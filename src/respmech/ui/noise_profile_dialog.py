"""Modal noise-profile picker.

Shows the raw EMG channels stacked on a shared time axis. Hovering draws a vertical
crosshair on every channel and a cursor-following label with the time (3 dp). A
click-drag marks a rest region — shaded on every channel — and the label then shows
the region's duration (Δt). A region wider than 0.5 s warns that a larger noise
profile can slow processing markedly. The selection persists until a new drag or a
plain click clears it. "Set noise profile" (enabled only once a region is marked)
accepts the dialog; "Annullér" rejects it without touching the settings.

The wheel zooms the time axis for all channels together (they are x-linked, and the
view is bounded to the recording); a bottom scrollbar pans the zoomed view, and a
double-click resets it — restoring the marked region its own leading click cleared.

The selection state (``_set_selection`` / ``_clear_selection`` / ``_maybe_warn`` /
``selected_region``) is factored out of the mouse handling so it is testable headless.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
                               QPushButton, QScrollBar, QVBoxLayout)

import pyqtgraph as pg

from respmech.ui.plot_overlays import add_flow_background

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

_WARN_SECONDS = 0.5
_TIME_STEPS = 1000.0                   # scrollbar ticks per second: QScrollBar is integer-only
_GLW_MARGIN = 10                       # GraphicsLayoutWidget's own margin around the plots
_REGION_BRUSH = (255, 152, 0, 60)      # brand orange, semi-transparent
_ACCENT = (255, 152, 0)


def _axis_width():
    """The pinned left-axis width the stacked plots share, so the scrollbar can be inset to
    sit under the DATA area rather than under the y-axis labels."""
    if _theme is not None:
        try:
            return int(_theme.PLOT_AXIS_WIDTH)
        except Exception:  # pragma: no cover - defensive
            pass
    return 68


def _plot_pal():
    """Active-theme plot colours (light table if the theme module is unavailable)."""
    if _theme is not None:
        try:
            return _theme.plot_palette()
        except Exception:  # pragma: no cover - defensive
            pass
    return {"bg": "#FCFDFE", "fg": (51, 64, 77), "noise_trace": (90, 150, 200)}


class NoiseProfileDialog(QDialog):
    """Pick a rest span across the raw EMG channels to use as the noise reference."""

    def __init__(self, raw, t, fs, cols, parent=None, file_name="", flow=None):
        super().__init__(parent)
        self.setWindowTitle("Set noise profile" + (f" — {file_name}" if file_name else ""))
        self.setModal(True)
        self.resize(940, 580)
        self._t = np.asarray(t, dtype=float)
        self._fs = fs
        self._flow = np.asarray(flow, dtype=float) if flow is not None else None
        self._selection = None             # (t0, t1) in seconds, or None
        self._dragging = False
        self._moved = False                # did the pointer move far enough to count as a drag?
        self._press_x = None
        self._press_px = None              # press point in viewport pixels (click vs drag test)
        self._sel_before_press = None      # selection as it was at the last press, so a double-
                                           # click (whose leading click clears it) can restore it
        self._label_plot = None            # the plot the cursor label currently lives on
        self._syncing = False              # scrollbar <-> ViewBox are wired BOTH ways, so each
                                           # must refuse to answer the other's update

        v = QVBoxLayout(self)
        hint = QLabel("Hover to read the time; click-drag over a quiet (EMG-free) span to "
                      "mark the rest region on all channels. Click once to clear it. "
                      "Scroll to zoom the time axis (all channels together); double-click to reset. "
                      "Use the bar below the plots to move through the recording when zoomed in.")
        hint.setProperty("status", "muted"); hint.setWordWrap(True)
        v.addWidget(hint)

        pal = _plot_pal()
        trace_pen = pal.get("noise_trace", (90, 150, 200))
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(pal["bg"])
        v.addWidget(self.glw, 1)
        self._plots, self._vlines, self._regions = [], [], []
        tmin = float(self._t[0]) if self._t.size else 0.0
        prev = None
        n = len(raw)
        for i, y in enumerate(raw):
            p = self.glw.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            p.plot(self._t[:len(y)], np.asarray(y, dtype=float), pen=pg.mkPen(trace_pen))
            add_flow_background(p, self._t, self._flow, pal)   # discrete respiration reference, behind
            if i == n - 1:
                p.setLabel("bottom", "Time (s)")
            vb = p.getViewBox()
            # Wheel-zoom the TIME axis only. y stays fixed: the channels have wildly different
            # scales (col 2 is ±0.5, col 3 is ±500), so a shared y zoom would be meaningless —
            # whereas x is already linked below, so zooming any channel zooms them ALL together.
            # Left-drag still marks the rest region: eventFilter() consumes those events before
            # the ViewBox can pan on them.
            vb.setMouseEnabled(x=True, y=False)
            vb.setMenuEnabled(False)
            # Bound the time axis to the recording: the view can never zoom or pan outside
            # the data. xMin/xMax bound the RANGE, not merely the pan, so no maxXRange is
            # needed. This must be set on EVERY ViewBox, not just the first: the plots are
            # x-linked in a chain, so a wheel over channel 3 originates the range change at
            # ITS ViewBox. An unlimited ViewBox scales freely and only the far end of the
            # chain clamps, which desyncs the stack instead of stopping the zoom.
            if self._t.size > 1:
                vb.setLimits(xMin=float(self._t[0]), xMax=float(self._t[-1]))
            if prev is not None:
                p.setXLink(prev)
            prev = p
            vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(_ACCENT, width=1))
            vl.setVisible(False); p.addItem(vl, ignoreBounds=True)
            reg = pg.LinearRegionItem(values=(tmin, tmin), movable=False,
                                      brush=pg.mkBrush(*_REGION_BRUSH), pen=pg.mkPen(_ACCENT, width=1))
            reg.setZValue(-5); reg.setVisible(False); p.addItem(reg)
            self._plots.append(p); self._vlines.append(vl); self._regions.append(reg)
        # The x view is EXPLICIT, never auto-ranged: pyqtgraph's padded first auto-range
        # gets clamped against the xMax limit and, through the x-link chain, locks a
        # rightward drift in — the dialog then opens with the start of the recording
        # cropped off-screen and the pan bar already at its maximum. y stays auto per
        # channel (their scales differ wildly).
        if self._plots and self._t.size > 1:
            for p in self._plots:
                p.getViewBox().enableAutoRange(x=False)
            self._plots[0].getViewBox().setXRange(
                float(self._t[0]), float(self._t[-1]), padding=0)
        self._label = pg.TextItem("", color=pal["fg"], anchor=(0, 1))
        self._label.setZValue(50)

        # Pan control for the zoomed time axis. Left-drag belongs to the region picker (the
        # eventFilter consumes it before the ViewBox can pan), so this is the ONLY way to
        # move through the recording once zoomed. Inset to sit under the data, not the axis.
        self.scroll = QScrollBar(Qt.Horizontal)
        self.scroll.setEnabled(False)
        _sr = QHBoxLayout()
        _sr.setContentsMargins(_axis_width() + _GLW_MARGIN, 0, _GLW_MARGIN, 0)
        _sr.addWidget(self.scroll)
        v.addLayout(_sr)

        self.warn = QLabel(""); self.warn.setObjectName("noiseWarn")
        self.warn.setWordWrap(True); self.warn.setVisible(False)
        self.warn.setStyleSheet("#noiseWarn { color: #E08A4F; font-weight: 600; }")
        v.addWidget(self.warn)

        row = QHBoxLayout()
        self.info = QLabel(""); self.info.setProperty("status", "muted")
        row.addWidget(self.info, 1)
        self.btn_cancel = QPushButton("Annullér"); self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("Set noise profile"); self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.accept)
        row.addWidget(self.btn_cancel); row.addWidget(self.btn_ok)
        v.addLayout(row)

        vp = self.glw.viewport()
        vp.installEventFilter(self)
        vp.setMouseTracking(True)

        self.scroll.valueChanged.connect(self._scroll_to)
        if self._plots:
            _vb0 = self._plots[0].getViewBox()
            _vb0.sigXRangeChanged.connect(self._sync_scroll)
            self._sync_scroll(_vb0, _vb0.viewRange()[0])    # seed range/enabled state

    # -- coordinate mapping -------------------------------------------------
    def _scene_x(self, view_pos):
        scene = self.glw.mapToScene(view_pos)
        return float(self._plots[0].getViewBox().mapSceneToView(scene).x())

    def _plot_at(self, view_pos):
        scene = self.glw.mapToScene(view_pos)
        for p in self._plots:
            vb = p.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(scene):
                return p, vb.mapSceneToView(scene)
        return None, None

    def _clamp(self, x):
        if not self._t.size:
            return x
        return max(float(self._t[0]), min(float(self._t[-1]), x))

    # -- interaction --------------------------------------------------------
    def eventFilter(self, obj, ev):
        # The left button belongs to the REGION picker, the wheel belongs to the ViewBox's
        # x zoom. Returning True on the left-button events stops them reaching the ViewBox,
        # which would otherwise pan the view under the very drag that marks the region
        # (mouse-x is enabled for the wheel). Wheel/other events fall through untouched.
        if obj is self.glw.viewport() and self._plots:
            try:
                et = ev.type()
                if et == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                    self._press_px = ev.position().toPoint()
                    self._press_x = self._clamp(self._scene_x(self._press_px))
                    self._dragging = True
                    self._moved = False
                    self._sel_before_press = self._selection   # so a double-click can restore it
                    return True
                elif et == QEvent.MouseMove:
                    self._on_move(ev.position().toPoint())
                    if self._dragging:
                        return True
                elif et == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                    self._on_release(ev.position().toPoint())
                    return True
                elif et == QEvent.MouseButtonDblClick and ev.button() == Qt.LeftButton:
                    self._reset_zoom()                 # double-click anywhere -> whole recording
                    # Qt delivers a double-click as Press-Release-DblClick-Release, so the
                    # leading Release already ran _on_release -> _clear_selection. Put the
                    # user's marked region back: resetting the view must not lose it.
                    if self._sel_before_press is not None:
                        self._set_selection(*self._sel_before_press)
                    return True
                elif et == QEvent.Leave:
                    self._hide_cursor()
            except Exception:                          # noqa: BLE001 — interaction is cosmetic
                pass
        return super().eventFilter(obj, ev)

    # -- time scrollbar -----------------------------------------------------
    def _scroll_to(self, value):
        """Scrollbar moved -> pan the view, preserving the current zoom width."""
        if self._syncing or not self._plots or not self._t.size:
            return
        vb = self._plots[0].getViewBox()
        lo, hi = vb.viewRange()[0]
        span = hi - lo                       # read live: panning must never change the zoom
        x0 = float(self._t[0]) + value / _TIME_STEPS
        self._syncing = True
        try:
            vb.setXRange(x0, x0 + span, padding=0)   # x-linked -> every channel follows
        finally:
            self._syncing = False

    def _sync_scroll(self, vb, rng):
        """View changed (wheel zoom, double-click reset) -> match the scrollbar to it.

        Disabled rather than hidden when the whole recording is visible: hiding it would
        make the plot stack jump by the scrollbar's height on every zoom in and out.
        """
        if self._syncing or not self._t.size:
            return
        lo, hi = float(rng[0]), float(rng[1])
        t0, t1 = float(self._t[0]), float(self._t[-1])
        span = max(1e-9, hi - lo)
        hidden = max(0.0, (t1 - t0) - span)          # how much time is off-screen
        self._syncing = True
        try:
            self.scroll.setRange(0, int(round(hidden * _TIME_STEPS)))
            self.scroll.setPageStep(max(1, int(round(span * _TIME_STEPS))))
            self.scroll.setSingleStep(max(1, int(round(span * _TIME_STEPS / 10.0))))
            self.scroll.setValue(int(round((lo - t0) * _TIME_STEPS)))
            self.scroll.setEnabled(hidden > 1.0 / _TIME_STEPS)
        finally:
            self._syncing = False

    def _reset_zoom(self):
        """Back to the whole recording. The plots are x-linked, so setting one resets all."""
        if not self._plots or not self._t.size:
            return
        self._plots[0].getViewBox().setXRange(float(self._t[0]), float(self._t[-1]), padding=0)

    def _on_move(self, view_pos):
        x = self._clamp(self._scene_x(view_pos))
        for vl in self._vlines:
            vl.setValue(x); vl.setVisible(True)
        p, vpt = self._plot_at(view_pos)
        if p is None:                                  # cursor between plots -> pin to the top
            p = self._plots[0]
            y = p.getViewBox().viewRange()[1][1]
        else:
            y = vpt.y()
        if self._label_plot is not p:
            if self._label_plot is not None:
                self._label_plot.removeItem(self._label)
            p.addItem(self._label, ignoreBounds=True); self._label_plot = p
        if self._dragging and self._press_x is not None:
            if not self._moved and self._press_px is not None \
                    and (view_pos - self._press_px).manhattanLength() > QApplication.startDragDistance():
                self._moved = True                     # far enough -> this is a drag, not a click
            if self._moved:
                lo, hi = min(self._press_x, x), max(self._press_x, x)
                for reg in self._regions:
                    reg.setRegion((lo, hi)); reg.setVisible(True)
                self._maybe_warn(hi - lo)
                self._label.setText(f"Δ {hi - lo:.3f} s")
            else:
                self._label.setText(f"{x:.3f} s")
        else:
            self._label.setText(f"{x:.3f} s")
        self._label.setPos(x, y)

    def _on_release(self, view_pos):
        if not self._dragging:
            return
        self._dragging = False
        x = self._clamp(self._scene_x(view_pos))
        if not self._moved:                            # a plain click (within jitter) -> dismiss
            self._clear_selection()
        else:
            x0 = self._press_x if self._press_x is not None else x
            self._set_selection(min(x0, x), max(x0, x))

    def _hide_cursor(self):
        for vl in self._vlines:
            vl.setVisible(False)
        self._label.setText("")

    # -- selection state (headless-testable) --------------------------------
    def _set_selection(self, t0, t1):
        t0, t1 = float(min(t0, t1)), float(max(t0, t1))
        self._selection = (t0, t1)
        for reg in self._regions:
            reg.setRegion((t0, t1)); reg.setVisible(True)
        self.btn_ok.setEnabled(True)
        self.info.setText(f"Rest region {t0:.3f}–{t1:.3f} s (Δ {t1 - t0:.3f} s)")
        self._maybe_warn(t1 - t0)

    def _clear_selection(self):
        self._selection = None
        for reg in self._regions:
            reg.setVisible(False)
        self.btn_ok.setEnabled(False)
        self.info.setText("")
        self.warn.setVisible(False)

    def _maybe_warn(self, width):
        if width > _WARN_SECONDS:
            self.warn.setText(f"⚠ A noise profile wider than {_WARN_SECONDS:g} s "
                              f"(selected {width:.3f} s) can increase processing time markedly.")
            self.warn.setVisible(True)
        else:
            self.warn.setVisible(False)

    def selected_region(self):
        """The chosen (t0, t1) in seconds, or None if nothing is marked."""
        return self._selection

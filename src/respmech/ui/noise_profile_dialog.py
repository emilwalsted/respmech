"""Modal noise-profile picker.

Shows the raw EMG channels stacked on a shared time axis. Hovering draws a vertical
crosshair on every channel and a cursor-following label with the time (3 dp). A
click-drag marks a rest region — shaded on every channel — and the label then shows
the region's duration (Δt). A region wider than 0.5 s warns that a larger noise
profile can slow processing markedly. The selection persists until a new drag or a
plain click clears it. "Set noise profile" (enabled only once a region is marked)
accepts the dialog; "Annullér" rejects it without touching the settings.

The selection state (``_set_selection`` / ``_clear_selection`` / ``_maybe_warn`` /
``selected_region``) is factored out of the mouse handling so it is testable headless.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout)

import pyqtgraph as pg

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

_WARN_SECONDS = 0.5
_REGION_BRUSH = (255, 152, 0, 60)      # brand orange, semi-transparent
_ACCENT = (255, 152, 0)


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

    def __init__(self, raw, t, fs, cols, parent=None, file_name=""):
        super().__init__(parent)
        self.setWindowTitle("Set noise profile" + (f" — {file_name}" if file_name else ""))
        self.setModal(True)
        self.resize(940, 580)
        self._t = np.asarray(t, dtype=float)
        self._fs = fs
        self._selection = None             # (t0, t1) in seconds, or None
        self._dragging = False
        self._moved = False                # did the pointer move far enough to count as a drag?
        self._press_x = None
        self._press_px = None              # press point in viewport pixels (click vs drag test)
        self._label_plot = None            # the plot the cursor label currently lives on

        v = QVBoxLayout(self)
        hint = QLabel("Hover to read the time; click-drag over a quiet (EMG-free) span to "
                      "mark the rest region on all channels. Click once to clear it.")
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
            p.plot(self._t[:len(y)], np.asarray(y, dtype=float), pen=pg.mkPen(trace_pen))
            if i == n - 1:
                p.setLabel("bottom", "Time (s)")
            vb = p.getViewBox()
            vb.setMouseEnabled(x=False, y=False)   # our drag marks a region, it does not pan
            vb.setMenuEnabled(False)
            if prev is not None:
                p.setXLink(prev)
            prev = p
            vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(_ACCENT, width=1))
            vl.setVisible(False); p.addItem(vl, ignoreBounds=True)
            reg = pg.LinearRegionItem(values=(tmin, tmin), movable=False,
                                      brush=pg.mkBrush(*_REGION_BRUSH), pen=pg.mkPen(_ACCENT, width=1))
            reg.setZValue(-5); reg.setVisible(False); p.addItem(reg)
            self._plots.append(p); self._vlines.append(vl); self._regions.append(reg)
        self._label = pg.TextItem("", color=pal["fg"], anchor=(0, 1))
        self._label.setZValue(50)

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
        if obj is self.glw.viewport() and self._plots:
            try:
                et = ev.type()
                if et == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                    self._press_px = ev.position().toPoint()
                    self._press_x = self._clamp(self._scene_x(self._press_px))
                    self._dragging = True
                    self._moved = False
                elif et == QEvent.MouseMove:
                    self._on_move(ev.position().toPoint())
                elif et == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                    self._on_release(ev.position().toPoint())
                elif et == QEvent.Leave:
                    self._hide_cursor()
            except Exception:                          # noqa: BLE001 — interaction is cosmetic
                pass
        return super().eventFilter(obj, ev)

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

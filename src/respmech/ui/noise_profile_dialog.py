"""Modal noise-profile picker.

Shows the EMG channels the noise profile is actually built from — the ECG-reduced
matrix when "Remove ECG" is on, or the raw channels when it is off (D08, UI-overhaul:
the caller resolves this via ``workers.stage_ecg_reduction``, never here) — stacked on
a shared time axis. R-peak markers (``peak_times``) are drawn on every channel when
ECG removal found any, the same ▼ the ECG tab uses, so a residual heartbeat reads as
cardiac rather than muscle activity. Hovering draws a vertical crosshair on every
channel and a cursor-following label with the time (3 dp). A click-drag marks a rest
region — shaded on every channel — and the label then shows the region's duration
(Δt). A region wider than 0.5 s warns that a larger noise profile can slow processing
markedly. The selection persists until a new drag or a plain click clears it. "Set
noise profile" (enabled only once a region is marked) accepts the dialog; "Cancel"
rejects it without touching the settings.

The reference can also be defined as "every expiration" rather than a marked span — the two
are alternatives in the core, so they are offered as one choice here instead of a checkbox
on another screen that this dialog silently contradicted.

The wheel zooms the time axis for all channels together (they are x-linked, and the
view is bounded to the recording); a bottom scrollbar pans the zoomed view, and a
double-click resets it — restoring the marked region its own leading click cleared.

The selection state (``_set_selection`` / ``_clear_selection`` / ``_maybe_warn`` /
``selected_region``) is factored out of the mouse handling so it is testable headless.

Opening the picker on a test that already has a saved reference seeds it via
``_seed_reference`` (D07, UI-overhaul) — the caller resolves the reference against the
CURRENT (possibly re-opened) settings and calls it after construction; the dialog itself
never reads settings. That makes "OK" without dragging anything a valid, no-op accept,
instead of the picker opening empty and forcing a fresh drag to see what was already
saved. When ``reference_file`` names a file other than the one the picker was opened on,
a persistent banner explains that accepting moves the WHOLE test's reference to this
file, and the accept button is relabelled "Replace rest reference" so the click
describes its own effect.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QEvent, QSize
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QScrollArea, QScrollBar, QVBoxLayout)

import pyqtgraph as pg

from respmech.ui.flow_layout import ElidingCheckBox
from respmech.ui.stft_frames import (DEFAULT_HOP_LENGTH, DEFAULT_WIN_LENGTH,
                                     MIN_STABLE_FRAMES, min_seconds_for_frames,
                                     stft_frame_count)
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

#: sentinel returned by ``selected_region`` for the whole-expiration option
EXPIRATION = "expiration"

_TIME_STEPS = 1000.0                   # scrollbar ticks per second: QScrollBar is integer-only
_GLW_MARGIN = 10                       # GraphicsLayoutWidget's own margin around the plots
#: Minimum height (px) for one channel row, so a rest span stays judgeable by eye however
#: many EMG channels the rig has. Chosen against the measured collapse: at the old fixed
#: pane a 12-channel recording gave each channel ZERO pixels of data area.
#: (row height, not data height — the axis and label take ~36 px of it, so this leaves
#: ~85 px of actual trace.)
_ROW_MIN_H = 120
_REGION_BRUSH = (255, 152, 0, 60)      # brand orange, semi-transparent
#: The one accent this picker uses for "a marked noise reference", unthemed (it reads the
#: same on light and dark plot grounds). Public: the Detail-plot indicator that mirrors the
#: saved reference (preview_screen's EMG tab) imports this so the two never drift apart —
#: before D07 that indicator was painted the SAME hue as the breathing shading it sits
#: inside (theme.py's old noise_region token), which made it invisible in practice.
NOISE_ACCENT = (255, 152, 0)
_ACCENT = NOISE_ACCENT


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

    def __init__(self, raw, t, fs, cols, parent=None, file_name="", flow=None,
                reference_file="", peak_times=None, ecg_applied=True,
                win_length=DEFAULT_WIN_LENGTH, hop_length=DEFAULT_HOP_LENGTH):
        super().__init__(parent)
        self.setWindowTitle("Set noise profile" + (f" — {file_name}" if file_name else ""))
        self.setModal(True)
        self._preferred = QSize(940, 580)   # opening size, clamped to the screen in showEvent
        self._t = np.asarray(t, dtype=float)
        self._fs = fs
        self._flow = np.asarray(flow, dtype=float) if flow is not None else None
        # R-peak times (s), same axis as _t — the caller passes stage_ecg_reduction's
        # 'peaks', which are populated whether or not removal is ON (D08): even with
        # removal off, the reference channel's peaks are still detected (cheaply) so the
        # markers can say "these are heartbeats" instead of leaving them unlabelled.
        self._peak_times = (np.asarray(peak_times, dtype=float)
                            if peak_times is not None else np.array([], dtype=float))
        self._ecg_applied = bool(ecg_applied)
        # The STFT geometry the noise profile will actually be built with, threaded in
        # from settings.processing.emg.noise by the caller (D09). Without it the frame
        # threshold cannot be computed and any warning would silently assume 1000 Hz.
        self._win_length = int(win_length)
        self._hop_length = int(hop_length)
        self._selection = None             # (t0, t1) in seconds, or None
        # the file the TEST's shared reference currently lives on (may differ from the
        # file this picker was opened on, or be "" if no reference is set yet) — drives
        # the persistent cross-file warning below, never touched again after construction
        self._reference_file = reference_file or ""
        self._mismatch = bool(self._reference_file and file_name
                              and self._reference_file != file_name)
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
        # Asks for what the criterion can actually mean on real data (D08): "quiet
        # (EMG-free)" sent users hunting for a gap between heartbeat spikes that are not
        # even in the signal the profile is built from when ECG removal is on. Says
        # plainly, when it is OFF, that the heartbeats are still there — in this signal
        # AND in the profile it builds — rather than leaving that to be discovered later.
        _rest_hint = ("click-drag over a span where the diaphragm is at rest, typically "
                     "late in expiration, to mark the rest region on all channels")
        if self._ecg_applied:
            _ecg_hint = ("Heartbeats are marked ▼ and have already been removed from "
                         "this signal.")
        else:
            _ecg_hint = ("'Remove ECG' is off, so heartbeats (marked ▼ where found) are "
                         "still in this signal, and will still be in the profile it builds.")
        self.hint = QLabel(f"Hover to read the time; {_rest_hint}. {_ecg_hint} Click once "
                          "to clear the selection. Scroll to zoom the time axis (all "
                          "channels together); double-click to reset. Use the bar below "
                          "the plots to move through the recording when zoomed in.")
        self.hint.setProperty("status", "muted"); self.hint.setWordWrap(True)
        v.addWidget(self.hint)

        # Persistent, not tied to the drag/selection state below (self.warn IS): the test
        # has exactly one shared reference regardless of which file is open, and accepting
        # here always moves it to THIS file — the user needs to know that before they ever
        # touch the plots, not only after a drag. First shown widget, so it cannot be missed.
        self.file_warn = QLabel("")
        self.file_warn.setObjectName("noiseFileWarn")
        self.file_warn.setWordWrap(True)
        self.file_warn.setVisible(self._mismatch)
        _fwarn = (_theme.active_theme().get("st_warn_fg", "#8A5A12")
                 if _theme is not None else "#8A5A12")
        self.file_warn.setStyleSheet("#noiseFileWarn { color: %s; font-weight: 600; }" % _fwarn)
        if self._mismatch:
            self.file_warn.setText(
                f"This test's rest reference is currently {self._reference_file}. Setting "
                f"it here moves the whole test's reference to {file_name} — the profile is "
                "still built once, from this one span, and applied identically to every "
                "file in the test.")
        v.addWidget(self.file_warn)

        pal = _plot_pal()
        trace_pen = pal.get("noise_trace", (90, 150, 200))
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(pal["bg"])
        # The plot stack goes in a scroll area and is given a per-channel minimum height
        # below. Without both, every channel shares one fixed pane: measured at the old
        # 940x580 opening size the data area per channel was 325 px at 1 channel, 41 px at
        # 5, 14 px at 8 and ZERO at 12 — the dialog silently stopped showing the signal the
        # user is being asked to pick a rest span from.
        #
        # Deliberately NOT wheel-guarded: this dialog's own hint tells the user to scroll to
        # zoom the time axis, so forwarding the wheel to the scroll area would take away a
        # documented interaction. pyqtgraph accepts the wheel over a plot, so zoom keeps
        # working and vertical travel is via the scroll bar.
        self._plot_area = QScrollArea()
        self._plot_area.setWidgetResizable(True)
        self._plot_area.setFrameShape(QFrame.NoFrame)
        self._plot_area.setWidget(self.glw)
        v.addWidget(self._plot_area, 1)
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
            add_ecg_capture_markers(p, self._peak_times, pal)  # same ▼ the ECG tab uses (D08)
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
        # the warning token, not a literal: #E08A4F is tuned for the dark ground and scored
        # 2.65:1 against the light theme's surface, which is below the 3:1 floor.
        _warn = (_theme.active_theme().get("st_warn_fg", "#8A5A12")
                 if _theme is not None else "#8A5A12")
        self.warn.setStyleSheet("#noiseWarn { color: %s; font-weight: 600; }" % _warn)
        v.addWidget(self.warn)

        # The two ways to define the reference, in ONE place. They are mutually exclusive in
        # the core (expiration wins whenever use_expiration or no intervals are set), and
        # they used to be split across two screens: a checkbox on Setup that this dialog
        # silently unticked whenever a span was marked, and which — re-ticked — silently made
        # the marked span inert. Neither screen showed what the other had done.
        # Eliding: this caption is one unbreakable item and was the dialog's ENTIRE minimum
        # width (598 px of 620 on Windows metrics, 1217 px at 225% text scaling), which put
        # the commit button off the right edge of a 1080p laptop at 175% display scaling.
        self.use_expiration = ElidingCheckBox("Use the whole expiration of this recording")
        self.use_expiration.setToolTip(
            "Sample the noise profile from every expiratory phase, which is diaphragm-quiet "
            "and gives a more stable estimate than one hand-marked span. Untick to mark a "
            "rest span yourself.")
        self.use_expiration.toggled.connect(self._on_mode_changed)
        v.addWidget(self.use_expiration)

        row = QHBoxLayout()
        self.info = QLabel(""); self.info.setProperty("status", "muted")
        row.addWidget(self.info, 1)
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.clicked.connect(self.reject)
        # Relabelled when accepting would move the reference off another file (see
        # file_warn above): the click then describes what it actually does.
        self.btn_ok = QPushButton("Replace rest reference" if self._mismatch
                                  else "Set noise profile")
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.accept)
        # Enter commits, Esc cancels. Without this Qt promotes Cancel (added first) as the
        # default and Enter throws away the marked region. The accepting button starts
        # disabled — Qt simply ignores Enter until a region makes it live, which is right.
        self.btn_cancel.setAutoDefault(False)
        self.btn_ok.setDefault(True)
        row.addWidget(self.btn_cancel); row.addWidget(self.btn_ok)
        v.addLayout(row)

        # Enough height per channel to judge a rest span by eye. The stack grows with the
        # channel count and the scroll area absorbs the overflow, so the DIALOG's minimum
        # stays small however many EMG channels the rig has.
        self.glw.setMinimumHeight(max(1, n) * _ROW_MIN_H)

        vp = self.glw.viewport()
        vp.installEventFilter(self)
        vp.setMouseTracking(True)

        self.scroll.valueChanged.connect(self._scroll_to)
        if self._plots:
            _vb0 = self._plots[0].getViewBox()
            _vb0.sigXRangeChanged.connect(self._sync_scroll)
            self._sync_scroll(_vb0, _vb0.viewRange()[0])    # seed range/enabled state

    def showEvent(self, ev):                # noqa: N802 - Qt API
        """Open at the preferred size, but never larger than the screen."""
        super().showEvent(ev)
        if not getattr(self, "_clamped", False):
            self._clamped = True
            from respmech.ui import screen_fit  # noqa: PLC0415
            screen_fit.clamp_to_screen(self, prefer=self._preferred)

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

    def _seed_reference(self, t0, t1):
        """Show the reference already saved for this test (D07, UI-overhaul): shade it via
        the normal selection path — so it behaves exactly like a fresh drag, right down to
        the width warning — then say plainly that this is the CURRENT reference, not a new
        pick, so accepting without touching anything is understood as "keep it". The
        caller resolves ``(t0, t1)`` against whatever is current in settings; this method
        never reads settings itself. Any later drag calls ``_set_selection`` directly and
        overwrites this wording with the ordinary "Rest region …" text, which is correct:
        at that point it IS a new pick, not the saved one."""
        self._set_selection(t0, t1)
        where = self._reference_file or "the current file"
        self.info.setText(f"Current reference: {where}, {t0:.3f}–{t1:.3f} s")

    def _maybe_warn(self, width):
        """Judge the dragged span by the SAME frame arithmetic the EMG tab applies
        (D09) — live, on every ``_set_selection``. The old rule warned above a fixed
        0.5 s about processing time, which the reference's width does not drive
        (the costly sweep runs on the ACTIVE clip), and at 1000 Hz it pointed the
        user away from the only spans the tab would call good. Below the stability
        threshold the label says how far to drag, in seconds; at or past it, the
        label goes quiet."""
        frames = stft_frame_count(int(round(width * self._fs)),
                                  self._win_length, self._hop_length)
        if frames < MIN_STABLE_FRAMES:
            need = min_seconds_for_frames(MIN_STABLE_FRAMES, self._fs,
                                          self._win_length, self._hop_length)
            enkelt = "frame" if frames == 1 else "frames"
            self.warn.setText(
                f"⚠ {width:.2f} s gives {frames} STFT {enkelt} — too short for a stable "
                f"noise estimate. Drag to at least {need:.2f} s ({MIN_STABLE_FRAMES} frames).")
            self.warn.setVisible(True)
        else:
            self.warn.setVisible(False)

    def _on_mode_changed(self, on):
        """Whole-expiration and a marked span are alternatives, so choosing one visibly
        retires the other rather than leaving both on screen looking active."""
        self.glw.setEnabled(not on)
        for reg in self._regions:
            reg.setVisible(bool(self._selection) and not on)
        if on:
            self.warn.setVisible(False)
            self.info.setText("The profile will be built from every expiration.")
        elif self._selection:
            self._set_selection(*self._selection)
        else:
            self.info.setText("")
        self.btn_ok.setEnabled(on or self._selection is not None)

    def selected_region(self):
        """The chosen (t0, t1) in seconds, ``EXPIRATION`` for the whole-expiration option,
        or None if nothing is chosen."""
        if self.use_expiration.isChecked():
            return EXPIRATION
        return self._selection

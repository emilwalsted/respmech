"""Screen 2 — source-data preview & tuning.

Two sub-tabs (the EMG one appears only when EMG channels are configured):

* **Mechanics** — the flow/volume/pressure channels on a shared time axis with
  detected breath boundaries and the trim window; a Campbell diagram and the
  per-breath table. Breaths are shaded/numbered and can be included/excluded by
  clicking.
* **EMG processing** — a view of the raw EMG channels; a "result" view of the
  conditioned EMG (pick which channels to show); a single-channel detail (raw ▸
  ECG-removed ▸ noise-reduced, time + PSD); a draggable rest-region selector that
  sets the shared noise reference; and the noise-window options (active once a
  reference file is set) with a fidelity-frontier preview.

**Reactive auto-run.** Selecting a file automatically kicks off every runnable
computation. Independent jobs (mechanics preview; the full test run that yields
the Campbell/table/fidelity in one pass; the EMG staging) run concurrently, each
on its own worker thread; panels that depend on a value (the auto-selected noise
suppression) are re-run in order once that value is known. Every sub-panel shows
its own spinner while its job is in flight and clears it on completion.

The scheduling entry points are reached only through ``QTimer.singleShot`` (or a
button click), so under a headless test — which never spins the Qt event loop —
no worker threads start and the direct render/compute methods stay synchronous.
All computation reuses the core; nothing here writes to disk.
"""
from __future__ import annotations

import copy
import glob
import os
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QThread, QTimer, Signal

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.io.loaders import load  # noqa: F401 (re-exported for tests/back-compat)
from respmech.core.pipeline import run_batch
from respmech.core._legacy_ns import to_legacy_ns  # noqa: F401 (back-compat import)
from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_mechanics_preview, stage_noise_fidelity)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

# channel -> (axis label with units, pen colour by physiological meaning)
_CHANNELS = [
    ("flow", "Flow (L/s)", (44, 110, 155)),
    ("volume", "Volume (L)", (31, 122, 77)),
    ("poes", "Poes (cmH₂O)", (180, 50, 42)),
    ("pgas", "Pgas (cmH₂O)", (183, 121, 31)),
    ("pdi", "Pdi (cmH₂O)", (125, 91, 166)),
]
# distinct pens for EMG channels in the raw / result views
_EMG_PENS = [(44, 110, 155), (180, 50, 42), (31, 122, 77), (183, 121, 31),
             (125, 91, 166), (14, 124, 123), (180, 80, 122), (92, 107, 122)]

# reactive job kinds -> which panel overlays they own, and the spinner caption.
# 'batch' (the test run: table + Campbell) is NOT auto-run — only the button triggers
# it. 'noise' auto-computes the fidelity frontier + suppression, decoupled from batch.
_PANELS = {"mech": ["channels", "raw"], "batch": ["campbell"],
           "emg_all": ["result"], "emg_detail": ["detail", "detail_psd"], "noise": ["fidelity"]}
_SPIN_TEXT = {"mech": "Loading channels…", "batch": "Running test…",
              "emg_all": "Conditioning channels…", "emg_detail": "Staging detail…",
              "noise": "Measuring fidelity…"}
# human labels for status lines + panel error cards
_KIND_LABEL = {"mech": "Channel preview", "batch": "Test run",
               "emg_all": "EMG result", "emg_detail": "EMG detail",
               "noise": "Noise fidelity"}
# the kinds that run automatically (on file select / settings change / Refresh). The
# test run ('batch') is now automatic too, but MECHANICS-ONLY (no ECG/EMG work).
_AUTO_KINDS = ("mech", "batch", "emg_all", "emg_detail", "noise")


def _pen(colour, width=1):
    return pg.mkPen(colour, width=width)


def _rms_envelope(x, win):
    """Sliding-window RMS envelope of a 1-D signal (the quantity each breath's EMG
    amplitude is taken from), same length as ``x``. Front-padded so it aligns in time."""
    x = np.asarray(x, dtype=float)
    win = max(1, int(win))
    if x.size == 0 or win >= x.size:
        return np.sqrt(np.maximum(np.mean(x * x) if x.size else 0.0, 0.0)) * np.ones_like(x)
    c = np.cumsum(np.insert(x * x, 0, 0.0))
    ma = (c[win:] - c[:-win]) / win
    env = np.sqrt(np.maximum(ma, 0.0))
    return np.concatenate([np.full(x.size - env.size, env[0]), env])


# Fallback plot palette used only if the theme module failed to import (it never
# raises in practice); mirrors theme._PLOT_LIGHT so light behaviour is unchanged.
_FALLBACK_PAL = {
    "bg": "#FCFDFE", "fg": "#33404D", "grid_alpha": 0.15,
    "channels": {k: c for k, _l, c in _CHANNELS},
    "emg_cycle": list(_EMG_PENS),
    "breath_incl_brush": (44, 110, 155, 32), "breath_excl_brush": (180, 50, 42, 70),
    "breath_incl_label": (90, 107, 122), "breath_excl_label": (180, 50, 42),
    "separator": (150, 165, 180), "noise_region": (44, 110, 155, 45),
    "raw_trace": (150, 165, 180), "noise_trace": (90, 150, 200),
    "legend_bg": (255, 255, 255, 0),
    "mpl_bg": "#FFFFFF",
    "mpl_accent": "#2C6E9B", "mpl_ok": "#1F7A4D", "mpl_warn": "#B7791F",
    "mpl_error": "#B4322A", "mpl_muted": "0.6",
    "mpl_loop": "0.55", "mpl_zeroline": "0.85", "mpl_target": "0.35",
}


def _plot_pal():
    """The active theme's plot colour table (light before any theme is applied)."""
    if _theme is not None:
        try:
            return _theme.plot_palette()
        except Exception:  # pragma: no cover - defensive
            pass
    return _FALLBACK_PAL


_CHECK_ICON_PATH = None


def _check_icon_url():
    """A white checkmark PNG (generated once, cached in the temp dir) used as the
    ``:checked`` image of the colour-filled result-channel checkboxes, so the tick sits
    inside the channel's coloured box. Qt QSS ``url()`` needs a real file, not a data URI."""
    global _CHECK_ICON_PATH
    if _CHECK_ICON_PATH is None:
        import tempfile
        from PySide6.QtGui import QImage, QPainter, QPen, QColor
        from PySide6.QtCore import QPointF
        img = QImage(24, 24, QImage.Format_ARGB32); img.fill(Qt.transparent)
        p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("white")); pen.setWidth(3)
        pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); p.setPen(pen)
        p.drawPolyline([QPointF(5, 12), QPointF(10, 18), QPointF(19, 6)]); p.end()
        path = os.path.join(tempfile.gettempdir(), "respmech_check.png")
        img.save(path)
        _CHECK_ICON_PATH = path.replace("\\", "/")
    return _CHECK_ICON_PATH


@dataclass(eq=False)
class _Job:
    """A single in-flight reactive computation (one kind at a time is current).

    ``eq=False`` keeps identity-based equality/hashing so a job can live in the
    ``_draining`` set. ``error`` is per-job (never kind-keyed) so a superseded
    job's late failure cannot poison its fresh successor's result."""
    kind: str
    token: int
    thread: object
    worker: object
    error: object = None


# threads that would not stop within the shutdown budget are parked here (kept
# referenced forever) so CPython never GCs a still-running QThread and aborts.
_ORPHANED_THREADS = []


class _FileRunError(RuntimeError):
    """A per-file analysis error (the core returned FileResult.error). Raised so
    _on_job_done labels it 'failed' — not a 'display error' — but still routes it
    to the copyable error card."""


class BusyOverlay(QWidget):
    """A translucent overlay that covers one panel — either a spinner while the
    panel's job runs, or an error card (short message + a round info button that
    opens the full, copyable trace) if the job failed.

    State lives in plain attributes (``busy``/``error``/``message``), so it is
    inspectable under a headless test without any pixels. It re-covers its parent
    on resize/move (splitter drags) and is only ever shown/hidden from the GUI
    thread."""

    def __init__(self, panel: QWidget):
        super().__init__(panel)
        self.busy = False
        self.error = None
        self.message = ""
        self._detail_dialog = None
        self.setObjectName("busyOverlay")

        # -- busy card: caption + indeterminate bar --
        self._busy_box = QWidget(self)
        self._busy_box.setObjectName("busyBox")
        bl = QVBoxLayout(self._busy_box)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.setSpacing(8)
        self._label = QLabel("", self._busy_box)
        self._label.setObjectName("busyLabel")
        self._label.setAlignment(Qt.AlignCenter)
        self._bar = QProgressBar(self._busy_box)
        self._bar.setObjectName("busyBar")
        self._bar.setRange(0, 0)          # indeterminate = animated spinner
        self._bar.setTextVisible(False)
        self._bar.setFixedWidth(150)
        bl.addWidget(self._label)
        bl.addWidget(self._bar, 0, Qt.AlignCenter)

        # -- error card: ⚠ + summary + round info button --
        self._error_box = QWidget(self)
        self._error_box.setObjectName("errorBox")
        self._error_box.setMaximumWidth(360)
        el = QVBoxLayout(self._error_box)
        el.setContentsMargins(16, 14, 16, 14)
        el.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(9)
        self._err_icon = QLabel("⚠")
        self._err_icon.setObjectName("errIcon")
        self._err_label = QLabel("")
        self._err_label.setObjectName("errLabel")
        self._err_label.setWordWrap(True)
        self._info_btn = QPushButton("i")
        self._info_btn.setObjectName("infoBtn")
        self._info_btn.setFixedSize(24, 24)
        self._info_btn.setToolTip("Show the full error detail (copyable)")
        self._info_btn.setCursor(Qt.PointingHandCursor)
        self._info_btn.clicked.connect(self._open_detail)
        head.addWidget(self._err_icon, 0, Qt.AlignTop)
        head.addWidget(self._err_label, 1)
        head.addWidget(self._info_btn, 0, Qt.AlignTop)
        el.addLayout(head)
        hint = QLabel("Click the info button for the full trace.")
        hint.setObjectName("errHint")
        hint.setWordWrap(True)
        el.addWidget(hint)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._busy_box, 0, Qt.AlignCenter)
        lay.addWidget(self._error_box, 0, Qt.AlignCenter)
        self._error_box.hide()

        # the round info button tracks the active theme's accent (so it is the dark
        # accent in dark mode, not the light-mode steel-blue) — everything else on the
        # overlay already sits on a translucent dark scrim and works in both themes.
        t = _theme.active_theme() if _theme is not None else {}
        acc = t.get("accent", "#2C6E9B")
        acc_fg = t.get("accent_fg", "white")
        acc_hover = t.get("accent_hover", "#3E8AC0")
        self.setStyleSheet(
            "#busyOverlay { background-color: rgba(18, 22, 28, 0.42); }"
            "#busyBox, #errorBox { background-color: palette(window);"
            " border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; }"
            "#errorBox { border-color: rgba(180,50,42,0.55); }"
            "#busyLabel, #errLabel { color: palette(window-text); font-weight: 600; }"
            "#errIcon { color: #E08A4F; font-size: 17px; font-weight: 700; }"
            "#errHint { color: rgba(147,164,183,0.95); font-size: 11px; }"
            "#infoBtn { background-color: " + acc + "; color: " + acc_fg + "; border: none;"
            " border-radius: 12px; font-weight: 700; font-style: italic; }"
            "#infoBtn:hover { background-color: " + acc_hover + "; }")
        panel.installEventFilter(self)
        self.setGeometry(panel.rect())
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() in (QEvent.Resize, QEvent.Move):
            self.setGeometry(self.parent().rect())
        return False

    def start(self, message="Working…"):
        self.busy = True
        self.error = None
        self.message = message
        self._label.setText(message)
        self._error_box.hide()
        self._busy_box.show()
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def stop(self):
        self.busy = False
        self.error = None
        self.message = ""
        self.hide()

    def show_error(self, summary, detail):
        """Switch to the error card: a short summary + a round info button that
        opens the full ``detail`` (traceback) in a copyable dialog."""
        self.busy = False
        self.error = detail or ""
        self._err_label.setText(summary or "Something went wrong")
        self._busy_box.hide()
        self._error_box.show()
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def _open_detail(self):
        if not self.error:
            return
        if self._detail_dialog is not None:   # replace, don't accumulate windows
            self._detail_dialog.close()
            self._detail_dialog.deleteLater()
            self._detail_dialog = None
        dlg = TextViewerDialog(
            "Error detail", self.error, self,
            intro="Full error trace — select the text or use Copy to clipboard.")
        self._detail_dialog = dlg          # keep a reference so it isn't GC'd
        dlg.show()
        dlg.raise_()

    def is_busy(self):
        return self.busy


class PreviewScreen(QWidget):
    status_changed = Signal(str)
    # emitted when a noise reference is chosen on the graph (feature B)
    noise_reference_changed = Signal(str, object, bool)
    # emitted to process AND write just the previewed file (P19)
    process_file_requested = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        # reactive-job bookkeeping (replaces the old single self._thread/_worker)
        self._jobs = {}                  # kind -> _Job (the current owner)
        self._draining = set()           # superseded jobs, kept alive until self-cleanup
        self._tokens = {"mech": 0, "batch": 0, "emg_all": 0, "emg_detail": 0, "noise": 0}
        self._overlays = {}              # panel key -> BusyOverlay
        self._autorun_pending = False
        self._previewed_file = None
        self._loading_noise = False
        # breath-overlay interaction state (feature A)
        self._channel_plots = []
        self._breath_spans = {}
        self._breath_regions = {}
        self._breath_texts = {}
        # draggable noise-selection region (feature B)
        self._noise_region = None
        # staged all-channel EMG (result view)
        self._emg_all = None
        self.result_checks = []          # list of (col:int, QCheckBox)
        # breath overlays in the EMG views (Mechanics keeps its own _breath_* state)
        self._breaths = []               # last mech spans [(num, t0, t1, ignored), ...] (TRIMMED s)
        self._trim_offset_s = 0.0        # startix/fs — maps a trimmed span to absolute EMG time
        self._bov = {}                   # view -> {'items':[(plot,item)], 'regions':{num:[reg]}, 'texts':{num:txt}}
        self._emg_raw_subplots = []      # per-channel PlotItems of the raw stack (hit-testing)
        self._raw_label_y = None
        self._detail_label_y = None
        self._result_label_y = None
        self._build()
        # render dispatch by job kind (built after _build so the methods exist)
        self._RENDER = {
            "mech": self._render_preview,
            "batch": self._on_batch_result,
            "emg_all": self._on_emg_all_result,
            "emg_detail": self._on_emg_detail_result,
            "noise": self._on_noise_result,
        }

    # -- construction -------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.file_combo = QComboBox()
        # P27: step through files without leaving the plots — buttons + PageUp/PageDown.
        self.btn_prev_file = QPushButton("◀")
        self.btn_next_file = QPushButton("▶")
        for b, tip in ((self.btn_prev_file, "Previous file (PgUp)"),
                       (self.btn_next_file, "Next file (PgDn)")):
            b.setFixedWidth(30); b.setToolTip(tip)
        self.btn_prev_file.clicked.connect(lambda: self._step_file(-1))
        self.btn_next_file.clicked.connect(lambda: self._step_file(+1))
        # 'Refresh' recomputes every auto panel (not the test run); the file list itself
        # refreshes automatically when file-related settings change.
        self.btn_refresh_all = QPushButton("Refresh")
        self.btn_refresh_all.setToolTip("Recompute all preview panels for the current file.")
        self.btn_refresh_all.clicked.connect(self._refresh_all)
        bar.addWidget(QLabel("File:")); bar.addWidget(self.btn_prev_file)
        bar.addWidget(self.file_combo, 1); bar.addWidget(self.btn_next_file)
        bar.addWidget(self.btn_refresh_all)
        root.addLayout(bar)
        from PySide6.QtGui import QShortcut, QKeySequence  # noqa: PLC0415
        QShortcut(QKeySequence(Qt.Key_PageUp), self, activated=lambda: self._step_file(-1))
        QShortcut(QKeySequence(Qt.Key_PageDown), self, activated=lambda: self._step_file(+1))
        # Status is shown ONLY in the main-window bottom status bar (via status_changed).
        # Keep the label object (state holder + tests read pv.status) but never place it
        # under the file selector.
        self.status = QLabel("Pick a file — everything runs automatically.")
        self.status.setWordWrap(True)
        self.status.setParent(self)
        self.status.hide()

        self.subtabs = QTabWidget()
        self._mech_tab = self._build_mech_tab()
        self._emg_tab = self._build_emg_tab()
        self.subtabs.addTab(self._mech_tab, "Mechanics")
        root.addWidget(self.subtabs, 1)

        # one spinner overlay per panel (each fed by exactly one job kind)
        self._overlays = {
            "channels": BusyOverlay(self.plots),
            "campbell": BusyOverlay(self._mech_lower),
            "raw": BusyOverlay(self.emg_raw_plots),
            "result": BusyOverlay(self.emg_result_plots),
            "detail": BusyOverlay(self.emg_plots),
            "detail_psd": BusyOverlay(self.emg_psd_canvas),   # the detail job also renders the PSD
            "fidelity": BusyOverlay(self.fidelity_canvas),
        }

        self._refresh_emg_channels()
        self._load_noise_params()
        self._ensure_noise_region()
        self._update_emg_tab_visibility()

        self.file_combo.currentTextChanged.connect(self._on_file_selected)
        # the chip's themed height isn't known until the EMG sub-tab is first laid out;
        # match the 'Set noise profile' button to it then, so the strip is one band.
        self.subtabs.currentChanged.connect(lambda *_: QTimer.singleShot(0, self._align_noise_strip))
        self.refresh_files()

    def _build_mech_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)
        split = QSplitter(Qt.Vertical)
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setBackground(_plot_pal()["bg"])
        self.plots.scene().sigMouseClicked.connect(self._on_plot_clicked)
        # P17: a crosshair that tracks the cursor across the X-linked channel stack and
        # reads out (time, value); built once, the per-channel lines are (re)made per render.
        self._crosshair_lines = []
        self._crosshair_proxy = pg.SignalProxy(
            self.plots.scene().sigMouseMoved, rateLimit=60, slot=self._on_mech_mouse_moved)
        split.addWidget(self.plots)
        lower = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 0)
        lower.addWidget(self.table)
        self.campbell = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        lower.addWidget(self.campbell)
        self._mech_lower = lower
        # the per-breath table takes ~3/4 of the width; the Campbell diagram ~1/4 (default)
        lower.setStretchFactor(0, 3); lower.setStretchFactor(1, 1)
        lower.setSizes([720, 240])
        split.addWidget(lower)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        v.addWidget(split)
        # P16/P17: a thin action bar — batch QC overview + crosshair read-out + export
        bar = QHBoxLayout()
        self.qc_overview = QLabel("")
        self.qc_overview.setProperty("status", "muted")
        self.qc_overview.setToolTip("Quality overview of the most recent test run.")
        self.crosshair_label = QLabel("")
        self.crosshair_label.setProperty("status", "muted")
        self.btn_export_fig = QPushButton("Export Campbell…")
        self.btn_export_fig.setEnabled(False)          # enabled once a diagram is drawn
        self.btn_export_fig.setToolTip("Save the Campbell diagram as a PNG or PDF.")
        self.btn_export_fig.clicked.connect(self._export_campbell)
        # P19: process AND write just this file, so a tuned file can be produced without
        # re-running the whole batch (reuses the Run screen's write machinery).
        self.btn_process_file = QPushButton("Process && write this file")
        self.btn_process_file.setEnabled(False)
        self.btn_process_file.setToolTip("Run and write output for the previewed file only.")
        self.btn_process_file.clicked.connect(self._process_this_file)
        bar.addWidget(self.qc_overview); bar.addStretch(1)
        bar.addWidget(self.crosshair_label); bar.addSpacing(12)
        bar.addWidget(self.btn_process_file); bar.addWidget(self.btn_export_fig)
        v.addLayout(bar)
        return w

    def _process_this_file(self):
        name = self._previewed_file or self.file_combo.currentText()
        if name:
            self.process_file_requested.emit(os.path.basename(name))

    def _update_qc_overview(self, fr):
        """P16: a persistent, at-a-glance quality summary of the current test run —
        breaths used/excluded plus a conservative flag for any non-physiological
        per-breath value (so a suspect run is obvious without reading the table)."""
        bt = getattr(fr, "breaths_table", None)
        total = len(fr.breaths) if getattr(fr, "breaths", None) else (len(bt) if bt is not None else 0)
        used = len(bt) if bt is not None else 0
        excl = max(0, total - used)
        flags = []
        if bt is not None and len(bt):
            for c in ("vt", "ti", "te", "ttot"):
                if c in bt.columns and (bt[c] <= 0).any():
                    flags.append(f"{c}≤0")
            for c in ("wobtotal", "vt", "ve"):          # core metrics must be finite
                if c in bt.columns and not np.isfinite(bt[c].to_numpy(dtype=float)).all():
                    flags.append(f"{c} NaN")
        msg = f"QC:  {used} breaths used" + (f",  {excl} excluded" if excl else "")
        status = "ok"
        if flags:
            msg += "   ⚠ " + "; ".join(dict.fromkeys(flags))
            status = "warn"
        else:
            msg += "   ·  no flags"
        self.qc_overview.setText(msg)
        self.qc_overview.setProperty("status", status)
        self.qc_overview.style().unpolish(self.qc_overview)
        self.qc_overview.style().polish(self.qc_overview)

    # -- P17 crosshair + figure export -------------------------------------
    def _on_mech_mouse_moved(self, evt):
        if not self._channel_plots or not self._crosshair_lines:
            return
        pos = evt[0]
        for p, ln in zip(self._channel_plots, self._crosshair_lines):
            if p.sceneBoundingRect().contains(pos):
                mp = p.getViewBox().mapSceneToView(pos)
                for l in self._crosshair_lines:
                    l.setPos(mp.x()); l.show()
                lab = p.getAxis("left").labelText or "value"
                self.crosshair_label.setText(f"t = {mp.x():.3f} s     {lab} = {mp.y():.3g}")
                return

    def _export_campbell(self):
        from PySide6.QtWidgets import QFileDialog       # noqa: PLC0415
        base = (self._previewed_file or "campbell").rsplit(".", 1)[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Campbell diagram", f"{base} – Campbell.png",
            "PNG image (*.png);;PDF document (*.pdf)")
        if not path:
            return
        try:
            self.campbell.figure.savefig(path, dpi=150, bbox_inches="tight",
                                         facecolor=self.campbell.figure.get_facecolor())
            self._set_status(f"Saved Campbell diagram → {path}")
        except Exception as e:                          # noqa: BLE001
            self._set_status(f"Could not save figure: {short_error(str(e))}")

    def _build_emg_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)

        # detail-channel dropdown (pinned into the Detail panel title, top-right) and the
        # result-channel picker (pinned into the Conditioned-result panel title, top-right)
        self.emg_channel = QComboBox()
        self.emg_channel.setToolTip("EMG channel shown in the single-channel Detail panel.")
        self.emg_channel.currentIndexChanged.connect(self._on_emg_channel_changed)
        self.result_checks_holder = QWidget()
        self.result_checks_layout = QHBoxLayout(self.result_checks_holder)
        self.result_checks_layout.setContentsMargins(0, 0, 0, 0)
        self.result_checks_layout.setSpacing(24)      # generous gap between channels

        # -- one compact controls strip above the plots ----------------------
        # the 'Set noise profile' button + the noise-reduction params (in a subtle
        # rounded chip) on a single row, instead of two full-width rows.
        self.btn_set_noise = QPushButton("Set noise profile")
        self.btn_set_noise.setToolTip("Open a picker to mark a rest span across the raw EMG "
                                      "channels as the shared noise reference for this test.")
        self.btn_set_noise.clicked.connect(self._open_noise_profile_dialog)

        self.noise_auto = QCheckBox("Auto")
        self.noise_auto.setToolTip(_help_tip(
            "processing.emg.noise.auto_prop",
            "Automatically picks the strongest suppression that still keeps every channel at or "
            "above the fidelity target; on by default."))
        self.noise_auto.toggled.connect(self._on_noise_param_changed)
        self.noise_prop = QDoubleSpinBox(); self.noise_prop.setRange(0.0, 1.0); self.noise_prop.setSingleStep(0.05); self.noise_prop.setDecimals(2)
        self.noise_prop.setToolTip(_help_tip(
            "processing.emg.noise.prop_decrease",
            "How aggressively to remove noise when auto is off, from 0 (none) to 1 (maximum); default 0.6."))
        self.noise_prop.valueChanged.connect(self._on_noise_param_changed)
        self.noise_target = QDoubleSpinBox(); self.noise_target.setRange(0.50, 0.99); self.noise_target.setSingleStep(0.05); self.noise_target.setDecimals(2)
        self.noise_target.setToolTip(_help_tip(
            "processing.emg.noise.fidelity_target",
            "Smallest fraction of inspiratory EMG power that must survive noise removal, from 0 to 1; default 0.8."))
        self.noise_target.valueChanged.connect(self._on_noise_param_changed)
        self.noise_nstd = QDoubleSpinBox(); self.noise_nstd.setRange(0.0, 10.0); self.noise_nstd.setSingleStep(0.1); self.noise_nstd.setDecimals(1)
        self.noise_nstd.setToolTip(_help_tip(
            "processing.emg.noise.n_std_thresh",
            "Signal below the mean noise level plus this many standard deviations is treated as noise "
            "and removed; default 1.0."))
        self.noise_nstd.valueChanged.connect(self._on_noise_param_changed)

        def _cap(text, tip=""):        # a compact muted caption; full text lives in the tooltip
            la = QLabel(text); la.setProperty("status", "muted")
            if tip:
                la.setToolTip(tip)
            return la

        # noise-reduction params grouped in a subtle rounded chip (active once a
        # reference file is set — _update_actions enables/disables the whole chip)
        self.noise_opts = QFrame()
        self.noise_opts.setObjectName("noiseChip")
        self.noise_opts.setStyleSheet(
            "#noiseChip { border: 1px solid rgba(128, 128, 128, 0.30); border-radius: 8px; }")
        # tight caption→field pairing (4 px) with a wider gap (10 px) BETWEEN groups so
        # 'Strength [ ]', 'Keep ≥ [ ]', 'Gate [ ]' read as clusters, not an even bead line
        nrow = QHBoxLayout(self.noise_opts); nrow.setContentsMargins(11, 3, 11, 3); nrow.setSpacing(4)
        nrow.addWidget(_cap("Noise reduction"))
        nrow.addSpacing(8); nrow.addWidget(self.noise_auto)
        nrow.addSpacing(10); nrow.addWidget(_cap("Strength", self.noise_prop.toolTip())); nrow.addWidget(self.noise_prop)
        nrow.addSpacing(10); nrow.addWidget(_cap("Keep ≥", self.noise_target.toolTip())); nrow.addWidget(self.noise_target)
        nrow.addSpacing(10); nrow.addWidget(_cap("Gate", self.noise_nstd.toolTip())); nrow.addWidget(self.noise_nstd)

        strip = QHBoxLayout(); strip.setSpacing(10)
        strip.addWidget(self.btn_set_noise)
        strip.addWidget(self.noise_opts)
        strip.addStretch(1)
        v.addLayout(strip)

        # plots — LEFT column: all-channel views (raw stack, conditioned result);
        #         RIGHT column: single detail-channel views (time detail, its PSD).
        # (Detail and Conditioned result are swapped vs. the old layout.)
        _bg = _plot_pal()["bg"]
        self.emg_raw_plots = pg.GraphicsLayoutWidget()
        self.emg_raw_plots.setBackground(_bg)
        self.emg_result_plots = pg.PlotWidget()
        self._style_legend(self.emg_result_plots.addLegend())
        self.emg_result_plots.setBackground(_bg)
        self.emg_result_plots.setLabel("bottom", "Time (s)")
        self.emg_result_plots.setLabel("left", "Conditioned EMG (a.u.)")
        self.emg_plots = pg.PlotWidget()
        self._style_legend(self.emg_plots.addLegend())
        self.emg_plots.setBackground(_bg)
        self.emg_plots.setLabel("bottom", "Time (s)"); self.emg_plots.setLabel("left", "EMG (a.u.)")
        self.emg_psd_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))

        split = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(self._titled("Raw EMG channels", self.emg_raw_plots))
        top.addWidget(self._titled("Detail channel — raw ▸ ECG-removed ▸ noise-reduced",
                                   self.emg_plots, corner=self.emg_channel))
        split.addWidget(top)
        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(self._titled("Conditioned result (selected channels)",
                                   self.emg_result_plots, corner=self.result_checks_holder))
        mid.addWidget(self._titled("Detail PSD", self.emg_psd_canvas))
        self._emg_mid = mid
        split.addWidget(mid)
        self.fidelity_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        split.addWidget(self._titled("Noise fidelity frontier", self.fidelity_canvas))
        v.addWidget(split, 1)
        # clicking a numbered breath in any EMG plot toggles its exclusion too
        self.emg_raw_plots.scene().sigMouseClicked.connect(self._on_emg_raw_clicked)
        self.emg_plots.scene().sigMouseClicked.connect(self._on_emg_detail_clicked)
        self.emg_result_plots.scene().sigMouseClicked.connect(self._on_emg_result_clicked)
        return w

    @staticmethod
    def _style_legend(leg):
        """In DARK mode only, give a pyqtgraph legend a near-opaque dark backing and
        light text so entries stay legible where they float over bright trace fills
        (e.g. 'noise-reduced' over the green conditioned EMG). Light mode is left
        untouched so its legends render exactly as before. Never raises."""
        if leg is None or _theme is None or not _theme.is_dark():
            return
        pal = _plot_pal()
        try:
            leg.setBrush(pg.mkBrush(*pal["legend_bg"]))
            leg.setLabelTextColor(pg.mkColor(*pal["fg"]) if isinstance(pal["fg"], tuple)
                                  else pg.mkColor(pal["fg"]))
            leg.setPen(pg.mkPen(*pal["separator"]))
        except Exception:                            # noqa: BLE001 — legend chrome is cosmetic
            pass

    @staticmethod
    def _titled(title, widget, corner=None):
        """A titled panel. ``corner`` is an optional widget pinned to the top-right of
        the title row (e.g. the detail-channel dropdown or the result-channel picker)."""
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        header = QHBoxLayout(); header.setContentsMargins(0, 0, 0, 0); header.setSpacing(8)
        lab = QLabel(title); lab.setProperty("status", "muted")
        header.addWidget(lab); header.addStretch(1)
        if corner is not None:
            header.addWidget(corner, 0, Qt.AlignRight | Qt.AlignVCenter)
        lay.addLayout(header); lay.addWidget(widget, 1)
        return box

    def _align_noise_strip(self):
        """Match the 'Set noise profile' button to the noise chip's height so the strip
        reads as one aligned band. The chip's themed height is only known once the EMG
        sub-tab is laid out, so this is deferred to the sub-tab's first show."""
        h = self.noise_opts.height()
        if h > 1 and self.btn_set_noise.minimumHeight() != h:
            self.btn_set_noise.setMinimumHeight(h)

    # -- EMG tab visibility / channels -------------------------------------
    def _update_emg_tab_visibility(self):
        has_emg = bool(self.state.settings.input.channels.emg)
        idx = self.subtabs.indexOf(self._emg_tab)
        if has_emg and idx < 0:
            self.subtabs.addTab(self._emg_tab, "EMG processing")
        elif not has_emg and idx >= 0:
            self.subtabs.removeTab(idx)

    def _refresh_emg_channels(self):
        cols = list(self.state.settings.input.channels.emg)
        cur = self.emg_channel.currentIndex()
        self.emg_channel.blockSignals(True)
        self.emg_channel.clear()
        for c in cols:
            self.emg_channel.addItem(f"EMGdi col {c}")
        if 0 <= cur < self.emg_channel.count():
            self.emg_channel.setCurrentIndex(cur)
        self.emg_channel.blockSignals(False)
        # rebuild the result-channel picker: each checkbox's indicator IS the channel's
        # coloured box with the tick inside it, and 'col N' sits right beside its colour;
        # generous spacing keeps the channels visually separate.
        while self.result_checks_layout.count():
            item = self.result_checks_layout.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.setParent(None)
        self.result_checks = []
        check = _check_icon_url()
        cycle = _plot_pal()["emg_cycle"]
        for i, c in enumerate(cols):
            r, g, b = cycle[i % len(cycle)][:3]
            cb = QCheckBox(f"col {c}")
            cb.setChecked(i < 3)
            cb.setStyleSheet(
                "QCheckBox { spacing: 6px; }"
                "QCheckBox::indicator { width: 15px; height: 15px; border-radius: 3px; "
                f"border: 1px solid rgba({r}, {g}, {b}, 0.9); background: rgba({r}, {g}, {b}, 0.25); }}"
                f"QCheckBox::indicator:checked {{ background: rgb({r}, {g}, {b}); "
                f"image: url('{check}'); }}")
            cb.toggled.connect(self._render_emg_result)
            self.result_checks_layout.addWidget(cb)
            self.result_checks.append((c, cb))

    # -- noise-window options (owned here, gated on reference file) ---------
    def _load_noise_params(self):
        n = self.state.settings.processing.emg.noise
        self._loading_noise = True
        try:
            self.noise_auto.setChecked(n.auto_prop)
            self.noise_prop.setValue(n.prop_decrease)
            self.noise_target.setValue(n.fidelity_target)
            self.noise_nstd.setValue(n.n_std_thresh)
        finally:
            self._loading_noise = False

    def _on_noise_param_changed(self, *_):
        if self._loading_noise:
            return
        n = self.state.settings.processing.emg.noise
        n.auto_prop = self.noise_auto.isChecked()
        n.prop_decrease = self.noise_prop.value()
        n.fidelity_target = self.noise_target.value()
        n.n_std_thresh = self.noise_nstd.value()
        self._update_actions()
        self._request_autorun()          # re-condition with the new noise params

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- file list (reactive) ----------------------------------------------
    def refresh_files(self):
        s = self.state.settings
        prev = self.file_combo.currentText()
        folder = (s.input.folder or "").strip()
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        files = []
        if folder and os.path.isdir(folder):
            from respmech.ui.validation import matching_files
            files = [os.path.basename(f) for f in matching_files(folder, s.input.files)]
            self.file_combo.addItems(files)
        self.file_combo.blockSignals(False)
        if prev in files:
            self.file_combo.setCurrentText(prev)
        elif prev:
            # the current file changed silently while signals were blocked (prev is
            # gone from the list -> the combo auto-selected files[0], or emptied):
            # take the same cleanup path an interactive switch would, so stale breath
            # overlays never linger clickable against the wrong (or no) file
            self._begin_file_switch()
        if not folder:
            self._set_status("Set an input folder in Settings.")
        elif not os.path.isdir(folder):
            self._set_status(f"Input folder not found: {folder}")
        elif not files:
            self._set_status(f"No files match '{s.input.files}' in that folder.")
        else:
            self._set_status(f"{len(files)} file(s) — pick one; everything runs automatically.")
        self._update_actions()

    def _refresh_files(self):        # kept for existing wiring + tests
        self.refresh_files()

    def sync_from_settings(self):
        self._refresh_emg_channels()
        self._load_noise_params()
        self._update_emg_tab_visibility()
        self._update_actions()
        self._cancel_inflight()      # settings changed -> abort the now-stale panel jobs
        self._request_autorun()

    def _current_file(self):
        name = self.file_combo.currentText()
        return os.path.join(self.state.settings.input.folder, name) if name else None

    def _step_file(self, delta: int):
        """P27: move the file selection by ``delta`` (keyboard/▲▼), clamped to the list."""
        n = self.file_combo.count()
        if n <= 1:
            return
        i = min(max(self.file_combo.currentIndex() + delta, 0), n - 1)
        if i != self.file_combo.currentIndex():
            self.file_combo.setCurrentIndex(i)

    # -- enablement / guidance ---------------------------------------------
    def _settings_ok(self):
        try:
            self.state.settings.validate()
            return True, ""
        except Exception as e:              # noqa: BLE001
            return False, str(e)

    def _job_running(self, kind):
        return kind in self._jobs

    def _update_actions(self, status=True):
        """Refresh button/widget enablement. ``status=False`` updates enablement
        only and never touches the status line — used after a job completes so a
        fresh result/error message is not clobbered by a guidance hint."""
        has_file = bool(self.file_combo.currentText())
        ok, why = self._settings_ok()
        emg = self.state.settings.processing.emg
        noise_on, ref = emg.noise.enabled, emg.noise.reference_file
        has_emg = bool(self.state.settings.input.channels.emg)
        jr = self._job_running
        self.btn_refresh_all.setEnabled(has_file)
        self.emg_channel.setEnabled(has_emg)
        self.btn_set_noise.setEnabled(has_file and has_emg)
        # noise-window options only active with a reference file (Emil's requirement)
        self.noise_opts.setEnabled(bool(has_emg and noise_on and ref))
        if noise_on:
            self.noise_prop.setEnabled(bool(ref) and not self.noise_auto.isChecked())
        if not status:
            return
        if noise_on and not ref:
            self._set_status("Noise reduction is on — pick a reference file (Settings) or "
                             "click 'Set noise profile' to mark a rest span in this file.")
        elif has_file and not ok:
            self._set_status(f"Settings incomplete: {why}")

    def _invalidate_inflight(self):
        """Bump every kind's token so any in-flight job for the *previous* file is
        dropped by the acceptance check in _on_job_done when it lands — even if it
        finished in the same event-loop iteration as the switch, before the deferred
        autorun bumps tokens. A superseded job still clears its own spinner. Safe
        against _schedule's bump-last invariant: after a file change EVERY in-flight
        job is known-stale (wrong file), so none of them must survive."""
        for k in self._tokens:
            self._tokens[k] += 1

    def _cancel_inflight(self):
        """Abort every in-flight job: cancel its worker (cooperative), invalidate its
        token so a late result is dropped, and move it to draining so its thread is
        still joined on completion. Used on a settings change / file switch / Refresh."""
        for job in list(self._jobs.values()):
            if hasattr(job.worker, "cancel"):
                try:
                    job.worker.cancel()
                except Exception:                      # noqa: BLE001
                    pass
            self._draining.add(job)
        self._jobs.clear()
        self._invalidate_inflight()

    def _begin_file_switch(self):
        """Shared cleanup when the current file changes: clear every panel to its
        start/blank state, abort the old file's in-flight jobs, and re-dispatch the
        auto panels for the new file. Reached both from an interactive combo change and
        from refresh_files silently switching the current file when the prev vanished."""
        self._clear_all_panels()     # blank every panel + show the test-run CTA (as at start)
        self._cancel_inflight()      # abort the old file's in-flight jobs
        self._request_autorun()

    def _refresh_all(self):
        """The 'Refresh' button: recompute every auto panel (NOT the test run) for the
        current file. Aborts any in-flight auto jobs first, then re-dispatches."""
        if not self._current_file():
            return
        self._cancel_inflight()
        self._request_autorun()

    def _clear_all_panels(self):
        """Blank every panel to its start/empty state (as when the screen first opens):
        mechanics channels, the test-run table + Campbell, the raw EMG stack, the
        conditioned result, the detail time + PSD, and the fidelity frontier.
        _reset_breath_state additionally drops breath overlays + staged EMG."""
        self.plots.clear(); self._channel_plots = []
        self.table.setRowCount(0); self.table.setColumnCount(0)
        self.campbell.figure.clear(); self.campbell.draw()
        self.emg_raw_plots.clear()
        self.emg_plots.clear()
        self.emg_psd_canvas.figure.clear(); self.emg_psd_canvas.draw()
        self.fidelity_canvas.figure.clear(); self.fidelity_canvas.draw()
        self._reset_breath_state()   # clears _bov/_breaths/_emg_all/result plot/_previewed_file
        self._ensure_noise_region()  # re-attach the rest-region selector to the cleared detail plot
        self._clear_panel_overlays(*self._overlays)   # dismiss stale spinners / error cards

    def _on_file_selected(self, name):
        self._update_actions()
        if name and name != self._previewed_file:
            self._begin_file_switch()

    # -- reactive scheduling ------------------------------------------------
    def _request_autorun(self):
        """Coalesce many change events into a single full re-dispatch on the next
        event-loop tick. Never starts a thread synchronously (so it is inert in
        headless tests, which never spin the loop)."""
        if self._autorun_pending:
            return
        self._autorun_pending = True
        QTimer.singleShot(0, self._run_autorun)

    def _run_autorun(self):
        self._autorun_pending = False
        self._schedule_all()

    def _schedule_all(self):
        for k in _AUTO_KINDS:
            self._schedule(k)

    def _schedule(self, kind):
        """Evaluate the kind's gate, build its worker, and (only if it will
        actually launch) bump the kind's token and start it concurrently. The
        token is bumped LAST so a gated-out reschedule cannot supersede — and
        discard the result of — an in-flight job it never replaced."""
        path = self._current_file()
        if not path:
            return
        has_emg = bool(self.state.settings.input.channels.emg)
        # snapshot the settings on the GUI thread so a worker never reads them while
        # the GUI mutates them in place (Settings is a plain, deep-copyable model)
        snap = copy.deepcopy(self.state.settings)
        worker = None
        if kind == "mech":
            worker = FnWorker(stage_mechanics_preview, snap, path)
        elif kind == "batch":
            ok, _ = self._settings_ok()
            if not ok:
                return
            # MECHANICS-ONLY test run: drop EMG so run_batch skips ECG removal, EMG RMS
            # and noise reduction (that work is shown separately in the EMG tab).
            snap.input.channels.emg = []
            snap.processing.emg.remove_ecg = False
            snap.processing.emg.noise.enabled = False
            worker = BatchWorker(snap, write=False, only_files=[os.path.basename(path)])
        elif kind == "emg_all":
            if not has_emg:
                self._clear_panel_overlays(*_PANELS[kind])   # gated out -> drop a stale spinner/card
                return
            worker = EmgAllChannelsWorker(snap, path)
        elif kind == "emg_detail":
            if not has_emg:
                self._clear_panel_overlays(*_PANELS[kind])
                return
            ch = max(0, self.emg_channel.currentIndex())
            worker = EmgConditioningWorker(snap, path, ch)
        elif kind == "noise":
            ncfg = self.state.settings.processing.emg.noise
            if not has_emg or not ncfg.enabled or not ncfg.reference_file:
                self._clear_panel_overlays(*_PANELS[kind])   # no reference -> no fidelity panel
                return                                 # no fidelity without a noise reference
            worker = FnWorker(stage_noise_fidelity, snap)
        if worker is None:
            return
        self._tokens[kind] += 1
        self._launch(kind, self._tokens[kind], worker)

    def _launch(self, kind, token, worker):
        old = self._jobs.pop(kind, None)
        if old is not None:
            if hasattr(old.worker, "cancel"):
                try:
                    old.worker.cancel()
                except Exception:                      # noqa: BLE001
                    pass
            self._draining.add(old)                    # keep referenced until it self-cleans
        for p in _PANELS[kind]:
            self._overlays[p].start(_SPIN_TEXT[kind])
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        job = _Job(kind, token, thread, worker)
        self._jobs[kind] = job
        # Connect BOUND METHODS of self (a GUI-thread QObject), never lambdas: a
        # lambda has no receiver QObject, so Qt cannot infer thread affinity and
        # falls back to a DIRECT connection — which would run the render on the
        # worker thread and crash Qt. Bound methods force a queued (GUI-thread)
        # delivery. The emitting worker is recovered via sender().
        worker.finished.connect(self._job_finished)
        if hasattr(worker, "failed"):
            worker.failed.connect(self._job_failed_slot)
        thread.start()
        self._update_actions(status=False)

    def _find_job(self, worker):
        for j in list(self._jobs.values()) + list(self._draining):
            if j.worker is worker:
                return j
        return None

    def _job_failed_slot(self, msg):
        job = self._find_job(self.sender())
        if job is not None:
            job.error = msg                # per-job, so a drained job can't poison its successor

    def _job_finished(self, result):
        job = self._find_job(self.sender())
        if job is not None:
            self._on_job_done(job, result)

    def _on_job_done(self, job, result):
        # join + release the thread (the ONLY place a worker thread is joined);
        # a thread that will not stop is parked, never GC'd while running
        job.thread.quit()
        if not job.thread.wait(2000):
            _ORPHANED_THREADS.append((job.thread, job.worker))
        if self._jobs.get(job.kind) is job:
            del self._jobs[job.kind]
        self._draining.discard(job)
        # superseded by a newer same-kind job: drop the payload silently, and
        # only clear the spinner if no newer owner took over the panels — but
        # never wipe an error card the current owner already painted (a stale
        # draining job finishing late must not erase the live job's error).
        if job.token != self._tokens[job.kind]:
            if job.kind not in self._jobs:
                for p in _PANELS[job.kind]:
                    if not self._overlays[p].error:
                        self._overlays[p].stop()
            self._update_actions(status=False)
            return
        label = _KIND_LABEL.get(job.kind, job.kind)
        err = job.error
        if err or result is None:
            detail = err or "The computation returned no data."
            self._set_status(f"{label} failed — {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(f"{label} failed", detail)
            self._update_actions(status=False)
            return
        try:
            self._RENDER[job.kind](result)
        except _FileRunError as e:                     # a per-file analysis error -> "failed"
            detail = str(e)
            self._set_status(f"{label} failed — {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(f"{label} failed", detail)
            self._update_actions(status=False)
            return
        except Exception:                              # noqa: BLE001 — a rendering bug
            detail = traceback.format_exc()
            self._set_status(f"{label} — display error: {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(f"{label} — display error", detail)
            self._update_actions(status=False)
            return
        for p in _PANELS[job.kind]:
            self._overlays[p].stop()
        self._update_actions(status=False)

    def shutdown(self, wait_ms=5000):
        """Cancel and join every worker thread (current + draining). Called from
        MainWindow.closeEvent so nothing is destroyed while still running.

        A QThread whose worker is still mid-compute cannot be interrupted (the
        staging/`run_batch` calls are monolithic), so we join with a budget and,
        for any thread that will not stop in time, keep it referenced forever in
        the module-level parking list rather than let CPython GC a running
        QThread (which would call std::terminate and abort the process)."""
        jobs = list(self._jobs.values()) + list(self._draining)
        for j in jobs:
            if hasattr(j.worker, "cancel"):
                try:
                    j.worker.cancel()
                except Exception:                      # noqa: BLE001
                    pass
        for j in jobs:
            j.thread.quit()
        for j in jobs:
            if not j.thread.wait(wait_ms):
                _ORPHANED_THREADS.append((j.thread, j.worker))
        self._jobs.clear()
        self._draining.clear()

    def panel_busy(self, key):
        ov = self._overlays.get(key)
        return bool(ov and ov.busy)

    def busy_panels(self):
        return {k for k, o in self._overlays.items() if o.busy}

    def panel_error(self, key):
        """The full error detail shown in a panel's error card, or None."""
        ov = self._overlays.get(key)
        return ov.error if ov is not None else None

    def _clear_panel_overlays(self, *keys):
        """Dismiss any spinner/error card on these panels — called by the
        synchronous render paths so a successful redraw never leaves a stale
        error card raised over fresh content."""
        for k in keys:
            ov = self._overlays.get(k)
            if ov is not None:
                ov.stop()

    # -- channel preview (Mechanics), synchronous render -------------------
    def _preview(self):
        path = self._current_file()
        if not path:
            return
        try:
            data = stage_mechanics_preview(self.state.settings, path)
            self._render_preview(data)               # render also inside the guard
        except Exception:                            # noqa: BLE001 — copyable error card
            detail = traceback.format_exc()
            self._set_status(f"Channel preview failed — {short_error(detail)}")
            for p in ("channels", "raw"):
                self._overlays[p].show_error("Channel preview failed", detail)

    def _render_preview(self, data):
        """Draw the mechanics channel stack + breath overlays and the raw EMG
        stack from staged arrays. GUI-thread only; shared by the synchronous
        _preview() and the async 'mech' job."""
        fs = data["fs"]
        series = data["series"]
        t = data["t"]
        spans = data["spans"]
        self._trim_offset_s = data["startix"] / fs      # trimmed span -> absolute EMG time
        self._breaths = list(spans)
        self._clear_panel_overlays("channels", "raw")   # dismiss any stale error card
        self.plots.clear()
        self._channel_plots = []
        self._crosshair_lines = []                       # cleared with the plots; rebuilt below
        prev = None
        n = len(_CHANNELS)
        pal = _plot_pal()
        for i, (key, label, colour) in enumerate(_CHANNELS):
            p = self.plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=pal["grid_alpha"])
            p.setLabel("left", label)
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            y = series[key]
            p.plot(t[:len(y)], y, pen=_pen(pal["channels"].get(key, colour)))
            # P22: a subtle zero-reference baseline on every channel, so the sign and
            # excursion of each signal read at a glance.
            p.addLine(y=0, pen=pg.mkPen(pal["separator"], width=1, style=Qt.DashLine))
            if i == n - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
            self._limit_x(p, t[:len(y)])          # can't zoom/pan past the data
            self._channel_plots.append(p)
        # P17: one crosshair line per channel, hidden until the cursor enters the stack
        self._crosshair_lines = []
        for p in self._channel_plots:
            ln = p.addLine(x=0, pen=pg.mkPen(pal["separator"], width=1, style=Qt.DashLine))
            ln.hide()
            self._crosshair_lines.append(ln)
        self._draw_breath_overlays(spans, data["label_y"])
        self._render_raw_stack(data["emg"], fs)
        # breaths are now known -> (re)number any EMG detail/result already rendered
        self._repaint_view_breaths("detail")
        self._repaint_view_breaths("result")

        self._previewed_file = data["name"]
        self.btn_process_file.setEnabled(True)       # a file is loaded → can process just it
        if data.get("trim_error"):
            self._set_status(f"{data['name']}: showing raw channels — could not detect "
                             f"breaths. {data['trim_error']}")
        else:
            nign = sum(1 for _n, _a, _b, ig in spans if ig)
            self._set_status(
                f"{data['name']}: {data['nbreaths']} breaths"
                + (f" ({nign} excluded)" if nign else "")
                + f", trimmed to {data['startix'] / fs:.2f}–{data['endix'] / fs:.2f} s. "
                "Click a shaded breath to include/exclude (red = excluded).")

    def _render_raw_stack(self, emg, fs):
        """Draw the stacked raw EMG channels and keep the noise region alive."""
        emg = np.asarray(emg, dtype=float)
        if emg.ndim == 1:
            emg = emg[:, None]
        self._emg_raw_subplots = []
        self._raw_label_y = None
        self.emg_raw_plots.clear()
        if emg.size == 0 or emg.ndim != 2 or emg.shape[1] == 0:
            self._paint_breaths("raw", [], self._trim_offset_s, 0.0)   # drop stale overlays
            self._ensure_noise_region()
            return
        cols = list(self.state.settings.input.channels.emg)
        t = np.arange(emg.shape[0]) / fs
        prev = None
        cycle = _plot_pal()["emg_cycle"]
        for i in range(emg.shape[1]):
            p = self.emg_raw_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            p.plot(t, emg[:, i], pen=_pen(cycle[i % len(cycle)]))
            if i == emg.shape[1] - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
            self._limit_x(p, t)
            self._emg_raw_subplots.append(p)
        self._ensure_noise_region()
        self._raw_label_y = self._safe_top(emg[:, 0])
        self._repaint_view_breaths("raw")

    # -- feature A: breath overlays + include/exclude ----------------------
    @staticmethod
    def _breath_brush(ignored):
        pal = _plot_pal()
        return pg.mkBrush(*(pal["breath_excl_brush"] if ignored else pal["breath_incl_brush"]))

    @staticmethod
    def _breath_label_color(ignored):
        pal = _plot_pal()
        return pg.mkColor(*(pal["breath_excl_label"] if ignored else pal["breath_incl_label"]))

    @staticmethod
    def _limit_x(plot, t):
        """Bound a plot's x view to the data's time extent so it cannot zoom/pan past
        the recording (req: no zooming out beyond the data)."""
        try:
            t = np.asarray(t, dtype=float)
            t = t[np.isfinite(t)]
            if t.size < 2:
                return
            x0, x1 = float(t.min()), float(t.max())
            if x1 > x0:
                vb = plot.getViewBox()
                vb.setLimits(xMin=x0, xMax=x1)
                vb.setXRange(x0, x1, padding=0)
        except Exception:                        # noqa: BLE001 — cosmetic
            pass

    def _label_headroom(self, plot, frac=0.22):
        """Expand the label-carrying plot's y range upward so breath labels sit clearly
        above the signal — the short mechanics channel plots would otherwise clip them.
        Re-fits y to the DATA first so repeated repaints of a persistent plot (detail /
        result) don't compound the headroom."""
        try:
            vb = plot.getViewBox()
            vb.enableAutoRange(y=True)           # snap y back to the data extent…
            vb.updateAutoRange()                 # …(not the previous, already-expanded view)
            (x0, x1), (y0, y1) = vb.viewRange()
            span = y1 - y0
            if span > 0:
                vb.setYRange(y0, y1 + frac * span, padding=0)   # (this disables y auto again)
        except Exception:                        # noqa: BLE001 — cosmetic
            pass

    @staticmethod
    def _breath_label(num, short):
        """The breath label: full 'Breath #N', or the compact '#N' when full labels
        would overlap (dense breaths)."""
        return f"#{num}" if short else f"Breath #{num}"

    @staticmethod
    def _labels_would_overlap(centers, seconds_per_pixel, label_px=72.0):
        """True if full 'Breath #N' labels placed at these x-centres would collide at
        the given seconds-per-pixel scale (``label_px`` ≈ a full label's pixel width)."""
        if len(centers) < 2 or not seconds_per_pixel or seconds_per_pixel <= 0:
            return False
        cs = sorted(centers)
        min_gap = min(b - a for a, b in zip(cs, cs[1:]))
        return min_gap < label_px * seconds_per_pixel

    def _breath_labels_short(self, plot, centers):
        """Decide whether ``plot`` should use the compact '#N' breath label — yes when
        the full labels would overlap at the plot's current x scale."""
        try:
            vb = plot.getViewBox()
            if vb is None:
                return False
            # a freshly-created plot still sits at the default [0,1] range until its first
            # paint; flush the pending auto-range so the scale reflects the real data
            if any(ax is not False for ax in vb.autoRangeEnabled()):
                vb.updateAutoRange()
            px = vb.viewPixelSize()[0]
            if px and np.isfinite(px) and px > 0:
                spp = float(px)
            else:                                # not laid out yet -> assume a ~600px plot
                xr = vb.viewRange()[0]
                spp = abs((xr[1] - xr[0]) or 1.0) / 600.0
        except Exception:                        # noqa: BLE001 — labels are cosmetic
            return False
        return self._labels_would_overlap(centers, spp)

    def _draw_breath_overlays(self, spans, label_y=0.0):
        self._breath_spans = {n: (t0, t1) for (n, t0, t1, _ig) in spans}
        self._breath_regions = {n: [] for (n, _0, _1, _ig) in spans}
        self._breath_texts = {}
        centers = [(t0 + t1) / 2.0 for (_n, t0, t1, _ig) in spans]
        short = bool(self._channel_plots) and self._breath_labels_short(self._channel_plots[0], centers)
        sep = _plot_pal()["separator"]
        for n, t0, t1, ignored in spans:
            for plot in self._channel_plots:
                reg = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                          brush=self._breath_brush(ignored), pen=pg.mkPen(None))
                reg.setZValue(-10)
                plot.addItem(reg)
                self._breath_regions[n].append(reg)
                plot.addLine(x=t0, pen=pg.mkPen(sep, width=1, style=Qt.DashLine))
            if self._channel_plots:
                txt = pg.TextItem(self._breath_label(n, short),
                                  color=self._breath_label_color(ignored), anchor=(0.5, 1.0))
                txt.setPos((t0 + t1) / 2.0, label_y)
                self._channel_plots[0].addItem(txt)
                self._breath_texts[n] = txt
        if self._channel_plots and self._breath_texts:
            self._label_headroom(self._channel_plots[0])   # keep the labels above the signal

    def _breath_at(self, t):
        for n, (t0, t1) in self._breath_spans.items():
            if t0 <= t <= t1:
                return n
        return None

    def _on_plot_clicked(self, ev):
        if not self._breath_spans:
            return
        try:
            pos = ev.scenePos()
        except Exception:                              # noqa: BLE001
            return
        for plot in self._channel_plots:
            vb = plot.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(pos):
                bno = self._breath_at(vb.mapSceneToView(pos).x())
                if bno is not None:
                    self._toggle_breath(bno)
                return

    def _toggle_breath(self, breath_no):
        name = self.file_combo.currentText()
        if not name or breath_no not in self._breath_spans:
            return None
        excl = self.state.settings.processing.exclude_breaths
        entry = next((e for e in excl if e.file == name), None)
        if entry is None:
            entry = ExcludeEntry(file=name, breaths=[])
            excl.append(entry)
        now_excluded = breath_no not in entry.breaths
        if now_excluded:
            entry.breaths = sorted(set(entry.breaths) | {breath_no})
        else:
            entry.breaths = [b for b in entry.breaths if b != breath_no]
            if not entry.breaths:
                excl.remove(entry)
        for reg in self._breath_regions.get(breath_no, []):
            reg.setBrush(self._breath_brush(now_excluded)); reg.update()
        txt = self._breath_texts.get(breath_no)
        if txt is not None:
            txt.setColor(self._breath_label_color(now_excluded))
        # recolour the same breath in every EMG view that has it painted
        for view in ("raw", "detail", "result"):
            rec = self._bov.get(view)
            if not rec:
                continue
            for reg in rec["regions"].get(breath_no, []):
                try:
                    reg.setBrush(self._breath_brush(now_excluded)); reg.update()
                except Exception:                      # noqa: BLE001
                    pass
            t = rec["texts"].get(breath_no)
            if t is not None:
                try:
                    t.setColor(self._breath_label_color(now_excluded))
                except Exception:                      # noqa: BLE001
                    pass
        nexcl = len({b for e in excl if e.file == name for b in e.breaths} & set(self._breath_spans))
        # excluding/including a breath must update the AVERAGED result in lockstep with the
        # overlay — otherwise the Campbell loop + per-breath table stay stale and the user
        # tunes blind. Recompute the (mechanics-only) test run, debounced.
        self._request_batch_recompute()
        self._set_status(
            f"{name}: breath {breath_no} {'excluded' if now_excluded else 'included'} "
            f"({nexcl}/{len(self._breath_spans)} excluded). Recomputing the average…")
        return now_excluded

    def _request_batch_recompute(self):
        """Debounced recompute of the mechanics test run (Campbell + per-breath table) after
        a breath is toggled, so the averaged result tracks the overlay live. Inert headless
        (never spins the loop), like _request_autorun."""
        if getattr(self, "_batch_recompute_pending", False):
            return
        self._batch_recompute_pending = True
        QTimer.singleShot(0, self._run_batch_recompute)

    def _run_batch_recompute(self):
        self._batch_recompute_pending = False
        self._schedule("batch")

    # -- breath overlays on the EMG views ----------------------------------
    def _excluded_now(self):
        """Live set of excluded 1-based breath numbers for the current file — the
        single source of truth for overlay colour (matches what the core ignores)."""
        name = self.file_combo.currentText()
        entry = next((e for e in self.state.settings.processing.exclude_breaths if e.file == name), None)
        return set(entry.breaths) if entry else set()

    def _safe_top(self, *arrays, default=1.0):
        """A finite, headless-safe top-of-signal value for placing the number label."""
        best = None
        for a in arrays:
            if a is None:
                continue
            a = np.asarray(a, dtype=float)
            finite = a[np.isfinite(a)] if a.size else a
            if finite.size == 0:
                continue
            m = float(finite.max())
            best = m if best is None else max(best, m)
        return best if best is not None else default

    def _paint_breaths(self, view, plot_items, offset, label_y):
        """Idempotent per-view overlay: remove the view's previous items (guarded),
        then shade every breath + a boundary line on each plot and a number on the
        first plot. Colour is taken from the LIVE exclusion set."""
        rec = self._bov.get(view)
        if rec:
            for plot, item in rec.get("items", []):
                try:
                    plot.removeItem(item)
                except Exception:                      # noqa: BLE001
                    pass
        self._bov[view] = {"items": [], "regions": {}, "texts": {}}
        if not plot_items or not self._breaths:
            return
        excl = self._excluded_now()
        reg_map = self._bov[view]["regions"]; txt_map = self._bov[view]["texts"]
        items = self._bov[view]["items"]
        centers = [(t0 + t1) / 2.0 + offset for (_n, t0, t1, _ig) in self._breaths]
        short = self._breath_labels_short(plot_items[0], centers)
        sep = _plot_pal()["separator"]
        for (num, t0, t1, _ig) in self._breaths:
            ignored = num in excl
            a, b = t0 + offset, t1 + offset
            for p in plot_items:
                reg = pg.LinearRegionItem(values=(a, b), movable=False,
                                          brush=self._breath_brush(ignored), pen=pg.mkPen(None))
                reg.setZValue(-10)
                p.addItem(reg)
                items.append((p, reg)); reg_map.setdefault(num, []).append(reg)
                line = p.addLine(x=a, pen=pg.mkPen(sep, width=1, style=Qt.DashLine))
                items.append((p, line))
            txt = pg.TextItem(self._breath_label(num, short),
                              color=self._breath_label_color(ignored), anchor=(0.5, 1.0))
            txt.setPos((a + b) / 2.0, label_y)
            plot_items[0].addItem(txt)
            items.append((plot_items[0], txt)); txt_map[num] = txt
        self._label_headroom(plot_items[0])       # keep the labels above the signal

    def _repaint_view_breaths(self, view):
        """Repaint one EMG view IF it has rendered real data (label_y sentinel set)."""
        if view == "raw":
            plot_items, label_y = self._emg_raw_subplots, self._raw_label_y
        elif view == "detail":
            plot_items, label_y = [self.emg_plots.getPlotItem()], self._detail_label_y
        elif view == "result":
            plot_items, label_y = [self.emg_result_plots.getPlotItem()], self._result_label_y
        else:
            return
        if label_y is None or not plot_items:
            return
        try:
            self._paint_breaths(view, plot_items, self._trim_offset_s, label_y)
        except Exception:                              # noqa: BLE001 — overlay is cosmetic
            pass

    def _reset_breath_state(self):
        """On a file change, drop stale overlays + spans so a new-file EMG job that
        finishes before the new mech never shows the previous file's numbers."""
        for rec in self._bov.values():
            for plot, item in rec.get("items", []):
                try:
                    plot.removeItem(item)
                except Exception:                      # noqa: BLE001
                    pass
        self._bov = {}
        self._breaths = []
        self._breath_spans = {}
        self._breath_regions = {}
        self._breath_texts = {}
        self._emg_raw_subplots = []
        self._raw_label_y = self._detail_label_y = self._result_label_y = None
        self._trim_offset_s = 0.0
        # a result-checkbox toggle re-renders self._emg_all synchronously (no token
        # gate); drop the previous file's staged result so it can't repaint the old
        # curves and re-arm the result sentinel with them
        self._emg_all = None
        self.emg_result_plots.clear()
        # _previewed_file only advances on a COMPLETED mech render, so leaving it set
        # here would make a flip back to the last-rendered file look like "no change"
        # and skip this very reset while the new file's jobs are still in flight
        self._previewed_file = None

    def _toggle_from_emg_click(self, ev, plot_items, offset):
        if not self._breath_spans:
            return
        try:
            if ev.isAccepted():
                return
            pos = ev.scenePos()
        except Exception:                              # noqa: BLE001
            return
        for p in plot_items:
            # a click on the plot's legend (text/frame) is unaccepted by pyqtgraph
            # and would otherwise fall through to the breath underneath it — the
            # legend sits top-left over breath 1 on the detail/result plots
            leg = getattr(p, "legend", None)
            if leg is not None and leg.isVisible() and leg.sceneBoundingRect().contains(pos):
                return
            vb = p.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(pos):
                bno = self._breath_at(vb.mapSceneToView(pos).x() - offset)
                if bno is not None:
                    self._toggle_breath(bno)
                return

    def _on_emg_raw_clicked(self, ev):
        self._toggle_from_emg_click(ev, self._emg_raw_subplots, self._trim_offset_s)

    def _on_emg_detail_clicked(self, ev):
        if getattr(self._noise_region, "mouseHovering", False):
            return                                     # let the movable noise region own the click
        self._toggle_from_emg_click(ev, [self.emg_plots.getPlotItem()], self._trim_offset_s)

    def _on_emg_result_clicked(self, ev):
        self._toggle_from_emg_click(ev, [self.emg_result_plots.getPlotItem()], self._trim_offset_s)

    # -- feature B: noise-profile region -----------------------------------
    def _ensure_noise_region(self):
        reg = self._noise_region
        if reg is None:
            ivals = self.state.settings.processing.emg.noise.reference_intervals
            try:                                     # a malformed settings file must not abort construction
                t0, t1 = (float(ivals[0][0]), float(ivals[0][1])) if ivals else (0.0, 1.0)
            except Exception:                        # noqa: BLE001 — any bad shape -> cosmetic default
                t0, t1 = 0.0, 1.0
            # a read-only indicator of the current noise reference span; selection now
            # happens in the 'Set noise profile' modal, so this no longer needs dragging
            reg = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                      brush=pg.mkBrush(*_plot_pal()["noise_region"]))
            reg.setZValue(10)
            self._noise_region = reg
        if reg.scene() is None:
            self.emg_plots.addItem(reg)

    def _apply_noise_reference(self, t0, t1):
        """Apply [t0, t1] s of the current file as the shared noise reference: write it
        into settings, mirror it (signal + the inline detail-plot indicator), and
        re-condition. Shared by the noise-profile modal and the legacy inline region."""
        name = self.file_combo.currentText()
        if not name:
            return None
        t0, t1 = float(min(t0, t1)), float(max(t0, t1))
        if t1 - t0 <= 0:
            self._set_status("The selected noise region has zero width.")
            return None
        n = self.state.settings.processing.emg.noise
        n.reference_file = name
        n.reference_intervals = [[t0, t1]]
        n.use_expiration = False
        fs = self.state.settings.input.format.sampling_frequency or 0
        span = int(round((t1 - t0) * fs))
        nframes = 1 + max(0, span - n.win_length) // max(1, n.hop_length)
        self._ensure_noise_region()                  # reflect the choice on the detail plot
        try:
            self._noise_region.setRegion((t0, t1))
        except Exception:                            # noqa: BLE001 — indicator is cosmetic
            pass
        self.noise_reference_changed.emit(name, [[t0, t1]], False)
        self._set_status(
            f"Noise profile ← {name} [{t0:.2f}–{t1:.2f} s]: {span} samples ≈ {nframes} STFT frames "
            + ("(good)" if nframes >= 8 else "(short — pick a longer rest span)")
            + ". Enable 'Reduce EMG noise' to apply it.")
        self._update_actions()
        self._request_autorun()          # re-condition against the new reference
        return (t0, t1)

    def _use_region_as_noise(self):
        reg = self._noise_region
        if reg is None or not self.file_combo.currentText():
            self._set_status("Draw a rest region on the detail plot, then apply.")
            return None
        t0, t1 = reg.getRegion()
        return self._apply_noise_reference(t0, t1)

    def _open_noise_profile_dialog(self):
        """Open the modal noise-profile picker over the current file's raw EMG channels;
        on accept, apply the marked span as the shared noise reference."""
        path = self._current_file()
        if not path:
            return
        from respmech.ui.noise_profile_dialog import NoiseProfileDialog
        from respmech.ui.workers import stage_raw_emg
        try:
            data = stage_raw_emg(self.state.settings, path)
        except Exception:                            # noqa: BLE001 — copyable status
            self._set_status(f"Could not load EMG for the noise picker — {short_error(traceback.format_exc())}")
            return
        if not data["raw"]:
            self._set_status("This file has no EMG channels to pick a noise profile from.")
            return
        dlg = NoiseProfileDialog(data["raw"], data["t"], data["fs"], data["cols"],
                                 parent=self, file_name=self.file_combo.currentText())
        if dlg.exec() == QDialog.Accepted:
            sel = dlg.selected_region()
            if sel is not None:
                self._apply_noise_reference(sel[0], sel[1])

    # -- EMG detail / result / test triggers (manual re-runs) --------------
    def _on_emg_channel_changed(self, *_):
        if self._current_file():
            QTimer.singleShot(0, lambda: self._schedule("emg_detail"))

    # -- EMG detail render (single channel) --------------------------------
    def _on_emg_detail_result(self, data):
        self.render_emg_time(data)
        self.render_emg_psd(data)
        col = data.get("col", int(data.get("channel", 0)) + 1)
        if data.get("noise_applied"):
            applied = "noise reduction applied"
        elif data.get("noise_error"):
            applied = "noise conditioning failed — showing ECG-removed"
        else:
            applied = "no noise reference — raw vs ECG-removed"
        note = " · ECG removal failed (showing raw)" if data.get("ecg_error") else ""
        self._set_status(f"EMG detail (col {col}): {applied}{note} (band 20–250 Hz marked).")

    def render_emg_time(self, data):
        t = np.asarray(data["t"])
        pal = _plot_pal()
        ch = pal["channels"]
        self.emg_plots.clear()
        self.emg_plots.plot(t, np.asarray(data["raw"]), pen=_pen(pal["raw_trace"]), name="raw")
        self.emg_plots.plot(t, np.asarray(data["ecg"]), pen=_pen(ch["flow"]), name="ECG-removed")
        final = np.asarray(data["ecg"])
        if data.get("noise_applied"):
            final = np.asarray(data["noise"])
            self.emg_plots.plot(t, final, pen=_pen(ch["volume"]), name="noise-reduced")
        # P13: overlay the RMS envelope the analysis quantifies (± the conditioned signal),
        # so the picked amplitude is visible over the raw trace it is taken from.
        fs = self.state.settings.input.format.sampling_frequency or (1.0 / (t[1] - t[0]) if len(t) > 1 else 1.0)
        env = _rms_envelope(final, (self.state.settings.processing.emg.rms_window_s or 0.05) * fs)
        env_pen = pg.mkPen((180, 50, 42), width=2)
        self.emg_plots.plot(t, env, pen=env_pen, name="RMS envelope")
        self.emg_plots.plot(t, -env, pen=env_pen)
        self.emg_plots.setLabel("bottom", "Time (s)"); self.emg_plots.setLabel("left", "EMG (a.u.)")
        self._limit_x(self.emg_plots.getPlotItem(), t)
        self._ensure_noise_region()
        self._detail_label_y = self._safe_top(
            np.asarray(data["raw"]), np.asarray(data["ecg"]),
            np.asarray(data["noise"]) if data.get("noise_applied") else None)
        self._repaint_view_breaths("detail")

    def render_emg_psd(self, data):
        freqs = np.asarray(data["freqs"], dtype=float)
        band = data.get("band", (20.0, 250.0))
        pal = _plot_pal()
        fig = self.emg_psd_canvas.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])

        def _db(p):
            return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=float), 1e-30))

        ax.plot(freqs, _db(data["psd_raw"]), color=pal["mpl_muted"], lw=1.0, label="raw")
        ax.plot(freqs, _db(data["psd_ecg"]), color=pal["mpl_accent"], lw=1.2, label="ECG-removed")
        if data.get("noise_applied"):
            ax.plot(freqs, _db(data["psd_noise"]), color=pal["mpl_ok"], lw=1.2, label="noise-reduced")
        ax.axvspan(band[0], band[1], color=pal["mpl_warn"], alpha=0.12, label="EMG band 20–250 Hz")
        ax.set_xlim(0, min(500.0, float(freqs[-1]) if len(freqs) else 500.0))
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB)")
        ax.set_title(f"EMG PSD — col {data.get('col', int(data.get('channel', 0)) + 1)}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        self.emg_psd_canvas.draw()

    # -- result view (all channels, pick which to show) --------------------
    def _on_emg_all_result(self, data):
        self._emg_all = data
        self._render_emg_result()
        if data.get("noise_applied"):
            applied = "conditioned (ECG + noise)"
        elif data.get("ecg_applied"):
            applied = "ECG-removed"
        else:
            applied = "raw"
        notes = []
        if data.get("ecg_error"):
            notes.append("ECG removal failed")
        if data.get("noise_error"):
            notes.append("noise conditioning failed")
        note = (" · " + ", ".join(notes)) if notes else ""
        self._set_status(f"EMG result: {applied}{note}. Tick channels above to show/hide them.")

    def _render_emg_result(self, *_):
        """Overlay the conditioned result for the ticked channels."""
        data = self._emg_all
        self.emg_result_plots.clear()
        if not data:
            self._result_label_y = None
            self._bov["result"] = {"items": [], "regions": {}, "texts": {}}
            return
        t = np.asarray(data["t"])
        cols = data.get("cols", [])
        cycle = _plot_pal()["emg_cycle"]
        checked = {c for c, cb in self.result_checks if cb.isChecked()}
        for i, c in enumerate(cols):
            if c not in checked or i >= len(data["conditioned"]):
                continue
            self.emg_result_plots.plot(t, np.asarray(data["conditioned"][i]),
                                       pen=_pen(cycle[i % len(cycle)]), name=f"col {c}")
        self._limit_x(self.emg_result_plots.getPlotItem(), t)
        self._result_label_y = self._safe_top(*[np.asarray(c) for c in data.get("conditioned", [])])
        self._repaint_view_breaths("result")

    def _on_batch_result(self, result):
        """Render the automatic mechanics test run: the per-breath table + Campbell (the
        batch is mechanics-only, so no EMG/fidelity here). The noise_report branch is
        retained for the direct-call path that still carries one."""
        cur = self.file_combo.currentText()
        fr = None
        if getattr(result, "files", None):
            fr = result.files.get(cur) or next(iter(result.files.values()), None)
        if fr is None or getattr(fr, "error", None):
            err = getattr(fr, "error", None) or "The run produced no result for this file."
            if str(err).startswith("TrimError"):
                # same precondition failure the mech preview already handled softly
                # (raw channels + a 'could not detect breaths' status): leave the table
                # + Campbell blank, no hard 'Test run failed' card, don't touch the status.
                self.table.setRowCount(0); self.table.setColumnCount(0)
                self.campbell.figure.clear(); self.campbell.draw()
                return
            # raise so _on_job_done paints a copyable "Test run failed" error card
            raise _FileRunError(err)
        self._fill_table(fr.breaths_table)
        self._draw_campbell(fr.breaths)
        self._update_qc_overview(fr)                  # P16 batch QC overview
        nr = getattr(result, "noise_report", None)
        if nr:
            self.render_noise_report(result)         # sets its own status incl. prop_decrease
            # the auto-selected suppression lives only in the report — write it back so
            # the spinbox AND the re-conditioned EMG views use the value the batch chose
            # (the core never mutates settings), then re-dispatch those EMG views. Only
            # the direct-call path carries a report now; the auto mechanics-only batch
            # does not, so it never triggers this redundant re-condition (the separate
            # 'noise' job owns that).
            chosen = nr.get("prop_decrease")
            if chosen is not None and self.state.settings.processing.emg.noise.auto_prop:
                self.state.settings.processing.emg.noise.prop_decrease = float(chosen)
            self._load_noise_params()                # reflect the chosen prop_decrease
            noise = self.state.settings.processing.emg.noise
            if noise.enabled and noise.auto_prop and self.state.settings.input.channels.emg:
                self._schedule("emg_all")
                self._schedule("emg_detail")
        else:
            n = len(fr.breaths_table)
            self._set_status(f"Test run OK: {n} breaths (nothing written)")

    def _fill_table(self, df):
        df = df.reset_index(drop=True)
        self.table.setRowCount(len(df)); self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))

    def _draw_campbell(self, breaths):
        pal = _plot_pal()
        fig = self.campbell.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])
        kept = [b for b in breaths.values() if not b["ignored"]]
        for b in kept:
            ax.plot(b["poes"], b["volume"], color=pal["mpl_loop"], lw=0.7, alpha=0.5, zorder=1)
        # P12: overlay the average breath bold, draw the elastic recoil (relaxation)
        # line EELV→EILV, and shade the inspiratory work between the Poes trace and it.
        self._overlay_campbell_work(ax, kept, pal)
        ax.axvline(0, color=pal["mpl_zeroline"], lw=0.8, zorder=0)
        ax.set_xlabel("Oesophageal pressure  Poes (cmH₂O)")
        ax.set_ylabel("Lung volume above end-expiration (L)")
        ax.set_title("Campbell diagram")
        fig.tight_layout()
        self.campbell.draw()
        self.btn_export_fig.setEnabled(True)         # a diagram now exists to export

    def _overlay_campbell_work(self, ax, kept, pal):
        """Draw the average PV loop + elastic recoil line + shaded inspiratory work,
        so the Campbell diagram *shows* the work of breathing, not just the loops.
        Defensive: any missing average field falls back to just the loops (no raise)."""
        if not kept:
            return
        b = kept[0]
        insp = b.get("inspiration", {})
        vavg, pavg = b.get("volumeavg"), b.get("poesavg")
        iv, ip = insp.get("volumeavg"), insp.get("poesavg")
        eelv, eilv = b.get("eelvavg"), b.get("eilvavg")   # each [volume, poes]
        try:
            import numpy as np
            if vavg is not None and pavg is not None and len(vavg):
                ax.plot(pavg, vavg, color="#2C6E9B", lw=2.0, zorder=3, label="average breath")
            if eelv is not None and eilv is not None:
                # elastic recoil line: straight line between the two volume endpoints
                ax.plot([eelv[1], eilv[1]], [eelv[0], eilv[0]], color=pal["mpl_target"],
                        ls="--", lw=1.3, zorder=4, label="elastic recoil")
                if iv is not None and ip is not None and len(iv) > 1:
                    iv = np.asarray(iv, float); ip = np.asarray(ip, float)
                    # Poes on the recoil line at each inspiratory volume
                    line_p = np.interp(iv, sorted([eelv[0], eilv[0]]),
                                       [eelv[1], eilv[1]] if eelv[0] <= eilv[0] else [eilv[1], eelv[1]])
                    ax.fill_betweenx(iv, ip, line_p, color="#2C6E9B", alpha=0.18,
                                     zorder=2, label="inspiratory work")
            wob = dict(b.get("wob") or {})
            if "wobtotal" in wob:
                ax.text(0.02, 0.98, f"WOB {wob['wobtotal']:.2f} J·min⁻¹",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        color=pal["mpl_loop"])
            ax.legend(loc="lower right", frameon=False, fontsize=7)
        except Exception:                       # pragma: no cover - overlay is best-effort
            pass

    # -- fidelity render ----------------------------------------------------
    def _draw_fidelity(self, nr):
        """Draw the EMG noise fidelity frontier from a noise ``report`` dict onto the
        fidelity canvas. Shared by the auto noise job and the manual test run."""
        frontier = nr.get("frontier", {})
        chosen = nr.get("prop_decrease")
        target = nr.get("fidelity_target", 0.8)
        emg_cols = list(self.state.settings.input.channels.emg)
        pal = _plot_pal()
        fig = self.fidelity_canvas.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])
        if frontier:
            props = sorted(frontier)
            nch = len(next(iter(frontier.values())))
            for ch in range(nch):
                lbl = f"EMGdi col {emg_cols[ch]}" if ch < len(emg_cols) else f"EMG ch {ch + 1}"
                ax.plot(props, [frontier[p][ch] for p in props], marker="o", ms=3, label=lbl)
            ax.axhline(target, color=pal["mpl_target"], ls=":", lw=1)
            if chosen is not None:
                ax.axvline(chosen, color=pal["mpl_error"], ls="--", lw=1.2, label=f"chosen = {chosen}")
            ax.set_xlabel("Noise-suppression strength (prop_decrease, 0–1)")
            ax.set_ylabel("EMG fidelity retained (1 = untouched)")
            ax.set_ylim(0, 1.02)
            ax.set_title("EMG noise: fidelity frontier")
            ax.legend(fontsize=7)
        fig.tight_layout()
        self.fidelity_canvas.draw()

    def render_noise_report(self, result):
        """Draw the fidelity frontier + status from a full batch result (with ECG
        suppression from its ok_files). Used by the manual test run."""
        nr = result.noise_report
        self._draw_fidelity(nr)
        chosen = nr.get("prop_decrease")
        target = nr.get("fidelity_target", 0.8)
        worst = min((r["fidelity"] for r in nr.get("channels", [])), default=float("nan"))
        msg = (f"Noise: prop_decrease {chosen} chosen "
               f"(worst-channel fidelity {worst:.2f} vs target {target:.2f})")
        for fr in result.ok_files.values():
            if getattr(fr, "ecg", None):
                msg += f" · ECG suppression {fr.ecg.get('suppression', float('nan')):.0%}"
                break
        self._set_status(msg)

    # -- auto noise/fidelity job (decoupled from the test run) --------------
    def _on_noise_result(self, report):
        self._apply_noise_report(report)

    def _apply_noise_report(self, nr):
        """Render the fidelity frontier, adopt the auto-selected suppression (so the
        spinbox + re-conditioned EMG use the value the sweep chose — the core never
        mutates settings), and re-dispatch the EMG conditioning views with it."""
        self._draw_fidelity(nr)
        chosen = nr.get("prop_decrease")
        noise = self.state.settings.processing.emg.noise
        if chosen is not None and noise.auto_prop:
            noise.prop_decrease = float(chosen)
        self._load_noise_params()                    # reflect the chosen prop in the spinbox
        if nr.get("frontier"):
            target = nr.get("fidelity_target", 0.8)
            worst = min((r["fidelity"] for r in nr.get("channels", [])), default=float("nan"))
            self._set_status(f"Noise: prop_decrease {chosen} chosen "
                             f"(worst-channel fidelity {worst:.2f} vs target {target:.2f}).")
        if noise.enabled and noise.auto_prop and self.state.settings.input.channels.emg:
            self._schedule("emg_all")                # re-condition with the chosen suppression
            self._schedule("emg_detail")

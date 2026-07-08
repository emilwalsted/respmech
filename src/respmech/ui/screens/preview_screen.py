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
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QProgressBar, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget,
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
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_mechanics_preview)

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

# reactive job kinds -> which panel overlays they own, and the spinner caption
_PANELS = {"mech": ["channels", "raw"], "batch": ["campbell", "fidelity"],
           "emg_all": ["result"], "emg_detail": ["detail"]}
_SPIN_TEXT = {"mech": "Loading channels…", "batch": "Running test…",
              "emg_all": "Conditioning channels…", "emg_detail": "Staging detail…"}
# human labels for status lines + panel error cards
_KIND_LABEL = {"mech": "Channel preview", "batch": "Test run",
               "emg_all": "EMG result", "emg_detail": "EMG detail"}


def _pen(colour, width=1):
    return pg.mkPen(colour, width=width)


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

        self.setStyleSheet(
            "#busyOverlay { background-color: rgba(18, 22, 28, 0.42); }"
            "#busyBox, #errorBox { background-color: palette(window);"
            " border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; }"
            "#errorBox { border-color: rgba(180,50,42,0.55); }"
            "#busyLabel, #errLabel { color: palette(window-text); font-weight: 600; }"
            "#errIcon { color: #E08A4F; font-size: 17px; font-weight: 700; }"
            "#errHint { color: rgba(147,164,183,0.95); font-size: 11px; }"
            "#infoBtn { background-color: #2C6E9B; color: white; border: none;"
            " border-radius: 12px; font-weight: 700; font-style: italic; }"
            "#infoBtn:hover { background-color: #3E8AC0; }")
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

    def __init__(self, state):
        super().__init__()
        self.state = state
        # reactive-job bookkeeping (replaces the old single self._thread/_worker)
        self._jobs = {}                  # kind -> _Job (the current owner)
        self._draining = set()           # superseded jobs, kept alive until self-cleanup
        self._tokens = {"mech": 0, "batch": 0, "emg_all": 0, "emg_detail": 0}
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
        self._build()
        # render dispatch by job kind (built after _build so the methods exist)
        self._RENDER = {
            "mech": self._render_preview,
            "batch": self._on_batch_result,
            "emg_all": self._on_emg_all_result,
            "emg_detail": self._on_emg_detail_result,
        }

    # -- construction -------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.file_combo = QComboBox()
        self.btn_refresh = QPushButton("Refresh files"); self.btn_refresh.clicked.connect(self.refresh_files)
        self.btn_preview = QPushButton("Preview channels"); self.btn_preview.clicked.connect(self._preview)
        self.btn_test = QPushButton("Test run this file"); self.btn_test.clicked.connect(self._test_run)
        if _theme is not None:
            _theme.make_primary(self.btn_test)
        bar.addWidget(QLabel("File:")); bar.addWidget(self.file_combo, 1)
        bar.addWidget(self.btn_refresh); bar.addWidget(self.btn_preview); bar.addWidget(self.btn_test)
        root.addLayout(bar)
        self.status = QLabel("Pick a file — everything runs automatically.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

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
            "detail": BusyOverlay(self._emg_mid),
            "fidelity": BusyOverlay(self.fidelity_canvas),
        }

        self._refresh_emg_channels()
        self._load_noise_params()
        self._ensure_noise_region()
        self._update_emg_tab_visibility()

        self.file_combo.currentTextChanged.connect(self._on_file_selected)
        self.refresh_files()

    def _build_mech_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)
        split = QSplitter(Qt.Vertical)
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.scene().sigMouseClicked.connect(self._on_plot_clicked)
        split.addWidget(self.plots)
        lower = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 0)
        lower.addWidget(self.table)
        self.campbell = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        lower.addWidget(self.campbell)
        self._mech_lower = lower
        split.addWidget(lower)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        v.addWidget(split)
        return w

    def _build_emg_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)

        # controls
        row = QHBoxLayout()
        self.emg_channel = QComboBox(); self.emg_channel.setToolTip("EMG channel for the single-channel detail below.")
        self.emg_channel.currentIndexChanged.connect(self._on_emg_channel_changed)
        self.btn_emg_preview = QPushButton("Preview conditioning")
        self.btn_emg_preview.setToolTip("Re-stage the selected channel raw ▸ ECG-removed ▸ noise-reduced.")
        self.btn_emg_preview.clicked.connect(self._start_emg_preview)
        self.btn_emg_result = QPushButton("Show result (all channels)")
        self.btn_emg_result.setToolTip("Re-condition every EMG channel; tick channels below to show them.")
        self.btn_emg_result.clicked.connect(self._show_emg_result)
        self.btn_use_region = QPushButton("Use selection as noise profile")
        self.btn_use_region.setToolTip("Set the shaded rest span as the noise reference for this file.")
        self.btn_use_region.clicked.connect(self._use_region_as_noise)
        self.btn_noise = QPushButton("Noise & fidelity preview")
        self.btn_noise.setToolTip("Re-run the test to rebuild the noise profile and fidelity frontier.")
        self.btn_noise.clicked.connect(self._start_noise_preview)
        row.addWidget(QLabel("Detail channel:")); row.addWidget(self.emg_channel)
        row.addWidget(self.btn_emg_preview); row.addWidget(self.btn_emg_result)
        row.addStretch(1)
        row.addWidget(self.btn_use_region); row.addWidget(self.btn_noise)
        v.addLayout(row)

        # noise-window options (active once a reference file is set)
        self.noise_opts = QWidget()
        nrow = QHBoxLayout(self.noise_opts); nrow.setContentsMargins(0, 0, 0, 0)
        self.noise_auto = QCheckBox("Auto-select suppression")
        self.noise_auto.setToolTip("Pick the strongest suppression keeping every channel ≥ the fidelity target.")
        self.noise_auto.toggled.connect(self._on_noise_param_changed)
        self.noise_prop = QDoubleSpinBox(); self.noise_prop.setRange(0.0, 1.0); self.noise_prop.setSingleStep(0.05); self.noise_prop.setDecimals(2)
        self.noise_prop.setToolTip("Manual suppression strength (0 = none, 1 = full) when auto is off.")
        self.noise_prop.valueChanged.connect(self._on_noise_param_changed)
        self.noise_target = QDoubleSpinBox(); self.noise_target.setRange(0.50, 0.99); self.noise_target.setSingleStep(0.05); self.noise_target.setDecimals(2)
        self.noise_target.setToolTip("Minimum fraction of inspiratory EMG power to retain.")
        self.noise_target.valueChanged.connect(self._on_noise_param_changed)
        self.noise_nstd = QDoubleSpinBox(); self.noise_nstd.setRange(0.0, 10.0); self.noise_nstd.setSingleStep(0.1); self.noise_nstd.setDecimals(1)
        self.noise_nstd.setToolTip("Gate threshold: mean + n_std·SD of the noise spectrum. Lower = gentler.")
        self.noise_nstd.valueChanged.connect(self._on_noise_param_changed)
        nrow.addWidget(QLabel("Noise window:")); nrow.addWidget(self.noise_auto)
        nrow.addWidget(QLabel("suppression")); nrow.addWidget(self.noise_prop)
        nrow.addWidget(QLabel("fidelity target")); nrow.addWidget(self.noise_target)
        nrow.addWidget(QLabel("n_std")); nrow.addWidget(self.noise_nstd)
        nrow.addStretch(1)
        v.addWidget(self.noise_opts)

        # result-channel picker
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Result channels:"))
        self.result_checks_holder = QWidget()
        self.result_checks_layout = QHBoxLayout(self.result_checks_holder)
        self.result_checks_layout.setContentsMargins(0, 0, 0, 0)
        rrow.addWidget(self.result_checks_holder); rrow.addStretch(1)
        v.addLayout(rrow)

        # plots
        split = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)
        self.emg_raw_plots = pg.GraphicsLayoutWidget()
        self.emg_result_plots = pg.PlotWidget()
        self.emg_result_plots.addLegend()
        self.emg_result_plots.setLabel("bottom", "Time (s)")
        self.emg_result_plots.setLabel("left", "Conditioned EMG (a.u.)")
        top.addWidget(self._titled("Raw EMG channels", self.emg_raw_plots))
        top.addWidget(self._titled("Conditioned result (selected channels)", self.emg_result_plots))
        split.addWidget(top)
        mid = QSplitter(Qt.Horizontal)
        self.emg_plots = pg.PlotWidget()
        self.emg_plots.addLegend()
        self.emg_plots.setLabel("bottom", "Time (s)"); self.emg_plots.setLabel("left", "EMG (a.u.)")
        self.emg_psd_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        mid.addWidget(self._titled("Detail channel — raw ▸ ECG-removed ▸ noise-reduced", self.emg_plots))
        mid.addWidget(self._titled("Detail PSD", self.emg_psd_canvas))
        self._emg_mid = mid
        split.addWidget(mid)
        self.fidelity_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        split.addWidget(self._titled("Noise fidelity frontier", self.fidelity_canvas))
        v.addWidget(split, 1)
        return w

    @staticmethod
    def _titled(title, widget):
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        lab = QLabel(title); lab.setProperty("status", "muted")
        lay.addWidget(lab); lay.addWidget(widget, 1)
        return box

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
        # rebuild result-channel checkboxes
        for _c, cb in self.result_checks:
            cb.setParent(None)
        self.result_checks = []
        for i, c in enumerate(cols):
            cb = QCheckBox(f"col {c}")
            cb.setChecked(i < 3)
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
            pattern = os.path.join(folder, s.input.files or "*.*")
            files = [os.path.basename(f) for f in sorted(glob.glob(pattern)) if os.path.isfile(f)]
            self.file_combo.addItems(files)
        self.file_combo.blockSignals(False)
        if prev in files:
            self.file_combo.setCurrentText(prev)
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
        self._request_autorun()

    def _current_file(self):
        name = self.file_combo.currentText()
        return os.path.join(self.state.settings.input.folder, name) if name else None

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
        self.btn_preview.setEnabled(has_file and not jr("mech"))
        self.btn_test.setEnabled(has_file and ok and not jr("batch"))
        self.btn_noise.setEnabled(has_file and ok and noise_on and bool(ref) and not jr("batch"))
        self.emg_channel.setEnabled(has_emg)
        self.btn_use_region.setEnabled(has_file and has_emg)
        self.btn_emg_preview.setEnabled(has_file and has_emg and not jr("emg_detail"))
        self.btn_emg_result.setEnabled(has_file and has_emg and not jr("emg_all"))
        # noise-window options only active with a reference file (Emil's requirement)
        self.noise_opts.setEnabled(bool(has_emg and noise_on and ref))
        if noise_on:
            self.noise_prop.setEnabled(bool(ref) and not self.noise_auto.isChecked())
        if not status:
            return
        if noise_on and not ref:
            self._set_status("Noise reduction is on — pick a reference file (Settings) or "
                             "select a rest region below and 'Use selection as noise profile'.")
        elif has_file and not ok:
            self._set_status(f"Settings incomplete: {why}")

    def _on_file_selected(self, name):
        self._update_actions()
        if name and name != self._previewed_file:
            self._request_autorun()

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
        for k in ("mech", "batch", "emg_all", "emg_detail"):
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
            worker = BatchWorker(snap, write=False, only_files=[os.path.basename(path)])
        elif kind == "emg_all":
            if not has_emg:
                return
            worker = EmgAllChannelsWorker(snap, path)
        elif kind == "emg_detail":
            if not has_emg:
                return
            ch = max(0, self.emg_channel.currentIndex())
            worker = EmgConditioningWorker(snap, path, ch)
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
        self._clear_panel_overlays("channels", "raw")   # dismiss any stale error card
        self.plots.clear()
        self._channel_plots = []
        prev = None
        n = len(_CHANNELS)
        for i, (key, label, colour) in enumerate(_CHANNELS):
            p = self.plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLabel("left", label)
            y = series[key]
            p.plot(t[:len(y)], y, pen=_pen(colour))
            if i == n - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
            self._channel_plots.append(p)
        self._draw_breath_overlays(spans, data["label_y"])
        self._render_raw_stack(data["emg"], fs)

        nign = sum(1 for _n, _a, _b, ig in spans if ig)
        self._previewed_file = data["name"]
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
        self.emg_raw_plots.clear()
        if emg.size == 0 or emg.ndim != 2 or emg.shape[1] == 0:
            self._ensure_noise_region()
            return
        cols = list(self.state.settings.input.channels.emg)
        t = np.arange(emg.shape[0]) / fs
        prev = None
        for i in range(emg.shape[1]):
            p = self.emg_raw_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            p.plot(t, emg[:, i], pen=_pen(_EMG_PENS[i % len(_EMG_PENS)]))
            if i == emg.shape[1] - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
        self._ensure_noise_region()

    # -- feature A: breath overlays + include/exclude ----------------------
    @staticmethod
    def _breath_brush(ignored):
        return pg.mkBrush(180, 50, 42, 70) if ignored else pg.mkBrush(44, 110, 155, 32)

    @staticmethod
    def _breath_label_color(ignored):
        return pg.mkColor(180, 50, 42) if ignored else pg.mkColor(90, 107, 122)

    def _draw_breath_overlays(self, spans, label_y=0.0):
        self._breath_spans = {n: (t0, t1) for (n, t0, t1, _ig) in spans}
        self._breath_regions = {n: [] for (n, _0, _1, _ig) in spans}
        self._breath_texts = {}
        for n, t0, t1, ignored in spans:
            for plot in self._channel_plots:
                reg = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                          brush=self._breath_brush(ignored), pen=pg.mkPen(None))
                reg.setZValue(-10)
                plot.addItem(reg)
                self._breath_regions[n].append(reg)
                plot.addLine(x=t0, pen=pg.mkPen((150, 165, 180), width=1, style=Qt.DashLine))
            if self._channel_plots:
                txt = pg.TextItem(str(n), color=self._breath_label_color(ignored), anchor=(0.5, 1.0))
                txt.setPos((t0 + t1) / 2.0, label_y)
                self._channel_plots[0].addItem(txt)
                self._breath_texts[n] = txt

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
        nexcl = len({b for e in excl if e.file == name for b in e.breaths} & set(self._breath_spans))
        self._set_status(
            f"{name}: breath {breath_no} {'excluded' if now_excluded else 'included'} "
            f"({nexcl}/{len(self._breath_spans)} excluded). Re-running the test…")
        # excluding a breath changes the test result -> re-run it
        QTimer.singleShot(0, lambda: self._schedule("batch"))
        return now_excluded

    # -- feature B: noise-profile region -----------------------------------
    def _ensure_noise_region(self):
        reg = self._noise_region
        if reg is None:
            ivals = self.state.settings.processing.emg.noise.reference_intervals
            try:                                     # a malformed settings file must not abort construction
                t0, t1 = (float(ivals[0][0]), float(ivals[0][1])) if ivals else (0.0, 1.0)
            except Exception:                        # noqa: BLE001 — any bad shape -> cosmetic default
                t0, t1 = 0.0, 1.0
            reg = pg.LinearRegionItem(values=(t0, t1), movable=True, brush=pg.mkBrush(44, 110, 155, 45))
            reg.setZValue(10)
            self._noise_region = reg
        if reg.scene() is None:
            self.emg_plots.addItem(reg)

    def _use_region_as_noise(self):
        name = self.file_combo.currentText()
        reg = self._noise_region
        if not name or reg is None:
            self._set_status("Draw a rest region on the detail plot, then apply.")
            return None
        t0, t1 = reg.getRegion()
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
        self.noise_reference_changed.emit(name, [[t0, t1]], False)
        self._set_status(
            f"Noise profile ← {name} [{t0:.2f}–{t1:.2f} s]: {span} samples ≈ {nframes} STFT frames "
            + ("(good)" if nframes >= 8 else "(short — pick a longer rest span)")
            + ". Enable 'Reduce EMG noise' to apply it.")
        self._update_actions()
        self._request_autorun()          # re-condition against the new reference
        return (t0, t1)

    # -- EMG detail / result / test triggers (manual re-runs) --------------
    def _on_emg_channel_changed(self, *_):
        if self._current_file():
            QTimer.singleShot(0, lambda: self._schedule("emg_detail"))

    def _start_emg_preview(self):
        self._schedule("emg_detail")

    def _show_emg_result(self):
        self._schedule("emg_all")

    def _start_noise_preview(self):
        self._schedule("batch")

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
        self.emg_plots.clear()
        self.emg_plots.plot(t, np.asarray(data["raw"]), pen=_pen((150, 165, 180)), name="raw")
        self.emg_plots.plot(t, np.asarray(data["ecg"]), pen=_pen((44, 110, 155)), name="ECG-removed")
        if data.get("noise_applied"):
            self.emg_plots.plot(t, np.asarray(data["noise"]), pen=_pen((31, 122, 77)), name="noise-reduced")
        self.emg_plots.setLabel("bottom", "Time (s)"); self.emg_plots.setLabel("left", "EMG (a.u.)")
        self._ensure_noise_region()

    def render_emg_psd(self, data):
        freqs = np.asarray(data["freqs"], dtype=float)
        band = data.get("band", (20.0, 250.0))
        fig = self.emg_psd_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        def _db(p):
            return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=float), 1e-30))

        ax.plot(freqs, _db(data["psd_raw"]), color="0.6", lw=1.0, label="raw")
        ax.plot(freqs, _db(data["psd_ecg"]), color="#2C6E9B", lw=1.2, label="ECG-removed")
        if data.get("noise_applied"):
            ax.plot(freqs, _db(data["psd_noise"]), color="#1F7A4D", lw=1.2, label="noise-reduced")
        ax.axvspan(band[0], band[1], color="#B7791F", alpha=0.12, label="EMG band 20–250 Hz")
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
            return
        t = np.asarray(data["t"])
        cols = data.get("cols", [])
        checked = {c for c, cb in self.result_checks if cb.isChecked()}
        for i, c in enumerate(cols):
            if c not in checked or i >= len(data["conditioned"]):
                continue
            self.emg_result_plots.plot(t, np.asarray(data["conditioned"][i]),
                                       pen=_pen(_EMG_PENS[i % len(_EMG_PENS)]), name=f"col {c}")

    # -- test run (Mechanics), synchronous ---------------------------------
    def _test_run(self):
        path = self._current_file()
        if not path:
            return
        name = os.path.basename(path)
        self._preview()          # show breaths on the graph for the test run
        try:
            self.state.settings.validate()
            result = run_batch(self.state.settings, only_files=[name])  # never writes
            fr = result.files.get(name)
            if fr is None or fr.error:
                raise RuntimeError(fr.error if fr else "The run produced no result for this file.")
            self._fill_table(fr.breaths_table)
            self._draw_campbell(fr.breaths)
        except Exception:              # noqa: BLE001 — surface a copyable error card
            detail = traceback.format_exc()
            self._set_status(f"Test run failed — {short_error(detail)}")
            for p in ("campbell", "fidelity"):
                self._overlays[p].show_error("Test run failed", detail)
            return
        self._clear_panel_overlays("campbell", "fidelity")   # dismiss any stale error card
        n = len(fr.breaths_table)
        msg = f"Test run OK: {n} breaths (nothing written)"
        ecg = getattr(fr, "ecg", None)
        if ecg:
            msg += f" · ECG suppression {ecg.get('suppression', float('nan')):.0%}"
        self._set_status(msg)

    def _on_batch_result(self, result):
        """Render the async full-run: table + Campbell (from the file result) and,
        when noise reduction ran, the fidelity frontier (from the batch result).
        Then, if suppression was auto-selected, re-condition the EMG views with the
        chosen value (the one real cross-job value dependency)."""
        cur = self.file_combo.currentText()
        fr = None
        if getattr(result, "files", None):
            fr = result.files.get(cur) or next(iter(result.files.values()), None)
        if fr is None or getattr(fr, "error", None):
            # raise so _on_job_done paints a copyable "Test run failed" error card
            raise _FileRunError(getattr(fr, "error", None)
                                or "The run produced no result for this file.")
        self._fill_table(fr.breaths_table)
        self._draw_campbell(fr.breaths)
        nr = getattr(result, "noise_report", None)
        if nr:
            self.render_noise_report(result)         # sets its own status incl. prop_decrease
            # the auto-selected suppression lives only in the report — write it
            # back so the spinbox AND the re-conditioned EMG views use the value
            # the batch actually chose (the core never mutates settings). Only do so
            # if auto-selection is still on, so a user who un-ticked it (or loaded
            # different settings) mid-run isn't overwritten.
            chosen = nr.get("prop_decrease")
            if chosen is not None and self.state.settings.processing.emg.noise.auto_prop:
                self.state.settings.processing.emg.noise.prop_decrease = float(chosen)
        else:
            n = len(fr.breaths_table)
            msg = f"Test run OK: {n} breaths (nothing written)"
            ecg = getattr(fr, "ecg", None)
            if ecg:
                msg += f" · ECG suppression {ecg.get('suppression', float('nan')):.0%}"
            self._set_status(msg)
        self._load_noise_params()                    # reflect the chosen prop_decrease
        noise = self.state.settings.processing.emg.noise
        if noise.enabled and noise.auto_prop and self.state.settings.input.channels.emg:
            self._schedule("emg_all")
            self._schedule("emg_detail")

    def _fill_table(self, df):
        df = df.reset_index(drop=True)
        self.table.setRowCount(len(df)); self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c in range(len(df.columns)):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))

    def _draw_campbell(self, breaths):
        fig = self.campbell.figure
        fig.clear()
        ax = fig.add_subplot(111)
        for b in breaths.values():
            if b["ignored"]:
                continue
            ax.plot(b["poes"], b["volume"], color="0.55", lw=0.9, zorder=1)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel("Oesophageal pressure  Poes (cmH₂O)")
        ax.set_ylabel("Lung volume above end-expiration (L)")
        ax.set_title("Campbell diagram")
        fig.tight_layout()
        self.campbell.draw()

    # -- fidelity render ----------------------------------------------------
    def render_noise_report(self, result):
        nr = result.noise_report
        frontier = nr.get("frontier", {})
        chosen = nr.get("prop_decrease")
        target = nr.get("fidelity_target", 0.8)
        emg_cols = list(self.state.settings.input.channels.emg)
        fig = self.fidelity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if frontier:
            props = sorted(frontier)
            nch = len(next(iter(frontier.values())))
            for ch in range(nch):
                lbl = f"EMGdi col {emg_cols[ch]}" if ch < len(emg_cols) else f"EMG ch {ch + 1}"
                ax.plot(props, [frontier[p][ch] for p in props], marker="o", ms=3, label=lbl)
            ax.axhline(target, color="0.35", ls=":", lw=1)
            if chosen is not None:
                ax.axvline(chosen, color="#B4322A", ls="--", lw=1.2, label=f"chosen = {chosen}")
            ax.set_xlabel("Noise-suppression strength (prop_decrease, 0–1)")
            ax.set_ylabel("EMG fidelity retained (1 = untouched)")
            ax.set_ylim(0, 1.02)
            ax.set_title("EMG noise: fidelity frontier")
            ax.legend(fontsize=7)
        fig.tight_layout()
        self.fidelity_canvas.draw()
        worst = min((r["fidelity"] for r in nr.get("channels", [])), default=float("nan"))
        msg = (f"Noise: prop_decrease {chosen} chosen "
               f"(worst-channel fidelity {worst:.2f} vs target {target:.2f})")
        for fr in result.ok_files.values():
            if getattr(fr, "ecg", None):
                msg += f" · ECG suppression {fr.ecg.get('suppression', float('nan')):.0%}"
                break
        self._set_status(msg)

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
import os
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QThread, QTimer, Signal
from PySide6.QtGui import QFont

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers
from respmech.ui import wheel as _wheel
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)

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
# sub-tab titles — a leading chevron marks the intended flow: Mechanics › ECG reduction ›
# noise reduction. Kept as constants so the insert code and the tests share one source.
_TAB_MECH = "Mechanics"
_TAB_ECG = "› EMG – ECG reduction"
_TAB_NOISE = "› EMG – noise reduction"

_PANELS = {"mech": ["channels", "raw"], "batch": ["table", "campbell"],
           "ecg": ["ecg_capture", "ecg_stack"],
           "emg_all": ["result"], "emg_detail": ["detail", "detail_psd"], "noise": ["fidelity"]}
_SPIN_TEXT = {"mech": "Loading channels…", "batch": "Running test…",
              "ecg": "Removing ECG…",
              "emg_all": "Conditioning channels…", "emg_detail": "Staging detail…",
              "noise": "Measuring fidelity…"}
# human labels for status lines + panel error cards
_KIND_LABEL = {"mech": "Channel preview", "batch": "Test run", "ecg": "ECG reduction",
               "emg_all": "EMG result", "emg_detail": "EMG detail",
               "noise": "Noise fidelity"}
# the kinds that run automatically (on file select / settings change / Refresh). The
# test run ('batch') is now automatic too, but MECHANICS-ONLY (no ECG/EMG work).
_AUTO_KINDS = ("mech", "batch", "ecg", "emg_all", "emg_detail", "noise")
# the kinds whose result depends on the SELECTED file. 'noise' is deliberately absent: the
# fidelity/noise profile is test-wide (built from the reference file + the whole input set),
# so switching the previewed file must NOT blank or rebuild it. See _begin_file_switch.
_FILE_KINDS = ("mech", "batch", "ecg", "emg_all", "emg_detail")


def _kinds_for_settings_path(path):
    """Which auto kinds a changed Settings field (dotted path, as produced by
    ``dataclasses.asdict``) can affect, so a settings edit recomputes only the impacted
    panels. GOLDEN-SAFE: this only scopes the PREVIEW; the batch/CLI always use the real
    Settings, never this map. Erring WIDE is safe (over-recompute); erring narrow risks a
    stale panel — so any field NOT classified below falls through to ALL kinds. The
    exhaustive per-kind field lists were derived + adversarially verified against each
    stage_* function; the coarse buckets here stay on the safe (wide) side of them."""
    # output / diagnostics and the optional pre-resample never surface in any preview panel
    if path == "output" or path.startswith("output.") or path.startswith("processing.sampling"):
        return frozenset()
    # the EMG channel SET: mechanics shows the raw EMG traces and all EMG/noise panels use
    # them; the mechanics test run strips EMG, so it is unaffected
    if path == "input.channels.emg":
        return frozenset(("mech", "ecg", "emg_all", "emg_detail", "noise"))
    if path.startswith("processing.emg"):
        if path.startswith("processing.emg.robust_peak"):
            # Writes-only, draws-nothing — like output.* above. The cardiac-gated peak adds
            # columns to the workbook; no preview panel plots a per-breath maximum, so there
            # is nothing to redraw. Without this rule it would fall through to the EMG
            # conditioning case below and re-run ECG + both EMG panels + the noise frontier
            # on every tick of a checkbox that changes none of them.
            return frozenset()
        if path == "processing.emg.outlier_rms_sd_limit":
            return frozenset(("batch",))          # a batch-table outlier flag only
        if path in ("processing.emg.normalization", "processing.emg.save_sound",
                    "processing.emg.plot_yscale", "processing.emg.noise_profile"):
            return frozenset(("emg_all", "emg_detail"))   # display/output-ish -> redraw EMG panels
        # EMG conditioning (remove_ecg / detect_channel / ecg_* / rms_window_s / remove_noise) +
        # noise.* params: the ECG-reduction tab + the EMG/noise panels (the mechanics test run
        # forces EMG + ECG + noise off, so it is unaffected)
        return frozenset(("ecg", "emg_all", "emg_detail", "noise"))
    # mechanics-only compute that feeds the test-run table/Campbell exclusively
    if (path.startswith("processing.wob") or path.startswith("processing.ptp")
            or path.startswith("processing.entropy") or path.startswith("processing.breath_counts")):
        return frozenset(("batch",))
    # breath segmentation feeds the mechanics panels AND — via the noise reference clip's
    # expiration segmentation and the auto_prop gather — the EMG conditioning + fidelity
    # panels (which build NoiseProfile.from_clip on a buffer-dependent clip). So a
    # segmentation edit must recompute all five, or the EMG traces go stale when auto_prop
    # is off (nothing else re-dispatches them). The ECG cache keeps the extra work cheap.
    if path.startswith("processing.segmentation"):
        return frozenset(_AUTO_KINDS)
    # volume drift/trend + explicit breath exclusion: the mechanics panels (+ noise, a
    # cache hit); these do NOT change the flow-based EMG reference masks, so the EMG panels
    # are left alone.
    if (path.startswith("processing.volume.correct") or path.startswith("processing.volume.trend")
            or path.startswith("processing.exclude_breaths")):
        return frozenset(("mech", "batch", "noise"))
    # channels core / format / volume inverse+integrate (all applied inside load()),
    # channels.entropy (validated in every load path), input.folder/files, and anything not
    # matched above -> recompute everything (safe default; never leaves a panel stale)
    return frozenset(_AUTO_KINDS)


def _changed_settings_paths(old, new):
    """Dotted paths of the leaves that differ between two Settings snapshots."""
    from dataclasses import asdict
    changed = set()

    def _walk(a, b, prefix):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                _walk(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        elif a != b:
            changed.add(prefix)

    _walk(asdict(old), asdict(new), "")
    return changed


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

# Cap the number of worker QThreads running AT ONCE. A file select fans out up to 5 heavy,
# largely GIL-bound staging jobs; starting them all together on a 2-core machine thrashes the
# GIL + oversubscribes OpenBLAS, starving the GUI thread so queued `finished` signals (and the
# UI) stall for seconds — badly on Windows (coarse GIL hand-off + no offscreen message pump),
# where it tips reactive tests past their timeout. Running at most 2 keeps the GUI responsive;
# the rest queue and start as slots free. See _pump_pool / _launch.
_MAX_ACTIVE = 2


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
            # padding/min-* MUST be reset: the global QPushButton padding (7px 16px) is
            # wider than this 24x24 circle and would clip the 'i' away entirely.
            "#infoBtn { background-color: " + acc + "; color: " + acc_fg + "; border: none;"
            " border-radius: 12px; font-weight: 700; font-style: italic;"
            " padding: 0; min-width: 0; min-height: 0; }"
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
    # emitted when the USER edits a Preview-owned setting that lands in the saved .toml
    # (noise/ECG params, breath exclusions). main_window routes it to the Setup screen's
    # dirty funnel so the title, close guard and Save all reflect the edit. Programmatic
    # fills (_load_*_params, sync_from_settings, the detect_channel seed) and app-derived
    # writes (the auto-picked prop_decrease) deliberately do NOT emit it.
    settings_edited = Signal()
    # emitted to process AND write just the previewed file (P19)
    process_file_requested = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        # reactive-job bookkeeping (replaces the old single self._thread/_worker)
        self._jobs = {}                  # kind -> _Job (the current owner)
        self._draining = set()           # superseded jobs (thread started), kept alive until they self-clean
        self._launch_queue = []          # _Jobs registered but not yet thread.start()ed (concurrency cap)
        self._active = set()             # _Jobs whose thread is started, compute not yet delivered
        self._reaping = set()            # _Jobs delivered; thread quitting — ref held until thread.finished
        self._tokens = {"mech": 0, "batch": 0, "ecg": 0, "emg_all": 0, "emg_detail": 0, "noise": 0}
        self._overlays = {}              # panel key -> BusyOverlay
        # Debounced auto-recompute: rapid edits (a spinbox drag / multi-keystroke entry)
        # restart one single-shot timer, collapsing the burst into ONE re-dispatch after a
        # short quiet period. The timer only fires while an event loop spins, so it stays
        # inert in headless tests (no thread starts synchronously) — same property as the
        # previous QTimer.singleShot(0) coalescing. _pending_kinds accumulates which panels
        # the pending run should recompute.
        self._autorun_timer = QTimer(self)
        self._autorun_timer.setSingleShot(True)
        self._autorun_timer.setInterval(300)         # ms
        self._autorun_timer.timeout.connect(self._run_autorun)
        self._pending_kinds = set()
        self._last_synced_settings = None   # snapshot for dependency-scoped invalidation (diff)
        self._noise_has_result = False   # has the fidelity panel a current result? (first-compute guard)
        self._previewed_file = None
        self._loading_noise = False
        self._loading_ecg = False
        self._ecg_capture_subplots = []   # per-channel PlotItems of the ECG-processed stack
        # breath-overlay interaction state (feature A)
        self._channel_plots = []
        self._breath_spans = {}
        self._breath_regions = {}
        self._breath_texts = {}
        self._mech_unpin = lambda: None   # detaches the mechanics label-pin slot
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
            "ecg": self._on_ecg_result,
            "emg_all": self._on_emg_all_result,
            "emg_detail": self._on_emg_detail_result,
            "noise": self._on_noise_result,
        }

    # -- construction -------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(11, 11, 11, 11)   # deterministic, matches Setup
        bar = QHBoxLayout()
        self.file_combo = QComboBox()
        # Size the selector to the recording names, not the window: names are short
        # (~100px at 13pt), but a stretch-factor row hands an uncapped combo ALL the
        # slack. "wide" caps the contents at 320px; a longer name elides in the closed
        # field but shows in full in the popup.
        self.file_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.file_combo.setProperty("formField", "wide")
        # P27: step through files without leaving the plots — buttons + PageUp/PageDown.
        self.btn_prev_file = QPushButton("◀")
        self.btn_next_file = QPushButton("▶")
        for b, tip in ((self.btn_prev_file, "Previous file (PgUp)"),
                       (self.btn_next_file, "Next file (PgDn)")):
            # The nav QSS gives the arrow 6px side air, and its min-width:0 overwrites
            # any code-side minimum at polish — the width simply tracks sizeHint, so the
            # glyph cannot clip. The max only stops layout slack ballooning the button.
            b.setProperty("nav", True)
            b.setMaximumWidth(34); b.setToolTip(tip)
        self.btn_prev_file.clicked.connect(lambda: self._step_file(-1))
        self.btn_next_file.clicked.connect(lambda: self._step_file(+1))
        # 'Refresh' recomputes every auto panel (not the test run); the file list itself
        # refreshes automatically when file-related settings change.
        self.btn_refresh_all = QPushButton("Refresh")
        self.btn_refresh_all.setToolTip("Recompute all preview panels for the current file.")
        self.btn_refresh_all.clicked.connect(self._refresh_all)
        bar.addWidget(QLabel("File:")); bar.addWidget(self.btn_prev_file)
        bar.addWidget(self.file_combo); bar.addWidget(self.btn_next_file)
        bar.addSpacing(12)   # detach Refresh a little from the ◀ file ▶ cluster
        bar.addWidget(self.btn_refresh_all)
        # park the slack AFTER Refresh: QPushButton's Minimum h-policy would otherwise
        # let the buttons balloon to fill the row once the combo stops stretching
        bar.addStretch(1)
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
        self._ecg_tab = self._build_ecg_tab()      # inserted between Mechanics + noise when EMG cols exist
        self._emg_tab = self._build_emg_tab()
        self.subtabs.addTab(self._mech_tab, _TAB_MECH)
        root.addWidget(self.subtabs, 1)

        # The control strips sit directly above wheel-zoomable plots, so an overshoot while
        # zooming used to land on a spin box and step it — and every ECG/noise parameter here
        # writes straight into the analysis, marks it modified and schedules a recompute.
        # file_combo was worse still: one notch silently switched the previewed recording.
        # Nothing on this screen scrolls, so the wheel should do nothing at all.
        self._wheel_guard = _wheel.swallow_wheel(
            root=self, extra=[self.subtabs.tabBar()])

        # one spinner overlay per panel (each fed by exactly one job kind)
        self._overlays = {
            "channels": BusyOverlay(self.plots),
            # one overlay per REAL panel: parenting this to the lower splitter instead put a
            # single spinner across the table+Campbell pair, reading as a third panel.
            "table": BusyOverlay(self.table),
            "campbell": BusyOverlay(self.campbell),
            "ecg_capture": BusyOverlay(self.ecg_capture_plot),
            "ecg_stack": BusyOverlay(self.ecg_processed_plots),
            "raw": BusyOverlay(self.emg_raw_plots),
            "result": BusyOverlay(self.emg_result_plots),
            "detail": BusyOverlay(self.emg_plots),
            "detail_psd": BusyOverlay(self.emg_psd_canvas),   # the detail job also renders the PSD
            "fidelity": BusyOverlay(self.fidelity_canvas),
        }

        self._refresh_emg_channels()
        self._refresh_ecg_channels()
        self._load_noise_params()
        self._load_ecg_params()
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
        # the per-breath table takes ~3/4 of the width; the Campbell diagram ~1/4 (default)
        lower.setStretchFactor(0, 3); lower.setStretchFactor(1, 1)
        lower.setSizes([720, 240])
        split.addWidget(lower)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        v.addWidget(split)
        # P16/P17: a thin action bar — batch QC overview + crosshair read-out + export
        bar = QHBoxLayout()
        self.qc_overview = QLabel("")
        self.qc_overview.setProperty("banner", True)   # box baked at first polish (theme.py)
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
        # hug the natural height — never stretch vertically into the plot area
        self.noise_opts.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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

        # plots — three rows: raw stack | detail channel on top, the full-width
        # conditioned result in the middle, the two small diagnostics below (see the
        # splitter assembly further down for why each panel sits where it does).
        _bg = _plot_pal()["bg"]
        self.emg_raw_plots = pg.GraphicsLayoutWidget()
        self.emg_raw_plots.setBackground(_bg)
        self.emg_result_plots = pg.PlotWidget()
        self._style_legend(self.emg_result_plots.addLegend())
        self.emg_result_plots.setBackground(_bg)
        self.emg_result_plots.setLabel("bottom", "Time (s)")
        self.emg_result_plots.setLabel("left", "EMG (a.u.)")   # "Conditioned" is in the panel title
        self.emg_plots = pg.PlotWidget()
        self._style_legend(self.emg_plots.addLegend())
        self.emg_plots.setBackground(_bg)
        self.emg_plots.setLabel("bottom", "Time (s)"); self.emg_plots.setLabel("left", "EMG (a.u.)")
        self.emg_psd_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))

        split = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(self._titled("Raw EMG channels", self.emg_raw_plots))
        top.addWidget(self._titled("Detail channel",       # pipeline stages are in the plot legend
                                   self.emg_plots, corner=self.emg_channel))
        split.addWidget(top)
        # Conditioned result spans the full width: it is EMG against time, so width is what
        # makes it readable. The two small diagnostics below share a row — the fidelity
        # frontier and the PSD are compact curves that gain nothing from the extra width.
        split.addWidget(self._titled("Conditioned result",   # the corner picker names the channels
                                     self.emg_result_plots, corner=self.result_checks_holder))
        self.fidelity_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(self._titled("Noise fidelity frontier (1 = untouched)", self.fidelity_canvas))
        mid.addWidget(self._titled("Detail PSD", self.emg_psd_canvas))
        self._emg_diag_row = mid          # the two small diagnostics share the bottom row
        split.addWidget(mid)
        v.addWidget(split, 1)
        # clicking a numbered breath in any EMG plot toggles its exclusion too
        self.emg_raw_plots.scene().sigMouseClicked.connect(self._on_emg_raw_clicked)
        self.emg_plots.scene().sigMouseClicked.connect(self._on_emg_detail_clicked)
        self.emg_result_plots.scene().sigMouseClicked.connect(self._on_emg_result_clicked)
        return w

    # -- EMG – ECG reduction tab -------------------------------------------
    def _build_ecg_tab(self):
        """Tune the ECG artefact removal: a top settings row (capture channel + params +
        Auto-suggest), the RAW capture channel with the detected R-peaks marked, and the
        ECG-processed channels below — all with the flow background + capture markers."""
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)

        def _cap(text, tip=""):
            la = QLabel(text); la.setProperty("status", "muted")
            if tip:
                la.setToolTip(tip)
            return la

        self.ecg_capture_channel = QComboBox()
        self.ecg_capture_channel.setToolTip("EMG channel used to detect the R-waves (heartbeats). "
                                            "Pick the channel with the clearest ECG and weakest EMG "
                                            "(often the middle channel).")
        self.remove_ecg = QCheckBox("Remove ECG")
        self.remove_ecg.setToolTip(_help_tip("processing.emg.remove_ecg",
                                   "Subtract an averaged ECG template from every EMG channel; off by default."))
        self.ecg_min_height = QDoubleSpinBox()
        self.ecg_min_height.setRange(0.0, 1_000_000.0); self.ecg_min_height.setDecimals(6); self.ecg_min_height.setSingleStep(0.0001)
        self.ecg_min_height.setToolTip(_help_tip("processing.emg.ecg_min_height",
                                       "Minimum height of an R-wave peak (in signal units) on the capture channel."))
        self.ecg_min_distance = QDoubleSpinBox()
        self.ecg_min_distance.setRange(0.05, 2.0); self.ecg_min_distance.setSingleStep(0.05); self.ecg_min_distance.setDecimals(3); self.ecg_min_distance.setSuffix(" s")
        self.ecg_min_distance.setToolTip(_help_tip("processing.emg.ecg_min_distance_s",
                                         "Minimum time between heartbeats (the refractory gap)."))
        self.ecg_min_width = QDoubleSpinBox()
        self.ecg_min_width.setRange(0.0, 0.1); self.ecg_min_width.setSingleStep(0.001); self.ecg_min_width.setDecimals(4); self.ecg_min_width.setSuffix(" s")
        self.ecg_min_width.setToolTip(_help_tip("processing.emg.ecg_min_width_s",
                                      "Minimum width of an R-wave peak."))
        self.ecg_window = QDoubleSpinBox()
        self.ecg_window.setRange(0.05, 1.0); self.ecg_window.setSingleStep(0.05); self.ecg_window.setDecimals(3); self.ecg_window.setSuffix(" s")
        self.ecg_window.setToolTip(_help_tip("processing.emg.ecg_window_s",
                                   "Width of the ECG template averaged and subtracted around each beat (QRS-T)."))
        for wdg in (self.remove_ecg,):
            wdg.toggled.connect(self._on_ecg_param_changed)
        for wdg in (self.ecg_min_height, self.ecg_min_distance, self.ecg_min_width, self.ecg_window):
            wdg.valueChanged.connect(self._on_ecg_param_changed)
        self.ecg_capture_channel.currentIndexChanged.connect(self._on_ecg_param_changed)
        self.btn_ecg_autosuggest = QPushButton("Auto-suggest settings")
        self.btn_ecg_autosuggest.setToolTip("Analyse the raw EMG channels: pick the channel with the "
                                            "clearest ECG and propose settings that capture the R-peaks "
                                            "and only those.")
        self.btn_ecg_autosuggest.clicked.connect(self._on_ecg_autosuggest)

        self.ecg_opts = QFrame(); self.ecg_opts.setObjectName("ecgChip")
        self.ecg_opts.setStyleSheet("#ecgChip { border: 1px solid rgba(128, 128, 128, 0.30); border-radius: 8px; }")
        self.ecg_opts.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)   # hug natural height
        row = QHBoxLayout(self.ecg_opts); row.setContentsMargins(11, 3, 11, 3); row.setSpacing(4)
        row.addWidget(_cap("Capture channel")); row.addSpacing(4); row.addWidget(self.ecg_capture_channel)
        row.addSpacing(12); row.addWidget(self.remove_ecg)
        row.addSpacing(10); row.addWidget(_cap("Min height", self.ecg_min_height.toolTip())); row.addWidget(self.ecg_min_height)
        row.addSpacing(10); row.addWidget(_cap("Min gap", self.ecg_min_distance.toolTip())); row.addWidget(self.ecg_min_distance)
        row.addSpacing(10); row.addWidget(_cap("Min width", self.ecg_min_width.toolTip())); row.addWidget(self.ecg_min_width)
        row.addSpacing(10); row.addWidget(_cap("Window", self.ecg_window.toolTip())); row.addWidget(self.ecg_window)

        strip = QHBoxLayout(); strip.setSpacing(10)
        strip.addWidget(self.ecg_opts); strip.addStretch(1); strip.addWidget(self.btn_ecg_autosuggest)
        v.addLayout(strip)

        _bg = _plot_pal()["bg"]
        self.ecg_capture_plot = pg.PlotWidget(); self.ecg_capture_plot.setBackground(_bg)
        self.ecg_capture_plot.setLabel("bottom", "Time (s)")
        # Just the unit: this panel is the 1/4-height slot of the splitter below (stretch 1:3,
        # ~71px of viewbox at the default window), and a pyqtgraph left label is rotated, so
        # its LENGTH runs along that height. The panel title names the channel.
        self.ecg_capture_plot.setLabel("left", "a.u.")
        self.ecg_processed_plots = pg.GraphicsLayoutWidget(); self.ecg_processed_plots.setBackground(_bg)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self._titled("Raw capture channel — detected R-peaks (▼)", self.ecg_capture_plot))
        split.addWidget(self._titled("ECG-processed EMG channels", self.ecg_processed_plots))
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 3)
        v.addWidget(split, 1)
        return w

    def _refresh_ecg_channels(self):
        """Populate the capture-channel combo; default it to the MIDDLE EMG channel (typically
        the weakest-EMG / clearest-ECG electrode) when the analysis has not configured ECG yet."""
        cols = list(self.state.settings.input.channels.emg)
        e = self.state.settings.processing.emg
        if len(cols) > 1 and int(e.detect_channel) == 0 and not e.remove_ecg:
            e.detect_channel = len(cols) // 2         # preview-only seed (golden reads the model directly)
        self.ecg_capture_channel.blockSignals(True)
        self.ecg_capture_channel.clear()
        for c in cols:
            self.ecg_capture_channel.addItem(f"EMG col {c}")
        if cols:
            self.ecg_capture_channel.setCurrentIndex(max(0, min(int(e.detect_channel), len(cols) - 1)))
        self.ecg_capture_channel.blockSignals(False)

    def _load_ecg_params(self):
        """Reflect the ECG settings into the tab's widgets without firing a recompute."""
        self._loading_ecg = True
        try:
            e = self.state.settings.processing.emg
            self.remove_ecg.setChecked(bool(e.remove_ecg))
            self.ecg_min_height.setValue(float(e.ecg_min_height))
            self.ecg_min_distance.setValue(float(e.ecg_min_distance_s))
            self.ecg_min_width.setValue(float(e.ecg_min_width_s))
            self.ecg_window.setValue(float(e.ecg_window_s))
            cols = list(self.state.settings.input.channels.emg)
            if cols:
                self.ecg_capture_channel.setCurrentIndex(max(0, min(int(e.detect_channel), len(cols) - 1)))
        finally:
            self._loading_ecg = False

    def _on_ecg_param_changed(self, *_):
        if self._loading_ecg:
            return
        e = self.state.settings.processing.emg
        e.remove_ecg = self.remove_ecg.isChecked()
        e.ecg_min_height = float(self.ecg_min_height.value())
        e.ecg_min_distance_s = float(self.ecg_min_distance.value())
        e.ecg_min_width_s = float(self.ecg_min_width.value())
        e.ecg_window_s = float(self.ecg_window.value())
        e.detect_channel = max(0, self.ecg_capture_channel.currentIndex())
        self.settings_edited.emit()      # ECG params land in the .toml -> mark dirty
        # ECG removal feeds the EMG/noise panels too, so recompute those alongside this tab.
        self._request_autorun({"ecg", "emg_all", "emg_detail", "noise"})

    def _on_ecg_autosuggest(self):
        """Analyse the raw EMG and fill in the ECG settings (channel + params), then recompute."""
        path = self._current_file()
        if not path:
            self._set_status("Pick a file first, then Auto-suggest can analyse its EMG.")
            return
        try:
            from respmech.ui.workers import stage_raw_emg
            from respmech.core.emg import suggest_ecg_settings
            data = stage_raw_emg(self.state.settings, path)
            matrix = np.column_stack(data["raw"]) if data.get("raw") else np.empty((0, 0))
            fs = int(self.state.settings.input.format.sampling_frequency or data.get("fs") or 1)
            sug = suggest_ecg_settings(matrix, fs)
        except Exception:                              # noqa: BLE001
            self._set_status(f"Auto-suggest failed — {short_error(traceback.format_exc())}")
            return
        e = self.state.settings.processing.emg
        e.detect_channel = int(sug["detect_channel"])
        e.ecg_min_height = float(sug["ecg_min_height"])
        e.ecg_min_distance_s = float(sug["ecg_min_distance_s"])
        e.ecg_min_width_s = float(sug["ecg_min_width_s"])
        e.ecg_window_s = float(sug["ecg_window_s"])
        self._load_ecg_params()                        # reflect into the widgets (guarded)
        self.settings_edited.emit()      # Auto-suggest rewrote the ECG settings -> mark dirty
        diag = sug.get("_diagnostics", {})
        cols = list(self.state.settings.input.channels.emg)
        col = cols[e.detect_channel] if e.detect_channel < len(cols) else e.detect_channel
        if diag.get("confidence") == "low":
            self._set_status("Auto-suggest: no clear ECG found — set conservative defaults on the "
                             f"middle channel (col {col}). Tune the settings and watch the capture.")
        else:
            bpm = diag.get("est_bpm")
            self._set_status(f"Auto-suggest: capture on col {col}"
                             + (f" (~{bpm:.0f} bpm)" if bpm else "")
                             + ". Review the ▼ markers and tune if needed.")
        self._request_autorun({"ecg", "emg_all", "emg_detail", "noise"})

    def _on_ecg_result(self, data):
        """Render the ECG-reduction tab: the raw capture channel + detected R-peaks, and the
        ECG-processed channels below — each with the flow background + capture markers."""
        pal = _plot_pal()
        t = np.asarray(data["t"])
        peaks = data.get("peaks", [])
        self.ecg_capture_plot.clear()
        self.ecg_capture_plot.plot(t, np.asarray(data["raw_capture"]), pen=_pen(pal["raw_trace"]))
        add_flow_background(self.ecg_capture_plot.getPlotItem(), t, data.get("flow"), pal)
        add_ecg_capture_markers(self.ecg_capture_plot.getPlotItem(), peaks, pal)
        self._limit_x(self.ecg_capture_plot.getPlotItem(), t)

        self.ecg_processed_plots.clear()
        self._ecg_capture_subplots = []
        cols = data.get("cols", [])
        proc = data.get("processed", [])
        cycle = pal["emg_cycle"]
        prev = None
        for i, ch in enumerate(proc):
            p = self.ecg_processed_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            if _theme is not None:
                _theme.align_left_axis(p)
            p.plot(t, np.asarray(ch), pen=_pen(cycle[i % len(cycle)]))
            add_flow_background(p, t, data.get("flow"), pal)
            add_ecg_capture_markers(p, peaks, pal)
            if i == len(proc) - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
            self._limit_x(p, t)
            self._ecg_capture_subplots.append(p)

        npk = int(np.asarray(peaks).size)
        col = data.get("detect_col", "?")
        state = "ECG removed" if data.get("ecg_applied") else "capture preview (removal is OFF for the run)"
        if data.get("ecg_error"):
            self._set_status(f"ECG reduction — capture on col {col}: {short_error(data['ecg_error'])}")
        else:
            self._set_status(f"ECG reduction — capture on col {col}: {npk} R-peaks · {state}.")

    @staticmethod
    def _legend_one_row(plot, offset=(10, 4)):
        """Lay a plot's legend out as a single horizontal row tucked under the top edge.

        ``plot`` must be the PlotItem, never the PlotWidget: PlotWidget.__getattr__ only
        forwards callables, so a widget yields legend=None and this silently does nothing.

        The column count has to be re-applied on every render, not just at construction:
        PlotItem.clear() removes each entry from the legend, the following plot(name=...)
        calls re-add them, and the count varies (the detail plot gains 'noise-reduced' only
        when noise conditioning ran; the result plot has one entry per ticked channel).
        columnCount == entry count is what keeps it to one row.
        """
        leg = getattr(plot, "legend", None)
        if leg is None:
            return
        try:
            n = len(leg.items)
            if not n:
                return                       # setColumnCount(0) divides by zero
            leg.setColumnCount(n)
            leg.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=offset)
        except Exception:                    # noqa: BLE001 — legend chrome is cosmetic
            pass

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
        # The panel's own margin is the single source of air: it keeps the title, the plot
        # and the right-pinned corner widget (channel selectors) all off the panel edge, so
        # nothing butts against the neighbouring panel or the nav.
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(8, 4, 8, 6); lay.setSpacing(2)
        header = QHBoxLayout(); header.setContentsMargins(0, 0, 0, 0); header.setSpacing(8)
        lab = QLabel(title); lab.setProperty("status", "muted")
        header.addWidget(lab); header.addStretch(1)
        if corner is not None:
            header.addWidget(corner, 0, Qt.AlignRight | Qt.AlignVCenter)
        lay.addLayout(header); lay.addWidget(widget, 1)
        return box

    def _align_noise_strip(self):
        """Match the 'Set noise profile' button to the noise chip's height so the strip
        reads as one aligned band. Keyed off the chip's *sizeHint* (its natural, laid-out
        height), never its live allocated height: the button's minimum height feeds back
        into the strip's row height, so reading the live height would let the chip balloon
        a little more on every tab switch (the chip is clamped to Maximum vertically, but
        sizeHint keeps this idempotent regardless)."""
        h = self.noise_opts.sizeHint().height()
        if h > 1 and self.btn_set_noise.minimumHeight() != h:
            self.btn_set_noise.setMinimumHeight(h)

    # -- EMG tab visibility / channels -------------------------------------
    def _update_emg_tab_visibility(self):
        """Show the two EMG sub-tabs only when EMG channels exist, in the fixed pipeline order
        Mechanics(0) › EMG–ECG reduction(1) › EMG–noise reduction(2)."""
        has_emg = bool(self.state.settings.input.channels.emg)
        if has_emg:
            if self.subtabs.indexOf(self._ecg_tab) < 0:
                self.subtabs.insertTab(1, self._ecg_tab, _TAB_ECG)     # right after Mechanics(0)
            if self.subtabs.indexOf(self._emg_tab) < 0:
                self.subtabs.insertTab(2, self._emg_tab, _TAB_NOISE)   # after the ECG tab(1)
        else:
            for tab in (self._emg_tab, self._ecg_tab):
                idx = self.subtabs.indexOf(tab)
                if idx >= 0:
                    self.subtabs.removeTab(idx)

    def _refresh_emg_channels(self):
        cols = list(self.state.settings.input.channels.emg)
        cur = self.emg_channel.currentIndex()
        self.emg_channel.blockSignals(True)
        self.emg_channel.clear()
        for c in cols:
            self.emg_channel.addItem(f"EMG col {c}")
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
        self.settings_edited.emit()      # noise params land in the .toml -> mark dirty
        self._update_actions()
        self._request_autorun()          # re-condition with the new noise params

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- file list (reactive) ----------------------------------------------
    def refresh_files(self):
        s = self.state.settings
        # drop the preview caches only when the input folder/mask ACTUALLY changes (the file
        # set may differ) — not on every entry into the Preview tab, which would defeat the
        # cross-revisit reuse. Per-file freshness tokens in the keys guard in-place edits.
        fm = ((s.input.folder or ""), (s.input.files or ""))
        if fm != getattr(self, "_cache_fm", None):
            from respmech.ui.screens import _preview_cache
            _preview_cache.clear_all()
            self._cache_fm = fm
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
        self._refresh_ecg_channels()
        self._load_noise_params()
        self._load_ecg_params()
        self._update_emg_tab_visibility()
        self._update_actions()
        # Dependency-scoped invalidation: diff the settings against the last-synced snapshot
        # and recompute ONLY the panels whose inputs actually changed. An EMG-only edit no
        # longer re-runs the mechanics preview + test run; a mechanics-only edit no longer
        # re-conditions EMG or rebuilds the noise profile. The first sync (no baseline) and
        # any unclassified field recompute everything, so a panel can never be left stale.
        cur = self.state.settings
        if self._last_synced_settings is None:
            kinds = set(_AUTO_KINDS)
        else:
            kinds = set()
            for p in _changed_settings_paths(self._last_synced_settings, cur):
                kinds |= _kinds_for_settings_path(p)
        self._last_synced_settings = copy.deepcopy(cur)
        if not kinds:
            return                   # nothing preview-relevant changed -> leave panels as they are
        self._cancel_inflight(kinds)  # abort only the now-stale panels' jobs
        self._request_autorun(kinds)

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

    def _invalidate_inflight(self, kinds=None):
        """Bump the token of each targeted kind so any in-flight job for it is dropped by
        the acceptance check in _on_job_done when it lands — even if it finished in the
        same event-loop iteration as the change, before the (debounced) autorun bumps
        tokens. A superseded job still clears its own spinner. ``kinds`` defaults to ALL:
        on a file switch pass only the file-dependent kinds so the still-valid, test-wide
        noise job is not needlessly invalidated."""
        for k in (self._tokens if kinds is None else kinds):
            if k in self._tokens:
                self._tokens[k] += 1

    def _cancel_inflight(self, kinds=None):
        """Abort the in-flight jobs of ``kinds`` (default all): cancel each worker
        (cooperative), invalidate its token so a late result is dropped, and move it to
        draining so its thread is still joined on completion. Used on a settings change /
        file switch / Refresh — a file switch scopes to the file-dependent kinds so the
        test-wide noise job keeps running."""
        target = set(_AUTO_KINDS if kinds is None else kinds)
        for kind, job in list(self._jobs.items()):
            if kind not in target:
                continue
            if job in self._launch_queue:
                self._launch_queue.remove(job)         # never started -> drop (no thread/finished)
                self._clear_panel_overlays(*_PANELS[kind])   # ...and its spinner (no _on_job_done to stop it)
            else:
                if hasattr(job.worker, "cancel"):
                    try:
                        job.worker.cancel()
                    except Exception:                  # noqa: BLE001
                        pass
                self._draining.add(job)                # still running -> joined on completion
            del self._jobs[kind]
        self._invalidate_inflight(target)

    def _noise_applicable(self):
        """True when the noise/fidelity job would produce a panel (EMG channels present,
        noise enabled, a reference file set) — same gate as _schedule('noise')."""
        ncfg = self.state.settings.processing.emg.noise
        return bool(self.state.settings.input.channels.emg) and ncfg.enabled and bool(ncfg.reference_file)

    def _begin_file_switch(self):
        """Shared cleanup when the current file changes: blank the FILE-dependent panels,
        abort their in-flight jobs, and re-dispatch them for the new file. The test-wide
        noise/fidelity panel is left intact — switching the previewed file cannot change
        its result — so it is neither blanked nor rebuilt, EXCEPT on its first compute
        (no result yet). Reached from an interactive combo change and from refresh_files
        silently switching the current file when the previous one vanished."""
        kinds = set(_FILE_KINDS)
        if self._noise_applicable() and not self._noise_has_result:
            kinds.add("noise")                 # first compute only; an existing result survives the switch
        self._clear_file_panels("noise" in kinds)
        self._cancel_inflight(kinds)           # abort only the file-dependent jobs (noise keeps running)
        self._request_autorun(kinds)

    def _clear_file_panels(self, include_noise=False):
        """Blank the FILE-dependent panels for a file switch, leaving the test-wide
        fidelity panel (and its last result) intact unless ``include_noise`` (a first
        compute is being scheduled)."""
        self.plots.clear(); self._channel_plots = []
        self.table.setRowCount(0); self.table.setColumnCount(0)
        self.campbell.figure.clear(); self.campbell.draw()
        self.ecg_capture_plot.clear(); self.ecg_processed_plots.clear(); self._ecg_capture_subplots = []
        self.emg_raw_plots.clear()
        self.emg_plots.clear()
        self.emg_psd_canvas.figure.clear(); self.emg_psd_canvas.draw()
        if include_noise:
            self.fidelity_canvas.figure.clear(); self.fidelity_canvas.draw()
            self._noise_has_result = False
        self._reset_breath_state()   # clears _bov/_breaths/_emg_all/result plot/_previewed_file
        self._ensure_noise_region()  # re-attach the rest-region selector to the cleared detail plot
        panels = [p for k in _FILE_KINDS for p in _PANELS[k]]
        if include_noise:
            panels += _PANELS["noise"]
        self._clear_panel_overlays(*panels)   # dismiss stale spinners/cards on the cleared panels

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
        self.ecg_capture_plot.clear(); self.ecg_processed_plots.clear(); self._ecg_capture_subplots = []
        self.emg_raw_plots.clear()
        self.emg_plots.clear()
        self.emg_psd_canvas.figure.clear(); self.emg_psd_canvas.draw()
        self.fidelity_canvas.figure.clear(); self.fidelity_canvas.draw()
        self._noise_has_result = False
        self._reset_breath_state()   # clears _bov/_breaths/_emg_all/result plot/_previewed_file
        self._ensure_noise_region()  # re-attach the rest-region selector to the cleared detail plot
        self._clear_panel_overlays(*self._overlays)   # dismiss stale spinners / error cards

    def _on_file_selected(self, name):
        self._update_actions()
        if name and name != self._previewed_file:
            self._begin_file_switch()

    # -- reactive scheduling ------------------------------------------------
    def _request_autorun(self, kinds=None):
        """Coalesce change events into a single re-dispatch after a short quiet period
        (~300 ms). Restarting the single-shot timer on each edit debounces a spinbox drag
        / multi-keystroke entry into ONE recompute. ``kinds`` scopes which panels the
        pending run recomputes (default: all auto kinds); pending kinds accumulate across
        edits until the timer fires. Never starts a thread synchronously — the timer only
        fires while an event loop spins, so this is inert in headless tests."""
        self._pending_kinds |= set(_AUTO_KINDS if kinds is None else kinds)
        self._autorun_timer.start()          # start() on a running single-shot timer restarts it

    def _run_autorun(self):
        kinds = self._pending_kinds or set(_AUTO_KINDS)
        self._pending_kinds = set()
        self._schedule_all(kinds)

    def _schedule_all(self, kinds=None):
        for k in (_AUTO_KINDS if kinds is None else kinds):
            if k in _AUTO_KINDS:             # ignore anything that is not an auto kind
                self._schedule(k)

    def _schedule(self, kind):
        """Evaluate the kind's gate, build its worker, and (only if it will
        actually launch) bump the kind's token and start it concurrently. The
        token is bumped LAST so a gated-out reschedule cannot supersede — and
        discard the result of — an in-flight job it never replaced."""
        path = self._current_file()
        if not path:
            return
        # EVERY kind loads the recording, and every load resolves the channel columns — so an
        # incomplete mapping has to gate them all, not just "batch". This used to be masked by
        # the Setup spin boxes, whose minimum of 1 meant a required channel could never be
        # unset; with the assignment dialog as the sole writer it genuinely can be, and each
        # ungated kind would render a panel full of raw loader traceback instead.
        ok, why = self._settings_ok()
        if not ok:
            self._clear_panel_overlays(*_PANELS[kind])
            self._set_status(f"Settings incomplete: {why}")
            return
        has_emg = bool(self.state.settings.input.channels.emg)
        # snapshot the settings on the GUI thread so a worker never reads them while
        # the GUI mutates them in place (Settings is a plain, deep-copyable model)
        snap = copy.deepcopy(self.state.settings)
        worker = None
        if kind == "mech":
            worker = FnWorker(stage_mechanics_preview, snap, path)
        elif kind == "ecg":
            if not has_emg:
                self._clear_panel_overlays(*_PANELS[kind])   # gated out -> drop a stale spinner/card
                return
            worker = FnWorker(stage_ecg_reduction, snap, path)
        elif kind == "batch":
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
                self._noise_has_result = False
                return                                 # no fidelity without a noise reference
            worker = FnWorker(stage_noise_fidelity, snap)
        if worker is None:
            return
        self._tokens[kind] += 1
        self._launch(kind, self._tokens[kind], worker)

    def _launch(self, kind, token, worker):
        old = self._jobs.pop(kind, None)
        if old is not None:
            if old in self._launch_queue:
                # never started -> no running thread (no GC hazard) and its `finished` would
                # never fire, so drop it rather than draining (which would wedge the drain)
                self._launch_queue.remove(old)
            else:
                if hasattr(old.worker, "cancel"):
                    try:
                        old.worker.cancel()
                    except Exception:              # noqa: BLE001
                        pass
                self._draining.add(old)            # still running -> keep referenced until it self-cleans
        for p in _PANELS[kind]:
            self._overlays[p].start(_SPIN_TEXT[kind])
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        job = _Job(kind, token, thread, worker)
        self._jobs[kind] = job
        # Worker -> GUI delivery is EXPLICITLY queued. AutoConnection decides
        # queued-vs-direct at emit time, and under a fast-emit race it can resolve a
        # bound-method connection to DIRECT — running the render and the overlay
        # stop()/hide() on the worker thread, which starts the spinner QProgressBar's
        # style-animation timer cross-thread ("QBasicTimer::start: Timers cannot be
        # started from another thread" on cocoa). Qt.QueuedConnection pins delivery to
        # the GUI thread regardless of timing; sender() still resolves under it.
        # (Receivers must still be bound methods of self, never lambdas — a lambda has
        # no receiver QObject for the queued delivery to land on.)
        worker.finished.connect(self._job_finished, Qt.QueuedConnection)
        if hasattr(worker, "failed"):
            worker.failed.connect(self._job_failed_slot, Qt.QueuedConnection)
        thread.finished.connect(self._release_thread, Qt.QueuedConnection)   # non-blocking cleanup when the thread exits exec()
        self._launch_queue.append(job)                  # start it when a concurrency slot is free
        self._pump_pool()
        self._update_actions(status=False)

    def _pump_pool(self):
        """Start queued jobs up to the concurrency cap. Called after a launch and after each
        job is delivered (freeing a slot), so at most _MAX_ACTIVE worker threads run at once."""
        while self._launch_queue and len(self._active) < _MAX_ACTIVE:
            job = self._launch_queue.pop(0)
            self._active.add(job)
            job.thread.start()

    def _release_thread(self):
        """Queued to the GUI thread by thread.finished: the thread has exited its event loop,
        so its strong ref can be dropped + deletion scheduled. NEVER runs while the thread is
        alive (that is why the ref is held in _reaping until here) — preserving the crash-safe
        'never GC a running QThread' invariant without a blocking join on the GUI thread."""
        thread = self.sender()
        for job in list(self._reaping):
            if job.thread is thread:
                self._reaping.discard(job)
                job.thread.deleteLater()
                break

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
        # NON-BLOCKING teardown: post a quit and hold a strong ref in _reaping until the
        # thread's `finished` fires (_release_thread) — the GUI thread never blocks in a join,
        # so sibling jobs' queued `finished` signals (and the UI) keep flowing. Freeing this
        # slot lets the next queued job start.
        job.thread.quit()
        self._active.discard(job)
        self._reaping.add(job)
        if self._jobs.get(job.kind) is job:
            del self._jobs[job.kind]
        self._draining.discard(job)
        self._pump_pool()
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
        if job.kind == "noise":
            self._noise_has_result = True   # the fidelity panel now holds a current result
        self._update_actions(status=False)

    def shutdown(self, wait_ms=5000):
        """Cancel and join every worker thread (current + draining). Called from
        MainWindow.closeEvent so nothing is destroyed while still running.

        A QThread whose worker is still mid-compute cannot be interrupted (the
        staging/`run_batch` calls are monolithic), so we join with a budget and,
        for any thread that will not stop in time, keep it referenced forever in
        the module-level parking list rather than let CPython GC a running
        QThread (which would call std::terminate and abort the process)."""
        # Disarm the debounced auto-run FIRST: a torn-down screen must never keep a loaded
        # 300ms timer that could later fire into another screen's event loop and launch real,
        # never-cancelled jobs on a dead screen (a cross-test/tab thread leak). Clearing the
        # pending set makes a stray timeout that already slipped through a no-op.
        self._autorun_timer.stop()
        self._pending_kinds.clear()
        self._launch_queue.clear()                     # unstarted jobs: no thread to join
        # a superseded-running job can be in both _draining and _active; a delivered one in
        # _reaping (its thread may still be quitting) — join every started thread exactly once
        jobs = list(set(self._jobs.values()) | self._draining | self._active | self._reaping)
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
        self._active.clear()
        self._reaping.clear()

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
        self._render_raw_stack(data["emg"], fs, data.get("emg_flow"))
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

    def _render_raw_stack(self, emg, fs, flow=None):
        """Draw the stacked raw EMG channels and keep the noise region alive. A discrete
        full-length flow silhouette is superimposed behind each channel so EMG activity can
        be read against the respiration cycle (``flow`` is untrimmed, aligned to the emg)."""
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
            add_flow_background(p, t, flow, _plot_pal())   # discrete respiration reference, behind
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

    def _breath_text(self, num, ignored):
        """A breath-number TextItem, sized for the SHORT stacked mechanics channel plots
        (~46 px of data area each): at the 13 pt app font the box is 23 px, so the headroom
        needed to show it swallows half the plot.

        Both steps matter, and the second is the one that counts: the default box is 23 px
        almost entirely because QTextDocument adds a 4 px margin on every side — setting a
        smaller font ALONE leaves it at 23 px. Font 9 pt + documentMargin 0 gives ~15 px."""
        txt = pg.TextItem(self._breath_label(num),
                          color=self._breath_label_color(ignored), anchor=(0.5, 0.0))
        f = QFont(); f.setPointSizeF(9.0)
        txt.setFont(f)
        txt.textItem.document().setDocumentMargin(0)
        txt.updateTextPos()
        return txt

    @staticmethod
    def _label_px(texts, default=25.0, margin=4.0):
        """The tallest breath label's rendered height in pixels (+ a little breathing room),
        measured rather than assumed — a TextItem is taller than its font size."""
        try:
            h = max(t.boundingRect().height() for t in (texts.values() if hasattr(texts, "values") else texts))
            return float(h) + margin if h > 0 else default + margin
        except Exception:                        # noqa: BLE001 — cosmetic
            return default + margin

    def _label_headroom(self, plot, label_px=25.0, frac=0.22):
        """Expand the label-carrying plot's y range upward so the breath labels — pinned
        at the view top, hanging down into the view — start over an empty band instead of
        sitting on the signal. Re-fits y to the DATA first so repeated repaints of a
        persistent plot (detail / result) don't compound the headroom.

        The headroom is sized in PIXELS, not as a fraction of the data span: a TextItem is
        a fixed pixel height whatever the scale, so on a short plot a fixed fraction leaves
        less band than the label needs (five stacked mechanics channels leave each ~70 px,
        where 22% ≈ 15 px — less than the label). Solving
        ``new_span = span + label_px * new_span / height_px`` for the exact expansion makes
        the label occupy ``label_px`` at the top at any height; ``frac`` is only the
        fallback when the plot has no laid-out height yet."""
        try:
            vb = plot.getViewBox()
            vb.enableAutoRange(y=True)           # snap y back to the data extent…
            vb.updateAutoRange()                 # …(not the previous, already-expanded view)
            (x0, x1), (y0, y1) = vb.viewRange()
            span = y1 - y0
            if span <= 0:
                return
            h_px = float(vb.height() or 0.0)
            extra = (span * (label_px / (h_px - label_px))
                     if h_px > label_px + 6 else frac * span)
            vb.setYRange(y0, y1 + extra, padding=0)   # (this disables y auto again)
        except Exception:                        # noqa: BLE001 — cosmetic
            pass

    def _pin_breath_labels(self, plot, txt_map):
        """Keep the breath-number labels pinned to the TOP of the visible view under
        y-zoom/pan — the same behaviour as the red capture marks: a sigYRangeChanged
        slot that only setPos()es the existing TextItems (wheel-zoom fires this
        continuously, so nothing is re-created) and self-disconnects once its labels
        leave the view. The labels MUST be added with ignoreBounds=True: a view-pinned
        item that still fed childrenBounds would re-inflate every autorange pass.
        Returns an unpin callable for eager teardown on repaint."""
        vb = plot.getViewBox()
        texts = list(txt_map.values())
        if vb is None or not texts:
            return lambda: None

        def _unpin():
            try:
                vb.sigYRangeChanged.disconnect(_reposition)
            except Exception:                          # noqa: BLE001
                pass

        def _reposition(*_):
            try:
                if texts[0].getViewBox() is None:      # repainted/cleared -> self-remove
                    _unpin()
                    return
                top = vb.viewRange()[1][1]
                for t in texts:
                    t.setPos(t.pos().x(), top)
            except Exception:                          # noqa: BLE001 — cosmetic
                _unpin()

        vb.sigYRangeChanged.connect(_reposition)
        _reposition()                                  # place at the CURRENT view top now
        return _unpin

    @staticmethod
    def _breath_label(num):
        """The per-breath number label, e.g. '#3'. The word 'breath' is intentionally
        omitted from per-breath numbering — it is implicit from context on every graph."""
        return f"#{num}"

    def _draw_breath_overlays(self, spans, label_y=0.0):
        self._mech_unpin()               # the old labels are torn down with their pin slot
        self._breath_spans = {n: (t0, t1) for (n, t0, t1, _ig) in spans}
        self._breath_regions = {n: [] for (n, _0, _1, _ig) in spans}
        self._breath_texts = {}
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
                txt = self._breath_text(n, ignored)
                txt.setPos((t0 + t1) / 2.0, label_y)
                self._channel_plots[0].addItem(txt, ignoreBounds=True)
                self._breath_texts[n] = txt
        if self._channel_plots and self._breath_texts:
            # size the headroom from the label's REAL rendered height, not a guess
            self._label_headroom(self._channel_plots[0], label_px=self._label_px(self._breath_texts))
            self._mech_unpin = self._pin_breath_labels(self._channel_plots[0], self._breath_texts)

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
        self.settings_edited.emit()      # exclude_breaths lands in the .toml -> mark dirty
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
            rec.get("unpin", lambda: None)()           # detach the old pin slot eagerly
            for plot, item in rec.get("items", []):
                try:
                    plot.removeItem(item)
                except Exception:                      # noqa: BLE001
                    pass
        self._bov[view] = {"items": [], "regions": {}, "texts": {}, "unpin": lambda: None}
        if not plot_items or not self._breaths:
            return
        excl = self._excluded_now()
        reg_map = self._bov[view]["regions"]; txt_map = self._bov[view]["texts"]
        items = self._bov[view]["items"]
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
            txt = self._breath_text(num, ignored)
            txt.setPos((a + b) / 2.0, label_y)
            plot_items[0].addItem(txt, ignoreBounds=True)
            items.append((plot_items[0], txt)); txt_map[num] = txt
        self._label_headroom(plot_items[0], label_px=self._label_px(txt_map))   # labels above the signal
        self._bov[view]["unpin"] = self._pin_breath_labels(plot_items[0], txt_map)

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
        self._mech_unpin()
        self._mech_unpin = lambda: None
        for rec in self._bov.values():
            rec.get("unpin", lambda: None)()
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
                                 parent=self, file_name=self.file_combo.currentText(),
                                 flow=data.get("flow"))
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
        # discrete full-length flow silhouette behind the EMG, to read bursts against respiration
        add_flow_background(self.emg_plots.getPlotItem(), t, data.get("flow"), pal)
        self.emg_plots.setLabel("bottom", "Time (s)"); self.emg_plots.setLabel("left", "EMG (a.u.)")
        self._legend_one_row(self.emg_plots.getPlotItem())   # after every named plot() call
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
            self._bov.get("result", {}).get("unpin", lambda: None)()
            self._bov["result"] = {"items": [], "regions": {}, "texts": {}, "unpin": lambda: None}
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
        # discrete full-length flow silhouette behind the ticked EMG channels
        add_flow_background(self.emg_result_plots.getPlotItem(), t, data.get("flow"), _plot_pal())
        # re-run per render: the entry count tracks the ticked channels
        self._legend_one_row(self.emg_result_plots.getPlotItem())
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
            # Bottom-left: with Poes on x and volume on y that corner is empty (the loop's
            # EELV end sits bottom-right). The output PDF transposes these axes, so its
            # legend must NOT follow — see core/plots._pv_average.
            ax.legend(loc="lower left", frameon=False, fontsize=7)
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
                lbl = f"EMG col {emg_cols[ch]}" if ch < len(emg_cols) else f"EMG ch {ch + 1}"
                ax.plot(props, [frontier[p][ch] for p in props], marker="o", ms=3, label=lbl)
            ax.axhline(target, color=pal["mpl_target"], ls=":", lw=1)
            if chosen is not None:
                ax.axvline(chosen, color=pal["mpl_error"], ls="--", lw=1.2, label=f"chosen = {chosen}")
            ax.set_xlabel("Noise-suppression strength (prop_decrease, 0–1)")
            # Rotated, the y-label's length runs along the axes HEIGHT (~206px in a 300px
            # panel), so a wordy one overflows. The panel header carries the title and the
            # "1 = untouched" scale hint horizontally, where there is room for them.
            ax.set_ylabel("Fidelity retained")
            ax.set_ylim(0, 1.02)
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
        # A user-fixable precondition surfaced by stage_noise_fidelity as {"error": msg} (e.g. a
        # misassigned flow channel, so the reference has no segmentable breaths). Raise FIRST so
        # _on_job_done paints a clean 'Noise fidelity failed' card with the message — otherwise the
        # error dict would render a silently blank frontier, mark the panel as done, and even
        # re-dispatch the EMG views, hiding the failure and suppressing the auto-retry.
        if isinstance(report, dict) and report.get("error"):
            raise _FileRunError(report["error"])
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
            # This is an APP-driven change, not a user edit — fold it into the diff baseline
            # so the next user edit's dependency-scope diff does not mistake it for a change
            # and needlessly recompute the EMG/noise panels.
            if self._last_synced_settings is not None:
                self._last_synced_settings.processing.emg.noise.prop_decrease = float(chosen)
        self._load_noise_params()                    # reflect the chosen prop in the spinbox
        if nr.get("frontier"):
            target = nr.get("fidelity_target", 0.8)
            worst = min((r["fidelity"] for r in nr.get("channels", [])), default=float("nan"))
            self._set_status(f"Noise: prop_decrease {chosen} chosen "
                             f"(worst-channel fidelity {worst:.2f} vs target {target:.2f}).")
        if noise.enabled and noise.auto_prop and self.state.settings.input.channels.emg:
            self._schedule("emg_all")                # re-condition with the chosen suppression
            self._schedule("emg_detail")

"""Screen 2 — source-data preview & tuning.

Two sub-tabs (the EMG one appears only when EMG channels are configured):

* **Mechanics** — the flow/volume/pressure channels on a shared time axis with
  detected breath boundaries and the trim window; a Campbell diagram and the
  per-breath table after a test run. Breaths are shaded/numbered and can be
  included/excluded by clicking.
* **EMG processing** — a view of the raw EMG channels; a "result" view of the
  conditioned EMG (pick which channels to show); a single-channel detail (raw ▸
  ECG-removed ▸ noise-reduced, time + PSD); a draggable rest-region selector that
  sets the shared noise reference; and the noise-window options (active once a
  reference file is set) with a fidelity-frontier preview.

The file list refreshes when the input folder/mask changes; selecting a file
auto-previews. All computation reuses the core; nothing here writes to disk.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QThread, Signal

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core import compute
from respmech.core.io.loaders import load
from respmech.core.pipeline import run_batch
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.settings import ExcludeEntry
from respmech.ui.workers import BatchWorker, EmgConditioningWorker, EmgAllChannelsWorker

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


def _pen(colour, width=1):
    return pg.mkPen(colour, width=width)


class PreviewScreen(QWidget):
    status_changed = Signal(str)
    # emitted when a noise reference is chosen on the graph (feature B)
    noise_reference_changed = Signal(str, object, bool)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._thread = None
        self._worker = None
        self._noise_error = None
        self._emg_error = None
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
        self.status = QLabel("Pick a file and preview.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.subtabs = QTabWidget()
        self._mech_tab = self._build_mech_tab()
        self._emg_tab = self._build_emg_tab()
        self.subtabs.addTab(self._mech_tab, "Mechanics")
        root.addWidget(self.subtabs, 1)

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
        self.btn_emg_preview.setToolTip("Stage the selected channel raw ▸ ECG-removed ▸ noise-reduced (off the GUI thread).")
        self.btn_emg_preview.clicked.connect(self._start_emg_preview)
        self.btn_emg_result = QPushButton("Show result (all channels)")
        self.btn_emg_result.setToolTip("Condition every EMG channel; tick channels below to show them.")
        self.btn_emg_result.clicked.connect(self._show_emg_result)
        self.btn_use_region = QPushButton("Use selection as noise profile")
        self.btn_use_region.setToolTip("Set the shaded rest span as the noise reference for this file.")
        self.btn_use_region.clicked.connect(self._use_region_as_noise)
        self.btn_noise = QPushButton("Noise & fidelity preview")
        self.btn_noise.setToolTip("Build the shared noise profile for the test and show the fidelity frontier.")
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
            self._set_status(f"{len(files)} file(s) — pick one to preview.")
        self._update_actions()

    def _refresh_files(self):        # kept for existing wiring + tests
        self.refresh_files()

    def sync_from_settings(self):
        self._refresh_emg_channels()
        self._load_noise_params()
        self._update_emg_tab_visibility()
        self._update_actions()

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

    def _update_actions(self):
        has_file = bool(self.file_combo.currentText())
        ok, why = self._settings_ok()
        emg = self.state.settings.processing.emg
        noise_on, ref = emg.noise.enabled, emg.noise.reference_file
        has_emg = bool(self.state.settings.input.channels.emg)
        busy = self._thread is not None
        self.btn_preview.setEnabled(has_file and not busy)
        self.btn_test.setEnabled(has_file and ok and not busy)
        self.btn_noise.setEnabled(has_file and ok and noise_on and bool(ref) and not busy)
        self.emg_channel.setEnabled(has_emg and not busy)
        self.btn_use_region.setEnabled(has_file and has_emg and not busy)
        self.btn_emg_preview.setEnabled(has_file and has_emg and not busy)
        self.btn_emg_result.setEnabled(has_file and has_emg and not busy)
        # noise-window options only active with a reference file (Emil's requirement)
        self.noise_opts.setEnabled(bool(has_emg and noise_on and ref))
        if noise_on:
            self.noise_prop.setEnabled(bool(ref) and not self.noise_auto.isChecked())
        if noise_on and not ref:
            self._set_status("Noise reduction is on — pick a reference file (Settings) or "
                             "select a rest region below and 'Use selection as noise profile'.")
        elif has_file and not ok:
            self._set_status(f"Settings incomplete: {why}")

    def _on_file_selected(self, name):
        self._update_actions()
        if name and name != self._previewed_file and self._thread is None:
            self._preview()

    # -- channel preview (Mechanics) ---------------------------------------
    def _preview(self):
        path = self._current_file()
        if not path:
            return
        try:
            s = to_legacy_ns(self.state.settings)
            flow, volume, poes, pgas, pdi, ent, emg = load(path, s)
            fs = s.input.format.samplingfrequency
            tc = np.arange(len(flow)) / fs
            (tcT, flowT, volT, poesT, pgasT, pdiT, _e, startix, endix) = compute.trim(
                tc, flow, volume, poes, pgas, pdi,
                np.array(emg) if len(emg) else np.array([]), s)
            volc = compute.correctdrift(compute.zero(volT), s) if s.processing.mechanics.correctvolumedrift else compute.zero(volT)
            breaths = compute.separateintobreaths(
                s.processing.mechanics.separateby, os.path.basename(path), tcT, flowT,
                volc, poesT, pgasT, pdiT, [], [], s)
        except Exception as e:              # noqa: BLE001
            self._set_status(f"Preview failed: {e}")
            return

        t = np.arange(len(flowT)) / fs
        series = {"flow": flowT, "volume": volc, "poes": poesT, "pgas": pgasT, "pdi": pdiT}
        spans, cum = [], 0
        for bno, b in breaths.items():
            length = len(np.atleast_1d(b["poes"]))
            spans.append((bno, cum / fs, (cum + length) / fs, bool(b["ignored"])))
            cum += length

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
        label_y = float(np.nanmax(series["flow"])) if len(series["flow"]) else 0.0
        self._draw_breath_overlays(spans, label_y)
        # keep the raw-EMG view + detail in step with the selected file
        self._draw_raw_emg(path)

        nign = sum(1 for _n, _a, _b, ig in spans if ig)
        self._previewed_file = os.path.basename(path)
        self._set_status(
            f"{os.path.basename(path)}: {len(breaths)} breaths"
            + (f" ({nign} excluded)" if nign else "")
            + f", trimmed to {startix / fs:.2f}–{endix / fs:.2f} s. "
            "Click a shaded breath to include/exclude (red = excluded).")

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
            f"({nexcl}/{len(self._breath_spans)} excluded). Re-run Preview or Test run to apply.")
        return now_excluded

    # -- feature B: noise-profile region -----------------------------------
    def _ensure_noise_region(self):
        reg = self._noise_region
        if reg is None:
            ivals = self.state.settings.processing.emg.noise.reference_intervals
            t0, t1 = (float(ivals[0][0]), float(ivals[0][1])) if ivals else (0.0, 1.0)
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
        return (t0, t1)

    # -- feature C: single-channel conditioning detail ---------------------
    def _draw_raw_emg(self, path):
        """Stacked raw EMG channels view + the selected channel on the detail plot
        (so the noise region has context). Light — load only, no conditioning."""
        try:
            s = to_legacy_ns(self.state.settings)
            _f, _v, _p, _g, _d, _e, emg = load(path, s)
            emg = np.asarray(emg, dtype=float)
            if emg.ndim == 1:
                emg = emg[:, None]
            fs = s.input.format.samplingfrequency
            t = np.arange(emg.shape[0]) / fs
        except Exception:                              # noqa: BLE001
            return
        cols = list(self.state.settings.input.channels.emg)
        self.emg_raw_plots.clear()
        prev = None
        for i in range(emg.shape[1]):
            p = self.emg_raw_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i+1}")
            p.plot(t, emg[:, i], pen=_pen(_EMG_PENS[i % len(_EMG_PENS)]))
            if i == emg.shape[1] - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
        # detail plot: selected channel raw + noise region
        ch = max(0, min(self.emg_channel.currentIndex(), emg.shape[1] - 1))
        self.emg_plots.clear()
        self.emg_plots.plot(t, emg[:, ch], pen=_pen((150, 165, 180)), name="raw")
        self.emg_plots.setLabel("bottom", "Time (s)"); self.emg_plots.setLabel("left", "EMG (a.u.)")
        self._ensure_noise_region()

    def _on_emg_channel_changed(self, *_):
        path = self._current_file()
        if path and self._thread is None:
            self._draw_raw_emg(path)

    def _start_emg_preview(self):
        path = self._current_file()
        if not path or self._thread is not None:
            return
        if not self.state.settings.input.channels.emg:
            self._set_status("No EMG channels configured (Settings ▸ Channels).")
            return
        ch = max(0, self.emg_channel.currentIndex())
        self._set_status(f"Staging EMG {self.emg_channel.currentText()} (raw ▸ ECG-removed ▸ noise-reduced)…")
        self.btn_emg_preview.setEnabled(False)
        self._emg_error = None
        self._thread = QThread()
        self._worker = EmgConditioningWorker(self.state.settings, path, ch)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_emg_staged)
        self._worker.failed.connect(self._on_emg_failed)
        self._thread.start()

    def _on_emg_failed(self, msg):
        self._emg_error = msg

    def _on_emg_staged(self, data):
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(2000)
        self._thread = None; self._worker = None
        if self._emg_error or data is None:
            self._set_status(f"EMG preview failed: {self._emg_error or 'no data'}")
            self._update_actions(); return
        try:
            self.render_emg_time(data)
            self.render_emg_psd(data)
        except Exception as e:                          # noqa: BLE001
            self._set_status(f"Could not render EMG preview: {e}")
            self._update_actions(); return
        applied = ("noise reduction applied" if data.get("noise_applied")
                   else "no noise reference set — showing raw vs ECG-removed")
        self._set_status(f"EMG {self.emg_channel.currentText()}: {applied} (band 20–250 Hz marked).")
        self._update_actions()

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
        ax.set_title(f"EMG PSD — channel {int(data.get('channel', 0)) + 1}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        self.emg_psd_canvas.draw()

    # -- result view (all channels, pick which to show) --------------------
    def _show_emg_result(self):
        path = self._current_file()
        if not path or self._thread is not None:
            return
        if not self.state.settings.input.channels.emg:
            self._set_status("No EMG channels configured.")
            return
        self._set_status("Conditioning all EMG channels for the result view…")
        self.btn_emg_result.setEnabled(False)
        self._emg_error = None
        self._thread = QThread()
        self._worker = EmgAllChannelsWorker(self.state.settings, path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_emg_all_staged)
        self._worker.failed.connect(self._on_emg_failed)
        self._thread.start()

    def _on_emg_all_staged(self, data):
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(2000)
        self._thread = None; self._worker = None
        if self._emg_error or data is None:
            self._set_status(f"EMG result failed: {self._emg_error or 'no data'}")
            self._update_actions(); return
        self._emg_all = data
        self._render_emg_result()
        applied = "conditioned (ECG + noise)" if data.get("noise_applied") else "ECG-removed"
        self._set_status(f"EMG result: {applied}. Tick channels above to show/hide them.")
        self._update_actions()

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

    # -- test run (Mechanics) ----------------------------------------------
    def _test_run(self):
        path = self._current_file()
        if not path:
            return
        name = os.path.basename(path)
        self._preview()          # show breaths on the graph for the test run
        try:
            self.state.settings.validate()
            result = run_batch(self.state.settings, only_files=[name])  # never writes
        except Exception as e:              # noqa: BLE001
            self._set_status(f"Test run failed: {e}")
            return
        fr = result.files.get(name)
        if fr is None or fr.error:
            self._set_status(f"Test run: {fr.error if fr else 'no result'}")
            return
        self._fill_table(fr.breaths_table)
        self._draw_campbell(fr.breaths)
        n = len(fr.breaths_table)
        msg = f"Test run OK: {n} breaths (nothing written)"
        ecg = getattr(fr, "ecg", None)
        if ecg:
            msg += f" · ECG suppression {ecg.get('suppression', float('nan')):.0%}"
        self._set_status(msg)

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

    # -- noise & fidelity preview (non-blocking) ---------------------------
    def _start_noise_preview(self):
        path = self._current_file()
        if not path or self._thread is not None:
            return
        if not self.state.settings.processing.emg.noise.enabled:
            self._set_status("Enable EMG noise reduction (Settings) to preview fidelity.")
            return
        if not self.state.settings.processing.emg.noise.reference_file:
            self._set_status("Set a reference file (Settings) or select a rest region for the noise profile.")
            return
        try:
            self.state.settings.validate()
        except Exception as e:              # noqa: BLE001
            self._set_status(f"Cannot preview: {e}")
            return
        self._set_status("Building the shared noise profile for the test (this may take a moment)…")
        self.btn_noise.setEnabled(False)
        self._noise_error = None
        name = os.path.basename(path)
        self._thread = QThread()
        self._worker = BatchWorker(self.state.settings, write=False, only_files=[name])
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_noise_done)
        self._worker.failed.connect(self._on_noise_failed)
        self._thread.start()

    def _on_noise_failed(self, msg):
        self._noise_error = msg

    def _on_noise_done(self, result):
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(2000)
        self._thread = None; self._worker = None
        if self._noise_error:
            self._set_status(f"Noise preview failed: {self._noise_error}")
            self._update_actions(); return
        if result is None or result.noise_report is None:
            self._set_status("No noise report produced (check the reference file / intervals).")
            self._update_actions(); return
        try:
            self.render_noise_report(result)
        except Exception as e:              # noqa: BLE001
            self._set_status(f"Could not render noise report: {e}")
        self._load_noise_params()   # auto-selection may have chosen prop_decrease
        self._update_actions()

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

"""Screen 2 — source-data preview & tuning.

Pick one input file and inspect it before committing to the full batch:
* the raw channels (flow / volume / pressures) are drawn with pyqtgraph on a shared
  time axis with units and detected breath boundaries + trim window,
* a **test run** computes just this file (nothing written) and shows the per-breath
  table + a Campbell diagram, and
* a **noise & fidelity preview** builds the shared noise profile for the whole test
  (off the GUI thread) and shows the per-channel fidelity frontier + chosen strength.

The file list refreshes when the input folder/mask changes; selecting a file
auto-previews. All computation reuses the core; nothing here writes to disk.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QSplitter)
from PySide6.QtCore import Qt, QThread, Signal

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core import compute
from respmech.core.io.loaders import load
from respmech.core.pipeline import run_batch
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.settings import ExcludeEntry
from respmech.ui.workers import BatchWorker, EmgConditioningWorker

# channel -> (axis label with units, pen colour by physiological meaning)
_CHANNELS = [
    ("flow", "Flow (L/s)", (44, 110, 155)),
    ("volume", "Volume (L)", (31, 122, 77)),
    ("poes", "Poes (cmH₂O)", (180, 50, 42)),
    ("pgas", "Pgas (cmH₂O)", (183, 121, 31)),
    ("pdi", "Pdi (cmH₂O)", (125, 91, 166)),
]


class PreviewScreen(QWidget):
    status_changed = Signal(str)
    # emitted when a noise reference is chosen on the graph (feature B), so the
    # Settings screen + shared state stay coherent: (file, intervals, use_expiration)
    noise_reference_changed = Signal(str, object, bool)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._thread = None
        self._worker = None
        self._noise_error = None
        self._emg_error = None
        self._previewed_file = None
        # breath-overlay interaction state (feature A)
        self._channel_plots = []
        self._breath_spans = {}       # breath_no -> (t0, t1)  seconds
        self._breath_regions = {}     # breath_no -> [LinearRegionItem per channel plot]
        self._breath_texts = {}       # breath_no -> TextItem (number label)
        # draggable noise-selection region on the EMG view (feature B)
        self._noise_region = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.file_combo = QComboBox()
        self.btn_refresh = QPushButton("Refresh files"); self.btn_refresh.clicked.connect(self.refresh_files)
        self.btn_preview = QPushButton("Preview channels"); self.btn_preview.clicked.connect(self._preview)
        self.btn_test = QPushButton("Test run this file"); self.btn_test.clicked.connect(self._test_run)
        self.btn_noise = QPushButton("Noise & fidelity preview"); self.btn_noise.clicked.connect(self._start_noise_preview)
        bar.addWidget(QLabel("File:")); bar.addWidget(self.file_combo, 1)
        bar.addWidget(self.btn_refresh); bar.addWidget(self.btn_preview)
        bar.addWidget(self.btn_test); bar.addWidget(self.btn_noise)
        root.addLayout(bar)
        self.status = QLabel("Pick a file and preview.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        split = QSplitter(Qt.Vertical)
        self.plots = pg.GraphicsLayoutWidget()
        # single click -> hit-test a breath region -> toggle include/exclude (feature A).
        # Connected to the persistent scene (survives self.plots.clear()).
        self.plots.scene().sigMouseClicked.connect(self._on_plot_clicked)
        split.addWidget(self.plots)

        lower = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 0)
        lower.addWidget(self.table)
        self.campbell = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        lower.addWidget(self.campbell)
        self.fidelity_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        lower.addWidget(self.fidelity_canvas)
        split.addWidget(lower)

        # -- EMG conditioning / noise-region panel (features B + C) -----------
        emg_box = QWidget()
        ev = QVBoxLayout(emg_box); ev.setContentsMargins(0, 0, 0, 0)
        ebar = QHBoxLayout()
        self.emg_channel = QComboBox()
        self.emg_channel.setToolTip("EMG channel to inspect / condition.")
        self.btn_use_region = QPushButton("Use selection as noise profile")
        self.btn_use_region.setToolTip("Set the shaded rest span as processing.emg.noise "
                                       "reference_intervals for the current file.")
        self.btn_use_region.clicked.connect(self._use_region_as_noise)
        self.btn_emg_preview = QPushButton("Preview EMG conditioning")
        self.btn_emg_preview.setToolTip("Stage raw ▸ ECG-removed ▸ noise-reduced for the "
                                        "selected channel (off the GUI thread).")
        self.btn_emg_preview.clicked.connect(self._start_emg_preview)
        ebar.addWidget(QLabel("EMG channel:")); ebar.addWidget(self.emg_channel)
        ebar.addWidget(self.btn_use_region); ebar.addWidget(self.btn_emg_preview)
        ebar.addStretch(1)
        ev.addLayout(ebar)
        emg_split = QSplitter(Qt.Horizontal)
        self.emg_plots = pg.PlotWidget()
        self.emg_plots.setLabel("bottom", "Time (s)")
        self.emg_plots.setLabel("left", "EMG (a.u.)")
        emg_split.addWidget(self.emg_plots)
        self.emg_psd_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        emg_split.addWidget(self.emg_psd_canvas)
        ev.addWidget(emg_split, 1)
        split.addWidget(emg_box)
        self._refresh_emg_channels()
        self._ensure_noise_region()

        root.addWidget(split, 1)

        self.file_combo.currentTextChanged.connect(self._on_file_selected)
        self.refresh_files()

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- file list (reactive) ----------------------------------------------
    def refresh_files(self):
        s = self.state.settings
        prev = self.file_combo.currentText()
        folder = (s.input.folder or "").strip()
        self.file_combo.blockSignals(True)          # no auto-preview on programmatic rebuild
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
        if noise_on and not ref:
            self._set_status("Noise reduction is on but no reference file is set "
                             "(Settings ▸ Rest reference file).")
        elif has_file and not ok:
            self._set_status(f"Settings incomplete: {why}")

    def _on_file_selected(self, name):
        self._update_actions()
        if name and name != self._previewed_file and self._thread is None:
            self._preview()

    # -- channel preview ----------------------------------------------------
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

        t = np.arange(len(flowT)) / fs      # seconds, from start of trimmed data
        series = {"flow": flowT, "volume": volc, "poes": poesT, "pgas": pgasT, "pdi": pdiT}
        # per-breath spans (contiguous partition of the trimmed data): the breath key
        # is the 1-based `breathcnt` used by exclude_breaths / ignorebreaths.
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
            p.plot(t[:len(y)], y, pen=pg.mkPen(colour, width=1))
            if i == n - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
            self._channel_plots.append(p)
        label_y = float(np.nanmax(series["flow"])) if len(series["flow"]) else 0.0
        self._draw_breath_overlays(spans, label_y)

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
        """Overlay each breath as a shaded, labelled ``LinearRegionItem`` on every
        stacked channel plot (blue = included, red = excluded). ``spans`` is a list
        of ``(breath_no, t0, t1, ignored)``. Regions are stored so ``_toggle_breath``
        can recolour them in place. Non-movable — clicks are hit-tested at the scene
        level. Headless-safe (no mouse needed to construct)."""
        self._breath_spans = {n: (t0, t1) for (n, t0, t1, _ig) in spans}
        self._breath_regions = {n: [] for (n, _0, _1, _ig) in spans}
        self._breath_texts = {}
        for n, t0, t1, ignored in spans:
            for plot in self._channel_plots:
                reg = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                          brush=self._breath_brush(ignored),
                                          pen=pg.mkPen(None))
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
        """Scene click -> which channel viewbox -> time -> breath -> toggle. Guarded
        so ordinary pan/zoom on empty space is a no-op."""
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
        """Toggle include/exclude for ``breath_no`` (1-based) of the CURRENT file in
        ``settings.processing.exclude_breaths``, recolour its overlay, and update the
        status live. Returns the new excluded state (or ``None`` if not applicable).
        Callable headless — no mouse required."""
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
            reg.setBrush(self._breath_brush(now_excluded))
            reg.update()
        txt = self._breath_texts.get(breath_no)
        if txt is not None:
            txt.setColor(self._breath_label_color(now_excluded))
        nexcl = len({b for e in excl if e.file == name for b in e.breaths}
                    & set(self._breath_spans))
        self._set_status(
            f"{name}: breath {breath_no} "
            f"{'excluded' if now_excluded else 'included'} "
            f"({nexcl}/{len(self._breath_spans)} excluded). "
            "Re-run Preview or Test run to apply.")
        return now_excluded

    # -- feature B: select a noise-profile region on the graph -------------
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

    def _ensure_noise_region(self):
        """Add (once) a draggable highlighted time span on the EMG view; re-attach it
        after an ``emg_plots.clear()`` (which detaches scene items)."""
        reg = self._noise_region
        if reg is None:
            ivals = self.state.settings.processing.emg.noise.reference_intervals
            t0, t1 = (float(ivals[0][0]), float(ivals[0][1])) if ivals else (0.0, 1.0)
            reg = pg.LinearRegionItem(values=(t0, t1), movable=True,
                                      brush=pg.mkBrush(44, 110, 155, 45))
            reg.setZValue(10)
            self._noise_region = reg
        if reg.scene() is None:
            self.emg_plots.addItem(reg)

    def _use_region_as_noise(self):
        """Read the shaded span [t0, t1] and set it as the shared noise reference for
        the CURRENT file: ``noise.reference_file = file``,
        ``reference_intervals = [[t0, t1]]``, ``use_expiration = False``. Emits
        ``noise_reference_changed`` so the Settings screen mirrors it. Headless-safe."""
        name = self.file_combo.currentText()
        reg = self._noise_region
        if not name or reg is None:
            self._set_status("Draw a rest region on the EMG channel, then apply.")
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
            f"Noise profile ← {name} [{t0:.2f}–{t1:.2f} s]: {span} samples "
            f"≈ {nframes} STFT frames "
            + ("(good)" if nframes >= 8 else "(short — pick a longer rest span)")
            + ". Enable 'Reduce EMG noise' to apply it.")
        self._update_actions()
        return (t0, t1)

    # -- feature C: preview EMG conditioning (off-thread) ------------------
    def _start_emg_preview(self):
        path = self._current_file()
        if not path or self._thread is not None:
            return
        if not self.state.settings.input.channels.emg:
            self._set_status("No EMG channels configured (Settings ▸ Channels).")
            return
        ch = max(0, self.emg_channel.currentIndex())
        self._set_status(f"Staging EMG {self.emg_channel.currentText()} "
                         "(raw ▸ ECG-removed ▸ noise-reduced)…")
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
        self._set_status(f"EMG {self.emg_channel.currentText()}: {applied} "
                         "(band 20–250 Hz marked on the PSD).")
        self._update_actions()

    def render_emg_time(self, data):
        """Time-domain overlay of raw vs ECG-removed vs noise-reduced (pyqtgraph).
        Takes the precomputed arrays dict from ``workers.stage_emg_channel`` so it is
        directly unit-testable headless."""
        t = np.asarray(data["t"])
        self.emg_plots.clear()
        self.emg_plots.plot(t, np.asarray(data["raw"]),
                            pen=pg.mkPen((150, 165, 180), width=1), name="raw")
        self.emg_plots.plot(t, np.asarray(data["ecg"]),
                            pen=pg.mkPen((44, 110, 155), width=1), name="ECG-removed")
        if data.get("noise_applied"):
            self.emg_plots.plot(t, np.asarray(data["noise"]),
                                pen=pg.mkPen((31, 122, 77), width=1), name="noise-reduced")
        self.emg_plots.setLabel("bottom", "Time (s)")
        self.emg_plots.setLabel("left", "EMG (a.u.)")
        self._ensure_noise_region()

    def render_emg_psd(self, data):
        """Log-power PSD of each stage with the 20–250 Hz EMG band marked (matplotlib
        on ``emg_psd_canvas`` — kept separate from ``fidelity_canvas``). Takes the
        precomputed arrays dict; unit-testable headless. Always exactly 1 axis."""
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
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title(f"EMG PSD — channel {int(data.get('channel', 0)) + 1}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        self.emg_psd_canvas.draw()

    # -- test run -----------------------------------------------------------
    def _test_run(self):
        path = self._current_file()
        if not path:
            return
        name = os.path.basename(path)
        # Always show the breaths on the channel graph for a test run, reflecting the
        # current include/exclude state (so they can be toggled here too).
        self._preview()
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
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)          # atmospheric reference
        ax.set_xlabel("Oesophageal pressure  Poes (cmH₂O)")
        ax.set_ylabel("Lung volume above end-expiration (L)")
        ax.set_title("Campbell diagram")
        fig.tight_layout()
        self.campbell.draw()

    # -- noise / fidelity preview (non-blocking) ---------------------------
    def _start_noise_preview(self):
        path = self._current_file()
        if not path or self._thread is not None:
            return
        if not self.state.settings.processing.emg.noise.enabled:
            self._set_status("Enable EMG noise reduction (Settings) to preview fidelity.")
            return
        if not self.state.settings.processing.emg.noise.reference_file:
            self._set_status("Set a rest reference file (Settings) for the noise profile.")
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
        # process just this file, but the shared profile is built from the WHOLE test.
        self._worker = BatchWorker(self.state.settings, write=False, only_files=[name])
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_noise_done)
        self._worker.failed.connect(self._on_noise_failed)
        self._thread.start()

    def _on_noise_failed(self, msg):
        self._noise_error = msg             # store; don't display-then-lose

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
        self._update_actions()

    def render_noise_report(self, result):
        """Plot the per-channel fidelity frontier + summarise. The chosen prop_decrease
        and the fidelity-target line make the per-test tuning visible."""
        nr = result.noise_report
        frontier = nr.get("frontier", {})
        chosen = nr.get("prop_decrease")
        target = nr.get("fidelity_target", 0.8)
        emg_cols = list(self.state.settings.input.channels.emg)
        fig = self.fidelity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)                    # unconditional -> always 1 axis
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

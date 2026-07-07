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
from respmech.ui.workers import BatchWorker

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

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._thread = None
        self._worker = None
        self._noise_error = None
        self._previewed_file = None
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
        split.addWidget(self.plots)

        lower = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 0)
        lower.addWidget(self.table)
        self.campbell = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        lower.addWidget(self.campbell)
        self.fidelity_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        lower.addWidget(self.fidelity_canvas)
        split.addWidget(lower)
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
        busy = self._thread is not None
        self.btn_preview.setEnabled(has_file and not busy)
        self.btn_test.setEnabled(has_file and ok and not busy)
        self.btn_noise.setEnabled(has_file and ok and noise_on and bool(ref) and not busy)
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
        # breath start times + ignored flag (contiguous partition of the trimmed data)
        bounds, cum = [], 0
        for b in breaths.values():
            bounds.append((cum / fs, b["ignored"]))
            cum += len(b["poes"])

        self.plots.clear()
        prev = None
        n = len(_CHANNELS)
        for i, (key, label, colour) in enumerate(_CHANNELS):
            p = self.plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLabel("left", label)
            y = series[key]
            p.plot(t[:len(y)], y, pen=pg.mkPen(colour, width=1))
            for x, ignored in bounds:
                p.addLine(x=x, pen=pg.mkPen((180, 50, 42) if ignored else (150, 165, 180),
                                            width=1, style=Qt.DashLine))
            if i == n - 1:
                p.setLabel("bottom", "Time (s)")
            if prev is not None:
                p.setXLink(prev)
            prev = p
        nign = sum(1 for _, ig in bounds if ig)
        self._previewed_file = os.path.basename(path)
        self._set_status(
            f"{os.path.basename(path)}: {len(breaths)} breaths"
            + (f" ({nign} excluded)" if nign else "")
            + f", trimmed to {startix / fs:.2f}–{endix / fs:.2f} s "
            "(dashed = breath start; red = excluded)")

    # -- test run -----------------------------------------------------------
    def _test_run(self):
        path = self._current_file()
        if not path:
            return
        name = os.path.basename(path)
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

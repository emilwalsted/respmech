"""Screen 2 — source-data preview & tuning.

Pick one input file and inspect it before committing to the full batch:
* the raw channels (flow/volume/pressures) are drawn with pyqtgraph (fast pan/zoom),
* detected breath boundaries and the trim window are overlaid, and
* a **test run** computes just this file (no files written) and shows the per-breath
  table plus a Campbell diagram (matplotlib-Qt), so settings can be fine-tuned first.

All computation reuses the core; nothing here writes to disk.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QSplitter)
from PySide6.QtCore import Qt, QThread

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core import compute
from respmech.core.io.loaders import load
from respmech.core.pipeline import run_batch
from respmech.core._legacy_ns import to_legacy_ns
from respmech.ui.workers import BatchWorker

_CHANNELS = ["flow", "volume", "poes", "pgas", "pdi"]


class PreviewScreen(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.file_combo = QComboBox()
        btn_refresh = QPushButton("Refresh files"); btn_refresh.clicked.connect(self._refresh_files)
        btn_preview = QPushButton("Preview channels"); btn_preview.clicked.connect(self._preview)
        btn_test = QPushButton("Test run this file"); btn_test.clicked.connect(self._test_run)
        self.btn_noise = QPushButton("Noise & fidelity preview"); self.btn_noise.clicked.connect(self._start_noise_preview)
        bar.addWidget(QLabel("File:")); bar.addWidget(self.file_combo, 1)
        bar.addWidget(btn_refresh); bar.addWidget(btn_preview); bar.addWidget(btn_test)
        bar.addWidget(self.btn_noise)
        root.addLayout(bar)
        self.status = QLabel("Pick a file and preview.")
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
        self._thread = None
        self._worker = None
        self._refresh_files()

    def _refresh_files(self):
        s = self.state.settings
        pattern = os.path.join(s.input.folder, s.input.files)
        self.file_combo.clear()
        self.file_combo.addItems([os.path.basename(f) for f in sorted(glob.glob(pattern))])

    def _current_file(self):
        name = self.file_combo.currentText()
        return os.path.join(self.state.settings.input.folder, name) if name else None

    def _preview(self):
        path = self._current_file()
        if not path:
            return
        try:
            s = to_legacy_ns(self.state.settings)
            flow, volume, poes, pgas, pdi, ent, emg = load(path, s)
            tc = np.arange(len(flow)) / s.input.format.samplingfrequency
            (tcT, flowT, volT, poesT, pgasT, pdiT, _e, startix, endix) = compute.trim(
                tc, flow, volume, poes, pgas, pdi,
                np.array(emg) if len(emg) else np.array([]), s)
            volc = compute.correctdrift(compute.zero(volT), s) if s.processing.mechanics.correctvolumedrift else compute.zero(volT)
            breaths = compute.separateintobreaths(
                s.processing.mechanics.separateby, os.path.basename(path), tcT, flowT,
                volc, poesT, pgasT, pdiT, [], [], s)
        except Exception as e:
            self.status.setText(f"Preview failed: {e}")
            return

        series = {"flow": flowT, "volume": volc, "poes": poesT, "pgas": pgasT, "pdi": pdiT}
        self.plots.clear()
        prev = None
        # breath boundaries (in trimmed sample index)
        bounds, cum = [], 0
        for b in breaths.values():
            bounds.append((cum, b["ignored"]))
            cum += len(b["poes"])
        for i, ch in enumerate(_CHANNELS):
            p = self.plots.addPlot(row=i, col=0)
            p.setLabel("left", ch)
            p.plot(series[ch], pen=pg.mkPen(width=1))
            for x, ignored in bounds:
                p.addLine(x=x, pen=pg.mkPen('r' if ignored else (100, 100, 255), style=Qt.DashLine))
            if prev is not None:
                p.setXLink(prev)
            prev = p
        self.status.setText(
            f"{os.path.basename(path)}: trimmed {startix}..{endix}, {len(breaths)} breaths "
            f"(dashed = breath starts; red = excluded)")

    def _test_run(self):
        path = self._current_file()
        if not path:
            return
        name = os.path.basename(path)
        try:
            self.state.settings.validate()
            # run_batch never writes; the GUI simply doesn't call the writer here.
            result = run_batch(self.state.settings, only_files=[name])
        except Exception as e:
            self.status.setText(f"Test run failed: {e}")
            return
        fr = result.files.get(name)
        if fr is None or fr.error:
            self.status.setText(f"Test run: {fr.error if fr else 'no result'}")
            return
        self._fill_table(fr.breaths_table)
        self._draw_campbell(fr.breaths)
        self.status.setText(f"Test run OK: {len(fr.breaths_table)} breaths (no files written)")

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
            ax.plot(b["volume"], b["poes"], linewidth=1)
        ax.set_xlabel("Inspired volume (L)")
        ax.set_ylabel("Poes (cmH2O)")
        ax.invert_xaxis()
        ax.set_title("Campbell diagram (test run)")
        self.campbell.draw()

    # -- noise / fidelity preview (non-blocking) ----------------------------
    def _start_noise_preview(self):
        path = self._current_file()
        if not path or self._thread is not None:
            return
        if not self.state.settings.processing.emg.noise.enabled:
            self.status.setText("Enable noise reduction (Settings) to preview fidelity.")
            return
        try:
            self.state.settings.validate()
        except Exception as e:
            self.status.setText(f"Cannot preview: {e}")
            return
        self.status.setText("Building shared noise profile for the test (this may take a moment)…")
        self.btn_noise.setEnabled(False)
        name = os.path.basename(path)
        self._thread = QThread()
        # process just this file, but the shared profile is built from the WHOLE test.
        self._worker = BatchWorker(self.state.settings, write=False, only_files=[name])
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_noise_done)
        self._worker.failed.connect(lambda msg: self.status.setText(f"Noise preview failed: {msg}"))
        self._thread.start()

    def _on_noise_done(self, result):
        self.btn_noise.setEnabled(True)
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(2000)
        self._thread = None; self._worker = None
        if result is None or result.noise_report is None:
            self.status.setText("No noise report produced (check settings/reference).")
            return
        self._render_noise(result)

    def render_noise_report(self, result):
        """Testable: plot the per-channel fidelity frontier + summarise. The chosen
        prop_decrease and the fidelity ≥ target line make the per-test tuning visible."""
        nr = result.noise_report
        frontier = nr.get("frontier", {})
        chosen = nr.get("prop_decrease")
        target = nr.get("fidelity_target", 0.8)
        fig = self.fidelity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if frontier:
            props = sorted(frontier)
            nch = len(next(iter(frontier.values())))
            for ch in range(nch):
                ax.plot(props, [frontier[p][ch] for p in props], marker="o", ms=3, label=f"ch{ch}")
            ax.axhline(target, color="k", ls=":", lw=1)
            if chosen is not None:
                ax.axvline(chosen, color="r", ls="--", lw=1, label=f"chosen={chosen}")
            ax.set_xlabel("prop_decrease"); ax.set_ylabel("fidelity")
            ax.set_title("Noise fidelity frontier (per channel)")
            ax.legend(fontsize=7)
        self.fidelity_canvas.draw()
        parts = [f"chosen prop_decrease={chosen} (fidelity target {target})"]
        for row in nr.get("channels", []):
            parts.append(f"ch{row['channel']}: fid={row['fidelity']:.2f} ΔSNR={row['delta_snr_db']:+.1f}dB")
        # ECG suppression for the previewed file, if available
        for fr in result.ok_files.values():
            if getattr(fr, "ecg", None):
                parts.append(f"ECG suppression={fr.ecg.get('suppression', float('nan')):.0%}")
                break
        self.status.setText("  |  ".join(parts))

    def _render_noise(self, result):
        self.render_noise_report(result)

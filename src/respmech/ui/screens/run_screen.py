"""Screen 3 — run the batch with live output and status.

Computation runs off the GUI thread (``BatchWorker`` on a ``QThread``); the worker
forwards the core's progress events as signals, so the log and progress bar update
live and the UI never freezes. Start/Cancel; a results table + summary fill on
completion. Run/Dry-run are disabled while settings are invalid or a run is busy.
"""
from __future__ import annotations

from PySide6.QtCore import Signal, QThread
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QTableWidget, QTableWidgetItem, QPlainTextEdit,
                               QVBoxLayout, QWidget)

from respmech.ui.workers import BatchWorker

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover - theme is import-safe, but stay defensive
    _theme = None


class RunScreen(QWidget):
    status_changed = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._thread = None
        self._worker = None
        self._build()
        self.refresh_actions()

    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.btn_run = QPushButton("Run batch")
        self.btn_dry = QPushButton("Dry run (no files written)")
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.setEnabled(False)
        if _theme is not None:
            _theme.make_primary(self.btn_run)
        self.btn_run.clicked.connect(lambda: self._start(write=True))
        self.btn_dry.clicked.connect(lambda: self._start(write=False))
        self.btn_cancel.clicked.connect(self._cancel)
        bar.addWidget(self.btn_run); bar.addWidget(self.btn_dry)
        bar.addWidget(self.btn_cancel); bar.addStretch(1)
        root.addLayout(bar)

        self.progress = QProgressBar()
        root.addWidget(self.progress)
        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        root.addWidget(self.log, 2)

        self.table = QTableWidget(0, 0)
        root.addWidget(self.table, 1)

    # -- helpers ------------------------------------------------------------
    def _settings_ok(self):
        try:
            self.state.settings.validate()
            return True, ""
        except Exception as e:              # noqa: BLE001
            return False, str(e)

    def refresh_actions(self):
        ok, why = self._settings_ok()
        busy = self._thread is not None
        self.btn_run.setEnabled(ok and not busy)
        self.btn_dry.setEnabled(ok and not busy)
        if not ok:
            self._set_status(f"Settings incomplete: {why}")

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- run lifecycle ------------------------------------------------------
    def _start(self, write: bool):
        if self._thread is not None:
            return
        ok, why = self._settings_ok()
        if not ok:
            self._set_status(f"Cannot run: {why}")
            return
        self.log.clear(); self.progress.setRange(0, 0)  # busy until first file
        self._set_running(True)

        self._thread = QThread()
        self._worker = BatchWorker(self.state.settings, write=write)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_failed.connect(lambda f, e: self._append(f"  {f}: FAILED — {e}"))
        self._worker.failed.connect(lambda msg: self._append(f"ERROR: {msg}"))
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self._append("cancelling…")

    def _on_progress(self, ev):
        if ev.kind == "file_start":
            self._append(f"\n{ev.file}: {ev.message}")
        elif ev.kind == "stage":
            self._append(f"  {ev.message}…")
        elif ev.kind == "breath":
            if ev.total_breaths:
                self.progress.setRange(0, ev.total_breaths)
                self.progress.setValue(ev.breath)
            self._set_status(f"{ev.file}: breath {ev.breath}/{ev.total_breaths}")
        elif ev.kind == "file_done":
            self._append(f"  done ({ev.message})")
        elif ev.kind == "finished":
            self._set_status(ev.message)

    def _on_finished(self, result):
        self.progress.setRange(0, 1); self.progress.setValue(1)
        self._set_running(False)                  # always re-enable first
        if result is not None:
            try:
                self._fill_table(result)
                self._append_summary(result)
            except Exception as e:                # noqa: BLE001
                self._append(f"\n(results table error: {e})")
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(2000)
        self._thread = None; self._worker = None
        self.refresh_actions()

    def _append_summary(self, result):
        self._append(f"\nFinished: {len(result.ok_files)} ok, {len(result.failed_files)} failed")
        nr = getattr(result, "noise_report", None)
        if nr:
            self._append(f"  noise: prop_decrease {nr.get('prop_decrease')} "
                         f"(fidelity target {nr.get('fidelity_target')})")
        for fname, fr in result.ok_files.items():
            n = len(fr.breaths_table) if fr.breaths_table is not None else 0
            line = f"  {fname}: {n} breaths"
            ecg = getattr(fr, "ecg", None)
            if ecg:
                line += f", ECG suppression {ecg.get('suppression', float('nan')):.0%}"
            self._append(line)

    def _fill_table(self, result):
        if result.average_table is None:
            return
        df = result.average_table.reset_index(drop=True)
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))

    def _set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_dry.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _append(self, text):
        self.log.appendPlainText(text)

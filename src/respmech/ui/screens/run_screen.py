"""Screen 3 — run the batch with live output and status.

Computation runs off the GUI thread (``BatchWorker`` on a ``QThread``); the worker
forwards the core's progress events as signals, so the log and progress bar update
live and the UI never freezes. Start/Cancel; a results table + summary fill on
completion. Run/Dry-run are disabled while settings are invalid or a run is busy.
"""
from __future__ import annotations

import copy
import os

from PySide6.QtCore import Signal, QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QPlainTextEdit, QVBoxLayout, QWidget)

from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.validation import matching_files, path_problem
from respmech.ui.workers import BatchWorker

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover - theme is import-safe, but stay defensive
    _theme = None

# threads that will not stop within the shutdown budget are parked here (kept
# referenced) rather than GC'd while running, which would abort the process.
_ORPHANED_THREADS = []


class RunScreen(QWidget):
    status_changed = Signal(str)
    run_started = Signal()          # a batch run began (main window locks Settings)
    run_finished = Signal()         # the batch run ended

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._thread = None
        self._worker = None
        self._fatal_msg = None
        self._error_dialog = None
        self._incomplete_shown = False
        self._last_write = False        # did the most recent completed run write files?
        self._build()
        self.refresh_actions()

    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.btn_run = QPushButton("Run batch")
        self.btn_dry = QPushButton("Dry run (no files written)")
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.setEnabled(False)
        self.btn_open = QPushButton("Open output folder")
        self.btn_open.setEnabled(False)     # enabled once a run has written results
        if _theme is not None:
            _theme.make_primary(self.btn_run)
        self.btn_run.clicked.connect(lambda: self._start(write=True))
        self.btn_dry.clicked.connect(lambda: self._start(write=False))
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_open.clicked.connect(self._open_output_folder)
        bar.addWidget(self.btn_run); bar.addWidget(self.btn_dry)
        bar.addWidget(self.btn_cancel); bar.addStretch(1)
        bar.addWidget(self.btn_open)
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

    def _quick_path_problem(self):
        """Cheap path check for enablement (the full glob-based check runs at start)."""
        s = self.state.settings
        if not (s.input.folder or "").strip() or not os.path.isdir(s.input.folder):
            return "input folder is not set or does not exist"
        if not (s.output.folder or "").strip():
            return "output folder is not set"
        return None

    def refresh_actions(self):
        ok, why = self._settings_ok()
        if ok:
            why = self._quick_path_problem() or ""
            ok = not why
        busy = self._thread is not None
        self.btn_run.setEnabled(ok and not busy)
        self.btn_dry.setEnabled(ok and not busy)
        if not ok:
            self._set_status(f"Settings incomplete: {why}")
            self._incomplete_shown = True
        elif self._incomplete_shown and not busy:
            self._set_status("Ready to run.")     # clear the stale warning once valid
            self._incomplete_shown = False

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- pre-flight & output (P5/P7) ---------------------------------------
    def _planned_outputs(self, files):
        """The output files a real run WOULD write, given the current save flags."""
        d = self.state.settings.output.data
        names = []
        if d.save_breath_by_breath:
            names += [f"data/{os.path.basename(f)}.breathdata.xlsx" for f in files]
        if d.save_processed:
            names += [f"data/{os.path.basename(f)} – Processed data.csv" for f in files]
        if d.save_average:
            names.append("data/Average breathdata.xlsx")
        names += ["analysis-used.toml", "run-report.txt"]      # always emitted
        return names

    def _append_plan(self, write: bool):
        """Log a pre-flight plan so a run is never a black box: what is read, and
        (for a real run) exactly what will be written — or, for a dry run, would be."""
        s = self.state.settings
        files = matching_files(s.input.folder, s.input.files)
        head = "DRY RUN — nothing will be written." if not write else "RUN — writing output."
        self._append(f"── {head} ──")
        self._append(f"Input:  {len(files)} file(s) matching '{s.input.files}' in {s.input.folder}")
        for f in files:
            self._append(f"    • {os.path.basename(f)}")
        planned = self._planned_outputs(files)
        verb = "Would write" if not write else "Will write"
        self._append(f"Output: {verb} {len(planned)} file(s) into {s.output.folder}")
        for n in planned:
            self._append(f"    • {n}")
        self._append("")     # blank line before the live run log

    def _existing_output(self):
        """(newest_mtime, count) of results already in the output folder, or None."""
        out = (self.state.settings.output.folder or "").strip()
        if not out or not os.path.isdir(out):
            return None
        hits = []
        for p in (os.path.join(out, "analysis-used.toml"),
                  os.path.join(out, "run-report.txt")):
            if os.path.isfile(p):
                hits.append(p)
        datadir = os.path.join(out, "data")
        if os.path.isdir(datadir):
            hits += [os.path.join(datadir, n) for n in os.listdir(datadir)
                     if os.path.isfile(os.path.join(datadir, n))]
        if not hits:
            return None
        newest = max(os.path.getmtime(p) for p in hits)
        return newest, len(hits)

    def _confirm_overwrite(self, existing) -> bool:
        from datetime import datetime
        newest, count = existing
        when = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
        out = self.state.settings.output.folder
        ans = QMessageBox.question(
            self, "Output folder already has results",
            f"{out}\n\nalready contains {count} result file(s) from a run on {when}.\n\n"
            "Running now will overwrite them. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return ans == QMessageBox.Yes

    def _open_output_folder(self):
        out = (self.state.settings.output.folder or "").strip()
        if out and os.path.isdir(out):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(out)))
        else:
            self._set_status("Output folder does not exist yet.")

    # -- run lifecycle ------------------------------------------------------
    def _start(self, write: bool):
        if self._thread is not None:
            return
        ok, why = self._settings_ok()
        if not ok:
            self._set_status(f"Cannot run: {why}")
            return
        problem = path_problem(self.state.settings)     # full filesystem check
        if problem:
            self._set_status(f"Cannot run: {problem}")
            return
        # overwrite guard: a real run into a folder that already holds results asks first
        if write:
            existing = self._existing_output()
            if existing and not self._confirm_overwrite(existing):
                self._set_status("Run cancelled — existing results kept.")
                return
        self.log.clear(); self.progress.setRange(0, 0)  # busy until first file
        self._fatal_msg = None
        self._last_write = write
        self._append_plan(write)                        # pre-flight: what will be read/written
        self._set_running(True)
        self.run_started.emit()

        self._thread = QThread()
        # snapshot the settings so edits mid-run can't change what the worker reads
        self._worker = BatchWorker(copy.deepcopy(self.state.settings), write=write)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_failed.connect(self._on_file_failed)   # bound -> GUI thread
        self._worker.failed.connect(self._on_fatal)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_file_failed(self, f, e):
        self._append(f"  {f}: FAILED — {e}")

    def _on_fatal(self, msg):
        self._fatal_msg = msg
        self._append(f"ERROR: {msg}")

    def shutdown(self, wait_ms=5000):
        """Cancel and join a running batch (called from MainWindow.closeEvent) so
        the worker thread is never destroyed while running."""
        if self._worker is not None:
            try:
                self._worker.cancel()
            except Exception:                       # pragma: no cover
                pass
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(wait_ms):
                _ORPHANED_THREADS.append((self._thread, self._worker))
        self._thread = None
        self._worker = None

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
        self._set_running(False)                  # always re-enable first
        fatal = self._fatal_msg is not None
        cancelled = result is None and not fatal
        if result is not None:
            try:
                self._fill_table(result)
                self._append_summary(result)
            except Exception as e:                # noqa: BLE001
                self._append(f"\n(results table error: {e})")
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(2000):       # never GC a still-running thread
                _ORPHANED_THREADS.append((self._thread, self._worker))
        self._thread = None; self._worker = None
        # a fatal message with a result present means the ANALYSIS succeeded but the
        # WRITE failed — a distinct outcome from a run that never completed
        write_failed = fatal and result is not None
        # enablement first (never lets a stale hint win over the outcome status)
        self.refresh_actions()
        self.progress.setRange(0, 1)
        if cancelled:
            self.progress.setValue(0)
            self._set_status("Run cancelled — no output written.")
        elif write_failed:
            self.progress.setValue(1)     # analysis finished; only the write failed
            self._set_status(f"Analysis complete, but writing failed — {short_error(self._fatal_msg)}")
        elif fatal:
            self.progress.setValue(0)
            self._set_status(f"Run failed — {short_error(self._fatal_msg)}")
        else:
            self.progress.setValue(1)
        # a clean real-write run leaves results on disk — offer to open them
        if self._last_write and result is not None and not fatal and not cancelled:
            self.btn_open.setEnabled(True)
        self.run_finished.emit()
        # surface a copyable error-log window if anything failed
        failed = getattr(result, "failed_files", {}) if result is not None else {}
        if failed or self._fatal_msg:
            self._show_error_window(self._error_report(result, failed or {}),
                                    write_failed=write_failed, fatal=fatal)

    def _error_report(self, result, failed):
        parts = []
        if result is None:
            parts.append("The batch run did not complete.\n")
            if self._fatal_msg:
                parts.append(self._fatal_msg)
        else:
            if failed:
                parts.append(f"{len(failed)} file(s) failed during the run:\n")
                for fname, fr in failed.items():
                    parts.append(f"── {fname} ──\n{getattr(fr, 'error', '') or '(no detail)'}\n")
            if self._fatal_msg:
                if parts:
                    parts.append("")
                parts.append(self._fatal_msg)
        return "\n".join(parts).strip()

    def _show_error_window(self, text, write_failed=False, fatal=False):
        if self._error_dialog is not None:   # replace, don't accumulate windows
            self._error_dialog.close()
            self._error_dialog.deleteLater()
            self._error_dialog = None
        if write_failed:
            intro = "The analysis completed, but writing the output failed. The full log below is copyable."
        elif fatal:
            intro = "The run stopped with an error before completing. The full log below is copyable."
        else:
            intro = "One or more files failed during the run. The full log is copyable."
        dlg = TextViewerDialog("Batch run — error log", text, self, intro=intro)
        self._error_dialog = dlg          # keep a reference so it isn't GC'd
        dlg.show()
        dlg.raise_()

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

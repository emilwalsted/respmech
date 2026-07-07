"""Background worker that runs the batch off the GUI thread.

The core's progress-event callback is the seam (see ``core.pipeline``): the worker
simply forwards core :class:`ProgressEvent`s as Qt signals, so the GUI never blocks
and can stream output live. Cancellation is cooperative (checked between files).

Usage (moveToThread pattern):

    thread = QThread()
    worker = BatchWorker(settings, write=True)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    ... connect worker.progress / worker.failed ...
    thread.start()
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from respmech.core.pipeline import run_batch
from respmech.core.io.writers import write_batch
from respmech.core.settings import Settings


class BatchWorker(QObject):
    progress = Signal(object)          # ProgressEvent
    file_done = Signal(str, object)    # filename, average-row DataFrame
    file_failed = Signal(str, str)     # filename, error
    finished = Signal(object)          # BatchResult (or None if cancelled/failed)
    failed = Signal(str)               # fatal error message

    def __init__(self, settings: Settings, write: bool = True, only_files=None):
        super().__init__()
        self._settings = settings
        self._write = write
        self._only = only_files
        self._cancel = False

    @Slot()
    def cancel(self):
        self._cancel = True

    def _on_event(self, ev):
        # Surface per-file completion/failure as dedicated signals for live updates.
        if ev.kind == "file_done":
            self.file_done.emit(ev.file, None)
        elif ev.kind == "file_error":
            self.file_failed.emit(ev.file, ev.message)
        self.progress.emit(ev)

    @Slot()
    def run(self):
        try:
            result = run_batch(self._settings, progress=self._on_event,
                               cancel_check=lambda: self._cancel,
                               only_files=self._only)
            if self._write and not self._cancel:
                write_batch(result, self._settings, self._settings.output.folder)
            # Final result carries every file's table for the results view.
            self.finished.emit(None if self._cancel else result)
        except Exception as e:  # fatal (e.g. no files, invalid settings)
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit(None)

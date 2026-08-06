"""A ``QApplication`` that also captures a native "open this file" launch.

macOS delivers a double-clicked or Dock "Open With…" document as a ``QEvent.FileOpen``
(Cocoa's ``application:openFile:``) sent straight to the application object — it is never
put on ``sys.argv``, unlike a Windows/Linux file-association launch. Nothing in a plain
``QApplication`` surfaces that event to application code; it has to be caught by overriding
``event()``.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication


def _capture_or_forward(target, path):
    """Record or forward a FileOpen path onto ``target`` (an app instance, or any stand-in
    exposing the same two attributes). Pulled out of ``RespMechApplication.event`` as a
    plain function so it is testable without a second live ``QApplication`` — Qt permits
    only one per process, and the shared test session already owns the one there is."""
    if target.on_file_open is not None:
        target.on_file_open(path)
    elif target.opened_path is None:
        target.opened_path = path


class RespMechApplication(QApplication):
    """Captures ``QEvent.FileOpen`` for ``ui.app.main()`` to fold into normal startup.

    The FIRST such event can arrive before the main window exists — on a cold launch it is
    typically delivered during app construction or the first ``processEvents()`` call, well
    before ``MainWindow`` is built — so it is simply remembered in ``opened_path`` rather
    than acted on immediately; ``main()`` reads it once it is ready to decide what to open.
    Any LATER one (the app already running; a second document opened from the Dock while
    the window is up) is instead forwarded live to ``on_file_open``, which ``main()`` points
    at the running window's own guarded open handler once one exists.
    """

    def __init__(self, argv):
        super().__init__(argv)
        self.opened_path: str | None = None
        self.on_file_open = None   # set by main() once a window exists; callable(path) -> None

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            _capture_or_forward(self, event.file())
            return True
        return super().event(event)

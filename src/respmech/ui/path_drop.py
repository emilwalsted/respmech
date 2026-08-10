"""Let a Finder/Explorer file or folder drag fill a path ``QLineEdit``.

A desktop user expects to drag a study folder onto "Recordings folder" and have the path
appear. Measured on the live Setup screen before this existed: the field advertises
``acceptDrops: True`` (``QLineEdit`` turns that on itself, to support dragging *selected
text* onto itself), but a real platform file/folder drag carries only ``text/uri-list`` on
the pasteboard/clipboard — never ``text/plain`` — and ``QLineEdit``'s own drag handling only
recognises ``text/plain``. So ``dragEnterEvent``/``dropEvent`` both come back
``accepted: False`` and the field's text is untouched; there is nothing broken to repair,
only a gesture the field never implemented.

``PathDropFilter`` is installed *on* the line edit as a Qt event filter, ahead of the
widget's own handling, and takes over the URL-carrying drag gesture entirely: a qualifying
drag is accepted and applied, a disqualifying one (several URLs, a remote URL, or — for a
``folder=True`` field — a URL that is an existing file rather than a directory) is rejected
outright, never partially applied and never handed back to ``QLineEdit``. Only a drag that
carries no URLs at all (an ordinary selected-text drag) is not our gesture and is left for
``QLineEdit``'s own, unchanged handling — that is the one case Qt's native drag-to-insert
keeps working through unmodified.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QObject


def _single_local_path(mime, folder=False):
    """The dropped path, iff ``mime`` carries exactly one local file/folder URL — else
    ``None``. A drag of several files, or of a remote/non-local URL, must be rejected
    outright rather than partially applied (e.g. by silently taking the first item).

    ``folder=True`` additionally rejects a path that exists and is a plain file: the
    Recordings/Output fields' own Browse… button only ever offers a folder picker
    (``QFileDialog.getExistingDirectory``), so a drag onto the same field accepting a file
    would be a real inconsistency, not a convenience — dropping an analysis .toml/.py
    squarely onto one of these two fields is meant to be caught here and rejected, so the
    window's own "drop this file to open it as an analysis" handling gets it instead (found
    in self-review: both features accept a drag on the SAME window, and the field is the
    more specific — and, before this check, the more permissive — target under the cursor).
    A path that does not exist yet is not rejected: a real platform drag only ever carries an
    existing filesystem entry, so this can only affect a synthetic drag anyway."""
    if mime is None or not mime.hasUrls():
        return None
    urls = mime.urls()
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = urls[0].toLocalFile()
    if not path:
        return None
    if folder and os.path.exists(path) and not os.path.isdir(path):
        return None
    # Native separators before the path reaches a field or the settings. QUrl.toLocalFile()
    # returns forward slashes on EVERY platform — including Windows ('C:/Users/…') — while
    # everything downstream (prefs, carried-folder matching, the user's own eyes) expects
    # the OS's native form. Measured on the Windows CI runner 10-08-2026: five path-drop
    # tests failed on exactly this, 'C:/Users/…' != 'C:\\Users\\…'; a no-op on macOS/Linux.
    return os.path.normpath(path)


class PathDropFilter(QObject):
    """Fills the target ``QLineEdit`` with a single dropped local path.

    Owns exactly the URL-carrying drag gesture: a qualifying single local URL is accepted
    and replaces the field's text; a disqualifying one is explicitly rejected so the OS shows
    a "not allowed" cursor rather than inviting a drop that would be silently discarded. A
    drag that carries no URLs at all (plain text) is not our gesture — it is left untouched
    for ``QLineEdit``'s own handling. See ``_single_local_path`` for what "qualifying" means,
    including the ``folder``-only file rejection."""

    def __init__(self, parent=None, folder=False):
        super().__init__(parent)
        self._folder = folder

    def eventFilter(self, obj, event):
        et = event.type()
        if et not in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            return super().eventFilter(obj, event)
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return False   # not a URL drag: QLineEdit's own handling decides (e.g. text/plain)
        path = _single_local_path(mime, folder=self._folder)
        if path is None:
            event.ignore()
            return True     # a disqualifying URL drag: reject it outright, field untouched
        event.acceptProposedAction()
        if et == QEvent.Type.Drop:
            obj.setText(path)
        return True


def install_path_drop(line_edit, folder=False) -> PathDropFilter:
    """Install a ``PathDropFilter`` on ``line_edit`` and return it.

    ``folder`` mirrors ``_with_browse``'s own flag: pass the same value the field's Browse…
    button was built with, so the drag and the button stay consistent about what the field
    accepts. The caller does not need to keep the return value: parenting the filter to the
    line edit (so it is destroyed together with it) is enough to keep it alive, same as any
    other Qt child object — unlike a filter installed on some other object with no owner of
    its own, there is no risk here of the filter being garbage-collected out from under a
    still-live widget."""
    f = PathDropFilter(line_edit, folder=folder)
    line_edit.installEventFilter(f)
    return f

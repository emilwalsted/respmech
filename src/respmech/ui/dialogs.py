"""Reusable dialogs — a copyable text/detail viewer used for error traces and the
batch error log. Qt-only; no core dependency."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QVBoxLayout)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None


def short_error(detail: str) -> str:
    """Reduce a full traceback (or multi-line message) to a one-line summary —
    the last non-empty line, which for a Python traceback is ``Type: message``."""
    if not detail:
        return "unknown error"
    lines = [ln.strip() for ln in str(detail).splitlines() if ln.strip()]
    return lines[-1] if lines else str(detail).strip()


class TextViewerDialog(QDialog):
    """A modeless window showing read-only, selectable, copyable monospace text
    (an error trace or the batch error log). A Copy button puts it on the
    clipboard; the text is also selectable directly."""

    def __init__(self, title: str, text: str, parent=None, intro: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(560, 340)
        self.resize(780, 500)
        lay = QVBoxLayout(self)
        if intro:
            lab = QLabel(intro)
            lab.setWordWrap(True)
            lab.setProperty("status", "muted")
            lay.addWidget(lab)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setPlainText(text or "")
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy to clipboard")
        self.copy_btn.setAutoDefault(False)
        self.copy_btn.clicked.connect(self._copy)
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)                 # Enter dismisses; single emphasised action
        close_btn.clicked.connect(self.accept)
        if _theme is not None:
            _theme.make_primary(close_btn)
        row.addStretch(1)
        row.addWidget(self.copy_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)

    def _copy(self):
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self.view.toPlainText())
        self.copy_btn.setText("Copied ✓")
        # bind the reset to the button's lifetime, so it never fires on a deleted widget
        QTimer.singleShot(1200, self.copy_btn,
                          lambda b=self.copy_btn: b.setText("Copy to clipboard"))

    def text(self) -> str:
        return self.view.toPlainText()


def open_error_dialog(parent, title: str, detail: str, intro: str | None = None, prior=None):
    """Show a copyable full-detail (traceback) dialog, replacing ``prior`` if given
    so repeated errors don't accumulate windows. Returns the new dialog to keep a
    reference on the caller."""
    if prior is not None:
        try:
            prior.close()
            prior.deleteLater()
        except Exception:                          # pragma: no cover - prior already gone
            pass
    dlg = TextViewerDialog(title, detail, parent, intro=intro)
    dlg.show()
    dlg.raise_()
    return dlg

"""'Duplicate for another recordings folder…' confirmation (ticket C03 point 5): shows
the new recordings folder (already chosen by the caller) and a SUGGESTED output folder
for confirmation/editing — never applied silently, per the ticket's own instruction.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None


class DuplicateFolderDialog(QDialog):
    """Confirm/edit the output folder for a duplicated analysis. ``new_input`` is shown
    read-only (already picked by the caller); ``suggested_output`` pre-fills the editable
    output field, or is left blank (with a muted hint) when the caller could not derive
    one (``ui.duplicate.derive_sibling_output`` returned ``None``)."""

    def __init__(self, new_input: str, suggested_output: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Duplicate for another recordings folder")
        self.setModal(True)

        v = QVBoxLayout(self)
        v.setContentsMargins(26, 22, 26, 20)
        v.setSpacing(10)

        title = QLabel("Duplicate this analysis for a new recordings folder")
        title.setProperty("role", "heading")
        title.setWordWrap(True)
        v.addWidget(title)
        sub = QLabel("Every other setting is kept as it is. Confirm where the results "
                     "should go before saving this as a new analysis.")
        sub.setProperty("status", "muted")
        sub.setWordWrap(True)
        v.addWidget(sub)
        v.addSpacing(6)

        v.addWidget(QLabel("New recordings folder"))
        in_lab = QLabel(new_input)
        in_lab.setWordWrap(True)
        in_lab.setProperty("banner", True)
        v.addWidget(in_lab)

        v.addSpacing(4)
        v.addWidget(QLabel("Output folder for this duplicate"))
        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
        self.out_line = QLineEdit(suggested_output or "")
        row.addWidget(self.out_line, 1)
        browse = QPushButton("Browse…")
        browse.setProperty("compact", True)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        v.addLayout(row)
        if not suggested_output:
            hint = QLabel("Could not guess an output folder from the current analysis "
                          "(its output is not a sibling of its recordings folder) — "
                          "choose one.")
            hint.setProperty("status", "warn")
            hint.setWordWrap(True)
            v.addWidget(hint)

        foot = QHBoxLayout()
        foot.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        ok = QPushButton("Continue")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        if _theme is not None:
            _theme.make_primary(ok)
        foot.addWidget(ok)
        v.addLayout(foot)

        self.setMinimumWidth(520)
        self.adjustSize()

    def _browse(self):
        start = self.out_line.text().strip() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if d:
            self.out_line.setText(d)

    def output_folder(self) -> str:
        return self.out_line.text().strip()

    def showEvent(self, ev):                # noqa: N802 - Qt API
        """Never taller than the screen — same guard every dialog in this codebase uses."""
        super().showEvent(ev)
        if not getattr(self, "_clamped", False):
            self._clamped = True
            from respmech.ui import screen_fit  # noqa: PLC0415
            screen_fit.clamp_to_screen(self, prefer=QSize(560, self.height()))

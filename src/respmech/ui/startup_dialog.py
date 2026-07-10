"""Startup chooser — the first thing shown when RespMech opens.

A clear fork: start a brand-new analysis (guided, step by step) or open an existing
one (a saved analysis, or a legacy .py setup). "Analysis" is the user-facing name for
a settings file; the word "TOML" never appears here. The dialog exposes its outcome as
``mode`` ("new"/"open") and, for open, ``path``; the caller performs the actual load,
so this stays purely presentational and easy to test headless.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCommandLinkButton, QDialog, QFileDialog,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout)

# File-dialog filter used both here and by the Settings screen's "Open analysis…".
OPEN_FILTER = ("Analysis files (*.toml *.py);;Saved analysis (*.toml);;"
               "Legacy setup (*.py);;All files (*)")


class StartupDialog(QDialog):
    """Ask the user to start a New analysis or Open an existing one.

    After ``exec()``: ``mode`` is "new" or "open"; when "open", ``path`` is the chosen
    file (else ``None``). Cancelling/closing leaves the default (``mode == "new"``,
    ``path is None``) and rejects, so the caller can treat a cancel as "start new"."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RespMech — get started")
        self.setModal(True)
        self.mode = "new"
        self.path = None

        v = QVBoxLayout(self)
        v.setContentsMargins(26, 22, 26, 20)
        v.setSpacing(8)

        title = QLabel("Start a new analysis or open an existing one")
        title.setProperty("role", "heading")
        title.setWordWrap(True)
        v.addWidget(title)
        sub = QLabel("An analysis holds every setting for one batch of recordings.")
        sub.setProperty("status", "muted")
        sub.setWordWrap(True)
        v.addWidget(sub)
        v.addSpacing(6)

        self.new_btn = QCommandLinkButton(
            "New analysis",
            "Set it up step by step, starting from the input recordings.")
        self.open_btn = QCommandLinkButton(
            "Open analysis…",
            "Continue from a saved analysis, or import a legacy .py setup.")
        self.new_btn.clicked.connect(self._choose_new)
        self.open_btn.clicked.connect(self._choose_open)
        v.addWidget(self.new_btn)
        v.addWidget(self.open_btn)

        foot = QHBoxLayout()
        foot.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        v.addLayout(foot)

        self.resize(480, 300)

    def _choose_new(self):
        self.mode = "new"
        self.path = None
        self.accept()

    def _choose_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open analysis", ".", OPEN_FILTER)
        if p:                                   # a cancelled file picker keeps the chooser open
            self.mode = "open"
            self.path = p
            self.accept()

"""Startup chooser — the first thing shown when RespMech opens.

A clear fork: start a brand-new analysis (guided, step by step) or open an existing
one (a saved analysis, or a legacy .py setup). "Analysis" is the user-facing name for
a settings file; the word "TOML" never appears here. The dialog exposes its outcome as
``mode`` and, for open, ``path``; the caller performs the actual load, so this stays
purely presentational and easy to test headless.

Modes: ``new`` (guided from scratch), ``new_rig`` (guided, inheriting the last channel
rig — P25), ``open`` (a chosen/recent analysis — P26), ``sample`` (explore bundled
synthetic data — P23). Cancelling rejects and leaves ``mode == "new"``.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCommandLinkButton, QDialog, QFileDialog,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout)

# File-dialog filter used both here and by the Settings screen's "Open analysis…".
OPEN_FILTER = ("Analysis files (*.toml *.py);;Saved analysis (*.toml);;"
               "Legacy setup (*.py);;All files (*)")


class StartupDialog(QDialog):
    """Ask the user how to start. After ``exec()``: ``mode`` is one of
    new/new_rig/open/sample; when "open", ``path`` is the chosen file. Cancelling
    leaves the default (``mode == "new"``, ``path is None``) and rejects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RespMech — Get started")
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

        # P25: reuse the channel mapping + sampling of the last analysis
        from respmech.ui import prefs  # noqa: PLC0415
        if prefs.last_rig():
            self.rig_btn = QCommandLinkButton(
                "New from last rig",
                "Start a new analysis with the channel mapping you used last time.")
            self.rig_btn.clicked.connect(self._choose_new_rig)
            v.addWidget(self.rig_btn)

        v.addWidget(self.open_btn)

        # P23: a no-setup door into the app for first-time users
        self.sample_btn = QCommandLinkButton(
            "Explore with sample data",
            "Open a small built-in synthetic recording to see how RespMech works.")
        self.sample_btn.clicked.connect(self._choose_sample)
        v.addWidget(self.sample_btn)

        # P26: recent analyses as one-click entries
        recents = prefs.recent_analyses()
        if recents:
            lab = QLabel("Recent analyses")
            lab.setProperty("status", "muted")
            v.addSpacing(4); v.addWidget(lab)
            for path in recents[:5]:
                btn = QPushButton(os.path.basename(path))
                btn.setToolTip(path)
                btn.setProperty("compact", True)
                btn.clicked.connect(lambda _=False, p=path: self._choose_recent(p))
                v.addWidget(btn)

        foot = QHBoxLayout()
        foot.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        v.addLayout(foot)

        # size to content — the door/recent count varies, so a fixed height would clip
        # the two-line command-link descriptions
        self.setMinimumWidth(480)
        self.adjustSize()

    def _choose_new(self):
        self.mode = "new"; self.path = None; self.accept()

    def _choose_new_rig(self):
        self.mode = "new_rig"; self.path = None; self.accept()

    def _choose_sample(self):
        self.mode = "sample"; self.path = None; self.accept()

    def _choose_recent(self, path):
        self.mode = "open"; self.path = path; self.accept()

    def _choose_open(self):
        from respmech.ui import prefs  # noqa: PLC0415
        p, _ = QFileDialog.getOpenFileName(
            self, "Open analysis", prefs.last_folder("analysis", "."), OPEN_FILTER)
        if p:                                   # a cancelled file picker keeps the chooser open
            self.mode = "open"; self.path = p; self.accept()

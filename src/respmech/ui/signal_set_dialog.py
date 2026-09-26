"""SignalSetDialog — choosing which signals a new analysis declares (part of R7).

Opened before a fresh analysis is set up (the startup chooser's 'New analysis' door,
and 'File > New analysis'), never over an already-configured one — Setup's own
'Signals' row and its 'Change...' door (a later release) are a separate entry point
onto the same underlying model. The outcome is exposed as ``signals``: a list of signal
names (``'flow'``, ``'poes'``, ``'pgas'``, ``'pdi'``, ``'emg'``) after ``exec()``
returns ``QDialog.Accepted``, or ``None`` if the dialog is cancelled — the caller must
leave the previous analysis untouched in that case.

Every preset shown here matches ``core.analysis.signals``'s vocabulary exactly, so a
chosen set is always a valid ``analysis.signals`` value. Only 'Flow + Poes + Pgas +
Pdi' (the full set, today's shape, with or without EMG via the 'Also EMG' toggle) is
reachable in THIS milestone: 'Flow only', 'Flow + Poes', 'EMG only' and 'Custom...'
are shown — so the dialog's eventual, unchanging shape is visible early, and the
dark-mode/lone-ampersand/windows-metrics checks already cover it — but disabled, with
'Available in a later step.' as their description, until the compute guards (a later
ticket) and the relevance-driven UI / EMG-only recording-content question (later
tickets still) actually support running an analysis on a reduced set. Widening which
buttons are enabled is exactly that later ticket's job; nothing else about this dialog
should need to change for it.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QCommandLinkButton, QDialog, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout)

from respmech.ui.help_text import tooltip as _tip

#: Shown as the description of every preset this milestone does not yet support.
_LATER_STEP = "Available in a later step."


class SignalSetDialog(QDialog):
    """Choose a new analysis's signal set. After ``exec()`` returns ``QDialog.Accepted``,
    ``signals`` is the chosen list (e.g. ``['flow', 'poes', 'pgas', 'pdi']``, optionally
    with ``'emg'`` appended); a rejected/cancelled dialog leaves it ``None``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RespMech — Signal set")
        self.setModal(True)
        self.signals: list[str] | None = None

        v = QVBoxLayout(self)
        v.setContentsMargins(26, 22, 26, 20)
        v.setSpacing(8)

        title = QLabel("Choose the signals this analysis uses")
        title.setProperty("role", "heading")
        title.setWordWrap(True)
        v.addWidget(title)
        sub = QLabel("This decides which channels, panels and results the rest of the "
                     "app offers for this analysis.")
        sub.setProperty("status", "muted")
        sub.setWordWrap(True)
        v.addWidget(sub)
        v.addSpacing(6)

        self.flow_only_btn = QCommandLinkButton(
            "Flow only", f"Breath timing from flow (and volume) alone. {_LATER_STEP}")
        self.flow_poes_btn = QCommandLinkButton(
            "Flow + Poes",
            f"Adds work of breathing from oesophageal pressure. {_LATER_STEP}")
        self.full_btn = QCommandLinkButton(
            "Flow + Poes + Pgas + Pdi",
            "The complete mechanics set: work of breathing, gastric and "
            "transdiaphragmatic pressure, and ventilatory muscle ratio.")
        self.also_emg = QCheckBox("Also EMG")
        self.emg_only_btn = QCommandLinkButton(
            "EMG only", f"Diaphragm EMG alone, no breath mechanics. {_LATER_STEP}")
        self.custom_btn = QCommandLinkButton(
            "Custom…", f"Choose exactly which signals to declare. {_LATER_STEP}")

        for btn in (self.flow_only_btn, self.flow_poes_btn, self.emg_only_btn, self.custom_btn):
            btn.setEnabled(False)
            btn.setToolTip(_tip("analysis.signals", _LATER_STEP))
        self.full_btn.setToolTip(_tip(
            "analysis.signals",
            "flow, poes, pgas, pdi — the default, and today's only reachable preset."))
        self.also_emg.setToolTip(_tip(
            "analysis.signals",
            "Adds 'emg' to the flow preset chosen above."))

        v.addWidget(self.flow_only_btn)
        v.addWidget(self.flow_poes_btn)
        v.addWidget(self.full_btn)
        v.addWidget(self.also_emg)
        v.addWidget(self.emg_only_btn)
        v.addWidget(self.custom_btn)

        self.full_btn.clicked.connect(self._choose_full)

        foot = QHBoxLayout()
        foot.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        v.addLayout(foot)

        # size to content, same reasoning as StartupDialog: a fixed height would clip
        # the multi-line command-link descriptions
        self.setMinimumWidth(480)
        self.adjustSize()

    def _choose_full(self):
        signals = ["flow", "poes", "pgas", "pdi"]
        if self.also_emg.isChecked():
            signals.append("emg")
        self.signals = signals
        self.accept()

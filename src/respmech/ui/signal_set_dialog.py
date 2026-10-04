"""SignalSetDialog — choosing which signals a new analysis declares (part of R7).

Opened before a fresh analysis is set up (the startup chooser's 'New analysis' door,
and 'File > New analysis'), never over an already-configured one — Setup's own
'Signals' row and its 'Change...' door are a separate entry point onto the same
underlying model (``settings_screen.apply_signal_set``). The outcome is exposed as
``signals``: a list of signal names (``'flow'``, ``'poes'``, ``'pgas'``, ``'pdi'``,
``'emg'``) after ``exec()`` returns ``QDialog.Accepted``, or ``None`` if the dialog is
cancelled — the caller must leave the previous analysis untouched in that case.
``segmentation_method`` carries the EMG-only preset's own extra choice (see below); it
is ``None`` for every other preset.

Every preset shown here matches ``core.analysis.signals``'s vocabulary exactly, so a
chosen set is always a valid ``analysis.signals`` value. Four of the five presets are
reachable now: the three flow-family presets ('Flow only', 'Flow + Poes' and
'Flow + Poes + Pgas + Pdi', each with or without EMG via the 'Also EMG' toggle), and
'EMG only'. 'Custom...' is still shown — so the dialog's eventual, unchanging
shape is visible early, and the dark-mode/lone-ampersand/windows-metrics checks already
cover it — but disabled, with 'Available in a later step.' as its description, until
the Custom checkbox picker (a later release) actually supports running an analysis on
an arbitrary reduced set. Widening it is exactly that later release's job; nothing
else about this dialog should need to change for it.

'EMG only' asks ONE extra question — via :class:`EmgRecordingContentDialog`, stacked
on top of this one — because unlike the flow-family presets, which all use the same
breath-timing-from-flow segmentation regardless of which pressures are added,
declaring EMG alone leaves open HOW the recording is split into segments at all
(there is no flow to detect a breath boundary from). Cancelling that sub-dialog
leaves this one open and unchanged, exactly like cancelling any other in-app dialog
leaves its caller's state untouched.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QCommandLinkButton, QDialog, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout)

from respmech.ui.help_text import tooltip as _tip

#: Shown as the description of every preset this milestone does not yet support.
_LATER_STEP = "Available in a later step."


class EmgRecordingContentDialog(QDialog):
    """The ONE place the EMG-only recording-content question is asked: what does a
    single file in this analysis actually CONTAIN, since there is no flow channel to
    detect a breath boundary from. Reused unchanged by ``SignalSetDialog`` (choosing
    the EMG-only preset for a new analysis) and by Setup's own 'Change…' door
    re-asking it for an already-EMG-only analysis — the SAME dialog both times, never
    two copies drifting apart.

    After ``exec()`` returns ``QDialog.Accepted``, ``method`` is one of
    ``'whole_file'``/``'separators'``/``'emg_burst'`` (``core.settings.
    SegmentationSettings.method``'s own EMG-only vocabulary; ``'emg_burst'`` is the
    automatic detection of tidal-breathing bursts, whose thresholds are starting values
    until they have been calibrated on real recordings); a cancelled/rejected dialog
    leaves it ``None``. ``'fixed_windows'`` is a method of the same family but not one of
    the three questions here: it is chosen in the analysis file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RespMech — Recording content")
        self.setModal(True)
        self.method: str | None = None

        v = QVBoxLayout(self)
        v.setContentsMargins(26, 22, 26, 20)
        v.setSpacing(8)

        title = QLabel("What does this recording contain?")
        title.setProperty("role", "heading")
        title.setWordWrap(True)
        v.addWidget(title)
        sub = QLabel("EMG alone has no flow channel to detect a breath boundary "
                     "from, so this decides how each file is split into segments "
                     "instead.")
        sub.setProperty("status", "muted")
        sub.setWordWrap(True)
        v.addWidget(sub)
        v.addSpacing(6)

        self.whole_file_btn = QCommandLinkButton(
            "One maximal manoeuvre",
            "The whole file is one effort — a sniff or a maximal voluntary "
            "contraction.")
        self.separators_btn = QCommandLinkButton(
            "Several efforts or breaths",
            "Place the segment boundaries yourself in Preview && QC.")
        self.emg_burst_btn = QCommandLinkButton(
            "Tidal breathing — detect bursts automatically",
            "Split each file into segments from the EMG activity itself, one per "
            "inspiratory burst.")

        self.emg_burst_btn.setToolTip(_tip(
            "processing.segmentation.method",
            "emg_burst — one segment per detected burst of EMG activity, with neural "
            "timing. The thresholds are adjustable in Preview & QC ▸ EMG – segments ▸ "
            "Advanced…"))
        self.whole_file_btn.setToolTip(_tip(
            "processing.segmentation.method",
            "whole_file — the entire recording is one segment."))
        self.separators_btn.setToolTip(_tip(
            "processing.segmentation.method",
            "separators — place the segment boundaries yourself, in "
            "Preview & QC ▸ EMG – segments."))

        v.addWidget(self.whole_file_btn)
        v.addWidget(self.separators_btn)
        v.addWidget(self.emg_burst_btn)

        self.whole_file_btn.clicked.connect(self._choose_whole_file)
        self.separators_btn.clicked.connect(self._choose_separators)
        self.emg_burst_btn.clicked.connect(self._choose_emg_burst)

        foot = QHBoxLayout()
        foot.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        v.addLayout(foot)

        self.setMinimumWidth(480)
        self.adjustSize()

    def _choose_whole_file(self):
        self.method = "whole_file"
        self.accept()

    def _choose_separators(self):
        self.method = "separators"
        self.accept()

    def _choose_emg_burst(self):
        self.method = "emg_burst"
        self.accept()


class SignalSetDialog(QDialog):
    """Choose a new analysis's signal set. After ``exec()`` returns ``QDialog.Accepted``,
    ``signals`` is the chosen list (e.g. ``['flow', 'poes', 'pgas', 'pdi']``, optionally
    with ``'emg'`` appended); a rejected/cancelled dialog leaves it ``None``.
    ``segmentation_method`` is set alongside ``signals`` ONLY for the EMG-only preset
    (see :class:`EmgRecordingContentDialog`) — ``None`` for every other preset, since
    they all share the ordinary flow-based breath segmentation and have nothing new to
    decide here."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RespMech — Signal set")
        self.setModal(True)
        self.signals: list[str] | None = None
        self.segmentation_method: str | None = None

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
            "Flow only", "Breath timing from flow (and volume) alone.")
        self.flow_poes_btn = QCommandLinkButton(
            "Flow + Poes", "Adds work of breathing from oesophageal pressure.")
        self.full_btn = QCommandLinkButton(
            "Flow + Poes + Pgas + Pdi",
            "The complete mechanics set: work of breathing, gastric and "
            "transdiaphragmatic pressure, and ventilatory muscle ratio.")
        self.also_emg = QCheckBox("Also EMG")
        self.emg_only_btn = QCommandLinkButton(
            "EMG only", "Diaphragm EMG alone, no breath mechanics.")
        self.custom_btn = QCommandLinkButton(
            "Custom…", f"Choose exactly which signals to declare. {_LATER_STEP}")

        self.custom_btn.setEnabled(False)
        self.custom_btn.setToolTip(_tip("analysis.signals", _LATER_STEP))
        self.emg_only_btn.setToolTip(_tip(
            "analysis.signals",
            "emg — diaphragm EMG alone, no flow or pressure channels; asks how "
            "each recording is split into segments (R7)."))
        self.flow_only_btn.setToolTip(_tip(
            "analysis.signals", "flow — breath timing (and volume) alone, no pressure "
            "channels; the Mechanics stack and Campbell panel adjust to match (R7)."))
        self.flow_poes_btn.setToolTip(_tip(
            "analysis.signals", "flow, poes — adds work of breathing from oesophageal "
            "pressure, without Pgas/Pdi."))
        self.full_btn.setToolTip(_tip(
            "analysis.signals",
            "flow, poes, pgas, pdi — the complete, default set."))
        self.also_emg.setToolTip(_tip(
            "analysis.signals",
            "Adds 'emg' to the flow preset chosen above."))

        v.addWidget(self.flow_only_btn)
        v.addWidget(self.flow_poes_btn)
        v.addWidget(self.full_btn)
        v.addWidget(self.also_emg)
        v.addWidget(self.emg_only_btn)
        v.addWidget(self.custom_btn)

        self.flow_only_btn.clicked.connect(self._choose_flow_only)
        self.flow_poes_btn.clicked.connect(self._choose_flow_poes)
        self.full_btn.clicked.connect(self._choose_full)
        self.emg_only_btn.clicked.connect(self._choose_emg_only)

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

    def _choose_flow_only(self):
        signals = ["flow"]
        if self.also_emg.isChecked():
            signals.append("emg")
        self.signals = signals
        self.accept()

    def _choose_flow_poes(self):
        signals = ["flow", "poes"]
        if self.also_emg.isChecked():
            signals.append("emg")
        self.signals = signals
        self.accept()

    def _choose_full(self):
        signals = ["flow", "poes", "pgas", "pdi"]
        if self.also_emg.isChecked():
            signals.append("emg")
        self.signals = signals
        self.accept()

    def _choose_emg_only(self):
        """Ask the recording-content question BEFORE accepting this dialog — an
        EMG-only preset with no method chosen would leave
        ``processing.segmentation.method`` at its stale, irrelevant flow-family
        default. Cancelling the sub-dialog leaves THIS dialog open and completely
        unchanged (``self.signals``/``self.segmentation_method`` both stay whatever
        they were before the click), the same as clicking 'Cancel' on any other
        button here never partially commits."""
        sub = EmgRecordingContentDialog(self)
        if sub.exec() != QDialog.Accepted or sub.method is None:
            return
        self.signals = ["emg"]
        self.segmentation_method = sub.method
        self.accept()

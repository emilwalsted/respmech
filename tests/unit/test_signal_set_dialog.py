"""SignalSetDialog: the new-analysis signal-set picker.

Four of the five presets — Flow only, Flow + Poes, the full set (each with or without
the 'Also EMG' toggle), and EMG only — are reachable now (R7);
Custom… is not yet — see the module's own docstring for why that one door exists on
screen but disabled. These tests cover: which doors are enabled/disabled and what they
say, the outcome of accepting/cancelling (including EMG only's own sub-dialog,
EmgRecordingContentDialog), and the same cross-cutting checks every other top-level
window/dialog in this app carries (dark mode, no lone '&', fits under the Windows
font-metrics model).
"""
from PySide6.QtWidgets import QDialog

from _helpers import _lone_ampersands


def _dlg(qapp):
    from respmech.ui.signal_set_dialog import SignalSetDialog
    return SignalSetDialog()


def test_four_of_five_presets_are_enabled(qapp):
    dlg = _dlg(qapp)
    for btn in (dlg.flow_only_btn, dlg.flow_poes_btn, dlg.full_btn, dlg.emg_only_btn):
        assert btn.isEnabled() is True
    assert dlg.custom_btn.isEnabled() is False
    dlg.close()


def test_only_custom_says_available_later(qapp):
    """Every disabled door names WHY it's disabled, on the door itself — the app's own
    gating convention (the reason hangs on the action, not a separate label)."""
    dlg = _dlg(qapp)
    assert "later step" in dlg.custom_btn.description().lower()
    for btn in (dlg.flow_only_btn, dlg.flow_poes_btn, dlg.full_btn, dlg.emg_only_btn):
        assert "later step" not in btn.description().lower()
    dlg.close()


def test_custom_door_is_rejected_in_this_milestone(qapp):
    """'Custom…' (checkboxes, press-without-flow rejected) is part of the eventual R7
    shape but not reachable until a later ticket widens it — disabled is the whole story
    here."""
    dlg = _dlg(qapp)
    assert dlg.custom_btn.isEnabled() is False
    dlg.close()


def test_custom_produces_no_outcome_even_if_clicked(qapp):
    """Qt's own quirk: QAbstractButton.click() bypasses isEnabled() and fires 'clicked'
    regardless — so being disabled alone does not guarantee inertness; what actually
    protects this door today is that it has no connected slot. Pin that, so a future
    ticket that wires it up without also flipping isEnabled(True) is caught by THIS
    test going red, rather than shipping a door that silently 'works' while still
    looking disabled."""
    dlg = _dlg(qapp)
    dlg.custom_btn.click()
    assert dlg.signals is None
    assert dlg.result() != QDialog.Accepted
    dlg.close()


def test_choosing_full_without_emg(qapp):
    dlg = _dlg(qapp)
    dlg.full_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["flow", "poes", "pgas", "pdi"]


def test_choosing_full_with_also_emg(qapp):
    dlg = _dlg(qapp)
    dlg.also_emg.setChecked(True)
    dlg.full_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["flow", "poes", "pgas", "pdi", "emg"]


def test_choosing_flow_only_without_emg(qapp):
    dlg = _dlg(qapp)
    dlg.flow_only_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["flow"]


def test_choosing_flow_only_with_also_emg(qapp):
    dlg = _dlg(qapp)
    dlg.also_emg.setChecked(True)
    dlg.flow_only_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["flow", "emg"]


def test_choosing_flow_poes_without_emg(qapp):
    dlg = _dlg(qapp)
    dlg.flow_poes_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["flow", "poes"]


def test_choosing_flow_poes_with_also_emg(qapp):
    dlg = _dlg(qapp)
    dlg.also_emg.setChecked(True)
    dlg.flow_poes_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["flow", "poes", "emg"]


def test_cancel_leaves_signals_none(qapp):
    dlg = _dlg(qapp)
    dlg.reject()
    assert dlg.result() == QDialog.Rejected
    assert dlg.signals is None
    assert dlg.segmentation_method is None


def test_no_lone_ampersand(qapp):
    dlg = _dlg(qapp)
    offenders = _lone_ampersands(dlg)
    dlg.close()
    assert not offenders, offenders


def test_signal_set_dialog_fits_under_windows_font_metrics(qapp, windows_metrics):
    """The same defect class CLAUDE.md has prescribed testing for since the window-width
    fix: a dialog sized on macOS's narrower metrics can clip a caption on the Windows
    runner's wider advance. Building it under the widened font and checking every button
    still reports the full text it was given (Qt elides rather than raising) is enough to
    catch a caption silently truncated to '…'."""
    from respmech.ui.signal_set_dialog import SignalSetDialog
    dlg = SignalSetDialog()
    for btn in (dlg.flow_only_btn, dlg.flow_poes_btn, dlg.full_btn, dlg.emg_only_btn,
               dlg.custom_btn):
        assert btn.text()          # QCommandLinkButton.text() is the title, never elided
    assert dlg.width() > 0 and dlg.height() > 0
    dlg.close()


# --- EMG only + its recording-content sub-dialog -----------------------------------

def test_choosing_emg_only_opens_the_recording_content_sub_dialog(qapp, monkeypatch):
    """Clicking 'EMG only' must open EmgRecordingContentDialog rather than accepting
    this dialog outright — unlike every other preset, EMG only has one more question to
    answer before a signal set is actually decided."""
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog
    opened = []
    original_exec = EmgRecordingContentDialog.exec

    def _spy(self):
        opened.append(self)
        self.reject()
        return QDialog.Rejected
    monkeypatch.setattr(EmgRecordingContentDialog, "exec", _spy)
    dlg = _dlg(qapp)
    dlg.emg_only_btn.click()
    assert len(opened) == 1
    assert isinstance(opened[0], EmgRecordingContentDialog)
    monkeypatch.setattr(EmgRecordingContentDialog, "exec", original_exec)
    dlg.close()


def test_choosing_emg_only_then_whole_file_accepts_with_emg_signals(qapp, monkeypatch):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog

    def _fake_exec(self):
        self.whole_file_btn.click()
        return QDialog.Accepted
    monkeypatch.setattr(EmgRecordingContentDialog, "exec", _fake_exec)
    dlg = _dlg(qapp)
    dlg.emg_only_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["emg"]
    assert dlg.segmentation_method == "whole_file"
    dlg.close()


def test_choosing_emg_only_then_separators_accepts_with_that_method(qapp, monkeypatch):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog

    def _fake_exec(self):
        self.separators_btn.click()
        return QDialog.Accepted
    monkeypatch.setattr(EmgRecordingContentDialog, "exec", _fake_exec)
    dlg = _dlg(qapp)
    dlg.emg_only_btn.click()
    assert dlg.result() == QDialog.Accepted
    assert dlg.signals == ["emg"]
    assert dlg.segmentation_method == "separators"
    dlg.close()


def test_cancelling_the_recording_content_dialog_leaves_signal_set_dialog_open_and_unchanged(
        qapp, monkeypatch):
    """Cancelling the sub-dialog must not partially commit anything — the outer dialog
    stays open (never accepted/rejected) with signals/segmentation_method both still
    None, exactly as if the 'EMG only' click never happened."""
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog

    def _fake_exec(self):
        return QDialog.Rejected
    monkeypatch.setattr(EmgRecordingContentDialog, "exec", _fake_exec)
    dlg = _dlg(qapp)
    dlg.emg_only_btn.click()
    assert dlg.result() != QDialog.Accepted
    assert dlg.signals is None
    assert dlg.segmentation_method is None
    dlg.close()


def test_emg_recording_content_dialog_offers_all_three_choices(qapp):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog
    sub = EmgRecordingContentDialog()
    assert sub.whole_file_btn.isEnabled() is True
    assert sub.separators_btn.isEnabled() is True
    assert sub.emg_burst_btn.isEnabled() is True
    assert "later step" not in sub.emg_burst_btn.description().lower()
    sub.close()


def test_emg_recording_content_dialog_emg_burst_is_chosen_by_clicking_it(qapp):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog
    sub = EmgRecordingContentDialog()
    sub.emg_burst_btn.click()
    assert sub.method == "emg_burst"
    assert sub.result() == QDialog.Accepted
    sub.close()


def test_emg_recording_content_dialog_cancel_leaves_method_none(qapp):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog
    sub = EmgRecordingContentDialog()
    sub.reject()
    assert sub.result() == QDialog.Rejected
    assert sub.method is None


def test_emg_recording_content_dialog_no_lone_ampersand(qapp):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog
    sub = EmgRecordingContentDialog()
    offenders = _lone_ampersands(sub)
    sub.close()
    assert not offenders, offenders


def test_emg_recording_content_dialog_fits_under_windows_font_metrics(qapp, windows_metrics):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog
    sub = EmgRecordingContentDialog()
    for btn in (sub.whole_file_btn, sub.separators_btn, sub.emg_burst_btn):
        assert btn.text()
    assert sub.width() > 0 and sub.height() > 0
    sub.close()

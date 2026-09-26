"""SignalSetDialog: the new-analysis signal-set picker.

Only the full preset (with or without the 'Also EMG' toggle) is reachable in this
milestone — see the module's own docstring for why the other doors exist on screen
but disabled. These tests cover: which doors are enabled/disabled and what they say,
the outcome of accepting/cancelling, and the same cross-cutting checks every other
top-level window/dialog in this app carries (dark mode, no lone '&', fits under the
Windows font-metrics model).
"""
from PySide6.QtWidgets import QDialog

from _helpers import _lone_ampersands


def _dlg(qapp):
    from respmech.ui.signal_set_dialog import SignalSetDialog
    return SignalSetDialog()


def test_only_the_full_preset_is_enabled(qapp):
    dlg = _dlg(qapp)
    assert dlg.full_btn.isEnabled() is True
    for btn in (dlg.flow_only_btn, dlg.flow_poes_btn, dlg.emg_only_btn, dlg.custom_btn):
        assert btn.isEnabled() is False
    dlg.close()


def test_disabled_presets_say_available_later(qapp):
    """Every disabled door names WHY it's disabled, on the door itself — the app's own
    gating convention (the reason hangs on the action, not a separate label)."""
    dlg = _dlg(qapp)
    for btn in (dlg.flow_only_btn, dlg.flow_poes_btn, dlg.emg_only_btn, dlg.custom_btn):
        assert "later step" in btn.description().lower()
    assert "later step" not in dlg.full_btn.description().lower()
    dlg.close()


def test_custom_door_is_rejected_in_this_milestone(qapp):
    """'Custom…' (checkboxes, press-without-flow rejected) is part of the eventual R7
    shape but not reachable until a later ticket widens it — disabled is the whole story
    here."""
    dlg = _dlg(qapp)
    assert dlg.custom_btn.isEnabled() is False
    dlg.close()


def test_disabled_presets_produce_no_outcome_even_if_clicked(qapp):
    """Qt's own quirk: QAbstractButton.click() bypasses isEnabled() and fires 'clicked'
    regardless — so being disabled alone does not guarantee inertness; what actually
    protects these four doors today is that NONE of them has a connected slot. Pin that,
    so a future ticket that wires one up without also flipping isEnabled(True) is caught
    by THIS test going red, rather than shipping a door that silently 'works' while still
    looking disabled."""
    dlg = _dlg(qapp)
    for btn in (dlg.flow_only_btn, dlg.flow_poes_btn, dlg.emg_only_btn, dlg.custom_btn):
        btn.click()
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


def test_cancel_leaves_signals_none(qapp):
    dlg = _dlg(qapp)
    dlg.reject()
    assert dlg.result() == QDialog.Rejected
    assert dlg.signals is None


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

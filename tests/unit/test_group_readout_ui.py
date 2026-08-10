"""D19 (UI-overhaul): a live "Group files by" read-out on Setup, and the same line
repeated in the dry-run plan -- so a study's grouping pattern can be checked before a
run instead of only being visible afterwards by opening the written "By group" sheet.

The pure computation (``group_readout``, three states: normal/warn/error) has its own
Qt-free tests in ``test_manifest.py``; this file covers the WIRING -- that Setup's field
actually shows each state, that an invalid pattern does not stop settings from being
saved, and that Run's dry-run plan carries the same text.
"""
from PySide6.QtWidgets import QApplication

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _screen(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(synth_settings(str(tmp_path))))
    sc.show()
    qapp.processEvents()
    return sc


def test_setup_shows_the_normal_state_for_the_default_two_file_batch(qapp, tmp_path):
    """synth_case_A.csv / synth_case_B.csv both share the leading token 'synth' under
    the default (blank-pattern) grouping -- one group, and the row must be visible with
    an "info"-status line naming it, not left empty/hidden the way it starts before any
    folder is scanned."""
    sc = _screen(qapp, tmp_path)
    assert sc.group_readout.isVisible()
    assert sc.group_readout.property("status") == "info"
    assert "2 files" in sc.group_readout.text() and "1 group" in sc.group_readout.text()


def test_setup_warns_and_names_files_a_partial_pattern_pools_as_all(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    sc.group_regex.setText(r"_([a-z]+)\.csv")   # neither synth_case_A.csv nor _B.csv has a
    sc._on_field_changed()                       # trailing lowercase-only token before .csv
    qapp.processEvents()
    assert sc.group_readout.property("status") == "warn"
    text = sc.group_readout.text()
    assert "(all)" in text
    assert "synth_case_A.csv" in text and "synth_case_B.csv" in text


def test_setup_shows_re_errors_own_message_for_an_invalid_pattern(qapp, tmp_path):
    """The read-out must not just say 'something is wrong' -- it repeats re.error's own
    message, and the pattern must still be a perfectly saveable setting (core.summary's
    OWN except re.error fallback exists precisely so a bad pattern never crashes a run;
    this UI-level check confirms wiring this read-out did not accidentally change that)."""
    sc = _screen(qapp, tmp_path)
    sc.group_regex.setText("[unclosed")
    sc._on_field_changed()
    qapp.processEvents()
    assert sc.group_readout.property("status") == "error"
    assert "unterminated" in sc.group_readout.text().lower()
    sc.to_state()   # must not raise
    assert sc.state.settings.output.group_regex == "[unclosed"


def test_the_row_is_hidden_when_nothing_is_matched_yet(qapp, tmp_path):
    """A fresh analysis with no input folder set: nothing on screen should claim '0
    files -> 0 groups' -- the row is hidden entirely (the SAME convention
    ``format_readout`` already uses), matching the pre-B01 "nothing scanned yet" state."""
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(synth_settings(str(tmp_path))))
    sc.in_folder.setText("")
    sc.in_files.setText("*.nope-such-extension")
    sc._on_inputs_changed()
    qapp.processEvents()
    assert sc.group_readout.text() == ""
    assert not sc._output_form.isRowVisible(sc.group_readout)


def test_dry_run_plan_repeats_the_same_grouping_line_setup_shows(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s))
    win.settings_screen.group_regex.setText(r"_([a-z]+)\.csv")
    win.settings_screen._on_field_changed()
    qapp.processEvents()
    setup_text = win.settings_screen.group_readout.text()
    assert setup_text and "(all)" in setup_text

    rn = win.run_screen
    rn._append_plan(write=False)
    log = rn.log.toPlainText()
    assert f"Grouping: {setup_text}" in log
    win.close()

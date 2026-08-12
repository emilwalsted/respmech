"""D19 (UI-overhaul): a live "Group files by" read-out on Setup, and the same line
repeated in the dry-run plan -- so a study's grouping pattern can be checked before a
run instead of only being visible afterwards by opening the written "By group" sheet.

The pure computation (``group_readout``, three states: normal/warn/error) has its own
Qt-free tests in ``test_manifest.py``; this file covers the WIRING -- that Setup's field
actually shows each state, that an invalid pattern does not stop settings from being
saved, and that Run's dry-run plan carries the same text.
"""
import os
import shutil

from PySide6.QtWidgets import QApplication

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings, write_delim

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


def test_setup_no_longer_gives_a_false_all_clear_for_a_column_count_outlier(qapp, tmp_path):
    """Self-review fix (10-08-2026): the read-out originally built its filename list from
    ``self._manifest.included_files`` (the UI's own majority-column-count subset). That
    gave a false all-clear in exactly the scenario D19 exists to catch:
    ``core.pipeline.run_batch`` has no column-count pre-filter at all and processes
    EVERY matched file, so an outlier a custom pattern fails to match still gets pooled
    into "(all)" in the actually-written "By group" sheet, while Setup showed a clean
    "info" line claiming otherwise. Reproduced end to end against a real Manifest with
    one genuine column-count outlier."""
    from respmech.ui.screens.settings_screen import SettingsScreen
    indir = tmp_path / "in"; indir.mkdir()
    shutil.copyfile(os.path.join(INPUT, "synth_case_A.csv"), indir / "P01_rest.csv")
    shutil.copyfile(os.path.join(INPUT, "synth_case_B.csv"), indir / "P02_rest.csv")
    write_delim(indir / "notes_extra.csv", ncols=13)   # majority (from the two copies) is 12
    sc = SettingsScreen(AppState(synth_settings(str(tmp_path))))
    sc.in_folder.setText(str(indir))
    sc.in_files.setText("*.csv")
    sc._on_inputs_changed()
    qapp.processEvents()
    assert len(sc._manifest.outliers) == 1   # confirms the fixture actually produced one

    sc.group_regex.setText(r"^(P\d+)_")   # matches the two P0N_rest.csv files, not notes_extra.csv
    sc._on_field_changed()
    qapp.processEvents()
    assert sc.group_readout.property("status") == "warn"
    assert "notes_extra.csv" in sc.group_readout.text()


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


def test_the_plan_omits_grouping_when_no_cohort_summary_will_be_written(qapp, tmp_path):
    """Self-review fix (10-08-2026): the "Grouping:" line used to appear unconditionally,
    even when the write it previews would never happen -- a subset re-run, or
    save_average turned off, both write NO "By group" sheet at all
    (core.io.plan.plan_outputs / write_batch gate it on ``save_average and
    cohort_outputs``), so the old line sat three lines above an output list that
    correctly listed no such file, directly contradicting _append_plan's own docstring
    ("this can never again drift from what a real run actually writes")."""
    from respmech.ui.main_window import MainWindow

    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); rn = win.run_screen
    rn._append_plan(write=False)
    assert any(ln.startswith("Grouping:") for ln in rn.log.toPlainText().splitlines())
    win.close()

    s2 = synth_settings(str(tmp_path))
    s2.output.data.save_average = False
    win2 = MainWindow(AppState(s2)); rn2 = win2.run_screen
    rn2._append_plan(write=False)
    assert not any(ln.startswith("Grouping:") for ln in rn2.log.toPlainText().splitlines())
    win2.close()

    s3 = synth_settings(str(tmp_path))
    win3 = MainWindow(AppState(s3)); rn3 = win3.run_screen
    rn3._only_files = ["synth_case_A.csv"]   # a subset re-run
    rn3._append_plan(write=False)
    assert not any(ln.startswith("Grouping:") for ln in rn3.log.toPlainText().splitlines())
    win3.close()

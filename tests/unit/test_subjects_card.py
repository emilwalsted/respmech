"""The 'Subjects && lung volumes' card on Setup: always reachable, and editable in place.

It used to be a read-only table that hid itself while ``input.subjects`` was empty, so a
fresh analysis, an opened file with ``subjects = []`` and a new analysis whose "Group files
by" had just been edited all had no card, and the only way to name a participant was to
edit the TOML by hand. The card is now an ordinary always-visible card with Add / Add from
files / Remove and cell editing, writing the model directly like the Preview-owned settings.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from respmech.core.settings import Settings, SubjectEntry
from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _screen(qapp, tmp_path, subjects=None, group_regex=None):
    from respmech.ui.screens.settings_screen import SettingsScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    if subjects is not None:
        s.input.subjects = subjects
    if group_regex is not None:
        s.output.group_regex = group_regex
    sc = SettingsScreen(AppState(s))
    sc.show()
    qapp.processEvents()
    return sc


def _edit(sc, row, col, text):
    """Type ``text`` into a cell the way the delegate would: set the item, let
    ``itemChanged`` fire."""
    sc.subjects_table.item(row, col).setText(text)


# -- visibility ---------------------------------------------------------------------------


def test_card_is_visible_with_no_subjects(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    assert sc.state.settings.input.subjects == []
    assert sc._card_subjects.isVisible()
    assert sc.subjects_table.rowCount() == 0


def test_card_is_visible_after_opening_a_toml_with_empty_subjects(qapp, tmp_path):
    from respmech.settingsio.toml_io import save_toml
    s = synth_settings(str(tmp_path / "out"), data_out=_OUT)
    s.input.subjects = []
    path = tmp_path / "a.toml"
    save_toml(s, path)
    sc = _screen(qapp, tmp_path)
    assert sc.open_analysis(str(path))
    qapp.processEvents()
    assert sc.state.settings.input.subjects == []
    assert sc._card_subjects.isVisible()


def test_card_is_visible_in_a_new_analysis_and_after_editing_the_group_pattern(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    sc.new_analysis_from_startup()
    qapp.processEvents()
    assert sc._card_subjects.isVisible()
    sc.group_regex.setText(r"^([A-Za-z]+)")
    sc._on_field_changed()
    qapp.processEvents()
    assert sc._card_subjects.isVisible()


def test_card_stays_visible_when_the_last_subject_is_removed(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth", tlc_l=6.0)])
    assert sc._card_subjects.isVisible()
    sc.subjects_table.selectRow(0)
    sc._remove_selected_subjects()
    qapp.processEvents()
    assert sc.state.settings.input.subjects == []
    assert sc._card_subjects.isVisible()


# -- adding and removing ------------------------------------------------------------------


def test_add_subject_proposes_a_group_key_from_the_files_and_stamps_the_folder(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    sc._add_subject()
    qapp.processEvents()
    subs = sc.state.settings.input.subjects
    assert [x.key for x in subs] == ["synth"]            # the files' own group key
    assert subs[0].folder == sc.state.settings.input.folder
    assert sc.subjects_table.rowCount() == 1
    assert sc.is_dirty()


def test_add_subject_never_duplicates_a_key(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    sc._add_subject()
    sc._add_subject()          # every file's group is already taken: a free placeholder key
    sc._add_subject()
    keys = [x.key for x in sc.state.settings.input.subjects]
    assert len(keys) == 3 and len(set(keys)) == 3
    assert keys[0] == "synth"
    assert sc.state.settings.validate()                  # unique keys: the model is valid


def test_add_from_files_adds_each_group_once(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, group_regex=r"synth_case_([AB])\.")
    sc._add_subjects_from_files()
    keys = sorted(x.key for x in sc.state.settings.input.subjects)
    assert keys == ["A", "B"]                            # "(all)" (no match) is not proposed
    sc._add_subjects_from_files()
    assert sorted(x.key for x in sc.state.settings.input.subjects) == ["A", "B"]
    assert not sc.btn_subjects_from_files.isEnabled()    # nothing left to add


def test_remove_selected_removes_exactly_those_rows(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="a"), SubjectEntry(key="b"),
                                           SubjectEntry(key="c")])
    sc.subjects_table.item(0, 0).setSelected(True)
    sc.subjects_table.item(2, 0).setSelected(True)
    sc._remove_selected_subjects()
    assert [x.key for x in sc.state.settings.input.subjects] == ["b"]
    assert sc.subjects_table.rowCount() == 1


# -- editing ------------------------------------------------------------------------------


def test_editing_a_number_writes_the_model_and_accepts_a_decimal_comma(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth")])
    _edit(sc, 0, 1, "6,5")
    _edit(sc, 0, 5, "150")
    sub = sc.state.settings.input.subjects[0]
    assert sub.tlc_l == 6.5 and sub.mvv_lpm == 150.0
    assert sc.subjects_table.item(0, 1).text() == "6.5"   # shown back in the card's own format
    assert sc.is_dirty()


def test_clearing_a_cell_makes_the_value_unset(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth", fev1_l=4.1)])
    _edit(sc, 0, 4, "")
    assert sc.state.settings.input.subjects[0].fev1_l is None
    assert sc.subjects_table.item(0, 4).text() == "—"


def test_a_bad_number_is_refused_and_the_old_value_is_restored(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth", tlc_l=6.0)])
    for bad in ("abc", "-1", "0", "nan", "inf"):
        _edit(sc, 0, 1, bad)
        assert sc.state.settings.input.subjects[0].tlc_l == 6.0, bad
        assert sc.subjects_table.item(0, 1).text() == "6", bad
        assert sc.subjects_message.text(), bad            # says why, in the card
    _edit(sc, 0, 1, "5")
    assert sc.state.settings.input.subjects[0].tlc_l == 5.0
    assert sc.subjects_message.text() == ""               # a good edit clears the message


def test_sex_is_normalised_and_an_unknown_value_is_refused(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth")])
    _edit(sc, 0, 6, "F")
    assert sc.state.settings.input.subjects[0].sex == "female"
    assert sc.subjects_table.item(0, 6).text() == "female"
    _edit(sc, 0, 6, "other")
    assert sc.state.settings.input.subjects[0].sex == "female"
    assert sc.subjects_message.text()
    _edit(sc, 0, 6, "")
    assert sc.state.settings.input.subjects[0].sex is None


def test_a_key_must_be_present_and_unique(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="a"), SubjectEntry(key="b")])
    _edit(sc, 1, 0, "a")
    assert [x.key for x in sc.state.settings.input.subjects] == ["a", "b"]
    assert sc.subjects_table.item(1, 0).text() == "b"
    _edit(sc, 1, 0, "   ")
    assert [x.key for x in sc.state.settings.input.subjects] == ["a", "b"]
    _edit(sc, 1, 0, "  c ")
    assert [x.key for x in sc.state.settings.input.subjects] == ["a", "c"]


def test_editing_never_restamps_the_folder_tag(qapp, tmp_path):
    """B06: the folder tag records where an entry was CREATED; a plain edit must not
    'confirm' a subject carried over from another recordings folder."""
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth", folder="/elsewhere")])
    _edit(sc, 0, 1, "6")
    assert sc.state.settings.input.subjects[0].folder == "/elsewhere"


def test_an_out_of_range_value_surfaces_through_the_normal_setup_validation(qapp, tmp_path):
    """The card refuses only what is not a positive number; ranges stay with
    Settings.validate(), whose message already names this card."""
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth")])
    _edit(sc, 0, 1, "99")
    assert sc.state.settings.input.subjects[0].tlc_l == 99.0
    assert "Subjects" in sc.qc.text()


def test_an_edit_survives_a_to_state_round_trip(qapp, tmp_path):
    """to_state() runs on every tab change; it must not revert a subjects edit."""
    sc = _screen(qapp, tmp_path, subjects=[SubjectEntry(key="synth")])
    _edit(sc, 0, 2, "5.1")
    sc.to_state()
    sc.from_state()
    assert sc.state.settings.input.subjects[0].vc_l == 5.1
    assert sc.subjects_table.item(0, 2).text() == "5.1"


def test_the_card_explains_where_fev1_comes_from(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    text = sc.subjects_hint.text()
    assert "FEV1" in text and "FVC" in text
    assert "reference" in text

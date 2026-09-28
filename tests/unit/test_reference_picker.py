"""ReferencePickerDialog (M-37): browsing/linking cross-file reference manoeuvres.

Settings-only, like the dialog itself (see its own module docstring) -- these tests
build a Settings object with ``BreathTypeEntry`` rows already typed, never rendering a
real recording, since the dialog only ever reasons about ``processing.breath_types``/
``processing.references`` and ``core.analysis.references.resolve_reference``.
"""
from respmech.core.settings import BreathTypeEntry
from respmech.ui.reference_picker_dialog import ReferencePickerDialog

from _helpers import _lone_ampersands, requires_synth, synth_settings

pytestmark = requires_synth()

_FILES = ["synth_case_A.csv", "synth_case_B.csv"]


def _settings_with_typed_ic(tmp_path, file="synth_case_B.csv", breath=3):
    s = synth_settings(str(tmp_path))
    s.processing.breath_types.append(
        BreathTypeEntry(file=file, breath=breath, kind="ic", t_onset_s=1.0))
    return s


def test_opening_shows_no_source_for_every_slot_when_nothing_is_typed(qapp, tmp_path):
    s = synth_settings(str(tmp_path))
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    assert all(v is None for v in dlg.staged().values())
    assert dlg.touched_slots() == set()
    dlg.close()


def test_touched_slots_is_empty_until_a_row_is_actually_edited(qapp, tmp_path):
    """Self-review finding: an untouched row must never be mistaken for an edited one
    -- opening the dialog on an already-resolved value (below) must not itself count
    as touching it."""
    s = _settings_with_typed_ic(tmp_path, file="synth_case_A.csv", breath=2)
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    assert dlg.touched_slots() == set(), "pre-selecting the resolved value touched it"
    row = dlg._rows["ic"]
    row.file_combo.setCurrentIndex(row.file_combo.findData("synth_case_B.csv"))
    assert dlg.touched_slots() == {"ic"}
    dlg.close()


def test_clicking_clear_counts_as_touching_the_row(qapp, tmp_path):
    s = _settings_with_typed_ic(tmp_path, file="synth_case_A.csv", breath=2)
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    dlg._rows["ic"].clear_btn.click()
    assert dlg.touched_slots() == {"ic"}
    dlg.close()


def test_picking_another_files_typed_ic_breath_stages_a_breathref(qapp, tmp_path):
    """The ticket's own acceptance criterion: choose an IC breath in a DIFFERENT file."""
    s = _settings_with_typed_ic(tmp_path)
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    row = dlg._rows["ic"]
    idx = row.file_combo.findData("synth_case_B.csv")
    assert idx >= 0
    row.file_combo.setCurrentIndex(idx)
    b_idx = row.breath_combo.findData(3)
    assert b_idx >= 0
    row.breath_combo.setCurrentIndex(b_idx)
    staged = dlg.staged()
    assert staged["ic"].file == "synth_case_B.csv"
    assert staged["ic"].breaths == [3]
    assert staged["fvc"] is None                  # untouched slots stay unresolved
    dlg.close()


def test_only_typed_ic_kinds_are_offered_for_the_ic_slot(qapp, tmp_path):
    """A breath typed 'fvc' (not ic/ic_fvc) must not be offered as an IC source, but IS
    offered for the fvc slot -- _SLOT_SOURCE_KINDS is per-slot, not a shared list."""
    s = synth_settings(str(tmp_path))
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_B.csv", breath=1, kind="fvc", t_onset_s=0.5))
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    ic_row = dlg._rows["ic"]
    ic_row.file_combo.setCurrentIndex(ic_row.file_combo.findData("synth_case_B.csv"))
    assert ic_row.breath_combo.isEnabled() is False
    fvc_row = dlg._rows["fvc"]
    fvc_row.file_combo.setCurrentIndex(fvc_row.file_combo.findData("synth_case_B.csv"))
    assert fvc_row.breath_combo.isEnabled() is True
    dlg.close()


def test_baseline_ic_offers_any_typed_breath_regardless_of_kind(qapp, tmp_path):
    """baseline_ic has no BREATH_KINDS member of its own (a baseline is by nature a
    different recording, typically quiet breathing) -- any typed breath qualifies."""
    s = synth_settings(str(tmp_path))
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_B.csv", breath=7, kind="rest", t_onset_s=0.2))
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    row = dlg._rows["baseline_ic"]
    row.file_combo.setCurrentIndex(row.file_combo.findData("synth_case_B.csv"))
    assert row.breath_combo.isEnabled() is True
    assert row.breath_combo.findData(7) >= 0
    dlg.close()


def test_pre_resolved_reference_is_shown_as_the_opening_value(qapp, tmp_path):
    """resolve_reference's OWN-typed-breath fallback is read on open, so the file's own
    already-typed IC breath appears pre-selected even with no explicit ReferenceEntry."""
    s = _settings_with_typed_ic(tmp_path, file="synth_case_A.csv", breath=2)
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    row = dlg._rows["ic"]
    assert row.file_combo.currentData() == "synth_case_A.csv"
    assert row.breath_combo.currentData() == 2
    dlg.close()


def test_clear_resets_the_slot_to_no_source(qapp, tmp_path):
    s = _settings_with_typed_ic(tmp_path, file="synth_case_A.csv", breath=2)
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    row = dlg._rows["ic"]
    row.clear_btn.click()
    assert row.file_combo.currentData() is None
    assert dlg.staged()["ic"] is None
    dlg.close()


def test_no_lone_ampersand(qapp, tmp_path):
    s = synth_settings(str(tmp_path))
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    offenders = _lone_ampersands(dlg)
    dlg.close()
    assert not offenders, offenders


def test_fits_under_windows_font_metrics(qapp, tmp_path, windows_metrics):
    """Acceptance criterion: '...pickeren...består windows_metrics-ratiotests'.

    An UNSHOWN QDialog's ``.width()``/``.height()`` are Qt's fixed pre-layout default,
    the same whether the font is native or windows_metrics-widened -- self-review
    found this the hard way (the first cut of this test asserted exactly that, and
    stayed unchanged even with deliberately absurd label text). ``minimumSizeHint()``,
    read AFTER a real show()+layout pass, is the number that actually tracks the font
    -- checked here against the tightest usable screen this app ships to, the same
    budget ``test_dialog_fits_screen.py`` uses for every other dialog."""
    _shortest_usable_h = min(864, 800, 720) - 70    # win 1080p@125%/mac 1280x800/win@150% minus chrome
    _narrowest_w = min(1536, 1280, 1280)
    s = synth_settings(str(tmp_path))
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    dlg.show()
    for _ in range(6):
        qapp.processEvents()
    got = dlg.minimumSizeHint()
    assert got.width() > 0 and got.height() > 0
    assert got.height() <= _shortest_usable_h, (
        f"minimum height {got.height()}px exceeds the {_shortest_usable_h}px usable on "
        "the shortest screen this app ships to")
    assert got.width() <= _narrowest_w, (
        f"minimum width {got.width()}px exceeds the {_narrowest_w}px narrowest screen")
    dlg.close()


def test_paints_in_dark_mode(qapp, tmp_path, dark_app):
    s = synth_settings(str(tmp_path))
    dlg = ReferencePickerDialog("synth_case_A.csv", s, _FILES)
    dlg.show()
    for _ in range(4):
        qapp.processEvents()
    assert dlg.isVisible()
    dlg.close()

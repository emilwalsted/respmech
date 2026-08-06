"""The visual channel-assignment feature: the raw all-columns loader, the
ChannelSetupDialog modal (role dropdowns, mutual exclusion, dark-mode plots), and its
integration into the Settings screen (apply a mapping, graceful no-file handling)."""
import glob
import os

import pytest




from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from respmech.core.settings import Settings  # noqa: E402
from respmech.ui.state import AppState  # noqa: E402


from _helpers import INPUT, requires_synth  # noqa: F401

pytestmark = requires_synth()




def _load():
    from respmech.ui.workers import load_raw_matrix
    f = sorted(glob.glob(os.path.join(INPUT, "synth_case_*.csv")))[0]
    return load_raw_matrix(Settings(), f)


def _matrix():
    return _load()[0]


def _set_role(dlg, col_index, role_key):
    from respmech.ui.channel_setup_dialog import _ROLES
    idx = next(i for i, (k, _l) in enumerate(_ROLES) if k == role_key)
    dlg._combos[col_index].setCurrentIndex(idx)


def _files():
    return sorted(glob.glob(os.path.join(INPUT, "synth_case_*.csv")))


def _loader():
    from respmech.ui.workers import load_raw_matrix
    return lambda p: load_raw_matrix(Settings(), p)


def _dialog(initial=None, files=None):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    return ChannelSetupDialog(files or _files(), 1000, initial=initial, loader=_loader())


# --------------------------------------------------------------------------- #
# raw loader
# --------------------------------------------------------------------------- #
def test_load_raw_matrix_reads_all_columns_and_names(qapp):
    m, names = _load()
    assert m.ndim == 2 and m.shape[1] == 12 and m.shape[0] > 1000
    assert names[4] == "flow" and names[6] == "poes" and "EMG1" in names


def test_load_raw_matrix_reads_txt_with_names(qapp, tmp_path):
    from respmech.ui.workers import load_raw_matrix
    p = tmp_path / "d.txt"
    p.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
    m, names = load_raw_matrix(Settings(), str(p))
    assert m.shape == (2, 3) and names == ["a", "b", "c"]


def test_load_raw_matrix_unsupported_type_raises(qapp, tmp_path):
    from respmech.ui.workers import load_raw_matrix
    p = tmp_path / "data.bin"
    p.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError):
        load_raw_matrix(Settings(), str(p))


# --------------------------------------------------------------------------- #
# dialog
# --------------------------------------------------------------------------- #
def test_dialog_has_a_control_and_plot_per_column(qapp):
    dlg = _dialog()
    assert len(dlg._combos) == 12 and len(dlg._plots) == 12
    assert dlg.selected_mapping() == {"flow": None, "volume": None, "poes": None,
                                      "pgas": None, "pdi": None, "emg": [], "entropy": []}


def test_dialog_lists_valid_files_and_defaults_to_first(qapp):
    files = _files()
    dlg = _dialog(files=files)
    assert dlg.file_combo.count() == len(files)
    assert dlg.file_combo.itemText(0) == os.path.basename(files[0])
    assert dlg.file_combo.currentIndex() == 0        # first valid file is the default


def test_dialog_switching_file_replots_but_keeps_assignments(qapp):
    dlg = _dialog()
    _set_role(dlg, 4, "flow"); _set_role(dlg, 6, "poes")
    _set_role(dlg, 7, "pgas"); _set_role(dlg, 8, "pdi"); _set_role(dlg, 1, "emg")
    before = dlg.selected_mapping()
    n_before = len(dlg._curves[1].getData()[1])      # EMG column length on file A
    dlg.file_combo.setCurrentIndex(1)                # switch to file B
    assert dlg._file_idx == 1
    assert len(dlg._curves[1].getData()[1]) != n_before   # B is a different length -> re-plotted
    assert dlg.selected_mapping() == before          # assignments persist across files


def test_dialog_first_column_is_time_without_a_dropdown(qapp):
    dlg = _dialog()
    assert dlg._combos[0] is None                    # time column: shown but not assignable
    assert dlg._role_of(0) == ""
    assert dlg._headers[0].text().endswith("time")
    _set_role(dlg, 4, "flow"); _set_role(dlg, 6, "poes")
    _set_role(dlg, 7, "pgas"); _set_role(dlg, 8, "pdi")
    assert dlg._ok_btn.isEnabled()                   # required gate ignores the time column
    m = dlg.selected_mapping()
    assert 1 not in (m["flow"], m["poes"], m["pgas"], m["pdi"])   # col 1 is never a role


def test_dialog_file_selector_is_not_full_width(qapp):
    dlg = _dialog()
    assert 0 < dlg.file_combo.maximumWidth() <= 500  # bounded, not stretched to the edge


def test_dialog_loads_files_lazily_and_caches(qapp):
    import numpy as np
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    data = (np.zeros((100, 4)), ["a", "b", "c", "d"])
    calls = []
    dlg = ChannelSetupDialog(["A", "B", "C"], 1000,
                             loader=lambda p: (calls.append(p), data)[1])
    assert calls == ["A"]                            # only the first file loaded up front
    dlg.file_combo.setCurrentIndex(1)
    assert calls == ["A", "B"]                       # B loaded on selection
    dlg.file_combo.setCurrentIndex(0)                # revisiting A -> cache hit, no new load
    assert calls == ["A", "B"]


def test_dialog_falls_forward_past_an_unreadable_first_file(qapp):
    import numpy as np
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    good = (np.zeros((100, 4)), ["a", "b", "c", "d"])

    def loader(p):
        if p == "A":
            raise ValueError("ragged row")           # passes the cheap probe, fails full read
        return good
    dlg = ChannelSetupDialog(["A", "B", "C"], 1000, loader=loader)
    assert dlg._file_idx == 1                        # A unreadable -> default to the next file
    assert dlg.file_combo.currentIndex() == 1 and dlg._ncols == 4


def test_dialog_switch_to_unreadable_file_reverts(qapp):
    import numpy as np
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    good = (np.zeros((100, 4)), ["a", "b", "c", "d"])

    def loader(p):
        if p == "B":
            raise ValueError("corrupt")
        return good
    dlg = ChannelSetupDialog(["A", "B"], 1000, loader=loader)
    dlg.file_combo.setCurrentIndex(1)                # B fails to load
    assert dlg._file_idx == 0                        # reverted to the last good file
    assert dlg.file_combo.currentIndex() == 0
    assert "could not read" in dlg.info.text().lower()


# --------------------------------------------------------------------------- #
# ticket D01: a diagnosis, not a bare sentence, when nothing could be read
# --------------------------------------------------------------------------- #
def test_dialog_diagnoses_a_field_count_mismatch_instead_of_a_bare_sentence(qapp):
    """An instrument export with a header block above the real channel data is unreadable
    end to end; the raised error must name the file and translate pandas' 'Expected N
    fields...saw M' into RespMech's own words, plus the header-block hint — not the
    previous bare 'None of the matching data files could be read.'"""
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog, NoReadableFileError

    def loader(_p):
        raise ValueError("Error tokenizing data. C error: Expected 3 fields in line 4, saw 9")
    with pytest.raises(NoReadableFileError) as excinfo:
        ChannelSetupDialog(["P05_60W.txt"], 1000, loader=loader)
    assert isinstance(excinfo.value, ValueError)     # still a plain ValueError to old callers
    msg = str(excinfo.value)
    assert "P05_60W.txt" in msg
    assert "row 4" in msg and "9 values" in msg and "the first row has 3" in msg
    assert "header block" in msg.lower()
    assert "Traceback" not in msg


def test_dialog_diagnosis_omits_the_header_hint_for_a_mismatch_deep_in_the_file(qapp):
    """Self-review finding: the header-block guidance ('delete the lines above the channel
    data') only makes sense for a mismatch NEAR THE TOP of the file. A ragged row 5000
    lines in (a truncated export, a quoted delimiter pandas miscounts) is a different
    problem, and telling the user to look at the top would send them the wrong way."""
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog

    def loader(_p):
        raise ValueError("Error tokenizing data. C error: Expected 6 fields in line 5000, saw 2")
    with pytest.raises(ValueError) as excinfo:
        ChannelSetupDialog(["deep.csv"], 1000, loader=loader)
    msg = str(excinfo.value)
    assert "row 5000" in msg and "2 values" in msg and "the first row has 6" in msg
    assert "header block" not in msg.lower()


def test_dialog_diagnosis_names_the_first_of_several_failed_files(qapp):
    """Every file's own exception is kept (not just discarded), but the raised message
    leads with the FIRST failure — the one a user would investigate first — while still
    saying how many files failed in total."""
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog

    def loader(_p):
        raise ValueError("boom")
    with pytest.raises(ValueError) as excinfo:
        ChannelSetupDialog(["A.txt", "B.txt", "C.txt"], 1000, loader=loader)
    msg = str(excinfo.value)
    assert msg.startswith("None of the 3 matching files could be read.")
    assert "A.txt" in msg and "boom" in msg
    assert "B.txt" not in msg and "C.txt" not in msg


def test_dialog_diagnosis_for_a_single_file_reads_naturally(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog

    def loader(_p):
        raise ValueError("boom")
    with pytest.raises(ValueError) as excinfo:
        ChannelSetupDialog(["A.txt"], 1000, loader=loader)
    assert str(excinfo.value) == "A.txt could not be read: boom"


def test_dialog_diagnosis_falls_back_to_the_raw_message_for_an_unrecognised_failure(qapp):
    """Not every read failure is a field-count mismatch — an unrelated exception must still
    surface, just without the header-block guess grafted onto it."""
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog

    def loader(_p):
        raise PermissionError("Permission denied")
    with pytest.raises(ValueError) as excinfo:
        ChannelSetupDialog(["locked.csv"], 1000, loader=loader)
    msg = str(excinfo.value)
    assert msg == "locked.csv could not be read: Permission denied"
    assert "header block" not in msg.lower()


def test_dialog_updates_column_names_on_file_switch(qapp):
    import numpy as np
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    a = (np.zeros((50, 3)), ["ax", "ay", "az"])
    b = (np.zeros((50, 3)), ["bx", "by", "bz"])
    dlg = ChannelSetupDialog(["A", "B"], 1000, loader=lambda p: a if p == "A" else b)
    assert dlg._headers[0].text().endswith("ax")
    dlg.file_combo.setCurrentIndex(1)
    assert dlg._headers[0].text().endswith("bx")     # header re-labels to the new file's names


def test_dialog_initial_mapping_single_role_wins_over_emg(qapp):
    dlg = _dialog(initial={"poes": 3, "emg": [3, 4]})   # column 3 claimed by both
    assert dlg._role_of(2) == "poes"                 # the single (pressure) role wins col 3
    assert dlg._role_of(3) == "emg"


def test_dialog_preselects_from_initial_mapping(qapp):
    dlg = _dialog(initial={"flow": 5, "poes": 7, "emg": [2, 3, 4]})
    assert dlg._role_of(4) == "flow" and dlg._role_of(6) == "poes"
    assert dlg._role_of(1) == "emg" and dlg._role_of(2) == "emg" and dlg._role_of(3) == "emg"


def test_dialog_selected_mapping_reflects_choices(qapp):
    dlg = _dialog()
    _set_role(dlg, 4, "flow"); _set_role(dlg, 5, "volume"); _set_role(dlg, 6, "poes")
    _set_role(dlg, 7, "pgas"); _set_role(dlg, 8, "pdi")
    _set_role(dlg, 1, "emg"); _set_role(dlg, 2, "emg"); _set_role(dlg, 3, "emg")
    dlg._entropy_boxes[9].setChecked(True); dlg._entropy_boxes[10].setChecked(True)
    assert dlg.selected_mapping() == {"flow": 5, "volume": 6, "poes": 7, "pgas": 8,
                                      "pdi": 9, "emg": [2, 3, 4], "entropy": [10, 11]}


def test_dialog_single_roles_are_mutually_exclusive(qapp):
    dlg = _dialog()
    _set_role(dlg, 4, "flow")
    assert dlg.selected_mapping()["flow"] == 5
    _set_role(dlg, 3, "flow")                 # picking flow again on col 4 clears col 5
    assert dlg.selected_mapping()["flow"] == 4
    assert dlg._role_of(4) == ""              # the previous flow column was reset to unused
    # EMG is NOT exclusive — several columns can be EMG
    _set_role(dlg, 1, "emg"); _set_role(dlg, 2, "emg")
    assert dlg.selected_mapping()["emg"] == [2, 3]


# -- a column may carry several roles: opening the dialog must never destroy them ----
# Regression: _roles_from_mapping seeds emg then entropy then the single roles, each
# overwriting the last, and selected_mapping reads only the combos. So merely opening the
# dialog and pressing OK deleted every role that lost. The worst case is the SHIPPED example
# config (legacy/example.py:56-57), which puts EMG and entropy on the same five columns:
# OK returned emg=[], silently switching the entire EMG analysis off.

def _shared(extra=None):
    m = {"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "entropy": [5]}
    m.update(extra or {})
    return _dialog(initial=m)


def test_emg_and_entropy_on_the_same_columns_both_survive_ok(qapp):
    """The shipped example config's shape (legacy/example.py:56-57). Before entropy had its
    own control this silently returned emg=[], switching the EMG analysis off."""
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9,
                           "emg": [2, 3, 4], "entropy": [2, 3, 4]})
    m = dlg.selected_mapping()
    assert m["emg"] == [2, 3, 4] and m["entropy"] == [2, 3, 4]
    # and it is VISIBLE as both, not merely remembered
    assert [dlg._role_of(i) for i in (1, 2, 3)] == ["emg"] * 3
    assert all(dlg._entropy_on(i) for i in (1, 2, 3))


def test_entropy_sharing_a_column_with_a_role_is_shown_on_both(qapp):
    dlg = _shared()
    assert dlg._role_of(4) == "flow"          # the dropdown still shows the exclusive role
    assert dlg._entropy_on(4)                 # and the box shows the non-exclusive one
    m = dlg.selected_mapping()
    assert m["flow"] == 5 and m["entropy"] == [5]


def test_ticking_entropy_is_independent_of_the_role_dropdown(qapp):
    """Structurally orthogonal: no conflict to resolve, no cross-column rule to get wrong."""
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9})
    dlg._entropy_boxes[6].setChecked(True)               # entropy on the Poes column
    assert dlg.selected_mapping() == {"flow": 5, "volume": None, "poes": 7, "pgas": 8,
                                      "pdi": 9, "emg": [], "entropy": [7]}
    _set_role(dlg, 6, "")                                # drop the role, keep the entropy
    m = dlg.selected_mapping()
    assert m["poes"] is None and m["entropy"] == [7]


def test_entropy_can_now_be_removed_which_the_kept_role_memory_could_not(qapp):
    """P1 preserved a hidden entropy assignment but gave no way to clear it. The box does."""
    dlg = _shared()
    dlg._entropy_boxes[4].setChecked(False)
    assert dlg.selected_mapping()["entropy"] == []


def test_entropy_does_not_follow_a_column_that_is_re_assigned(qapp):
    """It stays on the column the user ticked, whatever role that column later takes."""
    dlg = _shared()
    _set_role(dlg, 4, "pgas")
    m = dlg.selected_mapping()
    assert m["pgas"] == 5 and m["entropy"] == [5] and m["flow"] is None


def test_an_entropy_only_column_is_not_drawn_as_unused(qapp):
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "entropy": [11]})
    assert dlg._role_of(10) == "" and dlg._entropy_on(10)
    assert dlg._display_role(10) == "entropy"            # coloured as entropy, not muted grey
    assert dlg.selected_mapping()["entropy"] == [11]


def test_entropy_only_columns_count_as_assigned(qapp):
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "entropy": [11]})
    assert "5 columns assigned" in dlg.info.text()


def test_entropy_the_dialog_cannot_show_is_kept_and_flagged(qapp):
    """Column 20 is past this file's width and column 1 is the time axis: neither can carry
    a checkbox, but OK must not delete them either."""
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "entropy": [1, 20]})
    assert dlg.selected_mapping()["entropy"] == [1, 20]
    assert "#1" in dlg.info.text() and "#20" in dlg.info.text()
    assert "kept but not shown" in dlg.info.text()


def test_emg_hidden_behind_a_single_role_is_still_kept_and_announced(qapp):
    """EMG remains exclusive in the dropdown, so it still needs the memory that entropy no
    longer does — including releasing it when the user edits that column."""
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "emg": [5, 3]})
    assert dlg.selected_mapping()["emg"] == [3, 5]
    assert "EMG is also kept on column #5" in dlg.info.text()
    _set_role(dlg, 4, "pgas")                            # column 5 is something else now
    assert dlg.selected_mapping()["emg"] == [3]


def test_nothing_is_kept_and_nothing_is_said_when_no_roles_collide(qapp):
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "entropy": [11]})
    assert dlg._shadowed == {} and dlg._hidden_roles() == {}
    assert "kept" not in dlg.info.text().lower()


def test_two_single_roles_on_one_column_are_not_preserved(qapp):
    """Only emg/entropy are remembered. Two pressures on one column is an invalid mapping
    the QC strip blocks outright, not a state worth resurrecting."""
    dlg = _dialog(initial={"flow": 5, "poes": 5, "pgas": 8, "pdi": 9})
    m = dlg.selected_mapping()
    assert m["poes"] == 5 and m["flow"] is None       # last single role written wins
    assert dlg._shadowed == {}


# -- the column previews are read-only, and the wheel belongs to the list ----------
# Two separate mechanisms, worth keeping straight. setMouseEnabled(False, False) is what
# makes the preview read-only: pyqtgraph's ViewBox zooms on the wheel only while mouse
# interaction is on, and ignores it once off. The wheel guard then turns that "nothing
# happens" into "the column list scrolls". The dropdowns are the sharper case — measured
# before the guard, one wheel over a role dropdown set that column to Flow, and because
# Flow is mutually exclusive that moved the flow channel off the column it was on.
# Exhaustive per-screen coverage lives in test_wheel_guard.py.

def _wheel(qapp, target, notches=-3):
    from PySide6.QtCore import Qt, QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    qapp.sendEvent(target, QWheelEvent(
        QPointF(10, 10), target.mapToGlobal(QPoint(10, 10)), QPoint(0, 0),
        QPoint(0, notches * 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    qapp.processEvents()


def _shown(qapp):
    dlg = _dialog(initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9})
    dlg.resize(940, 660)
    dlg.show()
    qapp.processEvents()
    return dlg


def test_the_column_previews_are_not_interactive(qapp):
    dlg = _shown(qapp)
    for plot in dlg._plots:
        assert plot.getViewBox().state["mouseEnabled"] == [False, False]


def test_wheel_over_a_graph_scrolls_the_list_instead_of_zooming(qapp):
    dlg = _shown(qapp)
    vb = dlg._plots[3].getViewBox()
    before, bar = vb.viewRange()[1][:], dlg._scroll.verticalScrollBar()
    assert bar.maximum() > 0, "the column list must be scrollable for this to mean anything"
    _wheel(qapp, dlg._plots[3].viewport())
    assert bar.value() > 0, "the wheel did not scroll the column list"
    assert vb.viewRange()[1][:] == before, "the graph zoomed"


def test_wheel_over_a_role_dropdown_scrolls_instead_of_re_assigning(qapp):
    dlg = _shown(qapp)
    bar = dlg._scroll.verticalScrollBar()
    before = dlg.selected_mapping()
    at = bar.value()
    _wheel(qapp, dlg._combos[3])
    assert bar.value() > at, "the wheel did not scroll the column list"
    assert dlg.selected_mapping() == before, "the wheel re-assigned a channel"


def test_dialog_shows_source_column_names(qapp):
    dlg = _dialog()
    assert dlg._name_suffix(4).endswith("flow")      # column 5 == "flow"
    assert dlg._name_suffix(0).endswith("time")
    assert dlg._name_suffix(1).endswith("EMG1")


def test_dialog_ok_is_gated_on_the_required_roles(qapp):
    dlg = _dialog()
    assert not dlg._ok_btn.isEnabled()               # nothing assigned yet
    _set_role(dlg, 4, "flow"); _set_role(dlg, 6, "poes"); _set_role(dlg, 7, "pgas")
    assert not dlg._ok_btn.isEnabled()               # pdi still missing
    assert "Pdi" in dlg.info.text()
    _set_role(dlg, 8, "pdi")
    assert dlg._ok_btn.isEnabled()                   # all required present (volume optional)
    _set_role(dlg, 8, "")                            # remove pdi again -> re-locks OK
    assert not dlg._ok_btn.isEnabled()


def test_dialog_reassigning_a_single_role_recolors_both_plots(qapp):
    from respmech.ui.channel_setup_dialog import _role_color
    dlg = _dialog()
    _set_role(dlg, 4, "flow")
    _set_role(dlg, 3, "flow")                        # steal flow -> col 5 (idx 4) cleared
    pal = dlg._pal

    def pen_rgb(i):
        return tuple(dlg._curves[i].opts["pen"].color().getRgb()[:3])
    assert pen_rgb(3) == tuple(_role_color(pal, "flow")[:3])   # new flow column in flow colour
    assert pen_rgb(4) == tuple(_role_color(pal, "")[:3])       # cleared column back to grey


def test_dialog_uses_dark_plot_background_in_dark_mode(qapp, monkeypatch):
    from respmech.ui import theme
    monkeypatch.setattr(theme, "_IS_DARK", True)
    dlg = _dialog()
    name = dlg._plots[0].backgroundBrush().color().name().lower()
    assert name == theme._PLOT_DARK["bg"].lower()


# --------------------------------------------------------------------------- #
# Settings-screen integration (the real modal execs, so we stub it)
# --------------------------------------------------------------------------- #
class _StubDialog:
    _MAP = {"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9,
            "emg": [2, 3, 4], "entropy": [10, 11, 12]}

    def __init__(self, *a, accept=True, **k):
        self._accept = accept

    def exec(self):
        return QDialog.Accepted if self._accept else QDialog.Rejected

    def selected_mapping(self):
        return dict(self._MAP)


def _screen_pointed_at_input(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.in_folder.setText(INPUT)
    sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    return win, sc


def test_open_channel_setup_applies_mapping_on_ok(qapp, monkeypatch):
    import respmech.ui.channel_setup_dialog as csd
    monkeypatch.setattr(csd, "ChannelSetupDialog", lambda *a, **k: _StubDialog(accept=True))
    win, sc = _screen_pointed_at_input(qapp)
    assert sc._open_channel_setup(initial={}) is True
    ch = sc.state.settings.input.channels
    assert (ch.flow, ch.volume, ch.poes, ch.pgas, ch.pdi) == (5, 6, 7, 8, 9)
    assert ch.emg == [2, 3, 4] and ch.entropy == [10, 11, 12]
    # and it reached the readout the user actually sees
    assert any(t.startswith("Flow signal  ·  Column 5") for t in sc.channel_summary.texts())
    win.close()


def test_shared_entropy_column_survives_a_full_settings_round_trip(qapp):
    """End to end through the real dialog, not a stub: settings -> _current_channel_mapping
    -> dialog -> selected_mapping -> _apply_channel_mapping -> settings. The dialog is the
    only channel-assignment path, so a round trip through it must be lossless."""
    win, sc = _screen_pointed_at_input(qapp)
    ch = sc.state.settings.input.channels
    ch.flow, ch.poes, ch.pgas, ch.pdi, ch.entropy = 5, 7, 8, 9, [5, 11]
    dlg = _dialog(initial=sc._current_channel_mapping())
    sc._apply_channel_mapping(dlg.selected_mapping())
    out = sc.to_state().input.channels
    assert out.flow == 5
    assert out.entropy == [5, 11], "the entropy column shared with flow was lost"
    win.close()


def test_open_channel_setup_cancel_changes_nothing(qapp, monkeypatch):
    import respmech.ui.channel_setup_dialog as csd
    monkeypatch.setattr(csd, "ChannelSetupDialog", lambda *a, **k: _StubDialog(accept=False))
    win, sc = _screen_pointed_at_input(qapp)
    before = list(sc.state.settings.input.channels.emg)
    assert sc._open_channel_setup(initial={}) is False
    assert sc.state.settings.input.channels.emg == before
    win.close()


def test_open_channel_setup_shows_a_diagnosis_not_a_traceback(qapp, monkeypatch):
    """Ticket D01: when ChannelSetupDialog raises its diagnostic ``NoReadableFileError``
    (every matching file unreadable), the screen must lead with that diagnosis — no
    'Traceback' text and no path into RespMech's own source on the primary surface — with
    the full trace still one click away behind Details, not silently dropped.

    Uses ``isVisibleTo(dlg)``, not bare ``isVisible()``: the latter also asks whether the
    whole ancestor chain up to the screen is realised, and would give a false answer for an
    unshown top-level widget regardless of the child's own explicit visibility flag — the
    same class of gotcha CLAUDE.md documents for QShortcut. It happens to read correctly
    here only because ``open_error_dialog`` calls ``dlg.show()``; asserting the form that is
    correct regardless is the point."""
    import respmech.ui.channel_setup_dialog as csd
    diagnosis = ("None of the 2 matching files could be read. P05_60W.txt: row 4 has 9 "
                "values but the first row has 3. Instrument exports often start with a "
                "header block.")

    def _raise(*a, **k):
        raise csd.NoReadableFileError(diagnosis)
    monkeypatch.setattr(csd, "ChannelSetupDialog", _raise)
    win, sc = _screen_pointed_at_input(qapp)
    assert sc._open_channel_setup(initial={}) is False
    assert diagnosis in sc.status.text()
    dlg = sc._err_dialog
    assert dlg is not None
    assert dlg.intro_label is not None
    assert dlg.intro_label.text() == diagnosis
    assert "Traceback" not in dlg.intro_label.text()
    assert ".py" not in dlg.intro_label.text()            # no source path on the primary surface
    assert dlg.view.isVisibleTo(dlg) is False              # the trace starts collapsed
    assert dlg.details_btn is not None
    dlg.details_btn.setChecked(True)
    assert dlg.view.isVisibleTo(dlg) is True
    assert "NoReadableFileError" in dlg.text()             # still reachable, one click away
    win.close()


def test_open_channel_setup_shows_a_full_traceback_for_an_unrelated_valueerror(qapp, monkeypatch):
    """Self-review finding: an unrelated ``ValueError`` from elsewhere in
    ``ChannelSetupDialog.__init__`` (e.g. a numpy reshape error) is NOT the dialog's own
    diagnosed no-files-readable case, and must not be mistaken for one — it must still go
    through the ordinary, full-traceback error surface, not be presented as a trustworthy
    plain-language diagnosis. This is what distinguishes ``NoReadableFileError`` from
    catching bare ``ValueError``."""
    import respmech.ui.channel_setup_dialog as csd

    def _raise(*a, **k):
        raise ValueError("cannot reshape array of size 12 into shape (5,3)")
    monkeypatch.setattr(csd, "ChannelSetupDialog", _raise)
    win, sc = _screen_pointed_at_input(qapp)
    assert sc._open_channel_setup(initial={}) is False
    assert "channel setup failed" in sc.status.text().lower()
    dlg = sc._err_dialog
    assert dlg is not None
    assert dlg.details_btn is None                         # NOT the collapsed-diagnosis mode
    assert dlg.view.isVisibleTo(dlg) is True                # the trace is visible immediately
    assert "reshape" in dlg.text()
    win.close()


def test_open_channel_setup_without_a_file_is_graceful(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path))       # exists but has no matching data files
    sc.in_files.setText("*.csv")
    assert sc._open_channel_setup(initial={}) is False
    assert "no valid data" in sc.status.text().lower()
    win.close()


def test_valid_input_files_lists_only_loadable_consistent_files(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    # two real 4-col data files, one junk .txt that matches the mask, one odd-shape file
    (tmp_path / "rec_a.txt").write_text("f\tp\tg\td\n1\t2\t3\t4\n5\t6\t7\t8\n")
    (tmp_path / "rec_b.txt").write_text("f\tp\tg\td\n9\t8\t7\t6\n5\t4\t3\t2\n")
    (tmp_path / "readme.txt").write_text("this is not data at all\n")
    (tmp_path / "odd.txt").write_text("x\ty\n1\t2\n3\t4\n")     # only 2 cols -> minority, excluded
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.txt")
    sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    names = [os.path.basename(f) for f in sc._valid_input_files()]
    assert names == ["rec_a.txt", "rec_b.txt"]     # readme (1 col) + odd (minority) excluded
    win.close()


def test_probe_data_columns(qapp, tmp_path):
    from respmech.ui.workers import probe_data_columns
    p = tmp_path / "d.csv"
    p.write_text("a,b,c,d\n1,2,3,4\n5,6,7,8\n")
    assert probe_data_columns(Settings(), str(p)) == 4
    assert probe_data_columns(Settings(), str(tmp_path / "missing.csv")) is None


def test_matching_files_handles_multi_pattern_mask(qapp, tmp_path):
    from respmech.ui.validation import matching_files
    (tmp_path / "a.csv").write_text("x\n")
    (tmp_path / "b.txt").write_text("x\n")
    (tmp_path / "c.md").write_text("x\n")            # doesn't match either pattern
    got = sorted(os.path.basename(f) for f in matching_files(str(tmp_path), "*.csv; *.txt"))
    assert got == ["a.csv", "b.txt"]
    assert matching_files(str(tmp_path), "*.csv") == [str(tmp_path / "a.csv")]


def test_detect_decimal(qapp, tmp_path):
    from respmech.ui.workers import detect_decimal
    (tmp_path / "eu.txt").write_text("a\tb\n1,5\t2,7\n3,1\t4,9\n")
    (tmp_path / "us.txt").write_text("a\tb\n1.5\t2.7\n3.1\t4.9\n")
    assert detect_decimal(str(tmp_path / "eu.txt")) == ","
    assert detect_decimal(str(tmp_path / "us.txt")) == "."


def test_probe_narrows_multi_mask_and_detects_decimal(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    for name in ("r1.txt", "r2.txt"):                # tab-separated, comma-decimal, 5 cols
        (tmp_path / name).write_text("t\tf\tp\tg\td\n0\t1,0\t2,0\t3,0\t4,0\n0,1\t1,1\t2,1\t3,1\t4,1\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()                              # mask = *.csv; *.txt
    # Not exercising the guided channel-modal auto-open here (that is
    # test_startup_flow.py's job) — B04 hangs it on the input folder alone having
    # matching files, so filling in_folder below would otherwise schedule a REAL
    # (unstubbed) modal dialog via QTimer.singleShot(0, ...) that this test never pumps
    # the event loop for; disarm it up front instead of leaving a pending exec() for some
    # unrelated later test's processEvents() to stumble into.
    sc._channel_modal_done = True
    sc.in_folder.setText(str(tmp_path)); sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    assert sc.state.settings.input.files == "*.txt"          # multi-mask narrowed on input change
    note = sc._probe_and_apply_file_settings(sc._valid_input_files())
    assert sc.state.settings.input.format.decimal == ","     # comma decimal auto-detected
    assert "decimal" in note                                 # probe reports the detected format
    win.close()


def test_detect_decimal_ignores_thousands_separators(qapp, tmp_path):
    from respmech.ui.workers import detect_decimal
    (tmp_path / "us.txt").write_text("a\tb\n1,024\t2,048\n3,096\t4,000\n1,111\t2,222\n")
    (tmp_path / "eu.txt").write_text("a\tb\n1,5\t2,7\n3,1\t4,9\n12,34\t5,6\n")
    assert detect_decimal(str(tmp_path / "us.txt")) == "."   # comma-thousands, not comma-decimal
    assert detect_decimal(str(tmp_path / "eu.txt")) == ","   # genuine comma-decimal


def test_detect_sampling_frequency(qapp):
    import numpy as np
    from respmech.ui.workers import detect_sampling_frequency
    assert detect_sampling_frequency(np.arange(5000) / 1000.0) == 1000     # 1 kHz seconds
    assert detect_sampling_frequency(np.arange(5000, dtype=float)) is None  # a sample index
    assert detect_sampling_frequency(np.array([0.0, 0.1, 0.5, 0.51])) is None  # too few / irregular


def test_normalize_mask_narrows_multi_pattern_to_the_extension_present(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    (tmp_path / "a.csv").write_text("t,f\n0,1\n1,2\n")
    (tmp_path / "b.csv").write_text("t,f\n0,1\n1,2\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()                              # *.csv; *.txt
    sc._channel_modal_done = True   # disarm the guided auto-open modal (see the sibling test above)
    sc.in_folder.setText(str(tmp_path)); sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    assert sc.state.settings.input.files == "*.csv"  # narrowed so the single-glob core runner works
    win.close()


def test_all_ok_rejects_a_multi_pattern_mask(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    (tmp_path / "a.csv").write_text("t,f,p,g,d\n0,1,2,3,4\n1,1,2,3,4\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.samp_freq.setValue(1000)
    sc._apply_channel_mapping({"flow": 2, "volume": None, "poes": 3, "pgas": 4,
                               "pdi": 5, "emg": [], "entropy": []})
    sc._apply_channel_mapping({"flow": 5, "volume": None, "poes": 7, "pgas": 8,
                               "pdi": 9, "emg": [], "entropy": []})
    sc.state.settings.processing.volume.integrate_from_flow = True
    sc.out_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv"); sc._on_field_changed()
    assert sc._all_ok()                              # a single, valid mask is ready
    sc.state.settings.input.files = "*.csv, *.txt"   # a mask the core runner cannot glob
    assert not sc._all_ok()
    win.close()


def test_probe_detects_sampling_frequency_from_the_time_column(qapp, tmp_path):
    import numpy as np
    from respmech.ui.main_window import MainWindow
    n = 400; t = np.arange(n) / 500.0               # a real 500 Hz seconds time axis
    rows = "\n".join(f"{t[i]:.6f},1,2,3,4" for i in range(n))
    (tmp_path / "rec.csv").write_text("time,f,p,g,d\n" + rows + "\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv"); sc.samp_freq.setValue(2000)
    sc._on_inputs_changed()
    sc._probe_and_apply_file_settings(sc._valid_input_files())
    assert sc.samp_freq.value() == 500              # auto-detected, correcting the 2000 default
    win.close()


def test_probe_leaves_csv_decimal_untouched(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    (tmp_path / "r.csv").write_text("t,f,p,g,d\n0,1,2,3,4\n1,1,2,3,4\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()
    sc._channel_modal_done = True   # disarm the guided auto-open modal (see test_probe_narrows…)
    sc.in_folder.setText(str(tmp_path)); sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    assert sc.state.settings.input.files == "*.csv"
    before = sc.state.settings.input.format.decimal
    sc._probe_and_apply_file_settings(sc._valid_input_files())
    assert sc.state.settings.input.format.decimal == before   # decimal is a .txt-only concern
    win.close()


def test_probe_keeps_a_specific_single_mask(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    (tmp_path / "case_a.csv").write_text("t,f,p,g,d\n0,1,2,3,4\n1,1,2,3,4\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("case_*.csv")
    sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    sc._probe_and_apply_file_settings(sc._valid_input_files())
    assert sc.state.settings.input.files == "case_*.csv"     # a specific single mask is kept
    win.close()


def test_valid_input_files_tie_break_prefers_widest_and_missing_folder(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    def write(name, ncols):
        row = "\t".join(str(i) for i in range(ncols))
        (tmp_path / name).write_text(f"{row}\n{row}\n{row}\n")
    write("a3.txt", 3); write("b3.txt", 3); write("a5.txt", 5); write("b5.txt", 5)  # 2-2 tie
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.txt")
    sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    got = sorted(os.path.basename(f) for f in sc._valid_input_files())
    assert got == ["a5.txt", "b5.txt"]              # a tie in count -> the widest layout wins
    sc.in_folder.setText("/nonexistent-xyz-123"); sc._on_inputs_changed()
    assert sc._valid_input_files() == []            # missing folder -> no files
    win.close()


def test_apply_channel_mapping_writes_the_model_and_the_readout(qapp):
    """It is the ONLY writer of input.channels now — to_state deliberately leaves it alone,
    so anything this fails to set stays unset."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc._apply_channel_mapping({"flow": 3, "volume": None, "poes": 4, "pgas": 5,
                               "pdi": 6, "emg": [1, 2], "entropy": []})
    ch = sc.state.settings.input.channels
    assert (ch.flow, ch.poes, ch.pgas, ch.pdi) == (3, 4, 5, 6) and ch.emg == [1, 2]
    assert ch.volume is None                      # unassigned stays unassigned, not 0
    # no input folder here, so there is no file to plot: the summary falls back to the
    # plain list, which is the point — the mapping shows the moment it exists
    rows = sc.channel_summary.texts()
    assert "Flow signal: Column #3" in rows
    assert not any(r.startswith("Volume") for r in rows)
    assert "assigned" in sc.status.text().lower()
    win.close()


def test_manual_assign_channels_button_opens_setup_with_current_mapping(qapp, monkeypatch):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    assert sc.btn_assign_channels.text() == "Assign channels from data…"
    seen = {}
    monkeypatch.setattr(sc, "_open_channel_setup",
                        lambda initial=None: seen.setdefault("initial_is_none", initial is None))
    sc.btn_assign_channels.click()
    # opens with initial=None -> _open_channel_setup falls back to the CURRENT mapping,
    # so the picker pre-selects whatever channels are already set (not a blank fresh flow)
    assert seen.get("initial_is_none") is True
    win.close()


# ---------------------------------------------------------------------------
# Descriptive channel-role labels (P1) — from wave 1
# ---------------------------------------------------------------------------
def test_channel_role_labels_are_descriptive(qapp):
    from respmech.ui.channel_setup_dialog import _ROLES
    labels = dict(_ROLES)
    assert "oesophageal" in labels["poes"] and "gastric" in labels["pgas"]
    assert "transdiaphragmatic" in labels["pdi"] and "diaphragm" in labels["emg"]


# ---------------------------------------------------------------------------
# B01: the batch banner reconciles with the manifest's excluded files
# ---------------------------------------------------------------------------
def test_banner_names_excluded_files_when_given(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_files(), 1000, loader=_loader(),
                             excluded=[("odd_one.csv", 8), ("also_odd.csv", 11)])
    text = None
    for w in dlg.findChildren(type(dlg.info)):
        if "applied to all" in w.text():
            text = w.text()
            break
    assert text is not None
    assert "odd_one.csv" in text and "8" in text
    assert "also_odd.csv" in text
    assert "not shown here" in text
    dlg.close()


def test_banner_names_an_unreadable_excluded_file_without_literal_none(qapp):
    """Self-review regression (05-08-2026): an excluded file the manifest could not read at
    all carries columns=None (manifest.py's FileEntry), and the banner used to format that
    as the literal string 'brokenfile.csv (None cols)' — a bare f-string over None."""
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_files(), 1000, loader=_loader(),
                             excluded=[("brokenfile.csv", None), ("odd_one.csv", 8)])
    text = next(w.text() for w in dlg.findChildren(type(dlg.info)) if "applied to all" in w.text())
    assert "None" not in text
    assert "brokenfile.csv (unreadable)" in text
    assert "odd_one.csv (8 cols)" in text
    dlg.close()


def test_banner_says_nothing_extra_without_excluded_files(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_files(), 1000, loader=_loader())
    banners = [w.text() for w in dlg.findChildren(type(dlg.info)) if "applied to all" in w.text()]
    assert len(banners) == 1
    assert "not shown here" not in banners[0]
    dlg.close()


def test_settings_screen_passes_manifest_outliers_to_the_dialog(qapp, monkeypatch, tmp_path):
    """_open_channel_setup must hand the dialog exactly what its own manifest scan found —
    not silently drop the fact that a file was excluded."""
    from _helpers import write_delim
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b", "c"):
        write_delim(tmp_path / f"{n}.csv", 9)
    write_delim(tmp_path / "outlier.csv", 8)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    assert sc._manifest is not None and len(sc._manifest.outliers) == 1

    seen = {}

    def fake_dialog(files, fs, initial, loader=None, parent=None, excluded=None):
        seen["excluded"] = excluded
        raise ValueError("stop before actually opening a modal")

    monkeypatch.setattr("respmech.ui.channel_setup_dialog.ChannelSetupDialog", fake_dialog)
    sc._open_channel_setup(initial={})
    assert seen["excluded"] == [("outlier.csv", 8)]
    win.close()

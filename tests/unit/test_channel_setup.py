"""The visual channel-assignment feature: the raw all-columns loader, the
ChannelSetupDialog modal (role dropdowns, mutual exclusion, dark-mode plots), and its
integration into the Settings screen (apply a mapping, graceful no-file handling)."""
import glob
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from respmech.core.settings import Settings  # noqa: E402
from respmech.ui.state import AppState  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
    reason="synthetic input not present")


@pytest.fixture(scope="module")
def qapp():
    from respmech.ui import theme
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    yield app


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
    _set_role(dlg, 9, "entropy"); _set_role(dlg, 10, "entropy")
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
    assert sc.col_flow.value() == 5 and sc.col_volume.value() == 6
    assert sc.col_poes.value() == 7 and sc.col_pgas.value() == 8 and sc.col_pdi.value() == 9
    assert sc.cols_emg.text().replace(" ", "") == "2,3,4"
    assert sc.cols_entropy.text().replace(" ", "") == "10,11,12"
    win.close()


def test_open_channel_setup_cancel_changes_nothing(qapp, monkeypatch):
    import respmech.ui.channel_setup_dialog as csd
    monkeypatch.setattr(csd, "ChannelSetupDialog", lambda *a, **k: _StubDialog(accept=False))
    win, sc = _screen_pointed_at_input(qapp)
    before = sc.cols_emg.text()
    assert sc._open_channel_setup(initial={}) is False
    assert sc.cols_emg.text() == before
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


def test_apply_channel_mapping_writes_widgets(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc._apply_channel_mapping({"flow": 3, "volume": None, "poes": 4, "pgas": 5,
                               "pdi": 6, "emg": [1, 2], "entropy": []})
    assert sc.col_flow.value() == 3 and sc.col_poes.value() == 4
    assert sc.col_volume.value() == 0             # unassigned volume -> 0
    assert sc.cols_emg.text().replace(" ", "") == "1,2"
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

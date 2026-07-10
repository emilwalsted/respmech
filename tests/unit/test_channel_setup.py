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
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_matrix(), 1000, file_name="synth_case_A.csv")
    assert len(dlg._combos) == 12 and len(dlg._plots) == 12
    assert dlg.selected_mapping() == {"flow": None, "volume": None, "poes": None,
                                      "pgas": None, "pdi": None, "emg": [], "entropy": []}


def test_dialog_preselects_from_initial_mapping(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_matrix(), 1000,
                             initial={"flow": 5, "poes": 7, "emg": [2, 3, 4]})
    assert dlg._role_of(4) == "flow" and dlg._role_of(6) == "poes"
    assert dlg._role_of(1) == "emg" and dlg._role_of(2) == "emg" and dlg._role_of(3) == "emg"


def test_dialog_selected_mapping_reflects_choices(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_matrix(), 1000)
    _set_role(dlg, 4, "flow"); _set_role(dlg, 5, "volume"); _set_role(dlg, 6, "poes")
    _set_role(dlg, 7, "pgas"); _set_role(dlg, 8, "pdi")
    _set_role(dlg, 1, "emg"); _set_role(dlg, 2, "emg"); _set_role(dlg, 3, "emg")
    _set_role(dlg, 9, "entropy"); _set_role(dlg, 10, "entropy")
    assert dlg.selected_mapping() == {"flow": 5, "volume": 6, "poes": 7, "pgas": 8,
                                      "pdi": 9, "emg": [2, 3, 4], "entropy": [10, 11]}


def test_dialog_single_roles_are_mutually_exclusive(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_matrix(), 1000)
    _set_role(dlg, 4, "flow")
    assert dlg.selected_mapping()["flow"] == 5
    _set_role(dlg, 3, "flow")                 # picking flow again on col 4 clears col 5
    assert dlg.selected_mapping()["flow"] == 4
    assert dlg._role_of(4) == ""              # the previous flow column was reset to unused
    # EMG is NOT exclusive — several columns can be EMG
    _set_role(dlg, 1, "emg"); _set_role(dlg, 2, "emg")
    assert dlg.selected_mapping()["emg"] == [2, 3]


def test_dialog_shows_source_column_names(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    m, names = _load()
    dlg = ChannelSetupDialog(m, 1000, col_names=names)
    assert dlg._name_suffix(4).endswith("flow")      # column 5 == "flow"
    assert dlg._name_suffix(0).endswith("time")
    assert dlg._name_suffix(1).endswith("EMG1")


def test_dialog_ok_is_gated_on_the_required_roles(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    dlg = ChannelSetupDialog(_matrix(), 1000)
    assert not dlg._ok_btn.isEnabled()               # nothing assigned yet
    _set_role(dlg, 4, "flow"); _set_role(dlg, 6, "poes"); _set_role(dlg, 7, "pgas")
    assert not dlg._ok_btn.isEnabled()               # pdi still missing
    assert "Pdi" in dlg.info.text()
    _set_role(dlg, 8, "pdi")
    assert dlg._ok_btn.isEnabled()                   # all required present (volume optional)
    _set_role(dlg, 8, "")                            # remove pdi again -> re-locks OK
    assert not dlg._ok_btn.isEnabled()


def test_dialog_reassigning_a_single_role_recolors_both_plots(qapp):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog, _role_color
    dlg = ChannelSetupDialog(_matrix(), 1000)
    _set_role(dlg, 4, "flow")
    _set_role(dlg, 3, "flow")                        # steal flow -> col 5 (idx 4) cleared
    pal = dlg._pal

    def pen_rgb(i):
        return tuple(dlg._plots[i].listDataItems()[0].opts["pen"].color().getRgb()[:3])
    assert pen_rgb(3) == tuple(_role_color(pal, "flow")[:3])   # new flow column in flow colour
    assert pen_rgb(4) == tuple(_role_color(pal, "")[:3])       # cleared column back to grey


def test_dialog_uses_dark_plot_background_in_dark_mode(qapp, monkeypatch):
    from respmech.ui import theme
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    monkeypatch.setattr(theme, "_IS_DARK", True)
    dlg = ChannelSetupDialog(_matrix(), 1000)
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
    assert "no data file" in sc.status.text().lower()
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

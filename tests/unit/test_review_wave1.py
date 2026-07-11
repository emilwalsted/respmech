"""Wave 1 of the full-app review: correctness & trust.

P1 the channel-mapping hard gate (a required channel on the time column, or two required
channels sharing a column, must block Preview/Run); P2 excluding a breath live-recomputes the
averaged result; P6 the window title names the analysis + a real dirty flag; P9 a live QC
caution strip; plus the descriptive channel-role labels + the batch banner.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.ui.state import AppState  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")), reason="synthetic input absent")


@pytest.fixture(scope="module")
def qapp():
    from respmech.ui import theme
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    yield app


def _valid(sc, tmp):
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp))
    sc.col_flow.setValue(5); sc.col_volume.setValue(6); sc.col_poes.setValue(7)
    sc.col_pgas.setValue(8); sc.col_pdi.setValue(9); sc.cols_emg.setText("2,3,4")
    sc._on_field_changed()


# --------------------------------------------------------------------------- #
# P1 — hard channel-mapping gate (the cancel trap)
# --------------------------------------------------------------------------- #
def test_required_channel_on_the_time_column_blocks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    assert sc._all_ok()                                  # a good mapping is ready
    sc.col_poes.setValue(1); sc._on_field_changed()      # poes -> column 1 (the time axis)
    assert not sc._all_ok()
    assert "column 1" in sc._channel_collision().lower()
    win.close()


def test_two_required_channels_sharing_a_column_blocks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    sc.col_pgas.setValue(7); sc._on_field_changed()      # pgas == poes column
    assert not sc._all_ok()
    assert "same column" in sc._channel_collision().lower()
    win.close()


def test_cancelled_channel_modal_leaves_flow_gated(qapp, tmp_path, monkeypatch):
    """The cancel trap: cancelling the auto-opened modal must NOT unlock Preview/Run with
    every pressure defaulting to the time column."""
    import respmech.ui.channel_setup_dialog as csd
    from PySide6.QtWidgets import QDialog
    from respmech.ui.main_window import MainWindow

    class _Cancel:
        def __init__(self, *a, **k): pass
        def exec(self): return QDialog.Rejected
        def selected_mapping(self): return {}
    monkeypatch.setattr(csd, "ChannelSetupDialog", lambda *a, **k: _Cancel())
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    sc.out_folder.setText(str(tmp_path)); sc._on_field_changed()
    qapp.processEvents()                                 # auto-open -> cancelled
    assert not sc._all_ok()                              # channels default to col 1 -> gated
    assert not win.tabs.isTabVisible(win._i_preview)
    win.close()


# --------------------------------------------------------------------------- #
# P6 — dirty flag + window title
# --------------------------------------------------------------------------- #
def test_dirty_flag_and_window_title(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    assert not sc.is_dirty()
    assert "new analysis" in win.windowTitle().lower()
    sc.samp_freq.setValue(1234); sc._on_field_changed()  # a real edit
    assert sc.is_dirty()
    assert win.windowTitle().rstrip().endswith("•")      # dirty bullet
    sc._mark_clean()
    assert not sc.is_dirty() and not win.windowTitle().rstrip().endswith("•")
    win.close()


def test_new_analysis_only_confirms_when_dirty(qapp, monkeypatch):
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    calls = []
    monkeypatch.setattr(ss.QMessageBox, "question",
                        staticmethod(lambda *a, **k: calls.append(1) or ss.QMessageBox.Yes))
    sc._new_analysis()                                   # not dirty -> no confirmation
    assert calls == []
    sc.samp_freq.setValue(999); sc._on_field_changed()   # now dirty
    sc._new_analysis()
    assert calls == [1]                                  # confirmed exactly once
    win.close()


# --------------------------------------------------------------------------- #
# P9 — live QC strip
# --------------------------------------------------------------------------- #
def test_qc_strip_flags_cautions_and_clears(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    assert sc.qc.property("status") == "ok" and "no warnings" in sc.qc.text().lower()
    sc.cols_emg.setText("5,2,3"); sc._on_field_changed()  # EMG col 5 == the flow column
    assert sc.qc.property("status") == "warn"
    assert "overlap" in sc.qc.text().lower()
    win.close()


# --------------------------------------------------------------------------- #
# P2 — live recompute on breath exclusion
# --------------------------------------------------------------------------- #
def test_toggle_breath_requests_recompute(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    from respmech.settingsio.migrate import migrate_dict
    legacy = {"input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4],
                                 "columns_entropy": [10, 11, 12]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": False, "remove_noise": False}},
              "output": {"outputfolder": str(tmp_path), "data": {}}}
    s, _ = migrate_dict(legacy)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    assert pv._batch_recompute_pending is True            # the average is being recomputed…
    assert "recomputing" in pv.status.text().lower()      # …not deferred to a manual run
    win.close()


# --------------------------------------------------------------------------- #
# channel-role labels + batch banner
# --------------------------------------------------------------------------- #
def test_channel_role_labels_are_descriptive(qapp):
    from respmech.ui.channel_setup_dialog import _ROLES
    labels = dict(_ROLES)
    assert "oesophageal" in labels["poes"] and "gastric" in labels["pgas"]
    assert "transdiaphragmatic" in labels["pdi"] and "diaphragm" in labels["emg"]

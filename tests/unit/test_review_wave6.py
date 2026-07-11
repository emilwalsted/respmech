"""Wave 6 of the full-app review: onboarding & efficiency.

P23 a bundled-sample-data door (generated on demand) + Cancel; P24 the Preview/Run
tabs are LOCKED (not hidden) so the whole workflow is visible; P25 "New from last rig"
inherits the last channel mapping; P26 recent analyses + sticky folders; P27 keyboard
file-stepping; P28 a detected-format read-out. (P29's promoted channel picker + the
co-located EMG-noise controls were delivered by earlier work.)
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.ui.state import AppState  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")


@pytest.fixture(scope="module")
def qapp():
    from respmech.ui import theme
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    yield app


@pytest.fixture
def isolated_prefs(monkeypatch):
    """Point prefs at a throwaway QSettings scope so tests never touch real user prefs."""
    from respmech.ui import prefs
    monkeypatch.setattr(prefs, "ORG", "RespMechTest")
    monkeypatch.setattr(prefs, "APP", "RespMechTestSuite")
    QSettings("RespMechTest", "RespMechTestSuite").clear()
    yield prefs
    QSettings("RespMechTest", "RespMechTestSuite").clear()


# --------------------------------------------------------------------------- #
# P25 / P26 — prefs store
# --------------------------------------------------------------------------- #
def test_prefs_sticky_folder_and_rig(qapp, isolated_prefs, tmp_path):
    prefs = isolated_prefs
    prefs.set_last_folder("analysis", str(tmp_path))
    assert prefs.last_folder("analysis") == str(tmp_path)
    assert prefs.last_folder("missing", "fallback") == "fallback"

    from respmech.core.settings import Settings
    s = Settings()
    s.input.channels.poes = 7; s.input.channels.emg = [2, 3, 4]
    s.input.format.sampling_frequency = 500
    prefs.save_rig(s)
    rig = prefs.last_rig()
    assert rig["poes"] == 7 and rig["emg"] == [2, 3, 4] and rig["sampling_frequency"] == 500
    s2 = Settings()
    prefs.apply_rig(s2, rig)
    assert s2.input.channels.poes == 7 and s2.input.channels.emg == [2, 3, 4]
    assert s2.input.format.sampling_frequency == 500


def test_prefs_recent_analyses(qapp, isolated_prefs, tmp_path):
    prefs = isolated_prefs
    a = tmp_path / "a.toml"; a.write_text("x")
    b = tmp_path / "b.toml"; b.write_text("y")
    prefs.add_recent_analysis(str(a)); prefs.add_recent_analysis(str(b))
    prefs.add_recent_analysis(str(a))                    # re-adding moves it to the front
    assert prefs.recent_analyses()[0] == str(a)
    assert set(prefs.recent_analyses()) == {str(a), str(b)}
    a.unlink()
    assert str(a) not in prefs.recent_analyses()         # vanished files are dropped


# --------------------------------------------------------------------------- #
# P23 — sample data
# --------------------------------------------------------------------------- #
def test_sample_recording_is_analysable(tmp_path):
    from respmech.core.sample import write_sample_recording
    from respmech.core.pipeline import run_batch
    from respmech.core.settings import Settings
    desc = write_sample_recording(str(tmp_path / "input"))
    assert os.path.isfile(desc["path"])
    s = Settings()
    s.input.folder = desc["folder"]; s.input.files = desc["filename"]
    s.input.format.sampling_frequency = desc["sampling_frequency"]
    m, ch = desc["mapping"], s.input.channels
    ch.flow, ch.volume, ch.poes, ch.pgas, ch.pdi = m["flow"], m["volume"], m["poes"], m["pgas"], m["pdi"]
    ch.emg = m["emg"]; s.processing.segmentation.buffer = 200
    result = run_batch(s)
    fr = next(iter(result.ok_files.values()))
    assert len([b for b in fr.breaths.values() if not b["ignored"]]) >= 4   # real breaths detected


def test_startup_sample_and_cancel(qapp, isolated_prefs):
    from respmech.ui.startup_dialog import StartupDialog
    dlg = StartupDialog()
    assert hasattr(dlg, "sample_btn")                    # the explore door (P23)
    dlg._choose_sample()
    assert dlg.mode == "sample"
    dlg2 = StartupDialog()
    dlg2.reject()                                        # Cancel leaves the safe default
    assert dlg2.mode == "new"


# --------------------------------------------------------------------------- #
# P24 — tabs are locked, not hidden
# --------------------------------------------------------------------------- #
def test_downstream_tabs_locked_not_hidden(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()
    assert win.tabs.isTabVisible(win._i_preview)          # still VISIBLE (a stepper)…
    assert not win.tabs.isTabEnabled(win._i_preview)      # …but LOCKED until valid
    assert win.tabs.tabToolTip(win._i_preview)            # explains why
    assert sc.open_sample_analysis()
    assert win.tabs.isTabEnabled(win._i_preview) and win.tabs.isTabEnabled(win._i_run)
    win.close()


# --------------------------------------------------------------------------- #
# P25 — new from last rig
# --------------------------------------------------------------------------- #
def test_new_from_last_rig(qapp, isolated_prefs):
    from respmech.core.settings import Settings
    from respmech.ui.main_window import MainWindow
    from respmech.ui.startup_dialog import StartupDialog
    s = Settings(); s.input.channels.poes = 7; s.input.channels.pdi = 9
    s.input.channels.emg = [2, 3, 4]; s.input.format.sampling_frequency = 800
    isolated_prefs.save_rig(s)
    dlg = StartupDialog()
    assert hasattr(dlg, "rig_btn")                        # offered because a rig exists
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode(use_last_rig=True)
    assert sc.col_poes.value() == 7 and sc.cols_emg.text() == "2,3,4"
    assert sc.samp_freq.value() == 800
    assert not win.tabs.isTabEnabled(win._i_preview)      # still guided (folders blank)
    win.close()


# --------------------------------------------------------------------------- #
# P27 — keyboard file-stepping
# --------------------------------------------------------------------------- #
def test_file_stepping(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); pv = win.preview_screen
    pv.file_combo.clear(); pv.file_combo.addItems(["f1.csv", "f2.csv", "f3.csv"])
    pv.file_combo.setCurrentIndex(0)
    pv._step_file(+1); assert pv.file_combo.currentText() == "f2.csv"
    pv._step_file(+1); pv._step_file(+1)                 # clamps at the end
    assert pv.file_combo.currentIndex() == 2
    pv._step_file(-1); assert pv.file_combo.currentText() == "f2.csv"
    win.close()


# --------------------------------------------------------------------------- #
# P28 — detected-format read-out
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input absent")
def test_format_readout(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv"); sc._on_inputs_changed()
    assert "columns" in sc.format_readout.text() and "comma" in sc.format_readout.text()
    assert sc.format_readout.property("status") == "info"
    sc.in_files.setText("*.nope"); sc._on_inputs_changed()
    assert sc.format_readout.property("status") == "warn"
    win.close()

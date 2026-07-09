"""Headless (offscreen) tests for the PySide6 GUI logic.

These verify construction and the settings/preview/run data flow without a display
(QT_QPA_PLATFORM=offscreen). Interactive look-and-feel still needs a real display,
but the wiring — settings round-trip, preview/test-run, and the worker running the
core and producing output — is locked here.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.settingsio.migrate import migrate_dict  # noqa: E402
from respmech.ui.state import AppState  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
    reason="synthetic input not present")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _settings(outdir):
    legacy = {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": 1000},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": [10, 11, 12]}},
        "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                     "avgresamplingobs": 300},
                       "emg": {"remove_ecg": False, "remove_noise": False}},
        "output": {"outputfolder": outdir,
                   "data": {"saveaveragedata": True, "savebreathbybreathdata": True}},
    }
    s, _ = migrate_dict(legacy)
    return s


def test_mainwindow_constructs(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    assert win.tabs.count() == 3
    # settings form round-trips into shared state
    win.settings_screen.from_state()
    win.settings_screen.to_state()
    assert win.state.settings.input.folder == INPUT


def test_preview_and_test_run(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    assert pv.file_combo.count() == 2
    pv.file_combo.setCurrentIndex(0)
    pv._preview()
    assert "breaths" in pv.status.text()
    pv._test_run()
    assert pv.table.rowCount() > 0 and pv.table.columnCount() > 0
    assert "Test run OK" in pv.status.text()


def test_worker_runs_and_writes(qapp, tmp_path):
    from respmech.ui.workers import BatchWorker
    settings = _settings(str(tmp_path))
    results = {}
    w = BatchWorker(settings, write=True)
    w.finished.connect(lambda r: results.setdefault("r", r))
    w.run()  # synchronous — verifies the worker logic without a thread
    r = results["r"]
    assert r is not None
    assert set(r.ok_files) == {"synth_case_A.csv", "synth_case_B.csv"}
    assert os.path.exists(os.path.join(tmp_path, "data", "Average breathdata.xlsx"))


def test_preview_noise_fidelity_render(qapp, tmp_path):
    """The preview/tuning screen renders the per-test noise fidelity frontier."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = _settings(str(tmp_path))
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.use_expiration = False
    s.processing.emg.noise.reference_intervals = [[1.0, 5.0]]
    s.processing.emg.noise.auto_prop = True
    win = MainWindow(AppState(s))
    result = run_batch(s, only_files=["synth_case_B.csv"])
    assert result.noise_report is not None
    win.preview_screen.render_noise_report(result)
    assert "prop_decrease" in win.preview_screen.status.text()
    # the fidelity frontier figure has been drawn
    assert len(win.preview_screen.fidelity_canvas.figure.axes) == 1


def test_splash_resolves_fonts_to_installed_families(qapp):
    """The splash used CSS-style font stacks as SVG font-family, which Qt cannot
    resolve (the 'Populating font family aliases … missing font family' warning).
    make_splash must resolve them to a single installed family."""
    from PySide6.QtGui import QFontDatabase
    from respmech.ui import splash
    splash.make_splash(qapp)
    installed = set(QFontDatabase.families())
    assert splash._MONO in installed and splash._FONT in installed
    assert "," not in splash._MONO and "," not in splash._FONT   # single families, not stacks


def test_theme_applies_light_and_dark(qapp, monkeypatch):
    from respmech.ui import theme
    assert theme.apply_theme(qapp) in ("light", "dark")
    monkeypatch.setenv("RESPMECH_THEME", "dark")
    assert theme.apply_theme(qapp) == "dark"
    assert len(qapp.styleSheet()) > 1000


def test_reactive_file_list_and_noise_gating(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    sc, pv = win.settings_screen, win.preview_screen
    assert pv.file_combo.count() == 2
    # changing the mask in Settings refreshes the Preview file list (reactive)
    sc.in_files.setText("synth_case_A.csv"); sc._on_inputs_changed()
    assert pv.file_combo.count() == 1
    sc.in_files.setText("synth_case_*.csv"); sc._on_inputs_changed()
    assert pv.file_combo.count() == 2
    # noise on but no reference -> the noise-window options are disabled + a hint
    sc.remove_noise.setChecked(True); sc.noise_ref.setText(""); sc._on_field_changed()
    pv.file_combo.setCurrentIndex(0); pv._update_actions()
    assert pv.noise_opts.isEnabled() is False
    assert "reference" in pv.status.text().lower()
    # setting a reference enables them
    sc.noise_ref.setText("synth_case_A.csv"); sc._on_field_changed(); pv._update_actions()
    assert pv.noise_opts.isEnabled() is True


def test_remove_noise_checkbox_binds_noise_enabled(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(_settings(str(tmp_path))))
    sc.remove_noise.setChecked(True)
    sc.to_state()
    # the checkbox drives the real gate (noise.enabled) and the legacy mirror
    assert sc.state.settings.processing.emg.noise.enabled is True
    assert sc.state.settings.processing.emg.remove_noise is True


def test_empty_input_folder_is_handled(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    s.input.folder = str(tmp_path / "does_not_exist")
    win = MainWindow(AppState(s))
    win.preview_screen.refresh_files()
    assert win.preview_screen.file_combo.count() == 0
    assert "not found" in win.preview_screen.status.text().lower()


def test_worker_cancellation(qapp, tmp_path):
    from respmech.ui.workers import BatchWorker
    w = BatchWorker(_settings(str(tmp_path)), write=False)
    w.cancel()  # cancel before running -> no files processed
    out = {}
    w.finished.connect(lambda r: out.setdefault("r", r))
    w.run()
    assert out["r"] is None  # cancelled -> None


def test_validate_checks_paths(qapp, tmp_path):
    """Validate reports filesystem problems the core (path-agnostic) validate misses."""
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(_settings(str(tmp_path))))
    # a good configuration passes the path checks
    assert sc._path_problem() is None
    # a non-existent input folder is caught
    sc.state.settings.input.folder = str(tmp_path / "nope")
    msg = sc._path_problem()
    assert msg and "input folder" in msg
    # restore folder; a missing noise reference file is caught when noise is on
    sc.state.settings.input.folder = INPUT
    sc.state.settings.processing.emg.noise.enabled = True
    sc.state.settings.processing.emg.noise.reference_file = "not_here.csv"
    msg = sc._path_problem()
    assert msg and "noise reference file" in msg

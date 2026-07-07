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


def test_worker_cancellation(qapp, tmp_path):
    from respmech.ui.workers import BatchWorker
    w = BatchWorker(_settings(str(tmp_path)), write=False)
    w.cancel()  # cancel before running -> no files processed
    out = {}
    w.finished.connect(lambda r: out.setdefault("r", r))
    w.run()
    assert out["r"] is None  # cancelled -> None

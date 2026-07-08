"""Regression tests for the GUI-hardening pass: graceful failure, threading
safety, copyable error traces, and shared filesystem validation."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

import numpy as np  # noqa: E402
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


def _settings(outdir, remove_ecg=True):
    legacy = {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": 1000},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": [10, 11, 12]}},
        "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                     "avgresamplingobs": 300},
                       "emg": {"remove_ecg": remove_ecg, "remove_noise": False}},
        "output": {"outputfolder": outdir,
                   "data": {"saveaveragedata": True, "savebreathbybreathdata": True}},
    }
    s, _ = migrate_dict(legacy)
    return s


# -- graceful failure: write failure keeps the result + a clear message --------
def test_batchworker_write_failure_delivers_result_and_message(qapp, tmp_path, monkeypatch):
    from respmech.ui import workers
    monkeypatch.setattr(workers, "write_batch",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("No space left on device")))
    w = workers.BatchWorker(_settings(str(tmp_path)), write=True)
    got = {}
    w.finished.connect(lambda r: got.setdefault("r", r))
    w.failed.connect(lambda m: got.setdefault("m", m))
    w.run()
    assert got.get("r") is not None                       # result delivered despite write failure
    assert "writing the output failed" in got["m"] and "No space left" in got["m"]


# -- staging honours remove_ecg and reports the real column + error state ------
def test_stage_emg_channel_respects_remove_ecg_and_reports_col(qapp, tmp_path):
    from respmech.ui.workers import stage_emg_channel
    s = _settings(str(tmp_path), remove_ecg=False)
    d = stage_emg_channel(s, os.path.join(INPUT, "synth_case_A.csv"), 0)
    assert d["ecg_applied"] is False
    assert np.allclose(np.asarray(d["raw"]), np.asarray(d["ecg"]))   # ECG stage skipped
    assert d["col"] == 2                                             # first EMG column
    assert d["noise_error"] is None and d["ecg_error"] is None


# -- run screen: bound file_failed slot, shutdown, run signals, path gating ----
def test_run_screen_bound_slot_shutdown_and_signals(qapp, tmp_path):
    from respmech.ui.screens.run_screen import RunScreen
    rs = RunScreen(AppState(_settings(str(tmp_path))))
    assert hasattr(rs, "run_started") and hasattr(rs, "run_finished")
    rs._on_file_failed("bad.csv", "ValueError: x")        # bound method (not a lambda)
    assert "bad.csv" in rs.log.toPlainText() and "FAILED" in rs.log.toPlainText()
    rs.shutdown()                                          # no running thread -> no crash


def test_run_screen_disabled_without_valid_paths(qapp, tmp_path):
    from respmech.ui.screens.run_screen import RunScreen
    s = _settings(str(tmp_path))
    s.input.folder = str(tmp_path / "does_not_exist")
    rs = RunScreen(AppState(s))
    rs.refresh_actions()
    assert rs.btn_run.isEnabled() is False
    assert "incomplete" in rs.status.text().lower()


# -- shared filesystem validation ---------------------------------------------
def test_path_problem_shared_helper(qapp, tmp_path):
    from respmech.ui.validation import path_problem
    s = _settings(str(tmp_path))
    assert path_problem(s) is None
    s.input.folder = str(tmp_path / "missing")
    assert "input folder" in (path_problem(s) or "")


# -- settings: copyable error surface -----------------------------------------
def test_settings_report_error_is_copyable(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(_settings(str(tmp_path))))
    sc._report_error("Load settings", "Traceback (most recent call last):\nValueError: boom")
    assert "load settings failed" in sc.status.text().lower()
    assert sc._err_dialog is not None and "boom" in sc._err_dialog.text()


# -- preview: enablement-only update never clobbers a fresh status -------------
def test_update_actions_status_false_preserves_status(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._set_status("RESULT MESSAGE")
    pv._update_actions(status=False)
    assert pv.status.text() == "RESULT MESSAGE"
    win.close()


# -- preview: a per-file batch error routes to the copyable error card ---------
def test_on_batch_result_raises_on_file_error(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import BatchResult, FileResult
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")
    result = BatchResult(files={
        "synth_case_A.csv": FileResult(file="synth_case_A.csv", error="ValueError: bad column 7")})
    with pytest.raises(RuntimeError):          # _on_job_done's except turns this into an error card
        pv._on_batch_result(result)
    win.close()


def test_batch_file_error_uses_failed_label_not_display_error(qapp, tmp_path):
    """A per-file analysis error is labelled 'Test run failed' (via _FileRunError),
    not the generic 'display error' that a rendering bug would produce."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    from respmech.core.pipeline import BatchResult, FileResult
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")
    result = BatchResult(files={
        "synth_case_A.csv": FileResult(file="synth_case_A.csv", error="ValueError: bad column 7")})
    job = _Job("batch", pv._tokens["batch"], QThread(), object())
    pv._jobs["batch"] = job
    pv._on_job_done(job, result)               # -> _on_batch_result raises _FileRunError
    assert "test run failed" in pv.status.text().lower()
    assert "display error" not in pv.status.text().lower()
    assert "bad column 7" in (pv.panel_error("campbell") or "")
    win.close()


# -- graceful failure: a malformed settings file must not abort startup --------
def test_malformed_reference_intervals_does_not_abort_construction(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    # a hand-edited TOML shape (inline table) stored raw for the list[Any] field
    s.processing.emg.noise.reference_intervals = {"start": 1.0, "end": 5.0}
    win = MainWindow(AppState(s))              # must NOT raise (falls back to a default region)
    assert win.preview_screen._noise_region is not None
    win.close()

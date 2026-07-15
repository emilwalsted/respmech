"""Regression tests for the GUI-hardening pass: graceful failure, threading
safety, copyable error traces, and shared filesystem validation."""
import os

import pytest




import numpy as np  # noqa: E402

from respmech.ui.state import AppState  # noqa: E402


from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


pytestmark = requires_synth()




def _settings(outdir, remove_ecg=True):
    return synth_settings(outdir, remove_ecg=remove_ecg, data_out=_DATA_OUT)


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


# -- settings labels + variable-path/description tooltips ----------------------
def test_settings_widgets_expose_varpath_and_description_on_hover(qapp, tmp_path):
    from PySide6.QtWidgets import QLabel
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(_settings(str(tmp_path))))
    checks = {
        sc.samp_freq: "input.format.sampling_frequency",
        sc.col_flow: "input.channels.flow",
        sc.cols_emg: "input.channels.emg",
        sc.emg_rms_window: "processing.emg.rms_window_s",
        sc.remove_noise: "processing.emg.noise.enabled",
        sc.noise_ref: "processing.emg.noise.reference_file",
        sc.out_folder: "output.folder",
        sc.integrate: "processing.volume.integrate_from_flow",
    }
    for w, var in checks.items():
        tip = w.toolTip()
        assert var in tip, f"{var} missing from tooltip: {tip!r}"
        assert len(tip) > len(var) + 15, f"no description for {var}: {tip!r}"
    # visible labels are human, never the raw variable name
    labels = {la.text() for la in sc.findChildren(QLabel)}
    assert "Sampling frequency" in labels
    assert "input.format.sampling_frequency" not in labels


def test_preview_noise_params_expose_varpath_on_hover(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    assert "processing.emg.noise.auto_prop" in pv.noise_auto.toolTip()
    assert "processing.emg.noise.prop_decrease" in pv.noise_prop.toolTip()
    assert "processing.emg.noise.fidelity_target" in pv.noise_target.toolTip()
    assert "processing.emg.noise.n_std_thresh" in pv.noise_nstd.toolTip()
    # the ECG-removal params moved from Setup onto the ECG-reduction tab; varpaths follow
    assert "processing.emg.remove_ecg" in pv.remove_ecg.toolTip()
    assert "processing.emg.ecg_min_distance_s" in pv.ecg_min_distance.toolTip()
    assert "processing.emg.ecg_window_s" in pv.ecg_window.toolTip()
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

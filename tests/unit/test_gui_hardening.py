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


# -- A05: BatchWorker decides cohort_outputs via is_subset_run, consistently --------
def test_batchworker_sets_cohort_outputs_false_for_a_real_subset(qapp, tmp_path, monkeypatch):
    from respmech.ui import workers
    calls = {}
    monkeypatch.setattr(workers, "write_batch",
                        lambda *a, **k: calls.setdefault("cohort_outputs", k.get("cohort_outputs")))
    w = workers.BatchWorker(_settings(str(tmp_path)), write=True, only_files=["synth_case_A.csv"])
    w.run()
    assert calls["cohort_outputs"] is False


def test_batchworker_keeps_cohort_outputs_true_when_only_files_names_everything(qapp, tmp_path, monkeypatch):
    """only_files happening to list EVERY matching file (a 'Re-run failed' where every file
    failed) is not a real subset — must not skip the cohort outputs it fully covers."""
    from respmech.ui import workers
    calls = {}
    monkeypatch.setattr(workers, "write_batch",
                        lambda *a, **k: calls.setdefault("cohort_outputs", k.get("cohort_outputs")))
    w = workers.BatchWorker(_settings(str(tmp_path)), write=True,
                            only_files=["synth_case_A.csv", "synth_case_B.csv"])
    w.run()
    assert calls["cohort_outputs"] is True


def test_batchworker_full_run_defaults_cohort_outputs_true(qapp, tmp_path, monkeypatch):
    from respmech.ui import workers
    calls = {}
    monkeypatch.setattr(workers, "write_batch",
                        lambda *a, **k: calls.setdefault("cohort_outputs", k.get("cohort_outputs")))
    w = workers.BatchWorker(_settings(str(tmp_path)), write=True)   # only_files=None -> full run
    w.run()
    assert calls["cohort_outputs"] is True


def test_batchworker_is_subset_run_failure_reports_write_failed_not_a_crash(qapp, tmp_path, monkeypatch):
    """is_subset_run is called INSIDE the write try-block on purpose: a transiently
    unreadable input folder must surface as an ordinary write failure (finished + failed
    signals), not crash the worker thread with an unhandled exception that leaves the Run
    screen with no signal at all — the same review fix already applied to the earlier
    (data-losing) revision of this ticket."""
    from respmech.ui import workers
    from respmech.core import pipeline
    monkeypatch.setattr(pipeline, "is_subset_run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("permission denied")))
    w = workers.BatchWorker(_settings(str(tmp_path)), write=True, only_files=["synth_case_A.csv"])
    got = {}
    w.finished.connect(lambda r: got.setdefault("r", r))
    w.failed.connect(lambda m: got.setdefault("m", m))
    w.run()
    assert got.get("r") is not None
    assert "writing the output failed" in got["m"] and "permission denied" in got["m"]


# -- ticket A06: WriteWorker — "write elsewhere", no re-analysis ---------------
def test_writeworker_writes_to_the_given_folder(qapp, tmp_path):
    from respmech.ui import workers
    from respmech.core.io.plan import plan_outputs
    from respmech.core.pipeline import run_batch
    s = _settings(str(tmp_path / "original"), remove_ecg=False)
    result = run_batch(s)
    files = [os.path.join(INPUT, "synth_case_A.csv"), os.path.join(INPUT, "synth_case_B.csv")]
    plan = plan_outputs(s, files)
    elsewhere = tmp_path / "elsewhere"
    w = workers.WriteWorker(result, s, plan, str(elsewhere))
    got = {}
    w.finished.connect(lambda paths: got.setdefault("paths", paths))
    w.failed.connect(lambda m: got.setdefault("m", m))
    w.run()
    assert "m" not in got
    assert got["paths"] and all(p.startswith(str(elsewhere)) for p in got["paths"])
    assert not (tmp_path / "original").exists()


def test_writeworker_reports_failure_without_crashing(qapp, tmp_path, monkeypatch):
    from respmech.ui import workers
    from respmech.core.io.plan import plan_outputs
    from respmech.core.pipeline import run_batch
    s = _settings(str(tmp_path), remove_ecg=False)
    result = run_batch(s)
    plan = plan_outputs(s, [os.path.join(INPUT, "synth_case_A.csv")])
    monkeypatch.setattr(workers, "write_planned",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    w = workers.WriteWorker(result, s, plan, str(tmp_path / "elsewhere"))
    got = {}
    w.finished.connect(lambda paths: got.setdefault("paths", paths))
    w.failed.connect(lambda m: got.setdefault("m", m))
    w.run()
    assert "paths" not in got
    assert "disk full" in got["m"]


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


def test_path_problem_names_the_rest_reference_not_a_bare_noise_reference(qapp, tmp_path):
    """Ticket D21 (UI-overhaul): 'noise reference' is ambiguous — the same word also names
    the ECG auto-detect reference and the per-file EMG normalisation reference, none of
    which this check is about. The message must call it what the Preview & QC chip and
    'Set noise profile' button call it ('rest reference'), and must say where to fix it —
    there is no control for this anywhere on Setup."""
    from respmech.ui.validation import path_problem
    s = synth_settings(str(tmp_path), noise=True, data_out=_DATA_OUT)
    assert path_problem(s) is None                      # the synthetic reference file exists
    s.processing.emg.noise.reference_file = "does_not_exist.csv"
    problem = path_problem(s) or ""
    assert "rest reference" in problem
    assert "noise reference" not in problem
    assert "does_not_exist.csv" in problem
    assert "Preview & QC" in problem and "EMG – noise reduction" in problem


# -- ticket A06: a real write probe, not os.access ------------------------
def test_path_problem_probe_write_is_off_by_default(qapp, tmp_path, monkeypatch):
    """Settings' live, every-keystroke validation must never touch disk beyond the
    existing os.path.isdir checks — probe_write_folder is only called when explicitly
    asked for (probe_write=True)."""
    from respmech.ui import validation
    s = _settings(str(tmp_path))
    called = []
    import respmech.core.io.plan as plan_mod
    monkeypatch.setattr(plan_mod, "probe_write_folder",
                        lambda folder: called.append(folder) or plan_mod.WriteProbe(True))
    assert validation.path_problem(s) is None
    assert called == []
    assert validation.path_problem(s, probe_write=True) is None
    assert called == [str(tmp_path)]


def test_path_problem_probe_write_reports_an_unwritable_folder(qapp, tmp_path):
    from respmech.ui.validation import path_problem
    out = tmp_path / "readonly-out"
    out.mkdir()
    out.chmod(0o500)                       # read + execute, no write
    try:
        if os.access(str(out), os.W_OK):   # root (some CI runners) ignores the mode bit
            pytest.skip("running as a user that can still write a 0500 directory")
        s = _settings(str(out))
        problem = path_problem(s, probe_write=True)
        assert problem is not None and "write" in problem.lower()
    finally:
        out.chmod(0o700)                   # restore so pytest's tmp_path cleanup can remove it


def test_probe_write_folder_creates_and_cleans_up_missing_parents(tmp_path):
    """A dry run's probe (and the output-folder picker) must never leave an empty folder
    behind — ticket A06 point 6. Every directory the probe itself created is removed
    again, deepest first, even though the probe SUCCEEDED."""
    from respmech.core.io.plan import probe_write_folder
    target = tmp_path / "a" / "b" / "c"
    probe = probe_write_folder(str(target))
    assert probe.ok
    assert not target.exists()
    assert not (tmp_path / "a").exists()   # the whole chain it created is gone
    assert tmp_path.exists()               # but tmp_path itself (pre-existing) is untouched


def test_probe_write_folder_leaves_a_pre_existing_folder_alone(tmp_path):
    from respmech.core.io.plan import probe_write_folder
    existing = tmp_path / "already-here"
    existing.mkdir()
    probe = probe_write_folder(str(existing))
    assert probe.ok
    assert existing.is_dir()               # never removed — it was not this call's to remove


def test_probe_write_folder_cleans_up_created_parents_even_when_the_probe_fails(tmp_path, monkeypatch):
    """Self-review fix: os.makedirs() creates parent directories one at a time, so a
    failure could previously happen AFTER some ancestors were already created but before
    the function's own cleanup ran — reproduced here by failing the write step itself
    (tempfile.mkstemp), which happens strictly after makedirs succeeds."""
    from respmech.core.io import plan as plan_mod
    monkeypatch.setattr(plan_mod.tempfile, "mkstemp",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no space left")))
    target = tmp_path / "x" / "y" / "z"
    probe = plan_mod.probe_write_folder(str(target))
    assert not probe.ok and "write" in probe.message.lower()
    assert not target.exists()
    assert not (tmp_path / "x").exists()   # the whole chain this call created is still gone


def test_probe_write_folder_succeeds_even_if_deleting_the_probe_file_fails(tmp_path, monkeypatch):
    """Self-review fix: a failure to delete the temp probe file happens AFTER the write
    that actually matters already succeeded, so it must never be reported as a write
    failure (the original version put both steps in one try/except and did exactly that)."""
    from respmech.core.io import plan as plan_mod
    monkeypatch.setattr(plan_mod.os, "remove",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("permission denied")))
    probe = plan_mod.probe_write_folder(str(tmp_path))
    assert probe.ok and probe.message == ""


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
    pv.file_rail.select_filename("synth_case_A.csv")
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
    pv.file_rail.select_filename("synth_case_A.csv")
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
        sc.out_folder: "output.folder",
    }
    for w, var in checks.items():
        tip = w.toolTip()
        assert var in tip, f"{var} missing from tooltip: {tip!r}"
        assert len(tip) > len(var) + 15, f"no description for {var}: {tip!r}"
    # the gated-peak controls moved to the Preview EMG tab (test_robust_peak_ui.py) and
    # its five guards into the EMG Advanced modal (test_advanced_dialog.py)
    # the channel columns have no fields any more: their settings paths moved onto the
    # read-only summary rows, covered exhaustively by test_channel_summary.py
    # visible labels are human, never the raw variable name. fullText() where available
    # (ElidingLabel, built by _row()/_browse_row()): text() only reflects what a resize
    # event has actually laid out and would silently pass here for the wrong reason on an
    # unshown window (CLAUDE.md: never assert on a QLabel's rendered text() when
    # flow_layout.elide could have shortened it).
    labels = {getattr(la, "fullText", la.text)() for la in sc.findChildren(QLabel)}
    assert "Sampling frequency" in labels
    assert "input.format.sampling_frequency" not in labels


def test_every_settings_tooltip_names_a_real_settings_field(qapp, tmp_path):
    """The dict above is hand-maintained, so a widget added without an entry is unchecked and
    a typo in its dotted path ships silently. This resolves whatever path each tooltip claims
    against the live Settings object, which covers every widget automatically."""
    import re
    from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(_settings(str(tmp_path))))
    seen = 0
    widgets = [w for cls in (QAbstractSpinBox, QCheckBox, QComboBox, QLineEdit)
               for w in sc.findChildren(cls)]     # PySide6 findChildren takes one type only
    for w in widgets:
        m = re.search(r"<b>([a-z_]+(?:\.[a-z_0-9]+)+)</b>", w.toolTip() or "")
        if not m:
            continue
        obj = sc.state.settings
        for part in m.group(1).split("."):
            assert hasattr(obj, part), f"{m.group(1)} does not resolve: no '{part}' on {obj!r}"
            obj = getattr(obj, part)
        seen += 1
    assert seen > 12, f"only {seen} tooltips carried a settings path — the regex likely broke"
    # (Setup is lean now — most conditioning tooltips live in the Preview Advanced modals)


def test_preview_noise_params_expose_varpath_on_hover(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    assert "processing.emg.noise.auto_prop" in pv.noise_auto.toolTip()
    # prop_decrease / fidelity_target / n_std_thresh live only in the Advanced modal now,
    # whose every row names its dotted path by construction — that is covered by
    # test_every_row_names_its_settings_variable rather than duplicated here.
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

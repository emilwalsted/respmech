"""Run screen — the tune-to-run loop & batch control: re-run only the failed files (P18),
"Process & write this file" from Preview (P19), and the per-file results table with
drill-back into Preview (P20). Previously test_review_wave5.py; the run-screen tests from
wave 2 (dry-run plan P5, overwrite guard, open-folder P7) are appended below."""
from types import SimpleNamespace

import pandas as pd
import pytest

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _win(tmp):
    from respmech.ui.main_window import MainWindow
    return MainWindow(AppState(synth_settings(tmp)))


def _result_with_failure():
    ok = SimpleNamespace(error=None, breaths_table=pd.DataFrame({"vt": [0.5, 0.6]}))
    bad = SimpleNamespace(error="TrimError: too few breaths", breaths_table=None)
    return SimpleNamespace(files={"synth_case_A.csv": ok, "synth_case_B.csv": bad},
                           failed_files={"synth_case_B.csv": bad},
                           ok_files={"synth_case_A.csv": ok})


# --------------------------------------------------------------------------- #
# P20 — per-file results + drill-back
# --------------------------------------------------------------------------- #
def test_per_file_results_table(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    rn._fill_files_table(_result_with_failure())
    assert rn.files_table.rowCount() == 2
    rows = {rn.files_table.item(r, 0).text(): (rn.files_table.item(r, 1).text(),
                                               rn.files_table.item(r, 2).text())
            for r in range(2)}
    assert rows["synth_case_A.csv"] == ("OK", "2")
    assert rows["synth_case_B.csv"][0] == "Failed"
    win.close()


def test_double_click_drills_into_preview(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    rn._fill_files_table(_result_with_failure())
    seen = []
    rn.open_file_requested.connect(seen.append)
    rn._on_file_row_activated(0, 0)
    assert seen == ["synth_case_A.csv"]
    win._open_file_in_preview("synth_case_A.csv")        # MainWindow routing
    assert win.tabs.currentIndex() == win._i_preview
    win.close()


# --------------------------------------------------------------------------- #
# P18 — re-run failed
# --------------------------------------------------------------------------- #
def test_rerun_failed_restricts_to_failed_subset(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    calls = []
    rn._start = lambda write, only_files=None: calls.append((write, only_files))
    rn._last_failed = ["synth_case_B.csv"]
    rn._last_write = True
    rn._rerun_failed()
    assert calls == [(True, ["synth_case_B.csv"])]       # only the failed file, same write mode
    win.close()


def test_rerun_button_enabled_only_with_failures(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    assert not rn.btn_rerun.isEnabled()                  # nothing has run yet
    rn._set_running(True)
    assert not rn.btn_rerun.isEnabled()                  # never enabled mid-run
    win.close()


# --------------------------------------------------------------------------- #
# P19 — process & write this file from Preview
# --------------------------------------------------------------------------- #
def test_process_this_file_routes_to_single_file_run(qapp, tmp_path):
    win = _win(tmp_path); pv = win.preview_screen; rn = win.run_screen
    calls = []
    rn._start = lambda write, only_files=None: calls.append((write, only_files))
    pv._previewed_file = "synth_case_A.csv"
    pv._process_this_file()                              # emits → MainWindow → run_single_file
    assert calls == [(True, ["synth_case_A.csv"])]       # write=True, just this file
    assert win.tabs.currentIndex() == win._i_run         # and it switched to the Run screen
    win.close()


def test_subset_run_is_noted_in_the_plan(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    rn._only_files = ["synth_case_B.csv"]
    rn._append_plan(write=False)
    log = rn.log.toPlainText()
    assert "subset" in log and "synth_case_B.csv" in log
    win.close()


# ---------------------------------------------------------------------------
# Dry-run pre-flight plan + overwrite guard (P5) — from wave 2
# ---------------------------------------------------------------------------
def test_dry_run_plan_lists_inputs_and_planned_outputs(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    rn._append_plan(write=False)
    log = rn.log.toPlainText()
    assert "DRY RUN" in log and "nothing will be written" in log.lower()
    assert log.count("synth_case_") >= 2                      # both inputs listed
    assert "analysis-used.toml" in log and "run-report.txt" in log
    win.close()


def test_planned_outputs_follow_save_flags(qapp, tmp_path):
    """The plan (ticket A06: core.io.plan.plan_outputs, rendered by _append_plan) follows
    the save flags exactly like the old hand-built list used to — just built from the one
    shared source of truth now, instead of its own copy of the same logic."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); rn = win.run_screen
    s.output.data.save_processed = False
    rn._append_plan(write=False)
    assert not any("Processed data" in ln for ln in rn.log.toPlainText().splitlines())
    rn.log.clear()
    s.output.data.save_processed = True
    rn._append_plan(write=False)
    assert any("Processed data" in ln for ln in rn.log.toPlainText().splitlines())
    win.close()


def test_overwrite_guard_detects_prior_results(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    assert rn._existing_output() is None                     # empty folder — nothing to clobber
    (tmp_path / "analysis-used.toml").write_text("x")
    data = tmp_path / "data"; data.mkdir()
    (data / "Average breathdata.xlsx").write_text("y")
    ex = rn._existing_output()
    assert ex is not None and ex[1] == 2                      # 2 prior files detected
    win.close()


def test_open_output_button_starts_disabled(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    assert not rn.btn_open.isEnabled()                        # enabled only after a real write
    win.close()


# ---------------------------------------------------------------------------
# A06 — "Write results to another folder…" (write_planned, no re-analysis)
# ---------------------------------------------------------------------------
def test_show_plan_button_enabled_only_after_a_plan_exists(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    assert not rn.btn_show_plan.isEnabled()
    rn._append_plan(write=False)
    assert rn.btn_show_plan.isEnabled()
    win.close()


def test_write_elsewhere_button_enabled_only_when_writing_just_failed(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    assert not rn.btn_write_elsewhere.isEnabled()

    rn._last_write = True
    rn._run_settings_snapshot = rn.state.settings
    rn._fatal_msg = "writing failed: disk full"
    rn._on_finished(_result_with_failure())              # analysis succeeded, "write" failed
    assert rn.btn_write_elsewhere.isEnabled()
    win.close()


def test_write_elsewhere_button_stays_disabled_after_an_ordinary_successful_run(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    rn._last_write = True
    rn._run_settings_snapshot = rn.state.settings
    rn._fatal_msg = None
    rn._on_finished(_result_with_failure())
    assert not rn.btn_write_elsewhere.isEnabled()
    win.close()


def test_write_elsewhere_button_stays_disabled_after_a_fatal_run_with_no_result(qapp, tmp_path):
    """A fatal message with NO result means the analysis itself never completed — 'write
    elsewhere' has nothing to write, so it must not be offered (distinct from write_failed,
    which needs BOTH a fatal message AND a delivered result)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    rn._last_write = True
    rn._run_settings_snapshot = rn.state.settings
    rn._fatal_msg = "no input files found"
    rn._on_finished(None)
    assert not rn.btn_write_elsewhere.isEnabled()
    win.close()


def test_a_new_run_is_refused_while_write_elsewhere_is_in_flight(qapp, tmp_path):
    """refresh_actions()/_start() must treat self._write_thread exactly like self._thread —
    a review finding: without this, a 'write elsewhere' retry and a fresh run could drive
    the same log/status/result state from two threads at once."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    rn._write_thread = object()          # any non-None sentinel — _start only checks identity
    rn.refresh_actions()
    assert not rn.btn_run.isEnabled()
    assert not rn.btn_dry.isEnabled()
    calls = []
    rn._settings_ok = lambda: (True, "")
    rn._append_plan = lambda write: calls.append(write)
    rn._start(write=True)
    assert calls == []                    # refused before it ever gets to building a plan
    rn._write_thread = None
    win.close()


def test_write_elsewhere_does_nothing_without_a_prior_failed_write(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    rn._write_elsewhere()                 # no _last_result/_last_plan yet — must be a no-op
    assert rn._write_thread is None
    win.close()


def test_write_elsewhere_locks_run_buttons_immediately_and_emits_run_started(qapp, tmp_path, monkeypatch):
    """Two self-review findings in one test: (1) refresh_actions() ran with
    self._write_thread still None, so it briefly re-enabled Run/Dry-run for the whole
    write, not just a moment; (2) run_started/run_finished were never emitted for a
    write-elsewhere at all, so MainWindow never locked Settings/the Analysis menu the way
    it does for a normal run's write phase. Both must hold the instant _write_elsewhere()
    returns — a plain synchronous call, since QThread.start() itself returns immediately."""
    import os
    from PySide6.QtWidgets import QFileDialog
    from respmech.ui.main_window import MainWindow
    from respmech.core.io.plan import plan_outputs
    from respmech.core.pipeline import run_batch
    s = synth_settings(tmp_path / "original")
    win = MainWindow(AppState(s)); rn = win.run_screen

    files = [os.path.join(INPUT, "synth_case_A.csv"), os.path.join(INPUT, "synth_case_B.csv")]
    rn._last_result = run_batch(s)
    rn._run_settings_snapshot = s
    rn._last_plan = plan_outputs(s, files)
    rn._last_write = True

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(tmp_path / "elsewhere"))
    started = []
    rn.run_started.connect(lambda: started.append(True))

    rn._write_elsewhere()
    assert started == [True]
    assert rn._write_thread is not None
    assert not rn.btn_run.isEnabled()          # busy — refresh_actions saw the thread
    assert not rn.btn_dry.isEnabled()
    assert not rn.btn_write_elsewhere.isEnabled()   # can't start a second one mid-write

    win.close()                                # shutdown() joins the write thread safely


# ---------------------------------------------------------------------------
# A05 — subset writes get their own overwrite dialog, routing, and a
# permanent post-write note that the study's cohort outputs were left alone.
# ---------------------------------------------------------------------------
def test_confirm_overwrite_subset_names_artifacts_and_leaves_cohort_alone(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    seen = {}
    def _question(*a, **k):
        seen["text"] = a[2]
        return QMessageBox.Yes
    monkeypatch.setattr(QMessageBox, "question", _question)
    ok = rn._confirm_overwrite_subset(["synth_case_B.csv"])
    assert ok is True
    text = seen["text"]
    assert "synth_case_B.csv" in text
    assert "Average breathdata.xlsx" in text and "Cohort summary.xlsx" in text
    assert "will NOT be touched" in text
    assert "run-report.txt" not in text            # never claims to touch the full-run report
    win.close()


def test_confirm_overwrite_subset_mentions_the_cohort_figure_only_when_enabled(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    seen = {}
    def _question(*a, **k):
        seen["text"] = a[2]
        return QMessageBox.Yes
    monkeypatch.setattr(QMessageBox, "question", _question)
    rn.state.settings.output.diagnostics.save_pv_individual = False
    rn._confirm_overwrite_subset(["synth_case_A.csv"])
    assert "cohort Campbell figure" not in seen["text"]
    rn.state.settings.output.diagnostics.save_pv_individual = True
    rn._confirm_overwrite_subset(["synth_case_A.csv"])
    assert "cohort Campbell figure" in seen["text"]
    win.close()


def test_confirm_overwrite_subset_is_accurate_after_a_prior_subset_write(qapp, tmp_path, monkeypatch):
    """Regression: the dialog used to derive its "from a full run on {when}" claim from
    _existing_output()'s generic newest-mtime, which a PRIOR subset write's own per-file
    artefact would also bump — making the dialog lie about a subset write being a full run.
    It must instead be anchored to Average breathdata.xlsx's own mtime, which only a full
    run ever touches, and say plainly when no full run has happened yet."""
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    # no full run yet, but a PRIOR subset write's own per-file artefact is newer than nothing
    data = tmp_path / "data"; data.mkdir()
    (data / "synth_case_B.csv.breathdata.xlsx").write_text("y")   # a prior subset write's own file
    seen = {}
    def _question(*a, **k):
        seen["text"] = a[2]
        return QMessageBox.Yes
    monkeypatch.setattr(QMessageBox, "question", _question)
    rn._confirm_overwrite_subset(["synth_case_A.csv"])
    assert "no full run has produced them yet" in seen["text"]
    assert "full run at" not in seen["text"]           # never a timestamp for a run that didn't happen
    win.close()


def test_start_routes_a_real_subset_to_the_subset_dialog(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    (tmp_path / "analysis-used.toml").write_text("x")
    data = tmp_path / "data"; data.mkdir()
    (data / "Average breathdata.xlsx").write_text("y")
    calls = []
    rn._confirm_overwrite_subset = lambda files: (calls.append(("subset", files)), False)[1]
    rn._confirm_overwrite = lambda existing: calls.append(("generic",)) or False
    rn._start(write=True, only_files=["synth_case_A.csv"])
    assert calls == [("subset", ["synth_case_A.csv"])]        # the subset dialog, not the generic one
    assert rn.status.text() == "Run cancelled — existing results kept."
    win.close()


def test_start_routes_a_full_run_to_the_generic_dialog(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    (tmp_path / "analysis-used.toml").write_text("x")
    data = tmp_path / "data"; data.mkdir()
    (data / "Average breathdata.xlsx").write_text("y")
    calls = []
    rn._confirm_overwrite_subset = lambda files: calls.append(("subset", files)) or False
    rn._confirm_overwrite = lambda existing: (calls.append(("generic",)), False)[1]
    rn._start(write=True, only_files=None)                    # a plain "Run batch"
    assert calls == [("generic",)]
    win.close()


def test_start_routes_only_files_naming_everything_to_the_generic_dialog(qapp, tmp_path):
    """only_files listing every matching file is not a real subset (is_subset_run agrees) —
    the generic dialog is correct because the write really does cover the whole study."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    (tmp_path / "analysis-used.toml").write_text("x")
    data = tmp_path / "data"; data.mkdir()
    (data / "Average breathdata.xlsx").write_text("y")
    calls = []
    rn._confirm_overwrite_subset = lambda files: calls.append(("subset", files)) or False
    rn._confirm_overwrite = lambda existing: (calls.append(("generic",)), False)[1]
    rn._start(write=True, only_files=["synth_case_A.csv", "synth_case_B.csv"])
    assert calls == [("generic",)]
    win.close()


def test_safe_is_subset_run_degrades_to_false_on_a_transient_error(qapp, tmp_path, monkeypatch):
    """Both call sites of is_subset_run in this screen go through _safe_is_subset_run, which
    must never let a transient OSError (a dropped network drive) escape a Qt slot — see its
    docstring for why False (not True) is the safe default here."""
    from respmech.ui.main_window import MainWindow
    from respmech.core import pipeline
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    monkeypatch.setattr(pipeline, "is_subset_run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("dropped")))
    assert rn._safe_is_subset_run(["synth_case_A.csv"]) is False
    win.close()


def test_subset_cohort_note_reads_the_full_runs_timestamp_from_disk(qapp, tmp_path):
    import os as _os
    from datetime import datetime
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    data = tmp_path / "data"; data.mkdir()
    avg = data / "Average breathdata.xlsx"; avg.write_text("y")
    stamp = datetime(2026, 7, 11, 14, 0, 0).timestamp()
    _os.utime(avg, (stamp, stamp))
    note = rn._subset_cohort_note(["synth_case_B.csv"])
    assert "2026-07-11 14:00" in note
    assert "1 file processed in this write" in note
    assert "Average breathdata.xlsx" in note and "Cohort summary.xlsx" in note
    win.close()


def test_subset_cohort_note_when_no_full_run_exists_yet(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen
    note = rn._subset_cohort_note(["synth_case_A.csv"])
    assert "No full-batch" in note and "exist yet" in note
    win.close()


def test_persistent_subset_note_appears_only_after_a_successful_subset_write(qapp, tmp_path):
    """The note belongs on a clean, real-write, non-subset-cancelled outcome — not on a
    cancelled run, a fatal error, or a plain full run."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path))); rn = win.run_screen

    rn._last_write = True
    rn._only_files = ["synth_case_A.csv"]
    rn._fatal_msg = None
    rn._on_finished(_result_with_failure())
    log1 = rn.log.toPlainText()
    assert "Average breathdata.xlsx" in log1 and "Cohort summary.xlsx" in log1

    win2 = MainWindow(AppState(synth_settings(tmp_path)))
    rn2 = win2.run_screen
    rn2._last_write = True
    rn2._only_files = None                     # a full run — no subset note expected
    rn2._fatal_msg = None
    rn2._on_finished(_result_with_failure())
    log2 = rn2.log.toPlainText()
    assert "Average breathdata.xlsx" not in log2 and "Cohort summary.xlsx" not in log2
    win.close(); win2.close()

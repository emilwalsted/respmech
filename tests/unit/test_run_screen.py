"""Run screen — the tune-to-run loop & batch control: re-run only the failed files (P18),
"Process & write this file" from Preview (P19), and the per-file results table (P20/B02,
now shown in the ONE file rail shared with Preview & QC — see the B03 section near the
end for the drawer/collapse/shared-rail tests specifically). Previously
test_review_wave5.py; the run-screen tests from wave 2 (dry-run plan P5, overwrite guard,
open-folder P7) are appended below."""
from types import SimpleNamespace

import pandas as pd
import pytest

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _win(tmp):
    from respmech.ui.main_window import MainWindow
    return MainWindow(AppState(synth_settings(tmp)))


def _pump_until_thread_done(qapp, rn, timeout=30.0):
    """Spin a REAL Qt event loop until ``rn``'s batch thread has finished and cleared
    itself, or ``timeout`` seconds pass. A hand-rolled ``processEvents()`` loop does not
    reliably deliver a WORKER THREAD's cross-thread queued ``finished`` signal on every
    platform (see test_gui_reactive.py's own ``_pump_until``, which exists for the same
    reason) — a real ``QEventLoop`` does. Needed whenever a test lets ``_start()`` create
    a real thread and must not close the window (destroying the slot the signal targets)
    before that signal is actually delivered."""
    from PySide6.QtCore import QElapsedTimer, QEventLoop, QTimer
    if rn._thread is None:
        return True
    loop = QEventLoop()
    clock = QElapsedTimer(); clock.start()
    state = {"ok": False}
    timer = QTimer(); timer.setInterval(10)

    def _tick():
        if rn._thread is None:
            state["ok"] = True
            loop.quit()
        elif clock.elapsed() > timeout * 1000:
            loop.quit()

    timer.timeout.connect(_tick)
    timer.start()
    loop.exec()
    timer.stop()
    return state["ok"] or rn._thread is None


def _result_with_failure():
    ok = SimpleNamespace(error=None, breaths_table=pd.DataFrame({"vt": [0.5, 0.6]}))
    bad = SimpleNamespace(error="TrimError: too few breaths", breaths_table=None)
    return SimpleNamespace(files={"synth_case_A.csv": ok, "synth_case_B.csv": bad},
                           failed_files={"synth_case_B.csv": bad},
                           ok_files={"synth_case_A.csv": ok})


# --------------------------------------------------------------------------- #
# P20/B02 — per-file results (file rail) + drill-back
# --------------------------------------------------------------------------- #
def test_per_file_results_rail(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    rn._fill_file_rail(_result_with_failure())
    assert rn.file_rail.count() == 2
    a, b = rn.file_rail.entry("synth_case_A.csv"), rn.file_rail.entry("synth_case_B.csv")
    assert a.verdict == "ok" and a.breaths == 2
    assert b.verdict == "failed"
    win.close()


def test_failed_files_sort_first_after_a_run_with_failures(qapp, tmp_path):
    """B02: reaching the one failed file in a large batch must take a glance, not a
    scroll past every success — the rail brings failures to the top once a run's result
    names any."""
    win = _win(tmp_path); rn = win.run_screen
    rn._fill_file_rail(_result_with_failure())
    assert rn.file_rail.visible_filenames()[0] == "synth_case_B.csv"
    win.close()


def test_a_full_run_replaces_stale_rows_from_a_changed_input_folder(qapp, tmp_path):
    """A FULL run's file set is the complete truth — self-review finding: unioning it
    with the rail's existing rows (right for a P18/P19 subset) is wrong for a plain
    second full run, since it would leave a PREVIOUS folder's files (and their stale
    verdicts) permanently mixed into the rail after the user changes the input folder."""
    win = _win(tmp_path); rn = win.run_screen
    ok = SimpleNamespace(error=None, breaths_table=pd.DataFrame({"vt": [0.5]}))
    old_folder_result = SimpleNamespace(files={"old_case.csv": ok}, failed_files={},
                                        ok_files={"old_case.csv": ok})
    rn._fill_file_rail(old_folder_result)
    assert rn.file_rail.filenames() == ["old_case.csv"]

    rn._only_files = None                    # a plain full run, not a P18/P19 subset
    new_folder_result = _result_with_failure()   # synth_case_A.csv / synth_case_B.csv
    rn._fill_file_rail(new_folder_result)
    assert rn.file_rail.filenames() == ["synth_case_A.csv", "synth_case_B.csv"], (
        "old_case.csv from the previous folder must not survive a full run's rebuild")
    win.close()


def test_subset_rerun_keeps_an_untouched_failed_file_sorted_first(qapp, tmp_path):
    """Self-review finding: sort_failed_first must be judged from the rail's OWN current
    state, not just the just-finished run's result — a P18/P19 subset re-run that fixes
    ONE file must not silently un-sort another, untouched file that is still failed."""
    win = _win(tmp_path); rn = win.run_screen
    ok = SimpleNamespace(error=None, breaths_table=pd.DataFrame({"vt": [0.5]}))
    bad = SimpleNamespace(error="TrimError: too few breaths", breaths_table=None)
    full = SimpleNamespace(files={"a.csv": ok, "b.csv": bad, "c.csv": bad},
                           failed_files={"b.csv": bad, "c.csv": bad}, ok_files={"a.csv": ok})
    rn._fill_file_rail(full)
    assert rn.file_rail.visible_filenames()[0] in ("b.csv", "c.csv")

    fixed = SimpleNamespace(error=None, breaths_table=pd.DataFrame({"vt": [0.6]}))
    rn._only_files = ["b.csv"]              # re-run ONLY b, which now succeeds
    subset = SimpleNamespace(files={"b.csv": fixed}, failed_files={}, ok_files={"b.csv": fixed})
    rn._fill_file_rail(subset)
    assert rn.file_rail.entry("b.csv").verdict == "ok"
    # c.csv was never touched by the subset run and is STILL failed -> must stay first
    assert rn.file_rail.visible_filenames()[0] == "c.csv"
    win.close()


def test_run_and_preview_share_one_file_rail(qapp, tmp_path):
    """B03: Run & results has no rail of its own any more — it shares Preview & QC's, the
    ONE place per-file rows are shown. Selecting a row there already re-renders Preview &
    QC for it, so there is no separate destination left to 'drill back' to."""
    win = _win(tmp_path); rn = win.run_screen; pv = win.preview_screen
    assert rn.file_rail is pv.file_rail
    rn._fill_file_rail(_result_with_failure())
    seen = []
    pv.file_rail.selectionChanged.connect(seen.append)
    # the rail already auto-adopted its first file quietly on construction (no signal for
    # that) — select the OTHER one to actually observe a real change
    pv.file_rail.select_filename("synth_case_B.csv")
    assert seen == ["synth_case_B.csv"]
    assert pv.file_rail.current_filename() == "synth_case_B.csv"
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
    """B03: the request originates on Preview & QC, and the Run drawer lives on that same
    tab now — there is no separate 'Run screen' left to switch to, so the tab never moves."""
    win = _win(tmp_path); pv = win.preview_screen; rn = win.run_screen
    before = win.tabs.currentIndex()
    calls = []
    rn._start = lambda write, only_files=None: calls.append((write, only_files))
    pv._previewed_file = "synth_case_A.csv"
    pv._process_this_file()                              # emits → MainWindow → run_single_file
    assert calls == [(True, ["synth_case_A.csv"])]       # write=True, just this file
    assert win.tabs.currentIndex() == before             # no tab to switch to any more
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


# ---------------------------------------------------------------------------
# B03 — Run & results folded into the workspace as a drawer under the file rail
# ---------------------------------------------------------------------------
def test_run_has_no_tab_of_its_own(qapp, tmp_path):
    win = _win(tmp_path)
    assert win.tabs.count() == 2
    tab_widgets = [win.tabs.widget(i) for i in range(win.tabs.count())]
    assert win.run_screen not in tab_widgets
    assert win.preview_screen in tab_widgets
    win.close()


def test_run_and_preview_share_the_same_file_rail_instance(qapp, tmp_path):
    win = _win(tmp_path)
    assert win.run_screen.file_rail is win.preview_screen.file_rail
    win.close()


def test_io_info_shows_folder_mask_count_and_output_before_running(qapp, tmp_path):
    """Ticket acceptance: before Run is ever pressed, the input folder, the mask, the file
    count and the output folder must already be on screen. Asserted on toolTip() (always
    the untruncated text) and on label text, never on pixel widths."""
    win = _win(tmp_path); rn = win.run_screen
    assert rn.file_rail.count() == 2                  # synth_case_A.csv + synth_case_B.csv
    read_tip = rn.read_info.toolTip()
    write_tip = rn.write_info.toolTip()
    assert "2 file" in read_tip
    assert "synth_case_*.csv" in read_tip
    assert INPUT in read_tip
    assert "Writing:" in write_tip
    assert str(tmp_path) in write_tip
    # the label text (unshown widget, width 0 -> ElidingLabel shows the full string) matches
    assert rn.read_info.text() == read_tip
    assert rn.write_info.text() == write_tip
    win.close()


def test_refresh_actions_never_globs_for_the_file_count(qapp, tmp_path, monkeypatch):
    """The file count in the two-line io info must come from the file rail's already-built
    manifest, never a fresh directory scan on every settings tick."""
    import respmech.ui.screens.run_screen as run_screen_mod
    win = _win(tmp_path); rn = win.run_screen
    calls = []
    monkeypatch.setattr(run_screen_mod, "matching_files",
                        lambda *a, **k: calls.append(a) or [])
    for i in range(5):
        rn.state.settings.output.folder = str(tmp_path / f"out{i}")
        rn.refresh_actions()
    assert calls == []
    win.close()


def test_progress_bar_hidden_when_idle_and_after_a_run_finishes(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    assert rn.progress.isHidden()
    assert rn.progress.isTextVisible() is False
    rn._set_running(True)
    assert not rn.progress.isHidden()
    rn._last_write = True
    rn._only_files = None
    rn._fatal_msg = None
    rn._on_finished(_result_with_failure())
    assert rn.progress.isHidden()               # hidden again once the run is over
    assert rn.progress.isTextVisible() is False
    win.close()


def test_log_placeholder_explains_dry_run_vs_run(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    ph = rn.log.placeholderText().lower()
    assert "dry run" in ph and "run batch" in ph
    win.close()


def test_table_empty_hint_toggles_with_a_real_result(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    assert not rn._table_empty_hint.isHidden()
    rn._fill_table(SimpleNamespace(average_table=pd.DataFrame({"vt": [0.5, 0.6]})))
    assert rn._table_empty_hint.isHidden()
    win.close()


def test_run_report_path_resolves_the_written_report(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    assert rn._run_report_path() is None
    (tmp_path / "run-report.txt").write_text("hello")
    assert rn._run_report_path() == str(tmp_path / "run-report.txt")
    win.close()


def test_run_report_path_prefers_a_partial_report_name(qapp, tmp_path):
    """A05: a subset write's report is named ``run-report (partial, <timestamp>).txt`` so
    it never overwrites the full run's own record — the button must still find it."""
    win = _win(tmp_path); rn = win.run_screen
    (tmp_path / "run-report (partial, 20260101-000000).txt").write_text("partial")
    assert rn._run_report_path() == str(tmp_path / "run-report (partial, 20260101-000000).txt")
    win.close()


def test_show_run_report_button_disabled_until_a_run_writes_one(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    assert not rn.btn_show_report.isEnabled()
    (tmp_path / "run-report.txt").write_text("Read 2 files.\nWrote 3 files.\n")
    rn._last_write = True
    rn._only_files = None
    rn._fatal_msg = None
    rn._on_finished(_result_with_failure())
    assert rn.btn_show_report.isEnabled()
    win.close()


def test_show_run_report_opens_the_written_report_in_the_text_viewer(qapp, tmp_path, monkeypatch):
    import respmech.ui.screens.run_screen as run_screen_mod
    win = _win(tmp_path); rn = win.run_screen
    (tmp_path / "run-report.txt").write_text("Wrote 3 files.")
    seen = {}

    class _StubDialog:
        def __init__(self, title, text, parent=None, intro=None):
            seen["title"] = title; seen["text"] = text
        def show(self):
            pass
        def raise_(self):
            pass

    monkeypatch.setattr(run_screen_mod, "TextViewerDialog", _StubDialog)
    rn._last_report_path = str(tmp_path / "run-report.txt")   # as a completed run would set it
    rn._show_run_report()
    assert seen["title"] == "Run report"
    assert "Wrote 3 files." in seen["text"]
    win.close()


def test_show_run_report_is_a_noop_before_any_report_exists(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    rn._show_run_report()                        # must not raise
    assert "no run report" in rn.status.text().lower()
    win.close()


def test_finish_message_names_the_primary_deliverable(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    rn._last_write = True
    rn._only_files = None
    rn._fatal_msg = None
    result = SimpleNamespace(
        files={"synth_case_A.csv": SimpleNamespace(error=None, breaths_table=pd.DataFrame({"vt": [0.5, 0.6]}))},
        failed_files={}, ok_files={"synth_case_A.csv": SimpleNamespace(
            error=None, breaths_table=pd.DataFrame({"vt": [0.5, 0.6]}))},
        average_table=pd.DataFrame({"vt": [0.5, 0.6]}))
    rn._on_finished(result)
    log = rn.log.toPlainText()
    assert "Average breathdata.xlsx (2 rows) is the file to open first." in log
    win.close()


# ---------------------------------------------------------------------------
# B03 self-review follow-ups — the drawer's own collapse/expand, the shared
# rail's caveat preservation, and the run-report path's snapshot/reset fixes.
# ---------------------------------------------------------------------------
def test_results_section_starts_collapsed_and_toggles(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    assert not rn.btn_toggle_results.isChecked()
    assert rn._results_section.isHidden()
    assert rn.btn_toggle_results.text() == "Run & results ▸"
    rn.btn_toggle_results.setChecked(True)
    assert not rn._results_section.isHidden()
    assert rn.btn_toggle_results.text() == "Run & results ▾"
    rn.btn_toggle_results.setChecked(False)
    assert rn._results_section.isHidden()
    win.close()


def test_starting_a_run_auto_expands_the_results_section(qapp, tmp_path):
    win = _win(tmp_path); rn = win.run_screen
    assert rn._results_section.isHidden()
    rn._set_running(True)
    assert not rn._results_section.isHidden()
    assert rn.btn_toggle_results.isChecked()
    win.close()


def test_run_report_path_prefers_the_newer_of_two_reports(qapp, tmp_path):
    """A05: a subset write's report never overwrites the full run's own run-report.txt, so
    both can legitimately exist at once — the newer one (by mtime) is the one to show."""
    import os as _os
    win = _win(tmp_path); rn = win.run_screen
    old = tmp_path / "run-report.txt"; old.write_text("full run")
    new = tmp_path / "run-report (partial, 20260101-000000).txt"; new.write_text("subset")
    now = _os.path.getmtime(old)
    _os.utime(old, (now - 100, now - 100))       # make the full-run report demonstrably older
    assert rn._run_report_path() == str(new)
    win.close()


def test_run_report_path_uses_the_runs_own_snapshot_not_live_settings(qapp, tmp_path):
    """Self-review finding: a user can change the output folder in Setup the instant a run
    ends. 'Show run report' must keep pointing at the folder the JUST-FINISHED run actually
    wrote to, never wherever Setup happens to point at when the button is clicked."""
    import copy
    win = _win(tmp_path); rn = win.run_screen
    (tmp_path / "run-report.txt").write_text("the real one")
    rn._run_settings_snapshot = copy.deepcopy(win.state.settings)    # frozen, as _start() does
    other = tmp_path / "elsewhere"; other.mkdir()
    (other / "run-report.txt").write_text("a different study's report")
    win.state.settings.output.folder = str(other)     # live edit, after the snapshot was taken
    assert rn._run_report_path() == str(tmp_path / "run-report.txt")
    win.close()


def test_dry_run_finishing_does_not_leave_a_stale_report_button_enabled(qapp, tmp_path):
    """A run that starts after a previous run enabled the button, but is itself still in
    flight (or fails before writing), must not go on offering the PREVIOUS run's report."""
    win = _win(tmp_path); rn = win.run_screen
    (tmp_path / "run-report.txt").write_text("from an earlier run")
    rn._last_write = True; rn._only_files = None; rn._fatal_msg = None
    rn._on_finished(_result_with_failure())
    assert rn.btn_show_report.isEnabled() and rn._last_report_path
    rn._start(write=False)                        # a fresh dry run begins; resets happen
    assert not rn.btn_show_report.isEnabled()      # before the worker thread is even created
    assert rn._last_report_path is None
    # A REAL QThread/BatchWorker is now running (write=False skips the overwrite guard that
    # would otherwise short-circuit _start early). It must be let finish and its queued
    # `finished` signal delivered BEFORE the window closes — closing (and so destroying
    # `rn`) while that signal is still queued left it undelivered until a LATER, unrelated
    # test's own processEvents() call finally reached it, segfaulting there (found via a
    # reproducible crash in test_section_flow.py two full-suite runs in a row).
    _pump_until_thread_done(qapp, rn)
    win.close()


def test_a_run_preserves_the_shared_rails_manifest_caveats(qapp, tmp_path):
    """Self-review finding: _fill_file_rail's manifest is deliberately caveat-free (it only
    knows the resolved run file list), but the rail is now SHARED with Preview & QC, whose
    own build_manifest scan is the thing that actually knows about column/frequency
    outliers. A run finishing must not silently erase those warnings from the one rail both
    screens show."""
    win = _win(tmp_path); rn = win.run_screen
    rn.file_rail.set_caveat("synth_case_A.csv", "detected 500 Hz sampling — settings say 1000 Hz")
    rn._fill_file_rail(_result_with_failure())
    assert rn.file_rail.entry("synth_case_A.csv").caveat == \
        "detected 500 Hz sampling — settings say 1000 Hz"
    win.close()


def test_standalone_run_screen_without_a_shared_rail_still_works(qapp, tmp_path):
    """RunScreen(state) with no file_rail argument (several existing tests construct it
    this way directly) must still get a usable, private rail rather than fail."""
    from respmech.ui.screens.run_screen import RunScreen
    from respmech.ui.state import AppState
    rn = RunScreen(AppState(synth_settings(tmp_path)))
    assert rn.file_rail is not None
    assert rn.read_info.toolTip()          # io-info still renders with the private rail
    rn.deleteLater()


def test_install_run_drawer_actually_embeds_the_widget(qapp, tmp_path):
    win = _win(tmp_path); pv = win.preview_screen
    assert pv._run_drawer is win.run_screen
    assert pv.layout().indexOf(win.run_screen) != -1
    win.close()


def test_drawer_summary_reads_sensibly_with_folder_and_output_unset(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    win.state.settings.input.folder = ""       # AppState()'s own default is the placeholder
    win.state.settings.output.folder = ""      # "input"/"output", not an empty string
    win.run_screen.refresh_actions()
    text = win.run_screen._drawer_summary.toolTip()
    assert "(input not set)" in text and "(output not set)" in text
    assert "  " not in text                    # no accidental double space from empty bits
    win.close()

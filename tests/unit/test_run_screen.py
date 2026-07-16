"""Run screen — the tune-to-run loop & batch control: re-run only the failed files (P18),
"Process & write this file" from Preview (P19), and the per-file results table with
drill-back into Preview (P20). Previously test_review_wave5.py; the run-screen tests from
wave 2 (dry-run plan P5, overwrite guard, open-folder P7) are appended below."""
from types import SimpleNamespace

import pandas as pd
import pytest

from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings

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
    from respmech.ui.main_window import MainWindow
    from respmech.ui.validation import matching_files
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); rn = win.run_screen
    files = matching_files(s.input.folder, s.input.files)
    s.output.data.save_processed = False
    assert not any("Processed data" in n for n in rn._planned_outputs(files))
    s.output.data.save_processed = True
    assert any("Processed data" in n for n in rn._planned_outputs(files))
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

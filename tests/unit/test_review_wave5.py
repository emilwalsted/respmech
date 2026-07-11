"""Wave 5 of the full-app review: the tune-to-run loop & batch control.

P18 re-run only the files that failed (the BatchWorker already accepts a file subset);
P19 "Process & write this file" from Preview, so one tuned file can be produced without
re-running the whole batch; P20 a per-file Run-results table whose rows drill back into
Preview & QC. All GUI wiring — no core, no golden exposure.
"""
import os
from types import SimpleNamespace

import pandas as pd
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


def _win(tmp):
    from respmech.settingsio.migrate import migrate_dict
    from respmech.ui.main_window import MainWindow
    legacy = {"input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4],
                                 "columns_entropy": [10, 11, 12]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": False, "remove_noise": False}},
              "output": {"outputfolder": str(tmp), "data": {}}}
    s, _ = migrate_dict(legacy); s.input.folder = INPUT; s.output.folder = str(tmp)
    return MainWindow(AppState(s))


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
    assert rows["synth_case_A.csv"] == ("ok", "2")
    assert rows["synth_case_B.csv"][0] == "failed"
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

"""Integration tests for the reactive auto-run on the Preview & tuning screen.

Unlike the other GUI tests (which call render/compute methods directly and never
spin the event loop), these DRIVE the real Qt event loop so the ``QTimer.singleShot``
auto-run triggers fire and the worker threads actually run. That is the only way to
verify that selecting a file dispatches every runnable job concurrently, shows and
then clears each panel's spinner, supersedes stale jobs cleanly when the file
changes mid-run, and shuts the threads down without leaking.
"""
import os
import time

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


def _settings(outdir, noise=False):
    legacy = {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": 1000},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": [10, 11, 12]}},
        "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                     "avgresamplingobs": 300},
                       "emg": {"remove_ecg": True, "remove_noise": noise}},
        "output": {"outputfolder": outdir,
                   "data": {"saveaveragedata": True, "savebreathbybreathdata": True}},
    }
    s, _ = migrate_dict(legacy)
    if noise:
        s.processing.emg.noise.enabled = True
        s.processing.emg.noise.reference_file = "synth_case_A.csv"
        s.processing.emg.noise.use_expiration = False
        s.processing.emg.noise.reference_intervals = [[1.0, 5.0]]
        s.processing.emg.noise.auto_prop = True
    return s


def _pump_until(qapp, predicate, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    qapp.processEvents()
    return predicate()


def test_selecting_a_file_autoruns_all_panels(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=True)))
    pv = win.preview_screen
    pv._refresh_files()
    assert pv.file_combo.count() == 2
    # select a file -> the singleShot auto-run must dispatch jobs once the loop turns
    pv.file_combo.setCurrentIndex(1)
    assert _pump_until(qapp, lambda: bool(pv._jobs) or bool(pv.busy_panels()), 10), \
        "no jobs auto-started on file selection"
    # every runnable job finishes and every spinner clears
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60), \
        "reactive jobs did not all finish"
    assert pv.busy_panels() == set(), "a spinner was left running"
    # the auto panels are populated: mechanics channels, EMG result, and (auto) fidelity
    assert len(pv._channel_plots) == 5
    assert pv._emg_all is not None and len(pv._emg_all["conditioned"]) == 3
    assert len(pv.fidelity_canvas.figure.axes) == 1     # fidelity frontier drawn by the noise job
    # the test run is MANUAL now -> its table stays empty until the button is clicked
    assert pv.table.rowCount() == 0
    assert "failed" not in pv.status.text().lower()     # ended on a success message
    win.close()


def test_autorun_without_noise_skips_fidelity_but_runs_the_rest(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=False)))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining and pv._previewed_file, 60)
    assert pv.busy_panels() == set()
    assert len(pv._channel_plots) == 5
    assert pv._emg_all is not None            # EMG staged (ECG-removed, no noise)
    assert pv.table.rowCount() == 0          # test run is manual
    assert len(pv.fidelity_canvas.figure.axes) == 0   # no noise reference -> no fidelity job
    win.close()


def test_switching_file_mid_run_supersedes_cleanly(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=True)))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)          # start jobs for synth_case_B
    qapp.processEvents()                       # fire the singleShot -> launch B's jobs
    pv.file_combo.setCurrentIndex(0)          # immediately switch to synth_case_A
    # everything drains and clears with no crash
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)
    assert pv.busy_panels() == set()
    # the last-selected file (A) is the one whose mechanics preview won
    assert pv._previewed_file == "synth_case_A.csv"
    win.close()


def test_shutdown_joins_running_jobs(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=True)))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)
    qapp.processEvents()                       # launch the jobs
    pv.shutdown()                              # join them while (likely) still running
    assert not pv._jobs and not pv._draining   # registry cleared, no leaked threads


def test_shutdown_parks_a_thread_it_cannot_join(qapp, tmp_path):
    """The crash-safety guarantee: a worker that will not stop within the join
    budget is PARKED (kept referenced), never GC'd while running (which would
    abort the process). Exercised deterministically with an uncancellable slow
    job and a tiny budget."""
    import time as _t
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import FnWorker
    from respmech.ui.screens import preview_screen as ps
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    before = len(ps._ORPHANED_THREADS)
    worker = FnWorker(lambda: (_t.sleep(1.0), None)[1])   # runs past the budget, ignores cancel
    pv._tokens["mech"] += 1
    pv._launch("mech", pv._tokens["mech"], worker)
    qapp.processEvents()                       # let the thread actually start the slow fn
    pv.shutdown(wait_ms=50)                     # cannot join in 50ms -> must park, not clear-into-GC
    assert not pv._jobs and not pv._draining
    assert len(ps._ORPHANED_THREADS) == before + 1
    ps._ORPHANED_THREADS[-1][0].wait(3000)     # let the parked thread finish cleanly
    win.close()


def test_gated_out_reschedule_does_not_supersede_running_job(qapp, tmp_path):
    """Finding: bumping the token before the gate would let a gated-out
    reschedule discard a still-running job's result. The token must only move
    when a job actually launches."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)
    qapp.processEvents()                           # let the auto-run dispatch its jobs
    pv._schedule("batch")                          # the test run is manual -> start it explicitly
    assert _pump_until(qapp, lambda: "batch" in pv._jobs, 10)
    tok = pv._tokens["batch"]
    pv._settings_ok = lambda: (False, "invalid")   # force the batch gate to fail
    pv._schedule("batch")                          # gated out -> must NOT bump the token
    assert pv._tokens["batch"] == tok              # running job's result stays valid
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)
    win.close()


def test_autoprop_writes_chosen_prop_back_to_settings(qapp, tmp_path):
    """Finding: the batch's auto-selected suppression lives only in the report;
    it must be written into settings so the spinbox and re-conditioned EMG views
    use the value the batch actually chose (not the stale default)."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = _settings(str(tmp_path), noise=True)
    s.processing.emg.noise.prop_decrease = 0.05    # deliberately-not-chosen default
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_B.csv")
    result = run_batch(s, only_files=["synth_case_B.csv"])
    chosen = float(result.noise_report["prop_decrease"])
    pv._on_batch_result(result)                    # writes chosen prop, re-dispatches EMG jobs
    assert abs(s.processing.emg.noise.prop_decrease - chosen) < 1e-9
    assert abs(pv.noise_prop.value() - chosen) < 1e-9
    pv.shutdown()


def test_settings_change_cancels_inflight_jobs(qapp, tmp_path):
    """A settings change aborts every in-flight panel-refresh job (cancel + token
    invalidation), then re-dispatches — nothing is left running or leaked."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=True)))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)
    assert _pump_until(qapp, lambda: bool(pv._jobs), 10)     # jobs in flight
    toks = dict(pv._tokens)
    pv.sync_from_settings()                                  # settings changed
    assert not pv._jobs                                      # current jobs were aborted...
    assert all(pv._tokens[k] == toks[k] + 1 for k in toks)  # ...and their tokens invalidated
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)
    assert pv.busy_panels() == set()
    win.close()


def test_refresh_recomputes_auto_panels_but_not_the_test_run(qapp, tmp_path):
    """The 'Refresh' button re-dispatches only the auto kinds — never the test run."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=True)))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining and pv._previewed_file, 60)
    launched = []
    orig = pv._schedule
    pv._schedule = lambda k: (launched.append(k), orig(k))[1]
    try:
        pv._refresh_all()
        qapp.processEvents()                                # fire the deferred auto-run
    finally:
        pv._schedule = orig
    assert "batch" not in launched                          # never the test run
    assert {"mech", "emg_all", "emg_detail", "noise"} & set(launched)
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)
    win.close()


def test_noise_job_draws_fidelity_and_adopts_prop(qapp, tmp_path):
    """The auto noise job (decoupled from the test run) produces the fidelity frontier
    and writes the auto-selected suppression back into settings + the spinbox."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_noise_fidelity
    s = _settings(str(tmp_path), noise=True)
    s.processing.emg.noise.prop_decrease = 0.05        # deliberately-not-chosen default
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    report = stage_noise_fidelity(s)                   # what the noise worker returns
    chosen = float(report["prop_decrease"])
    pv._on_noise_result(report)
    assert len(pv.fidelity_canvas.figure.axes) == 1    # frontier drawn (no test run needed)
    assert abs(s.processing.emg.noise.prop_decrease - chosen) < 1e-9
    assert abs(pv.noise_prop.value() - chosen) < 1e-9
    pv.shutdown()

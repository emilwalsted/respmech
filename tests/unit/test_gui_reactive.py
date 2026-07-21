"""Integration tests for the reactive auto-run on the Preview & tuning screen.

Unlike the other GUI tests (which call render/compute methods directly and never
spin the event loop), these DRIVE the real Qt event loop so the ``QTimer.singleShot``
auto-run triggers fire and the worker threads actually run. That is the only way to
verify that selecting a file dispatches every runnable job concurrently, shows and
then clears each panel's spinner, supersedes stale jobs cleanly when the file
changes mid-run, and shuts the threads down without leaking.
"""
import os

# These heavy full-autorun / supersede tests used to be xfail'd on headless Windows CI
# because the drain sometimes exceeded 60s. Root cause (diagnosed via a multi-agent
# investigation): the reactive dispatch started up to 5 GIL-bound worker QThreads at once
# and _on_job_done blocked the GUI thread on thread.wait(2000), starving the event loop that
# delivers the workers' `finished` signals — a high-variance stall, worst on Windows (coarse
# GIL hand-off + offscreen has no native message pump). Fixed at the source: the Preview now
# caps concurrency at _MAX_ACTIVE and tears threads down non-blockingly (see preview_screen.py
# _pump_pool / _release_thread), so the drain is fast and deterministic on all platforms — the
# xfail is gone.

from respmech.ui.state import AppState  # noqa: E402


from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


pytestmark = requires_synth()




def _settings(outdir, noise=False):
    return synth_settings(outdir, remove_ecg=True, noise=noise, data_out=_DATA_OUT)


def _pump_until(qapp, predicate, timeout=60.0):
    """Spin a REAL Qt event loop until ``predicate`` holds (or ``timeout`` seconds pass).

    Uses ``QEventLoop.exec()`` rather than a hand-rolled ``processEvents()``/``sleep`` pump:
    a manual pump does not reliably deliver cross-thread *queued* signals on every platform
    (headless Windows delivers the same-thread ``singleShot`` dispatch but not the workers'
    cross-thread ``finished`` — so jobs looked stuck), whereas a real event loop delivers
    them exactly as the running app does. This makes the async-orchestration tests exercise
    the true signal-delivery path and behave identically on macOS and Windows."""
    from PySide6.QtCore import QEventLoop, QTimer, QElapsedTimer
    if predicate():
        return True
    loop = QEventLoop()
    clock = QElapsedTimer()
    clock.start()
    state = {"ok": False}
    timer = QTimer()
    timer.setInterval(10)

    def _tick():
        if predicate():
            state["ok"] = True
            loop.quit()
        elif clock.elapsed() > timeout * 1000:
            loop.quit()

    timer.timeout.connect(_tick)
    timer.start()
    loop.exec()
    timer.stop()
    return state["ok"] or predicate()


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
    # the mechanics test run is automatic now -> the table + Campbell fill on their own
    assert pv.table.rowCount() > 0
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
    assert pv.table.rowCount() > 0           # mechanics test run is automatic
    assert len(pv.fidelity_canvas.figure.axes) == 0   # no noise reference -> no fidelity job
    win.close()


def test_switching_file_mid_run_supersedes_cleanly(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=True)))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)          # request jobs for synth_case_B (debounced)
    pv.file_combo.setCurrentIndex(0)          # switch to synth_case_A before the debounce fires
    # the debounce coalesces the rapid A/B switch into ONE run for the last-selected file;
    # everything drains and clears with no crash and A's mechanics preview is the one that won
    assert _pump_until(qapp, lambda: pv._previewed_file == "synth_case_A.csv"
                       and not pv._jobs and not pv._draining, 60)
    assert pv.busy_panels() == set()
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
    """Finding: bumping the token before the gate would let a gated-out reschedule discard a
    still-running job's result. The token must only move when a job actually launches.

    This is about a gate on WHAT IS SHOWN — no EMG channels, no ECG panel. The job that is
    running is still computing something the current settings can produce, so its result
    stays valid and must not be thrown away."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)
    qapp.processEvents()                           # auto-run dispatches jobs (incl. the batch)
    assert _pump_until(qapp, lambda: "batch" in pv._jobs, 10)
    tok = pv._tokens["batch"]
    pv.state.settings.input.channels.emg = []      # the ECG panel has nothing to show
    pv._schedule("ecg")                            # gated out -> must NOT bump any token
    assert pv._tokens["batch"] == tok              # the running job's result stays valid
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)
    win.close()


def test_settings_going_invalid_DOES_supersede_a_running_job(qapp, tmp_path):
    """The opposite case, and the distinction is the point. A job launched while the mapping
    was valid is computing a result for a mapping the user has since abandoned; letting it
    land repaints the panels we just blanked, under a status claiming success."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(1)
    qapp.processEvents()
    assert _pump_until(qapp, lambda: "batch" in pv._jobs, 10)
    tok = pv._tokens["batch"]
    pv.state.settings.input.channels.flow = None   # the mapping is abandoned
    pv._schedule("batch")
    assert pv._tokens["batch"] != tok, "the obsolete job's result would still have landed"
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


def test_refresh_recomputes_all_auto_panels(qapp, tmp_path):
    """The 'Refresh' button re-dispatches every auto kind — including the (now
    automatic, mechanics-only) test run."""
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
        # wait out the debounce, then confirm every auto kind was dispatched
        _pump_until(qapp, lambda: {"mech", "batch", "emg_all", "emg_detail", "noise"}
                    <= set(launched), 5)
    finally:
        pv._schedule = orig
    assert {"mech", "batch", "emg_all", "emg_detail", "noise"} <= set(launched)
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


def test_edits_debounce_into_one_recompute(qapp, tmp_path):
    """Rapid settings changes coalesce: many _request_autorun calls within the debounce
    window fire the recompute ONCE (each kind scheduled once), not once per edit."""
    from collections import Counter
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentIndex(1)
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining and pv._previewed_file, 60)
    calls = []
    orig = pv._schedule
    pv._schedule = lambda k: (calls.append(k), orig(k))[1]
    try:
        for _ in range(8):
            pv._request_autorun()                 # 8 rapid edits within the debounce window
        assert calls == []                         # debounced: nothing dispatched synchronously
        _pump_until(qapp, lambda: bool(calls), 5)  # let the single debounced run fire
    finally:
        pv._schedule = orig
    assert calls and max(Counter(calls).values()) == 1   # each kind scheduled exactly once
    _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)
    win.close()


def test_settings_change_scopes_recompute_to_affected_panels(qapp, tmp_path):
    """Dependency-scoped recompute: an EMG-only edit re-runs only the EMG/noise panels
    (not the mechanics preview / test run); a mechanics-only edit does the reverse."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path), noise=True)))
    pv = win.preview_screen
    import copy as _copy
    pv._refresh_files(); pv.file_combo.setCurrentIndex(1)
    assert _pump_until(qapp, lambda: not pv._jobs and not pv._draining and pv._previewed_file, 60)
    # establish a clean diff baseline without kicking off another recompute
    pv._last_synced_settings = _copy.deepcopy(pv.state.settings)
    pv._pending_kinds.clear(); pv._autorun_timer.stop()

    def _capture(mutate):
        # capture the kinds the debounced dispatch SCOPES to (the dependency-scope decision),
        # not the downstream noise-adopt cascade (a noise-profile change legitimately
        # re-conditions EMG afterwards via _apply_noise_report)
        captured = {}
        orig = pv._schedule_all
        pv._schedule_all = lambda kinds=None: (captured.update(k=set(kinds or _AUTO_KINDS)),
                                               orig(kinds))[1]
        try:
            mutate(pv.state.settings)
            pv.sync_from_settings()
            _pump_until(qapp, lambda: "k" in captured, 5)
        finally:
            pv._schedule_all = orig
        _pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)
        return captured.get("k", set())

    emg = _capture(lambda s: setattr(s.processing.emg, "rms_window_s",
                                     s.processing.emg.rms_window_s + 0.01))
    assert {"emg_all", "emg_detail"} & emg          # EMG panels recompute
    assert "mech" not in emg and "batch" not in emg  # mechanics panels do NOT

    # a volume-drift edit is genuinely mechanics-only (it does not change the flow-based EMG
    # reference masks), so the EMG conditioning panels are left alone
    mech = _capture(lambda s: setattr(s.processing.volume, "correct_drift",
                                      not s.processing.volume.correct_drift))
    assert {"mech", "batch"} <= mech                 # mechanics panels recompute
    assert "emg_all" not in mech and "emg_detail" not in mech  # EMG panels do NOT
    win.close()

"""Write the diagnostic figures in a separate process, falling back to in-process.

WHY. ``write_batch`` runs on a worker thread in the GUI (ui/workers.py BatchWorker), and
figure writing is the one part of it that uses matplotlib. The GUI thread uses matplotlib too
— the Preview screen embeds three ``FigureCanvasQTAgg`` canvases — so the two threads share
matplotlib's global state, and matplotlib is not thread-safe. Emil's traced symptom was Qt
printing ``QBasicTimer::start: Timers cannot be started from another thread`` repeatedly with
a Python stack ending inside ``plots._emg_overview`` on the worker thread.

A spawned child is genuinely isolated: it imports ``respmech.core.plots``, which pulls in
matplotlib but no Qt at all, so there is no shared state and no cross-thread affinity left to
violate. Measured cost on the production set: a BatchResult pickles to ~22 MB per file in
under 10 ms, which is negligible beside the figure writing itself.

SAFETY. This can never make things worse than doing the work in-process:

  * A PACKAGED app (briefcase bundle) whose ``sys.executable`` is the GUI binary would, on a
    ``spawn``, re-launch the whole app instead of running a Python worker — and it does so
    TWICE (the resource tracker and the worker). ``multiprocessing.freeze_support()`` does NOT
    prevent this: a briefcase bundle is not ``sys.frozen``, so ``freeze_support()`` is a no-op.
    So we do not probe at all unless ``sys.executable`` is a real Python interpreter; in a
    bundle we go straight to the in-process path (no extra windows, no probe-timeout hang).
    In a normal dev/venv install ``sys.executable`` IS python, so the child isolation is used.
  * The child (when used) is still probed once with a trivial task; a caching verdict means a
    spawn failure is paid for once per process, not per batch.
  * Every failure mode — no probe, spawn error, pickling error, timeout, child crash —
    falls back to calling ``plots.write_figures`` directly, which is exactly today's
    behaviour.
"""
from __future__ import annotations

import os
import sys

# One trivial spawn decides whether this environment can run children at all. None = not yet
# probed; True/False = the answer for the rest of the process.
_CAN_SPAWN: bool | None = None

_PROBE_TIMEOUT_S = 30.0
# Generous: a large test with every diagnostic on writes a lot of figures. A hung child costs
# this much before the in-process fallback runs, so it must be well above any real run.
_WORK_TIMEOUT_S = 900.0

_DISABLE_ENV = "RESPMECH_NO_FIGURE_SUBPROCESS"


def _probe_ok() -> str:
    """Trivial child payload — must be module-level so 'spawn' can pickle it by reference."""
    return "ok"


def _run_write_figures(result, settings, outputfolder: str):
    """Child-side entry point: import the plotting stack fresh and do the work.

    Runs in a process that has never imported Qt, which is the whole point.
    """
    from respmech.core import plots
    return plots.write_figures(result, settings, outputfolder)


def _in_process(result, settings, outputfolder: str):
    from respmech.core import plots
    return plots.write_figures(result, settings, outputfolder)


def _executor():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    # "spawn" explicitly: "fork" would clone a process that already has Qt and a live worker
    # thread, which is its own class of crash.
    return ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))


def _spawn_relaunches_the_app() -> bool:
    """True when a multiprocessing ``spawn`` would re-launch THIS app instead of running a Python
    child — i.e. a packaged build. That is the shipped v2.3.0 bug: on every batch's figure step
    the probe opened two extra GUI windows (the resource tracker and the worker) and then hung
    for the probe timeout. ``multiprocessing.freeze_support()`` does not help (a briefcase bundle
    is not ``sys.frozen``). So when this is true we never spawn/probe — figures go in-process.

    Packaged is detected three ways: a real frozen build (``sys.frozen``, PyInstaller/py2exe);
    ``sys.executable`` not being a Python interpreter (the briefcase binary is named after the
    app); or ``sys.executable`` living inside a macOS ``.app`` bundle. Any false positive only
    costs the (safe, identical) in-process path, so we err toward not spawning."""
    if getattr(sys, "frozen", False):
        return True
    exe = (sys.executable or "").replace("\\", "/")         # normalise for a Windows path anywhere
    base = os.path.basename(exe).lower()
    if not (base.startswith("python") or base.startswith("pypy")):
        return True
    if ".app/contents/" in exe.lower():
        return True
    return False


def _can_spawn() -> bool:
    global _CAN_SPAWN
    if _CAN_SPAWN is None:
        if os.environ.get(_DISABLE_ENV) or _spawn_relaunches_the_app():
            _CAN_SPAWN = False
            return _CAN_SPAWN
        try:
            with _executor() as ex:
                _CAN_SPAWN = ex.submit(_probe_ok).result(timeout=_PROBE_TIMEOUT_S) == "ok"
        except BaseException:      # noqa: BLE001 - any failure at all means "do not use it"
            _CAN_SPAWN = False
    return _CAN_SPAWN


def write_figures(result, settings, outputfolder: str, *, on_fallback=None):
    """``plots.write_figures``, in a child process when that works, else here.

    ``on_fallback(reason)`` is called when the in-process path is taken, so a caller can
    record why in the run report rather than have it happen invisibly.
    """
    if not _can_spawn():
        if on_fallback:
            on_fallback("figure subprocess unavailable; wrote figures in-process")
        return _in_process(result, settings, outputfolder)

    try:
        with _executor() as ex:
            fut = ex.submit(_run_write_figures, result, settings, outputfolder)
            return fut.result(timeout=_WORK_TIMEOUT_S)
    except BaseException as e:     # noqa: BLE001 - never let isolation break a run
        # One failure is not necessarily permanent (a timeout may just be a huge test), so the
        # probe verdict is left alone; the next batch tries the child again.
        if on_fallback:
            on_fallback(f"figure subprocess failed ({type(e).__name__}: {e}); "
                        "wrote figures in-process")
        return _in_process(result, settings, outputfolder)

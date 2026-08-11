"""Point 6 durability (claude-ops ticket 20260811-1735): the invariant "closing a plot
owner releases its menus" used to live in ONE place — MainWindow.closeEvent's explicit
orchestration (screen.shutdown() calls + channel_summary.close_plots()). Qt never
delivers closeEvent to a child widget when its PARENT window closes, so any composition
that builds PreviewScreen/RunScreen/ColumnStack WITHOUT a MainWindow around it bypassed
that orchestration entirely — real today, not hypothetical: the point 6 investigation
counted 15 tests constructing PreviewScreen standalone, 6 RunScreen, 17 ColumnStack.

Each of those widgets now has its own closeEvent that runs its EXISTING cleanup function,
so close() works whether or not a MainWindow is the one calling it. This file is Layer 2:
the invariant as a NAMED test per widget type, so a future refactor that silently drops a
closeEvent override (or the shutdown()/close_plots() call inside it) fails one of these
the same day, instead of resurfacing as an unattributable multi-hour CI wall or sandbox
OOM a year from now — which is exactly how point 6 itself was originally found.
"""
import gc
import os

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu

from respmech.ui.column_stack import ColumnStack
from respmech.ui.screens.preview_screen import PreviewScreen
from respmech.ui.screens.run_screen import RunScreen
from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

# MainWindow's own initial Setup refresh reads the real synth CSV to build the channel
# summary's ColumnStack (see test_preview_screen.py's own point 6 tests), and the
# RunScreen guard below runs a real (small, synthetic) batch write — every test here
# needs the committed synthetic input.
pytestmark = requires_synth()


def _menu_census():
    """Count of parentless (top-level) ``QMenu``s alive right now. Point 6's own
    diagnostic (``RESPMECH_NET_CENSUS``, conftest.py) established that pyqtgraph's
    ``PlotItem``/``ViewBox`` build their context menus with NO Qt parent, so a leaked one
    shows up here as its own top-level widget — the exact census the whole investigation
    was built on. Every assertion below reads the DELTA around one construct/close cycle,
    never the absolute count: earlier tests in the same session leave their own
    closed-but-undeleted windows (and menus) alive by design (conftest.py's
    ``_close_top_level_windows`` never deletes a closed top-level window — deleting one
    segfaults nondeterministically on Python 3.11), so an absolute count would be hostage
    to test order and flake."""
    app = QApplication.instance()
    return sum(1 for w in app.topLevelWidgets() if isinstance(w, QMenu))


def _settle():
    """Reap freshly-dropped C++ wrappers without spinning the event loop: a gc pass plus a
    DeferredDelete-only dispatch, mirroring conftest.py's own ``_close_top_level_windows``
    net exactly — a bare ``processEvents()`` would also fire pending timers, which is
    precisely what that net exists to avoid."""
    gc.collect()
    app = QApplication.instance()
    try:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    except Exception:                       # pragma: no cover - defensive, matches conftest
        pass


def _matrix(n=200, cols=4):
    rng = np.random.RandomState(0)
    return rng.randn(n, cols), [f"ch{i}" for i in range(cols)]


def test_previewscreen_closeevent_releases_plot_menus_without_a_mainwindow(qapp, tmp_path):
    """The mechanics stack's PlotItems, closed via close()'s new self-cleanup rather than
    via a parent MainWindow's orchestrated shutdown() call."""
    from respmech.ui.workers import stage_mechanics_preview

    before = _menu_census()

    s = synth_settings(str(tmp_path))
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))

    plot_items = list(pv.plots.ci.items.keys())
    assert plot_items and all(p.ctrlMenu is not None for p in plot_items), (
        "the render must have built real PlotItems with menus, or this test proves nothing"
    )
    assert _menu_census() > before, (
        "the render should have added new top-level QMenus, or the close-side assertion "
        "below would pass vacuously"
    )

    pv.close()
    _settle()

    assert _menu_census() <= before, (
        "PreviewScreen.closeEvent must release every menu its own render built, exactly as "
        "MainWindow.closeEvent's orchestrated shutdown() call already does -- the invariant "
        "is NO NET GROWTH, not exact equality: gc.collect() could also reclaim an unrelated "
        "earlier test's leftover menus and push the count below `before`, which is fine"
    )
    assert all(p.ctrlMenu is None for p in plot_items), (
        "closeEvent must call shutdown(), which drops ctrlMenu via plot_perf.close_plots()"
    )
    pv.close()                              # idempotent: a second close must not raise


def _pump_until_write_done(qapp, rn, timeout=60.0):
    """``test_run_screen.py``'s own ``_pump_until_thread_done``, adapted to
    ``_write_thread`` instead of ``_thread``. Spins a REAL Qt event loop rather than
    blocking on ``QThread.wait()`` directly: measured directly in THIS sandbox that a
    plain blocking ``wait()`` on the calling (main) thread does not reliably observe the
    write finishing at all — the worker's cross-thread queued ``finished`` signal needs
    the main thread's event loop actually spinning to be delivered and processed, exactly
    the reason ``test_run_screen.py``'s own helper (and ``test_gui_reactive.py``'s
    ``_pump_until``) exists in the first place. Reusing the same proven pattern here."""
    from PySide6.QtCore import QElapsedTimer, QEventLoop, QTimer
    if rn._write_thread is None:
        return True
    loop = QEventLoop()
    clock = QElapsedTimer(); clock.start()
    state = {"ok": False}
    timer = QTimer(); timer.setInterval(10)

    def _tick():
        if rn._write_thread is None:
            state["ok"] = True
            loop.quit()
        elif clock.elapsed() > timeout * 1000:
            loop.quit()

    timer.timeout.connect(_tick)
    timer.start()
    loop.exec()
    timer.stop()
    return state["ok"] or rn._write_thread is None


def test_runscreen_closeevent_joins_a_running_write_thread_without_a_mainwindow(
        qapp, tmp_path, monkeypatch):
    """RunScreen owns no PlotWidget of its own (grep confirms — its point 6 exposure was
    never QMenu accumulation), so its self-cleanup contract is the OTHER hazard
    shutdown() exists for: a QThread destroyed while still running aborts the process.
    Mirrors test_run_screen.py's own ``test_write_elsewhere_locks_run_buttons…`` write,
    the same real-but-small synthetic write used there, but through a standalone
    RunScreen's own close() rather than a MainWindow's orchestrated one.

    The write is explicitly pumped to completion (``_pump_until_write_done``) BEFORE
    close(), not raced against it. Two things were measured directly in this sandbox
    while building this test, both surprising: (1) a plain blocking
    ``rn._write_thread.wait(60_000)`` on the calling thread genuinely never observed the
    write finish at all — the worker's cross-thread queued ``finished`` signal needs the
    caller's OWN Qt event loop spinning to be delivered (exactly why
    ``test_run_screen.py``'s own ``_pump_until_thread_done`` exists; reused as
    ``_pump_until_write_done`` here); (2) closing immediately after starting the write
    (the FIRST version of this test) let ``shutdown()``'s own 5000 ms ``wait_ms`` time
    out and park a REAL, still-running background thread — and its spawned child
    process — in ``_ORPHANED_THREADS`` for the rest of the test session. That parking is
    correct, INTENTIONAL production behaviour (a slow write must never block the user
    from closing the app), but a unit test asserting on it must not depend on winning
    that race or leave a live thread it created running behind it. This still exercises
    the thing point 6 durability is actually about — RunScreen's OWN bookkeeping reset
    running via the standalone close() path, not only via MainWindow's orchestrated one
    — without either hazard.
    """
    from respmech.core.io.plan import plan_outputs
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path / "original"))
    rn = RunScreen(AppState(s))
    files = [os.path.join(INPUT, "synth_case_A.csv"), os.path.join(INPUT, "synth_case_B.csv")]
    rn._last_result = run_batch(s)
    rn._run_settings_snapshot = s
    rn._last_plan = plan_outputs(s, files)
    rn._last_write = True
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                         lambda *a, **k: str(tmp_path / "elsewhere"))

    rn._write_elsewhere()
    assert rn._write_thread is not None, (
        "the write must actually have started, or this test proves nothing"
    )
    assert _pump_until_write_done(qapp, rn), (
        "the synthetic write did not finish within 60s -- something is genuinely stuck, "
        "not just slow"
    )

    rn.close()

    assert rn._write_thread is None, (
        "RunScreen.closeEvent must reset the write-thread bookkeeping via shutdown(), "
        "exactly as MainWindow.closeEvent already does for the orchestrated case"
    )
    rn.close()                              # idempotent: a second close must not raise


def test_columnstack_closeevent_releases_plot_menus_without_an_owner(qapp):
    """No ChannelSummary/ChannelSetupDialog/MainWindow around it to call close_plots()
    for it — the exact composition 17 standalone ColumnStack tests already use."""
    before = _menu_census()

    m, names = _matrix()
    st = ColumnStack(1000).build(m, names)
    plot_items = [p.getPlotItem() for p in st.plots]
    assert plot_items and all(pi.ctrlMenu is not None for pi in plot_items), (
        "build() must have built real PlotWidgets with menus, or this test proves nothing"
    )
    assert _menu_census() > before, (
        "build() should have added new top-level QMenus, or the close-side assertion "
        "below would pass vacuously"
    )

    st.close()
    _settle()

    assert _menu_census() <= before, (
        "ColumnStack.closeEvent must release every menu its own build() created, exactly "
        "as ChannelSummary.close_plots()/ChannelSetupDialog's finished-signal handler "
        "already do for the orchestrated cases (delta <= 0 -- see the PreviewScreen "
        "guard's own comment for why exact equality is the wrong assertion here)"
    )
    assert st.plots == []
    assert all(pi.ctrlMenu is None for pi in plot_items)
    st.close()                              # idempotent: a second close must not raise


def test_mainwindow_closeevent_still_orchestrates_every_owned_plot(qapp, tmp_path):
    """Pins the ALREADY-existing MainWindow.closeEvent orchestration (the three screen/
    summary shutdown() and close_plots() calls) as its own named regression test — the
    thing Layer 1's per-widget overrides must never become an excuse to simplify away.

    Point 6 measured a bare, freshly constructed MainWindow (real channel assignments via
    synth_settings(), no Preview interaction at all) at 83 surviving QMenus after close()
    BEFORE the orchestration existed, 6 after — the window's own File/Edit/Help menu-bar
    menus, not a leak (see CLAUDE.md's Point 6 section and
    test_preview_screen.py::test_closing_the_window_closes_setups_channel_summary_plots).
    Independently re-measured in this sandbox while writing this guard (Qt 6.11.1,
    pyqtgraph 0.14.0, offscreen): delta 6, an exact match. This asserts a ceiling well
    above that legitimate baseline but far below the pre-fix population, not zero: a
    MainWindow is CLOSED here, not deleted (deleting one segfaults nondeterministically on
    Python 3.11, see conftest.py), and its own menu bar's QMenus are expected to still be
    there — only a return toward the 83-QMenu pyqtgraph population would mean the
    orchestration broke.
    """
    from respmech.ui.main_window import MainWindow

    s = synth_settings(str(tmp_path))
    before = _menu_census()

    win = MainWindow(AppState(s))
    summary = win.settings_screen.channel_summary
    assert summary.stack is not None and summary.stack.plots, (
        "synth_settings() assigns real channels, so Setup's initial refresh should have "
        "built a real ColumnStack with real plots -- if this ever fails, the test below "
        "is vacuous (a MainWindow with nothing to close proves no orchestration at all)"
    )

    win.close()
    _settle()

    delta = _menu_census() - before
    assert delta <= 15, (
        f"a closed, empty MainWindow should leave only its own menu-bar QMenus behind "
        f"(measured 6 in the point 6 investigation), not an unclosed pyqtgraph population "
        f"(measured 83 before MainWindow.closeEvent's orchestration existed): "
        f"{before} -> {before + delta} (delta {delta})"
    )
    win.close()                             # idempotent: a second close must not raise

"""Shared pytest scaffolding for the unit suite.

Centralises what was previously copy-pasted across ~15 GUI test files: the headless-Qt
environment, the single themed ``QApplication``, and the prefs/dark-mode fixtures. Pure
helper functions (the synthetic-settings factory, colour helpers) live in ``_helpers.py``
so they can be imported at module level; fixtures live here so pytest injects them.
"""
import os
import sys

# Headless Qt + a Qt matplotlib backend, before any QApplication / pyplot import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Pin pyqtgraph to PySide6 (the app's binding) so importing pyqtgraph before PySide6 in a
# test module can't bind it to PyQt6 — loading two Qt bindings in one process crashes.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
try:
    import matplotlib
    matplotlib.use("QtAgg")
except Exception:                       # pragma: no cover - matplotlib always present in CI
    pass

# Make src/ importable without installing (until packaging lands), and this test dir
# importable so ``from _helpers import …`` works regardless of the invoking cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402


_EXIT_STATUS = {"code": 0}


def pytest_sessionfinish(session, exitstatus):
    _EXIT_STATUS["code"] = int(exitstatus)


def pytest_unconfigure(config):
    """Exit with pytest's final status *before* Qt's static destructors run.

    On some platforms (seen on offscreen macOS + Python 3.11 in CI) a worker QThread can
    outlive its window and, at interpreter shutdown, Qt aborts — "QThread: Destroyed while
    thread is still running" → Abort trap 6 → the process exits 134 even though every test
    passed. This is pytest's LAST hook (after the summary has printed), so we flush and
    exit immediately with the captured status — a failing run keeps its non-zero code, a
    passing run sidesteps the abort-prone teardown. No-op unless a QApplication was created
    (pure-core runs are unaffected)."""
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        return
    if QApplication.instance() is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS["code"])


@pytest.fixture(scope="session")
def qapp():
    """One themed ``QApplication`` for the whole session (Qt permits only one). Skips the
    depending test when the GUI stack is unavailable."""
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from respmech.ui import theme
    # Native file dialogs are OS panels that ignore the offscreen platform: a test (or a
    # crashed test's teardown) reaching an unmocked QFileDialog would pop a REAL macOS
    # save panel on the developer's screen and block the suite until a human dismisses
    # it. Widget-based dialogs stay inside the (offscreen) app instead.
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    return app


@pytest.fixture(autouse=True)
def _close_top_level_windows(request):
    """Systemic GUI-test isolation. After EVERY test, close any lingering top-level window
    so its ``closeEvent`` runs ``MainWindow`` → ``PreviewScreen.shutdown()``: that cancels +
    joins any worker QThread and disarms the 300 ms auto-run debounce timer.

    Why this is needed and not just per-test ``win.close()``:
      * A test whose assertion FAILS skips its own trailing ``win.close()``, leaking a screen
        (and, without cancellation, a running batch thread) into the next test.
      * Several GUI tests arm the debounce (via a ``file_combo`` change) but never spin an
        event loop, so the timer stays LOADED on a still-alive screen. When a later test
        spins the session's first real event loop, that expired timer fires and dispatches
        real, never-cancelled jobs on a dead screen — on the 2-core headless Windows runner
        the accumulated GIL-bound threads starve the reactive tests into timing out.

    Deliberately does NOT ``processEvents()`` before closing — that would fire the very
    timers this net exists to neutralise. Closing is synchronous (``closeEvent`` runs inline),
    so the timers are stopped without ever letting a pending one dispatch. A no-op until a
    QApplication exists (pure-core tests are unaffected) and cheap when no windows are open."""
    yield
    # A test that never asked for `qapp` never got the GUI machinery, so it has no windows
    # to close — and this net is not free. Measured 03-08-2026: one gc.collect() plus one
    # DeferredDelete dispatch costs ~0.34 s per test locally, and on the macOS CI runner the
    # teardown of a PURE LOADER test (test_loader_returns_2d_for_a_single_column, which opens
    # no window at all) was 17 s. 246 of the 715 tests are in that category.
    #
    # Skipping is not "never clean up": it is "clean up at the next GUI test's teardown",
    # which is already what happens to a window dropped mid-test. If a non-qapp test ever
    # did build a widget — a latent bug in that test — the next GUI test's pass still reaps
    # it, so this cannot leak across a run.
    #
    # The 469 GUI tests keep the identical net, including the ordering the comment below
    # depends on. Deliberately NOT touched: the gc/DeferredDelete/scan sequence itself. The
    # segfault it guards against is py3.11-specific and cannot be reproduced on the
    # maintainer's 3.13, so changing it could only be validated by shipping it.
    if "qapp" not in getattr(request, "fixturenames", ()):
        return
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:                       # pragma: no cover - PySide6 always present in CI
        return
    app = QApplication.instance()
    if app is None:
        return
    # Reap any widget torn down during the test BEFORE scanning: the test's local vars
    # are gone by now, so an unreferenced window's C++ half may be mid-destruction, and
    # topLevelWidgets() SEGFAULTS while wrapping such a pointer on py3.11 (py3.12's GC
    # timing hides it) — before any per-widget isValid check can run. A gc pass finalises
    # the dropped wrappers and a DeferredDelete-only dispatch drains the resulting
    # deleteLater's, so topLevelWidgets() then returns only live windows. DeferredDelete
    # is dispatched alone (not processEvents) so the timers this net exists to stop never
    # fire.
    import gc  # noqa: PLC0415
    from PySide6.QtCore import QEvent  # noqa: PLC0415
    gc.collect()
    try:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    except Exception:                       # pragma: no cover - defensive
        pass
    try:
        import shiboken6  # noqa: PLC0415
        _alive = shiboken6.isValid
    except Exception:                       # pragma: no cover - shiboken ships with PySide6
        _alive = lambda _w: True            # noqa: E731
    for w in list(app.topLevelWidgets()):
        if not _alive(w):                   # belt-and-braces after the reap above
            continue
        try:
            # Hide BEFORE closing: this net is plumbing, not a user closing a window, and
            # MainWindow.closeEvent only asks about unsaved edits when the window is
            # visible. A test that left a shown, dirty window behind would otherwise pop a
            # REAL modal here — after monkeypatch teardown restored the real QMessageBox —
            # and hang the whole run with nobody to answer it.
            w.hide()
            w.close()
        except Exception:                   # pragma: no cover - defensive; never fail teardown
            pass


@pytest.fixture(autouse=True)
def _prefs_never_touch_real_settings(monkeypatch):
    """EVERY test runs against a throwaway prefs scope. This is autouse because the damage
    mode is silent and real: any test that opens or saves an analysis writes recents /
    last-folder / last-rig through ui.prefs, and without isolation those writes land in the
    USER'S real QSettings — a suite run once left pytest tmp paths in the developer's
    actual recents menu and replaced their remembered rig with the synth test rig."""
    try:
        from respmech.ui import prefs  # noqa: PLC0415
    except Exception:                   # pragma: no cover - respmech.ui always importable here
        return
    monkeypatch.setattr(prefs, "ORG", "RespMechTest")
    monkeypatch.setattr(prefs, "APP", "RespMechTestSuite")


@pytest.fixture
def isolated_prefs(_prefs_never_touch_real_settings):
    """The throwaway prefs scope (see the autouse isolation above), CLEARED around the
    test — for tests that assert on prefs contents and need a clean slate."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from respmech.ui import prefs
    QSettings("RespMechTest", "RespMechTestSuite").clear()
    yield prefs
    QSettings("RespMechTest", "RespMechTestSuite").clear()


@pytest.fixture
def windows_metrics(qapp):
    """Model the Windows runner's font metrics for the duration of a test, then restore.

    macOS has the narrowest metrics we ship to, so a layout budget verified only here is
    verified on the friendliest platform there is — which is exactly how a modal that fits
    on the maintainer's Mac shipped 254 px too tall for a Windows laptop.

    Widen the horizontal ADVANCE and leave the height alone: ``QFont.setStretch(145)`` lands
    within a few percent of the runner (modelled 1506-1538 px against CI's measured 1516 px).
    Scaling the point size instead is the wrong instrument — it inflates row heights too, so
    it understates the width problem and invents height failures Windows does not have.
    CLAUDE.md has prescribed this since the window-width fix; this is the first time it has
    been executable rather than prose.
    """
    orig = qapp.font()
    f = qapp.font()
    f.setStretch(145)
    qapp.setFont(f)
    try:
        yield qapp
    finally:
        qapp.setFont(orig)


@pytest.fixture
def dark_app(qapp):
    """Force the dark theme for the duration of a test, then restore the OS default so the
    global theme state never leaks into other tests. Yields ``(app, theme)``."""
    from respmech.ui import theme
    orig = theme._prefers_dark
    theme._prefers_dark = lambda _a: True
    theme.apply_theme(qapp)
    try:
        yield qapp, theme
    finally:
        theme._prefers_dark = orig
        theme.apply_theme(qapp)


def pytest_collection_modifyitems(items):
    """Run the theme-switching tests FIRST.

    ``theme.apply_theme`` sets a stylesheet on the QApplication, and Qt re-polishes every
    live widget when it does — so its cost grows with everything the session has built so
    far. Nothing else in the suite calls it, which makes this purely a scheduling problem.

    Measured on 03-08-2026, same 13 tests, same machine:

        at their alphabetical position (late)   ~800 s
        collected first                           12.6 s

    A single call was up to 82 s late in the run against ~0.2 s early. That is 72% of the
    whole suite's wall time, and it is what took CI from ~28 min to over three hours per job.

    The alternative — deleting the accumulated windows in teardown so the population stays
    small — was tried first and SEGFAULTS on py3.11: the existing reaper in
    _close_top_level_windows documents why touching those half-destroyed wrappers is unsafe.
    Reordering costs nothing and risks nothing: these tests are independent, ``dark_app``
    restores the OS default in its own teardown, and pytest's order here is otherwise just
    alphabetical accident (no ordering plugin is installed).
    """
    # Matching on fixturenames ALONE is not enough, and the first cut of this hook proved it
    # by saving only 230 s of the ~800 s: test_theme_paints_both_modes.py pulls the fixture
    # with request.getfixturevalue("dark_app") INSIDE the test body, so at collection time it
    # is nowhere in fixturenames — and that file is the expensive one. Module names carry it.
    THEME_MODULES = {"test_theme_paints_both_modes.py", "test_dialogs_dark_mode.py"}

    def theme_switching(item):
        if os.path.basename(str(getattr(item, "fspath", ""))) in THEME_MODULES:
            return True
        if "dark_app" in getattr(item, "fixturenames", ()):
            return True
        # the two in test_gui.py that drive apply_theme directly rather than via the fixture
        return item.name in ("test_theme_applies_light_and_dark",
                             "test_plot_palette_tracks_theme_and_has_a_complete_contract")

    items.sort(key=lambda it: 0 if theme_switching(it) else 1)

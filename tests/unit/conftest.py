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
    from PySide6.QtWidgets import QApplication
    from respmech.ui import theme
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    return app


@pytest.fixture
def isolated_prefs(monkeypatch):
    """Point ``ui.prefs`` (QSettings) at a throwaway scope so tests never touch — or
    persist into — the real user preferences."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from respmech.ui import prefs
    monkeypatch.setattr(prefs, "ORG", "RespMechTest")
    monkeypatch.setattr(prefs, "APP", "RespMechTestSuite")
    QSettings("RespMechTest", "RespMechTestSuite").clear()
    yield prefs
    QSettings("RespMechTest", "RespMechTestSuite").clear()


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

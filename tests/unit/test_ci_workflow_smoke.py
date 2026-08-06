"""Guards the exact regression that made CI's own smoke step red on ``ui-overhaul``.

``.github/workflows/ci.yml``'s "GUI constructs headless" step is a plain inline
``python -c "..."`` command, not something pytest runs — so when ticket B03 folded the
Run tab into Preview & QC as a drawer (``MainWindow.tabs.count()`` dropped from 3 to 2),
nothing in the unit suite noticed that the workflow's hardcoded ``assert
w.tabs.count()==N`` had gone stale. ``test_mainwindow_constructs`` in ``test_gui.py``
was updated to ``== 2`` at the time, but the *workflow file* was not, and the two drifted
without any local signal — the CI run itself was the only place the mismatch surfaced.

This test closes that gap generically: it parses the asserted tab count straight out of
ci.yml and compares it against a MainWindow really constructed the same way CI constructs
one (``MainWindow(AppState())``, no synthetic settings). A future ticket that changes the
tab count again will fail this test locally instead of relying on someone remembering to
also grep the workflow file.
"""
import os
import re

from respmech.ui.state import AppState

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CI_WORKFLOW = os.path.join(ROOT, ".github", "workflows", "ci.yml")


def _asserted_tab_count():
    with open(CI_WORKFLOW, encoding="utf-8") as f:
        text = f.read()
    m = re.search(
        r"GUI constructs headless.*?assert\s+w\.tabs\.count\(\)\s*==\s*(\d+)", text, re.S
    )
    assert m, (
        "Could not find the 'assert w.tabs.count()==N' smoke check in "
        "'GUI constructs headless' step of .github/workflows/ci.yml — has the step "
        "been renamed or rewritten? Update this test's regex to match."
    )
    return int(m.group(1))


def test_ci_workflow_gui_smoke_tab_count_matches_mainwindow(qapp):
    from respmech.ui.main_window import MainWindow

    # No explicit close/deleteLater here: the autouse `_close_top_level_windows` fixture
    # in conftest.py is the established pattern for every other GUI test in this suite
    # (see test_gui.py) and already reaps this window at teardown.
    win = MainWindow(AppState())
    actual = win.tabs.count()
    asserted = _asserted_tab_count()
    assert actual == asserted, (
        f"MainWindow actually has {actual} tab(s), but .github/workflows/ci.yml's "
        f"'GUI constructs headless' step still asserts w.tabs.count()=={asserted}. "
        "Update the assert in ci.yml to match — this is exactly the drift that made "
        "CI red after ticket B03 folded Run & results into a Preview & QC drawer."
    )

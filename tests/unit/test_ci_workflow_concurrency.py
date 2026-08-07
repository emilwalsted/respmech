"""Guards against CI's concurrency group cancelling a DIFFERENT commit's still-running check.

``.github/workflows/ci.yml`` used ``group: ci-${{ github.ref }}`` with
``cancel-in-progress: true``. That group key is identical for every push to the same
branch, so when a second commit landed on a branch before the first commit's GUI-smoke
matrix (Windows + macOS, full unit suite) had finished, GitHub Actions cancelled the
first commit's run outright. On a branch that receives commits close together — as
``ui-overhaul`` does from chained ticket dispatch, sometimes under 30 minutes apart — the
in-flight run is cancelled before it can report a result almost every time, so no commit
ever gets a completed, genuinely green check. That reads as "CI is red" even though
nothing in the code or tests is actually broken.

This test parses the concurrency group expression straight out of ci.yml and requires it
to key push-triggered runs by something that differs per commit (``github.sha``), so a
later push can never cancel an earlier, unrelated commit's run. Pull-request runs may
still key by PR number: cancelling a stale review of an old PR head is the intended,
safe use of ``cancel-in-progress``.
"""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CI_WORKFLOW = os.path.join(ROOT, ".github", "workflows", "ci.yml")


def _concurrency_block():
    with open(CI_WORKFLOW, encoding="utf-8") as f:
        text = f.read()
    m = re.search(
        r"^concurrency:\s*\n"
        r"(?:\s*#.*\n)*"  # the group key is explained by a comment block above it
        r"\s*group:\s*(.+?)\s*\n"
        r"\s*cancel-in-progress:\s*(\S+)",
        text,
        re.M,
    )
    assert m, (
        "Could not find a top-level 'concurrency: group: ... / cancel-in-progress: ...' "
        "block in .github/workflows/ci.yml — has it been renamed or restructured? "
        "Update this test's regex to match."
    )
    return m.group(1), m.group(2)


def test_ci_workflow_concurrency_group_is_not_shared_by_every_push_to_a_branch():
    group_expr, cancel = _concurrency_block()
    assert cancel == "true", (
        "cancel-in-progress changed to something other than 'true' — this test only "
        "guards the group KEY; re-check whether the new value still needs a per-commit "
        "group."
    )
    # Not just "does github.sha appear somewhere": require the actual fallback shape,
    # PR number falling through to the commit SHA, in that order. A group like
    # 'ci-${{ github.sha }}' alone would satisfy a looser "contains github.sha" check
    # while silently dropping the PR-number branch, which is what makes a stale push to
    # an open PR get cancelled by its own successor (the one case cancel-in-progress is
    # FOR). Both operands must be present, joined by '||', PR number first.
    assert re.search(
        r"github\.event\.pull_request\.number\s*\|\|\s*github\.sha", group_expr
    ), (
        f"CI's concurrency group is {group_expr!r}. It must fall back from "
        "github.event.pull_request.number to github.sha (in that order), not just "
        "contain github.sha somewhere. Without the PR-number branch, a later push to an "
        "open PR would stop cancelling its own stale review run. Without github.sha as "
        "the push-event key, a later push to a branch would cancel an EARLIER, unrelated "
        "commit's still-running check before it can report a result — exactly what made "
        "ui-overhaul's CI look permanently red under chained ticket pushes."
    )

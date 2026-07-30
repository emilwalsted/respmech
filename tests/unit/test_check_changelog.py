"""``tools/check_changelog.py`` must catch a change that never reached the changelog.

CHANGELOG.md is the single source for what a release tells anyone: docs/RELEASING.md folds its
entry into the dated section, respmech.dk's changelog page is the reader-facing rewrite of the
same list, and the mailing-list notification is built from that page. A change that never
reaches the entry is therefore a change nobody is ever told about, and "did I remember
everything?" is the question a human answers worst at the end of a release.

WHAT THE TOOL PROMISES, AND WHAT THESE TESTS PIN DOWN
-----------------------------------------------------
The tool is a word comparison, so it can be certain of exactly one thing: that a commit has no
lexical overlap with the entry at all. That is the forgotten-change case, and it is a hard
failure. Everything weaker is handed over as a worksheet for a human to read, sorted weakest
match first, rather than dressed up as a verdict.

That boundary is deliberate and was arrived at by being wrong three times. Earlier versions
tried to judge adequacy: first "any shared word anywhere in the entry" (which passed a deleted
bullet, because a word from a different bullet matched), then "two shared words in one bullet"
(which passed the same deletion, because a long neighbouring bullet happened to contain two of
them), then "two shared words, one unique to that bullet" (which passed too, because deleting
the right bullet is exactly what MAKES a neighbour's words unique). Each rule looked stronger
and was measurably not. So these tests hold the tool to the promise it can keep, and the
worksheet cases assert that a weak match is *reported as weak* rather than silently accepted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_changelog.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "T",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "T",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(repo),
        },
    )


def _commit(repo: Path, path: str, body: str, subject: str) -> None:
    f = repo / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(body, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", subject)


def _run(repo: Path, *extra: str):
    """Run the tool against a scratch repo. Returns (exit code, output)."""
    # The tool locates CHANGELOG.md relative to its own file, so it is copied into
    # the scratch repo rather than pointed at it: that keeps the test honest about
    # how the tool is actually invoked in CI.
    tools = repo / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "check_changelog.py").write_text(TOOL.read_text(encoding="utf-8"), encoding="utf-8")
    p = subprocess.run(
        [sys.executable, str(tools / "check_changelog.py"), *extra],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout + p.stderr


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A repo with one tag, then three commits: two user-visible, one CI-only."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _commit(r, "src/respmech/core.py", "x = 1\n", "initial")
    _git(r, "tag", "v1.0.0")

    _commit(
        r,
        "src/respmech/emg.py",
        "def ecg_auto_detect(sig):\n    return sig\n",
        "Add ecg_auto_detect for batch runs",
    )
    _commit(
        r,
        "src/respmech/ui/mechanics.py",
        "resample_enabled = True\n",
        "Mechanics advanced: detail fields follow their checkbox",
    )
    _commit(r, ".github/workflows/ci.yml", "on: push\n", "CI: bump the runner image")
    return r


def _changelog(repo: Path, body: str) -> None:
    (repo / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\nA release.\n\n" + body,
        encoding="utf-8",
    )
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-m", "CHANGELOG: record the changes")


def test_a_complete_entry_passes(repo: Path) -> None:
    _changelog(
        repo,
        "- Added `ecg_auto_detect` so ECG detection can drive batch runs\n"
        "- The Mechanics advanced dialog's detail fields now follow their checkbox\n",
    )
    code, out = _run(repo)
    assert code == 0, out
    # The CI-only commit must be skipped, and said to be skipped: a classification
    # nobody can see is a classification nobody can dispute.
    assert "ikke brugersynlige" in out
    assert "bump the runner image" in out


def test_a_forgotten_change_fails_and_is_named(repo: Path) -> None:
    """The one thing a word comparison can be certain of: no overlap at all."""
    _changelog(repo, "- Added `ecg_auto_detect` so ECG detection can drive batch runs\n")
    code, out = _run(repo)
    assert code == 1, out
    assert "UDEN SPOR" in out
    assert "Mechanics advanced" in out
    # and it must point at the paths, so the reader can judge for themselves
    assert "src/respmech/ui/mechanics.py" in out


def test_a_waiver_records_the_omission_instead_of_hiding_it(repo: Path) -> None:
    sha = subprocess.run(
        ["git", "log", "--format=%H", "--grep", "Mechanics", "-1"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()[:7]
    _changelog(
        repo,
        "- Added `ecg_auto_detect` so ECG detection can drive batch runs\n"
        f"\n<!-- changelog-skip {sha} internal polish, not worth a release note -->\n",
    )
    code, out = _run(repo)
    assert code == 0, out
    assert "Bevidst udeladt" in out
    assert "internal polish" in out


def test_a_waiver_without_a_reason_does_not_count(repo: Path) -> None:
    """A bare sha would be a mute switch. The reason is the point of the waiver."""
    sha = subprocess.run(
        ["git", "log", "--format=%H", "--grep", "Mechanics", "-1"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()[:7]
    _changelog(
        repo,
        "- Added `ecg_auto_detect` so ECG detection can drive batch runs\n"
        f"\n<!-- changelog-skip {sha} nope -->\n",
    )
    code, out = _run(repo)
    assert code == 1, out
    assert "UDEN SPOR" in out


def test_a_weak_match_is_reported_as_weak_not_accepted_silently(repo: Path) -> None:
    """The case three earlier, cleverer rules got wrong.

    The Mechanics commit is not described here at all; the only bullet happens to share
    the words "detail" and "fields" while talking about something else entirely. The
    tool must not claim that is coverage, and must not claim it is a forgotten change
    either. It must put it at the top of the worksheet as the weakest match, with the
    candidate bullet quoted, so a human decides."""
    _changelog(
        repo,
        "- Added `ecg_auto_detect` so ECG detection can drive batch runs\n"
        "- The run report now lists detail fields for every processed file\n",
    )
    code, out = _run(repo)
    assert code == 0, out
    assert "svagt match" in out
    linjer = [l for l in out.splitlines() if "Mechanics advanced" in l]
    assert linjer, out
    assert linjer[0].strip().startswith("?"), f"skulle være markeret som svagt: {linjer[0]}"
    assert "run report now lists detail fields" in out
    # --strict is for whoever wants the tighter gate, and must actually bite
    code_strict, _ = _run(repo, "--strict")
    assert code_strict == 1


def test_warnings_only_never_fails(repo: Path) -> None:
    _changelog(repo, "- Something entirely unrelated to either change\n")
    code, out = _run(repo, "--warnings-only")
    assert code == 0, out
    assert "UDEN SPOR" in out


def test_an_empty_range_is_not_an_error(repo: Path) -> None:
    _changelog(repo, "- Added `ecg_auto_detect` so ECG detection can drive batch runs\n")
    _git(repo, "tag", "v1.1.0")
    code, out = _run(repo)
    assert code == 0, out
    assert "Intet at kontrollere" in out


def test_a_shallow_clone_is_refused_not_silently_useless(tmp_path: Path) -> None:
    """A shallow checkout has no tags, so there is no range to measure.

    Measured in a real ``git clone --depth 1`` of this repo before the guard existed:
    without ``--version`` the tool reported "1 commits, 1 to review" and exited 0 — a
    completely contentless green light — and *with* ``--version`` it failed with "not a
    tag in this repo", which in the release gate would have stopped every single
    release. Both are worse than refusing outright, so it refuses outright and names
    the fix. The workflows set ``fetch-depth: 0`` for the same reason."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit(origin, "src/respmech/core.py", "x = 1\n", "initial")
    _git(origin, "tag", "v1.0.0")
    _commit(origin, "src/respmech/emg.py", "y = 2\n", "a user-visible change")

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "file://" + str(origin), str(shallow)],
        check=True,
        capture_output=True,
    )
    (shallow / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\nA release.\n\n- something\n", encoding="utf-8"
    )
    code, out = _run(shallow)
    assert code == 2, out
    assert "overfladisk" in out or "shallow" in out
    assert "fetch-depth" in out

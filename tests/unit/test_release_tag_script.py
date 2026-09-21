"""``.github/scripts/release_tag.sh`` must never create a release tag it should not.

The script turns a "release request" (an empty ``Release vX.Y.Z`` commit on the branch
``release-request``) into the annotated tag that cuts a release. A PyPI version is immutable
and a pushed tag is never moved, so every mistake here is permanent: a tag on the wrong
commit, a tag for a version the package does not carry, a tag that silently leaves out the
change the requester thought they were releasing. The Actions workflow around the script
(``.github/workflows/release-request.yml``) cannot be exercised from a test, so the script
holds all the decisions and these tests hold the script to them.

Every test builds its own throwaway bare "origin" plus a working repo, and runs the script
in a fresh clone of that origin, which is what ``actions/checkout`` with ``fetch-depth: 0``
gives the workflow. Git is run hermetically (no global or system config), so a developer's
own signing setup neither helps nor breaks the annotated-tag step.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "release_tag.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-request.yml"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None or shutil.which("git") is None,
    reason="runs a bash script against local git repositories; the workflow itself runs on Linux",
)

# Nothing from the caller's shell may steer git (GIT_DIR set by a hook would send every
# sandbox command into THAT repository) or the script under test (an exported DRY_RUN).
_INHERITED = {
    key: value for key, value in os.environ.items()
    if not key.startswith("GIT_")
    and key not in ("DRY_RUN", "VERSION_FILE", "DEFAULT_BRANCH", "REQUEST_SHA")
}

_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_TERMINAL_PROMPT": "0",
}


class Sandbox:
    """A bare origin, a working repo on ``master``, and a way to run the script against them."""

    def __init__(self, root: Path):
        self.root = root
        self.bare = root / "origin.git"
        self.repo = root / "repo"
        self._clones = 0
        self.git("init", "-q", "--bare", "-b", "master", str(self.bare), cwd=root)
        self.repo.mkdir()
        self.git("init", "-q", "-b", "master")
        self.git("remote", "add", "origin", str(self.bare))
        self.set_version("1.2.3")

    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        done = subprocess.run(
            ["git", *args], cwd=cwd or self.repo, env={**_INHERITED, **_GIT_ENV},
            capture_output=True, text=True,
        )
        if check and done.returncode != 0:
            raise AssertionError(f"git {' '.join(args)} failed: {done.stderr}")
        return done.stdout.strip()

    def set_version(self, version: str, line: str | None = None) -> str:
        """Commit ``__version__`` on master and push it; returns master's new tip."""
        self.git("checkout", "-q", "-B", "master")
        target = self.repo / "src" / "respmech" / "__init__.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((line or f'__version__ = "{version}"') + "\n", encoding="utf-8")
        self.git("add", "src/respmech/__init__.py")
        self.git("commit", "-q", "-m", f"version {version}")
        self.git("push", "-q", "origin", "master")
        return self.git("rev-parse", "master")

    def advance_master(self) -> str:
        """An unrelated later commit on master (the version stays what it was)."""
        self.git("checkout", "-q", "master")
        self.git("commit", "-q", "--allow-empty", "-m", "unrelated later work")
        self.git("push", "-q", "origin", "master")
        return self.git("rev-parse", "master")

    def request(self, message: str, *, start: str = "master", change_a_file: bool = False) -> str:
        """Push a release request built on ``start``; returns the request commit."""
        self.git("checkout", "-q", "-B", "release-request", start)
        if change_a_file:
            (self.repo / "late-fix.txt").write_text("not on master\n", encoding="utf-8")
            self.git("add", "late-fix.txt")
            self.git("commit", "-q", "-m", message)
        else:
            self.git("commit", "-q", "--allow-empty", "-m", message)
        self.git("push", "-q", "--force", "origin", "release-request")
        return self.git("rev-parse", "release-request")

    def run(self, request_sha: str, *, prepare=None, **env: str) -> tuple[int, str]:
        """Run the script in a fresh clone of origin, as the workflow's checkout would.

        ``prepare(clone)`` may leave something behind in the clone first (a local-only tag).
        """
        self._clones += 1
        clone = self.root / f"clone-{self._clones}"
        self.git("clone", "-q", str(self.bare), str(clone), cwd=self.root)
        # A clone only tracks branches; fetch the request commit the way the workflow has it.
        self.git("fetch", "-q", "origin", "+refs/heads/*:refs/remotes/origin/*", cwd=clone)
        if prepare is not None:
            prepare(clone)
        done = subprocess.run(
            ["bash", str(SCRIPT)], cwd=clone, capture_output=True, text=True,
            env={**_INHERITED, **_GIT_ENV, "DEFAULT_BRANCH": "master", "DRY_RUN": "false",
                 "REQUEST_SHA": request_sha, **env},
        )
        return done.returncode, done.stdout + done.stderr

    def all_tags(self) -> str:
        """Every tag on origin ('' when there is none at all)."""
        return self.git("ls-remote", "--tags", str(self.bare), cwd=self.root)

    def tag_commit(self, tag: str) -> str:
        """The commit ``tag`` points at on origin ('' if origin has no such tag)."""
        listed = self.git("ls-remote", "--tags", str(self.bare), f"refs/tags/{tag}", cwd=self.root)
        if not listed:
            return ""
        return self.git("rev-parse", f"{tag}^{{commit}}", cwd=self.bare)


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


# ---------------------------------------------------------------- the path that releases

def test_request_on_masters_tip_tags_master_not_the_request_commit(box: Sandbox):
    tip = box.git("rev-parse", "master")
    request = box.request("Release v1.2.3")

    rc, out = box.run(request)

    assert rc == 0, out
    assert "TAG: v1.2.3" in out
    assert box.tag_commit("v1.2.3") == tip
    assert box.tag_commit("v1.2.3") != request


def test_the_tag_is_annotated_and_carries_the_bot_identity(box: Sandbox):
    rc, out = box.run(box.request("Release v1.2.3"))

    assert rc == 0, out
    assert box.git("cat-file", "-t", "refs/tags/v1.2.3", cwd=box.bare) == "tag"
    tagger = box.git("for-each-ref", "--format=%(taggername) %(taggeremail)",
                     "refs/tags/v1.2.3", cwd=box.bare)
    assert tagger == "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"


@pytest.mark.parametrize("line", [
    "__version__ = '1.2.3'",
    '__version__ = "1.2.3"  # bump together with pyproject.toml',
    '__version__="1.2.3"',
])
def test_version_line_is_read_in_the_spellings_the_project_uses(box: Sandbox, line: str):
    box.set_version("1.2.3", line=line)

    rc, out = box.run(box.request("Release v1.2.3"))

    assert rc == 0, out
    assert box.tag_commit("v1.2.3") == box.git("rev-parse", "master")


def test_dry_run_creates_nothing(box: Sandbox):
    rc, out = box.run(box.request("Release v1.2.3"), DRY_RUN="true")

    assert rc == 0, out
    assert "dry run" in out
    assert box.tag_commit("v1.2.3") == ""


# ---------------------------------------------------------------- idempotency, and never moving a tag

def test_repeating_a_request_that_already_succeeded_is_success(box: Sandbox):
    request = box.request("Release v1.2.3")
    assert box.run(request)[0] == 0

    rc, out = box.run(request)

    assert rc == 0, out
    assert "already present" in out


def test_a_rerun_stays_green_after_master_has_moved_on(box: Sandbox):
    tip = box.git("rev-parse", "master")
    request = box.request("Release v1.2.3")
    assert box.run(request)[0] == 0
    box.advance_master()

    rc, out = box.run(request)

    assert rc == 0, out
    assert "already present" in out
    assert box.tag_commit("v1.2.3") == tip


def test_an_existing_tag_is_never_moved(box: Sandbox):
    tip = box.git("rev-parse", "master")
    assert box.run(box.request("Release v1.2.3"))[0] == 0
    box.advance_master()

    rc, out = box.run(box.request("Release v1.2.3"))

    assert rc == 1, out
    assert "REFUSING" in out
    assert box.tag_commit("v1.2.3") == tip


# ---------------------------------------------------------------- requests that must be refused

def test_stale_request_is_refused_once_master_has_moved(box: Sandbox):
    request = box.request("Release v1.2.3")
    box.advance_master()

    rc, out = box.run(request)

    assert rc == 1, out
    assert "stale" in out
    assert box.tag_commit("v1.2.3") == ""


def test_request_with_a_commit_in_between_is_refused(box: Sandbox):
    box.git("checkout", "-q", "-B", "release-request", "master")
    box.git("commit", "-q", "--allow-empty", "-m", "something else first")
    request = box.request("Release v1.2.3", start="release-request")

    rc, out = box.run(request)

    assert rc == 1, out
    assert box.tag_commit("v1.2.3") == ""


def test_request_that_changes_files_is_refused(box: Sandbox):
    """The tag lands on the parent, so the change would silently miss the release."""
    request = box.request("Release v1.2.3", change_a_file=True)

    rc, out = box.run(request)

    assert rc == 1, out
    assert "not empty" in out
    assert box.tag_commit("v1.2.3") == ""


def test_request_for_a_version_the_package_does_not_carry_is_refused(box: Sandbox):
    """publish-pypi.yml would refuse later, but by then the tag exists and is never moved."""
    request = box.request("Release v9.9.9")

    rc, out = box.run(request)

    assert rc == 1, out
    assert "__version__" in out and "1.2.3" in out
    assert box.tag_commit("v9.9.9") == ""


def test_unreadable_version_file_is_refused(box: Sandbox):
    rc, out = box.run(box.request("Release v1.2.3"), VERSION_FILE="src/respmech/missing.py")

    assert rc == 1, out
    assert "could not read __version__" in out
    assert box.tag_commit("v1.2.3") == ""


@pytest.mark.parametrize("message", [
    "Bump version to 1.2.3",          # not the agreed form
    "Release v1.2.3-rc1",             # only plain semver is ever released
    "Release v1.02.3",                # a leading zero is not semver, and PyPI would normalise it away
    "Release 1.2.3",                  # the v is part of the tag name
    "release v1.2.3",                 # exact form, exact case
    "Please Release v1.2.3",          # the WHOLE first line, not a match somewhere in it
])
def test_malformed_request_message_is_refused(box: Sandbox, message: str):
    rc, out = box.run(box.request(message))

    assert rc == 1, out
    assert "is not of the form 'Release vX.Y.Z'" in out
    assert box.all_tags() == ""


@pytest.mark.parametrize("message, carried", [
    ("Release v1.2.3-rc1", "1.2.3-rc1"),
    ("Release v1.02.3", "1.02.3"),
])
def test_message_form_is_refused_even_when_the_package_carries_that_version(
        box: Sandbox, message: str, carried: str):
    """Takes the version check away as a possible refuser: the message check must do it."""
    box.set_version(carried)

    rc, out = box.run(box.request(message))

    assert rc == 1, out
    assert "is not of the form 'Release vX.Y.Z'" in out
    assert box.all_tags() == ""


def test_first_line_means_first_line_not_first_paragraph(box: Sandbox):
    """git's %s folds "Release\\nv1.2.3" into "Release v1.2.3"; the script must not."""
    request = box.request("Release\nv1.2.3")

    rc, out = box.run(request)

    assert rc == 1, out
    assert "is not of the form 'Release vX.Y.Z'" in out
    assert box.all_tags() == ""


def test_a_body_below_the_first_line_is_allowed(box: Sandbox):
    """Only the first line is the contract; a body may say anything, even another version."""
    request = box.request("Release v1.2.3\n\nSupersedes the withdrawn Release v9.9.9 attempt.")

    rc, out = box.run(request)

    assert rc == 0, out
    assert box.all_tags().endswith("refs/tags/v1.2.3^{}")


def test_root_commit_is_refused_cleanly(box: Sandbox):
    box.git("checkout", "-q", "--orphan", "release-request")
    box.git("commit", "-q", "--allow-empty", "-m", "Release v1.2.3")
    box.git("push", "-q", "--force", "origin", "release-request")

    rc, out = box.run(box.git("rev-parse", "release-request"))

    assert rc == 1, out
    assert "no parent" in out
    assert box.tag_commit("v1.2.3") == ""


@pytest.mark.parametrize("value", ["True", "1", " true", "yes", "TRUE"])
def test_dry_run_flag_is_strict(box: Sandbox, value: str):
    """Anything but the exact words must fail loudly, never quietly mean "run for real"."""
    rc, out = box.run(box.request("Release v1.2.3"), DRY_RUN=value)

    assert rc == 2, out
    assert box.tag_commit("v1.2.3") == ""


def test_unknown_request_commit_is_a_usage_error(box: Sandbox):
    rc, out = box.run("0123456789abcdef0123456789abcdef01234567")

    assert rc == 2, out


@pytest.mark.parametrize("carried", ["1.2.30", "1.2.3.dev0", "1.2.3rc1"])
def test_a_version_that_merely_starts_like_the_request_is_refused(box: Sandbox, carried: str):
    box.set_version(carried)

    rc, out = box.run(box.request("Release v1.2.3"))

    assert rc == 1, out
    assert "__version__" in out
    assert box.all_tags() == ""


def test_a_commented_out_version_line_is_not_the_version(box: Sandbox):
    box.set_version("1.2.3", line='# __version__ = "9.9.9"\n__version__ = "1.2.3"')

    rc, out = box.run(box.request("Release v1.2.3"))

    assert rc == 0, out
    assert box.tag_commit("v1.2.3") == box.git("rev-parse", "master")


def test_dry_run_still_refuses_a_bad_request(box: Sandbox):
    request = box.request("Release v1.2.3")
    box.advance_master()

    rc, out = box.run(request, DRY_RUN="true")

    assert rc == 1, out
    assert "stale" in out


def test_missing_default_branch_is_refused_with_an_explanation(box: Sandbox):
    rc, out = box.run(box.request("Release v1.2.3"), DEFAULT_BRANCH="no-such-branch")

    assert rc == 1, out
    assert "does not exist" in out
    assert box.all_tags() == ""


def test_a_tag_named_like_the_branch_cannot_stand_in_for_master(box: Sandbox):
    """git resolves the bare name "origin/master" to refs/tags/ first; the script must not."""
    box.git("checkout", "-q", "-b", "side", "master")
    box.git("commit", "-q", "--allow-empty", "-m", "never merged to master")
    box.git("tag", "origin/master", "side")
    box.git("push", "-q", "origin", "side", "refs/tags/origin/master")
    request = box.request("Release v1.2.3", start="side")

    rc, out = box.run(request)

    assert rc == 1, out
    assert "stale" in out
    assert box.tag_commit("v1.2.3") == ""


def test_request_that_changes_files_is_refused_even_when_its_tag_already_exists(box: Sandbox):
    """Without this the run ends green as "already present" and the change is in no release."""
    assert box.run(box.request("Release v1.2.3"))[0] == 0

    rc, out = box.run(box.request("Release v1.2.3", change_a_file=True))

    assert rc == 1, out
    assert "not empty" in out


def test_merge_commit_is_not_a_request(box: Sandbox):
    box.git("checkout", "-q", "-b", "side", "master")
    box.git("commit", "-q", "--allow-empty", "-m", "side work")
    box.git("checkout", "-q", "-B", "release-request", "master")
    box.git("merge", "-q", "-s", "ours", "--no-ff", "-m", "Release v1.2.3", "side")
    box.git("push", "-q", "--force", "origin", "release-request")

    rc, out = box.run(box.git("rev-parse", "release-request"))

    assert rc == 1, out
    assert "merge commit" in out
    assert box.all_tags() == ""


def test_a_tag_that_never_reached_origin_is_not_reported_as_released(box: Sandbox):
    """A clone left over from a failed push has the tag locally only."""
    tip = box.git("rev-parse", "master")
    request = box.request("Release v1.2.3")

    rc, out = box.run(request, prepare=lambda clone: box.git("tag", "v1.2.3", tip, cwd=clone))

    assert rc == 1, out
    assert "NOT on origin" in out
    assert box.all_tags() == ""


# ---------------------------------------------------------------- the workflow around the script

def test_workflow_runs_this_script_for_this_branch_only():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "bash .github/scripts/release_tag.sh" in text
    assert "branches: [release-request]" in text
    assert "DEFAULT_BRANCH: master" in text
    assert "token: ${{ secrets.RELEASE_TAG_TOKEN }}" in text
    assert "github.token" not in text, "no fallback: a GITHUB_TOKEN tag starts no release"
    assert "contents: read" in text and "contents: write" not in text
    assert "Refuse without RELEASE_TAG_TOKEN" in text
    assert "environment: release-tag" in text
    assert "fetch-depth: 0" in text
    assert "github.repository == 'emilwalsted/respmech'" in text

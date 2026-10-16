"""Guard: internal planning identifiers must not appear in the published sources.

Comments and docstrings describe behaviour in plain words; a planning label like
"M-" followed by two digits says nothing to a reader of the public repository or of
the PyPI package. The pattern is assembled from two pieces so that this file does
not match itself.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"\b" + "M" + r"-[0-9]{2}\b")
SUFFIXES = {".py", ".md", ".toml", ".txt", ".rst", ".cfg", ".yml", ".yaml"}
SKIP_DIRS = {"__pycache__", ".git", "legacy", "input", "output"}


def _files():
    for sub in ("src", "tests", "docs"):
        base = ROOT / sub
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix in SUFFIXES and path.is_file() and not (SKIP_DIRS & set(path.parts)):
                yield path


def test_no_internal_plan_ids_under_src_tests_docs():
    offenders = []
    for path in _files():
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if PATTERN.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}")
    assert not offenders, "internal planning ids found:\n" + "\n".join(offenders[:20])


def test_the_pattern_itself_matches_a_planted_id():
    assert PATTERN.search("see " + "M" + "-17 for details")
    assert not PATTERN.search("ATM-171 or M-1")

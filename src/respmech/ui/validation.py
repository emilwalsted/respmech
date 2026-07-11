"""Filesystem-level validation shared by the Settings and Run screens.

The core ``Settings.validate()`` is deliberately filesystem-agnostic; this adds the
path checks (folders exist, files match, noise reference present) that both the
Settings "Validate" action and the Run screen need, so the two stay consistent.
Qt-free.
"""
from __future__ import annotations

import os

from respmech.core.pipeline import match_input_files


def matching_files(folder: str, mask: str) -> list:
    """Files under ``folder`` matching a possibly multi-pattern ``mask`` — patterns split on
    ';' or ',' so a mask like '*.csv; *.txt' works in the UI. Delegates to the core matcher
    (``match_input_files``) so the file list the UI shows is exactly the set the batch will
    process — case-insensitive and folder-metacharacter-safe on both platforms."""
    patterns = [p.strip() for p in (mask or "*.*").replace(";", ",").split(",") if p.strip()]
    out = set()
    for pat in (patterns or ["*.*"]):
        out.update(match_input_files(folder, pat))
    return sorted(out)


def path_problem(settings) -> str | None:
    """Return a human message for the first filesystem problem, or None if all paths
    are usable for a run."""
    s = settings
    folder = (s.input.folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return f"input folder does not exist: {folder or '(unset)'}"
    matches = matching_files(folder, s.input.files)
    if not matches:
        return f"no files match '{s.input.files}' in the input folder"
    out = (s.output.folder or "").strip()
    if not out:
        return "output folder is not set"
    parent = out if os.path.isdir(out) else os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(parent):
        return f"output folder's location does not exist: {out}"
    n = s.processing.emg.noise
    if n.enabled and n.reference_file:
        ref = n.reference_file
        if not os.path.isabs(ref):
            ref = os.path.join(folder, ref)
        if not os.path.isfile(ref):
            return f"noise reference file not found: {n.reference_file}"
    return None

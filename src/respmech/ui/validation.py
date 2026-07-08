"""Filesystem-level validation shared by the Settings and Run screens.

The core ``Settings.validate()`` is deliberately filesystem-agnostic; this adds the
path checks (folders exist, files match, noise reference present) that both the
Settings "Validate" action and the Run screen need, so the two stay consistent.
Qt-free.
"""
from __future__ import annotations

import glob
import os


def path_problem(settings) -> str | None:
    """Return a human message for the first filesystem problem, or None if all paths
    are usable for a run."""
    s = settings
    folder = (s.input.folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return f"input folder does not exist: {folder or '(unset)'}"
    matches = [f for f in glob.glob(os.path.join(folder, s.input.files or "*.*"))
               if os.path.isfile(f)]
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

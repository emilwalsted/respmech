"""Filesystem-level validation shared by the Settings and Run screens.

The core ``Settings.validate()`` is deliberately filesystem-agnostic; this adds the
path checks (folders exist, files match, noise reference present) that both the
Settings screen's live validation and the Run screen need, so the two stay consistent.
Qt-free.
"""
from __future__ import annotations

import os

# NB ``match_input_files`` is imported lazily inside ``matching_files`` below. It needs
# nothing but os/fnmatch, but importing it at module level drags the whole compute core
# -- scipy.interpolate, pandas, scipy.signal -- into GUI startup, which cost 1.4 s of the
# 2.0 s it took to open a window. See tests/unit/test_startup_imports.py.


def matching_files(folder: str, mask: str) -> list:
    """Files under ``folder`` matching a possibly multi-pattern ``mask`` — patterns split on
    ';' or ',' so a mask like '*.csv; *.txt' works in the UI. Delegates to the core matcher
    (``match_input_files``) so the file list the UI shows is exactly the set the batch will
    process — case-insensitive and folder-metacharacter-safe on both platforms."""
    from respmech.core.pipeline import match_input_files
    patterns = [p.strip() for p in (mask or "*.*").replace(";", ",").split(",") if p.strip()]
    out = set()
    for pat in (patterns or ["*.*"]):
        out.update(match_input_files(folder, pat))
    return sorted(out)


def path_problem(settings, probe_write: bool = False, matches: list | None = None) -> str | None:
    """Return a human message for the first filesystem problem, or None if all paths
    are usable for a run.

    ``probe_write`` (default False) additionally tries to actually create/write/clean up a
    file in the output folder (``core.io.plan.probe_write_folder`` — a real write, never
    ``os.access``, which is unreliable against Windows ACLs). Leave it False for Settings'
    live, every-keystroke validation: this function is called from there on every field
    edit (``ui/screens/settings_screen.py``'s ``_path_problem``), and a disk round-trip per
    keystroke is worst on exactly the kind of network/removable drive this whole check
    exists for. Only ``RunScreen._start`` (covers both Run and Dry run) and the output
    folder picker pass ``probe_write=True`` — see their call sites.

    ``matches`` (default None) lets a caller that has ALREADY globbed the input folder
    hand the result in, instead of this function globbing again — ``RunScreen`` (ticket
    B04) keeps its own glob memoised on (folder, mask, folder mtime) and passes it here so
    its every-keystroke enablement check stays cheap without this function growing its own,
    second cache. Omit it (the default) to glob fresh, as Settings' live validation does."""
    s = settings
    folder = (s.input.folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return f"input folder does not exist: {folder or '(unset)'}"
    if matches is None:
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
    if probe_write:
        from respmech.core.io.plan import probe_write_folder
        probe = probe_write_folder(out)
        if not probe.ok:
            return probe.message
    return None


def channel_collision(settings) -> str | None:
    """A HARD channel-mapping error (message, else None): a required channel
    (flow/poes/pgas/pdi) not assigned at all, one pointing at column 1 — the time axis —
    or two of them sharing a column.

    Moved here from the Settings screen (ticket B04) so the Run screen's always-visible
    commitment sheet can name exactly the same blocker Setup's live QC strip already
    names for the identical mapping — the two must never disagree about why a run is
    blocked, the same reasoning that keeps ``path_problem`` above shared rather than
    duplicated per screen."""
    ch = settings.input.channels
    req = [("flow", ch.flow), ("poes", ch.poes), ("pgas", ch.pgas), ("pdi", ch.pdi)]
    unset = [n for n, c in req if c is None]
    if unset:
        return (f"{', '.join(unset)} not assigned — "
                "click 'Assign channels from data…'")
    on_time = [n for n, c in req if c == 1]
    if on_time:
        return (f"{', '.join(on_time)} point at column 1 (the time axis) — "
                "click 'Assign channels from data…'")
    cols = [c for _n, c in req if c]
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        names = [n for n, c in req if c in dup]
        return f"{', '.join(names)} are mapped to the same column"
    return None

"""Load/save :class:`~respmech.core.settings.Settings` as TOML.

TOML is declarative (no code execution — unlike the legacy ``.py`` settings),
comment-friendly and diffable. Reading uses the stdlib ``tomllib`` (Python 3.11+);
writing uses ``tomli_w``.
"""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import tomli_w

# The one authoritative table of per-folder-tagged batch state — defined once in
# core.settings (as `_CARRIED_KINDS`; CarriedOverState/carried_over_state/
# clear_carried_over are its other consumers) and imported here under the name this
# module's own functions were already written against, so the two can never drift the
# way the six hand-spread call sites this replaces already had (B06/M-07).
from respmech.core.settings import Settings, _CARRIED_KINDS as _FOLDER_TAG_PATHS, _walk

_INDEX_RE = re.compile(r"^\[(\d+)\]$")


def _rebase_folders(settings: Settings, base: str) -> None:
    """Resolve relative input/output folders against ``base`` (the analysis file's own
    directory), not the process CWD. A frozen app's CWD is the install dir / System32, so
    a shared analysis carrying the default relative 'input'/'output' folders would
    otherwise read from and write to the wrong place (often a PermissionError after a full
    run). The EMG noise reference FILE is left as-is — a bare filename is resolved against
    the input folder downstream.

    The carried-folder provenance tags (every row in ``_FOLDER_TAG_PATHS`` — see
    core.settings.carried_over_state) are rebased the SAME way, for the same reason: they
    are compared directly against the live, rebased ``settings.input.folder``
    (core.settings.is_carried_folder), and a shared/moved study that only rebased
    input.folder itself would make every entry look falsely carried over the moment it was
    reopened somewhere else."""
    for obj, attr in ((settings.input, "folder"), (settings.output, "folder")):
        val = getattr(obj, attr)
        if val and not os.path.isabs(val):
            setattr(obj, attr, os.path.normpath(os.path.join(base, val)))
    for path, _kind, _name_of, _clear_fn in _FOLDER_TAG_PATHS:
        container, attr = _walk(settings, path)
        val = getattr(container, attr)
        if isinstance(val, list):
            for entry in val:
                if entry.folder and not os.path.isabs(entry.folder):
                    entry.folder = os.path.normpath(os.path.join(base, entry.folder))
        elif val and not os.path.isabs(val):
            setattr(container, attr, os.path.normpath(os.path.join(base, val)))


def load_toml(path: str | Path) -> Settings:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    settings = Settings.from_dict(data)
    _rebase_folders(settings, os.path.dirname(os.path.abspath(str(path))))
    return settings


def dumps_toml(settings: Settings) -> str:
    return tomli_w.dumps(_merge_unknown(_toml_clean(settings.to_dict()), settings.unknown))


def _relativize_folder(folder: str, base: str) -> str:
    """Inverse of :func:`_rebase_folders`: an absolute folder that lives at/under ``base``
    (the analysis file's own directory) is written back as a path relative to it, so a
    shared analysis stays portable. Folders outside ``base`` (an explicit external
    location) and different-drive paths are left absolute."""
    if not folder or not os.path.isabs(folder):
        return folder
    try:
        rel = os.path.relpath(folder, base)
    except ValueError:                    # different drive on Windows -> not relativizable
        return folder
    if rel == os.curdir or (rel != os.pardir and not rel.startswith(os.pardir + os.sep)):
        return rel
    return folder


def _walk_dict(data: dict, dotted_path: str) -> tuple[dict | None, str]:
    """Dict-shaped counterpart of ``core.settings._walk``, for the ``to_dict()``/TOML
    shape ``save_toml`` writes: resolve all but the last segment of ``dotted_path``
    against nested dicts (dataclass field names ARE the TOML/dict key names, so the SAME
    path strings from ``_FOLDER_TAG_PATHS`` apply unchanged). Returns ``(None, last
    segment)`` if an intermediate table is absent (dropped by ``_toml_clean`` because it
    was never set) — the caller then has nothing to relativize."""
    segments = dotted_path.split(".")
    cur = data
    for seg in segments[:-1]:
        cur = cur.get(seg) if isinstance(cur, dict) else None
    return (cur if isinstance(cur, dict) else None), segments[-1]


def save_toml(settings: Settings, path: str | Path) -> None:
    # Re-relativize input/output folders against the file's own directory — the inverse of
    # load_toml's rebase — so an Open→edit→Save cycle preserves a portable relative-path
    # analysis instead of silently baking in machine-absolute paths (which would break the
    # shared/moved-study workflow the rebase exists to enable). The run manifest
    # (dumps_toml, written into the output folder) deliberately keeps absolute paths for
    # reproducibility, so it is not routed through here.
    base = os.path.dirname(os.path.abspath(str(path)))
    data = _merge_unknown(_toml_clean(settings.to_dict()), settings.unknown)
    for section in ("input", "output"):
        sec = data.get(section)
        if isinstance(sec, dict) and sec.get("folder"):
            sec["folder"] = _relativize_folder(sec["folder"], base)
    # inverse of the carried-folder rebase in _rebase_folders — see its docstring.
    for path_, _kind, _name_of, _clear_fn in _FOLDER_TAG_PATHS:
        parent, attr = _walk_dict(data, path_)
        if parent is None:
            continue
        val = parent.get(attr)
        if isinstance(val, list):
            for entry in val:
                if isinstance(entry, dict) and entry.get("folder"):
                    entry["folder"] = _relativize_folder(entry["folder"], base)
        elif val:
            parent[attr] = _relativize_folder(val, base)
    with open(path, "wb") as f:
        f.write(tomli_w.dumps(data).encode("utf-8"))


def _toml_clean(obj):
    """TOML has no null. Drop keys whose value is None (they carry no info; the
    defaults fill them back in on load)."""
    if isinstance(obj, dict):
        return {k: _toml_clean(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_toml_clean(v) for v in obj]
    return obj


def _merge_unknown(data: dict, unknown: dict) -> dict:
    """Fold ``Settings.unknown``'s dotted-path entries back into ``data`` before it is
    written as TOML, so a key/table this version does not recognise survives a save
    instead of being silently dropped (a hand-edited or newer-version analysis file
    otherwise lost its extra tables the moment RespMech re-saved it).

    ``core.settings._build`` archives an unrecognised key in exactly three shapes, and
    every path in ``unknown`` is one of them:

    * a whole unknown top-level or nested TABLE (``"processing.lung_volumes"`` ->
      a dict) -- inserted as a table at that path;
    * a whole unknown LIST OF TABLES (``"processing.breath_types"`` -> a list) --
      inserted as a list at that path;
    * a single unrecognised FIELD inside one element of an otherwise-known list
      dataclass (``"processing.exclude_breaths.[0].some_future_field"``) -- inserted
      into that element only.

    Every path segment before the last therefore already corresponds to an
    already-serialised known field in ``data`` (an unknown value is archived whole,
    never recursed into further) -- except a list index whose entry has since been
    removed from the in-memory settings, which is dropped silently: the element it
    belonged to no longer exists, so there is nowhere left to put it back.

    KNOWN, ACCEPTED LIMITATIONS (found by self-review, deliberately not fixed here --
    this ticket's scope is making the existing M-01 archival format round-trip through
    a save, not redesigning it; each is pinned by a test in test_settings.py so a
    future change to this function doesn't silently alter the accepted shape):

    * A per-element unknown field is keyed by its list POSITION at load time
      (``"...[0].kind"``), not by a stable identity. If an EARLIER entry in that same
      list is removed from the in-memory settings before a save (not merely appended
      to or left alone), the value reattaches to whatever entry now sits at that index
      instead of being dropped -- silent misattribution to the wrong entry, which is
      worse than the "index now out of range" case above. This cannot happen from
      today's code (nothing yet lets a list carrying unknown per-entry data be edited
      in the same session), but will need a stable per-entry identity or per-entry
      unknown storage instead of this flat, position-keyed dict once such editing
      exists.
    * A literal ``.`` inside a TOML key name (e.g. a quoted ``"weird.key" = 5``) is
      indistinguishable from a path separator once archived as a string, so a save
      re-splits it and rebuilds it as a NESTED table instead of the original flat key.
      The value survives; only the shape changes. This can never collide with a real
      dataclass field (Python field names cannot contain ``.``), so it is a narrow
      concern for a hand-edited or foreign-tool TOML file using dotted key namespacing,
      not for RespMech's own output.
    """
    for path, value in unknown.items():
        segments = path.split(".")
        cur = data
        for seg in segments[:-1]:
            m = _INDEX_RE.match(seg)
            if m:
                idx = int(m.group(1))
                if not isinstance(cur, list) or idx >= len(cur):
                    cur = None
                    break
                cur = cur[idx]
            else:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.setdefault(seg, {})
        if isinstance(cur, dict):
            cur[segments[-1]] = value
    return data

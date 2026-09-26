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

from respmech.core.settings import Settings

_INDEX_RE = re.compile(r"^\[(\d+)\]$")


def _rebase_folders(settings: Settings, base: str) -> None:
    """Resolve relative input/output folders against ``base`` (the analysis file's own
    directory), not the process CWD. A frozen app's CWD is the install dir / System32, so
    a shared analysis carrying the default relative 'input'/'output' folders would
    otherwise read from and write to the wrong place (often a PermissionError after a full
    run). The EMG noise reference FILE is left as-is — a bare filename is resolved against
    the input folder downstream.

    The carried-folder provenance tags (ExcludeEntry/BreathCountEntry.folder,
    NoiseSettings.reference_folder — see core.settings.carried_over_state) are rebased the
    SAME way, for the same reason: they are compared directly against the live, rebased
    ``settings.input.folder`` (core.settings.is_carried_folder), and a shared/moved study
    that only rebased input.folder itself would make every entry look falsely carried over
    the moment it was reopened somewhere else."""
    for obj, attr in ((settings.input, "folder"), (settings.output, "folder")):
        val = getattr(obj, attr)
        if val and not os.path.isabs(val):
            setattr(obj, attr, os.path.normpath(os.path.join(base, val)))
    for entry in (*settings.processing.exclude_breaths, *settings.processing.breath_counts):
        if entry.folder and not os.path.isabs(entry.folder):
            entry.folder = os.path.normpath(os.path.join(base, entry.folder))
    noise = settings.processing.emg.noise
    if noise.reference_folder and not os.path.isabs(noise.reference_folder):
        noise.reference_folder = os.path.normpath(os.path.join(base, noise.reference_folder))


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
    proc = data.get("processing")
    if isinstance(proc, dict):
        for key in ("exclude_breaths", "breath_counts"):
            for entry in proc.get(key) or ():
                if isinstance(entry, dict) and entry.get("folder"):
                    entry["folder"] = _relativize_folder(entry["folder"], base)
        noise = (proc.get("emg") or {}).get("noise")
        if isinstance(noise, dict) and noise.get("reference_folder"):
            noise["reference_folder"] = _relativize_folder(noise["reference_folder"], base)
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
    belonged to no longer exists, so there is nowhere left to put it back."""
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

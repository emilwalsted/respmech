"""Load/save :class:`~respmech.core.settings.Settings` as TOML.

TOML is declarative (no code execution — unlike the legacy ``.py`` settings),
comment-friendly and diffable. Reading uses the stdlib ``tomllib`` (Python 3.11+);
writing uses ``tomli_w``.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w

from respmech.core.settings import Settings


def _rebase_folders(settings: Settings, base: str) -> None:
    """Resolve relative input/output folders against ``base`` (the analysis file's own
    directory), not the process CWD. A frozen app's CWD is the install dir / System32, so
    a shared analysis carrying the default relative 'input'/'output' folders would
    otherwise read from and write to the wrong place (often a PermissionError after a full
    run). The EMG noise reference is left as-is — a bare filename is resolved against the
    input folder downstream."""
    for obj, attr in ((settings.input, "folder"), (settings.output, "folder")):
        val = getattr(obj, attr)
        if val and not os.path.isabs(val):
            setattr(obj, attr, os.path.normpath(os.path.join(base, val)))


def load_toml(path: str | Path) -> Settings:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    settings = Settings.from_dict(data)
    _rebase_folders(settings, os.path.dirname(os.path.abspath(str(path))))
    return settings


def dumps_toml(settings: Settings) -> str:
    return tomli_w.dumps(_toml_clean(settings.to_dict()))


def save_toml(settings: Settings, path: str | Path) -> None:
    with open(path, "wb") as f:
        f.write(dumps_toml(settings).encode("utf-8"))


def _toml_clean(obj):
    """TOML has no null. Drop keys whose value is None (they carry no info; the
    defaults fill them back in on load)."""
    if isinstance(obj, dict):
        return {k: _toml_clean(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_toml_clean(v) for v in obj]
    return obj

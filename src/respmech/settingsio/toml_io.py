"""Load/save :class:`~respmech.core.settings.Settings` as TOML.

TOML is declarative (no code execution — unlike the legacy ``.py`` settings),
comment-friendly and diffable. Reading uses the stdlib ``tomllib`` (Python 3.11+);
writing uses ``tomli_w``.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w

from respmech.core.settings import Settings


def load_toml(path: str | Path) -> Settings:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Settings.from_dict(data)


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

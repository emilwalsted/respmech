"""Small, cross-platform user-preferences store (features P25 / P26).

Wraps ``QSettings`` (native backend per OS — registry / plist / ini) so the app can
remember, across sessions:

* recently-opened analyses (P26) — shown on the startup screen;
* the last folder used for each kind of file dialog (P26) — so a browse starts where
  the last one left off, not at the CWD;
* the last "rig" (P25) — the channel mapping + sampling format of the last analysis,
  so a new analysis can inherit the hardware layout rather than re-entering it.

Everything is best-effort: a missing/backend-less environment degrades to sane
defaults and never raises, so headless tests and first runs are unaffected.
"""
from __future__ import annotations

ORG = "RespMech"
APP = "RespMech"
_MAX_RECENT = 8


def _qsettings():
    try:
        from PySide6.QtCore import QSettings
        return QSettings(ORG, APP)
    except Exception:                           # pragma: no cover - Qt-less environment
        return None


# --- recent analyses (P26) --------------------------------------------------
def recent_analyses() -> list[str]:
    import os
    s = _qsettings()
    if s is None:
        return []
    raw = s.value("recent_analyses", []) or []
    if isinstance(raw, str):                    # some backends collapse a 1-item list
        raw = [raw]
    # drop entries that no longer exist on disk, keep order
    return [p for p in raw if isinstance(p, str) and os.path.isfile(p)]


def add_recent_analysis(path: str) -> None:
    import os
    s = _qsettings()
    if s is None or not path:
        return
    path = os.path.abspath(path)
    items = [p for p in ([path] + recent_analyses()) if p]
    seen, ordered = set(), []
    for p in items:
        if p not in seen:
            seen.add(p); ordered.append(p)
    s.setValue("recent_analyses", ordered[:_MAX_RECENT])


# --- sticky folders (P26) ---------------------------------------------------
def last_folder(key: str, fallback: str = "") -> str:
    import os
    s = _qsettings()
    if s is None:
        return fallback
    p = s.value(f"last_folder/{key}", "") or ""
    return p if p and os.path.isdir(p) else fallback


def set_last_folder(key: str, path: str) -> None:
    import os
    s = _qsettings()
    if s is None or not path:
        return
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    if folder:
        s.setValue(f"last_folder/{key}", folder)


# --- last rig (P25) ---------------------------------------------------------
def save_rig(settings) -> None:
    """Remember the channel mapping + sampling format of an analysis as the 'last rig'."""
    s = _qsettings()
    if s is None:
        return
    try:
        ch = settings.input.channels
        rig = {"poes": ch.poes, "pgas": ch.pgas, "pdi": ch.pdi, "volume": ch.volume,
               "flow": ch.flow, "emg": list(ch.emg), "entropy": list(ch.entropy),
               "sampling_frequency": settings.input.format.sampling_frequency}
        s.setValue("last_rig", rig)
    except Exception:                           # pragma: no cover - defensive
        pass


def last_rig() -> dict | None:
    s = _qsettings()
    if s is None:
        return None
    rig = s.value("last_rig", None)
    return rig if isinstance(rig, dict) and rig else None


def apply_rig(settings, rig: dict) -> None:
    """Apply a saved rig onto a Settings object (channel columns + sampling only)."""
    if not rig:
        return
    ch = settings.input.channels
    for k in ("poes", "pgas", "pdi", "volume", "flow"):
        if rig.get(k) is not None:
            setattr(ch, k, int(rig[k]))
    if rig.get("emg"):
        ch.emg = [int(x) for x in rig["emg"]]
    if rig.get("entropy"):
        ch.entropy = [int(x) for x in rig["entropy"]]
    if rig.get("sampling_frequency"):
        settings.input.format.sampling_frequency = int(rig["sampling_frequency"])

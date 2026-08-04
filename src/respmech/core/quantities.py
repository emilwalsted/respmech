"""The size registry: what a result column is called, what unit it is in, and (in time)
what a human should call it (features P10 / P21).

The result tables use terse, script-friendly column names (``poes_mininsp``,
``wobtotal``, ``vt`` …). For a *publishable* export the reader needs the unit each
number is in — but the golden suite pins the exact column-*names*, so units must be
carried alongside the data (a Units sheet / a second header row / a table header),
never baked into the column names.

Units are resolved by matching a column name against ordered prefix/suffix rules.
We only assert a unit where the physiology is unambiguous (pressures, volumes,
flows, times, rates, work); EMG amplitude and sample entropy are reported as
arbitrary / dimensionless because this pipeline does not calibrate them, and a
handful of derived ratios are deliberately left blank rather than mislabelled.

This module was ``core/units.py`` (ticket A04, UI-overhaul); it is renamed here
because the registry now also carries a ``display`` field (a future human-readable
name, e.g. "Poes PTP, inspiratory"), so "quantities" better names what it is than
"units" did. ``unit_for()``/``units_map()`` are unchanged: same rules, same unit
strings, same behaviour. The display names themselves are NOT filled in by this
ticket — roughly 50 physiological result columns would each need a reviewed name,
and that table is Emil's call, not a guess made here. ``display_for()`` falls back
to the column's own identifier until that table exists.

This module stays Qt-free: it is read by both ``core/io/writers.py`` (the Excel
Units sheet) and ``ui/`` (the on-screen result tables), and only ``ui/`` may import
Qt.
"""
from __future__ import annotations

CMH2O = "cmH₂O"
LITRE = "L"
LPS = "L/s"
SECOND = "s"
DIMLESS = "—"          # a genuine ratio / dimensionless quantity
ARB = "a.u."           # uncalibrated (EMG amplitude in this pipeline)

# Ordered (predicate, unit) rules — first match wins. Predicates take the lower-cased
# column name. Order matters: the most specific patterns come first.
_RULES: list[tuple] = [
    # --- EMG (checked first: names contain 'emg'/'rms', not a pressure) ----------
    (lambda c: c.startswith("integral_emg") or c.startswith("integralemg"), f"{ARB}·s"),
    (lambda c: c.startswith("rms"), ARB),
    # --- sample entropy ----------------------------------------------------------
    (lambda c: c.startswith("sample_entropy"), DIMLESS),
    # --- pressure-time products / integrals (before the plain-pressure rule) -----
    # PTP is scaled by breaths·min⁻¹ in calcptp (× bcnt · vefactor), so it is a rate;
    # the paired int_* value is the un-scaled per-breath integral.
    (lambda c: c.startswith("ptp_"), f"{CMH2O}·{SECOND}·min⁻¹"),
    (lambda c: c.startswith("int_"), f"{CMH2O}·{SECOND}"),
    # --- pressures ---------------------------------------------------------------
    (lambda c: c.startswith("poes") or c.startswith("pgas") or c.startswith("pdi"), CMH2O),
    (lambda c: c.endswith("tidal_swing") or c.endswith("_rise"), CMH2O),
    # --- flows (before volumes: 'flow' names also contain 'vol') ------------------
    (lambda c: "flow" in c, LPS),
    # --- volumes -----------------------------------------------------------------
    (lambda c: c.startswith("vol_") or c == "vt", LITRE),
    # --- work of breathing (scaled × bcnt · vefactor → a per-minute power) --------
    (lambda c: c.startswith("wob"), "J·min⁻¹"),
    # --- timing ------------------------------------------------------------------
    (lambda c: c == "ti_ttot", DIMLESS),
    (lambda c: c in ("ti", "te", "ttot"), SECOND),
    # --- ventilation / rate ------------------------------------------------------
    (lambda c: c == "bf", "min⁻¹"),
    (lambda c: c == "ve", "L/min"),
]

# Human-readable display names, keyed by the exact lower-cased column identifier.
# Deliberately empty (see the module docstring): populating this needs a reviewed
# physiological name for each of ~50 result columns, which is out of scope here.
# display_for() falls back to the column's own identifier while this stays empty.
_DISPLAY: dict[str, str] = {}


def unit_for(column: str) -> str:
    """The unit string for a result column, or "" when it is unknown/left blank."""
    c = str(column).lower()
    if c in ("file", "breath_no", "breathno"):
        return ""
    for pred, unit in _RULES:
        try:
            if pred(c):
                return unit
        except Exception:                       # pragma: no cover - defensive
            continue
    return ""



def units_map(columns) -> dict[str, str]:
    """{column: unit} keeping only the columns that resolved to a real unit."""
    out = {}
    for c in columns:
        u = unit_for(c)
        if u:
            out[str(c)] = u
    return out


def display_for(column: str) -> str:
    """The human-readable name for a result column, or the column's own identifier
    when no name is registered yet (see the module docstring: the name table is not
    populated by this change)."""
    return _DISPLAY.get(str(column).lower(), str(column))

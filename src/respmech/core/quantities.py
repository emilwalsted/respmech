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
LPS = "L·s⁻¹"
SECOND = "s"
DIMLESS = "—"          # a genuine ratio / dimensionless quantity
ARB = "a.u."           # uncalibrated (EMG amplitude in this pipeline)

# Ordered (predicate, unit) rules — first match wins. Predicates take the lower-cased
# column name. Order matters: the most specific patterns come first.
_RULES: list[tuple] = [
    # --- generic naming conventions (checked FIRST: a suffix/prefix convention
    # future columns opt into, so it must win over the more specific rules below
    # that would otherwise misclassify it, e.g. 'rms_col_2_pct' would hit the
    # EMG 'rms' rule and 't_peak_in_flow' would hit the 'flow' rule). Suffixes
    # are all checked before any of this block's prefixes, not interleaved: a
    # name can match both (e.g. a future coefficient-of-variation of a timing
    # value, 't_peak_cv', or of a PEEPi value, 'peepi_dyn_cv') and the suffix is
    # always the more specific, intended classification in that case. -----------
    (lambda c: c.endswith("_pct") or "_pct_" in c, "%"),
    (lambda c: c.endswith("_frac"), DIMLESS),
    (lambda c: c.endswith("_cv"), "%"),
    (lambda c: c.endswith("_db"), "dB"),
    (lambda c: c.startswith("t_"), SECOND),
    (lambda c: c.startswith("peepi_lag"), SECOND),
    (lambda c: c.startswith("peepi"), CMH2O),
    (lambda c: c.startswith("tt_"), DIMLESS),
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
    (lambda c: c == "ve", "L·min⁻¹"),
]

# Human-readable display names, keyed by the exact lower-cased column identifier.
# Deliberately empty (see the module docstring): populating this needs a reviewed
# physiological name for each of ~50 result columns, which is out of scope here.
# display_for() falls back to the column's own identifier while this stays empty.
_DISPLAY: dict[str, str] = {}


def _registry_unit_for(c: str) -> str | None:
    """Consult ``core/analysis/registry.py``'s ``REGISTRY`` for a column that
    ``_RULES`` above leaves unclassified, matching by exact name or by
    ``prefix=``. Returns ``None`` when the registry has no opinion either
    (every entry today is ``unit=None`` — the legacy mechanics block still gets
    its units from ``_RULES`` alone, unchanged; see registry.py's own
    docstring). A registered ``unit=None`` is treated the same as no entry at
    all (never returned): an exact name always wins over a prefix family
    regardless of ``REGISTRY`` order, and a spec with no opinion is skipped
    rather than short-circuiting the search, so a broad, early ``prefix=``
    family (like today's ``rms_``/``sample_entropy_``) can never hide a later,
    more specific entry that does declare a real unit.

    Imported lazily, not at module level: ``quantities.py`` is read by both
    ``core/io/writers.py`` and ``ui/`` (module docstring), and
    ``test_startup_imports.py`` pins that importing the GUI shell must not drag
    ``core.analysis.registry`` in at import time (it is "not wired into the GUI
    yet"). A lazy import here only touches ``sys.modules`` the first time
    ``unit_for`` actually falls through to it, never on a bare ``import``."""
    from respmech.core.analysis.registry import REGISTRY

    for spec in REGISTRY:
        if spec.name is not None and spec.unit is not None and c == spec.name.lower():
            return spec.unit
    for spec in REGISTRY:
        if spec.prefix is not None and spec.unit is not None and c.startswith(spec.prefix.lower()):
            return spec.unit
    return None


def unit_for(column: str) -> str:
    """The unit string for a result column, or "" when it is unknown/left blank.

    ``_RULES`` (generic naming conventions) is consulted first; for a name it
    leaves unclassified, the registry's declared ``ColumnSpec.unit`` (exact
    name or ``prefix=``) is used when one is registered and not itself
    ``None``. Neither source having an opinion also resolves to ""."""
    c = str(column).lower()
    if c in ("file", "breath_no", "breathno"):
        return ""
    for pred, unit in _RULES:
        try:
            if pred(c):
                return unit
        except Exception:                       # pragma: no cover - defensive
            continue
    reg_unit = _registry_unit_for(c)
    return reg_unit if reg_unit is not None else ""



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

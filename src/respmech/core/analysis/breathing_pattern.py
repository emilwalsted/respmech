"""Opt-in breathing-pattern columns from flow and volume alone
(``processing.breathing_pattern.extended`` / ``.variability``).

Everything here is computed from the raw breath dicts (or, for the per-file variability, from
the per-breath table the ordinary mechanics already produced) and reads NO pressure channel,
so it applies to a Flow-only analysis just as to a full one. Nothing here changes an existing
column: with both flags off the module is never called, and with either on it only ADDS
columns (docs/REVERSE_ENGINEERING.md §5.18).

Per breath (``extended``), stored in ``breath["breathing_pattern_ext"]`` for
``core.results.build_breath_table`` to join after the other blocks:

``mean_in_flow`` = ``vt / ti`` and ``mean_ex_flow`` = ``vt / te`` (L·s⁻¹): the mean
    inspiratory flow is the drive component of the breathing pattern (Milic-Emili & Grunstein
    1976; Tobin 1983), so it is defined on the breath's own ``vt``, exactly as the ordinary
    timing columns define ``vt``, ``ti`` and ``te``;
``vol_insp`` / ``vol_exp`` (L): the volume moved in each phase (range of the phase's volume
    trace), which equals ``vt`` for a clean breath and differs from it when the volume signal
    drifts within the breath;
``bf_inst`` = ``60 / ttot`` (min⁻¹): the frequency this single breath would give if repeated,
    unlike ``bf``, which scales the file's breath count;
``t_peak_in_flow`` / ``t_peak_ex_flow`` (s): time from the start of the phase to its peak flow,
    counted in samples like ``ti`` and ``te``; ``t_peak_in_flow_frac`` (—) is the inspiratory
    one as a fraction of ``ti``.

Per file (``variability``), added to the file's average row: the coefficient of variation (%)
of ``vt``, ``ti``, ``te``, ``ttot`` and ``ti_ttot`` over the included breaths (``vt_cv`` …
``ti_ttot_cv``), and ``n_breaths``, the number of breaths they rest on. The CV follows the
cohort summary's own convention (``core.summary._stats_frame``): sample SD (ddof = 1) over a
positive mean, otherwise NaN. Below ``MIN_BREATHS_FOR_CV`` breaths it is also NaN, because a
spread from two numbers is not a variability (Tobin 1983; Wysocki 2006 report it over
series of breaths).
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

#: Fewest included breaths a per-file CV is reported for; below it the CV columns are NaN
#: while ``n_breaths`` still says how many breaths there were.
MIN_BREATHS_FOR_CV = 3

EXTENDED_COLUMNS = (
    "mean_in_flow", "mean_ex_flow", "vol_insp", "vol_exp", "bf_inst",
    "t_peak_in_flow", "t_peak_ex_flow", "t_peak_in_flow_frac",
)

#: (output column, source column in the per-breath table)
VARIABILITY_SOURCES = (
    ("vt_cv", "vt"), ("ti_cv", "ti"), ("te_cv", "te"),
    ("ttot_cv", "ttot"), ("ti_ttot_cv", "ti_ttot"),
)
VARIABILITY_COLUMNS = tuple(c for c, _s in VARIABILITY_SOURCES) + ("n_breaths",)


def _nan_columns() -> "OrderedDict[str, float]":
    return OrderedDict((c, math.nan) for c in EXTENDED_COLUMNS)


def _range(x) -> float:
    a = np.asarray(x, dtype=float).reshape(-1)
    return float(np.max(a) - np.min(a)) if a.size else math.nan


def _div(num: float, den: float) -> float:
    return float(num / den) if den and math.isfinite(num) and math.isfinite(den) else math.nan


def breath_columns(breath, fs: float) -> "OrderedDict[str, float]":
    """The ``extended`` columns for one raw breath dict (``inspiration`` / ``expiration``
    phases plus the whole-breath ``flow`` / ``volume``), sampled at ``fs`` Hz. Inspiratory
    flow is negative and expiratory flow positive, as everywhere else in the mechanics.
    A degenerate breath (an empty phase) gives NaN for what it cannot define."""
    insp, exp = breath["inspiration"], breath["expiration"]
    n_insp = int(np.asarray(insp["flow"]).size)
    n_exp = int(np.asarray(exp["flow"]).size)
    n_tot = int(np.asarray(breath["flow"]).size)
    ti, te, ttot = n_insp / fs, n_exp / fs, n_tot / fs
    vt = _range(breath["volume"])

    out = _nan_columns()
    out["mean_in_flow"] = _div(vt, ti)
    out["mean_ex_flow"] = _div(vt, te)
    out["vol_insp"] = _range(insp["volume"])
    out["vol_exp"] = _range(exp["volume"])
    out["bf_inst"] = _div(60.0, ttot)
    if n_insp:
        out["t_peak_in_flow"] = float(np.argmin(np.asarray(insp["flow"], dtype=float))) / fs
        out["t_peak_in_flow_frac"] = _div(out["t_peak_in_flow"], ti)
    if n_exp:
        out["t_peak_ex_flow"] = float(np.argmax(np.asarray(exp["flow"], dtype=float))) / fs
    return out


def attach(breath, settings) -> "str | None":
    """Stamp ``breath["breathing_pattern_ext"]`` (the ``extended`` columns). A fault in one
    breath leaves NaN for that breath's new columns and returns a short notice instead of
    failing the file, the same isolation the other opt-in column blocks use."""
    fs = settings.input.format.samplingfrequency
    try:
        breath["breathing_pattern_ext"] = breath_columns(breath, fs)
    except Exception as e:                       # noqa: BLE001 - isolate to the new columns
        breath["breathing_pattern_ext"] = _nan_columns()
        return f"breathing pattern failed: {type(e).__name__}: {e}"
    return None


def cv_pct(values) -> float:
    """Coefficient of variation in percent over the finite entries of ``values``: sample SD
    (ddof = 1) over a positive mean, NaN otherwise or with fewer than
    :data:`MIN_BREATHS_FOR_CV` values."""
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < MIN_BREATHS_FOR_CV:
        return math.nan
    mean = float(np.mean(v))
    if not mean > 0:
        return math.nan
    return float(100.0 * np.std(v, ddof=1) / mean)


def file_variability(breaths_table) -> "OrderedDict[str, float]":
    """The per-file ``variability`` columns from one file's per-breath table (one row per
    included breath). A source column the table lacks gives NaN."""
    out: "OrderedDict[str, float]" = OrderedDict()
    for col, src in VARIABILITY_SOURCES:
        out[col] = cv_pct(breaths_table[src].to_numpy(dtype=float)) if src in breaths_table else math.nan
    out["n_breaths"] = float(len(breaths_table))
    return out


def attach_variability(breaths_table, average_row) -> None:
    """Add the per-file variability columns to the file's average row, in place."""
    for col, value in file_variability(breaths_table).items():
        average_row[col] = value

"""Cohort-level aggregation of the per-file averages (features P8 / P15).

The core already emits one average row per file (``result.average_table``). For a
study you also want the numbers you would put in a paper: for each variable, the
**mean ± SD across files**, the **n**, and the **coefficient of variation** (CV%);
and, when files fall into groups (subjects / conditions), the same broken down
**by group** so conditions can be compared.

This is computed at write time from the finished average table — it never touches
the per-file / per-breath tables the golden suite pins, so it is purely additive.

Grouping (P15) uses a documented default: the **leading token** of the file name
(everything up to the first space, underscore or hyphen), e.g. ``P03_120W.csv`` →
``P03``. A study can override it with an ``output.group_regex`` whose first capture
group is the key; if the regex does not match, the file falls in group ``"(all)"``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_LEADING_TOKEN = re.compile(r"^([^ _\-]+)")


def group_key(filename: str, settings=None) -> str:
    """The group a file belongs to (default: its leading filename token)."""
    name = str(filename)
    regex = getattr(getattr(settings, "output", None), "group_regex", None) if settings else None
    if regex:
        try:
            m = re.search(regex, name)
            if m:
                # an optional capture group that didn't participate is None — fall back
                # to the whole match rather than returning None (which would break sort)
                key = m.group(1) if (m.groups() and m.group(1) is not None) else m.group(0)
                return str(key)
        except re.error:                        # a bad pattern shouldn't crash a run
            pass
        return "(all)"
    m = _LEADING_TOKEN.match(name)
    return m.group(1) if m else name


def _stats_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """n / mean / SD / CV% for every numeric column of ``rows`` (one row per file)."""
    numeric = rows.select_dtypes(include="number")
    recs = []
    for col in numeric.columns:
        v = numeric[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        n = int(v.size)
        mean = float(np.mean(v)) if n else float("nan")
        sd = float(np.std(v, ddof=1)) if n > 1 else float("nan")
        # CV% is only defined for a positive, ratio-scale mean; a signed quantity
        # (e.g. poes_mininsp, which is negative) has no meaningful CV, so report NaN.
        cv = float(100.0 * sd / mean) if (n > 1 and np.isfinite(mean) and mean > 0) else float("nan")
        recs.append({"variable": col, "n": n, "mean": mean, "sd": sd, "cv_pct": cv})
    return pd.DataFrame(recs, columns=["variable", "n", "mean", "sd", "cv_pct"])


def normalize_emg_table(breaths_table, settings) -> "pd.DataFrame | None":
    """Per-file-normalised EMG RMS (feature P14).

    Each RMS amplitude column is re-expressed as a percentage of a per-file reference
    (the maximum breath, or the mean breath) so EMG amplitudes — which are otherwise
    in arbitrary, electrode-dependent units — can be compared across files/subjects.
    Returns a ``breath_no`` + normalised-columns DataFrame, or None when normalisation
    is off, there is no EMG, or the table is empty. The raw RMS is never changed."""
    mode = getattr(getattr(getattr(settings, "processing", None), "emg", None), "normalization", "none")
    if mode in (None, "none") or breaths_table is None or len(breaths_table) == 0:
        return None
    rms_cols = [c for c in breaths_table.columns if str(c).lower().startswith("rms")]
    if not rms_cols:
        return None
    out = pd.DataFrame()
    if "breath_no" in breaths_table.columns:
        out["breath_no"] = breaths_table["breath_no"].to_numpy()
    for c in rms_cols:
        v = breaths_table[c].to_numpy(dtype=float)
        if not np.isfinite(v).any():
            # An entirely blank column is now a normal outcome, not a fault: the
            # cardiac-gated columns are NaN for a whole file whose quality guards refused it
            # (too fast a heart rate, too little heartbeat-free signal). nanmax/nanmean would
            # emit "All-NaN slice encountered" and return NaN anyway, so short-circuit to the
            # same answer without the noise.
            out[f"{c}_pct"] = np.full_like(v, np.nan)
            continue
        ref = np.nanmax(v) if mode == "per_file_max" else np.nanmean(v)
        out[f"{c}_pct"] = (100.0 * v / ref) if (ref and not np.isnan(ref)) else np.full_like(v, np.nan)
    return out


def build_cohort_summary(result, settings=None) -> dict:
    """Return {"summary": df, "by_group": df or None} aggregated across files.

    ``summary`` is the whole-cohort n/mean/SD/CV% per variable. ``by_group`` is the
    same per group (subject/condition), stacked, with a leading ``group`` column;
    it is None when every file lands in a single group (nothing to compare)."""
    avg = getattr(result, "average_table", None)
    if avg is None or len(avg) == 0:
        return {"summary": pd.DataFrame(), "by_group": None}
    rows = avg.reset_index(drop=True)

    summary = _stats_frame(rows)

    by_group = None
    if "file" in rows.columns:
        keys = [group_key(f, settings) for f in rows["file"]]
        if len(set(keys)) > 1:                  # only worth a breakdown if >1 group
            parts = []
            for g in sorted(set(keys)):
                sub = rows.loc[[k == g for k in keys]]
                sf = _stats_frame(sub)
                sf.insert(0, "group", g)
                sf.insert(1, "n_files", len(sub))
                parts.append(sf)
            by_group = pd.concat(parts, ignore_index=True)
    return {"summary": summary, "by_group": by_group}

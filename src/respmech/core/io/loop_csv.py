"""The averaged Campbell / flow-volume loop as a table (opt-in ``output.data.save_pv_average_csv``).

The average breath is computed by ``core.compute.calculateaveragebreaths`` (every kept
breath's inspiration and expiration resampled to the same number of points and averaged
point by point) and drawn by the Campbell PDF and by Preview. This module writes the very
arrays those figures are drawn from, so a loop can be compared between conditions
numerically instead of by eye.

Nothing here computes anything: ``volumeavg``/``poesavg`` are read as the engine left
them (same sign and drift correction as the workbooks), the figure's inverted x-axis and
transposition are a drawing choice and do not reach the table. Qt-free, so the Preview
export and the batch writer share one definition of the columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = ("inspiration", "expiration")


def loop_csv_name(fname: str) -> str:
    """File name of one recording's table (the sibling of ``<file> – Campbell (average).pdf``)."""
    return f"{fname} – Campbell (average).csv"


COHORT_CSV_NAME = "All files – Campbell (average).csv"


def average_loop_table(breaths) -> pd.DataFrame | None:
    """The average breath of one recording as ``Phase``, ``Point``, ``Volume`` [, ``Poes``].

    ``breaths`` is any iterable of the file's breath dicts (``FileResult.breaths.values()``
    or Preview's cache); excluded breaths are skipped and the first kept one is read, since
    every breath of a file carries the same average (the figures read it from there too).
    ``Point`` counts 1..N within each phase. ``Poes`` is present only when the analysis has
    Poes. ``None`` when there is no average breath (no kept breath, or none with an average).
    """
    kept = [b for b in (breaths or []) if not b.get("ignored")]
    if not kept:
        return None
    b = kept[0]
    phases = {}
    for ph in PHASES:
        v = (b.get(ph) or {}).get("volumeavg")
        if v is None or not len(v):
            return None
        phases[ph] = (np.asarray(v, dtype=float),
                      (b.get(ph) or {}).get("poesavg"))
    has_poes = all(p is not None and len(p) == len(phases[ph][0]) for ph, (_v, p) in phases.items())
    cols = {
        "Phase": [ph for ph in PHASES for _ in range(len(phases[ph][0]))],
        "Point": [i + 1 for ph in PHASES for i in range(len(phases[ph][0]))],
        "Volume": np.concatenate([phases[ph][0] for ph in PHASES]),
    }
    if has_poes:
        cols["Poes"] = np.concatenate([np.asarray(phases[ph][1], dtype=float) for ph in PHASES])
    return pd.DataFrame(cols)


def cohort_loop_table(tables: dict) -> pd.DataFrame | None:
    """One table for the whole batch: ``File`` first, then each recording's own columns.

    ``tables`` maps file name to its :func:`average_loop_table`. ``Poes`` is filled where a
    recording has it and left empty where it does not (a batch has one signal set, so in
    practice every file agrees)."""
    parts = []
    for fname, t in tables.items():
        if t is None:
            continue
        t = t.copy()
        t.insert(0, "File", fname)
        parts.append(t)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)

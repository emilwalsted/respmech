#!/usr/bin/env python3
"""Generate/verify the synthetic golden from the NEW core (src/respmech).

The golden reference is the newest code's output (Emil's ground-truth decision).
This regenerates golden_reference.json from the new core and, on a plain run,
reports any column that differs from the committed reference so deliberate changes
(e.g. the bug #2 entropy fix) are visible.

    python golden_newcore.py            # compare to committed golden
    python golden_newcore.py --write    # regenerate golden_reference.json
"""
import os
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import make_golden as mg  # noqa: E402  (scenario definitions + synthetic settings)
from respmech.settingsio.migrate import migrate_dict  # noqa: E402
from respmech.settingsio.toml_io import load_toml  # noqa: E402
from respmech.core.pipeline import run_batch  # noqa: E402

GOLDEN = os.path.join(HERE, "golden_reference.json")


def _jsonify(df):
    out = {}
    for col in df.columns:
        vals = []
        for v in df[col].tolist():
            vals.append("NaN" if isinstance(v, float) and np.isnan(v) else v)
        out[str(col)] = vals
    return out


def _processed_summary(df):
    summary = {"shape": list(df.shape), "columns": list(map(str, df.columns)), "colstats": {}}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            v = df[col].values
            summary["colstats"][str(col)] = {
                "sum": float(np.nansum(v)), "mean": float(np.nanmean(v)),
                "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    return summary


def run_scenario(name):
    """Build and run one scenario's settings, then serialise its numeric output.

    A LEGACY_SCENARIOS entry is a dict override on ``base_settings()``, migrated
    through ``migrate_dict`` exactly as before (unchanged, so every existing golden
    stays byte-identical). A V2_SCENARIOS entry is a committed
    ``tests/golden/scenarios/<name>.toml`` file loaded via ``load_toml`` instead —
    the only way to express settings a legacy dict has no shape for (typed breaths,
    references, separators, ...). There is deliberately no ``V2_OVERRIDES``-style
    hook layered on top of ``migrate_dict``: a v2 scenario is its own committed TOML
    file, not a legacy dict with v2 fields bolted on."""
    if name in mg.V2_SCENARIOS:
        settings = load_toml(os.path.join(HERE, mg.V2_SCENARIOS[name]))
        settings.output.folder = os.path.join(HERE, "_work", "newcore", name)
    else:
        d = mg.base_settings(os.path.join(HERE, "_work", "newcore", name))
        mg.deep_update(d, mg.SCENARIOS[name])
        settings, _ = migrate_dict(d)
    res = run_batch(settings)

    # Guarded on breaths_table (rather than assumed present): a reference-only file
    # (no tidal breaths, a future scenario kind) reports role=="reference" with
    # breaths_table/average_row both None — nothing for THIS harness to serialise,
    # not an error. Every scenario today produces a tidal breaths_table for every
    # ok file, so this guard changes nothing yet.
    out = {"average_breathdata": None, "per_file": {}, "processed": {}}
    if res.average_table is not None:
        out["average_breathdata"] = _jsonify(res.average_table.reset_index(drop=True))
    for fname, fr in res.ok_files.items():
        if fr.breaths_table is not None:
            out["per_file"][f"{fname}.breathdata.xlsx"] = _jsonify(fr.breaths_table.reset_index(drop=True))
        if fr.processed is not None:
            out["processed"][f"{fname} – Processed data.csv"] = _processed_summary(fr.processed)
        # Future FileResult fields (manoeuvre extraction, per-file scalar summaries)
        # -- getattr-guarded and emitted only when non-empty, so today's scenarios
        # (none of which set either) produce an IDENTICAL dict to before this ticket.
        manoeuvres = getattr(fr, "manoeuvres", None)
        if manoeuvres:
            out.setdefault("manoeuvres", {})[f"{fname}"] = manoeuvres
        per_file_scalars = getattr(fr, "per_file_scalars", None)
        if per_file_scalars:
            out.setdefault("per_file_scalars", {})[f"{fname}"] = per_file_scalars
    return out


def build_all():
    return {name: run_scenario(name) for name in sorted(mg.SCENARIOS)}


def main():
    write = "--write" in sys.argv
    results = build_all()
    if write:
        json.dump(results, open(GOLDEN, "w"), indent=2, sort_keys=True)
        print("Wrote", GOLDEN)
        return
    committed = json.load(open(GOLDEN))
    import re
    unexpected = 0
    for sc in sorted(results):
        for key, table in results[sc].get("per_file", {}).items():
            ref = committed[sc]["per_file"].get(key, {})
            for col, vals in table.items():
                rv = ref.get(col)
                if rv is None:
                    continue
                a = pd.to_numeric(pd.Series(rv), errors="coerce").to_numpy(float)
                b = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(float)
                n = min(len(a), len(b))
                nan = np.isnan(a[:n]) & np.isnan(b[:n])
                if not np.all(np.isclose(a[:n], b[:n], rtol=1e-9, atol=1e-12, equal_nan=True) | nan):
                    tag = "entropy(bug#2)" if re.search("entropy", col) else "UNEXPECTED"
                    if tag == "UNEXPECTED":
                        unexpected += 1
                    print(f"  {sc}/{key}/{col}: differs  [{tag}]")
    print("UNEXPECTED diffs:" , unexpected)


if __name__ == "__main__":
    main()

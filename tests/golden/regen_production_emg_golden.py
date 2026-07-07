#!/usr/bin/env python3
"""Regenerate the production golden's EMG scenarios (H5/H6) from the NEW canonical
core, and categorise every changed column so we can prove that only EMG (this
milestone) and PTP (the earlier approved window-baseline, Commit B) changed —
timing / WOB / pressures / entropy stay intact.

    python regen_production_emg_golden.py            # compare + report only
    python regen_production_emg_golden.py --write     # also update production_golden.json
"""
import os
import re
import sys
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import build_production_golden as bpg          # scenario defs + expected_files + PROD
from respmech.settingsio.migrate import migrate_dict
from respmech.core.pipeline import run_batch

GOLDEN = os.path.join(HERE, "production_golden.json")
RTOL, ATOL = 1e-6, 1e-9

PTP_RE = re.compile(r"^(int_|ptp_)")
EMG_RE = re.compile(r"(rms|emg|integral)", re.IGNORECASE)

EMG_SCENARIOS = ["emg_h5_ecg_noise_outlier", "emg_h6_ecg_noise_outlier"]


def _classify(col):
    if EMG_RE.search(col):
        return "EMG"
    if PTP_RE.search(col):
        return "PTP"
    return "OTHER"


def _jsonify(df):
    out = {}
    for c in df.columns:
        out[str(c)] = ["NaN" if isinstance(v, float) and np.isnan(v) else v for v in df[c].tolist()]
    return out


def run_scenario(scname):
    sc = next(s for s in bpg.SCENARIOS if s["name"] == scname)
    raw = bpg.extract_settings(os.path.join(bpg.PROD, sc["settings_py"]))
    subj = "H5" if "h5" in scname.lower() else "H6"
    s, _ = migrate_dict(raw)
    s.input.folder = os.path.join(bpg.PROD, sc["input_dir"])
    s.input.files = f"RIU_{subj}_*.txt"
    s.output.folder = os.path.join(HERE, "_prod_work_emg", scname)
    s.output.data.save_processed = False
    # the shared noise reference is the rest/Baseline file (already set by migrator);
    # ensure it points at the local Baseline.
    s.processing.emg.noise.reference_file = f"RIU_{subj}_Baseline.txt"
    res = run_batch(s)
    return res


def main():
    write = "--write" in sys.argv
    golden = json.load(open(GOLDEN))
    summary = {"changed": {"EMG": 0, "PTP": 0, "OTHER": 0}, "other_cols": [], "new_ok": []}

    for scname in EMG_SCENARIOS:
        print(f"\n===== {scname} =====")
        res = run_scenario(scname)
        gold_files = golden.get(scname, {}).get("files", {})
        for fname, fr in sorted(res.ok_files.items()):
            new_tbl = fr.breaths_table.reset_index(drop=True)
            old = gold_files.get(fname, {})
            if old.get("status") != "ok" or "golden" not in old:
                summary["new_ok"].append(f"{scname}/{fname} (was {old.get('status','absent')})")
                print(f"  {fname}: NEW run (old status={old.get('status','absent')}), "
                      f"{len(new_tbl)} breaths")
                continue
            og = old["golden"]
            cats = {}
            for col in og:
                if col not in new_tbl.columns:
                    continue
                a = pd.to_numeric(pd.Series(og[col]), errors="coerce").to_numpy(float)
                b = pd.to_numeric(new_tbl[col], errors="coerce").to_numpy(float)
                n = min(len(a), len(b))
                nan = np.isnan(a[:n]) & np.isnan(b[:n])
                if np.all(np.isclose(a[:n], b[:n], rtol=RTOL, atol=ATOL, equal_nan=True) | nan):
                    continue
                cat = _classify(col)
                mr = float(np.nanmax(np.abs(a[:n] - b[:n]) / np.maximum(np.abs(b[:n]), 1e-12)))
                cats.setdefault(cat, []).append((col, mr))
                summary["changed"][cat] += 1
                if cat == "OTHER":
                    summary["other_cols"].append(f"{scname}/{fname}/{col}")
            emgn = len(cats.get("EMG", [])); ptpn = len(cats.get("PTP", [])); othn = len(cats.get("OTHER", []))
            emg_max = max([m for _, m in cats.get("EMG", [])], default=0)
            print(f"  {fname}: changed EMG={emgn} (max_rel {emg_max:.2f}) PTP={ptpn} OTHER={othn}"
                  + (f"  !! OTHER: {[c for c,_ in cats['OTHER']]}" if othn else ""))

        # regenerate this scenario's file entries from the new core
        if write:
            golden.setdefault(scname, {"covers": next(x['covers'] for x in bpg.SCENARIOS if x['name']==scname), "files": {}})
            for fname, fr in res.ok_files.items():
                golden[scname]["files"][fname] = {"status": "ok",
                                                  "golden": _jsonify(fr.breaths_table.reset_index(drop=True))}
            for fname, fr in res.failed_files.items():
                golden[scname]["files"][fname] = {"status": "error", "error": fr.error}

    print("\n=== SUMMARY ===")
    print("changed columns by category:", summary["changed"])
    print("newly-running files (bug #1 fix):", summary["new_ok"])
    if summary["other_cols"]:
        print("!!! UNEXPECTED non-EMG/non-PTP changes:", summary["other_cols"])
    else:
        print("OK: only EMG (this milestone) and PTP (approved Commit B) columns changed.")

    if write:
        json.dump(golden, open(GOLDEN, "w"), indent=2, sort_keys=True)
        print(f"\nWrote {GOLDEN}")


if __name__ == "__main__":
    main()

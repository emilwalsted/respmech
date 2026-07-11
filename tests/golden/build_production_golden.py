#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the PRODUCTION golden from Emil's real validation datasets.

For each scenario (a real settings .py + input recordings + Emil's expected output
spreadsheets under tests/golden/production/):
  1. Safely extract the settings dict from the .py (ast.literal_eval — no code run).
  2. Rewrite the machine-specific absolute paths to the local production tree, force
     a single-file mask and a local output dir.
  3. Run the UNMODIFIED respmech.py (pinned env) on each file, in a subprocess.
  4. Read the produced per-file breathdata, and COMPARE it against Emil's expected
     spreadsheet for the same file — split into a "mechanics/WOB/pressure" column
     group and an "EMG-derived" column group (which depends on the ECG/noise
     algorithm version).
  5. Freeze the CURRENT code's captured numbers as production_golden.json (derived,
     de-identified) and write a manifest (input hashes/sizes) + a human comparison
     report. NO raw data or expected spreadsheets are written into the repo.

Outputs (committed): production_golden.json, production_manifest.json,
production_comparison.md. Everything else stays local / gitignored.
"""
import os
import re
import ast
import sys
import json
import glob
import time
import shutil
import hashlib
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(HERE, "production")
WORK = os.path.join(HERE, "_prod_work")
PYEXE = sys.executable  # the pinned interpreter running this script
RUNNER = os.path.join(HERE, "prod_runner.py")

GOLDEN = os.path.join(HERE, "production_golden.json")
MANIFEST = os.path.join(HERE, "production_manifest.json")
REPORT = os.path.join(HERE, "production_comparison.md")

# Comparison tolerances.
RTOL = 1e-6
ATOL = 1e-9

# Columns whose values are EMG/ECG-algorithm dependent (compared separately).
EMG_COL_RE = re.compile(r"(rms|emg|integral)", re.IGNORECASE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_settings(pyfile):
    src = open(pyfile, encoding="utf-8", errors="replace").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "settings":
                    return ast.literal_eval(node.value)
    raise ValueError(f"no `settings` dict in {pyfile}")


def localize(path, scenario_input_dir):
    """Rewrite a machine-specific absolute path to the local production tree by
    anchoring on '.../RespMech validation/<...>'."""
    if not isinstance(path, str) or not path:
        return path
    marker = "RespMech validation/"
    idx = path.find(marker)
    if idx >= 0:
        rel = path[idx + len(marker):]
        return os.path.join(PROD, rel)
    return path


def prepare_settings(raw, scenario_input_dir, single_file, out_dir):
    """Deep-copy + rewrite paths for a single-file run into out_dir."""
    s = json.loads(json.dumps(raw))  # deep copy
    s["input"]["inputfolder"] = scenario_input_dir
    s["input"]["files"] = single_file
    s["output"]["outputfolder"] = out_dir
    # Master's applysettings() crashes on a nested processing.* subsection that is
    # absent from its defaults (it only handles new *leaf* keys). Strip any such
    # subsection so the ORIGINAL code can load these newer settings files. The only
    # one present is `sampling` (resampling), which master does not implement anyway
    # (the settings here use resample=False) — recorded as a characterisation note.
    known = {"mechanics", "wob", "emg", "entropy"}
    stripped = [k for k in list(s.get("processing", {}))
                if isinstance(s["processing"][k], dict) and k not in known]
    for k in stripped:
        del s["processing"][k]
    s["_stripped_processing_sections"] = stripped
    # Rewrite noise_profile reference paths (2nd element of each entry).
    try:
        nps = s["processing"]["emg"].get("noise_profile", [])
        for entry in nps:
            if len(entry) >= 2 and isinstance(entry[1], str) and entry[1]:
                entry[1] = localize(entry[1], scenario_input_dir)
    except Exception:
        pass
    # Turn off optional plots for speed (the forced EMG overview plot cannot be
    # disabled; that is itself a finding). Numbers are unaffected.
    d = s["output"].get("diagnostics", {})
    for k in ["savepvaverage", "savepvoverview", "savepvindividualworkload",
              "savedataviewraw", "savedataviewtrimmed", "savedataviewdriftcor"]:
        d[k] = False
    s["output"]["diagnostics"] = d
    return s


def jsonify_df(df):
    out = {}
    for col in df.columns:
        vals = []
        for v in df[col].tolist():
            if isinstance(v, float) and (np.isnan(v) if isinstance(v, float) else False):
                vals.append("NaN")
            else:
                vals.append(v)
        out[str(col)] = vals
    return out


def read_breathdata(xlsx):
    df = pd.read_excel(xlsx, sheet_name="Data")
    return df


def compare_tables(cur, exp):
    """Compare two breathdata DataFrames aligned by row order. Returns a dict:
    per-column-group match counts + list of mismatching columns with max reldiff."""
    result = {"n_rows_cur": len(cur), "n_rows_exp": len(exp), "groups": {}, "columns": {}}
    common = [c for c in cur.columns if c in exp.columns]
    n = min(len(cur), len(exp))
    groups = {"emg": [], "mechanics": []}
    for c in common:
        grp = "emg" if EMG_COL_RE.search(str(c)) else "mechanics"
        a = pd.to_numeric(cur[c].iloc[:n], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(exp[c].iloc[:n], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        with np.errstate(invalid="ignore"):
            close = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True) | both_nan
        max_abs = float(np.nanmax(np.abs(a - b))) if np.any(~both_nan) else 0.0
        denom = np.maximum(np.abs(b), 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(a - b) / denom
        max_rel = float(np.nanmax(rel[~both_nan])) if np.any(~both_nan) else 0.0
        ok = bool(np.all(close))
        groups[grp].append(ok)
        if not ok:
            result["columns"][str(c)] = {"group": grp, "max_abs": max_abs, "max_rel": max_rel}
    for g, oks in groups.items():
        result["groups"][g] = {"n": len(oks), "matching": int(sum(oks)),
                               "all_match": bool(oks) and all(oks)}
    return result


# ---- scenario definitions -------------------------------------------------

SCENARIOS = [
    {
        "name": "zeros_debugging",
        "settings_py": "Zeros debugging/NEP303_example.py",
        "input_dir": "Zeros debugging/input",
        "expected_dir": "Zeros debugging/output/data",
        "covers": "flow separation, volume drift/zeroing, WOB (no EMG)",
    },
    {
        "name": "trimming_debugging",
        "settings_py": "Trimming debugging/NEP301_V2_140W 1 trimming debugging.py",
        "input_dir": "Trimming debugging",
        "expected_dir": "Trimming debugging/output/data",
        "covers": "breath trimming edge cases, flow separation (no EMG)",
    },
    {
        "name": "resampling_volume_sep",
        "settings_py": "Resampling test/example.py",
        "input_dir": "Resampling test/input",
        "expected_dir": "Resampling test/output 1000Hz original/data",
        "covers": "VOLUME-based breath separation + EMG/ECG/noise",
    },
    {
        "name": "emg_h5_ecg_noise_outlier",
        "settings_py": "EMG processing fix test/RIU_H5_example.py",
        "input_dir": "EMG processing fix test",
        "expected_dir": "EMG processing fix test/RIU_H5 output with outlier processing + new ECG removal/data",
        "covers": "ECG removal + noise reduction + RMS outlier processing + EMG RMS",
    },
    {
        "name": "emg_h6_ecg_noise_outlier",
        "settings_py": "EMG processing fix test/RIU_H6_example.py",
        "input_dir": "EMG processing fix test",
        "expected_dir": "EMG processing fix test/RIU_H6 output with outlier processing + new ECG removal/data",
        "covers": "ECG removal + noise reduction + RMS outlier processing + EMG RMS",
    },
]


def expected_files(expected_dir):
    """Map input filename -> expected breathdata xlsx, derived from the dir."""
    out = {}
    for x in glob.glob(os.path.join(expected_dir, "*.breathdata*.xlsx")):
        base = os.path.basename(x)
        if base.startswith("~$"):
            continue
        m = re.match(r"(.+?\.txt)\.breathdata", base)
        if m:
            out[m.group(1)] = x
    return out


def run_scenario(sc, manifest, only=None):
    input_dir = os.path.join(PROD, sc["input_dir"])
    expected_dir = os.path.join(PROD, sc["expected_dir"])
    raw = extract_settings(os.path.join(PROD, sc["settings_py"]))
    exp_map = expected_files(expected_dir)

    files = sorted(exp_map)
    if only:
        files = [f for f in files if f in only]

    scenario_result = {"covers": sc["covers"], "files": {}}
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        if not os.path.exists(in_path):
            scenario_result["files"][fname] = {"status": "input_missing"}
            continue
        out_dir = os.path.join(WORK, sc["name"], re.sub(r"[^\w.-]", "_", fname))
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

        settings = prepare_settings(raw, input_dir, fname, out_dir)
        stripped = settings.pop("_stripped_processing_sections", [])
        sp = os.path.join(out_dir, "_settings.json")
        with open(sp, "w") as f:
            json.dump(settings, f)

        manifest.setdefault(sc["name"], {})[fname] = {
            "input_sha256": sha256(in_path),
            "input_bytes": os.path.getsize(in_path),
            "expected_sha256": sha256(exp_map[fname]),
        }

        t0 = time.time()
        proc = subprocess.run([PYEXE, RUNNER, sp], capture_output=True, text=True,
                              timeout=3600)
        dt = time.time() - t0
        ok = proc.returncode == 0 and "PROD_RUNNER_OK" in proc.stdout

        entry = {"status": "ok" if ok else "error", "seconds": round(dt, 1)}
        if stripped:
            entry["stripped_processing_sections"] = stripped
        if not ok:
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            entry["error_tail"] = tail[-6:]
            scenario_result["files"][fname] = entry
            print(f"  [{sc['name']}] {fname}: ERROR ({dt:.0f}s)")
            continue

        cur_xlsx = os.path.join(out_dir, "data", f"{fname}.breathdata.xlsx")
        if not os.path.exists(cur_xlsx):
            entry["status"] = "no_output"
            scenario_result["files"][fname] = entry
            print(f"  [{sc['name']}] {fname}: no output produced")
            continue

        cur = read_breathdata(cur_xlsx)
        exp = read_breathdata(exp_map[fname])
        entry["golden"] = jsonify_df(cur)
        entry["comparison"] = compare_tables(cur, exp)
        scenario_result["files"][fname] = entry
        g = entry["comparison"]["groups"]
        print(f"  [{sc['name']}] {fname}: ok ({dt:.0f}s) "
              f"mech {g['mechanics']['matching']}/{g['mechanics']['n']} "
              f"emg {g['emg']['matching']}/{g['emg']['n']}")
    return scenario_result


def main():
    only_scenarios = None
    only_files = None
    args = sys.argv[1:]
    if args:
        only_scenarios = set(a for a in args if not a.endswith(".txt"))
        only_files = set(a for a in args if a.endswith(".txt")) or None

    manifest = {}
    golden = {}
    for sc in SCENARIOS:
        if only_scenarios and sc["name"] not in only_scenarios:
            continue
        print(f"\n===== {sc['name']} =====")
        golden[sc["name"]] = run_scenario(sc, manifest, only=only_files)

    # Merge into existing golden/manifest so partial runs accumulate.
    if os.path.exists(GOLDEN):
        prev = json.load(open(GOLDEN))
        prev.update(golden)
        golden = prev
    if os.path.exists(MANIFEST):
        pm = json.load(open(MANIFEST))
        pm.update(manifest)
        manifest = pm

    json.dump(golden, open(GOLDEN, "w"), indent=2, sort_keys=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2, sort_keys=True)
    print(f"\nWrote {GOLDEN}\nWrote {MANIFEST}")


if __name__ == "__main__":
    main()

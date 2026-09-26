#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden / characterisation harness for RespMech.

Runs the *unmodified* respmech.py over deterministic synthetic input across a
matrix of settings scenarios, then serialises every numeric output into
golden_reference.json. A refactor is proven equivalent when it reproduces this
JSON within the documented tolerance (see test_golden.py / GOLDEN_TOLERANCE).

This script is used two ways:
    python make_golden.py --write     # (re)generate the golden reference
    imported by test_golden.py        # recompute current outputs for comparison

The reference must only be regenerated deliberately (when behaviour is
intentionally changed), never to "make the test pass".
"""
import os
import sys
import json
import glob
import shutil
import importlib.util

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESPMECH_PY = os.path.join(REPO_ROOT, "legacy", "respmech.py")   # frozen v1 oracle (legacy/README.md)
INPUT_DIR = os.path.join(HERE, "input")
WORK_DIR = os.path.join(HERE, "_work")           # transient output (gitignored)
GOLDEN_JSON = os.path.join(HERE, "golden_reference.json")


def load_respmech():
    spec = importlib.util.spec_from_file_location("respmech", RESPMECH_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def base_settings(outputfolder):
    """Common settings; scenarios override selectively."""
    return {
        "input": {
            "inputfolder": INPUT_DIR,
            "files": "synth_case_*.csv",
            "format": {
                "samplingfrequency": 1000,
                "decimalcharacter": ".",
            },
            "data": {
                "column_poes": 7,
                "column_pgas": 8,
                "column_pdi": 9,
                "column_volume": 6,
                "column_flow": 5,
                "columns_emg": [2, 3, 4],
                "columns_entropy": [10, 11, 12],
            },
        },
        "processing": {
            "mechanics": {
                "breathseparationbuffer": 200,
                "separateby": "flow",
                "peakheight": 0.1,
                "peakdistance": 0.1,
                "peakwidth": 0.2,
                "inverseflow": False,
                "integratevolumefromflow": False,
                "inversevolume": False,
                "correctvolumedrift": True,
                "correctvolumetrend": False,
                "volumetrendadjustmethod": "linear",
                "volumetrendpeakminheight": 0.8,
                "volumetrendpeakmindistance": 0.4,
                "excludebreaths": [],
                "breathcounts": [],
            },
            "wob": {
                "calcwobfrom": "average",
                "avgresamplingobs": 300,
            },
            "emg": {
                "rms_s": 0.050,
                "remove_ecg": False,
                "minheight": 0.0005,
                "mindistance": 0.5,
                "minwidth": 0.001,
                "windowsize": 0.4,
                "outlierrmssdlimit": 0,
                "remove_noise": False,
                "save_sound": False,
                "emgplotyscale": [-0.1, 0.1],
            },
            # Pinned explicitly, independent of EntropySettings.epochs' live GUI default
            # (settings.py) -- this scenario characterises v1's fixed config, not "whatever
            # the app currently defaults to". Changing the GUI default (2 -> 3, 05-09-2026
            # content review) does NOT require regenerating this golden reference, and was
            # verified not to (tests/golden/test_golden.py stayed 4/4 green before and after).
            "entropy": {
                "entropy_epochs": 2,
                "entropy_tolerance": 0.1,
            },
        },
        "output": {
            "outputfolder": outputfolder,
            "data": {
                "saveaveragedata": True,
                "savebreathbybreathdata": True,
                # getprocesseddata() hardcodes exactly 5 EMG column names, so the
                # processed-data path is exercised only in the EMG-off scenario.
                "saveprocesseddata": False,
                "includeignoredbreaths": False,
            },
            "diagnostics": {
                # Plots do not affect numbers; disabled for speed & determinism.
                "savepvaverage": False,
                "savepvoverview": False,
                "savepvindividualworkload": False,
                "pvcolumns": 2,
                "pvrows": 2,
                "savedataviewraw": False,
                "savedataviewtrimmed": False,
                "savedataviewdriftcor": False,
            },
        },
    }


def deep_update(base, overrides):
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# NOTE: The forced "EMG raw channel overview" plot in legacy/emg.py USED TO crash on
# an *ignored* breath (latent bug #1 in README.md: the Rectangle width was built as a
# 1-element array). That is why EMG-on scenarios originally carried no excluded
# breaths and excludebreaths/breathcounts were covered by an EMG-off scenario only.
# The crash was fixed on 23-09-2026 with Emil's explicit sign-off -- the one
# sanctioned edit to the frozen oracle, see the repo-root CLAUDE.md -- so
# "flow_exclude_emg" below now locks that combination with the EMG pipeline ACTIVE.
# "flow_exclude_noemg" stays: it is the only scenario exercising the processed-data
# export, which getprocesseddata() only supports at exactly 5 EMG channels.
#
# LEGACY_SCENARIOS: expressed as legacy-dict overrides (deep_update onto
# base_settings()) and migrated via migrate_dict — run_all() below (the frozen v1
# oracle) can only run THESE, since the oracle has no notion of the newer,
# TOML-only settings (analysis.signals, breath_types, references, separators,
# subjects, pressure.peepi, ...). Every scenario here must stay byte-identical
# against the legacy oracle forever; a new scenario that needs any v2-only
# settings belongs in V2_SCENARIOS instead.
LEGACY_SCENARIOS = {
    "flow_wob_average": {},
    "flow_wob_individual": {
        "processing": {"wob": {"calcwobfrom": "individual"}},
    },
    # Volume-based breath separation is intentionally NOT covered by synthetic
    # data: the separator assumes a specific breath morphology and is brittle on
    # analytic signals. It is to be locked against Emil's real production data.
    "flow_integratevol": {
        # Exercises the flow->volume integration path (scipy cumtrapz), which
        # is a distinct real code path and is broken on modern scipy.
        "processing": {"mechanics": {"integratevolumefromflow": True}},
    },
    # The EMG-on half of the exclusion pair: same excludebreaths/breathcounts as
    # flow_exclude_noemg, but with the default EMG channels left in, so the RMS /
    # integral EMG columns are locked for a run that ignores a breath. Unreachable
    # until the legacy plot crash above was fixed.
    "flow_exclude_emg": {
        "processing": {"mechanics": {
            "excludebreaths": [["synth_case_A.csv", [3]]],
            "breathcounts": [["synth_case_B.csv", 6]],
        }},
    },
    "flow_exclude_noemg": {
        "input": {"data": {"columns_emg": [], "columns_entropy": [10, 11, 12]}},
        "processing": {"mechanics": {
            "excludebreaths": [["synth_case_A.csv", [3]]],
            "breathcounts": [["synth_case_B.csv", 6]],
        }},
        "output": {"data": {"saveprocesseddata": True}},
    },
}

# V2_SCENARIOS: name -> path (relative to this file's directory) of a committed
# tests/golden/scenarios/<name>.toml settings file, bagt directly from the v2 core
# (golden_newcore.py --write) rather than run through the legacy oracle. A v2
# scenario expresses settings the legacy dict/migrate_dict path cannot (typed
# breaths, references, separators, ...); it is never included in run_all()'s
# legacy-oracle run. Empty until the first feature ticket that needs one.
V2_SCENARIOS: dict = {}

# The union both test_golden.py (via golden_newcore.mg.SCENARIOS) and this
# module's own run_all() see the FULL set through; run_all() below filters back
# down to LEGACY_SCENARIOS alone, since the frozen oracle cannot run a v2 scenario.
SCENARIOS = {**LEGACY_SCENARIOS, **V2_SCENARIOS}


def _jsonify_df(df):
    """Turn a DataFrame into a plain dict of column -> list of values,
    with NaN represented as the string 'NaN' for stable round-tripping."""
    out = {}
    for col in df.columns:
        vals = []
        for v in df[col].tolist():
            if isinstance(v, float) and np.isnan(v):
                vals.append("NaN")
            else:
                vals.append(v)
        out[str(col)] = vals
    return out


def collect_outputs(outdir):
    """Read every numeric artefact produced into a JSON-serialisable dict."""
    result = {"average_breathdata": None, "per_file": {}, "processed": {}}

    avg_path = os.path.join(outdir, "data", "Average breathdata.xlsx")
    if os.path.exists(avg_path):
        df = pd.read_excel(avg_path, sheet_name="Data")
        result["average_breathdata"] = _jsonify_df(df)

    for xlsx in sorted(glob.glob(os.path.join(outdir, "data", "*.breathdata.xlsx"))):
        name = os.path.basename(xlsx)
        df = pd.read_excel(xlsx, sheet_name="Data")
        result["per_file"][name] = _jsonify_df(df)

    # Processed data CSVs: capture shape + per-column summary (compact but sensitive).
    for csvf in sorted(glob.glob(os.path.join(outdir, "data", "*Processed data.csv"))):
        name = os.path.basename(csvf)
        df = pd.read_csv(csvf)
        summary = {"shape": list(df.shape), "columns": list(map(str, df.columns))}
        colstats = {}
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                colstats[str(col)] = {
                    "sum": float(np.nansum(df[col].values)),
                    "mean": float(np.nanmean(df[col].values)),
                    "min": float(np.nanmin(df[col].values)),
                    "max": float(np.nanmax(df[col].values)),
                }
        summary["colstats"] = colstats
        result["processed"][name] = summary

    return result


def run_all():
    """Run every scenario through the frozen v1 oracle (legacy/respmech.py) and
    --write the reference JSON. Only LEGACY_SCENARIOS: the oracle has no concept
    of a v2-only settings shape, so a V2_SCENARIOS entry is bagt separately by
    golden_newcore.py --write instead (see tests/golden/README.md)."""
    rm = load_respmech()
    all_results = {}
    for name, override in LEGACY_SCENARIOS.items():
        outdir = os.path.join(WORK_DIR, name)
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

        settings = base_settings(outdir)
        deep_update(settings, override)

        print(f"\n===== SCENARIO: {name} =====")
        rm.analyse(settings)
        all_results[name] = collect_outputs(outdir)
    return all_results


def main():
    write = "--write" in sys.argv
    results = run_all()
    if write:
        # Merge onto whatever is already on disk rather than overwrite it outright:
        # this file only ever regenerates LEGACY_SCENARIOS (the frozen oracle cannot
        # run a V2_SCENARIOS entry at all), so a blind overwrite would silently drop
        # any v2 scenario golden_newcore.py --write had separately baked in.
        existing = {}
        if os.path.exists(GOLDEN_JSON):
            with open(GOLDEN_JSON) as f:
                existing = json.load(f)
        existing.update(results)
        with open(GOLDEN_JSON, "w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        print(f"\nWrote golden reference: {GOLDEN_JSON}")
    else:
        print("\nDry run complete (pass --write to save golden_reference.json).")
    return results


if __name__ == "__main__":
    main()

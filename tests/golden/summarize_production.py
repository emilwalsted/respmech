#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render production_comparison.md from production_golden.json.

Summarises, per scenario/file: run status, and — where the current code produced
output — how it compares to Emil's expected spreadsheet, split into a
mechanics/WOB/pressure group and an EMG-derived group. The narrative analysis
(root causes) is appended from ANALYSIS below.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "production_golden.json")
OUT = os.path.join(HERE, "production_comparison.md")

PTP_COLS = {"int_oesinsp", "ptp_oesinsp", "int_pdiinsp", "ptp_pdiinsp",
            "int_pgasexp", "ptp_pgasexp"}
WOB_COLS = {"wob_ex_total", "wob_in_res", "wobtotal", "wob_in_total", "wob_in_ela"}

ANALYSIS = """
## Root-cause analysis of the mismatches

All differences below are **characterised, not fixed** (fixes are Phase 2). The
current repo code (`master`) was run unmodified in the pinned env (Python 3.12 /
SciPy 1.13); "expected" = the spreadsheets Emil generated.

### A. PTP / pressure-time-integral columns differ by a large factor — *explained*
The six columns `int_oesinsp, ptp_oesinsp, int_pdiinsp, ptp_pdiinsp, int_pgasexp,
ptp_pgasexp` differ substantially (up to ~150 %) in the **Zeros** and **Trimming**
datasets. Root cause: commit **`1630c40` "Fixed PTP calculations (zeroing)"**
introduced an extra baseline subtraction in `calcptp` (`pressure = pressure.squeeze()
- pressure[0]`). Those expected spreadsheets were produced **before** that commit,
so the current code legitimately differs. Verified numerically: expected
`int_oesinsp` for Zeros/NEP303_V2_40W breath 1 = 13.0628 is reproduced exactly by
the pre-fix formula (no `- pressure[0]`); the current code gives 2.7111. It is **not**
an integration-method or SciPy-version effect — trapezoid and Simpson agree to 4
decimals on the same array. In the **Resampling** and **EMG H5/H6** datasets (which
Emil generated *after* the fix) these same PTP columns **match**.

### B. WOB columns differ at the ~1e-5 level — negligible / float-level
`wob_*` differ only at rtol ~1e-5 where they differ at all — consistent with the
`scipy.integrate.simps`→`simpson` rename and floating-point reassociation. No
behavioural change.

### C. EMG-derived columns differ (RMS / integrated EMG) — *algorithm version*
For every EMG dataset the EMG columns differ (relative differences up to ~90 %).
Emil's expected outputs come from the **"new ECG removal" / outlier-processing**
version he was validating; the current `master` EMG/ECG-removal + spectral
noise-reduction path differs from it. Mechanics/WOB (non-EMG) are unaffected, which
localises the difference to the EMG conditioning chain. (~20/42 EMG columns still
coincide with `master`; the rest track the ECG/noise algorithm.)

### D. Runtime failures on the current code — real latent bugs, on real data
- **`RIU_H5_IC`, `RIU_H6_IC`, `RIU_H6_Baseline` crash** with
  "setting an array element with a sequence … shape (4,)". This is latent bug #1
  (the EMG overview plot builds an ignored-breath `Rectangle` width as a 1-element
  array). These files have **excluded breaths + EMG**, so the always-on EMG plot
  crashes — the current code cannot process them at all.
- **`NEP301_V2_140W 1`, `…testbegin`, `…testend` (Trimming) fail** with "Could not
  trim data to whole breaths" — these are deliberate trim edge-case files whose
  data does not start in late expiration / end in early inspiration, so `trim()`
  produces an empty range. Characterises the trim precondition.
- **The `processing.sampling` (resampling) settings section** crashes `master`'s
  `applysettings` (KeyError) because its naive default-merge cannot handle a nested
  subsection absent from defaults. `master` does not implement resampling (that
  lives on the unmerged `resampling-options` branch); the section is stripped before
  running (recorded per file). `resample` was `False` here anyway.

### E. Segmentation sensitivity — `RIU_H5_Baseline`
Beyond the PTP columns, `RIU_H5_Baseline` also differs in timing/segmentation
columns (`ti, te, ttot, ti_ttot, vmr, flow_midvol*, *_endexp, tlr_insp`). Quiet
baseline breathing is sensitive to the flow buffer (`breathseparationbuffer=800`),
so the breath boundaries — and everything derived from them — diverge from the
expected version. The other H5/H6 files segment identically (only PTP/EMG differ).

## What the production golden locks

`production_golden.json` freezes the **current code's** captured per-breath output
for every file that runs (the regression lock for the refactor). Newly covered
versus the synthetic golden:
- **Volume-based breath separation** — `resampling_volume_sep` (mechanics match
  Emil's post-fix expected 48/48).
- **ECG removal + spectral noise reduction + RMS outlier processing** —
  `emg_h5/h6` on real diaphragm EMG.
- **Real trim edge cases and quiet-breathing segmentation**.

Because the raw recordings are gitignored (public repo), this golden is verified
locally when the data is present (`production_manifest.json` pins input hashes).
"""


def main():
    g = json.load(open(GOLDEN))
    lines = []
    lines.append("# Production golden — comparison against Emil's expected outputs\n")
    lines.append("_Generated from `production_golden.json` by `summarize_production.py`. "
                 "Raw data and expected spreadsheets are gitignored (public repo); only "
                 "these derived numbers are committed._\n")
    lines.append("## Summary\n")
    lines.append("| Scenario | Covers | File | Status | Mechanics match | EMG match | Notes |")
    lines.append("|---|---|---|---|---|---|---|")

    for scname in sorted(g):
        sc = g[scname]
        covers = sc.get("covers", "")
        for fn in sorted(sc["files"]):
            fe = sc["files"][fn]
            st = fe["status"]
            if st != "ok":
                note = ""
                for l in fe.get("error_tail", []):
                    if "Error message" in l:
                        note = l.split("Error message:")[-1].strip()[:60]
                strip = fe.get("stripped_processing_sections")
                if strip:
                    note += f" (stripped {strip})"
                lines.append(f"| {scname} | {covers} | {fn} | **{st}** | – | – | {note} |")
                continue
            cmp = fe["comparison"]
            m = cmp["groups"]["mechanics"]; e = cmp["groups"]["emg"]
            mm = [c for c, i in cmp["columns"].items() if i["group"] == "mechanics"]
            ptp = sorted(set(mm) & PTP_COLS)
            wob = sorted(set(mm) & WOB_COLS)
            other = sorted(set(mm) - PTP_COLS - WOB_COLS)
            note = []
            if ptp:
                note.append("PTP-fix")
            if wob:
                note.append("WOB~1e-5")
            if other:
                note.append("seg:" + ",".join(other[:4]) + ("…" if len(other) > 4 else ""))
            emgstr = f"{e['matching']}/{e['n']}" if e["n"] else "n/a"
            lines.append(f"| {scname} | {covers} | {fn} | ok {fe['seconds']}s | "
                         f"{m['matching']}/{m['n']} | {emgstr} | {'; '.join(note)} |")

    lines.append("")
    lines.append(ANALYSIS)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()

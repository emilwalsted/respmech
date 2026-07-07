# Production golden — comparison against Emil's expected outputs

_Generated from `production_golden.json` by `summarize_production.py`. Raw data and expected spreadsheets are gitignored (public repo); only these derived numbers are committed._

## Summary

| Scenario | Covers | File | Status | Mechanics match | EMG match | Notes |
|---|---|---|---|---|---|---|
| emg_h5_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H5_40W.txt | ok 6.1s | 38/48 | 20/42 | PTP-fix; WOB~1e-5 |
| emg_h5_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H5_60W.txt | ok 6.2s | 39/48 | 20/42 | PTP-fix; WOB~1e-5 |
| emg_h5_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H5_Baseline.txt | ok 15.0s | 26/48 | 0/42 | PTP-fix; WOB~1e-5; seg:ex_flow_midvol,flow_midvolexp,pdi_endexp,pgas_endexp… |
| emg_h5_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H5_IC.txt | **error** | – | – | setting an array element with a sequence. The requested arra (stripped ['sampling']) |
| emg_h5_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H5_Peak180W.txt | ok 4.1s | 38/48 | 20/42 | PTP-fix; WOB~1e-5 |
| emg_h6_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H6_40W.txt | ok 8.8s | 38/48 | 19/42 | PTP-fix; WOB~1e-5 |
| emg_h6_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H6_60W.txt | ok 8.4s | 40/48 | 19/42 | PTP-fix; WOB~1e-5 |
| emg_h6_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H6_Baseline.txt | **error** | – | – | setting an array element with a sequence. The requested arra |
| emg_h6_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H6_IC.txt | **error** | – | – | setting an array element with a sequence. The requested arra |
| emg_h6_ecg_noise_outlier | ECG removal + noise reduction + RMS outlier processing + EMG RMS | RIU_H6_Peak220W.txt | ok 4.8s | 42/48 | 21/42 | PTP-fix |
| resampling_volume_sep | VOLUME-based breath separation + EMG/ECG/noise | RIU_H1_60W.txt | ok 66.5s | 48/48 | 0/42 |  |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP301_V2_140W 1.txt | **error** | – | – | Could not trim data to whole breaths. Please ensure that the |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP301_V2_140W 2.txt | ok 1.2s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP301_V2_140W testbegin.txt | **error** | – | – | Could not trim data to whole breaths. Please ensure that the |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP301_V2_140W testend.txt | **error** | – | – | Could not trim data to whole breaths. Please ensure that the |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP307_V2_140W 1.txt | ok 1.3s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP307_V2_40W 1.txt | ok 1.4s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP307_V2_Baseline 1.txt | ok 1.7s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP307_V2_IC 1.txt | ok 1.3s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| trimming_debugging | breath trimming edge cases, flow separation (no EMG) | NEP307_V2_Peak340W 1.txt | ok 1.3s | 40/48 | n/a | PTP-fix; WOB~1e-5 |
| zeros_debugging | flow separation, volume drift/zeroing, WOB (no EMG) | NEP303_V2_140W.txt | ok 1.6s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| zeros_debugging | flow separation, volume drift/zeroing, WOB (no EMG) | NEP303_V2_40W.txt | ok 1.4s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| zeros_debugging | flow separation, volume drift/zeroing, WOB (no EMG) | NEP303_V2_Baseline.txt | ok 1.4s | 38/48 | n/a | PTP-fix; WOB~1e-5 |
| zeros_debugging | flow separation, volume drift/zeroing, WOB (no EMG) | NEP303_V2_IC.txt | ok 1.1s | 41/48 | n/a | PTP-fix; WOB~1e-5 |
| zeros_debugging | flow separation, volume drift/zeroing, WOB (no EMG) | NEP303_V2_Peak240W.txt | ok 1.4s | 39/48 | n/a | PTP-fix; WOB~1e-5 |


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

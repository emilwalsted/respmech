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


## Ground truth (Emil's decision)

**Ground truth = the input data processed by the NEWEST algorithms.** The old
runs' spreadsheets can be outdated and are **not** the correctness target where they
differ from the newest code. The golden reference is therefore the **newest code's
output** on the inputs (captured here from the current repo). Consequently, a
difference between "current code" and an "expected" spreadsheet below is attributed
to **outdated run data**, not a code error — unless it is one of the genuine bugs
called out in §D.

### Is `master` actually the newest algorithm? (verified)

Checked across all branches. Result:
- **ECG removal (`subtractecg`/`remove_ecg`) — `master` IS newest.** `master`'s only
  two post-divergence EMG commits are non-algorithmic (`de4d0f2` comments out a
  debug plot; `41097aa` renames `simps`→`simpson`). Numerically identical to the
  branches.
- **Noise reduction (`reducenoise`) — a NEWER variant exists OFF `master`.** Branch
  `origin/resampling-options` passes `win_length=len(noise)**2` to `removeNoise`
  (master does not) and also carries the **resampling** feature that the production
  settings' `processing.sampling` section requires. `master` and
  `resampling-options` have **diverged** (each has commits the other lacks), so
  neither is a strict superset.
- **Neither branch reproduces the old EMG spreadsheets** (both 0/42; the
  `win_length` change alone shifts noise-reduced RMS by >10×), consistent with the
  spreadsheets being outdated.

**FLAG for Emil:** the canonical "newest" for the EMG/noise path is a **merge of
`master` + `origin/resampling-options`** (ECG from master ≡ branch; noise-reduction
`win_length` and resampling from the branch). Until you confirm the canonical
baseline, the **EMG/noise numbers in `production_golden.json` are PROVISIONAL**
(captured from `master`) and should be regenerated from the confirmed newest code
and validated for physiological plausibility — not against the old spreadsheets.
The mechanics/WOB/volume-separation golden is unaffected and solid.

## Root-cause analysis of the mismatches

All differences below are **characterised, not fixed** (fixes are Phase 2). The
current repo code (`master`) was run unmodified in the pinned env (Python 3.12 /
SciPy 1.13); "expected" = the (possibly outdated) spreadsheets Emil generated.

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

**Decision (Emil): PTP baseline is PARKED for a post-refactor review** (see
`PLAN.md` "Post-refactor review"). The possible double baseline subtraction
(`- pressure[0]` on top of `adjustforintegration`) is **not** touched now; the
current `master` PTP behaviour is preserved as-is in the golden until then.

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

### C2. EMG columns and the ground-truth decision
Because ground truth is the newest code (not the old spreadsheets), the EMG
mismatches are **not** counted as errors. They must be re-established from the
confirmed canonical newest code (see the FLAG above), then validated for
physiological plausibility.

### D. Runtime failures on the current code — real latent bugs vs legitimate preconditions

Emil's decision: the newest code must actually **run on all valid real input**.
Distinguish two kinds of failure:

**(i) Genuine bugs — fix in Phase 2, then establish the golden for these files from
the fixed code** (validated for physiological plausibility, since the old
spreadsheets are outdated):
- **`RIU_H5_IC`, `RIU_H6_IC`, `RIU_H6_Baseline` crash** with
  "setting an array element with a sequence … shape (4,)". This is latent bug #1
  (the EMG overview plot builds an ignored-breath `Rectangle` width as a 1-element
  array). These files have **excluded breaths + EMG**, so the always-on EMG plot
  crashes — the current code cannot process them at all. These are **valid inputs**,
  so the refactor must fix this and produce output for them.
- **The `processing.sampling` (resampling) settings section** crashes `master`'s
  `applysettings` (KeyError) because its naive default-merge cannot handle a nested
  subsection absent from defaults (bug #6). Stripped before running here (recorded
  per file). The refactored settings loader must accept this gracefully.
- **Processed-data export hardcodes 5 EMG channels** (bug #3) — not hit here (these
  use 5), but a real bug for any other channel count.

**(ii) Legitimate precondition failures — must be handled with a clear error, not
forced to produce output:**
- **`NEP301_V2_140W 1`, `…testbegin`, `…testend` (Trimming) fail** with "Could not
  trim data to whole breaths" — these are deliberate trim edge-case files whose
  data does not start in late expiration / end in early inspiration, so `trim()`
  produces an empty range. The refactor should surface a **clear, actionable error**
  (which input/precondition failed) rather than a generic trace — but it is correct
  to refuse to process them.

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

# Cardiac-gated peak EMG (`processing.emg.robust_peak`)

_Opt-in, default off. Enabling it **adds** columns and never changes the existing ones;
`tests/golden` passes untouched with the feature both off and on._

Companion to [`NOISE_ECG_OPTIMIZATION.md`](NOISE_ECG_OPTIMIZATION.md), which governs the ECG
**removal** stage. This document is about the **statistic** computed afterwards.

## The problem

The per-breath EMG number RespMech reports is `np.max` of a rolling RMS
(`core.emg.calculate_rms`). A maximum over one short window is the most artefact-sensitive
statistic available, and averaged-template ECG subtraction leaves a sharp beat-to-beat
residual for it to find. Measured on the production sets:

| | H5 (strong cardiac coupling) | H6 (weak) |
|---|---|---|
| in-beat ÷ between-beat envelope, raw | 3.6 – 23× | 1.2 – 10× |
| breaths whose reported max sits on a heartbeat | 83 – 100 % | 6 – 60 % |

Beat-lock on H5 is ~85–90 % of breath-channel cases against a ~55 % chance baseline
(permutation test clustering on breaths, p = 0.0085; cluster bootstrap CI [0.82, 1.00]).
Dropping the single worst beat window moves the reported number by 16–22 %.

Worse for a **relative** measure: under `normalization = "per_file_max"` the denominator is
itself a per-breath maximum, and it is beat-locked in 60 % of RMS columns (81 % on H5). Every
reported percentage is then a fraction of a heartbeat.

Against a synthetic ladder with a known relative drive profile, the shipped statistic recovers
a slope of **0.187** (1.0 = faithful) — it flattens the drive curve to a fifth of its true
range — with MAE 26.9 percentage points and Spearman 0.301 against truth.

## What the feature does

Blank `± gate_half_width_s` of the RMS envelope around every detected R-peak and take the
maximum of what survives, per breath and per phase. The R-peaks are the ones the ECG-removal
stage already found (`pipeline._ecg_remove` keeps them in its diag dict), so nothing new is
detected. On the same synthetic ladder the gated statistic recovers slope **0.434**, MAE 21.9
pp, Spearman 0.582 — better than the shipped statistic wherever contamination exists, and
indistinguishable from it where it does not.

**Losing samples is not a cost.** EMG is reported relative to the patient's own maximum, so a
transformation applied identically to every breath *and* to the normalising maximum cancels in
the ratio — the same design constraint that governs `NOISE_ECG_OPTIMIZATION.md`. Verified two
ways: the residual `gated/shipped` ratio has no drive correlation (Spearman vs V<sub>T</sub>
+0.099, p = 0.38), and blanking the same number of samples at the *wrong* cardiac phase leaves
the reported value unchanged (median ratio 1.0000). Data loss is free in the median; it is
lossy in the tail (P(within 5 %) = 0.742 pooled, 0.620 at HR 162), which is what the survival
guard below exists to catch.

## Settings

```toml
[processing.emg.robust_peak]
enabled = true            # everything below has a working default
gate_half_width_s = 0.120 # ≈ rms_window_s + QRS duration, with margin
min_survival = 0.40       # phase guard: fraction of the phase outside the gates
min_island_s = 0.20       # phase guard: longest contiguous cardiac-free run
long_rr_factor = 1.6      # file guard: an RR interval this much above the median = missed beat
max_long_rr_frac = 0.02   # file guard: tolerated fraction of such intervals
hr_ceiling_margin = 0.10  # file guard: margin below the 60/ecg_min_distance_s ceiling
```

Requires `remove_ecg = true` — without it there are no R-peaks and the gated columns are NaN.

## Output

21 extra columns per file, in the same shape as the existing RMS columns:

```
rms_gated_col_<n>       rms_gated_max        rms_gated_mean
rms_gated_insp_col_<n>  rms_gated_insp_max   rms_gated_insp_mean
rms_gated_exp_col_<n>   rms_gated_exp_max    rms_gated_exp_mean
```

Note that `rms_gated_max` (max across channels) is **not** simply a rescale of `rms_max`: the
ungated cross-channel max preferentially selects the most cardiac-coupled electrode pair, and
the gated one does not. On H5_40W the per-channel ratio is 0.14 while the cross-channel ratio
is 0.66 — the aggregate is repaired too, not just the per-channel value.

## The guards — why NaN is the right answer

Gating is only as good as the R-peak detection it is built on. An **undetected** beat is
neither subtracted by the template nor blanked by the gate, so the gate discards the honest
samples and leaves the missed beat fully exposed. Measured by deleting peaks at random and
judging beat-lock against the true set:

| missed beats | gated beat-lock | reported value |
|---|---|---|
| 0 % | 0 % | 1.0× |
| 5 % | 18 – 20 % | 1.0× |
| 10 % | 28 – 34 % | 1.0 – 2.0× |
| 20 % | 54 – 60 % | 6.5 – 28× |
| 40 % | 82 – 93 % | 11 – 37× |

Gating never becomes *worse* than not gating, but by ~20 % dropout it has stopped doing
anything while the number has inflated by an order of magnitude. The inflation is largest
where the true EMG is smallest — quiet breathing is hit hardest. Hence two guard layers:

**Per file** (`core.emg.detection_quality`) — the fraction of RR intervals exceeding
`long_rr_factor` × the median (a missed beat produces a roughly doubled interval), and whether
the heart rate is approaching the detector's own refractory ceiling, `60 / ecg_min_distance_s`.
Either failing makes all gated columns NaN and emits a warning naming the reason.

**Per phase** (`core.emg.gated_peak_rms`) — surviving fraction and longest cardiac-free island.
Too little left to measure makes that phase NaN.

Separation on the production data is clean: the long-RR fraction is 0.000 on all five H5 files
and on H6_40W/60W/Baseline, 0.042 on H6_IC, 0.267 on H6_Peak220W — so the default 0.02
threshold flags exactly the file whose gating is known to be broken.

**Known limitation.** `ecg_min_distance_s` doubles as a refractory floor, so the shipped
production setting of 0.3613 s imposes a 166 bpm ceiling. A peak-exercise file at 162 bpm
(RIU_H5_Peak180W) is inside the margin and is refused. That refusal is correct — at that heart
rate detection is about to start missing beats — but it means the feature declines to report
on the fastest files rather than gating them badly.

## Approaches that were measured and rejected

Kept here so they are not re-attempted. All were tested against ground truth on the production
data; none reduced beat-lock, which is the metric that decides whether the reported number is
cardiac.

| Approach | Why it failed |
|---|---|
| Lung-volume-phase ECG template (5 bins) | 29 of 30 contaminated maxima stayed beat-locked; sign of the change flipped breath to breath; not significant after multiplicity correction |
| Envelope-power subtraction (NNLS bumps at R-peaks) | Beat-lock 89 % → 85 % at HR 83, 100 % → 100 % at HR 162. The residual is beat-to-beat **shape** variation; giving the model per-beat amplitude *and* timing did not move beat-lock at all |
| Multichannel BSS (ICA / CCA / piCA) | The five esophageal pairs span an effective rank of only ~1.35 (mean inter-channel r = 0.73). An oracle subspace projection destroys 93 % of the EMG for 20 dB of cardiac suppression and still leaves 1.9× contamination — worse on both axes than the template subtraction that already ships |
| Filter-first template estimation | Inside the beat window the coherent cardiac component already dominates EMG at every frequency below ~400 Hz (Wiener weight 0.88–1.00), so there is no EMG jitter to filter out: pre-filtered alignment picks bit-identical shifts on 55 % of beats. A filter aggressive enough to change the alignment makes the template *blunter* (peak −16 %, half-width ×3.5) |
| Plain high-pass of the RMS band | The typical sample gets cleaner (contamination 8.4× → 1.6× on raw at 250 Hz) but beat-lock **rises** (87 % → 95 %): the residual is a sharp transient, and high-pass preserves transients while removing the smooth EMG envelope |

The pattern is consistent: every method that tries to *model and subtract* the residual hits
the same wall, because what is left after template subtraction is incoherent beat-to-beat shape
variation that no fixed template, spatial filter or frequency band can represent. Only
**discarding** the contaminated samples fixes beat-lock — precisely because it makes no attempt
to model the residual.

## Implementation

| File | Change |
|---|---|
| `core/settings.py` | `RobustPeakSettings` dataclass; `EmgSettings.robust_peak` |
| `core/emg.py` | `rolling_rms`, `detection_quality`, `gated_peak_rms`. `calculate_rms` is **not** touched — it is golden-pinned; `rolling_rms` is a separate implementation and a unit test asserts its maximum equals `calculate_rms`'s value |
| `core/compute.py` | `_add_gated_peaks`; `calculatemechanics(..., peaks_s, detection_ok, detection_reason)` — all defaulted, so existing callers are unaffected |
| `core/pipeline.py` | per-file detection-quality judgement before the breath loop |
| `core/results.py` | the extra columns, guarded on the setting |
| `core/_legacy_ns.py` | passes `robust_peak` through |

Tests: `tests/unit/test_robust_peak.py` (14). `tests/golden` (9) and `tests/unit` (348) pass.

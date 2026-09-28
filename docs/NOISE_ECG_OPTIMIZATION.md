# EMGdi noise reduction & ECG removal — optimisation study

_Started as exploratory analysis; **Emil approved making this canonical**. The
shared-profile, fidelity-gated design below is now **implemented** in
`respmech.core.noise` + the pipeline (Phase-2 EMG milestone 1). The EMG golden is
regenerated from this canonical code in the settings/CLI/UI milestone. Measurements
were run on the real H5/H6 production sets (same individuals)._

## Design constraint (governs everything below)

EMG is quantified **relative to a patient+test-specific maximum**, so **every file
in a test must undergo the *identical* transformation** or the relative values are
not comparable. The optimisation target is therefore **not raw SNR** but *SNR
improvement subject to (a) not over-subtracting real EMG and (b) one shared,
reusable noise/ECG parameterisation applied identically to all files*.

## 1. Current state (precise mapping)

### 1.1 Noise reduction — `emg.reducenoise` → `emg.removeNoise`
A stationary **spectral-gating** filter (Sainburg method; the predecessor of the
`noisereduce` library): estimate a per-frequency threshold `mean + n_std·std` from a
noise clip, then attenuate signal STFT bins below threshold, smoothed in time/freq.

Hardcoded / effective parameters:

| Parameter | `removeNoise` default | What `reducenoise` actually passes |
|---|---|---|
| `n_fft` | 2048 | **`len(noise)**2`** ⚠️ |
| `win_length` | 2048 | 2048 (master) / **`len(noise)**2`** (`resampling-options`) ⚠️ |
| `hop_length` | 512 | 512 |
| `n_std_thresh` | 2 | 2 |
| `prop_decrease` | 1 (remove 100 % below threshold) | 1 |
| `n_grad_freq` / `n_grad_time` | 0 / 4 (mask smoothing) | 0 / 4 |
| mode | stationary (one mean+std profile) | stationary |
| frequency range | full 0…fs/2 (no band limiting) | full |

**The noise clip** is extracted from the `noise_profile` entry
`[file, source_file_or_empty, [t0, t1]]`: the sub-interval `[t0,t1]` of `file` (or of
`source_file` if given). In the H5 settings the exercise files point their
`source_file` at `RIU_H5_Baseline.txt` — so a shared *source* is already possible.

**Two serious problems with the current parameterisation** (verified):
1. **`n_fft = len(noise)²` is degenerate.** For H5 the noise interval is 0.05 s =
   **100 samples → n_fft = 10 000** (a 5 s STFT window at 2 kHz); the Baseline file
   uses 0.30 s → **n_fft = 360 000** (180 s window!). librosa warns *“n_fft too large
   for input length=100”*. With a 100-sample clip and n_fft ≥ its length there is
   effectively **one STFT frame → std = 0 → threshold = mean**, i.e. a meaningless,
   unstable noise estimate. Measured effect: the current setting is **almost a no-op**
   (ΔSNR ≈ −0.15 dB, fidelity ≈ 1.2 — it barely touches the signal).
2. **The transformation is NOT identical across files.** Because `n_fft` (and, on
   `resampling-options`, `win_length`) is tied to the clip length, the H5 **Baseline**
   file (0.30 s → n_fft 360 000) is filtered with completely different parameters than
   its siblings (0.05 s → n_fft 10 000). This **violates the shared-transformation
   requirement** even though the noise *source* is shared.

### 1.2 ECG removal — `emg.remove_ecg` → `emg.subtractecg`
R-peaks are detected with `scipy.signal.find_peaks(peakch, height=minheight,
distance=mindistance·fs, width=minwidth·fs)` on the `column_detect` channel. A
**time-aligned averaged ECG template** is built over `windowsize` (0.4 s) windows
around the peaks (alignment search ±0.2 s); each beat is then time-aligned
(`timeshift_average`) and amplitude-fitted (`amplitude_average`, range ±1.25, 1000
steps) and subtracted. Parameters (`minheight`, `mindistance`, `minwidth`,
`windowsize`, `column_detect`) are **per-file settings**; the **template is rebuilt
per file** from that file's own beats (adaptive — acceptable, since ECG morphology is
stable within a subject and ECG is a contaminant, not the EMG being normalised).
Measured on H5 Peak180W: 22 R-peaks, peak-window RMS 0.165 → 0.049 (**69.9 %
suppression**).

## 2. Metric (physiologically meaningful, over-subtraction-aware)

Computed in the diaphragm EMG active band **20–250 Hz** (Welch PSD):

- **In-band SNR** = `10·log10( P_band(inspiration) / P_band(expiration) )`.
  Inspiration = diaphragm-active; expiration = quiet reference (masks from flow
  segmentation).
- **Fidelity** = `P_band(inspiration, processed) / P_band(inspiration, raw)` — the
  fraction of true inspiratory EMG power retained. **This is the over-subtraction
  guard**: a filter can inflate SNR by removing power everywhere; fidelity catches it.
- **ECG residual suppression** = `1 − RMS_±40ms-around-R(processed) / RMS_raw`.

**Objective:** maximise ΔSNR **subject to fidelity ≥ 0.8** (retain ≥ 80 % of
inspiratory EMG power). Anything that raises SNR while fidelity collapses is
rejected as over-subtraction.

## 3. Parameter sweep (real H5, Peak180W, most diaphragm-like channel)

Raw in-band inspiratory-vs-expiratory SNR ≈ **3.5 dB** (modest separation — typical
for surface/oesophageal EMGdi).

### 3a. With the current 0.05 s noise clip
Every fixed `n_fft` warns *“too large for length=100”*; the estimate is unstable and
**over-subtraction is severe** — e.g. `n_fft=256, n_std=1.5, prop=1.0` gives ΔSNR
+6.6 dB but **fidelity 0.07** (93 % of EMG destroyed). No setting reaches
fidelity ≥ 0.8. Conclusion: **the noise clip is too short**.

### 3b. With a PROPER noise profile (≈16 s of concatenated EMG-free Baseline expiration)
A long, diaphragm-quiet noise sample gives a stable per-frequency estimate. Frontier
(fixed `n_fft`, `win=n_fft`, `hop=n_fft/4`, `n_grad_time=4`):

| n_fft | n_std | prop_decrease | ΔSNR (dB) | fidelity | verdict |
|---|---|---|---|---|---|
| 256 | 1.0 | 0.3 | +1.4 | 1.07 | safe, gentle |
| 256 | 1.0 | 0.5 | +2.1 | 0.97 | safe |
| 256 | 1.0 | 0.7 | +2.6 | 0.91 | safe |
| **256** | **1.0** | **1.0** | **+3.3** | **0.84** | **best ΔSNR with fidelity ≥ 0.8** |
| 256 | 1.5 | 1.0 | +5.9 | 0.43 | ❌ over-subtracts |
| 256 | 2.0 | 1.0 | +8.5 | 0.23 | ❌ over-subtracts |
| 256 | 3.0 | 1.0 | +8.0 | 0.006 | ❌ destroys EMG |
| 512 | 1.0 | 0.3 | +1.9 | 0.89 | safe |
| 512 | 3.0 | 1.0 | +13.9 | 0.000 | ❌ (the "great SNR" trap) |

**`n_std_thresh` dominates**: 1.0 is gentle (fidelity 0.84–1.07); ≥ 1.5 over-subtracts.
`prop_decrease` trades ΔSNR vs fidelity along a smooth frontier. `n_fft=256` (≈128 ms
at 2 kHz) slightly beats 512 for this band.

### 3c. Rendered figures (real H5/H6 EMG)
Signal figures are kept **out of this public repo** (real research recordings, not
committable here); they are kept privately alongside the other `tests/golden/production`
material referenced elsewhere in this file.
- `H5_RIU_H5_Peak180W_ch4_time.png` / `_psd.png` — raw vs current (near-no-op) vs
  proposed; proposed removes the 20–50 Hz noise floor while preserving the 80–200 Hz
  EMG (ΔSNR +2.35 dB, fidelity 0.94).
- `H5_RIU_H5_60W_ch2_time.png` / `_psd.png` and `..._variants_*` — an already-clean
  channel where prop_decrease=0.6 over-subtracts (fidelity 0.54); prop 0.3/0.2 lift
  fidelity to 0.76/0.88.
- `H6_RIU_H6_Peak220W_ch0_*` / `H6_RIU_H6_40W_ch4_*` — H6 confirms the pattern.
- `OVERVIEW_snr_fidelity_vs_prop_decrease.png` — ΔSNR & fidelity vs prop_decrease per
  channel with the fidelity ≥ 0.8 line: **the safe choice varies per channel/subject.**

### 3d. Per-channel frontier is now computed automatically (implemented)
The pipeline builds the shared profile from an **expiration-based rest reference** —
for H5 that is **16.0 s of Baseline expiration → 497 STFT frames** (vs the ~7 frames
of the legacy 0.05 s interval), a stable per-frequency estimate. It then auto-selects,
from the whole-test frontier, the highest `prop_decrease` keeping the **worst channel**
≥ target (0.8):

| Test | frames | auto `prop_decrease` | binding channel | per-channel fidelity / ΔSNR |
|---|---|---|---|---|
| H5 | 497 | **0.5** | ch4 = 0.812 | fidelity 0.81–1.06, ΔSNR +0.26…+1.71 dB |
| H6 | ~440 | **0.2** | ch2 = 0.839 | fidelity 0.84–1.13, ΔSNR +0.03…+1.20 dB |

The stable reference lets H5 denoise more (0.5 vs the 0.2 the borderline reference
forced) while every channel stays ≥ 0.8. The chosen value is applied **identically to
all files** (verified by test).

### Recommended noise-reduction settings
- **`n_fft = 256`, `win_length = 256`, `hop_length = 64`** (decouple from clip length),
- **`n_std_thresh = 1.0`**, **`prop_decrease = 0.5–0.7`** (safety margin; fidelity
  0.91–0.97, ΔSNR +2.1–2.6 dB) — or `prop_decrease = 1.0` for max ΔSNR +3.3 dB at
  fidelity 0.84,
- **noise clip built from ≥ several seconds of EMG-free expiration** of a rest
  reference, not a single 0.05 s gap,
- `n_grad_time = 4`, `n_grad_freq = 0` are fine.

### ECG removal (implemented — milestone 2)
The template-subtraction method is kept (it is sound). Detection/template parameters
(`detect_channel`, `ecg_min_height/distance/width`, `ecg_window_s`) are **test-level
settings applied identically to every file** — never re-tuned per file. The pipeline
now attaches an **ECG suppression report** to each `FileResult.ecg`
(`n_peaks`, `peak_rms_before/after`, `suppression`), computed as the peak-window RMS
reduction on the detect channel (H5 reached ~70 %), so consistency and quality are
visible per test. Tune `detect_channel` to the channel where the R-wave is most
prominent (H5 `0`, H6 `4`).

_Caveats: single-file/channel frontier on one subject; band 20–250 Hz; expiration
assumed diaphragm-quiet. Values illustrate the trade-off and the trap, not final
per-subject numbers — those should be set from each subject's rest reference._

## 4. Shared-profile workflow — **implemented** (`respmech.core.noise`)

The legacy code only *partially* supported this (shared *source file*, but
`n_fft=len(noise)**2`, per-file intervals, and noise stats recomputed per call → not
guaranteed identical). It is now replaced by:

1. **`NoiseProfile`** (`respmech/core/noise.py`) — a serialisable per-channel gate:
   fixed `n_fft/hop/win` + a precomputed `mean + n_std·std` threshold spectrum, built
   **once** from an EMG-free rest reference (explicit intervals or expiration).
2. **`NoiseProfileSet`** — one profile per EMG channel + one `prop_decrease`, applied
   **identically** to every file (`apply_columns`). Built once per test in
   `pipeline.run_batch` from the **whole** test's file list, so a single-file/GUI test
   run denoises exactly as the full batch (verified:
   `test_identical_transformation_regardless_of_batch_subset`).
3. **Fidelity gate** — `fidelity()` (retained inspiratory in-band power) +
   `select_prop_decrease()` picks, once per test, the highest `prop_decrease` whose
   worst active channel stays ≥ `fidelity_target`. Manual `prop_decrease` is also
   supported (`auto_prop=false`). Per-channel fidelity/ΔSNR + the full frontier are
   returned in `BatchResult.noise_report`.

The implementation is the established Sainburg spectral-gating math driven by a fixed,
precomputed threshold (equivalent in spirit to `noisereduce` with a fixed `y_noise` +
`stationary=True`, but with a serialisable profile and no per-call re-estimation). A
too-short reference raises a clear error / warning (≥ ~8 STFT frames recommended).

## 4a. Whole production golden regenerated from the canonical core

The **entire** production golden was regenerated from the new canonical core
(`tests/golden/regen_production_emg_golden.py --write`; locked by
`test_production_golden.py` for Zeros/Trimming and `test_production_emg_golden.py`
for Resampling/H5/H6 — both now drive the **new core**, not the legacy runner). Every
changed column was categorised against the previous golden across **all** scenarios:

| Category | Columns changed | Cause |
|---|---|---|
| **EMG** (`rms_*`, `integral_emg_*`) | 462 | shared-profile noise reduction (all EMG scenarios: Resampling, H5, H6) |
| **PTP** (`int_*`, `ptp_*`) | 64 | the **approved** end-expiratory window baseline (Commit B), now propagated to Zeros/Trimming/Resampling too |
| **OTHER** (timing, VE, WOB, pressures, entropy, volume-separation mechanics) | **0** | — **unchanged**, as required |

Per scenario: Zeros/Trimming changed **only PTP**; Resampling and H5/H6 changed EMG
(+ PTP where not already updated). Three previously-**crashing** files now run
(bug #1 fixed): `RIU_H5_IC`, `RIU_H6_Baseline`, `RIU_H6_IC`. Output is deterministic
(re-run reproduces with zero diff). The synthetic golden is unchanged (noise off
there).

## 4b. Downstream: the peak statistic

This document governs the ECG **removal** stage. A later study established that the residual
that survives removal is dominated by beat-to-beat *shape* variation, and that the shipped
per-breath statistic — `max` of the rolling RMS — lands on that residual in ~85–90 % of
strongly-coupled breath-channels. See
[`CARDIAC_GATED_PEAK_EMG.md`](CARDIAC_GATED_PEAK_EMG.md) for the measurements, the opt-in
cardiac-gated statistic (`processing.emg.robust_peak`), and the list of approaches that were
tried and rejected — including further template refinements, so they are not re-attempted here.

## 4c. ECG auto-detect exposed to the batch pipeline / CLI (opt-in)

`core.emg.suggest_ecg_settings` (the analysis behind the GUI ECG tab's *Auto-suggest*
button) was, until now, only reachable interactively — a settings.toml/CLI-only
workflow had no way to derive `detect_channel`/`ecg_min_height`/`ecg_min_distance_s`/
`ecg_min_width_s`/`ecg_window_s` without opening the GUI first. `processing.emg.
ecg_auto_detect` (off by default) closes that gap: `pipeline._auto_detect_ecg_settings`
runs the same analysis ONCE per test on a reference file (`processing.emg.
ecg_reference_file`, or the batch's first matched file) and applies the 5 derived
parameters identically to every file — mirroring `noise.auto_prop`'s "selected once,
applied identically, never re-tuned per file" model. This is exactly the split the
milestone-2 status above already established: **detection** parameters are shared
across the test; **template subtraction** (`subtractecg`) still rebuilds its ECG
template per file from that file's own beats, unchanged.

Because the 5 shared parameters (in particular `ecg_min_distance_s`, derived from the
reference file's own median R-R interval) come from ONE file, a sibling file with a
materially different heart rate or R-amplitude can have real beats missed without the
run failing. `run_batch` now runs `emg.detection_quality` per file whenever
`ecg_auto_detect` is on (the same guard `processing.emg.robust_peak` already uses) and
emits a `warnings.warn` naming the file and the reference it diverged from — it never
fails the run, so an unsupervised batch keeps going but leaves a visible trail of files
worth revisiting. `BatchResult.ecg_auto_report` (reference file, chosen channel,
confidence, estimated bpm) is also printed by `respmech run` and written to
`run-report.txt`, so a CLI-only user can actually see when auto-detect fell back to a
low-confidence/middle-channel guess.

The GUI now has an equivalent entry point: an **"Auto (whole batch)"** checkbox on the
Preview ECG tab (`ui/screens/preview_screen.py`, bound to `ecg_auto_detect`), placed
next to *Remove ECG* and mirroring `noise.auto_prop`'s "Auto" checkbox exactly —
including greying out the capture-channel/min-height/min-gap/Advanced controls it
overrides once ticked. Before this, the field could only be set by hand-editing a
settings.toml outside the app: the underlying analysis (`suggest_ecg_settings`) was
already the one function shared by both the GUI's manual *Auto-suggest* button and the
CLI's automatic per-batch detection, but only CLI/TOML users could reach the
batch-wide behaviour. `Settings.validate()` also gained the `ecg_auto_detect` ⇒
`remove_ecg` + `input.channels.emg` cross-check that `pipeline.run_batch` already
enforced at runtime, so both `respmech validate`/`run` and the GUI's Run-button gate
now catch the same misconfiguration before a batch starts, not just mid-run.

## 5. Status

**Done:** the `n_fft`/`win_length` bug fix (fixed STFT params); the shared,
serialisable `NoiseProfile`; an **expiration-based rest reference by default** (~500
STFT frames — stable, no borderline 7-frame estimate); the fidelity metric + gate +
once-per-test auto-selection; the identical-transformation guarantee; per-test ECG
parameters + suppression report; TOML settings + migrator mapping; a GUI fidelity
preview; and the **whole production golden regenerated** from the canonical core (only
EMG + the approved PTP change; everything else intact) — all with tests on real
H5/H6 data.

_(Historical note — the milestone breakdown below is retained for provenance.)_

**Superseded remaining (now done):** ECG detection parameters per test and exposing
them; the new `processing.emg.noise` settings in TOML + migrator mapping and a GUI
fidelity/noise preview; and regenerating the golden from this canonical code —
everything non-EMG/non-PTP is held constant.

## 6. EMG-only noise references

A flow-bearing analysis has always resolved its rest reference from one of two saved
choices: `use_expiration` (the reference file's own expiration, ~500 STFT frames,
stable) or, when that is false, explicit `reference_intervals`. An EMG-only signal set
(no flow channel — nothing to split into inspiration/expiration at all) needed its own
rule, since neither concept means anything without a flow-derived breath.

`core.settings.resolve_noise_reference_mode(settings)` is the one function that decides
which source `_reference_noise_clip` actually builds from, returning `'expiration'` |
`'intervals'` | `'rest_segments'` | `'interburst'` | `'unresolved'`. `processing.emg.
noise.reference_mode` (default `'auto'`) drives it:

* **With a flow channel declared**, `'auto'` reproduces the existing rule exactly —
  `'expiration'` when `use_expiration` is true (or no intervals are set), `'intervals'`
  otherwise. `use_expiration`/`reference_intervals` remain the two saved fields for this
  case; `reference_mode` adds nothing new here.
* **Without a flow channel**, `'auto'` looks for a segment typed `'rest'` in the
  reference file (`BREATH_KINDS` — a right-clicked, or hand-edited, segment; segment
  numbers are `whole_file`/`separators`' own segment numbers, not breaths). If one
  exists, the mode is `'rest_segments'` and the clip is the concatenation of every
  `'rest'`-typed segment's EMG. Otherwise, explicit `reference_intervals` still work
  (`'intervals'`); with neither, the mode is `'unresolved'` and `Settings.validate()`
  refuses to enable noise reduction at all, rather than let an unbuildable profile reach
  the pipeline mid-batch.
* `'rest_segments'`/`'interburst'` can also be set explicitly, but only for an EMG-only
  set — `Settings.validate()` rejects either while a flow channel is declared, and either
  without a `reference_file` while noise reduction is enabled. `'interburst'` is the
  quiet periods between the bursts of the automatic `emg_burst` segmentation (each
  shrunk by a guard band, measured from the wide coarse edges of the bursts; see
  `docs/REVERSE_ENGINEERING.md` §5.12) and is accepted only with that method.

**Decision:** a typed breath (any manoeuvre kind — IC, FVC, a maximal effort, `'rest'`
itself, or `'other'`) is excluded from the flow-bearing reference file's own expiration
mask when building the `'expiration'`-mode clip: its expiration is not the
diaphragm-quiet period the profile is trusted for. A no-op for the overwhelming common
case (no typed breaths at all) — every existing scenario is unaffected. See
`core.pipeline._emg_segmented`'s `exclude_typed_from_expiration` parameter.

**Automatic strength (`auto_prop`) for an EMG-only set** exists only for the `emg_burst`
segmentation, where "active" is the bursts and "quiet" the periods between them
(`_emg_segmented` returns those masks instead of inspiration/expiration ones). For the
other EMG-only methods there is no such split (a rest-typed active/quiet split is not
built), so `Settings.validate()` requires `auto_prop` off whenever noise reduction is
enabled for them: a clear, named restriction rather than an unguarded crash. A file with
no quiet period at all is left out of the pooled sets, like an unreadable one.

## 7. The expiration mask now follows the analysis's own segmentation, not a hardcoded copy

`core.pipeline._emg_segmented` builds the `'expiration'`-mode reference clip (and
gathers the active/quiet EMG `auto_prop` samples from) by segmenting a file into
breaths and reading each one's inspiration/expiration sample counts. Until now it did
this with its own private, hardcoded copy of the trim/zero/drift/segment sequence:
always flow-method breath separation, and volume drift correction only (never trend
correction), regardless of what `processing.mechanics.separateby`/`processing.volume.
correct_trend` actually said. A volume-segmented or trend-corrected analysis therefore
built its noise masks from breaths that did not match the ones the analysis itself
used — the mask and the analysis could quietly disagree about where inspiration ends
and expiration begins.

`_emg_segmented` now calls `core.pipeline.segment_file` (the same trim/zero/drift/
trend/segment sequence `run_batch`'s main loop and `_rest_segments_clip` already use)
instead of re-implementing a piece of it. The mask it returns is therefore always
built from the SAME breaths the analysis itself will use, whatever the configured
method or trend setting. This is a deliberate, documented numerical change for
exactly the combinations that could previously disagree — noise reduction enabled AND
(`separateby == 'volume'` OR `correct_trend` on) — every flow-method, no-trend
scenario (every existing golden/synthetic scenario, and the overwhelming common case
in practice) is unaffected, since `segment_file` reduces to byte-identical behaviour
there. See `tests/unit/test_noise.py::test_emg_segmented_mask_equals_main_loop_mask`.

A caching note for anyone touching this again: `segment_file`'s own per-file load/
ECG-removal cache lookup is consuming (`cache.pop`), by design, for the main loop's
single pass over each file. `_emg_segmented` is a SECOND, earlier caller of
`segment_file` for the same file (during noise-profile building, before the main
loop reaches it), so it must not let that pop drain the shared cache — see the
function's own docstring for how it hands `segment_file` a throwaway one-entry cache
instead, keeping the real one intact for the main loop's later call
(`tests/unit/test_load_cache.py` pins the "loaded/ECG-removed at most once per run"
invariant this preserves).

A UI-cache note, for the same reason: `ui.screens._preview_cache.ref_clip_key`/
`noise_report_key` (the Preview screen's memoisation of the reference-clip/
noise-fidelity computation) previously excluded volume drift and trend correction
from their keys on the stated grounds that the mask never depended on them. That was
already only half true for `method == 'volume'` — `separateintobreathsbyvolume`
searches for inspiratory/expiratory peaks by an absolute height threshold against the
drift-(and, once configured, trend-)corrected volume, so drift correction alone can
shift which samples are detected as peaks, independently of trend. Both keys now
include drift, trend and the volume-peak thresholds (`_method_sensitive_key` in that
module) wherever they already depend on the segmentation method, so a Preview panel
can no longer keep showing a fidelity/reference-clip result computed under a
volume-peak, drift or trend setting the user has since changed.

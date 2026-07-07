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
Signal figures are kept **out of this public repo**; they live in Emil's Dropbox:
`/Users/emilwalsted/Dropbox/RespMech/figures/`
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
Running the implemented shared-profile pipeline on the full H5 test auto-selects, from
the whole-test frontier, the highest `prop_decrease` keeping the **worst channel** ≥
target (0.8) — here **prop_decrease = 0.2** (ch0=0.874 is the binding channel; prop=0.3
would drop ch0 to 0.751). Per-channel at the chosen value: fidelity 0.87–1.23, ΔSNR
+0.14…+0.59 dB. Conservative but guaranteed non-destructive across every channel, and
**identical for all five files** (verified by test).

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

## 5. Status & remaining

**Done (this milestone):** the `n_fft`/`win_length` bug fix (fixed STFT params), the
shared serialisable `NoiseProfile`, the fidelity metric + gate + auto-selection, and
the identical-transformation guarantee — all with tests, on real H5/H6 data.

**Remaining (later milestones):** fix the ECG detection parameters per test and expose
them (milestone 2); expose the new `processing.emg.noise` settings in TOML + migrator
mapping and a GUI fidelity/noise preview (milestone 3); **regenerate the EMG golden**
from this canonical code and document the before/after on the EMG columns (milestone
3). Everything non-EMG in the golden is held constant.

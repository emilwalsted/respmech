# EMGdi noise reduction & ECG removal — optimisation study

_Exploratory analysis + recommendation. **No canonical behaviour or golden was
changed** — this awaits Emil's decision and is tied to the open "canonical EMG/noise
baseline" question (master + `resampling-options`). Measurements were run on the real
H5 production set (same individual: Baseline / 40W / 60W / IC / Peak180W)._

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

### Recommended noise-reduction settings
- **`n_fft = 256`, `win_length = 256`, `hop_length = 64`** (decouple from clip length),
- **`n_std_thresh = 1.0`**, **`prop_decrease = 0.5–0.7`** (safety margin; fidelity
  0.91–0.97, ΔSNR +2.1–2.6 dB) — or `prop_decrease = 1.0` for max ΔSNR +3.3 dB at
  fidelity 0.84,
- **noise clip built from ≥ several seconds of EMG-free expiration** of a rest
  reference, not a single 0.05 s gap,
- `n_grad_time = 4`, `n_grad_freq = 0` are fine.

### ECG removal
The template-subtraction method is sound. Suggested: keep it, tune `minheight`/
`mindistance`/`minwidth`/`column_detect` per subject from the channel where the R-wave
is most prominent (H5 used `column_detect=0`, H6 `=4`); verify suppression with the
peak-window-RMS metric (H5 reached ~70 %). Detection parameters should be **fixed per
test** so every file is treated identically.

_Caveats: single-file/channel frontier on one subject; band 20–250 Hz; expiration
assumed diaphragm-quiet. Values illustrate the trade-off and the trap, not final
per-subject numbers — those should be set from each subject's rest reference._

## 4. Shared-profile workflow (the mathematically-required design)

**Does the current code already do this? Partially — and not correctly.** It can
share the noise *source file*, but (i) `n_fft`/`win_length` tie the transform to clip
length, (ii) intervals differ per file (H5 Baseline ≠ siblings), and (iii) the noise
statistics are recomputed inside every `removeNoise` call. So the transformation is
**not guaranteed identical** across a test. **This needs a small refactor.**

**Proposed design — one reusable noise/ECG profile per test:**
1. Build a **`NoiseProfile` artefact once** from a rest reference: concatenate
   EMG-free expiration (or an apnoea segment), compute the per-frequency
   `mean+n_std·std` threshold spectrum with **fixed** `n_fft/hop/win`. Store it
   (e.g. the noise clip + fixed params, or the precomputed threshold array).
2. Apply the **same** profile + params to **every** file in the test (identical gate
   ⇒ comparable RMS). `noisereduce` (the maintained Sainburg successor) supports
   passing a fixed `y_noise` + `stationary=True`; even cleaner is to precompute the
   threshold spectrum once.
3. Likewise fix the **ECG detection parameters** per test; the per-file adaptive
   template is acceptable (contaminant removal), but the detection settings must be
   shared.
4. Settings shape: a single `[processing.noise_reference]` (file + interval(s) +
   fixed STFT params) reused for all files, replacing the per-file `noise_profile`
   list — this makes "identical transformation" the default, not a coincidence.

We had a representative multi-file, same-individual set (H5) to design against, so no
extra data is needed to validate the workflow; a dedicated rest/apnoea reference
recording per test would make the noise profile cleaner still.

## 5. Recommendation & next step (for Emil)

1. **Fix `n_fft`/`win_length`** (decouple from `len(noise)`) — the current values are
   a bug that makes noise reduction a near-no-op and non-identical across files.
2. Adopt **`n_fft=256, n_std_thresh=1.0, prop_decrease≈0.5–0.7`** with a **long
   EMG-free noise clip**, prioritising **fidelity ≥ 0.8**.
3. Introduce a **shared noise/ECG profile artefact** applied identically to every file
   in a test (§4).

These change EMG RMS outputs, so — like the canonical EMG/noise baseline question —
they must be decided by Emil and then locked with a regenerated EMG golden. **Nothing
here is applied yet.**

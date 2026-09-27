# RespMech — Reverse-engineering & correctness reference

_Phase 1 documentation for the production refactor. Describes the **current**
(`v1.0.0`) code as it actually behaves, with special attention to every
physiological calculation (formula, units, assumptions). This is the reference a
reviewer uses to confirm the refactor preserves behaviour._

Source files analysed (v1.0.0, now frozen under [`legacy/`](../legacy/README.md)):
[`respmech.py`](../legacy/respmech.py) (1660 lines), [`emg.py`](../legacy/emg.py) (585),
[`entropy.py`](../legacy/entropy.py) (329), [`example.py`](../legacy/example.py).

---

## 1. What the software does

RespMech analyses time-series respiratory recordings (e.g. exported from ADInstruments
LabChart) and computes, **breath-by-breath and averaged**:

- **Respiratory mechanics** — timing (Ti, Te, Ttot, Ti/Ttot), tidal volume (VT),
  breathing frequency (bf), minute ventilation (VE), oesophageal/gastric/
  transdiaphragmatic pressure descriptors, pressure–time products (PTP), an
  isovolume lung-resistance estimate, and a gastric/oesophageal pressure ratio.
- **Work of breathing (WOB)** — inspiratory elastic, inspiratory resistive and
  expiratory WOB from the Campbell diagram, in Joules and Joules·min⁻¹.
- **Diaphragm EMG** — root-mean-square (RMS) and integrated EMG per channel, with
  optional ECG removal and spectral noise reduction.
- **Sample entropy** — of selected channels (e.g. diaphragm EMG), per breath.

It also writes diagnostic PDF plots (raw/trimmed signals, volume-drift correction,
per-breath and averaged Campbell diagrams, EMG overviews) and optional WAV exports
of EMG channels.

---

## 2. Architecture & entry points

The code is a **single-run batch script**, not a library or a package.

```
example.py  (per-project "settings file": a Python script)
   │  1. importlib-loads respmech.py from an absolute path
   │  2. defines a nested `settings` dict
   └─ 3. calls respmech.analyse(settings)
                     │
respmech.py ─ analyse(usersettings)  ── the single pipeline entry point
                     │  merges settings over JSON defaults, validates
                     │  for each input file:
                     │     load → trim → (EMG: ECG-removal/noise) → drift-correct
                     │     → separate into breaths → per-breath mechanics/WOB/
                     │       entropy/RMS → average → write Excel + plots
                     ├─ emg.py       (RMS, ECG removal, spectral noise reduction, EMG plots)
                     └─ entropy.py   (sample entropy; vendored pyEntropy)
```

**Entry points / how it is run today**

- There is **no CLI, no `argparse`, no `__main__` in `respmech.py`**. The unit of
  execution is a *settings file* (`example.py` copied per project) that hardcodes
  the path to `respmech.py`, hardcodes all settings inline, and calls
  `analyse()`. "Batch" = the file glob inside one `analyse()` call (`inputfolder`
  + `files` mask), processed in a single loop.
- `analyse()` installs a global `sys.excepthook` (`catchexceptions`) that writes an
  `Error log.txt` to the current working directory and swallows the traceback.
- `import_file()` re-loads `emg.py`/`entropy.py` by absolute path on **every breath**
  that needs them (not cached).

**Implications for the refactor** (detail in [`PLAN.md`](PLAN.md)):
settings are executable Python (a security & tooling problem); the code path is
import-by-path; there is no separation between computation, I/O and plotting; and
global exception/state handling is unsuitable for a GUI or a testable library.

---

## 3. Data flow & I/O formats

### 3.1 Input

Loader dispatch by file extension (`load()`), input columns selected by **1-based**
column number in settings:

| Format | Handler | Notes |
|---|---|---|
| `.mat` | `loadmatmac` / `loadmatwin` | MATLAB export; **Mac vs Windows LabChart formats differ** (`matlabfileformat` 1=Win, 2=Mac). Win path reads `data["data_block1"]`. |
| `.xls`/`.xlsx` | `loadxls` | via `pandas.read_excel` |
| `.csv` | `loadcsv` | via `pandas.read_csv` |
| `.txt` | `loadtxt` | tab-separated, `decimalcharacter` configurable |

Required channels: **flow**, **poes** (oesophageal), **pgas** (gastric), **pdi**
(transdiaphragmatic). **volume** is optional (can be integrated from flow).
Optional: **EMG channels** (`columns_emg`), **entropy channels** (`columns_entropy`).

Signal-conditioning applied at load time, in order:
1. `inverseflow` → negate flow (convention: **inspiration = negative flow**).
2. `integratevolumefromflow` → `volume = -cumtrapz(flow, t)` (SciPy) if no volume
   channel is supplied.
3. `inversevolume` → negate volume (convention: **inspired volume positive**).
4. `validatedata()` — rejects NaN / non-numeric / unequal-length columns.

### 3.2 Output

Written under `outputfolder`, which **must already contain `data/` and `plots/`
subfolders** (the code `makedirs` them in most but not all paths — historically
required by the README).

- `data/Average breathdata.xlsx` — one row per input file (mean over non-ignored
  breaths), merged across all files. Two sheets: `Data`, `Version`.
- `data/<file>.breathdata.xlsx` — per-breath rows for one file + column formatting.
- `data/<file> – Processed data.csv` — the trimmed, per-breath processed signals
  (Time, Breathno, Flow, Volume, Poes, Pgas, Pdi [, EMG1..EMG5]).
- `plots/*.pdf` — raw / trimmed / volume-correction / Campbell (per-breath &
  averaged) / EMG overviews.
- `plots/*.wav` — optional EMG channel audio (`save_sound`).

---

## 4. Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `xlsxwriter` (Excel engine,
hardcoded), `librosa` (imported at `emg.py` top level — so **any** EMG run needs it),
`openpyxl` (reading `.xlsx`). Vendored: `entropy.py` (pyEntropy, `LICENSE pyentrp`).

**Version fragility (verified):** the current code does **not** run on a modern
SciPy (≥1.14):
- `scipy.integrate.cumtrapz` was **removed** in 1.14 (used for volume integration).
- `scipy.integrate.simpson` made `x` **keyword-only** in 1.14; the code calls it
  positionally (`simpson(y, x)`) in `calcptp`, `calculatewob`, and `emg.calculate_rms`.

The golden tests therefore pin Python 3.12 + SciPy 1.13 (see
[`tests/golden/requirements-golden.txt`](../tests/golden/requirements-golden.txt)).

---

## 5. Physiological calculations (the correctness core)

Conventions used throughout: **flow** negative on inspiration / positive on
expiration; **inspired volume** positive; pressures in **cmH₂O**; flow in **L·s⁻¹**;
volume in **L**; `fs` = `samplingfrequency` (Hz).

### 5.1 Trimming to whole breaths — `trim()`
`startix` = first index with `flow ≤ 0` (onset of first inspiration);
`endix` = last index with `flow ≥ 0` (end of last expiration). All channels are
sliced to `[startix:endix]`. Requires data to **start in late expiration and end in
early inspiration** (per README).
⚠️ `entropycolumns` is **not** trimmed but is later indexed with trimmed
coordinates — see §6 latent issue (2).

### 5.2 Volume conditioning
- `zero(v) = v − v[0]`.
- `correctdrift(v)` — linear de-trend. With `a = (v[N-1] − v[0])/(N-1)`, it adds a
  descending ramp `≈ a·(1 − i)` to sample `i` (removing a linear baseline slope).
  Boundary quirk: the ramp starts at `+a` (not 0) and the **last sample is left at
  0**. Behaviour is preserved by the golden reference; the refactor should keep the
  numeric result identical unless deliberately corrected.
- `correcttrend()` (optional) — subtracts an interpolated envelope through detected
  volume peaks (`volumetrendadjustmethod` = interp kind); also emits a plot.
  **Deliberate v2 divergence** (v2.3.3): legacy gates the troughs on
  `volumetrendpeakminheight`, an *absolute* depth below the file's global volume
  maximum, defaulting to `0.8`. No trough on ordinary tidal breathing reaches it, so
  `find_peaks` returns an empty array and `interp1d` raises `cannot reshape array of
  size 0 into shape (0,newaxis)`; one trough is worse, since `interp1d` accepts it and
  returns an all-NaN envelope silently. v2 keeps that gate byte-for-byte when the
  setting is given, and otherwise selects a per-trough **prominence** ≥
  `trend_peak_min_prominence_frac` × the recording's own volume range. The first and last
  samples are considered as anchors too (an edge is never a local maximum, so `find_peaks`
  can never return one), but only accepted when they actually sit at an end-expiratory
  level — `trim` ends the window at the last `flow >= 0` sample, which is *in* expiration,
  not *at* end-expiration, and returns 0 for a file that already begins mid-inspiration.
  See `compute.trend_anchors` / `_end_is_expiratory`; `legacy/` keeps the original.

### 5.3 Breath segmentation
- **By flow** (`separateintobreathsbyflow`, default): walk the signal; an
  inspiration is the run where `flow < 0` **or** the forward mean over
  `breathseparationbuffer` samples is `< 0` (buffer tolerates zero-crossing
  wobble); the following expiration is the run where `flow > 0` (same buffer rule).
  A breath = inspiration + following expiration.
- **By volume** (`separateintobreathsbyvolume`): `find_peaks` on inspired volume
  (end-inspiration peaks) and on inverted volume (end-expiration peaks), gated by
  `peakheight`, `peakdistance·fs`, `peakwidth·fs`.
- `excludebreaths` marks named breaths `ignored` (kept in plots, dropped from
  averages). `breathcounts` overrides the detected breath count per file for
  per-minute scaling.

### 5.4 Per-breath timing, volume, ventilation — `calculatemechanics()`
- `Ti = n_insp/fs`, `Te = n_exp/fs`, `Ttot = n_total/fs` (seconds); `Ti/Ttot`.
- `VT = max(volume) − min(volume)` over the breath (L).
- `vefactor = 60 / (file_duration_s)`; `bf = breathcount · vefactor` (breaths·min⁻¹);
  `VE = VT · breathcount · vefactor` (L·min⁻¹, minute ventilation).
- Flows: `max_in_flow = −min(flow_insp)`, `max_ex_flow = max(flow_exp)`, and
  mid-tidal-volume inspiratory/expiratory flows.

### 5.5 Pressure descriptors (cmH₂O)
Per breath: `poes_maxexp`, `poes_mininsp`, `poes_endinsp`, `poes_endexp`,
`poes_midvolexp/insp` (Poes at the mid-tidal-volume point); analogous `pgas_*`,
`pdi_*`; tidal swings `p*_tidal_swing = max(p) − min(p)` over the breath;
`insp_pdi_rise = max(pdi_insp) − min(pdi_insp)`;
`exp_pgas_rise = max(pgas_exp) − min(pgas_exp)`.

- **Gastric/oesophageal pressure ratio** `vmr = (pgas_endinsp − pgas_endexp) /
  (poes_endinsp − poes_endexp)` (dimensionless; divide-by-zero guarded → 0).
- **Isovolume inspiratory lung-resistance estimate**
  `tlr_insp = |(poes_midvolexp − poes_midvolinsp) / (flow_midvolexp −
  flow_midvolinsp)|` (cmH₂O·L⁻¹·s), i.e. ΔPressure/ΔFlow at mid-tidal volume.

### 5.6 Pressure–time product (PTP) — `calcptp()` / `adjustforintegration()`
For inspiratory Poes, inspiratory Pdi and expiratory Pgas, the pressure is
baseline-shifted (`adjustforintegration`: min→0, or if wholly negative, max→0; the
caller also subtracts the first sample), then integrated over time with **Simpson's
rule**: `integral = simpson(pressure, t)` where `t = linspace(0, n/fs, n)`.
- `int_* ` = per-breath integral (cmH₂O·s).
- `ptp_* = integral · breathcount · vefactor` (cmH₂O·s·min⁻¹).

**Deliberate v2 divergence (v2.0.0, commit `da0420b`; see
[`docs/PTP_INVESTIGATION.md`](PTP_INVESTIGATION.md)):** v2 drops `adjustforintegration`
and the `- pressure[0]` step. The baseline is the MEAN of the first `n` samples of the
phase, `n = max(1, round(processing.ptp.baseline_window_s · fs))` with
`baseline_window_s = 0.05` s by default (end-expiratory for the inspiratory Poes/Pdi
PTPs, end-inspiratory for the expiratory Pgas PTP): `int = simpson(p − mean(p[:n]),
x=t)`; `ptp = int · breathcount · vefactor`. A window short enough that
`round(baseline_window_s · fs) = 1` reproduces the legacy single-sample value.

### 5.7 Work of breathing — `calculatewob()`  ⭐ most correctness-sensitive
Uses the **Campbell diagram** (oesophageal pressure vs inspired volume). Unit
conversion `WOBUNITCHANGEFACTOR = 98.0638/1000` J·(cmH₂O·L)⁻¹
(1 cmH₂O = 98.0638 Pa; Pa·m³ = J; 1 L = 10⁻³ m³).

Endpoints: `EILV = [V_endinsp, Poes_endinsp]`, `EELV = [V_endexp, Poes_endexp]`.
- **Inspiratory elastic** = triangle EELV–EILV:
  `wob_in_ela = ½ · |ΔV| · |ΔPoes| · factor`.
- **Inspiratory resistive** = area between the Poes trace and the elastic recoil
  line over inspiration: fit line from `(V0,Poes0)` to `(V_end,Poes_end)`, take the
  positive part of `(−Poes) − (−line)`, `wob_in_res = |simpson(that, V)| · factor`.
- **Expiratory** = expiratory Poes above end-expiratory level, positive part
  integrated over volume: `wob_ex = |simpson(max(Poes − Poes_end, 0), V)| · factor`.
- **Totals**: `wob_in_total = ela + res`; `wobtotal = wob_in_total + wob_ex`.
  Each is scaled `· breathcount · vefactor` → **J·min⁻¹** (the per-breath J values
  are the pre-scaling quantities).

`calcwobfrom = "average"` computes WOB from an **averaged breath** (breaths are
resampled to `avgresamplingobs` points via `interp1d` then mean-averaged in
`calculateaveragebreaths()`); `"individual"` computes per breath then averages.
The averaged method is more robust to irregular breaths.

### 5.8 EMG RMS & integrated EMG — `emg.calculate_rms()`
Per channel: a **sliding-window RMS**, window length `rms_s·fs` samples, and the
breath's value is the **maximum** window RMS over the breath. Integrated EMG =
`simpson(|EMG|, t)`. Returns per-channel values plus max/mean across channels.
⚠️ The window is `[i, i+win−1]` (forward-looking), while the docstring/label claim a
±25 ms centred window — behaviour documented as-implemented.

### 5.9 Sample entropy — `entropy.sample_entropy()`
Standard SampEn (Chebyshev norm). `entropy_epochs` is passed to the vendored routine as
the LONGEST template length, i.e. `entropy_epochs = m + 1`; the reported statistic is the
last element of the returned vector, SampEn(m = entropy_epochs − 1). With the default
`entropy_epochs = 3` this is m = 2, the value conventional in the sample-entropy
literature (set 2 for the m = 1 RespMech reported before this default changed,
05-09-2026). Tolerance `r = entropy_tolerance · SD`, where SD is the standard deviation of
that entropy column within the analysed segment, recomputed for each segment (whole
breath, inspiration, expiration); result `SampEn = −ln(A/B)`, reduced to max/min/mean
across channels. Vendored pyEntropy implementation.

### 5.10 EMG signal conditioning (optional)
- **ECG removal** (`remove_ecg`, `emg.remove_ecg`): detect R-peaks (`find_peaks`,
  `minheight`/`mindistance`/`minwidth`), build a **time-aligned averaged ECG
  template** per window, amplitude-fit it to each beat (least squares) and subtract.
- **Noise reduction** (`remove_noise`, `emg.reducenoise`→`removeNoise`): spectral
  gating — STFT the signal and a **noise-profile interval**, threshold each
  frequency bin at `mean + n_std·SD` of the noise, smooth the mask, invert. Adapted
  from Tim Sainburg; **requires `librosa`**.
  **Deliberate v2 divergence** (see [`docs/NOISE_ECG_OPTIMIZATION.md`](NOISE_ECG_OPTIMIZATION.md)):
  v2 replaces the per-call estimate with ONE shared profile per test. `core/noise.py`
  builds one `NoiseProfile` per EMG channel from an EMG-free rest reference
  (`processing.emg.noise.reference_file`: by default every expiration of that
  recording, or the explicit `reference_intervals` when `use_expiration` is off)
  using fixed STFT parameters (`n_fft` 256, `hop_length` 64, `win_length` 256,
  rescaled only when a pre-analysis resample changes the sampling rate), thresholds
  each frequency bin at `mean + n_std_thresh · SD` (default 1.0; v1 used 2) of the
  reference's dB spectrum, and applies that identical profile to every file of the
  test. The suppression strength is also chosen once per test: with `auto_prop` on
  (the default) the strongest `prop_decrease` on the 0.1–1.0 grid whose
  worst-channel in-band (20–250 Hz) fidelity stays at or above `fidelity_target`
  (0.8), falling back to the gentlest value if none qualify; with `auto_prop` off,
  the fixed `prop_decrease` (default 0.6). Reconstruction rescales each STFT bin's
  magnitude while keeping its own complex phase exactly; the earlier reconstruction
  split the real/imaginary parts as if they were independent magnitude/phase terms,
  which inflated in-band power by roughly 30–40% even at `prop_decrease = 0` —
  fixed 05-09-2026 (`core/noise.py::NoiseProfile.apply`). `librosa` is still
  required.
- **RMS outlier handling** (`processoutliers`): `rms_poes = rms_max / poes_mininsp`;
  breaths outside `mean ± outlierrmssdlimit·SD` of the other breaths have their
  `rms_max`/`rms_mean` replaced by the others' mean.

### 5.11 Optional pressure channels (v2-only)
`calculateaveragebreaths()` and `calculatemechanics()` read
`caps = getattr(settings, "capabilities", Capabilities.FULL)` (the same defensive
idiom §5.1's `boundarynotice_*` settings already use, since a hand-built
`SimpleNamespace` settings object reaches the segmenterers from several call sites)
and guard every value that reads Poes, Pgas or Pdi behind `caps.poes`/`caps.pgas`/
`caps.pdi` — legacy `master` always required all three; v2 does not, once a
`core.analysis.signals.Capabilities` with one or more of them `False` reaches
compute (`core/_legacy_ns.py::to_legacy_ns` passes `Capabilities.from_settings(s)`
through as `capabilities=`, same precedent as `processing.emg.robust_peak`: kept as
one dataclass, not exploded into the legacy attribute shape). On the full-channel
path every guard is `True` and executes exactly the statements that ran
unconditionally before this — golden output is untouched (`tests/golden` 5/5
byte-identical).

- **`calculateaveragebreaths`**: resamples Poes only `if caps.poes`, returning
  `(avgpoesin, avgpoesex) = (None, None)` otherwise. Volume averaging is never
  guarded here (always computed, whatever the signal set) — volume itself becoming
  optional is a later ticket's scope.
- **`calculatemechanics`**: `eilv`/`eelv`/`eilvavg`/`eelvavg` keep `[volume, NaN]`
  when `caps.poes` is `False`, instead of indexing an empty `poes`/`poesavg` array.
  `retbreath["wob"]` is set only `if caps.poes` (`calculatewob` needs Poes). Every
  Poes/Pgas/Pdi-derived quantity (the pressure descriptors of §5.5, the PTP triad of
  §5.6, `vmr` — needs BOTH `caps.poes` and `caps.pgas` — and `tlr_insp`, which needs
  only `caps.poes`) is computed into a local `values` dict only when its
  capabilities are present; the flow/volume-only timing group of §5.4 is never
  guarded (always present once a recording has been segmented into breaths at all).
  The final `retbreath["mechanics"]` `OrderedDict` is built by walking
  `core.analysis.registry.LEGACY_MECHANICS_ORDER` (the same literal, pinned key
  order as before — `tests/unit/test_analysis_registry.py`) and keeping only the
  names present in `values`, instead of a hardcoded 42-entry literal.
- **Golden-locked timing-group quirks, documented, not fixed:** `in_flow_midvol ==
  flow_midvolinsp` and `flow_midvolexp == -ex_flow_midvol` (the legacy names do not
  describe what they sound like — both survive unchanged on every signal set,
  including flow-only).
- **`core/results.py`'s outlier guard** (`'poes_mininsp' in mechs.columns`, §5.10's
  RMS outlier handling, owned by an earlier ticket) already reads correctly once
  `poes_mininsp` is genuinely absent from a flow-only breath's mechanics — pinned at
  the compute level by `tests/unit/test_flow_only.py::test_outlier_guard_unchanged`.
  More generally, `core/pipeline.py`/`core/results.py` are written generically over
  whatever columns the mechanics table ends up containing, rather than hardcoding
  the pressure-channel columns — so `run_batch()` already succeeds end to end for a
  flow-only or poes-only `Settings` object with no further change (verified: before
  this change the same `run_batch()` call raised inside `calculateaveragebreaths`;
  after it, `result.ok_files` contains the file with exactly the expected reduced
  column set). `tests/unit/test_flow_only.py::test_flow_only_reaches_run_batch_end_to_end`
  pins this; `results.py::build_processed_data` was already generic too (its own
  `if len(breath[key]) == 0: continue` per-channel guard, from an earlier ticket),
  and entropy is a signal-set-independent capability (R8) that was never gated on
  Poes/Pgas/Pdi in the first place — both pinned end to end by
  `test_processed_csv_has_only_present_channels`/`test_entropy_columns_equal_full_channel_run`
  in `tests/unit/test_flow_only.py`/`test_poes_only.py`.
- **The "None means absent" contract is now explicit, not just harmless.** An absent
  pressure channel travels through `run_batch`'s `BatchResult.signals` dict as
  `None` (`raw_poes`/`raw_pgas`/`raw_pdi`, guarded on `s.capabilities.poes`/`pgas`/
  `pdi`) rather than an empty array, and `ui/workers.py::stage_mechanics_preview`'s
  `series` dict *omits* the `poes`/`pgas`/`pdi` key entirely instead of carrying an
  empty one — on both the normal path and the `TrimError` fallback. Every existing
  (full-channel) consumer is unaffected, since `capabilities.poes`/`pgas`/`pdi` are
  always `True` there.
  **A reduced signal set is already reachable today**, though — an earlier ticket's
  channel-assignment dialog gates its OK button on the analysis's own *declared*
  roles (`ui/channel_setup_dialog.py::_enabled_analyses`/`_refresh_info`), not a
  hardcoded full set, so a Flow-only or Flow+Poes mapping can already be saved and
  reach Preview & QC today, well before M-16/M-17's own UI work lands. Self-review
  caught the consequence before this ticket closed: `ui/screens/preview/_mechanics.py`'s
  channel-stack render loop indexed the now-narrower `series` dict unconditionally
  over its hardcoded 5-row `_CHANNELS` list, so it would raise `KeyError` on exactly
  that already-reachable configuration — a real regression, not a theoretical one,
  reproduced and fixed in the same commit: the row list is now filtered to
  `[c for c in _CHANNELS if c[0] in series]` before the loop, so it only ever
  iterates keys the dict actually carries. `_update_mech_stack_floor` still sizes
  the stack for a flat 5 rows regardless of how many are drawn — a cosmetic gap on a
  reduced set, left for M-17's own relevance-driven layout, not a correctness issue.
  `core/pipeline.py::segment_file`'s raw time axis (`timecolraw`) is built from the
  first non-empty of flow/volume/poes/pgas/pdi/emg (`_first_present_length()`)
  rather than assuming flow specifically — a no-op for today's E2 scope (flow is
  always present, so always first and chosen) but forward-compatible groundwork for
  M-21's flow-less EMG-only segmentation.
- **Tests**: `tests/unit/test_flow_only.py` (flow only: no Poes/Pgas/Pdi) and
  `tests/unit/test_poes_only.py` (flow + Poes, no Pgas/Pdi) call
  `calculateaveragebreaths`/`calculatemechanics` directly on segments built from
  `synth_case_A.csv` (`tests/unit/_helpers.py::segment_synth_case`/
  `compute_all_breaths`), bypassing `core.pipeline.run_batch` for a narrower,
  faster unit of test — not because `run_batch` itself fails (see above). Both
  files also carry real `run_batch()` and `stage_mechanics_preview()` end-to-end
  tests for the processed-CSV, entropy and preview-series claims above, plus a
  `qapp`-backed `MainWindow`/`PreviewScreen` regression test each
  (`test_mechanics_stack_renders_without_crashing_on_a_reduced_signal_set`) that
  renders the real Mechanics channel stack and pins the `_CHANNELS`-filtering fix
  above — confirmed to fail with the exact `KeyError` before that fix and pass
  after it.

### 5.12 EMG-only segmentation (v2-only) — `core/analysis/segments.py`

A recording with no flow channel at all (`analysis.signals = ["emg"]`) has no
inspiration/expiration split to compute — `§5.3`'s flow/volume segmenters both need a
flow or volume signal to find breath boundaries on. Two segmentation methods split such
a recording into **segments** instead, each carrying `has_phases=False` and the same
`OrderedDict` shape a real breath does (`flow`/`volume`/`poes`/`pgas`/`pdi` all empty —
the same absence convention every other optional channel already uses):

- **`whole_file`** — the entire recording is one segment, always numbered 1.
- **`separators`** — N user-placed times (`processing.segmentation.separators`, one
  `SeparatorEntry` per file) split the recording into N+1 segments numbered from 1. A
  time outside the recording, or two times close enough to round to the same sample,
  raises `EmgSegmentationError` naming the file — a per-file `FileResult` error, never a
  batch-stopping crash.

`compute.separateintobreaths` dispatches to these two (via `core.analysis.segments`)
before falling through to the flow/volume segmenters, so the dispatch point stays
single. `core.pipeline.segment_file`/`run_batch` skip flow-only steps for such a
recording (`Capabilities.mode == "emg_only"`): no `trim()` (there is no flow-based
zero-crossing to trim to — the whole raw recording IS the analysis window), no volume
zero/drift/trend correction, no `calculateaveragebreaths` (nothing to average across
phases), and `vefactor`/`bcnt` (which would divide by an empty flow's length) are never
computed. `compute.compute_segment_emg` — RMS/integral-EMG/gated-peak EMG and sample
entropy, a verbatim extraction of what `calculatemechanics` computes inline — is called
with `phases=False`, which computes the whole-segment values only (never the
inspiration/expiration-specific ones, which do not exist for a phase-less segment); a
phase-less segment's `mechanics` dict is built directly as `{seg_start_s, seg_end_s,
seg_duration_s}` rather than the flow-derived timing group `§5.4`'s
`LEGACY_MECHANICS_ORDER` computes.

`whole_file` additionally reports, per EMG channel, where in the recording the peak RMS
fell (`t_rms_file_max_col_N`, from the same sliding-window RMS grid `calculate_rms`
itself maximises over — `emg.rolling_rms`) and the mean of the three highest values in
that envelope (`rms_file_top3_col_N`), a steadier "peak level" than the single max alone
against a lone noise spike — nan-aware throughout (a NaN sample, e.g. from upstream
noise reduction, poisons every rolling-RMS window from that sample onward, since it sits
in a cumulative sum; a real earlier peak is still recovered, and a channel with nothing
left to recover from reports NaN for the value AND its timestamp together, never a
concrete-looking time paired with a NaN value). Whether that peak coincides with a
heartbeat is not itself checked here — this diagnostic only locates the peak; deciding
whether it is real muscle activity or cardiac contamination is what
`processing.emg.robust_peak` is for.

**The loader's flow requirement is conditional on this shape**, not lifted generally:
`core/io/loaders.py` still raises immediately on an unassigned flow whenever volume,
poes, pgas or pdi is STILL assigned (a real, previously-caught misconfiguration —
`tests/unit/test_unassigned_channels.py::test_an_unassigned_channel_is_named` pins
this unchanged); flow is treated as absent, exactly like its four siblings, only when
ALL FOUR of them are absent too — the one shape `Settings.validate()` itself allows an
unassigned flow in (poes/pgas/pdi declared without flow is rejected there).

Golden-neutral: the EMG-only branch is taken only when `Capabilities.mode ==
"emg_only"`, which no golden scenario declares — every existing (flow-bearing) run
takes the unchanged path byte-for-byte.

An automatic alternative to manual separators (fixed windows, burst detection) and a
resolved single noise-reference rule for an EMG-only set (`reference_mode`) are later
tickets' scope, not this one's.

### 5.13 Manoeuvre extraction (v2-only) — `core/analysis/manoeuvres.py`

A single breath can be TYPED (`processing.breath_types`, one `BreathTypeEntry` per
breath — `ic`/`fvc`/`ic_fvc`/`max_insp`/`sniff`/`rest`/`other`) as a named manoeuvre
rather than tidal breathing. On a flow-bearing signal set every typed kind is unioned
into `excludebreaths` (`§5.3`'s exclusion mechanism, extended by M-19), so a typed
breath **never reaches `calculatemechanics()`** — the ordinary mechanics loop's own
`if breath["ignored"]: continue` skips it exactly like a manually excluded breath.
`manoeuvres.extract(breath, kind, tidal_breaths, caps, s)` is called separately, once
per typed breath, straight from its raw breath dict (never from `calculatemechanics`'
output):

- **`ic`/`ic_fvc`** (an inspiratory-capacity manoeuvre; `ic_fvc` also carries a forced
  expiration on the SAME breath): `vol_ic = max(breath['volume']) − ic_eelv_pre`.
  `ic_eelv_pre` is the mean end-expiratory volume of up to
  `IcSettings.preceding_breaths` TIDAL breaths immediately preceding this one (by
  breath NUMBER, not list order), falling back to this breath's own
  `inspiration['volume'][0]` (`ic_eelv_pre_n=1`, `ic_eelv_pre_sd=0.0`) when fewer than
  `min_preceding_breaths` tidal breaths precede it. `ic_ti` = inspiration sample count
  / fs (the same "a sample COUNT, not `time[-1] - time[0]`" convention `§5.12`'s
  `seg_duration_s` uses). `ic_peak_in_flow = −min(inspiration['flow'])`.
  `ic_plateau_s` counts back from the very end of the inspiration while `|flow| <
  plateau_flow_lps`, uninterrupted — how long the manoeuvre was held at its inspiratory
  peak. `poes_ic_min`/`poes_ic_eelv`/`poes_ic_swing`/`poes_ic_peakvol`,
  `pdi_ic_max`/`pdi_ic_swing`, `pgas_ic_peakvol` mirror the volume fields for each
  pressure channel present (`peakvol` = the channel's value at the SAME sample index
  `max(breath['volume'])` was found at).
- **Quality flags** (`quality`, a list, joined with `", "` for the sheet):
  `EELV_UNSTABLE` (the preceding-breaths' EELV standard deviation, scaled by the
  MANOEUVRE'S OWN `vol_ic` — never by `ic_eelv_pre` itself: this codebase
  zero-references and drift-corrects volume by default, so a real `ic_eelv_pre` sits
  within a few mL of 0 L, and a relative tolerance measured against it would explode
  on ordinary breath-to-breath noise — exceeds `eelv_tolerance_frac`, only evaluable
  with ≥ 2 preceding breaths), `LOW_EFFORT` (this manoeuvre's own peak inspiratory
  flow, or Poes swing, is below `low_effort_frac` × the FILE's OWN tidal median —
  computed from raw arrays over the file's own non-ignored breaths, never the
  mechanics table, so a subset run and a full batch run agree), `NO_PLATEAU`
  (`ic_plateau_s < min_plateau_s`), `BOUNDARY` (this breath is the first- or
  last-numbered breath the file has), and — for `fvc`/`ic_fvc` — `FVC_TOO_SHORT`
  (`validate_fvc_manoeuvre`: the expiratory limb's own duration is under 1.0 s).
  `NOT_REPEATABLE` is NOT set by `extract()` itself (a single breath cannot see its
  file's OTHER typed breaths); `apply_repeatability(manoeuvres, ic_cfg)` runs once
  per file AFTER every typed breath's own `extract()` result is in hand, comparing
  each IC/IC+FVC breath's `vol_ic` against the mean (or median, `aggregate=`) of the
  file's OTHER eligible ICs — LEAVE-ONE-OUT, never including the row's own value in
  what it is compared against, or the effective tolerance for exactly two attempts
  would silently halve and a single bad attempt could drag a genuinely agreeing pair
  into looking unrepeatable too. One already carrying a `reject_flags` flag
  (`LOW_EFFORT` by default) is excluded from that comparison group entirely (on
  either side), and a lone IC with no sibling is never flagged (repeatability is a
  property of a pair). `aggregate='mean'` (the default) is still outlier-sensitive
  once there are ≥ 4 total eligible attempts — `aggregate='median'` is the
  outlier-robust choice for a study expecting more than three repeats.
- **`max_insp`/`sniff`** (a maximal-effort reference breath):
  `max_effort_from_breath` reports `poes_max_ref = inspiration['poes'][0] −
  min(inspiration['poes'])`, `pdi_max_ref = max(inspiration['pdi']) −
  inspiration['pdi'][0]` (both SWINGS from the breath's own immediate
  pre-inspiratory baseline — the same convention `poes_ic_swing`/`pdi_ic_swing`
  use above, not the raw absolute pressure, so a later normalisation ratio (M-47)
  never divides a baseline-subtracted swing by a channel's absolute resting offset),
  and `rms_max_ref` — the peak rolling-RMS envelope (`emg.rolling_rms`, the same grid
  `§5.12`'s whole-file diagnostics use) across every EMG channel in the breath,
  computed here rather than read off the breath dict because a typed breath's
  `compute_segment_emg` is never called (it is `ignored=True`). No `quality` flags —
  the IC-specific acceptance checks do not apply to a maximal-effort breath.
- **`fvc`** (pure, not combined with an IC): `validate_fvc_manoeuvre`'s
  `FVC_TOO_SHORT` flag only — the actual FVC/FEV1/PEF/flow-volume-curve numerics are
  `core/analysis/mfvl.py`'s scope (a later ticket), not this module's.
  `suggest_fvc(breaths)` is a separate, pure UI-facing heuristic (not called from
  `extract()` or the pipeline): the untyped, non-ignored breath with the longest
  expiration in the file — a deterministic hint, never an automatic choice.
- **`other`**: `{'kind': 'other', 'quality': []}` — recorded so the breath is visible
  in the Manoeuvres sheet, no numeric semantics defined for it. **`rest`** is never
  passed to `extract()` at all — the pipeline skips it (a noise-reference segment
  label, not a manoeuvre — see M-22/`§5.12`'s `rest_segments` reference mode).

`core.pipeline.run_batch` builds `FileResult.manoeuvres` (`{breath_no: extract(...)
result}`) and `.manoeuvres_table` (`core.results.build_manoeuvre_table`, `None` when
empty) right after the main mechanics loop, flow-bearing runs only (an EMG-only
segment has no `inspiration`/`expiration` for `extract` to read).
`core.io.writers.write_batch` adds it as an extra "Manoeuvres" sheet on the per-file
breathdata workbook, exactly like the existing "EMG normalised" sheet — present only
when at least one typed breath exists in that file.

Every numeric threshold in `IcSettings` (`eelv_tolerance_frac`, `plateau_flow_lps`,
`min_plateau_s`, `repeatability_frac`, `low_effort_frac`) is a documented PLACEHOLDER,
not a value measured against a real IC recording (no production recording exists in
the sandbox that implemented this) — see `docs/beslutninger.md` for what still needs
measuring before these are trusted clinically. The FORMULAS themselves are pinned by
analytical/synthetic tests (`tests/unit/test_manoeuvres.py`), independent of the exact
cut-offs.

### 5.13a Cross-file reference resolution (v2-only) — `run_batch`'s forepass/afterpass

`§7b`'s `processing.references`/`reference_defaults` tables only DECLARE where a
file's reference values come from; `core.analysis.references.attach` and a small
forepass in `core.pipeline.run_batch` are what actually resolve them into per-breath
columns, in two passes around the ordinary main per-file loop:

- **Forepass** (before the main loop): `core.analysis.references.
  external_reference_sources(settings, files)` lists every reference SOURCE filename
  `processing.references`/`reference_defaults` name that is not already in `files`
  (this run's own file list — `only_files`-restricted or not). Each one is loaded and
  segmented through the SAME `segment_file()` entry point the main loop itself uses,
  and every TYPED breath in it is run through `manoeuvres.extract()` exactly like an
  in-batch reference-only file (`§5.13`, M-30) already is — a source loaded here and
  one loaded as an ordinary in-batch file give byte-identical results for the same
  breath. Results land in `BatchResult.references`
  (`{source_filename: {breath_no: extract(...) result}}`); a source that fails to
  load/segment at all is recorded in `BatchResult.reference_errors` keyed
  `(filename, None)` and simply contributes nothing — the batch is never aborted over
  one failed reference source. This is why a SUBSET run (`only_files` excluding an
  in-batch reference source) still resolves the same reference value a full run would:
  the source falls outside `files`, so the forepass fetches it regardless.
- **Afterpass** (`references.attach(result, settings, allfiles)`, after every file's
  own manoeuvres/breath table are final but BEFORE `average_table` is concatenated
  from the individual `average_row`s): for every OK tidal file, resolves its `ic` slot
  (`resolve_reference`'s order) and looks the linked breaths up in EITHER an in-batch
  `FileResult.manoeuvres` or the forepass's `BatchResult.references` (whichever has
  them). The resolved value is the `ic_cfg.aggregate` ('mean'/'median') of `vol_ic`
  over the breaths that resolve AND are not flagged with one of
  `ic_cfg.reject_flags` (`LOW_EFFORT` by default — the same disqualifying rule
  `apply_repeatability`'s own leave-one-out group already uses). Three columns are
  APPENDED (never inserted before an existing one) to both `breaths_table` and
  `average_row`: `vol_ic_ref` (L), `ic_ref_n` (the accepted-breath count), and
  `ic_ref_source` (the resolved source filename, a text column).
- **Column family rule**: whether these three columns exist AT ALL for a given
  analysis is decided from SETTINGS across `allfiles` (the full matched set, never
  just this run's own subset) — `any(resolve_reference(name, "ic", settings) for name
  in allfiles)`. A subset run therefore writes the exact same column SET a full run
  would. A file with the family present but no resolution of its own gets the three
  columns as NaN plus a notice — "an unresolved link is a caution plus NaN and a
  notice" (`§7b`'s policy) applies to the whole family, not only to an explicitly
  configured link.
- **`processing.lung_volume.require_references`**: off by default, an unresolved `ic`
  reference is soft (NaN + notice, the file stays OK). When set, it escapes as
  `core.analysis.references.ReferenceLinkError` and DEMOTES just that one file to
  failed (`FileResult.error`/`error_kind` set on the same object already in
  `result.files` — no new object constructed, avoiding a pipeline/references import
  cycle) — the same policy `ui.validation.path_problem` already applies at
  validation time to a source file missing from the matched set, now also covering a
  runtime-only failure (a matched source that fails to load, say) `Settings.validate()`
  cannot see ahead of time. `ReferenceLinkError` is registered in
  `ui.screens.preview._mechanics._SOFT_FILE_ERRORS` and
  `ui.screens.run_screen._FIX_HINTS` alongside `TrimError`/`VolumeTrendError`/
  `NoBreathsError`/`EmgSegmentationError`.
- **Reporting**: `core.io.writers._write_run_report`'s PROCESSING block gets a
  "Reference manoeuvres:" line listing what `processing.references`/
  `reference_defaults` CONFIGURE (a study-wide setting, like "Breath types:"
  above it); a conditional "REFERENCE MANOEUVRES" block (DIAGNOSTICS' own
  convention) reports what actually RESOLVED this run — external sources loaded,
  per-file resolutions, unresolved files, and forepass errors. Each file's own
  Provenance sheet gets an "IC reference" row when its own reference resolved
  (`FileResult.references_used['ic']`, via `_ic_reference_provenance_value`).
  `fvc`/`max_insp` are extracted by the SAME forepass (any typed breath in a source
  file, of any kind) but have no consuming column of their own yet — that is
  `mfvl.py` (M-42) and the normalisation ticket (M-47)'s scope. `baseline_ic` gained
  its first consumer in §5.14 below (`delta_ic`) — it is aggregated the same way an
  `ic` reference is, on demand, rather than through this forepass/afterpass pair
  (see §5.14's own note on why).

---

### 5.14 Operating lung volumes (v2-only) — `core/analysis/lungvol.py`

`§5.13a`'s `references.attach` resolves a file's `vol_ic_ref` (the IC reference
VALUE) but stops there — it does not turn that single number into what a tidal
breath's own lung volumes actually are. `core.analysis.lungvol.attach`, wired into
`core.pipeline.run_batch` immediately AFTER `references.attach` (M-36), does that:
per-tidal-breath EELV/EILV/IRV (and their %VC/%TLC forms) plus four per-file scalars
(`tlc`, `vc`, `delta_ic`, `delta_eelv`, `delta_ic_pct`).

**Column family**: the SAME `resolve_reference(..., "ic", ...)` check
`references.attach` already uses across `allfiles` — operating lung volumes are
meaningless without a resolvable IC reference, so this ticket's family is exactly
that one. Once present, EVERY olv column is added to EVERY OK tidal file, never
conditionally per file (a file with no subject VC/TLC simply gets NaN in the
VC-/TLC-anchored triples) — this is what keeps a subset run's column SET identical
to a full run's even when only some files' groups have a VC/TLC entered.

**`ic_op` (the reference IC actually operating for this breath)**:
`processing.lung_volume.ic.eelv_tracking` (declared and validated by M-29's
`IcSettings`, unused until this ticket) decides how it moves:

- `"none"` (default): `ic_op = vol_ic_ref`, unchanged across the whole file — the
  ordinary "IC measured once, assumed constant" convention. No `d_eelv` column at
  all in this mode (a settings-uniform family decision, simpler than M-35's own
  multi-file family rule, since `eelv_tracking` is one flag for the whole analysis).
- `"within_file"`: `d_eelv = vol_endexp - ic_eelv_pre` (positive = EELV has RISEN
  since the reference IC's own end-expiratory level, i.e. hyperinflation), then
  `ic_op = vol_ic_ref - d_eelv` — a RISE in EELV SHRINKS the IC actually available
  (the minus sign is load-bearing: an earlier draft of this ticket had it backwards,
  caught by the analytical test that pins the direction). `ic_eelv_pre` is the SAME
  `ic_cfg.aggregate` (mean/median) of the resolved IC breaths' own `ic_eelv_pre`
  field the reference itself was aggregated from — but ONLY for a SAME-FILE IC
  reference (`FileResult.references_used['ic']['source'] == filename`); a
  CROSS-file reference NaNs `d_eelv`/`ic_op` with one per-file notice instead
  (two different recordings rarely share a common volume zero, so a cross-file
  end-expiratory comparison is not meaningful) — and this notice fires ONLY when
  the reference genuinely resolved to a different file (`references_used['ic']` is
  set, `source != filename`); when it never resolved at all, `references.attach`
  has already said so, and this module adds nothing further (self-review finding:
  an earlier draft conflated the two and reported "cross-file" even for a file
  whose reference simply never resolved). `processing.volume.correct_trend`
  being on ALSO forces NaN even for a same-file reference: the trend-correction pass
  subtracts the trough envelope so every `vol_endexp` sample sits at the same level
  by construction, which would otherwise report a flat, definite zero for a
  quantity the filter has actively erased.

**Two EELV data, reported side by side (Emil's decision 26-09-2026, see
`docs/beslutninger.md`)**: `vol_eelv = vc_src - ic_op` (volume above residual
volume at end-expiration — ERV by definition, no separate `vol_erv` column) is the
PRIMARY family, because it needs only a spirometry-derived VC, not a measured TLC
(`vc_src` = `input.subjects`' own `vc_l` for this file's group, else the linked
`fvc` reference's own `fvc` field — always `None` today, since
`manoeuvres.extract` computes no numeric FVC value until M-42's `mfvl.py` lands;
the fallback is already wired to read whichever key is there, so it starts working
unchanged the moment M-42 adds it). `vol_eelv_abs = tlc - ic_op` is the absolute,
TLC-anchored value reported ALONGSIDE it, present only when `input.subjects` names a
TLC for this file's group. `vol_eilv`/`vol_eilv_abs` add `vt`; `vol_irv = ic_op -
vt`; every `_pct_vc`/`_pct_tlc` column divides by `vc`/`tlc`. Missing VC/TLC is the
ordinary case (most studies measure neither) and NaNs the corresponding family
silently — never a notice.

`vol_eelv` is the OPERATING (per-breath, model-derived) ERV for this breath, not a
single spirometrically-measured ERV value — under `eelv_tracking="within_file"` it
varies breath to breath by design (that is the point: tracking dynamic
hyperinflation), so a reader expecting the one stable clinical ERV number should not
mistake this column for that. The algebra (`vc - ic_op == eelv - rv` when
`vc == tlc - rv` for that subject) only holds when the subject's entered
`tlc_l`/`vc_l`/`rv_l` are themselves mutually consistent — `Settings.validate()`
only checks `rv_l < tlc_l`, nothing cross-checks `vc_l` against `tlc_l - rv_l`, and
`rv_l` is not otherwise consumed anywhere in this codebase yet; an inconsistent
subject entry produces a silently wrong split between the RV- and TLC-anchored
families with no notice (self-review finding, not fixed by this ticket — flagged in
`docs/beslutninger.md`'s 26-09-2026 entry as a known gap). Separately, once
`vc_src` can fall back to a linked FVC manoeuvre (M-42), note that a *forced*
vital capacity under-reads true (slow) VC in obstructive disease (gas trapping) —
that fallback will systematically UNDERESTIMATE `vol_eelv` in exactly the
population dynamic-hyperinflation tracking is most useful for, once it is wired in.

**`tlc`/`vc`/`delta_ic`/`delta_eelv`/`delta_ic_pct`** are per-FILE scalars (constant
across every row of `breaths_table`, and the value in `average_row`): `tlc`/`vc` as
above; `delta_ic = vol_ic_ref - vol_ic_ref(baseline)`, `delta_eelv = -delta_ic`,
`delta_ic_pct = 100 · delta_ic / vol_ic_ref(baseline)`. The baseline's own
`vol_ic_ref` is resolved two ways: an explicit/group `baseline_ic` link names
specific breaths in some file, aggregated exactly like an `ic` reference
(`ic_cfg.aggregate` of `vol_ic` over accepted breaths — a baseline IS itself just
another IC measurement, hence the same rule); with no such link, a filename
matching `lung_volume.baseline_pattern` is searched for among this file's OWN
`group_key` siblings in `allfiles` (the full matched set, never just this run's own
subset — the same "decide from settings across the full set" rule the column
family above already follows, so a subset run picks the SAME baseline candidate a
full run would), and the alphabetically first match's ALREADY-RESOLVED
`vol_ic_ref` scalar is looked up from `result.ok_files` (deliberately simpler than
re-aggregating specific breaths: a whole file matched by name is naturally "this
file's own IC reference is the baseline", not a hand-picked subset) — every ok
tidal file has `vol_ic_ref` on its `average_row` by the time this module runs,
regardless of processing order, so looking up ANY other file's value is always
safe; if the matched candidate happens to fall outside THIS run's own subset,
`delta_ic` is honestly NaN for this run (the DECISION of which file is the
baseline stays consistent, only the VALUE'S availability depends on what this run
actually processed). More than one sibling matching the pattern is reported with
its own notice (never silent) even though a match is still chosen deterministically.
`delta_ic` is silently NaN when no baseline mechanism resolves at all — a notice
fires only when `baseline_ic` was EXPLICITLY configured but could not be resolved
(the "unresolved link is a caution plus NaN and a notice" policy, `§7b`); an
ABSENT baseline configuration (the common case) is not a caution at all.

**Notices** (once per file, never per breath): a within-file cross-file/trend-
correction NaN (above, only when the reference genuinely resolved elsewhere);
an ambiguous `baseline_pattern` match; `vol_irv < 0`; `vol_eelv < 0` (the operating
IC exceeds VC); `vol_eelv_abs < 0` (the operating IC exceeds TLC) — the last three
are checks for a physiologically implausible operating volume, independent of
whether VC/TLC itself was even available. **Per-file isolation**: an unanticipated
exception while computing one file's operating lung volumes is caught and turned
into a notice on that file alone (the same per-file isolation `core.pipeline.
run_batch`'s own main loop already gives every other failure mode) — it never
aborts the whole batch. The family decision and which files actually got the
column family are recorded once, in `BatchResult.analysis_plan['lung_volume']`
(`{'family': bool, 'active_files': [...]}`), and `core.io.writers` reads that back
rather than re-deriving "is this active" by inspecting DataFrame columns at each
call site (the same "record once in `attach()`, read it back" convention
`analysis_plan['ic']` already established for §5.13a's REFERENCE MANOEUVRES).

**Reporting**: `run-report.txt` gets a "LUNG VOLUMES" block (present only when at
least one OK tidal file has the `ic_op` column family) naming the study-wide
`eelv_tracking`/EELV-datum choice and, per file, `ic_op`/TLC/VC; each file's own
Provenance sheet gets "EELV tracking"/"EELV datum" rows under the same condition.

**Units** (`core/quantities.py`/`core/analysis/registry.py`): `d_eelv`, `ic_op`,
`delta_ic`, `delta_eelv`, `tlc`, `vc` do not match the generic `vol_`/`vt` naming
convention, so their `unit="L"` is registered explicitly; every other new column
(`vol_irv`, `vol_eelv`, `vol_eilv`, `vol_eelv_abs`, `vol_eilv_abs` via the `vol_`
prefix; every `_pct`/`_pct_` column via the generic suffix rule) already resolves
without a registry entry.

---

## 6. Latent issues found (to fix deliberately in the refactor)

Confirmed by running the current code (see [`tests/golden/README.md`](../tests/golden/README.md)):

1. **EMG overview plot crashes on an ignored breath** — `emg.saveemgplots()` builds
   the ignored-breath `Rectangle` width as a 1-element array → matplotlib
   "inhomogeneous shape" error. The (always-on) EMG overview plot cannot render
   when any breath is excluded.
2. **Entropy on untrimmed / overlapping columns** — `analyse()` never trims
   `entropycolumns` yet indexes them with trimmed breath coordinates; it also
   overwrites entropy columns that coincide with EMG columns, causing a shape
   mismatch. Overlapping EMG/entropy channels (as in `example.py`!) crash or
   silently misalign.
3. **Processed-data export assumes exactly 5 EMG channels** — `getprocesseddata()`
   hardcodes `["EMG1".."EMG5"]`; any other channel count raises a shape error.
4. **SciPy API breakage** — `cumtrapz` removed / `simpson` positional-`x`; the code
   will not run on current SciPy (§4).
5. **Settings drift** — `example.py` sets `calcwobfromaverage` and `volumetrendpeakminwidth`
   under `mechanics`, but the code reads `wob.calcwobfrom` and never reads
   `peakminwidth`; those example keys are silently ignored (§7).
6. **`applysettings` crashes on a new nested subsection** — the merge
   (`applysubsettings`) only handles new *leaf* keys; a settings file containing a
   `processing.sampling` subsection absent from `defaultsettings` raises
   `KeyError: 'sampling'`. Real production settings (targeting the unmerged
   resampling branch) hit this.

### 6a. Confirmed against real production data

Running the current code over Emil's real validation datasets (see
[`tests/golden/production_comparison.md`](../tests/golden/production_comparison.md))
confirmed the above on real recordings and added two version-difference findings —
important for "fully correct calculations":

- **PTP zeroing is version-dependent.** Commit `1630c40` "Fixed PTP calculations
  (zeroing)" added `pressure = pressure.squeeze() - pressure[0]` to `calcptp`. This
  changes all `int_*` / `ptp_*` outputs by a large factor (verified: 13.06 → 2.71
  on one breath). Expected spreadsheets generated before that commit differ from
  the current code accordingly. **Resolved in v2** (see §5.6 and
  `PTP_INVESTIGATION.md`): one end-expiratory baseline, the mean over a 50 ms
  window. There is no double subtraction: ∫(f − f[0]) is invariant to the constant
  pre-shift `adjustforintegration` applied, so only one baseline was ever in effect.
- **EMG/ECG-removal is version-dependent.** The current `master` EMG conditioning
  (ECG removal + spectral noise reduction) differs from the "new ECG removal"
  version Emil validated with; EMG RMS / integrated-EMG columns differ by up to
  ~90 % while mechanics/WOB are unaffected. Latent bug #1 additionally **crashes**
  real files that combine excluded breaths with EMG (`RIU_H5_IC`, `RIU_H6_IC`,
  `RIU_H6_Baseline`). **Partially addressed in v2** (see §5.10): the spectral
  gate's reconstruction had a real bug that inflated noise-reduced EMG power by
  ~30–40 % regardless of `prop_decrease`, fixed 05-09-2026, and the ECG
  R-peak detector now median-corrects before peak-finding and accepts
  negative-polarity R-peaks. Neither fix reconciles this section's original
  comparison against the separate "new ECG removal" variant Emil validated
  with, which predates v2 and is not otherwise documented here.

---

## 7. Settings structure (current)

Settings are a **nested Python dict** merged over an embedded JSON default
(`defaultsettings` in `respmech.py`) via `applysettings()` (JSON → `SimpleNamespace`,
recursive override). The effective shape:

```
input.inputfolder / input.files
input.format.{samplingfrequency, matlabfileformat, decimalcharacter}
input.data.{column_poes, column_pgas, column_pdi, column_volume, column_flow,
            columns_entropy[], columns_emg[]}
processing.mechanics.{breathseparationbuffer, separateby*, peakheight, peakdistance,
            peakwidth, inverseflow, integratevolumefromflow, inversevolume,
            correctvolumedrift, correctvolumetrend, volumetrendadjustmethod,
            volumetrendpeakminheight, volumetrendpeakmindistance,
            excludebreaths[], breathcounts[]}
processing.wob.{calcwobfrom("average"|"individual"), avgresamplingobs}
processing.emg.{rms_s, remove_ecg, column_detect, minheight, mindistance, minwidth,
            windowsize, outlierrmssdlimit, remove_noise, noise_profile[],
            save_sound, emgplotyscale[]}
processing.entropy.{entropy_epochs, entropy_tolerance}
output.outputfolder
output.data.{saveaveragedata, savebreathbybreathdata, saveprocesseddata,
            includeignoredbreaths}
output.diagnostics.{savepvaverage, savepvoverview, savepvindividualworkload,
            pvcolumns, pvrows, savedataviewraw, savedataviewtrimmed,
            savedataviewdriftcor}
```

Notable traps for the refactor:
- `separateby` and `column_detect` are **read by the code but absent from
  `defaultsettings`** → a settings file omitting them can `AttributeError`.
- The **1-based** column numbering is a common user error source.
- Paths, exclusions and per-file overrides are all **absolute-path, filename-keyed
  lists** embedded in Python — not portable between machines (see the two different
  usernames baked into `example.py`/`README.md`).
- The **README documents a *flat* settings dict** that no longer matches the
  nested structure the code expects — stale.

A field-by-field semantics table and the old→new migration mapping live in
[`PLAN.md`](PLAN.md) §_Settings redesign_.

### 7a. `analysis.signals` (v2 addition)

The v2 model (`core/settings.py`) adds one field this legacy shape had no equivalent
for: `Settings.analysis.signals`, a list naming which of `flow`/`poes`/`pgas`/`pdi`/
`emg` an analysis actually uses. Left empty (the default — and the shape every
migrated v1 analysis takes, since v1 always required all four pressure/flow roles),
the effective set is *derived* instead: whichever of those roles has a channel
assigned (`core/analysis/signals.py::derived_signals`/`effective_signals`). An
explicit, non-empty list overrides that derivation outright — a `Settings.validate()`
requirement gone as of this ticket: Poes/Pgas/Pdi may all be absent, as long as Flow or
EMG is present (a lone pressure channel with no Flow is still rejected, since breath
segmentation needs Flow either way). A channel assigned without being named in an
explicit list is reconciled upward on load (`Settings.from_dict`, never any other write
path), with a `Settings.notices` entry recording it. Saving an analysis omits the whole
`[analysis]` table while the explicit set still matches the derived one; the run
manifest (`analysis-used.toml`, via `dumps_toml`) always records the resolved,
effective set instead, explicit or not.

### 7b. Reference manoeuvres and subject lung volumes

Three new tables let one file's inspiratory-capacity/forced-vital-capacity/maximal-
effort/baseline reference values come from breaths typed in a DIFFERENT file (or from
the same file), and let per-participant spirometry (TLC/VC/RV/FEV1/MVV) apply across
every file that participant recorded:

```
[[processing.references]]         # per-file: overrides everything else for THIS file
file = "P03_peak.txt"
ic = { file = "P03_IC.txt", breaths = [2, 3, 4] }   # BreathRef: one or more breaths
fvc = { file = "P03_MFVL.txt", breaths = [1] }
baseline_ic = { file = "P03_rest.txt", breaths = [7] }
max_insp = { file = "P03_IC.txt", breaths = [4] }
folder = "recordings"              # carried-over-state provenance tag, as elsewhere

[[processing.reference_defaults]]  # per-group fallback (core.summary.group_key)
group = "P03"
ic = { file = "P03_IC.txt", breaths = [2, 3, 4] }
folder = "recordings"

[[input.subjects]]                 # keyed the same way as reference_defaults' group
key = "P03"
tlc_l = 6.12
vc_l = 4.30
rv_l = 1.90
fev1_l = 3.10                      # spirometry value, preferred over a derived one
mvv_lpm = 124.0
folder = "recordings"

[processing.lung_volume]
require_references = false         # true: an unresolved reference source becomes a
                                    # HARD path_problem() blocker instead of a soft
                                    # check_links() caution
baseline_pattern = "(?i)baseline|rest"   # not yet read by any code path
[processing.lung_volume.ic]
eelv_tracking = "none"             # "none" | "within_file" -- a later ticket's own
                                    # arithmetic; only declared and validated here
```

`ic`/`fvc`/`baseline_ic`/`max_insp` are all `BreathRef | None` — a nested optional
dataclass, which needs the earlier PEP 604 fix to `_unwrap_optional` (§2) to round-trip
through TOML at all.

**Resolution order** (`core.analysis.references.resolve_reference`), applied
independently per slot: an explicit `processing.references` entry for the file beats a
matching `processing.reference_defaults` group entry, which beats the file's OWN typed
breaths of a matching kind (`ic`/`ic_fvc` for the `ic` slot, `fvc`/`ic_fvc` for `fvc`,
`max_insp`/`sniff` for `max_insp`), which beats nothing (`None`). `baseline_ic` has no
own-typed fallback — a baseline is by nature a different recording, never inferred from
the manoeuvre breath itself.

**`check_links`** (same module) is the read-only, no-exception counterpart:
one caution per reference source not among the batch's own matched files
(`core.pipeline.match_input_files`, NOT a manifest's majority-column-count subset —
`ui.manifest.Manifest.included_files`), per linked breath not typed the matching kind,
per linked breath that is excluded, and per `reference_defaults`/`input.subjects` group
key matching no analysed file. One policy throughout: an unresolved link is always a
caution, plus NaN and a run-report notice once a later pipeline pass attaches it —
never a hard error, unless `processing.lung_volume.require_references` is set, in
which case `ui.validation.path_problem` turns a missing SOURCE FILE into a blocker
(the other three caution kinds stay soft cautions even then — only an unresolvable
source file is escalated).

Neither table is read by `core.compute`/`core._legacy_ns` — resolving the actual
values (loading the referenced file, running `core.analysis.manoeuvres.extract` on it,
attaching the result to the referencing file's own columns) is `core.pipeline.run_batch`'s
forepass/afterpass pass, well downstream of the ordinary per-breath mechanics loop this
never touches — see `§5.13a`.

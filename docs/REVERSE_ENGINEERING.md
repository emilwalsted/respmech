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
- **Manual segmentation repair (v2-only)** — `processing.segmentation.overrides`
  (`SegmentationOverrideEntry`, one per file): `cut_s`/`join_s` REPAIR the automatic
  flow-/volume-based boundary list above, they never replace it (contrast the
  EMG-only `separators` method in §5.12, which has no automatic detection to repair
  at all). Applied by `compute.apply_segmentation_overrides`, called from
  `core.pipeline.segment_file` right after the automatic segmentation call above and
  BEFORE `ignorebreaths`/`breathkinds`/numbering/`trim_boundary_notices` — so every
  downstream consumer (exclusions, typed breaths, references, `t_onset_s` anchors,
  boundary notices) sees one finished boundary list, same as the automatic
  segmenters already give their own consumers.
  - `join_s`: removes the nearest AUTOMATIC breath-start boundary to the named time
    (within `max(2/fs, 0.05 s)`), merging the two breaths on either side into one —
    the fix for a flow wobble that over-split one real breath into two.
  - `cut_s`: inserts a new boundary at the named time, splitting whatever breath
    currently spans it — the fix for a flat/leaky expiration that under-split two
    real breaths into one.
  - Each resulting segment gets its own inspiration/expiration split via
    `_walk_insp_end` — the SAME mean-buffered flow-sign criterion `separateintobreathsbyflow`'s
    own inspiration loop uses, bounded to the segment's own `[start, end)` instead of
    the whole recording, so a residual wobble already absorbed by a `join_s` cannot
    re-trigger a second split within that one segment. Everything after the found
    transition is that segment's expiration, however the flow signal behaves later in
    it — by construction a segment produced by the override list is meant to be
    exactly one breath.
  - **One-sample transition drop, reproduced exactly**: `separateintobreathsbyflow`'s
    own `inend = i - 1` / `exend = min(i - 1, j)` drops the sample AT every transition
    it finds (never assigned to either phase) — a real, existing property of this
    ported algorithm. `apply_segmentation_overrides` reproduces this drop at a
    segment's inspiration end ONLY when that end is a genuine transition
    `_walk_insp_end` found, OR the segment's own end is itself a NATURAL boundary
    (an untouched automatic breath start, or the file's own end) reached without
    finding one; a `cut_s`-inserted boundary is not a flow transition, so a segment
    that runs straight into its OWN cut without ever finding a transition gets no
    drop there — inventing one would silently lose the sample the two resulting
    breaths' Ti+Te otherwise sum back to the original breath's exactly. The
    expiration end follows the same natural-vs-cut rule independently.
  - An out-of-range/colliding cut, or a join with no automatic boundary nearby, is a
    soft per-file `SegmentationOverrideNotice` (never a hard failure), the same
    "advisory, never fails a file" posture §5's K-035 boundary-truncation notice has.
  - Empty `cut_s`/`join_s` (or no entry for a file at all) is never even passed to
    `apply_segmentation_overrides` — `segment_file` calls it only when at least one
    list is non-empty, which is what makes every existing analysis byte-identical by
    construction, not by this function happening to be a no-op on empty input.

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

#### Automatic segmentation: `fixed_windows` and `emg_burst`

Two more methods for a recording with nothing to place separators on. Their parameters
live in `processing.segmentation.emg` (`EmgSegmentationSettings`).

- **`fixed_windows`** — windows of `window_s` seconds every `hop_s` seconds from the start
  (5 s and 5 s: a plain tiling), numbered from 1, phase-less. Only complete windows count:
  a trailing piece shorter than a window is dropped, never padded; a recording shorter
  than one window raises `EmgSegmentationError`. A hop shorter than the window overlaps,
  a longer one leaves gaps.
- **`emg_burst`** — segment *k* runs from the onset of burst *k* to the onset of burst
  *k + 1* (the last to the end of the recording); what precedes the first onset belongs
  to no segment. Bursts are found by `segments.detect_bursts` on the ECG-removed signal
  **before** noise reduction (`_process_emg` exposes it as `stages["detect"]`; the noise
  profile is itself cut from the periods between bursts, so detecting on the reduced
  signal would be circular):
  1. per channel, a centred moving RMS envelope over `burst_smooth_s`; non-finite samples
     contribute nothing to their windows (a plain cumulative sum would let one NaN poison
     everything after it);
  2. the envelope is scaled so its median (the resting level, valid while the muscle is
     active for less than half the recording) is 0 and its 95th percentile is 1; a channel
     whose 95th percentile is under `burst_min_contrast` times its median has no bursts and
     is left out; no channel left → `EmgSegmentationError` (a pure-noise or silent
     recording is never cut). The bursts must fill more than about 5 % of the recording
     for the 95th percentile to stand for the burst level;
  3. the remaining channels are averaged into one activation trace *a*; a burst is a
     stretch with *a* ≥ `burst_threshold_frac`/2 that reaches `burst_threshold_frac`
     (hysteresis, Hodges & Bui 1996); gaps shorter than `burst_min_s` are bridged and
     bursts shorter than `burst_min_s` dropped, in that order;
  4. each edge is then moved to where the smoothed **power** envelope crosses half of
     that burst's own plateau (median over the middle half of the burst). A centred window
     smears a step over `burst_smooth_s`, and the on/off levels sit well down the ramp, so
     the coarse edges lead the true ones by up to about half a window; the half-power
     point of a linear ramp is the step itself. The search is confined to one window
     around the coarse edge and never crosses the midpoint to the neighbouring burst.

  Measured: onsets and offsets of a synthetic burst train with a steady carrier are
  recovered within 1 sample at 2 kHz (envelope 0.1 s); with *stochastic* bursts the envelope
  itself fluctuates, and the worst case measured was 26 samples (13 ms) at 2 kHz. Not
  measured: any production EMG recording — the four thresholds are starting values and
  are to be calibrated there before they are frozen.

  Each segment carries `neural_timing` — `ti_emg` (burst duration), `te_emg` (this burst's
  end to the next onset), `ttot_emg` (onset to onset), `ti_ttot_emg = ti_emg / ttot_emg`,
  `bf_emg = 60 / ttot_emg` — with `_emg` names so they are never mistaken for the
  mechanical `ti`/`te`/`ttot`/`ti_ttot`/`bf`; the last burst has no following onset, so
  everything but `ti_emg` is NaN there (a truncated cycle is not a short one). Units are
  declared explicitly in the registry (`s`, `s`, `s`, `—`, `min⁻¹`), because none of the
  five names reaches `quantities.py`'s exact-match rules. Each segment also carries the
  file's `emg_seg_n_bursts`, `emg_seg_burst_frac` (share of the recording spent in
  bursts) and `emg_seg_contrast` (median over the channels used of the envelope's
  95th percentile over its median), identical on every segment of the file.

The noise reference for an EMG-only set is resolved by `resolve_noise_reference_mode`.
`interburst` (only with `emg_burst`, only when named explicitly) cuts the reference from the
periods **between** two consecutive bursts, each shrunk by `burst_smooth_s` at both ends
(the envelope smears an edge by about half its window, so the samples next to a burst are
not yet quiet); the stretch before the first burst and after the last one is not between two
bursts and never counts. `auto_prop` pools bursts (active) against those periods (quiet)
across the batch; it is refused for the other EMG-only methods, which have no such split.


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
  use above, not the raw absolute pressure, so the normalisation ratio (§5.17)
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
  file, of any kind); their consumers are `mfvl.py` (§5.15) and the normalisation
  (§5.17). `baseline_ic` gained
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

### 5.15 MFVL, EFL and ventilatory capacity (v2-only) — `core/analysis/mfvl.py`

`§5.13`'s `manoeuvres.extract` types a breath `fvc`/`ic_fvc` but computes no
spirometric arithmetic for it at all — deliberately deferred to this module
(M-42), kept SEPARATE rather than folded into `manoeuvres.py` so the boundary that
module's own docstring states stays real. Two independent pieces:

**FVC/FEV1/PEF from ONE typed breath** (`mfvl.fvc_metrics`, merged into that
breath's Manoeuvres row by `core.pipeline.run_batch` itself, both in the main loop
and its `§5.13a` external-reference forepass — never by `manoeuvres.extract`,
which stays untouched by this ticket): `V_TLC = insp['volume'][-1]`; the maximal
expiratory flow-volume (MEFV) envelope `v = V_TLC − exp['volume']`, forced
non-decreasing (`np.maximum.accumulate`, a numerical-noise guard, not a
physiological correction); PEF over a ≥10 ms centred moving-average window (a
single-sample spike must not win), mapped back to a real sample for the
back-extrapolation tangent; `t0`/BEV via ATS/ERS 2019's own back-extrapolation
(the tangent through the PEF sample, extrapolated to `v=0`; BEV is the REAL curve
interpolated at `t0`, not the tangent); FEV1 (clipped to FVC when the whole
manoeuvre finishes under 1 s, never extrapolated past the last real sample);
FEV1/FVC; forced expiratory time (`t[-1] - t0`); the end-of-forced-expiration
criterion (FET ≥ 15 s, or < 25 mL change over the last second); a peak
inspiratory-flow reference from the file's own NEXT tidal breath, only when that
breath's own excursion reaches ≥90% of FVC (some protocols record a rapid
near-maximal re-inflation right after the forced exhalation — a genuine
maximal-effort inspiratory reference, not to be confused with an ordinary tidal
breath that happens to follow). `quality` gets `BEV_HIGH` when BEV exceeds
`max(0.1 L, 5% of FVC)`.

**TLC consistency** (`mfvl.apply_tlc_consistency`, a second pass mirroring
`manoeuvres.apply_repeatability`'s own timing — run once per file after every
typed breath's row is in hand): when the SAME file also has `ic`/`ic_fvc`
breath(s), `mfvl_tlc_consistency = V_TLC_fvc − V_TLC_ic` (the IC breaths' own
`vol_ic + ic_eelv_pre`, averaged), flagged `not_from_tlc` past a 0.15 L tolerance
(placeholder, `†`) — comparing a CROSS-file IC reference's TLC is deliberately
never attempted (two different recordings rarely share a volume zero, the same
reasoning `§5.14`'s within-file EELV tracking already applies).

**Placement against a tidal breath** (`mfvl.attach`, `mfvl.tidal_mfvl_ext`):
unlike `§5.13a`/`§5.14`'s `attach()` functions, this one is NOT a post-loop pass —
it runs INSIDE `core.pipeline.run_batch`'s main per-file loop, stamping
`breath['mfvl_ext']` on every tidal breath BEFORE `build_breath_table` joins it in
exactly like `breath['wob']` already is (`core.results.build_breath_table`). This
is why it is deliberately SAME-FILE ONLY: resolving a file's own `fvc`/`ic`
reference (`core.analysis.references.resolve_reference`, the SAME slot every
other reference shares) needs the raw MEFV curve's actual sample arrays, which
only a file already IN this loop iteration's own `breaths` dict has available — a
CROSS-file or external reference source (`§5.13a`'s forepass) only ever carries
`manoeuvres.extract`'s scalar fields, no raw arrays to rebuild a curve from. The
IC operating point this module resolves is therefore also its OWN, simpler
`eelv_tracking='none'`-equivalent (`ic_op = vol_ic_ref`, held constant across the
file — `mfvl.resolve_same_file_ic_op`), not `§5.14`'s full per-breath EELV
tracking (which has not run yet at this point in the loop — `lungvol.attach` is
still a post-loop pass). Both are documented, deliberate gaps for a future ticket
to reconcile once the two loop-timing models can be aligned properly.

**The flow-volume figure** (`mfvl.placed_tidal_loops`, drawn by
`core.plots.draw_flow_volume_mfvl` for both `flow-volume (tidal in MFVL).pdf` and the
Preview panel): no calculation of its own. It reuses `resolve_same_file_curve` and
`resolve_same_file_ic_op`, so a loop can never sit anywhere but where `efl_pct` placed
it. The x axis is volume below TLC (the MEFV curve's own axis, TLC on the left); each
non-ignored tidal breath is drawn at `x(t) = ic_op − (V(t) − vol_endexp)`, so the end
of expiration (EELV) sits at `ic_op` and the end of inspiration (EILV) one tidal volume
nearer TLC. The bold average loop resamples every breath onto 200 points by breath
fraction (a display construct, not a per-column value). Without an IC reference the
loops cannot be anchored and only the envelope is drawn, with a note. The job is
planned only when `processing.breath_types` names an `fvc`/`ic_fvc` breath and
`output.diagnostics.save_flow_volume` (default true) is on; a file with no resolvable
curve returns no figure.

The MEFV envelope itself (`mfvl.resolve_same_file_curve`): `processing.mfvl.source
= "single"` (default) picks the resolved `fvc`/`ic_fvc` attempt with the LARGEST
FVC (ATS/ERS 2019's "report the largest across acceptable attempts", reused here
for which single curve to compare against); `"envelope"` takes the per-volume
MAXIMUM flow across every resolved attempt (Johnson 1999's own composite-MEFV
construction; identical to "single" when only one attempt resolved).

Each tidal breath is placed on a TLC-anchored axis (Johnson 1999):
`v_below_tlc(t) = ic_op − (V(t) − vol_endexp_b)` — at end-expiration
(`V(t)=vol_endexp_b`) this equals `ic_op` (the reference IC's own distance below
TLC); at end-inspiration it equals `ic_op − vt`. `efl_coverage_pct` is the
volume-weighted fraction of the breath's own expiratory excursion whose
`v_below_tlc` falls inside the MEFV curve's own domain — below 100% (a domain
mismatch, typically a genuine `not_from_tlc` disagreement between the file's IC-
and FVC-implied TLC), every placement-dependent column NaNs with ONE per-file
notice (never per-breath); with NO IC reference at all, only the two IC-
independent peak-vs-peak ratios below are filled, everything else NaN with its
own per-file notice.

Columns (all in `breath['mfvl_ext']`, joined into `breaths_table`/`average_row`):
`efl_pct` (Johnson 1999: `100·Σ limited_i·ΔV_i / Σ ΔV_i`, volume-weighted, never
sample-counted, and normalised by the SAME measured path length its own numerator
is built from — self-review finding: normalising by `mechanics.vt` (a net
excursion) instead let sample-to-sample volume noise, e.g. cardiogenic
oscillation, push the result over 100%, since it adds to the |ΔV| path length
without adding to vt; `limited_i = flow_i ≥ mefv(v_i)·(1−efl_rel_tol) −
efl_abs_tol_lps`, both tolerances `†`-placeholder 0.0 — a sample must literally
reach the envelope); `efl_present` (`efl_pct ≥ efl_present_min_pct`, `†` 5.0);
`ex_flow_pct_mfvl_max`/`in_flow_pct_mfvl_max` (this breath's own peak ex/in flow
against the MEFV's LOCAL ceiling across its own operating range —
placement-dependent, NaN without full coverage); `max_ex_flow_pct_mfvl_peak`/
`max_in_flow_pct_mfvl_peak` (against the GLOBAL PEF/peak-in-flow scalars instead
— needs no placement at all, the one pair still filled without an IC reference);
`te_min_mfvl` (`∫ dv/mefv(v)` over `[ic_op−vt_exp, ic_op]`, where `vt_exp` is this
breath's OWN measured expiratory excursion (`max(exp.volume) − min(exp.volume)`),
never `mechanics.vt` (the whole breath's own max−min, inspiration included) —
self-review finding: the two can differ under volume drift, which let the
integration range silently fall partly outside the MEFV curve's own domain while
`efl_coverage_pct` (built from a slightly different quantity) still read 100%; NaN
where the envelope drops under 0.05 L/s, or where `[lo, hi]` is not FULLY inside
the curve's own domain — `_te_min` refuses outright rather than silently
integrating over a clipped-down remainder); `ve_cap = 60·vt_exp·(1−ti_ttot)/
te_min_mfvl`, `ve_pct_cap`, `ve_reserve_pct` (Johnson 1995/1999); `mvv_est`
(`input.subjects.mvv_lpm` first, else `fev1_used × mvv_fev1_multiplier`, `†` 40.0
— ATS/ACCP 2003; `fev1_used` prefers `input.subjects.fev1_l` over the largest
derived FEV1 among this file's own resolved FVC attempts, the SAME preference
`§5.14`'s VC-fallback comment already anticipated), `ve_pct_mvv`, `br_mvv_pct` —
the last three independent of MEFV placement (MVV needs no placement at all), so
they are NaN only when NO IC reference resolved, never merely for partial
coverage. `fev1_source` (`'spirometry'` | `'recorded'`, the ticket's own explicit
acceptance criterion) names which value fed `fev1_used` — a per-FILE constant,
written directly onto `breaths_table`/`average_row` by `core.pipeline.run_batch`
itself AFTER `build_breath_table` has already run (self-review finding: a text
column joined in through `breath['mfvl_ext']`, `tidal_mfvl_ext`'s usual path,
breaks `build_breath_table`'s own `mechanics.mean()` reduction for the WHOLE
file — the same post-hoc column-assignment pattern `§5.13a`'s `ic_ref_source`
(also text) already uses for exactly this reason, never `attach`'s own
per-breath dict).

**PEF used for the tidal ratio columns is the SAME smoothed value the Manoeuvres
sheet shows** (self-review finding): an earlier version re-derived it from the
resolved curve's own raw samples (`mefv_flow.max()`), which for `source=
"envelope"` in particular is a single-SAMPLE maximum of an interpolated curve and
can equal a spike `_pef`'s own ≥10 ms smoothing exists to reject. `attach` now
reads the participating attempt(s)' own already-smoothed `mfvl_peak_ex_flow`
instead (the max across attempts, for "envelope"). `_pef` itself was also
corrected to return the smoothed maximum as its VALUE (the previous version
returned the raw sample at the window's centre, which could still be the spike if
one happened to sit there).

**`apply_tlc_consistency` excludes `ic_cfg.reject_flags`-flagged IC breaths from
the comparison and ignores a non-finite `vol_ic`/`ic_eelv_pre` on any one IC row**
(self-review findings): the first now matches `resolve_same_file_ic_op`'s own
disqualifying rule (a `LOW_EFFORT` attempt should not drag either comparison); the
second means one bad IC row no longer silently poisons `np.mean` into NaN for
EVERY `fvc` row in the file. A shared `_finite(x)` helper (true only for a real,
non-NaN number) replaces the `x == x` idiom used throughout this module for "not
NaN" — that idiom is silently wrong for a MISSING dict key (`.get(...)` hands back
`None`, and `None == None` is `True`), which mattered because `core.pipeline`'s
own `attach` call is now wrapped in its own per-file `try`/`except` (self-review
finding: one FVC row an upstream extraction failure left half-built — `{"kind":
"fvc", "quality": [...]}`, no numeric fields — could otherwise crash the WHOLE
file via a `row["fev1"]` lookup a few functions later, taking down every other
breath's mechanics with it; every other failure mode in this loop already has
this same per-file isolation).

**`source="envelope"`'s composite curve no longer extrapolates a SHORTER attempt's
tail flat** past its own domain (self-review finding): the previous
`right=flow[-1]`/`left=flow[0]` held a truncated attempt's end flow constant
across every volume the LONGER attempt alone reaches, letting a shorter attempt
that was still flowing fast at its own cutoff win the per-volume maximum near RV
— raising the composite envelope exactly where a real curve is near zero, and so
understating `efl_pct` and overstating `te_min_mfvl`/`ve_cap` there without any
warning. Each curve now extrapolates to NaN outside its own domain instead
(`np.fmax`, which ignores NaN), so the maximum at any given volume is only ever
taken among attempts that genuinely cover it.

**Threshold provenance**: every `†`-marked field in `core.settings.MfvlSettings`
is a placeholder (the plan's own starting value), same K-035 provenance as
`§5.13`'s `IcSettings` — this sandbox has no production FVC recording to calibrate
against; the formulas are pinned by analytical/synthetic tests
(`tests/unit/test_mfvl.py`), the cut-offs want a pass against real recordings.

**Units**: `fvc`/`fev1`/`fvc_bev`/`mfvl_tlc_consistency` → L; `fev1_fvc` → `—`
(dimensionless); `fvc_fet`/`te_min_mfvl` → s; `fvc_eofe_ok`/`efl_present` → ""
(explicit blank); `ve_cap`/`mvv_est` → L·min⁻¹ — none match a generic `_RULES`
convention, so each is registered explicitly. `mfvl_peak_ex_flow`/
`mfvl_peak_in_flow` (contain "flow") and every `_pct`/`_pct_` column already
resolve via the generic rules, listed in `registry.py` for documentation only.

**Golden**: `typed_ic_fvc_same_file` (`§5.13`) re-baked with this ticket's new
keys — its own IC (breath #4) and FVC (breath #7) breaths do NOT share a common
TLC in that fixture (a fixture limitation, not a code bug: `generate_data.py`'s
own comment already noted "the actual FVC/FEV1/PEF numerics are a later feature's
scope, not this generator's"), which the `not_from_tlc` flag correctly catches and
which then drives `efl_coverage_pct` to 0% for every tidal breath — so the golden
scenario exercises the "domain mismatch" branch, not the "full coverage" one.
`tests/unit/test_mfvl.py` covers the full-coverage EFL/VEcap arithmetic instead,
with purpose-built fixtures where the IC and FVC TLC agree.

### 5.16 PEEPi and the modified Campbell diagram (v2-only, opt-in) — `core/analysis/pressure.py`

Off by default (`processing.pressure.peepi.enabled = false`); enabling it only ADDS
columns, so every existing output, and every golden scenario, is unchanged with it off.
Needs Flow and Poes. The construction and its five design points are the decision record
in [`beslutninger.md`](beslutninger.md) (28-09-2026); this section is the implementation.

**`t_flow`, the true zero crossing.** `separateintobreathsbyflow` (§5.3) ends an
expiration only when `flow > 0` OR the mean of the next `breathseparationbuffer`
samples is `> 0`. During an end-expiratory pause (flow exactly zero) that forward mean
turns negative first, so the breath boundary lands in the pause, up to `buffer` samples
before flow actually starts (a pause LONGER than `buffer` is refused by the segmenter
itself, as a flat-flow error). `t_flow` is the start of the inspiratory flow
proper: the sample after the LAST `flow >= 0` sample before the phase's peak inspiratory
flow (the last `>= 0` to `< 0` crossing leading into the real inspiration, so a stray
sub-zero sample of noise inside the pause is not taken for the start); a phase with no
such sample falls back to its first sample. It is a
separate quantity from the golden-locked PTP baseline (§5.6, the mean of the first
`ptp.baseline_window_s`), which is untouched. The phase slices are end-exclusive, so
each phase is one sample short of its boundaries; `t_flow` is unaffected.

**Search window and onset (`detect_peepi_onset`).** The window is the last
`search_window_s` of the PRECEDING breath's expiration (the breath immediately before
this one in the recording, ignored or not) followed by this breath's samples up to and
including `t_flow`. It is smoothed with a centred moving average of `smooth_s`
(ends averaged over the samples that exist, not padded), and differenced. Walking back
from `t_flow`, the deflection lasts while the smoothed step is still steeper than
`onset_slope_frac` times the steepest fall anywhere in the window; the sample where
the walk stops is `t_onset`. Because the smoothed slope reaches the threshold before
the raw corner does, `t_onset` sits a few samples early, in the flat stretch before a
real deflection; the pressures READ at `t_onset` and `t_flow` are the raw ones.

- `peepi_dyn = max(Poes[t_onset] − Poes[t_flow], 0)` (cmH₂O), and `0` when below
  `min_deflection`. A window with no falling step at all has no deflection (`t_onset =
  t_flow`, so `0`, not NaN). NaN, plus a per-file notice, only when the breath has no
  preceding breath (breath #1), the predecessor does not directly precede it in the
  recording, or the window is too short or not finite.
- `peepi_lag = (t_flow − t_onset) / fs` (s); `0` below `min_deflection`.
- `peepi_pgas_drop = max(Pgas[t_onset] − Pgas[t_flow], 0)` and `peepi_corr =
  max(peepi_dyn − peepi_pgas_drop, 0)` (only with Pgas): the expiratory-muscle
  contribution is removed by the Pgas fall over the SAME interval (`peepi_corr`, before the clamp, equals the
  pre-flow Pdi rise) (Zakynthinos 1997, 1999). With a constant Pgas the correction is
  the identity.
- `int_oes_preflow` = the area of the deflection, `∫ (Poes[t_onset] − Poes) dt` over
  `[t_onset, t_flow]` (trapezoid, cmH₂O·s), and `ptp_oes_preflow = int_oes_preflow ·
  bcnt · vefactor`. **Reported separately and never added to any `*_peepi` column:** the
  ordinary PTP is already referenced to its own end-expiratory baseline, so adding the
  pre-flow area would subtract that baseline twice (`PTP_INVESTIGATION.md`).

**The modified Campbell diagram.** With PEEPi on, the Campbell figures (the written PDFs and
the Preview panel) draw a hatched rectangle UNDER the elastic-recoil polygon: it spans the tidal
volume (EELV to EILV) and rises above the end-expiratory Poes by the amount added above, so its
area is `wob_in_thr` before the unit change and the per-minute scaling. On the average loop the
height is the mean over the breaths PEEPi was computed for (a breath with nothing to add counts
as 0); the rectangle uses the breath's own end-expiratory volumes, so it can differ marginally
from `vt` when the volume drifts within the breath. Nothing at all is drawn with the feature off.

**Threshold work, and what the existing columns already hold.** `peepi_source` is
`corrected` when Pgas exists, else `dynamic` (written to the Provenance sheet). The
rectangle is `PEEPi × VT` in J·min⁻¹ with the same cmH₂O·L → J factor and scaling as §5.7
(`· (98.0638 / 1000) · bcnt · vefactor`), BUT only the part the existing polygon does not
already contain is added. Where the segmenter put the boundary decides that: the polygon
measures against the Poes at the phase start (resistive part, `poesin[0]`) and at the
end of expiration (elastic part), and `calcptp` against the mean of the first
`ptp.baseline_window_s`. A boundary in the pause BEFORE the deflection (the usual case:
the pause belongs to the inspiratory phase) makes those references the pre-deflection
level, so `int_oesinsp`, `wob_in_total` and `wobtotal` already contain the deflection;
a boundary AFTER it (the fall happened under the previous breath's last expiratory
samples) makes them the already-fallen level, so they do not. With
`shift_wob = min(max(Poes[t_onset] − Poes[phase start], 0), peepi_dyn)` (what the polygon
does NOT yet hold), the polygon already holds `peepi_dyn − shift_wob`, and
`wob_in_thr = max(peepi_source_value − (peepi_dyn − shift_wob), 0) · vt · (98.0638 /
1000) · bcnt · vefactor` (never negative: a corrected PEEPi smaller than the polygon's own
share adds nothing rather than subtracting). `wob_in_total_thr = wob_in_total +
wob_in_thr` and `wobtotal_thr = wobtotal + wob_in_thr`; `calculatewob`, its five columns
and its V = 0 crossing are untouched, and `wobtotal` never absorbs `wob_in_thr`.
`int_oesinsp_peepi = int_oesinsp + shift_ptp · ti` with `shift_ptp = min(max(Poes[t_onset]
− mean(insp Poes[:ptp_bw]), 0), peepi_dyn)`, and `ptp_oesinsp_peepi = int_oesinsp_peepi ·
bcnt · vefactor`; with Pgas and Pdi, `int_pdiinsp_peepi` / `ptp_pdiinsp_peepi` use
`shift_pdi = min(max(mean(insp Pdi[:ptp_bw]) − Pdi[t_onset], 0), peepi_corr)` (Appendini
1996). The same recording therefore gives the same totals wherever the boundary falls: in
the golden fixture (boundary in the pause) the `*_peepi` columns equal the plain ones and
`wob_in_thr` is 0, in a boundary-after-deflection breath they add the full amount
(`tests/unit/test_peepi.py`). A NaN `peepi_dyn` makes every dependent column NaN.

All columns live in `breath["pressure_ext"]`, joined after `breath["wob"]` in
`build_breath_table`, and are computed in `run_batch` right after each breath's
`calculatemechanics`. Units come from `core/quantities.py`'s generic rules (`peepi*` →
cmH₂O, `peepi_lag` → s, `int_`/`ptp_`/`wob*`); the columns are also registered in
`core/analysis/registry.py`.

**Known limitations (measured on synthetic data, not yet on real recordings).** A
cardiac oscillation of 0.5–2 cmH₂O on Poes (1.2 Hz) is not removed by the 50 ms
smoothing; one rising smoothed step stops the backward walk, so `peepi_dyn` can be
over- or underestimated by more than the ripple (1.4–4.9 for a true 3.0 at 0.5–1 cmH₂O
ripple) and read 0 in a share of breaths at 2 cmH₂O. An expiratory Poes hump that
decays continuously into the pre-flow fall is counted as part of the deflection
(the well-known reason the dynamic value overestimates PEEPi with active expiration,
which the Pgas correction is meant to remove); one separated from the fall by a flat
stretch is not. Without Pgas the `dynamic` value carries that overestimate. `peepi_pgas_drop`
is clamped at 0, so the corrected value equals the Pdi rise before flow only when Pgas
falls. With `wob.calc_from = "average"` the polygon is an average-breath value while
PEEPi is per breath. Breath #1 of a file is blank in every new column, and the per-file average skips blanks,
so an averaged `wobtotal_thr` need not equal the averaged `wobtotal` plus the averaged
`wob_in_thr` when breath #1 differs from the rest. A 0 (no deflection) cannot be told from a detection that found
nothing; only an unlocatable window or a missing predecessor is NaN with a notice.

**Thresholds are provisional.** `search_window_s = 1.0`, `smooth_s = 0.05`,
`onset_slope_frac = 0.1` and `min_deflection = 0.5` are literature-informed starting
values, not measured ones. Measured so far only on synthetic data: the analytical
pause case (`tests/unit/test_peepi.py`: `peepi_dyn` equals the known fall to 1e-9 with the
segment boundary 400 samples before flow; an exact-zero pause puts the boundary at the pause
start for every workable `buffer`, so that test does not show buffer-independence beyond that) and the built-in sample recording, which has no PEEPi at
all but a Poes cardiac ripple and wander of 1–2 cmH₂O: with the starting thresholds 3
of its 8 measurable breaths report a 1.2–2.1 cmH₂O deflection, a wider smoothing
window (0.2 s, 0.4 s) does not remove them, and `min_deflection = 2.5` reports zero
everywhere. A steeper fall earlier in the search window (the tail of an expiratory
Poes hump decaying to baseline) raises the reference the slope threshold is a fraction
of; that too is not yet measured on real recordings.

**Golden**: `flow_peepi_on` (a dedicated `synth_peepi_A.csv`, whose breaths after the
first follow a 0.4 s zero-flow pause carrying a 3 cmH₂O Poes fall and a 1 cmH₂O Pgas
fall over the same 0.25 s), pinned analytically in `test_golden.py`: `peepi_dyn = 3.0`,
`peepi_corr = 2.0`, breath #1 NaN.

### 5.17 Normalisation to a maximal manoeuvre (v2-only, opt-in) — `core/analysis/normalisation.py`

Off by default (`processing.pressure.normalization.enabled = false`). It is a pass over the
finished per-breath tables, run after the operating-lung-volume pass; it adds no column to
the Data sheet or the averages and writes one extra sheet, "Pressure normalised", per file.
The design points are in [`beslutninger.md`](beslutninger.md) (28-09-2026).

**The reference.** `resolve_reference(file, "max_insp")` (§5.13a: explicit entry, group
default, the file's own typed `max_insp`/`sniff` breath). The values it reads are the ones
`max_effort_from_breath` (§5.13) already wrote on the manoeuvre row: `poes_max_ref`
(baseline sample minus the minimum of the inspiratory Poes), `pdi_max_ref` (maximum minus
the baseline sample of the inspiratory Pdi), `rms_max_ref` (the largest channel peak of the
rolling RMS over the whole breath) and `rms_max_ref_col_<channel>` (each channel's own
peak). Over the breaths a link names, the largest finite value of each is used. Only breaths
typed `max_insp` or `sniff` count; the kinds behind the reference are kept for Provenance.

**Per tidal breath** (`n_bw = max(1, round(ptp.baseline_window_s · fs))`, the `calcptp` baseline window,
with `fs = len(insp flow) / ti` recovered from the breath so a resampled run uses its own rate):

| column | formula |
|---|---|
| `poes_insp_swing` | `mean(insp.poes[:n_bw]) − poes_mininsp` |
| `pdi_insp_swing` | `pdi_maxinsp − mean(insp.pdi[:n_bw])` |
| `poes_mean_insp`, `pdi_mean_insp` | `int_oesinsp / ti`, `int_pdiinsp / ti` |
| `*_swing_pct`, `*_mean_insp_pct` | `100 · x / poes_max_ref` (or `pdi_max_ref`) |
| `tt_es`, `tt_di` | `int_oesinsp / (ttot · poes_max_ref)`, `int_pdiinsp / (ttot · pdi_max_ref)` |
| `rms_insp_max_pct` | `100 · rms_insp_max / rms_max_ref` |
| `nrdi` | `rms_insp_max_pct · bf` |

`tt_*` is the tension-time index `(Pmean/Pmax)·(Ti/Ttot)` (Bellemare & Grassino 1982;
Ramonatxo et al. 1995) written with the pressure-time integral, in which `Ti` cancels. `nrdi`
follows Murphy et al. 2011 (EMG as a percentage of maximum times breathing rate, arbitrary
units). Anything whose reference is missing, or not a positive number, is NaN, with one
notice per file; the column set follows the signal set alone, so a file without a
reference writes the same columns.

**EMG-normalised sheet.** `processing.emg.normalization_reference_file` naming a file that
has `max_insp`/`sniff` breaths typed in it is now read at those breaths: a `_col_<channel>`
column against that channel's peak (NaN if it has none), a `*_mean` column against the mean
of the channel peaks, every other RMS column against `rms_max_ref`. This applies only with
`normalization = "per_file_max"`; `per_file_mean`, and a file with no such breath, keep the
earlier behaviour (each column's own maximum or mean over that file's breaths).

**Sniff.** A nasal sniff has no mouth flow, so the inspiration/expiration boundary inside
its typed breath is arbitrary; for `kind = sniff` the pressure swings are therefore taken over
the whole typed breath (from the first sample's baseline), for `max_insp` over its inspiration.

**Known properties, not corrected here.** (a) `calcptp` integrates over `linspace(0, n/fs, n)`,
so every integral, and with it `tt_*` and `*_mean_insp`, is high by n/(n−1) (about 2 % at 50
samples); the golden suite locks it. (b) `bf` counts every breath in the file, typed ones
included, unless `breath_counts` is set, so `nrdi` is high in a tidal file with embedded
maximal breaths. (c) `tt_di` uses baseline-referenced Pdi in numerator and denominator,
where Bellemare & Grassino used absolute Pdi: it is lower than the classical index whenever
end-expiratory Pdi is above zero, so do not read it against the classical 0.15 threshold.
(d) An external reference source (outside the run's own files) is segmented without EMG
noise reduction, so its `rms_max_ref` can differ from an in-batch source's; this predates
normalisation and is not fixed here.

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

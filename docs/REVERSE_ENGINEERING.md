# RespMech — Reverse-engineering & correctness reference

_Phase 1 documentation for the production refactor. Describes the **current**
(`v1.0.0`) code as it actually behaves, with special attention to every
physiological calculation (formula, units, assumptions). This is the reference a
reviewer uses to confirm the refactor preserves behaviour._

Source files analysed: [`respmech.py`](../respmech.py) (1660 lines),
[`emg.py`](../emg.py) (585), [`entropy.py`](../entropy.py) (329),
[`example.py`](../example.py), [`README.md`](../README.md).

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
Standard SampEn (Chebyshev norm): template length `m = entropy_epochs`, tolerance
`r = entropy_tolerance · SD(channel)`; result `SampEn = −ln(A/B)`. Computed per
channel per breath (whole breath, inspiration, expiration), reduced to max/min/mean
across channels. Vendored pyEntropy implementation.

### 5.10 EMG signal conditioning (optional)
- **ECG removal** (`remove_ecg`, `emg.remove_ecg`): detect R-peaks (`find_peaks`,
  `minheight`/`mindistance`/`minwidth`), build a **time-aligned averaged ECG
  template** per window, amplitude-fit it to each beat (least squares) and subtract.
- **Noise reduction** (`remove_noise`, `emg.reducenoise`→`removeNoise`): spectral
  gating — STFT the signal and a **noise-profile interval**, threshold each
  frequency bin at `mean + n_std·SD` of the noise, smooth the mask, invert. Adapted
  from Tim Sainburg; **requires `librosa`**.
- **RMS outlier handling** (`processoutliers`): `rms_poes = rms_max / poes_mininsp`;
  breaths outside `mean ± outlierrmssdlimit·SD` of the other breaths have their
  `rms_max`/`rms_mean` replaced by the others' mean.

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
  the current code accordingly. **Which baseline convention is physiologically
  correct must be decided in Phase 2** (the extra `- pressure[0]` double-subtracts a
  baseline that `adjustforintegration` already set).
- **EMG/ECG-removal is version-dependent.** The current `master` EMG conditioning
  (ECG removal + spectral noise reduction) differs from the "new ECG removal"
  version Emil validated with; EMG RMS / integrated-EMG columns differ by up to
  ~90 % while mechanics/WOB are unaffected. Latent bug #1 additionally **crashes**
  real files that combine excluded breaths with EMG (`RIU_H5_IC`, `RIU_H6_IC`,
  `RIU_H6_Baseline`).

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

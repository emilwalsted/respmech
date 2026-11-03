# Profiling a full analysis on a large recording

Measured 03-11-2026 with `tools/profile_large_recording.py 60 2000 <dir>`: the
bundled synthetic sample scaled to one hour at 2 kHz (1125 breaths, 7.2 million
samples, 534 MB CSV) with every signal (flow, volume, Poes, Pgas, Pdi, three EMG
channels), ECG removal and noise reduction on (automatic prop_decrease) and all
diagnostic figures. Python 3.11.18, numpy 2.4.6, scipy 1.17.1, pandas 3.1.0,
librosa 0.11.0, Linux, one process. Wall times vary by about 5 % between runs.

## Where the time goes

| Phase | Wall time |
| --- | --- |
| Compute (`run_batch`) | 102 s before, 91 s after the two optimisations below |
| Writing tables (xlsx) | about 8 s |
| Diagnostic figures (separate process) | about 370 s |

The first run in a fresh environment also pays about 33 s of one-off numba JIT
compilation (librosa's STFT); numba caches it on disk, so later runs do not.

### The ten most expensive functions of the compute phase (own time, before)

| # | Function | Own time | Cumulative | Calls |
| --- | --- | --- | --- | --- |
| 1 | `numpy.array` (built-in) | 9.4 s | 9.4 s | 104 675 |
| 2 | `core/compute.py: correctdrift` | 5.1 s | 5.1 s | 2 |
| 3 | numpy ufunc `reduce` (sum/min/max) | 4.3 s | 4.3 s | 779 043 |
| 4 | `core/noise.py: apply` | 4.2 s | 19.0 s | 39 |
| 5 | `numpy.correlate` (ECG shift narrowing) | 3.7 s | 3.7 s | 25 971 |
| 6 | built-in `min` | 3.7 s | 3.7 s | 49 399 |
| 7 | `librosa db_to_power` | 3.1 s | 3.1 s | 39 |
| 8 | `numpy.ones_like` | 2.9 s | 2.9 s | 39 |
| 9 | built-in `max` | 2.8 s | 2.8 s | 41 302 |
| 10 | `core/compute.py: separateintobreathsbyflow` | 2.7 s | 4.1 s | 2 |

Where they come from: 4.7 s of `numpy.array` was `_ecg_remove` re-stacking the
per-sample row list `subtractecg` returned, and 1.5 s its window copies; the
built-in `min`/`max` are almost all `calculatemechanics` applying Python's `min()`
and `max()` to numpy arrays (6.8 s over 1125 breaths); `db_to_power` and `ones_like`
sit inside the noise-reduction spectral gate, which the automatic prop_decrease
search runs 39 times. By cumulative time the large blocks are the shared noise
profile (`_build_noise_set`, 41 s, including ECG removal of the reference file, 25 s)
and the per-breath mechanics table (`build_breath_table`, 17 s; `calculatemechanics`
20 s, `compute_segment_emg` 12 s).

### Figures

Per figure, from the output files' timestamps: Campbell (average) about 88 s,
Campbell (breaths, 94 pages) about 181 s, the two signal overviews 12 to 16 s each,
the three EMG overviews about 20 to 24 s each, the volume figures under 10 s.

## Optimisations made

Both are proven bit-identical: golden green, and every output table of the
one-hour run (17 hashes: the xlsx sheets read back at full precision, the in-memory
breath and average tables and the noise report) is byte for byte the same before
and after.

1. **`correctdrift`** was a per-sample Python loop. It now builds the same running
   offsets with `np.add.accumulate`, which adds strictly in sequence, so every offset
   is the same float the loop produced (including float32 input, where the loop
   accumulated in float32). 5.1 s to 1.0 s (two calls on 7.2 million samples).
   `tests/unit/test_correctdrift_reference.py` compares it with the original loop.
2. **`subtractecg` / `remove_ecg`** returned a list of 7.2 million one-row arrays,
   which every caller stacked again. They now return the C-ordered array that
   round trip produced (same values, dtype and memory layout; a test in
   `tests/unit/test_ecg.py` pins it). `numpy.array` own time 9.4 s to 4.3 s.

## Proposed, not made

- **`calculatemechanics`: `np.min`/`np.max` instead of Python's `min()`/`max()`**
  (about 6.5 s). Not proven identical: with a NaN in a breath, Python's `min()`
  returns an order-dependent value while numpy returns NaN, and for signed zeros the
  two can return a different zero. Worth doing only behind a golden scenario with NaN
  inside a breath.
- **Noise reduction**: the automatic prop_decrease search re-runs the full spectral
  gate for every candidate (39 calls, 19 s). Reusing the STFT and the mask across
  candidates would cut most of it, but reorders floating-point work and needs its
  own bit-identity proof per stage.
- **Campbell figures** (about 270 s of the 370 s of figures): the averaged figure
  draws every breath; decimating the drawn breath curves or rasterising them would
  shrink both time and the 3 MB PDFs. This changes output files, not numbers, so it
  is a design choice for the maintainer rather than a pure optimisation.
- `separateintobreathsbyflow` (4 s) and the ECG template sweeps (already narrowed,
  see `core/emg.py`) were left alone: little left to gain without changing the
  arithmetic.

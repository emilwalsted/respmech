# RespMech golden / characterisation tests

This directory locks the **current** numeric behaviour of `respmech.py` so that the
planned production refactor can be proven to produce identical results.

## Why

The refactor (calculation core / CLI / Qt UI split, settings restructure) must not
silently change any physiological result. These tests run the *unmodified* current
code over deterministic input, capture every number it produces, and store it as a
golden reference. After refactoring, the same test must reproduce those numbers
within a tight tolerance.

## What is covered

`make_golden.py` runs a matrix of settings scenarios over two synthetic recordings
(`input/synth_case_A.csv`, `input/synth_case_B.csv`, produced by
`generate_data.py`):

| Scenario | Separation | WOB method | EMG | Notes |
|---|---|---|---|---|
| `flow_wob_average`    | flow | average    | on  | core mechanics + WOB + entropy + EMG RMS |
| `flow_wob_individual` | flow | individual | on  | per-breath WOB path |
| `flow_integratevol`   | flow | average    | on  | volume integrated from flow (`cumtrapz`) |
| `flow_exclude_noemg`  | flow | average    | off | `excludebreaths` + `breathcounts` override + processed-data export |

For each scenario the reference stores: the merged **average** breath data, the
**per-file** breath-by-breath tables, and a compact **processed-data** summary
(shape + per-column sum/mean/min/max).

## Known gaps (to be locked against real production data)

The synthetic data cannot faithfully exercise every path. The following are
**deliberately not** covered here and should be locked using Emil's real
production dataset + expected results:

- **Volume-based breath separation** (`separateby: "volume"`) — the separator
  assumes a specific breath morphology and is brittle on analytic signals.
- **ECG removal** (`remove_ecg`) and **EMG noise reduction** (`remove_noise`,
  needs `librosa`) — signal-conditioning paths best validated on real EMG.
- **EMG + entropy on the same channels** — the current code mishandles this
  (see "Latent issues" below), so synthetic entropy channels are kept disjoint.

## Latent issues found while building these tests

These are current-code bugs discovered while characterising behaviour. They are
captured here so the refactor fixes them deliberately (and updates the golden
reference with justification):

1. **EMG overview plot crashes on an ignored breath** — in `emg.py`
   `saveemgplots()` the ignored-breath `Rectangle` width is built as a 1-element
   array, raising a matplotlib shape error. The forced EMG overview plot therefore
   cannot be produced when any breath is excluded.
2. **Entropy computed on untrimmed / misaligned data** — `analyse()` never trims
   `entropycolumns`, yet indexes it with trimmed-coordinate breath boundaries; it
   also overwrites entropy columns that overlap EMG columns, which additionally
   causes a shape mismatch. Overlapping EMG/entropy channels crash or silently
   misalign.
3. **Processed-data export assumes exactly 5 EMG channels** — `getprocesseddata()`
   hardcodes column names `EMG1..EMG5`, so exporting processed data with any other
   EMG channel count raises a shape error.

## Environment

The original code only runs faithfully on an older SciPy stack (see
`requirements-golden.txt` for why — removed `cumtrapz`, keyword-only `simpson`):

```bash
conda create -y -n rmgolden python=3.12
conda activate rmgolden
pip install -r tests/golden/requirements-golden.txt
```

## Usage

```bash
# Regenerate the synthetic input (rarely needed; output is committed)
python tests/golden/generate_data.py

# (Re)generate the golden reference — only when a change is intentional
python tests/golden/make_golden.py --write

# Verify current code still matches the golden reference
pytest tests/golden/test_golden.py -v
```

## Tolerance

`rtol = 1e-9`, `atol = 1e-12` (see `test_golden.py`). The refactor is a
restructuring, not a numerical change, so results should match to floating-point
reassociation. A legitimate, explained numerical change requires regenerating the
reference with `--write` and justifying the diff in the PR.

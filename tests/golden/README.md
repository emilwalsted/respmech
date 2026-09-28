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

`make_golden.py` runs a matrix of settings scenarios over synthetic recordings
(`input/synth_case_A.csv`, `input/synth_case_B.csv`, produced by
`generate_data.py`):

| Scenario | Oracle | Separation | WOB method | EMG | Notes |
|---|---|---|---|---|---|
| `flow_wob_average`    | legacy | flow | average    | on  | core mechanics + WOB + entropy + EMG RMS |
| `flow_wob_individual` | legacy | flow | individual | on  | per-breath WOB path |
| `flow_integratevol`   | legacy | flow | average    | on  | volume integrated from flow (`cumtrapz`) |
| `flow_exclude_emg`    | legacy | flow | average    | on  | `excludebreaths` + `breathcounts` with the EMG pipeline active |
| `flow_exclude_noemg`  | legacy | flow | average    | off | `excludebreaths` + `breathcounts` override + processed-data export |
| `flow_only`           | v2     | flow | average    | off | `analysis.signals = ["flow"]`, no pressure channels — entropy columns [10,11,12] kept, to prove entropy stays signal-set-independent |
| `poes_only`           | v2     | flow | average    | off | `analysis.signals = ["flow", "poes"]` — work of breathing, no Pgas/Pdi |
| `emg_only_whole_file`  | v2     | n/a (EMG-only) | n/a | on | `analysis.signals = ["emg"]`, `processing.segmentation.method = "whole_file"` — each entire file is one segment, over a DEDICATED `input/synth_emgonly_*.csv` pair (own RNG stream, never `synth_case_*.csv`) |
| `emg_only_separators`  | v2     | n/a (EMG-only) | n/a | on | same EMG-only input pair, `processing.segmentation.method = "separators"` — one manual boundary list PER FILE, giving a DIFFERENT segment count per file (4 and 3) |
| `typed_ic_fvc_same_file` | v2  | flow | average    | on  | `processing.breath_types` marks breath #4 as an inspiratory-capacity manoeuvre and breath #7 as a forced-vital-capacity manoeuvre, IN THE SAME FILE — over a DEDICATED `input/synth_manoeuvre_A.csv` (own RNG stream, trapezoid breath shapes, never `synth_case_*.csv`). `test_typed_ic_fvc_same_file_vol_ic_matches_analytical_value` (`test_golden.py`) checks the extracted `vol_ic` against the literal analytical constant `3.0` (not just the committed reference), to `abs_tol=1e-9` — see `generate_data.py`'s `make_manoeuvre_file()`/`_manoeuvre_breath()` docstring for why the generator's plateau design makes that exact, not merely close. **First extension, cross-file references (`core.analysis.references.attach`):** breath #4's own typed IC already makes `resolve_reference`'s "own typed breath" fallback resolve THIS SAME file as its own IC reference — no new `processing.references`/`reference_defaults` entry needed — so `vol_ic_ref`/`ic_ref_n`/`ic_ref_source` (self-referencing: source = `synth_manoeuvre_A.csv` itself, `n=1`, value = breath #4's own `vol_ic`) are added to the scenario's average/breath tables, keys added only. **v2-scenarios are extended by later work in the same programme**: operating-lung-volume derivation and FVC/MFVL numerics will each ADD further keys to this SAME scenario and regenerate it with their own justification — an added key alone is not a regression here. |
| `typed_ic_crossfile` | v2 | flow | average | on | The CROSS-file counterpart of `typed_ic_fvc_same_file` above: a dedicated, reference-only recording `input/synth_crossfile_ic.csv` (own RNG stream, own `synth_crossfile_*.csv` naming — never `synth_case_*.csv`/`synth_manoeuvre_*.csv`, each already a SIBLING scenario's own committed glob; reusing either prefix would silently pull this file into that scenario's own batch too) holds a SINGLE IC-typed breath and no tidal breathing at all, so `core.pipeline.run_batch`'s own detection gives it `role="reference"` — `golden_newcore.run_scenario`'s own `if fr.breaths_table is not None` guard (already built for an earlier scenario) is what skips it from `per_file`, exactly as `run_scenario`'s own comment already anticipated. The SECOND file, `input/synth_crossfile_stage.csv` (ordinary tidal breathing, `make_file`'s own smooth sin² model, its own fresh seed), names the IC file via an EXPLICIT `processing.references` entry — `resolve_reference`'s cross-file path, not `typed_ic_fvc_same_file`'s same-file "own typed breath" fallback. `[[input.subjects]]` supplies a VC (5.0 L) for group `"synth"` (both files' shared leading-token group) so the VC-anchored operating-lung-volume family (`vol_eelv`/`vol_eilv`/`*_pct_vc`) resolves to real numbers rather than NaN; no TLC is entered, exercising that side's NaN path too. `processing.lung_volume.ic.eelv_tracking` is left at its default `"none"` — the ONLY valid choice for a cross-file reference (`"within_file"` needs a SAME-file IC and NaNs otherwise, per `lungvol.py`'s own module docstring) — which makes `ic_op`, and therefore `vol_eelv = vc - ic_op`, an EXACT CONSTANT across every breath in the stage file (it never reads `vol_endexp` under `"none"` tracking). `test_typed_ic_crossfile_vol_ic_matches_analytical_value`/`test_typed_ic_crossfile_stage_file_resolves_the_crossfile_reference`/`test_typed_ic_crossfile_reference_only_file_is_skipped_from_per_file` (`test_golden.py`) pin: the reference file's own `vol_ic == 3.0` (same exact-match reasoning as `typed_ic_fvc_same_file`, but with `ic_eelv_pre_n == 1` and `quality == ["BOUNDARY"]` — NOT the empty list that scenario's breath #4 gets, since this breath is simultaneously the min- and max-numbered breath among an EMPTY tidal set, a documented, expected `run_batch` behaviour for a reference-only file's typed breath, not a bug); the stage file's constant `vol_eelv == 2.0` (`vc(5.0) − ic_op(3.0)`, exactly — analytically pinned rather than described as "moving in some direction", which only applies to `"within_file"` tracking's own per-breath drift term, already covered by `tests/unit/test_lungvol.py`'s own analytical test for THAT mode); and that the reference file never gets a `per_file` breathdata entry. The companion end-to-end workflow test, `tests/unit/test_workflow_ic_file_per_participant.py`, exercises the SAME feature stack through a realistic multi-file-per-participant batch (`reference_defaults` GROUP resolution across two participants, `run_batch` + `write_batch`, `group_readout`, `check_links`, `build_cohort_summary`, `respmech validate`) that no earlier golden scenario or unit test ran end to end. |

`Oracle` names which generator is authoritative for that scenario's committed
numbers: `legacy` = the frozen v1 oracle (`make_golden.py --write`, cross-checked
by `golden_newcore.py`); `v2` = bagt directly from the v2 core
(`golden_newcore.py --write`), for a scenario whose settings the legacy dict/
`migrate_dict` path has no shape at all. `flow_only`/`poes_only`/`emg_only_whole_file`/
`emg_only_separators`/`typed_ic_fvc_same_file` are all `v2` rows — each a committed
`tests/golden/scenarios/<name>.toml`, since `analysis.signals` (and, for the two
EMG-only rows, `processing.segmentation.method`/`separators`; for `typed_ic_fvc_same_file`,
`processing.breath_types`) has no legacy-dict shape to express it in; `V2_SCENARIOS` in
`make_golden.py` is the table a feature ticket adds a `v2` row to.

**Regenerating `golden_reference.json` after adding a `v2` scenario merges, it never
overwrites.** `golden_newcore.py --write` recomputes the ENTIRE `SCENARIOS` union
(legacy + v2) through the current v2 core, and a naive `json.dump` of that result
would rewrite every legacy entry's numbers too — two different NumPy/SciPy builds can
legitimately differ in a float's last one or two bits (well within the
`rtol=1e-9`/`atol=1e-12` the tests themselves use), which is enough for a plain
overwrite to silently stop being byte-for-byte identical to the previously committed
legacy entries. Adding `flow_only`/`poes_only` hit exactly this (two legacy
entries' `wob_ex_total` moved in their 17th significant digit against the sandbox's
freshly installed NumPy/pandas/SciPy). The fix: load the freshly written file back,
take ONLY the new scenario key(s) from it, and merge those onto the previously
committed dict before writing — never take the whole freshly written file as-is.
Verify with a value-level (not text-level) equality check across every pre-existing
key, since `git diff` on this file is not a reliable read here either — inserting a
new scenario's tens of KB in the middle of a `sort_keys=True` dump shifts everything
after it, and a naive text diff of the shifted region can look like a rewrite even
when every value is unchanged (these two new scenarios added ~34 KB combined and the
`git diff` still showed thousands of changed lines).

**Golden job runtime:** measured locally (sandbox, `pytest tests/golden -q`,
13 non-skipped + 5 skipped production tests) at ~7.5 s after adding
`typed_ic_fvc_same_file` plus its own dedicated
`test_typed_ic_fvc_same_file_vol_ic_matches_analytical_value` test — still the same
order of magnitude as the ~11 s measured for the 11 non-skipped tests before it
(`emg_only_whole_file`/`emg_only_separators`), since the new scenario is one more
`run_batch` over a small synthetic input file, same as every other scenario here; the
apparent drop is sandbox timing noise, not a real speed-up. `ci.yml`'s `golden` job
(`timeout-minutes: 15`) was not itself re-measured on the real runner by this ticket
(no Actions access from this environment) — confirm the actual CI duration on the
merge commit that adds this scenario, and raise `timeout-minutes` in the same commit
only if it is ever observed to exceed ~10 minutes.

Re-measured locally (sandbox) at ~9.4 s (17 non-skipped + 5 skipped) after adding
`typed_ic_crossfile` plus its own 3 dedicated tests — same order of magnitude
again, same reasoning as above (one more small `run_batch` per scenario). Not
re-measured on the real CI runner by this ticket either; same confirm-and-raise-if-
needed note applies to the merge commit that adds THIS scenario.

For each scenario the reference stores: the merged **average** breath data, the
**per-file** breath-by-breath tables, and a compact **processed-data** summary
(shape + per-column sum/mean/min/max).

**Breath numbers are engine-relative — mind this when reading the exclusion
scenarios.** `excludebreaths` names breaths by *number*, and v1 and v2 do not number
these files identically: on `input/synth_case_A.csv` the v1 oracle detects 9 breaths
where the v2 core detects 8, v1's extra one being a leading sliver (`integral_emg_col_2`
≈ 0.0005 against ≈ 0.025 for the real breaths). So `[["synth_case_A.csv", [3]]]` drops
v1's third breath and v2's third breath, which are *different physical breaths*. Both
scenarios above are baked from the v2 core, like every other entry in
`golden_reference.json`, so they lock v2's own behaviour and are internally consistent.
Whether the same off-by-one shows up on real recordings has **not** been checked here.

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

1. ~~**EMG overview plot crashes on an ignored breath**~~ — **FIXED 23-09-2026.** In
   `legacy/emg.py`'s `saveemgplots()` the ignored-breath `Rectangle` width was built as
   a 1-element array, so the *forced* EMG overview plot raised
   `ValueError: setting an array element with a sequence` for any run that excluded a
   breath — and `analyse()`'s excepthook swallowed it into `Error log.txt`, so the run
   produced no output rather than an obvious failure. It crashed on the pinned stack
   above as well as on a modern one. Fixed with Emil's explicit permission (the single
   sanctioned edit to `legacy/` — see `../../CLAUDE.md`), which is what allows the
   `flow_exclude_emg` scenario to exist.
2. **Entropy computed on untrimmed / misaligned data** — `analyse()` never trims
   `entropycolumns`, yet indexes it with trimmed-coordinate breath boundaries; it
   also overwrites entropy columns that overlap EMG columns, which additionally
   causes a shape mismatch. Overlapping EMG/entropy channels crash or silently
   misalign.
3. **Processed-data export assumes exactly 5 EMG channels** — `getprocesseddata()`
   hardcodes column names `EMG1..EMG5`, so exporting processed data with any other
   EMG channel count raises a shape error.

## Production golden (real validation data)

A second golden is built from Emil's real validation recordings under
`production/` (raw data + expected spreadsheets are **gitignored** — public repo).
Only the derived numbers are committed: `production_golden.json` (the current
code's captured output), `production_manifest.json` (input SHA-256 + sizes), and
`production_comparison.md` (comparison against Emil's expected spreadsheets, with
root-cause analysis).

Build/verify (raw data must be present locally):
```bash
python tests/golden/build_production_golden.py        # all scenarios
python tests/golden/summarize_production.py            # regenerate the report
pytest tests/golden/test_production_golden.py -v       # skips if data absent
```

It covers the paths the synthetic golden could not: **volume-based breath
separation**, **ECG removal + spectral noise reduction + RMS outlier processing**,
and real **trim / quiet-breathing** edge cases. See `production_comparison.md` for
the full match analysis (notably: the current code's PTP columns differ from the
older pre-`1630c40` expected spreadsheets by design, and the EMG columns track a
newer ECG-removal algorithm than `master`).

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

# (Re)generate the LEGACY_SCENARIOS entries — only when a change is intentional.
# Merges onto golden_reference.json rather than overwriting it, so any committed
# V2_SCENARIOS entry survives. Never touches (and cannot produce) a V2_SCENARIOS
# entry: the frozen v1 oracle has no shape for the settings a v2 scenario needs
# (typed breaths, references, separators, ...).
python tests/golden/make_golden.py --write

# (Re)generate a V2_SCENARIOS entry — bagt directly from the v2 core, never
# through the legacy oracle. Regenerates every scenario in SCENARIOS (legacy AND
# v2); the legacy entries it writes must still match make_golden.py's own output.
python tests/golden/golden_newcore.py --write

# Verify current code still matches the golden reference (both LEGACY_SCENARIOS
# and V2_SCENARIOS, run through the v2 core either way — see golden_newcore.py)
pytest tests/golden/test_golden.py -v
```

**LEGACY_SCENARIOS vs. V2_SCENARIOS.** `make_golden.py` splits its scenario table
in two: `LEGACY_SCENARIOS` (the five above — a legacy-dict override on
`base_settings()`, migrated via `migrate_dict`, runnable by the frozen v1 oracle)
and `V2_SCENARIOS` (name → a committed `tests/golden/scenarios/<name>.toml` file,
loaded via `load_toml`; empty until the first feature ticket that needs one).
`SCENARIOS = {**LEGACY_SCENARIOS, **V2_SCENARIOS}` is the union `test_golden.py`
parametrises over, but `make_golden.py`'s own `run_all()` (the legacy oracle) only
ever iterates `LEGACY_SCENARIOS` — a v2 scenario's settings have no legacy-dict
shape at all. `golden_newcore.py::run_scenario` dispatches per scenario: legacy
entries still go through `migrate_dict` exactly as before (byte-identical), v2
entries load their own TOML file. There is deliberately no override hook layered
on top of `migrate_dict` for v2 scenarios (a `V2_OVERRIDES`-style shortcut) — a v2
scenario is its own committed TOML file, not a legacy dict with v2-only fields
bolted on.

## Tolerance

`rtol = 1e-9`, `atol = 1e-12` (see `test_golden.py`). The refactor is a
restructuring, not a numerical change, so results should match to floating-point
reassociation. A legitimate, explained numerical change requires regenerating the
reference with `--write` and justifying the diff in the PR.

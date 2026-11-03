# Mutation testing of the calculation core

The golden suite pins the engine's output and the unit suite is large, but neither says
which changes to the engine would go unnoticed. Mutation testing measures exactly that:
it makes thousands of small changes (mutants) to the code, one at a time, and runs the
tests against each. A mutant the tests catch is *killed*; one they do not is a
*survivor*, and every survivor is either harmless or a hole in the tests.

## Running it

```bash
pip install -e '.[gui,dev]' mutmut          # mutmut 3.8 or later
mutmut run "respmech.core.entropy.*"        # one module at a time; omit the pattern for all
mutmut results                              # survivors; `mutmut show <name>` gives the diff
```

The configuration is the `[tool.mutmut]` section of `pyproject.toml`: only the engine
modules are mutated (`compute`, `emg`, `entropy`, `noise`, `analysis/segments`,
`analysis/manoeuvres`), and the test command is the golden suite plus the unit tests that
exercise those modules. The full `tests/unit` is deliberately not used: it is slow and
mostly GUI, and the GUI tests add nothing for engine mutants. mutmut works in a copy under
`mutants/` (gitignored) and keeps its results there, so a run can be stopped and resumed
module by module. Changing the test selection in `pyproject.toml` invalidates the stored
results.

mutmut picks the tests to run for each mutant from coverage it records first. That is
why the golden fixture runs each scenario lazily, the first time a test asks for it
(`tests/golden/test_golden.py`): when every scenario was computed up front in the
fixture, the whole engine was attributed to the first scenario's test, and the other
golden tests were never run against a mutant they would have caught (a sign flip in
`sample_entropy` survived mutmut but failed `entropy_on_volume` when run by hand).

## How survivors are classified

- **Equivalent**: the mutant cannot change any result the analysis produces, either
  because it is a no-op (`dtype=None` on data that is already float, `endpoint=None`,
  a sort kind), because it only touches performance (block sizes, search padding that
  is widened to a superset by design), or because it is unreachable with the values the
  engine actually passes (for instance `n_grad_freq=0` makes the frequency half of the
  noise smoothing filter a single 1, whatever the mutant does to it).
- **Not covered**: a real behaviour no test checks. Each one either gets a test that
  kills it, or is listed below with the reason it was left.
- **False survivor**: a test that would kill it exists but did not run against the
  mutant (the golden fixture case above).

## Status

| Module | Mutants | Killed | Survived | Score | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `core/entropy.py` | 364 | 302 | 62 | 83.0 % | `sample_entropy` alone: 182 of 214 (85.0 %); one timeout counted as killed |
| `core/noise.py` | 527 | 440 | 87 | 83.5 % | after `test_noise_characterisation.py` |
| `core/compute.py` | 2552 | 1934 | 618 | 75.8 % | 31 of the survivors have no covering test; two timeouts counted as killed; survivors not yet classified |
| `core/emg.py` | 1677 | 1298 | 379 | 77.4 % | after `test_emg_characterisation.py` and adding `test_emg_suggest.py` (42.4 % before) |
| `analysis/segments.py` | 1049 | 939 | 110 | 89.5 % | after `test_segments_characterisation.py` (83.6 % before) |
| `analysis/manoeuvres.py` | 570 | 550 | 20 | 96.5 % | after `test_manoeuvres_characterisation.py` (79.8 % before) |

### `core/entropy.py`

Before any new test, with the golden scenarios computed up front, `sample_entropy` had
72 survivors out of 214 mutants. Running the scenarios lazily removed 20 false survivors,
and the new tests below another 20, leaving 32.
`tests/unit/test_sample_entropy_reference.py` then compares the whole result with the
original per-template loop on integer series that put differences exactly on the
tolerance, and covers the default tolerance, the shortest series, a lone surviving pair,
a match ending on the last sample, the block split above a million candidate pairs and
the cancel checkpoint.

What survives in `sample_entropy` is equivalent: the search padding and its sign (the
candidate range is a superset by design and every candidate is re-tested with the
original predicate), the sort kind, the block size and block bookkeeping (any split
covers every template exactly once), and integer `dtype` defaults on a 64-bit platform.

The other functions in the module (multiscale, permutation and Shannon entropy, the
embedding helper) are vendored and not on the analysis path; their survivors are left.

### `core/noise.py`

No golden scenario switches noise reduction on, so the threshold, the mask smoothing, the
reconstruction and the selection report were barely constrained.
`tests/unit/test_noise_characterisation.py` pins the current numbers on one synthetic
EMG-like signal (rtol 1e-9). What survives:

- equivalent at the engine's settings: the frequency half of the smoothing filter
  (`n_grad_freq=0`), multiplying or dividing by `n_std_thresh=1.0`, `endpoint=None`,
  keyword arguments equal to their defaults, `dtype=None`, the `±1e-30` guards and the
  `denom < 0` guard on a power that cannot be negative, the rounding of the grid;
- equality at exact float ties (`<` against `<=` on dB values, `>=` against `>` on the
  fidelity target), which a real recording does not hit;
- not covered, left on purpose: the wording and threshold of the "too few STFT frames"
  warning and the exact boundary of the "reference too short" error.

### `core/emg.py`

The golden suite only reaches the default RMS window and the ECG subtraction it uses, so
the first run killed 711 of 1677 mutants (42.4 %). 647 of the holes were in
`suggest_ecg_settings`, whose own tests (`test_emg_suggest.py`) were not in the
selection; the rest were the fields, messages and boundaries of `detection_quality` and
`gated_peak_rms`, single-channel input to the gate, the short-input guards of
`rolling_rms` and `peak_window_rms`, `timeshift` at its edges, and the degenerate and
clipped-tail windows of `calculate_rms`. `tests/unit/test_emg_characterisation.py`
covers those from the docstrings (the exact 40 % survival and 0.4 s island, the 2 %
long-interval limit, the 10 % ceiling margin, the gate of `round(gate * fs)` samples on
both sides of a peak) and pins the auto-suggested settings on four fixed signals. With
both files in the selection 1298 of 1677 are killed (77.4 %).

The 379 survivors are not classified one by one yet. Most sit in `suggest_ecg_settings`
(167) and in the candidate narrowing of the two ECG template fits (`_shift_candidates`,
`_amplitude_candidates`, `_ssd_band` and the two sweeps, about 120): those helpers only
narrow the candidate list and the original loop still picks the winner, so a mutant
that widens the list is equivalent by design, while one that narrows it wrongly is a real
hole. Telling the two apart is the next step.

### `analysis/segments.py`

The existing segmentation tests check how many segments come out and how long they
are. Before any new test, 877 of 1049 mutants were killed (83.6 %): nothing checked a
segment's `name`, `breathcnt` or `filename`, the `ignored`/`kind` lookups for any segment
but number 1, the slicing of the entropy columns, the treatment of non-finite samples in
`rms_envelope` and `_detrend`, the boundaries of the fixed-window and separator rules,
or the onsets and offsets of the burst detector.
`tests/unit/test_segments_characterisation.py` compares `rms_envelope` and `_detrend`
with a per-sample loop written from their docstrings (dropouts at both ends and inside,
windows from 1 sample to longer than the signal), checks every field of the segments the
four methods return, the window that exactly fills the recording, the separator on the
last sample, and pins the burst detector on two synthetic recordings: separated bursts on
four channels (one all NaN, one noise) and crowded bursts with a burst from the first
sample, an augmenting burst and a spike exactly `burst_min_s` long.

After these tests 939 of 1049 are killed (89.5 %). What survives is equivalent or left
on purpose:

- equivalent: `dtype` defaults on data that is already float or integer, `reshape(None)`
  on data that is already 1-D, slicing a 2-D matrix with or without `, :`, `numpy.errstate`
  arguments (they only silence warnings), an unused variable (`n` in `detect_bursts`), a
  slice end of `None` instead of the recording length, the upper clip of a window index
  that never exceeds it, the `max(1, ...)` floor of windows that are never shorter than
  two samples at the engine's settings, the merge loop for a single burst, the scaling of
  the power trace (the edge refinement uses a level relative to its own peak), defaults
  of `dict.get` and keyword arguments that every caller passes, and the `nan`/`inf`
  spellings `float()` accepts in any case;
- not covered, left on purpose: exact float ties in the hysteresis, contrast, merge and
  minimum-duration comparisons; the lower clip of the activation traces, which only
  matters when one of several channels dips below its own resting level while another is
  near the threshold; a 21st instead of 20th percentile for the power baseline; the
  file name in the error raised when `emg_burst` finds no bursts; and those parts of the
  edge refinement (`_refine_edges`) that only matter when two bursts are so close that
  their search windows overlap. The refinement is pinned on the two recordings above, and
  its constants are starting values the decisions log says will be measured on real tidal
  EMG before they are frozen.

### `analysis/manoeuvres.py`

Before any new test, 455 of 570 mutants were killed (79.8 %). The holes were real: every
IC test used an end-expiratory volume of 0 (so `max - eelv` and `max + eelv` agreed), the
Poes limb of the low-effort check never fired, no sniff test had an expiration that went
further than its inspiration, the per-channel EMG reference was only checked for being
positive, no FVC expiration started after 0 s, and no repeatability test used the median
or a centre of exactly 1. `tests/unit/test_manoeuvres_characterisation.py` covers each
of these from the documented definitions, including the exact boundaries (half the tidal
median, the EELV tolerance, the 1 s FVC minimum, the repeatability fraction).

After these tests 550 of 570 are killed (96.5 %). What survives is equivalent: `dtype`
defaults on data that is already float, defaults of
`dict.get` for keys every breath dict carries, a typed breath that is never in its own
tidal list (`<` against `<=` on the breath number), a tidal median of exactly 0, a
`sd >= 0` guard whose ratio is then 0, and the starting value of the "longest so far" in
`suggest_fvc`, which any duration beats. Left uncovered: whether a tidal breath whose
inspiration has a single Poes sample (or none, which segmentation never produces) takes
part in the Poes median of the low-effort check.

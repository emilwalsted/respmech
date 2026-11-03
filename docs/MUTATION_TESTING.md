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
| `core/compute.py` | | | | | not run yet |
| `core/emg.py` | | | | | not run yet |
| `analysis/segments.py` | | | | | not run yet |
| `analysis/manoeuvres.py` | | | | | not run yet |

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

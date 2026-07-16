"""Coverage for the vendored entropy helpers (pyentrp-derived). Only ``sample_entropy``
is on the analysis path; the rest (multiscale / permutation / composite) shipped
untested (audit #30). These are characterisation tests — they pin basic invariants so
the helpers are exercised, not silently rotting."""
import numpy as np

from respmech.core import entropy as ent


def _rng():
    # deterministic pseudo-signal without Math.random / global seed leakage
    x = np.linspace(0, 20 * np.pi, 2000)
    return np.sin(x) + 0.1 * np.sin(7.3 * x)


def test_shannon_entropy_nonnegative_and_higher_for_more_symbols():
    assert ent.shannon_entropy([1, 1, 1, 1]) == 0.0                 # one symbol → zero
    assert ent.shannon_entropy([1, 2, 3, 4]) > ent.shannon_entropy([1, 1, 2, 2])


def test_sample_entropy_finite_and_length_matches_sample_length():
    se = ent.sample_entropy(_rng(), 3, tolerance=0.2)
    assert len(se) == 3 and np.all(np.isfinite(se))


def test_permutation_entropy_bounds():
    pe = ent.permutation_entropy(_rng(), order=3, delay=1, normalize=True)
    assert 0.0 <= pe <= 1.0                                          # normalised → [0, 1]
    assert ent.permutation_entropy(np.arange(100), order=3) < 0.1    # monotonic → near-zero


def test_multiscale_permutation_entropy_returns_one_value_per_scale():
    mspe = ent.multiscale_permutation_entropy(_rng(), m=3, delay=1, scale=4)
    assert len(mspe) == 4 and np.all(np.isfinite(mspe))


def test_multiscale_entropy_runs_over_scales():
    mse = ent.multiscale_entropy(_rng(), sample_length=2, tolerance=0.2, maxscale=3)
    assert len(mse) == 3

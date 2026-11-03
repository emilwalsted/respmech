"""``correctdrift`` must stay bit-identical to the original per-sample loop.

The vectorised implementation relies on ``ufunc.accumulate`` adding strictly in
sequence; these cases compare whole output arrays byte for byte against the loop it
replaced, across dtypes and lengths (including a recording-sized one).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from respmech.core.compute import correctdrift


def _reference(volume):
    xno = len(volume) - 1
    a = ((volume[xno]) - volume[0]) / xno
    val = a
    corvol = np.zeros(len(volume))
    for i in range(0, xno):
        corvol[i] = volume[i] + val
        val = val - a
    return corvol


_RNG = np.random.default_rng(20261103)
_CASES = {
    "two samples": np.array([0.25, -1.5]),
    "three samples": np.array([1.0, 2.0, 4.0]),
    "random walk": _RNG.normal(size=10_001).cumsum(),
    "drifting volume": np.linspace(0.0, -0.4, 50_000) + _RNG.normal(size=50_000) * 1e-3,
    "float32": _RNG.normal(size=20_000).astype(np.float32),
    "int64": np.arange(5_000, dtype=np.int64) * 3,
    "flat": np.full(1_000, 0.7),
    "long": _RNG.normal(size=2_000_000).cumsum() * 1e-3,
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_matches_original_loop_bit_for_bit(name):
    v = _CASES[name]
    got = correctdrift(v, SimpleNamespace())
    want = _reference(v)
    assert got.dtype == want.dtype
    assert got.tobytes() == want.tobytes()


def test_last_sample_stays_zero():
    out = correctdrift(np.array([1.0, 2.0, 3.0, 4.0]), SimpleNamespace())
    assert out[-1] == 0.0
    assert np.array_equal(out[:-1], [2.0, 2.0, 2.0])


def test_single_sample_is_unchanged_behaviour():
    with np.errstate(divide="ignore", invalid="ignore"):
        got = correctdrift(np.array([3.0]), SimpleNamespace())
    assert got.tolist() == [0.0]

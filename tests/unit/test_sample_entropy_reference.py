"""``sample_entropy`` against the original pyEntropy per-template loop, value for value.

The shipped implementation replaced that loop with a sorted-range lookup that is
argued to be bit-identical (see the comment in ``core/entropy.py``). The golden suite
only ever looks at the LAST element of the result, at one tolerance per run, on
smooth signals, so several ways of breaking the rewrite went unnoticed by every test
(found by mutation testing): the length-0 normalisation that only reaches
``sampen[0]``, a ``<`` relaxed to ``<=`` (only visible when a difference equals the
tolerance exactly), the default tolerance, the shortest series that still has a
template, and the block split that only starts beyond a million candidate pairs.

Each case below compares the whole returned array with the original algorithm, kept
here verbatim as the reference. Integer-valued series with an integer tolerance make
the comparisons exact and put many differences precisely ON the tolerance.
"""
import numpy as np
import pytest

from respmech.core import entropy as ent


def _reference(time_series, sample_length, tolerance=None):
    """The pyEntropy loop the fast version replaced (one template at a time)."""
    M = sample_length - 1
    time_series = np.array(time_series)
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)
    n = len(time_series)
    Ntemp = np.zeros(M + 2)
    Ntemp[0] = n * (n - 1) / 2
    for i in range(n - M - 1):
        template = time_series[i:(i + M + 1)]
        rem_time_series = time_series[i + 1:]
        searchlist = np.nonzero(np.abs(rem_time_series - template[0]) < tolerance)[0]
        go = len(searchlist) > 0
        length = 1
        Ntemp[length] += len(searchlist)
        while go:
            length += 1
            nextindxlist = searchlist + 1
            nextindxlist = nextindxlist[nextindxlist < n - 1 - i]
            nextcandidates = rem_time_series[nextindxlist]
            hitlist = np.abs(nextcandidates - template[length - 1]) < tolerance
            searchlist = nextindxlist[hitlist]
            Ntemp[length] += np.sum(hitlist)
            go = any(hitlist) and length < M + 1
    return -np.log(Ntemp[1:] / Ntemp[:-1])


def _both(series, sample_length, tolerance=None):
    with np.errstate(divide="ignore", invalid="ignore"):
        got = ent.sample_entropy(series, sample_length, tolerance)
        want = _reference(series, sample_length, tolerance)
    return got, want


def _integer_walk(n, seed):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.integers(-2, 3, size=n)).astype(float)


@pytest.mark.parametrize("sample_length", [2, 3, 4])   # the loop itself cannot do 1
@pytest.mark.parametrize("tolerance", [1.0, 2.0])
def test_whole_array_matches_reference_with_differences_on_the_tolerance(sample_length, tolerance):
    series = _integer_walk(300, seed=sample_length)
    got, want = _both(series, sample_length, tolerance)
    assert len(got) == sample_length
    np.testing.assert_array_equal(got, want)


def test_first_element_uses_the_n_choose_2_normalisation():
    series = _integer_walk(120, seed=11)
    got, _ = _both(series, 2, 1.0)
    n = len(series)
    pairs = np.abs(series[:, None] - series[None, :]) < 1.0
    matches_len1 = np.triu(pairs, k=1)[: n - 2].sum()   # templates start below n - M - 1
    assert got[0] == -np.log(matches_len1 / (n * (n - 1) / 2))


def test_default_tolerance_is_a_tenth_of_the_standard_deviation():
    series = np.sin(np.linspace(0, 12 * np.pi, 400)) + np.linspace(0, 1, 400)
    got_default, want = _both(series, 3)
    explicit = ent.sample_entropy(series, 3, 0.1 * np.std(series))
    np.testing.assert_array_equal(got_default, want)
    np.testing.assert_array_equal(got_default, explicit)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_shortest_series_match_reference(n):
    # sample_length 2 gives M = 1: n = 3 leaves exactly one template start.
    series = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])[:n]
    got, want = _both(series, 2, 0.5)
    np.testing.assert_array_equal(got, want)


def test_a_single_surviving_pair_is_still_counted_at_every_length():
    # Exactly one pair, (4, 5), matches at length 1 and keeps matching at lengths 2
    # and 3. Tolerance is not transitive (0.4 steps against 0.5), and the later
    # samples are past the last template start, so no other pair is ever counted.
    series = np.array([100.0, 200.0, 300.0, 400.0, 0.0, 0.4, 0.8, 1.2])
    got, want = _both(series, 3, 0.5)
    np.testing.assert_array_equal(got, want)
    assert np.all(np.isfinite(got))


def test_match_running_into_the_end_of_the_series():
    # The second occurrence of the pattern ends on the last sample.
    series = np.array([5.0, 1.0, 2.0, 3.0, 9.0, 7.0, 1.0, 2.0, 3.0])
    got, want = _both(series, 3, 0.5)
    np.testing.assert_array_equal(got, want)


def test_the_only_pair_ends_on_the_last_sample():
    # The one length-1 match pairs the first sample with the last, so it cannot be
    # extended: the function must still return the array, with no length-2 match.
    series = np.array([5.0, 1.0, 2.0, 9.0, 7.0, 5.0])
    got, want = _both(series, 2, 0.5)
    assert got is not None
    np.testing.assert_array_equal(got, want)


def test_cancel_check_is_consulted_inside_the_counting():
    from respmech.core._cancel import Cancelled

    with pytest.raises(Cancelled):
        ent.sample_entropy(_integer_walk(200, seed=3), 2, 1.0, cancel_check=lambda: True)
    # ...and a checker that never fires changes nothing.
    got = ent.sample_entropy(_integer_walk(200, seed=3), 2, 1.0, cancel_check=lambda: False)
    np.testing.assert_array_equal(got, ent.sample_entropy(_integer_walk(200, seed=3), 2, 1.0))


def test_more_than_a_million_candidate_pairs_is_split_into_blocks_without_changing_counts():
    # A near-constant series makes almost every pair a candidate: about 2.2 million
    # here, so the counting runs in several blocks.
    rng = np.random.default_rng(5)
    series = rng.integers(0, 3, size=2100).astype(float)
    got, want = _both(series, 3, 2.0)
    np.testing.assert_array_equal(got, want)

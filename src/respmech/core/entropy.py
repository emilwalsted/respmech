# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np
from math import factorial

from respmech.core._cancel import check


def _embed(x, order=3, delay=1):
    """Time-delay embedding.

    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.

    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series

    Args:
        time_series: Time series
        scale: Scale factor

    Returns:
        Vector of coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    temp = np.reshape(time_series[0:b*scale], (b, scale))
    cts = np.mean(temp, axis = 1)
    return cts


def shannon_entropy(time_series):
    """Return the Shannon Entropy of the sample data.

    Args:
        time_series: Vector or string of the sample data

    Returns:
        The Shannon Entropy as float value
    """

    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    ent = -ent
    return ent


def sample_entropy(time_series, sample_length, tolerance = None, cancel_check = None):
    """Calculates the sample entropy of degree m of a time_series.

    This method uses chebychev norm.
    It is quite fast for random data, but can be slower is there is
    structure in the input time series.

    Args:
        time_series: numpy array of time series
        sample_length: length of longest template vector
        tolerance: tolerance (defaults to 0.1 * std(time_series)))
    Returns:
        Array of sample entropies:
            SE[k] is ratio "#templates of length k+1" / "#templates of length k"
            where #templates of length 0" = n*(n - 1) / 2, by definition
    Note:
        The parameter 'sample_length' is equal to m + 1 in Ref[1].


    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
    """
    #The code below follows the sample length convention of Ref [1] so:
    M = sample_length - 1;

    time_series = np.array(time_series)
    if tolerance is None:
        tolerance = 0.1*np.std(time_series)

    n = len(time_series)
    x = time_series

    #Ntemp is a vector that holds the number of matches. N[k] holds matches templates of length k
    Ntemp = np.zeros(M + 2)
    #Templates of length 0 matches by definition:
    Ntemp[0] = n*(n - 1) / 2

    # --- Sorted-candidate counting -------------------------------------------------
    # This replaces the original per-template scan over the whole tail (O(n^2) element
    # work *and* O(n) Python iterations per template) with a sorted range lookup.
    #
    # WHY THIS IS BIT-IDENTICAL, not merely "within tolerance":
    # ``Ntemp`` accumulates nothing but integer counts, and the only predicate anywhere
    # in this function is ``|a - b| < tolerance`` -- an exact comparison of
    # exactly-computed floats. The sole floating-point arithmetic is the final
    # ``-np.log(Ntemp[1:] / Ntemp[:-1])``, evaluated once from those integers. So as long
    # as we count *the same pairs*, the returned array is identical: there is no
    # summation order to preserve and no reassociation risk.
    #
    # HOW the candidates are found: fl(a - b) is monotone in b, so for a fixed x[i] the
    # match set {j : |x[i] - x[j]| < tolerance} is a contiguous range once x is sorted.
    # ``np.searchsorted`` locates that range. The bounds are computed as x[i] +/- tolerance,
    # which rounds differently from the predicate's x[i] - x[j], so they are widened by a
    # few ULP to guarantee a *superset* -- and every surviving candidate is then re-tested
    # with the original predicate below. Widening can therefore only cost a few wasted
    # candidates; it can never change a count, which is why the constant need not be tight.
    imax = n - M - 1              # template starts -- exactly the old range(n - M - 1)
    if imax > 0:
        order = np.argsort(x, kind="stable")
        xs = x[order]
        xi = x[:imax]
        pad = 16.0 * np.spacing(np.abs(xi) + abs(tolerance) + 1.0)
        lo = np.searchsorted(xs, xi - tolerance - pad, side="left")
        hi = np.searchsorted(xs, xi + tolerance + pad, side="right")
        counts = (hi - lo).astype(np.int64)
        ends = np.cumsum(counts)

        # Work in blocks bounded by candidate-pair count so peak memory stays predictable
        # on pathological (near-constant) signals, where every pair matches. ~48 MB peak.
        MAXPAIRS = 1_000_000
        blocks = [0]
        while blocks[-1] < imax:
            a = blocks[-1]
            base = ends[a - 1] if a > 0 else 0
            b = int(np.searchsorted(ends, base + MAXPAIRS, side="right"))
            blocks.append(min(max(b, a + 1), imax))

        for bi in range(len(blocks) - 1):
            # Cooperative abort point (no-op when cancel_check is None -> golden-safe).
            # Granularity is now per block rather than per template; for breath-length
            # calls that is a single block, i.e. ~1.5 ms of work -- still far finer than
            # the per-file granularity the GUI needs. See tests/unit/test_cancel.py.
            check(cancel_check)
            a, b = blocks[bi], blocks[bi + 1]
            c = counts[a:b]
            tot = int(c.sum())
            if tot == 0:
                continue
            # Expand the per-template candidate ranges into one flat (owner, candidate)
            # pair list, then map back through ``order`` to original indices.
            owner = np.repeat(np.arange(a, b, dtype=np.int64), c)
            off = np.cumsum(c) - c
            flat = (np.arange(tot, dtype=np.int64) - np.repeat(off, c)
                    + np.repeat(lo[a:b].astype(np.int64), c))
            j = order[flat]
            keep = j > owner                             # the old loop only looked forward
            i_ = owner[keep]
            j_ = j[keep]
            live = np.abs(x[i_] - x[j_]) < tolerance     # THE original predicate
            i_ = i_[live]
            j_ = j_[live]
            Ntemp[1] += i_.size
            # Chain to longer templates exactly as the original did: a length-L match
            # requires a length-(L-1) match, so the surviving pair list only shrinks --
            # which is also why the original's ``go = any(hitlist) and ...`` early exit is
            # equivalent to the candidate list becoming empty.
            for L in range(2, M + 2):
                if i_.size == 0:
                    break
                ok = (j_ + (L - 1)) <= (n - 1)           # old: nextindxlist < n - 1 - i
                i_ = i_[ok]
                j_ = j_[ok]
                if i_.size == 0:
                    break
                live = np.abs(x[i_ + (L - 1)] - x[j_ + (L - 1)]) < tolerance
                i_ = i_[live]
                j_ = j_[live]
                Ntemp[L] += i_.size

    sampen =  - np.log(Ntemp[1:] / Ntemp[:-1])
    return sampen


def multiscale_entropy(time_series, sample_length, tolerance = None, maxscale = None):
    """Calculate the Multiscale Entropy of the given time series considering
    different time-scales of the time series.

    Args:
        time_series: Time series for analysis
        sample_length: Bandwidth or group of points
        tolerance: Tolerance (default = 0.1*std(time_series))

    Returns:
        Vector containing Multiscale Entropy

    Reference:
        [1] http://en.pudn.com/downloads149/sourcecode/math/detail646216_en.html
    """

    if tolerance is None:
        #we need to fix the tolerance at this level. If it remains 'None' it will be changed in call to sample_entropy()
        tolerance = 0.1*np.std(time_series)
    if maxscale is None:
        maxscale = len(time_series)

    mse = np.zeros(maxscale)

    for i in range(maxscale):
        temp = util_granulate_time_series(time_series, i+1)
        mse[i] = sample_entropy(temp, sample_length, tolerance)[-1]
    return mse


def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    """Permutation Entropy.

    Parameters
    ----------
    time_series : list or np.array
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    pe : float
        Permutation Entropy

    References
    ----------
    .. [1] Massimiliano Zanin et al. Permutation Entropy and Its Main
        Biomedical and Econophysics Applications: A Review.
        http://www.mdpi.com/1099-4300/14/8/1553/pdf

    .. [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural
        complexity measure for time series.
        http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf

    Notes
    -----
    Last updated (Oct 2018) by Raphael Vallat (raphaelvallat9@gmail.com):
    - Major speed improvements
    - Use of base 2 instead of base e
    - Added normalization

    Examples
    --------
    1. Permutation entropy with order 2

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value between 0 and log2(factorial(order))
        >>> print(permutation_entropy(x, order=2))
            0.918

    2. Normalized permutation entropy with order 3

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(permutation_entropy(x, order=3, normalize=True))
            0.589
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def multiscale_permutation_entropy(time_series, m, delay, scale):
    """Calculate the Multiscale Permutation Entropy

    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay
        scale: Scale factor

    Returns:
        Vector containing Multiscale Permutation Entropy

    Reference:
        [1] Francesco Carlo Morabito et al. Multivariate Multi-Scale Permutation Entropy for
            Complexity Analysis of Alzheimer’s Disease EEG. www.mdpi.com/1099-4300/14/7/1186
        [2] http://www.mathworks.com/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe/content/MPerm.m
    """
    mspe = []
    for i in range(scale):
        coarse_time_series = util_granulate_time_series(time_series, i + 1)
        pe = permutation_entropy(coarse_time_series, order=m, delay=delay)
        mspe.append(pe)
    return mspe


# NOTE: the vendored ``composite_multiscale_entropy`` was removed — it was never called
# and was broken (indexed a shape-(1, scale) array as cmse[i], which raises for i≥1).

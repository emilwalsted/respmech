"""Pure, Qt-free, file-free data-quality predicates on already-loaded arrays.

Both checks are precondition checks on the RAW recorded signal, evaluated independently
of the segmentation/analysis pipeline -- neither ever changes a computed number. They
exist to catch two failure modes that used to surface only as a confusing, misleading
downstream error (or, for a constant flow channel, an outright hang -- see
``core.compute.separateintobreathsbyflow``):

* :func:`detect_merged_time_blocks` -- several recordings (e.g. LabChart blocks)
  exported into one file and merged row-by-row by timestamp, so a segmentation pass
  jumps between different recordings from sample to sample.
* :func:`detect_constant_channel` -- an assigned channel that never varies (a
  grounded/unused input, or the wrong column assigned), which makes the channel's
  own analysis meaningless (and, for flow specifically, makes breath segmentation
  unable to make progress at all).

No pandas/file I/O here on purpose, so ``core.compute`` (deliberately "no file I/O, no
plotting" per its own docstring) can depend on :func:`detect_constant_channel` without
pulling pandas into its import graph. The file-reading probes that feed these two
functions from a path on disk live in ``core.io.loaders`` instead (already pandas-
dependent), and are what ``ui.manifest.build_manifest`` and ``respmech validate`` use
to run these checks over a whole batch cheaply, before or independently of a real run.
"""
from __future__ import annotations

import numpy as np


def detect_merged_time_blocks(time_seconds, *, min_duplicates=3, min_duplicate_fraction=0.02):
    """Return a human message when ``time_seconds`` looks like a regular time axis that
    has been corrupted by duplicated or decreasing timestamps -- the signature of more
    than one recording merged into a single file by timestamp -- else ``None``.

    "Looks like a regular time axis" mirrors ``ui.workers.detect_sampling_frequency``'s
    own criteria (monotonic-in-the-main, a uniform non-integer step close to ``1/fs``),
    computed over the STRICTLY POSITIVE steps only -- deliberately the same subset that
    function's own median/std check uses. The difference is what happens to the
    steps that check discards: ``detect_sampling_frequency`` silently ignores a
    non-positive ``dt`` when inferring the sampling rate (so a column can still read as
    "clean, regular time" there even when riddled with duplicates); this function's whole
    job is to name exactly what that silent filtering hides.

    ``min_duplicates``/``min_duplicate_fraction`` both have to be exceeded (an absolute
    floor AND a relative one) before this reports anything, so neither a single rounding
    glitch in a huge file (fails the fraction test) nor a couple of duplicates in a tiny
    file (fails the absolute-count test) is mistaken for a merged export. Both are
    conservative, documented defaults, not measured against a real reproduction (the
    reported case's own file is local-only and never reaches this repo -- see
    ``CLAUDE.md``) -- widen them if a real recording is ever found to trip this
    spuriously.
    """
    t = np.asarray(time_seconds, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 20:
        return None
    diffs = np.diff(t)
    pos = diffs[diffs > 0]
    if pos.size < 10:
        return None                      # not enough regular structure to call this "time"
    med = float(np.median(pos))
    if med <= 0:
        return None
    if med >= 1.0 and abs(med - round(med)) < 1e-6:
        return None                      # integer steps -> a sample index, not seconds
    if float(np.std(pos)) > 0.25 * med:
        return None                      # irregular spacing -> not a clean sample clock either
    non_increasing = int(np.sum(diffs <= 0))
    if non_increasing == 0:
        return None
    fraction = non_increasing / diffs.size
    if non_increasing < min_duplicates or fraction < min_duplicate_fraction:
        return None
    fs = 1.0 / med
    return (
        f"{non_increasing} of {diffs.size} time steps ({fraction:.1%}) are duplicated or "
        f"decreasing, even though the column otherwise looks like a regular ~{fs:.4g} Hz "
        f"time axis. This is the signature of more than one recording exported into the "
        f"same file and merged row-by-row (e.g. several LabChart blocks) — export each "
        f"block to its own file instead, or the analysis will jump between recordings "
        f"from sample to sample.")


def detect_constant_channel(values, *, tol=1e-9):
    """True when ``values`` (an assigned channel's samples) never varies.

    Ignores non-finite samples (NaN/inf) rather than letting them dominate the
    range check; an all-non-finite input is NOT reported as constant (there is
    nothing to compare -- a genuinely empty/unreadable channel is a different,
    already-reported problem, e.g. ``core.io.loaders.validatedata``'s NaN check).
    ``tol`` is an absolute tolerance on the observed range, not a relative one:
    these are physical channels (flow, pressures, EMG) whose "zero" is meaningful
    in its own physical units, so a relative tolerance would silently exempt a
    channel that only ever reads exactly 0.0 -- precisely the reported case."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool((np.max(arr) - np.min(arr)) <= tol)

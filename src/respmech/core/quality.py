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


def detect_merged_time_blocks(time_seconds, *, min_duplicates=3, min_duplicate_fraction=0.02,
                              min_jump_seconds=1.0, min_jump_multiple=100.0):
    """Return a human message when ``time_seconds`` looks like a regular time axis that
    has been corrupted by more than one recording merged into it, else ``None``. Two
    distinct merge shapes are checked, since a real export can be merged either way:

    * **Interleaved / sorted-by-time** (the reported case: three LabChart blocks sorted
      together by timestamp): the overlapping stretch has each timestamp repeated once
      per still-active block, so a run of DUPLICATED/decreasing steps appears, then
      stops once only one block remains.
    * **Concatenated**: each block simply appended after the previous one, each
      restarting its own time base at (or near) zero -- a single large BACKWARD JUMP at
      each boundary, not a run of duplicates.

    "Looks like a regular time axis" mirrors ``ui.workers.detect_sampling_frequency``'s
    own criteria (monotonic-in-the-main, a uniform non-integer step close to ``1/fs``,
    the same ``10-200000`` Hz plausibility band), computed over the STRICTLY POSITIVE
    steps only -- deliberately the same subset that function's own median/std check
    uses, so the two functions never disagree about what counts as "looks like time" in
    the first place. The difference is what happens to the steps that check discards:
    ``detect_sampling_frequency`` silently ignores a non-positive ``dt`` when inferring
    the sampling rate (so a column can still read as "clean, regular time" there even
    when riddled with duplicates or containing a block boundary); this function's whole
    job is to name exactly what that silent filtering hides.

    ``min_duplicates``/``min_duplicate_fraction`` both have to be exceeded (an absolute
    floor AND a relative one) before the interleaved shape is reported, so neither a
    single rounding glitch in a huge file nor a couple of duplicates in a tiny file is
    mistaken for a merged export. On top of that, the interleaved shape additionally
    requires the LAST ~10% of the file to look clean BY THE SAME BAR: a genuine merge
    ends when only one block is still running, so its own tail has few or no
    duplicates, whereas a column that is merely printed at coarser precision than the
    true sampling interval (e.g. 3 decimal places at 2000 Hz, giving pairs of samples
    that print identically) shows the SAME duplicate rate throughout the whole file,
    tail included -- verified empirically to tell the two apart (a 2000 Hz column
    rounded to 3 dp gives a uniform ~50% duplicate rate end to end and is correctly
    left unflagged; the reported case's own shape has a duplicate-free tail). The
    concatenated shape's jump floor (``max(min_jump_seconds, min_jump_multiple * step)``)
    is deliberately large relative to the sample period, so it fires on nothing smaller
    than an actual restarted time base.

    None of these thresholds were measured against a real reproduction (the reported
    case's own file is local-only and never reaches this repo -- see ``CLAUDE.md``) --
    revisit them if a real recording is ever found to trip this spuriously, or to slip
    past it.
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
    fs = 1.0 / med
    if not (10.0 <= fs <= 200000.0):
        return None                      # implausible as a real sample clock either way

    jump_floor = max(min_jump_seconds, min_jump_multiple * med)
    big_jump_mask = diffs <= -jump_floor
    if bool(np.any(big_jump_mask)):
        idx = int(np.argmax(big_jump_mask))
        return (
            f"the time column jumps backward by {abs(diffs[idx]):.3g} s at sample "
            f"{idx + 1} (t≈{t[idx]:.2f} s), even though it otherwise looks like a "
            f"regular ~{fs:.4g} Hz time axis. This is the signature of more than one "
            f"recording exported into the same file, each restarting its own time base "
            f"— export each block to its own file instead, or the analysis will jump "
            f"between recordings from sample to sample.")

    non_increasing = int(np.sum(diffs <= 0))
    if non_increasing == 0:
        return None
    fraction = non_increasing / diffs.size
    if non_increasing < min_duplicates or fraction < min_duplicate_fraction:
        return None
    tail_size = min(diffs.size, max(50, diffs.size // 10))
    tail_fraction = float(np.sum(diffs[-tail_size:] <= 0)) / tail_size
    if tail_fraction >= min_duplicate_fraction:
        return None                      # a uniform rate end to end (e.g. coarse timestamp
                                          # rounding), not a merge whose tail runs clean
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

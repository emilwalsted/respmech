"""EMG-only segmentation: split a recording that has no flow/pressure signal into
segments a downstream EMG/entropy analysis can run against, instead of the usual
inspiration/expiration breath split (which needs a flow channel to detect at all).

Four methods, all reachable only for an EMG-only signal set (``Settings.validate()``
already enforces this — a ``method`` from this module and a declared flow channel
never coexist):

* ``whole_file`` — the entire recording is one segment, always numbered 1. The
  natural choice for a single maximal manoeuvre (a sniff, a maximal voluntary
  contraction) captured as its own short recording.
* ``separators`` — N user-placed times split the recording into N+1 segments,
  numbered from 1. Zero separators is the same shape as ``whole_file`` (one
  segment), just reached explicitly through this method instead.

Two automatic alternatives, for recordings with nothing to place separators on:

* ``fixed_windows`` — equal windows of ``window_s`` seconds every ``hop_s`` seconds
  (default 5 s / 5 s, a plain tiling). Phase-less like the two above; a trailing piece
  shorter than a window is dropped, never padded.
* ``emg_burst`` — the recording is cut at the onset of every inspiratory EMG burst that
  :func:`detect_bursts` finds on the envelope of the ECG-removed (not noise-reduced)
  signal. Segment ``k`` runs from burst ``k``'s onset to burst ``k + 1``'s onset (the
  last one to the end of the recording); whatever precedes the first onset belongs to no
  segment. Each segment carries the neural timing of its own cycle
  (:func:`_neural_timing`) and the burst's own position, from which the inter-burst
  periods a noise reference is cut from (:func:`burst_masks`) are derived.

All four build the SAME ``OrderedDict`` shape ``core.compute._make_breath`` builds for a
real breath, so every existing consumer that reads ``breath['time']``/``['emgcols']``/
``['ignored']``/``['kind']``/etc. keeps working unchanged — but with:

* ``has_phases=False`` — there is no inspiration/expiration split to read.
* empty ``flow``/``volume``/``poes``/``pgas``/``pdi`` arrays — the same "absent
  channel" convention ``core/io/loaders.py`` already uses for any channel that is
  not assigned (an EMG-only signal set never has any of these).

This module is deliberately free of any dependency on ``core.compute``/
``core.pipeline`` (it takes plain arrays, not a legacy settings namespace it would
have to import compute.py's helpers to interpret) — ``compute.py`` imports THIS
module, not the other way around, so there is no import cycle. Qt-free.
"""
from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from respmech.core import emg as emglib


class EmgSegmentationError(ValueError):
    """Raised when an EMG-only segmentation method cannot place a segment boundary
    it was asked for — in practice, a ``separators`` time that falls outside the
    recording, or two boundaries that collapse into a zero-length segment. A
    precondition failure, not a bug: reported as a per-file error the same way
    ``compute.VolumeSegmentationError``/``ConstantFlowError`` already are, naming
    the file so a batch of otherwise-fine files is not obscured by one bad entry.
    """


def _slice_cols(cols, start: int, end: int):
    """Slice a (possibly absent) 2-D column matrix — mirrors ``compute._phase_dicts``'s
    own "an absent channel is an empty array/list, never sliced" handling."""
    if len(cols) == 0:
        return cols
    return cols[start:end, :] if np.ndim(cols) > 1 else cols[start:end]


def _make_segment(number: int, start: int, end: int, timecol, emgcolumns, entropycolumns,
                  filename: str, ignored: bool, kind: str | None) -> OrderedDict:
    # reshape(-1), not squeeze(): squeeze collapses a genuine 1-SAMPLE segment's (1,)
    # array to a 0-d scalar (no len(), no indexing), which every downstream consumer of
    # breath["time"] (build_processed_data, the diagnostic plots, this module's own
    # pipeline.py caller) assumes is at least 1-D. reshape(-1) still flattens an
    # incoming (n, 1) column-vector shape the same way squeeze did, without that
    # collapse at n=1.
    time = np.asarray(timecol[start:end]).reshape(-1)
    empty = np.array([])
    return OrderedDict([
        ('number', number),
        ('name', f'Segment #{number}'),
        ('time', time),
        ('flow', empty),
        ('volume', empty),
        ('poes', empty),
        ('pgas', empty),
        ('pdi', empty),
        ('breathcnt', number),
        ('ignored', ignored),
        ('kind', kind),
        ('has_phases', False),
        ('entcols', _slice_cols(entropycolumns, start, end)),
        ('emgcols', _slice_cols(emgcolumns, start, end)),
        ('filename', filename),
    ])


def _attach_whole_file_rms_diagnostics(segment: OrderedDict, rms_s: float, fs: float) -> None:
    """``rms_file_max_col_n``/``t_rms_file_max_col_n``/``rms_file_top3_col_n`` —
    whole-file-only diagnostics that let a reader spot WHERE in a long recording the
    peak EMG activity fell, and whether it was a single spike or sustained (top-3
    average close to the max) — neither of which the ordinary per-segment RMS
    (``compute.compute_segment_emg``'s ``rms``/``rms_max``/``rms_mean``, computed the
    same way for this segment as for any breath) can distinguish on its own.

    Built from :func:`emg.rolling_rms`, the exact sliding-window grid
    :func:`emg.calculate_rms` already maximises over (so ``rms_file_max_col_n`` is
    NOT a second, differently-computed peak estimate — it is the same number
    ``rms_col_n``'s own max would report, just also timestamped). ``rms_file_top3_col_n``
    is the mean of the three highest values in that same envelope — a steadier
    "peak level" than the single max, which a lone noise spike could otherwise
    dominate.

    No-op when there are no EMG columns (nothing to diagnose) or the window is too
    short for the envelope to exist at all (``rolling_rms`` returns empty arrays).

    nan-aware: ``rolling_rms``'s cumulative-sum grid means a single NaN sample poisons
    every window from that sample ONWARD, not just the ones directly overlapping it
    (a NaN in a cumulative sum propagates forward forever). A real peak earlier in the
    recording is still recovered correctly; a channel with no NaN-free window left at
    all reports NaN for all three values, in matching pairs — never a concrete-looking
    ``t_rms_file_max`` alongside a NaN ``rms_file_max``, which is a wrong answer that
    looks right and worse than one that is visibly missing.
    """
    emgcols = segment['emgcols']
    if len(emgcols) == 0:
        return
    n_ch = np.asarray(emgcols).shape[1]
    rms_max, t_rms_max, rms_top3 = [], [], []
    for ch in range(n_ch):
        values, starts = emglib.rolling_rms(np.asarray(emgcols)[:, ch], rms_s, fs)
        # nan-aware throughout: a plain argmax/sort would let a single NaN window (e.g.
        # from upstream noise reduction touching the edge of the recording) pick a NaN
        # as the "peak" while still reporting a concrete, plausible-looking t_rms_max —
        # a wrong answer that looks right, worse than a value that is visibly NaN.
        if values.size == 0 or np.all(np.isnan(values)):
            rms_max.append(float('nan'))
            t_rms_max.append(float('nan'))
            rms_top3.append(float('nan'))
            continue
        peak_ix = int(np.nanargmax(values))
        rms_max.append(float(values[peak_ix]))
        t_rms_max.append(float(starts[peak_ix]) / fs)
        finite = values[~np.isnan(values)]
        top3 = np.sort(finite)[-min(3, finite.size):]
        rms_top3.append(float(np.mean(top3)))
    # One value per EMG channel, no appended max/mean summary (unlike compute_segment_emg's
    # rms/intemg families) — results.py::build_breath_table expands each of these three
    # lists into rms_file_max_col_N / t_rms_file_max_col_N / rms_file_top3_col_N, one column
    # per channel number, and nothing else.
    segment['rms_file_max'] = rms_max
    segment['t_rms_file_max'] = t_rms_max
    segment['rms_file_top3'] = rms_top3


#: The five neural-timing columns an ``emg_burst`` segment reports, in output order.
#: Named with an ``_emg`` suffix so they can never be mistaken for the mechanical
#: ``ti``/``te``/``ttot``/``ti_ttot``/``bf`` a flow signal gives (a burst is the neural
#: drive, not airflow). Units are declared explicitly in ``core/analysis/registry.py``.
NEURAL_TIMING_COLUMNS = ("ti_emg", "te_emg", "ttot_emg", "ti_ttot_emg", "bf_emg")

#: Per-file quality columns an ``emg_burst`` segment carries (identical on every segment
#: of a file, so the file's own average row reports them unchanged).
BURST_QC_COLUMNS = ("emg_seg_n_bursts", "emg_seg_burst_frac", "emg_seg_contrast")


@dataclass(frozen=True)
class BurstDetection:
    """Result of :func:`detect_bursts`. ``onsets``/``offsets`` are sample indices into the
    array the detection ran on (``offsets`` exclusive), strictly increasing and
    non-overlapping. ``contrast`` is the median, across the channels that took part, of
    the ratio between the envelope's 95th percentile and its median."""
    onsets: np.ndarray
    offsets: np.ndarray
    contrast: float
    n_channels_used: int


def rms_envelope(x, window: int) -> np.ndarray:
    """Centred moving RMS of a 1-D signal over ``window`` samples, same length as ``x``.
    Non-finite samples contribute nothing to their windows (a window of nothing but
    non-finite samples gives 0), so one bad sample cannot poison everything after it the
    way a plain cumulative sum would. The window shrinks at the two edges instead of
    padding."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    window = max(1, min(int(window), n))
    ok = np.isfinite(x)
    sq = np.where(ok, x * x, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    ccnt = np.concatenate(([0], np.cumsum(ok.astype(np.int64))))
    idx = np.arange(n)
    lo = np.clip(idx - window // 2, 0, n)
    hi = np.clip(idx - window // 2 + window, 0, n)
    cnt = ccnt[hi] - ccnt[lo]
    total = csum[hi] - csum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.sqrt(np.maximum(total, 0.0) / np.maximum(cnt, 1)), 0.0)


def detect_bursts(emgcols, fs: float, *, threshold_frac: float, min_s: float,
                  smooth_s: float, min_contrast: float, filename: str = "") -> BurstDetection:
    """Find the EMG bursts of a tidal recording: on/off detection on the RMS envelope with
    a baseline from the median absolute level and hysteresis thresholds, after Hodges &
    Bui (1996) (threshold-plus-duration onset detection on a smoothed envelope).

    Per channel: the envelope (:func:`rms_envelope`, ``smooth_s`` long) is scaled so that
    its own median (the resting level, as long as the muscle is active for less than half
    of the recording, which tidal breathing satisfies) is 0 and its own 95th percentile is
    1. A channel whose 95th percentile is less than ``min_contrast`` times its median has
    no bursts to find (it is noise) and is left out; if no channel is left, the
    recording fails with :class:`EmgSegmentationError` rather than being cut into
    noise. The remaining channels are averaged into one activation trace ``a``; a burst
    is a stretch with ``a >= threshold_frac / 2`` (the off level) that reaches
    ``threshold_frac`` (the on level) somewhere. Gaps shorter than ``min_s`` between two
    bursts are then bridged, and bursts shorter than ``min_s`` discarded, in that order.

    What is measured and what is not (K-035 lesson): the defaults were exercised on
    synthetic burst trains with a known onset/offset (onsets and offsets recovered within
    2 samples at 2 kHz, envelope smoothing 0.1 s) and on the built-in sample recording's
    EMG channel; no production EMG recording was available. The four parameters are
    starting values until they have been measured on real tidal EMG and frozen in the
    decisions log."""
    x = np.asarray(emgcols, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    where = f" in {filename}" if filename else ""
    win = max(1, int(round(smooth_s * fs)))
    traces, ptraces, contrasts = [], [], []
    for ch in range(x.shape[1]):
        env = rms_envelope(x[:, ch], win)
        if env.size == 0:
            continue
        base = float(np.median(env))
        peak = float(np.percentile(env, 95))
        if peak <= 0 or peak <= base:
            continue
        contrast = peak / base if base > 0 else float("inf")
        if contrast < min_contrast:
            continue
        traces.append(np.clip((env - base) / (peak - base), 0.0, None))
        pw = env * env                       # same scaling on the power envelope, for _refine_edges
        pbase, ppeak = float(np.median(pw)), float(np.percentile(pw, 95))
        ptraces.append(np.clip((pw - pbase) / (ppeak - pbase), 0.0, None))
        contrasts.append(contrast)
    if not traces:
        raise EmgSegmentationError(
            f"No EMG bursts found{where}: the envelope of no channel rises to "
            f"{min_contrast:g} times its resting level, so there is nothing to segment "
            "(lower processing.segmentation.emg.burst_min_contrast only if the bursts "
            "are real, or use whole_file/separators/fixed_windows instead).")
    a = np.mean(traces, axis=0)
    on_level, off_level = threshold_frac, threshold_frac / 2.0

    above = np.concatenate(([False], a >= off_level, [False]))
    edges = np.flatnonzero(above[1:] != above[:-1])
    starts, ends = edges[0::2], edges[1::2]
    keep = np.array([a[s_:e_].max() >= on_level for s_, e_ in zip(starts, ends)], dtype=bool)
    starts, ends = starts[keep], ends[keep]

    min_n = max(1, int(round(min_s * fs)))
    if starts.size > 1:
        merged_s, merged_e = [int(starts[0])], [int(ends[0])]
        for s_, e_ in zip(starts[1:], ends[1:]):
            if s_ - merged_e[-1] < min_n:
                merged_e[-1] = int(e_)
            else:
                merged_s.append(int(s_)); merged_e.append(int(e_))
        starts, ends = np.asarray(merged_s), np.asarray(merged_e)
    long_enough = (ends - starts) >= min_n
    starts, ends = starts[long_enough], ends[long_enough]
    if starts.size == 0:
        raise EmgSegmentationError(
            f"No EMG bursts found{where}: activity above the threshold never lasts "
            f"{min_s:g} s (processing.segmentation.emg.burst_min_s).")
    starts, ends = _refine_edges(np.mean(ptraces, axis=0), starts, ends, win)
    return BurstDetection(starts, ends, float(np.median(contrasts)), len(traces))


def _refine_edges(power, starts, ends, window: int):
    """Move each coarse edge to where the smoothed POWER envelope crosses half of that
    burst's own plateau. A centred window smears a step over ``window`` samples, and the
    on/off levels (a fraction of the peak, chosen to be robust to noise) sit well down
    the ramp, so the coarse edges lead the true ones by up to about half a window; the
    half-power point of a linear ramp is exactly the step. The plateau is the median
    over the middle half of the burst, so a burst weaker than the recording's typical
    one is still located against its own level. The search stays within one window of
    the coarse edge and never beyond the midpoint to the neighbouring burst, so bursts
    cannot swap places or overlap."""
    n = power.size
    new_s, new_e = [], []
    for k, (s_, e_) in enumerate(zip(starts, ends)):
        s_, e_ = int(s_), int(e_)
        m = e_ - s_
        core = power[s_ + m // 4: max(e_ - m // 4, s_ + m // 4 + 1)]
        half = 0.5 * float(np.median(core))
        lo = max(0, s_ - window)
        hi = min(n, e_ + window)
        if k > 0:
            lo = max(lo, (int(ends[k - 1]) + s_) // 2)
        if k + 1 < len(starts):
            hi = min(hi, (e_ + int(starts[k + 1])) // 2)
        mid = s_ + m // 2
        rising = np.flatnonzero(power[lo:mid + 1] >= half)
        falling = np.flatnonzero(power[mid:hi] >= half)
        new_s.append(lo + int(rising[0]) if rising.size else s_)
        new_e.append(mid + int(falling[-1]) + 1 if falling.size else e_)
    return np.asarray(new_s, dtype=int), np.asarray(new_e, dtype=int)


def _neural_timing(onsets, offsets, k: int, fs: float) -> OrderedDict:
    """Neural timing of cycle ``k`` (0-based): ``ti_emg`` = burst duration, ``te_emg`` =
    from this burst's end to the next one's onset, ``ttot_emg`` = onset to onset,
    ``ti_ttot_emg`` = ``ti_emg / ttot_emg`` and ``bf_emg`` = ``60 / ttot_emg`` (min⁻¹).
    The last burst of a recording has no following onset, so everything but ``ti_emg`` is
    NaN there (a truncated final cycle is not a short one)."""
    ti = (offsets[k] - onsets[k]) / fs
    if k + 1 < len(onsets):
        te = (onsets[k + 1] - offsets[k]) / fs
        ttot = ti + te
        return OrderedDict([("ti_emg", ti), ("te_emg", te), ("ttot_emg", ttot),
                            ("ti_ttot_emg", ti / ttot), ("bf_emg", 60.0 / ttot)])
    nan = float("nan")
    return OrderedDict([("ti_emg", ti), ("te_emg", nan), ("ttot_emg", nan),
                        ("ti_ttot_emg", nan), ("bf_emg", nan)])


def fixed_windows(filename: str, timecol, emgcolumns, entropycolumns, fs: float,
                  *, window_s: float, hop_s: float, ignored_breaths: set[int],
                  kinds: dict) -> OrderedDict:
    """Equal windows of ``window_s`` seconds, one every ``hop_s`` seconds from the start
    of the recording, numbered from 1. Only complete windows count: a trailing piece
    shorter than a window is dropped rather than padded (it would bias the RMS and
    entropy of the last segment against the others). A recording shorter than one window
    raises :class:`EmgSegmentationError` naming the file."""
    n = len(np.atleast_1d(timecol))
    win = max(1, int(round(window_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    if win > n:
        raise EmgSegmentationError(
            f"{filename} is {n / fs:.2f} s long, shorter than one {window_s:g} s window "
            "(processing.segmentation.emg.window_s).")
    segments = OrderedDict()
    for number, start in enumerate(range(0, n - win + 1, hop), start=1):
        segments[number] = _make_segment(
            number, start, start + win, timecol, emgcolumns, entropycolumns, filename,
            ignored=number in ignored_breaths, kind=kinds.get(number))
    return segments


def emg_burst(filename: str, timecol, emgcolumns, entropycolumns, detect_emg, fs: float,
              *, threshold_frac: float, min_s: float, smooth_s: float, min_contrast: float,
              ignored_breaths: set[int], kinds: dict) -> OrderedDict:
    """One segment per detected inspiratory EMG burst (see the module docstring for the
    exact extent of a segment). ``detect_emg`` is the signal the bursts are FOUND on:
    the ECG-removed EMG before any noise reduction (a noise profile is itself built from
    the periods between bursts, so detecting on the reduced signal would be circular);
    ``emgcolumns`` is what the segments carry. Both span the whole recording.

    Every segment gets ``neural_timing`` (:data:`NEURAL_TIMING_COLUMNS`), ``burst_span``
    (the burst's own ``(onset, offset)`` as absolute sample indices, which
    :func:`burst_masks` turns into masks) and ``emg_seg_qc`` (:data:`BURST_QC_COLUMNS`)."""
    n = len(np.atleast_1d(timecol))
    det = detect_bursts(detect_emg, fs, threshold_frac=threshold_frac, min_s=min_s,
                        smooth_s=smooth_s, min_contrast=min_contrast, filename=filename)
    onsets, offsets = det.onsets, det.offsets
    burst_frac = float(np.sum(offsets - onsets)) / n if n else float("nan")
    qc = OrderedDict([("emg_seg_n_bursts", float(len(onsets))),
                      ("emg_seg_burst_frac", burst_frac),
                      ("emg_seg_contrast", det.contrast)])
    segments = OrderedDict()
    for k, number in enumerate(range(1, len(onsets) + 1)):
        end = int(onsets[k + 1]) if k + 1 < len(onsets) else n
        seg = _make_segment(
            number, int(onsets[k]), end, timecol, emgcolumns, entropycolumns, filename,
            ignored=number in ignored_breaths, kind=kinds.get(number))
        seg["neural_timing"] = _neural_timing(onsets, offsets, k, fs)
        seg["burst_span"] = (int(onsets[k]), int(offsets[k]))
        seg["emg_seg_qc"] = OrderedDict(qc)
        segments[number] = seg
    return segments


def burst_masks(breaths, n: int, fs: float, guard_s: float):
    """``(burst, interburst)`` boolean masks of length ``n`` over a recording segmented by
    :func:`emg_burst`. ``burst`` is the union of the bursts; ``interburst`` the periods
    BETWEEN two consecutive bursts, each shrunk by ``guard_s`` at both ends (the RMS
    envelope smears an edge by about half its window, so the samples right next to a
    burst are not yet quiet). The stretch before the first burst and after the last one
    is not between two bursts and never counts. A gap no longer than two guard bands
    contributes nothing. Every segment of ``breaths``, ignored or not, takes part: which
    periods are quiet does not depend on which breaths the user excluded."""
    spans = sorted(b["burst_span"] for b in breaths.values() if "burst_span" in b)
    burst = np.zeros(n, bool)
    inter = np.zeros(n, bool)
    guard = int(round(guard_s * fs))
    for k, (on, off) in enumerate(spans):
        burst[on:off] = True
        if k + 1 < len(spans):
            lo, hi = off + guard, spans[k + 1][0] - guard
            if hi > lo:
                inter[lo:hi] = True
    return burst, inter


def whole_file(filename: str, timecol, emgcolumns, entropycolumns, rms_s: float, fs: float,
              *, ignored_breaths: set[int], kinds: dict) -> OrderedDict:
    """The entire recording as one segment, numbered 1. ``ignored_breaths``/``kinds``
    are the already-resolved per-breath-number lookups ``compute.ignorebreaths``/
    ``compute.breathkinds`` build for the file (the same shape ``separateintobreathsbyflow``/
    ``...byvolume`` already consume) — this module never reads settings directly, to
    keep it free of any dependency on ``core.compute``."""
    n = len(np.atleast_1d(timecol))
    segment = _make_segment(1, 0, n, timecol, emgcolumns, entropycolumns, filename,
                            ignored=1 in ignored_breaths, kind=kinds.get(1))
    _attach_whole_file_rms_diagnostics(segment, rms_s, fs)
    return OrderedDict([(1, segment)])


def separators(filename: str, times_s, timecol, emgcolumns, entropycolumns, fs: float,
               *, ignored_breaths: set[int], kinds: dict) -> OrderedDict:
    """``times_s`` (already validated by ``Settings.validate()`` as non-negative and
    strictly increasing — see ``SeparatorEntry``) split the recording into
    ``len(times_s) + 1`` segments, numbered from 1. A time at or beyond the
    recording's own duration, or two times close enough to round to the same
    sample, raises :class:`EmgSegmentationError` naming the file — a blind per-file
    failure (``FileResult.error_kind``), never a crash that stops the whole batch."""
    n = len(np.atleast_1d(timecol))
    bounds = [0]
    for t in times_s:
        ix = int(round(t * fs))
        if not (bounds[-1] < ix < n):
            raise EmgSegmentationError(
                f"A separator at {t:g} s in {filename} does not fall strictly inside "
                f"the recording ({n / fs:.2f} s long) after the previous boundary — "
                "check Preview & QC ▸ EMG – segments.")
        bounds.append(ix)
    bounds.append(n)

    segments = OrderedDict()
    for number, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        segments[number] = _make_segment(
            number, start, end, timecol, emgcolumns, entropycolumns, filename,
            ignored=number in ignored_breaths, kind=kinds.get(number))
    return segments


def remap_segment_number(old_bounds_s, new_bounds_s, old_number: int) -> int:
    """Map a segment NUMBER under an OLD set of segment boundaries to its equivalent
    number under a NEW set (M-27's manual-separator edit: placing or removing one
    boundary at a time), by finding where the OLD segment's own START TIME now falls
    among the NEW boundaries — the containing segment (the largest new boundary at or
    before that instant) is "the same segment, renumbered". See
    ``ui.screens.preview._segments._SegmentsMixin._set_separators`` for the caller.

    ``old_bounds_s``/``new_bounds_s``: each the FULL sorted list of segment start
    times, beginning with ``0.0`` (i.e. ``[0.0] + times_s``, matching how
    :func:`separators` above builds its own ``bounds`` before slicing) — not just the
    separator times alone. ``old_number`` is 1-based, like every other segment/breath
    number in this codebase.

    One rule covers every edit a single placement or removal can make, with no special
    case for any of them:

    * an insertion strictly AFTER the old segment's own start leaves that start time
      an exact boundary in the new list too, so it maps to whichever (now higher)
      number that same instant sits at;
    * an insertion strictly BEFORE it, splitting an earlier segment, is the same case:
      the old segment's own start is untouched and still an exact new boundary, one
      position further along;
    * removing the boundary AT the old segment's own start (a merge into the
      preceding segment) means that instant is no longer a boundary at all, so it now
      falls INSIDE whatever segment covers it — exactly what "the largest new
      boundary at or before it" finds.

    ``old_number`` past the end of ``old_bounds_s`` (stale data referencing a segment
    the current separators no longer produce) clamps to the last known boundary rather
    than raising — the caller has no better instant to compare with, and this is a
    renumbering aid, not a validator (``Settings.validate()``/``EmgSegmentationError``
    already own rejecting a genuinely malformed configuration)."""
    idx = min(max(old_number - 1, 0), len(old_bounds_s) - 1)
    old_start = old_bounds_s[idx]
    i = bisect.bisect_right(new_bounds_s, old_start) - 1
    i = max(0, min(i, len(new_bounds_s) - 1))
    return i + 1

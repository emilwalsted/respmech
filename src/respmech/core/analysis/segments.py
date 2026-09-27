"""EMG-only segmentation: split a recording that has no flow/pressure signal into
segments a downstream EMG/entropy analysis can run against, instead of the usual
inspiration/expiration breath split (which needs a flow channel to detect at all).

Two methods, both reachable only for an EMG-only signal set (``Settings.validate()``
already enforces this — a ``method`` from this module and a declared flow channel
never coexist):

* ``whole_file`` — the entire recording is one segment, always numbered 1. The
  natural choice for a single maximal manoeuvre (a sniff, a maximal voluntary
  contraction) captured as its own short recording.
* ``separators`` — N user-placed times split the recording into N+1 segments,
  numbered from 1. Zero separators is the same shape as ``whole_file`` (one
  segment), just reached explicitly through this method instead.

A later, automatic alternative (fixed windows / burst detection) is a separate
ticket's scope and does not live here.

Both build the SAME ``OrderedDict`` shape ``core.compute._make_breath`` builds for a
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

from collections import OrderedDict

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
    time = np.asarray(timecol[start:end]).squeeze()
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
    """
    emgcols = segment['emgcols']
    if len(emgcols) == 0:
        return
    n_ch = np.asarray(emgcols).shape[1]
    rms_max, t_rms_max, rms_top3 = [], [], []
    for ch in range(n_ch):
        values, starts = emglib.rolling_rms(np.asarray(emgcols)[:, ch], rms_s, fs)
        if values.size == 0:
            rms_max.append(float('nan'))
            t_rms_max.append(float('nan'))
            rms_top3.append(float('nan'))
            continue
        peak_ix = int(np.argmax(values))
        rms_max.append(float(values[peak_ix]))
        t_rms_max.append(float(starts[peak_ix]) / fs)
        top3 = np.sort(values)[-min(3, values.size):]
        rms_top3.append(float(np.mean(top3)))
    # One value per EMG channel, no appended max/mean summary (unlike compute_segment_emg's
    # rms/intemg families) — results.py::build_breath_table expands each of these three
    # lists into rms_file_max_col_N / t_rms_file_max_col_N / rms_file_top3_col_N, one column
    # per channel number, and nothing else.
    segment['rms_file_max'] = rms_max
    segment['t_rms_file_max'] = t_rms_max
    segment['rms_file_top3'] = rms_top3


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

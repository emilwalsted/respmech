"""Pure respiratory-mechanics / WOB / segmentation computation.

Faithful port of the validated legacy ``respmech.py`` calculation functions
(locked by the golden tests). Differences vs legacy ``master``, all documented:

* No file I/O, no plotting, no ``print`` — this module only computes. Progress is
  reported by the pipeline via events.
* ``scipy.integrate.simpson`` is called with keyword ``x=`` (positional removed in
  SciPy >= 1.14) — numerically identical (legacy bug #4).
* **PTP baseline is a short end-expiratory window mean** (`calcptp`, default 0.05 s
  via ``ptp.baseline_window_s``) — a deliberate, golden-locked change from legacy's
  single-sample ``- pressure[0]``, made after the review in ``docs/PTP_INVESTIGATION.md``.

Settings are read via the legacy attribute shape (see ``_legacy_ns``) to keep the
numerics byte-identical to the golden reference.
"""
import warnings

import numpy as np
import scipy as sp
import scipy.interpolate  # noqa: F401  (sp.interpolate)
import scipy.integrate    # noqa: F401
from collections import OrderedDict
from dataclasses import dataclass

from respmech.core import emg as emglib
from respmech.core import entropy as entlib
from respmech.core._cancel import check
from respmech.core.analysis.registry import LEGACY_MECHANICS_ORDER
from respmech.core.analysis.signals import Capabilities
from respmech.core.quality import detect_constant_channel


# --- volume conditioning ---------------------------------------------------

def zero(indata):
    return indata - (indata[0])


def correctdrift(volume, settings):
    xno = len(volume) - 1
    a = ((volume[xno]) - volume[0]) / xno
    val = a
    corvol = np.zeros(len(volume))
    for i in range(0, xno):
        corvol[i] = volume[i] + val
        val = val - a
    return corvol


class VolumeTrendError(ValueError):
    """Raised when the end-expiratory trend envelope cannot be fitted (a precondition
    failure, not a bug): too few end-expiratory troughs were detected for the chosen
    interpolation, or the volume signal is not finite."""


# Default for the scale-free anchor rule: an end-expiratory trough must be flanked by
# an inspiration of at least this fraction of the recording's own volume range. On the 13
# real recordings measured for this (11 production + the 2 reported), the anchor set is
# identical anywhere in 0.005-0.20 — a noise gate, not a tuning knob. It is still exposed,
# because a recording combining very shallow breaths with a large manoeuvre (which inflates
# the range) is the one shape that can need a lower value.
TREND_MIN_PROMINENCE_FRAC = 0.05

# Anchors scipy's interp1d needs per kind; every other accepted kind needs 2. Below
# these it raises "The number of derivatives at boundaries does not match: ...", which
# names neither the setting nor the recording.
_TREND_MIN_ANCHORS = {"quadratic": 3, "cubic": 4}

# How far above the predicted end-expiratory level a record's own end may still sit and
# count as end-expiratory, in multiples of the minimum breath depth. Measured: at 3x, the
# accepted anchors are identical to accepting both ends unconditionally on all 11 real
# trend-on recordings, while a record cut at peak inspiration is correctly rejected (its
# invented trend was a full tidal volume). At 1x, legitimate ends on RIU_H5_IC and
# RIU_H6_60W are rejected and the residual end-expiratory error more than doubles.
_END_ANCHOR_TOL_MULT = 3.0


def trend_anchors(vol, fs, *, min_height=None,
                  min_prominence_frac=TREND_MIN_PROMINENCE_FRAC, min_distance_s=0.4):
    """Indices of the end-expiratory troughs the trend envelope is fitted through.

    ``vol`` is the drift-corrected volume; troughs are the peaks of ``max(vol) - vol``.
    Single source of truth for the compute path AND the diagnostic figure, which used
    to re-implement the detection and could therefore draw anchors that were never
    subtracted (or omit a figure for a run that computed fine).

    ``min_height is None`` (the default) selects the scale-free rule: a trough qualifies
    on its PROMINENCE — the smaller of the two inspiratory excursions flanking it, i.e.
    about one tidal volume — as a fraction of the recording's own volume range. Being a
    per-trough measure it is invariant to tidal volume and to the volume unit, and it
    does not care where the file's global maximum sits.

    The recording's own first and last samples are considered too, because an edge is
    never a local maximum and ``find_peaks`` can therefore never return one — without
    that, every recording loses its two outermost anchors and the envelope extrapolates
    past the outermost trough (which is what leaves a NaN head/tail under the
    'previous'/'next' kinds). They are only ACCEPTED when they really sit at an
    end-expiratory level, tested by ``_end_is_expiratory``: ``trim`` ends the window at
    the last sample with ``flow >= 0``, which is "in expiration", not "at end-expiration",
    and it returns 0 for a file that already begins mid-inspiration. Anchoring an end
    that sits part way up a breath invents a trend and distorts that breath.

    An explicit ``min_height`` keeps the legacy absolute gate byte-for-byte: the trough
    must lie at least that many litres below the file's GLOBAL volume maximum. It is
    retained only so an older analysis reproduces exactly; see the module docstring of
    ``respmech.settingsio.migrate`` for why it is no longer the default.
    """
    from scipy import signal
    vol = np.asarray(vol, float).ravel()
    # scipy rejects distance < 1 with a message that names none of our settings. The
    # clamp is a no-op for any sane rate (0.4 s x 2000 Hz = 800 samples).
    distance = max(1.0, float(min_distance_s) * float(fs))
    inv = (vol * -1) + max(vol)
    if min_height is not None:
        return signal.find_peaks(inv, height=min_height, distance=distance)[0]
    span = float(np.ptp(inv))
    if span <= 0:
        return np.empty(0, dtype=int)
    found = signal.find_peaks(inv, prominence=float(min_prominence_frac) * span,
                              distance=distance)[0]
    if found.size == 0:
        # No breath-sized trough anywhere. The two record ends alone would still fit a
        # straight line, so the run would "correct the trend" and subtract almost nothing
        # while the report claimed otherwise — refuse instead, and let the caller explain.
        return found
    tol = _END_ANCHOR_TOL_MULT * float(min_prominence_frac) * span
    ends = [i for i in (0, vol.size - 1) if _end_is_expiratory(vol, found, i, tol)]
    return np.unique(np.concatenate((ends, found))).astype(int)


def _end_is_expiratory(vol, found, idx, tol):
    """Is the recording's first/last sample at an end-expiratory level?

    Compared against the level the interior troughs themselves predict at that position
    (linear, extrapolated beyond the outermost one), NOT against a fixed level —
    end-expiratory volume is exactly what trends here, so a start sitting well above a
    later trough may still be a perfectly good end-expiratory point. An end left part way
    up a breath sits above that prediction by a large fraction of a tidal volume, and is
    rejected.
    """
    if found.size == 1:
        predicted = float(vol[found[0]])
    elif found[0] <= idx <= found[-1]:
        predicted = float(np.interp(idx, found, vol[found]))
    else:
        predicted = float(np.polyval(np.polyfit(found[[0, -1]], vol[found[[0, -1]]], 1), idx))
    return bool(vol[idx] <= predicted + tol)


def _trend_anchor_message(vol, found, need, kind, mech):
    """One line — the GUI's short_error only shows the last one."""
    span = float(np.ptp(np.asarray(vol, float)))
    head = (f"Could not correct the end-expiratory trend: found {found} end-expiratory "
            f"trough(s), but '{kind}' interpolation needs {need}.")
    if mech.volumetrendpeakminheight is not None:
        return (f"{head} This analysis pins the legacy absolute threshold "
                f"(processing.volume.trend_peak_min_height = "
                f"{mech.volumetrendpeakminheight:g}), which requires a trough at least "
                f"that far below the recording's highest volume — but this recording's "
                f"whole volume range is only {span:.2f}. Set 'Trend anchor — absolute "
                f"threshold (legacy)' back to Auto under Mechanics — Advanced…, or lower "
                f"it below {span:.2f}.")
    frac = getattr(mech, "trend_peak_min_prominence_frac", TREND_MIN_PROMINENCE_FRAC)
    return (f"{head} A trough must be flanked by an inspiration of at least {frac:g} × "
            f"the recording's volume range ({frac * span:.3f}), and troughs must be "
            f"{mech.volumetrendpeakmindistance:g} s apart. Lower 'Trend anchor — minimum "
            f"breath depth' under Mechanics — Advanced…, or turn 'Correct end-expiratory "
            f"trend' off.")


def correcttrend(volume, settings):
    """Compute-only volume trend correction (no plot). Returns the corrected
    volume; the diagnostic plot is produced separately by the plots layer."""
    m = settings.processing.mechanics
    vol = volume.squeeze()
    if not np.isfinite(vol).all():
        raise VolumeTrendError(
            "Could not correct the end-expiratory trend: the volume signal contains "
            "missing or infinite samples. Check the volume channel under Setup (or "
            "'Calculate volume from flow' under Mechanics — Advanced…).")
    peaks = trend_anchors(
        vol, settings.input.format.samplingfrequency,
        min_height=m.volumetrendpeakminheight,
        min_prominence_frac=getattr(m, "trend_peak_min_prominence_frac",
                                    TREND_MIN_PROMINENCE_FRAC),
        min_distance_s=m.volumetrendpeakmindistance)
    kind = m.volumetrendadjustmethod
    need = _TREND_MIN_ANCHORS.get(kind, 2)
    if peaks.size < need:
        # Below this, interp1d either raises numpy's "cannot reshape array of size 0
        # into shape (0,newaxis)" (0 anchors) or — worse — accepts a single anchor and
        # silently returns an ALL-NaN envelope, so the run "succeeds" with NaN in every
        # VT/VE/PTP/WOB cell. Both are one actionable failure now.
        raise VolumeTrendError(_trend_anchor_message(vol, peaks.size, need, kind, m))
    f = sp.interpolate.interp1d(peaks, vol[peaks], kind, fill_value="extrapolate")
    peaksresampled = f(np.linspace(0, vol.size - 1, vol.size))
    if not np.isfinite(peaksresampled).all():
        # 'previous'/'next' leave everything outside the outermost anchor undefined. The
        # scale-free rule anchors both ends so it cannot happen there; the legacy gate
        # can, and always has — warn rather than fail a run that used to complete, since
        # those samples were already NaN before this change.
        bad = int((~np.isfinite(peaksresampled)).sum())
        warnings.warn(
            f"end-expiratory trend: '{kind}' interpolation leaves {bad} of {vol.size} "
            f"samples undefined (outside the outermost detected trough); those samples "
            f"become NaN. Use 'Linear' to cover the whole recording.")
    return volume - peaksresampled


# --- trimming --------------------------------------------------------------

class TrimError(ValueError):
    """Raised when the data cannot be trimmed to whole breaths (a precondition
    failure, not a bug): the recording must start in late expiration and end in
    early inspiration."""


def trim(timecol, flow, volume, poes, pgas, pdi, emgcolumns, settings):
    below = np.argwhere(flow <= 0)
    above = np.argwhere(flow >= 0)
    if len(below) == 0 or len(above) == 0:
        raise TrimError(
            "Could not trim data to whole breaths: the flow signal never crosses "
            "zero as required (data must start in late expiration and end in early "
            "inspiration). Check Setup ▸ channel assignment, or 'Invert the flow "
            "signal' under Preview & QC ▸ Mechanics ▸ Advanced….")
    # Start at the first INSPIRATION sample (flow strictly < 0), not the first
    # non-positive one: a recording that begins at rest (flow == 0) would otherwise set
    # startix = 0, leaving the leading expiration in place — then the first breath begins
    # in expiration, its inspiration loop never advances, and inend underflows to -1 so
    # the "inspiration" slice [0:-1] spans the whole recording (a malformed breath #1
    # that was silently averaged into the mean breath / WOB).
    startix = int(np.argmax(flow < 0))
    endix = int(above[:, 0][len(above[:, 0]) - 1])
    if endix <= startix:
        raise TrimError(
            "Could not trim data to whole breaths: computed an empty range "
            "(the recording does not start in late expiration and end in early "
            "inspiration). Check Setup ▸ channel assignment, or 'Invert the flow "
            "signal' under Preview & QC ▸ Mechanics ▸ Advanced….")
    return (timecol[startix:endix], flow[startix:endix], volume[startix:endix],
            poes[startix:endix], pgas[startix:endix], pdi[startix:endix],
            emgcolumns[startix:endix], startix, endix)


# --- breath segmentation ---------------------------------------------------

class NoBreathsError(ValueError):
    """Raised when a recording yields no breath to analyse (a precondition failure, not
    a bug): either segmentation detected none, or every detected breath is excluded."""


def check_breaths(breaths, filename, settings):
    """Fail early, and by name, when a file has nothing to analyse.

    Without this the empty set travels on and only surfaces further downstream as
    ``AttributeError: 'list' object has no attribute 'mean'`` from the results layer's
    ``mechs.mean()`` — reported verbatim as that file's error, naming neither the file's
    real problem nor a setting to change.
    """
    used = sum(1 for b in breaths.values() if not b["ignored"])
    if used:
        return
    if breaths:
        raise NoBreathsError(
            f"All {len(breaths)} breath(s) detected in {filename} are excluded, so there "
            f"is nothing to analyse. Re-include at least one — click a shaded breath in "
            f"Preview & QC, or clear this file's entry under processing.exclude_breaths.")
    by = settings.processing.mechanics.separateby
    channel = "flow" if by == "flow" else "volume"
    raise NoBreathsError(
        f"No breaths were detected in {filename}. Breaths are split on the {by} signal, so "
        f"check that the {channel} channel is the right column under Setup, and the "
        f"'Signal used to split breaths' and 'Breath peak' settings under Preview & QC — "
        f"Mechanics — Advanced… (processing.segmentation).")


def ignorebreaths(curfile, settings):
    d = dict(settings.processing.mechanics.excludebreaths)
    return d[curfile] if curfile in d else []


def breathkinds(curfile, settings):
    """Mirrors :func:`ignorebreaths`' file-keyed lookup, for the kind (not just the
    exclusion) a typed breath carries (M-19). Returns ``{breath_no: kind}`` for
    ``curfile`` -- read alongside ``ignorebreaths``' result in both segmenterers so a
    typed breath's dict also gets ``kind`` set, not just ``ignored=True``."""
    d = dict(getattr(settings.processing.mechanics, "breathtypes", []))
    entries = d[curfile] if curfile in d else []
    return {no: kind for no, kind, _t_onset_s in entries}


class DegenerateBreathError(ValueError):
    """Raised when a detected breath's inspiration and expiration phases cannot be
    joined into one breath (a precondition failure, not a bug): typically an
    incomplete breath right at the start or end of a recording."""


class ConstantFlowError(ValueError):
    """Raised when the flow channel does not vary enough for breath separation to make
    progress -- either across the WHOLE recording, or across a stretch of it reached
    partway through (e.g. a flat, zero-flow pause between breaths).

    Breath separation on flow (``separateintobreathsbyflow``) walks the signal looking
    for a sign change between inspiration (flow < 0) and expiration (flow > 0); a flow
    that is exactly constant AT ZERO over some stretch (most commonly a mis-assigned or
    unused/grounded column, but also a genuine pause at exactly zero flow) satisfies
    neither condition there, so the walk's index cannot advance past that point. A
    constant NON-zero flow does not hang this same loop (one of the two conditions is
    always true, so the walk consumes the rest of the file as one bogus "breath" and
    returns instead of getting stuck) -- checked and rejected the same way regardless,
    since a flow that never crosses zero cannot be split into real breaths either way.

    The zero-flow case was traced, not assumed, and its failure mode depends on
    whether any EMG/entropy columns are configured: with none, the walk's index truly
    never advances and the loop runs forever (confirmed with a hard-killed subprocess);
    with at least one EMG/entropy column configured, ``_phase_dicts`` instead raises a
    bare ``ValueError: negative dimensions are not allowed`` on the very first
    iteration (the degenerate ``(0, -1)`` phase slice this loop produces here computes
    a negative array length for those columns specifically). Both are confusing dead
    ends compared to this named, actionable error, so both are checked the same way:
    up front (the whole-recording case, cheaply, before the loop even starts) and
    inside the loop itself (the general case, wherever a flat-at-zero stretch is
    reached) instead of ever reaching either one."""


def _breath_onset_time(insp, exp):
    """The breath's own start time in the recording, read straight from its (already
    phase-sliced) time arrays -- used only for a degenerate-breath message (see
    ``_make_breath`` below), so it must work even when the six-way concatenation that
    triggers ``DegenerateBreathError`` itself failed. Returns ``None`` when neither
    phase has a usable time sample (both empty, or all non-finite)."""
    for phase in (insp, exp):
        t = np.asarray(phase.get("time")).reshape(-1)
        t = t[np.isfinite(t)]
        if t.size:
            return float(t[0])
    return None


def _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename, is_boundary, kind=None):
    # The try is scoped to only the six joins below (not the whole dict literal), so a
    # ValueError from an unrelated future addition here is never mislabeled as a
    # degenerate breath (self-review finding, D25).
    try:
        time = np.concatenate((insp["time"], exp["time"])).squeeze()
        flow = np.concatenate((insp["flow"], exp["flow"])).squeeze()
        volume = np.concatenate((insp["volume"], exp["volume"])).squeeze()
        poes = np.concatenate((insp["poes"], exp["poes"])).squeeze()
        pgas = np.concatenate((insp["pgas"], exp["pgas"])).squeeze()
        pdi = np.concatenate((insp["pdi"], exp["pdi"])).squeeze()
    except ValueError as e:
        if is_boundary:
            raise DegenerateBreathError(
                f"Breath #{breathcnt} in {filename} is degenerate (its inspiration and "
                f"expiration phases could not be joined into one breath) and cannot be "
                f"analysed. This usually happens at an incomplete breath right at the "
                f"start or end of a recording — exclude it in Preview & QC, or check the "
                f"breath-separation settings under Preview & QC ▸ Mechanics ▸ Advanced… ▸ "
                f"Breath detection.") from e
        onset = _breath_onset_time(insp, exp)
        where = f"at t≈{onset:.2f} s into the recording" if onset is not None else "mid-recording"
        raise DegenerateBreathError(
            f"Breath #{breathcnt} in {filename} is degenerate (its inspiration and "
            f"expiration phases could not be joined into one breath) and cannot be "
            f"analysed. This breath is {where}, well away from either boundary of the "
            f"file, so a truncated recording is unlikely to be the cause here — more "
            f"likely candidates are noise around zero flow, a mis-assigned flow channel, "
            f"or more than one recording merged together in this file. Exclude it in "
            f"Preview & QC, or check the breath-separation settings under Preview & QC ▸ "
            f"Mechanics ▸ Advanced… ▸ Breath detection.") from e
    return OrderedDict([
        ('number', breathcnt),
        ('name', 'Breath #' + str(breathcnt)),
        ('expiration', exp),
        ('inspiration', insp),
        ('time', time),
        ('flow', flow),
        ('volume', volume),
        ('poes', poes),
        ('pgas', pgas),
        ('pdi', pdi),
        ('breathcnt', breathcnt),
        ('ignored', ignored),
        ('kind', kind),
        ('has_phases', True),
        ('entcols', entcols),
        ('emgcols', emgcols),
        ('filename', filename),
    ])


def _phase_dicts(sl_in, sl_ex, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns):
    (instart, inend) = sl_in
    (exstart, exend) = sl_ex
    exp = {'time': timecol[exstart:exend].squeeze(), 'flow': flow[exstart:exend].squeeze(),
           'poes': poes[exstart:exend].squeeze(), 'pgas': pgas[exstart:exend].squeeze(),
           'pdi': pdi[exstart:exend].squeeze(), 'volume': volume[exstart:exend].squeeze()}
    insp = {'time': timecol[instart:inend].squeeze(), 'flow': flow[instart:inend].squeeze(),
            'poes': poes[instart:inend].squeeze(), 'pgas': pgas[instart:inend].squeeze(),
            'pdi': pdi[instart:inend].squeeze(), 'volume': volume[instart:inend].squeeze()}
    # NO .squeeze() on the phase slices: they stay (samples, channels). A single-channel
    # recording would otherwise collapse to 1-D here even when the loader got it right (the
    # .mat path always did), and calculate_rms then iterates emgchannels.T over a 1-D array,
    # yielding scalars and raising "TypeError: object of type 'numpy.float64' has no len()".
    # For >= 2 channels the squeeze only ever fired on a one-sample phase, which crashes the
    # run today either way, so no analysis that currently completes can change.
    if len(entropycolumns) > 0:
        exp['entcols'] = entropycolumns[exstart:exend, :]
        insp['entcols'] = entropycolumns[instart:inend, :]
    if len(emgcolumns) > 0:
        exp['emgcols'] = emgcolumns[exstart:exend, :]
        insp['emgcols'] = emgcolumns[instart:inend, :]

    entlen = exend - instart
    if len(entropycolumns) > 0:
        entcols = np.zeros([entlen, entropycolumns.shape[1]])
        for ix in range(0, entropycolumns.shape[1]):
            entcols[:, ix] = entropycolumns[instart:exend, ix]
    else:
        entcols = []
    if len(emgcolumns) > 0:
        emgcols = np.zeros([entlen, emgcolumns.shape[1]])
        for ix in range(0, emgcolumns.shape[1]):
            emgcols[:, ix] = emgcolumns[instart:exend, ix]
    else:
        emgcols = []
    return exp, insp, entcols, emgcols


def separateintobreathsbyflow(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    # A flow constant at exactly zero satisfies neither `flow[i] < 0` nor `flow[i] > 0`
    # below, so `i` never advances past `instart` -- an infinite loop (or, with any
    # EMG/entropy column configured, a confusing "negative dimensions" crash instead;
    # see ConstantFlowError's own docstring for both, traced not assumed). Cheap
    # whole-recording case, checked up front.
    if detect_constant_channel(flow):
        raise ConstantFlowError(
            f"The flow channel in {filename} is constant (it does not vary at all), so "
            f"breaths cannot be split on it. This usually means the wrong column is "
            f"assigned as the flow channel, or this input has no real flow signal — "
            f"check the channel assignment in Setup ('Assign channels from data…'), or "
            f"switch 'Signal used to split breaths' to volume under Preview & QC ▸ "
            f"Mechanics ▸ Advanced… ▸ Breath detection.")
    breaths = OrderedDict()
    j = len(flow)
    bufferwidth = settings.processing.mechanics.breathseparationbuffer
    ib = ignorebreaths(filename, settings)
    bk = breathkinds(filename, settings)
    i = 0
    breathno = 0
    breathcnt = 0
    while i < j:
        breathcnt += 1
        instart = i
        while (i < j and ((flow[i] < 0) or (np.mean(flow[i:min(j, i + bufferwidth)]) < 0))):
            i += 1
        inend = i - 1
        exstart = i
        while (i < j and ((flow[i] > 0) or (np.mean(flow[i:min(j, i + bufferwidth)]) > 0))):
            i += 1
        exend = min(i - 1, j)
        # General case: the whole-array check above cannot see a flow that is only
        # LOCALLY flat (e.g. a genuine pause at exactly zero flow between breaths, or a
        # merged-block boundary that happens to land on zero) -- if NEITHER inner loop
        # advanced `i` past `instart`, the walk is stuck and would otherwise repeat this
        # same iteration forever (self-review finding: confirmed by tracing a
        # concatenated sine+zeros signal, which the whole-array check alone does not
        # catch since the array as a WHOLE still varies).
        if i == instart:
            raise ConstantFlowError(
                f"The flow channel in {filename} is flat (does not cross zero) at "
                f"t≈{timecol[i]:.2f} s, so breath separation cannot continue past this "
                f"point. This can happen at a genuine pause with exactly zero flow "
                f"between breaths, or where more than one recording has been merged "
                f"into this file — check the flow channel around this timestamp, or "
                f"switch 'Signal used to split breaths' to volume under Preview & QC ▸ "
                f"Mechanics ▸ Advanced… ▸ Breath detection.")
        is_boundary = (breathcnt == 1) or (i >= j)
        exp, insp, entcols, emgcols = _phase_dicts(
            (instart, inend), (exstart, exend), timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns)
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
        breaths[breathcnt] = _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename,
                                           is_boundary, kind=bk.get(breathcnt))
    return breaths


class VolumeSegmentationError(ValueError):
    """Raised when volume-based breath segmentation cannot pair every detected
    inspiratory peak with an expiratory one (a precondition failure, not a bug):
    a pause at zero flow between phases (slow, quiet breathing) can suppress an
    expiratory peak below the configured thresholds and leave too few to pair."""


def separateintobreathsbyvolume(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    from scipy import signal
    breaths = OrderedDict()
    ib = ignorebreaths(filename, settings)
    bk = breathkinds(filename, settings)
    breathno = 0
    breathcnt = 0
    invol = volume
    exvol = -1 * volume
    exvol = exvol + min(exvol) * -1
    samplingfrequency = settings.input.format.samplingfrequency
    peakheight = settings.processing.mechanics.peakheight
    peakdistance = settings.processing.mechanics.peakdistance
    peakwidth = settings.processing.mechanics.peakwidth
    inpeaks, _ = signal.find_peaks(invol, height=peakheight, distance=peakdistance * samplingfrequency, width=peakwidth * samplingfrequency)
    expeaks, _ = signal.find_peaks(exvol, height=peakheight, distance=peakdistance * samplingfrequency, width=peakwidth * samplingfrequency)
    # The loop below indexes expeaks[breathcnt - 2] and expeaks[breathcnt - 1] for
    # breathcnt up to len(inpeaks); the highest index it ever needs is
    # len(inpeaks) - 2, so it needs at least len(inpeaks) - 1 expiratory peaks.
    # Fewer than that used to read past the end of expeaks and raise a bare
    # IndexError instead of naming the problem.
    if len(inpeaks) > 1 and len(expeaks) < len(inpeaks) - 1:
        raise VolumeSegmentationError(
            f"Volume-based breath segmentation found {len(inpeaks)} inspiratory peaks "
            f"but only {len(expeaks)} expiratory peaks in {filename}, so at least one "
            f"breath could not be paired with its expiration. This can happen with slow "
            f"breathing and pauses at zero flow between phases. Check 'Signal used to "
            f"split breaths' and the 'Breath peak' thresholds under Preview & QC ▸ "
            f"Mechanics ▸ Advanced…, or switch to flow-based segmentation.")
    for inpeak in inpeaks:
        breathcnt += 1
        if breathcnt == 1:
            instart = 0
            inend = inpeak - 1
        else:
            instart = expeaks[breathcnt - 2]
            inend = inpeak - 1
        exstart = inend + 1
        if breathcnt < len(inpeaks):
            exend = expeaks[breathcnt - 1] - 1
        else:
            exend = len(invol) - 1
        is_boundary = (breathcnt == 1) or (breathcnt == len(inpeaks))
        exp, insp, entcols, emgcols = _phase_dicts(
            (instart, inend), (exstart, exend), timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns)
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
        breaths[breathcnt] = _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename,
                                           is_boundary, kind=bk.get(breathcnt))
    return breaths


@dataclass
class SegmentationOverrideNotice:
    """One per-file quality notice from :func:`apply_segmentation_overrides` — an
    out-of-range cut, a cut colliding with an existing boundary, or a join with no
    automatic boundary nearby. Never raised as an error: one bad override entry in a
    big batch must not abort an otherwise-fine run, the same "advisory, never fails a
    file" posture :func:`trim_boundary_notices` already has for a truncated boundary
    breath. ``message`` is plain text, ready to append to ``FileResult.notices``
    alongside every other per-file notice."""
    message: str


def _walk_insp_end(flow, start: int, end: int, bufferwidth: int) -> int:
    """The SAME criterion :func:`separateintobreathsbyflow`'s own inspiration
    while-loop uses (mean-buffered flow sign), bounded to ``[start, end)`` instead of
    the whole recording — used by :func:`apply_segmentation_overrides` to re-split
    EACH resulting segment into inspiration/expiration once the override list has
    decided where that segment's own boundaries are: the transition-finding rule
    itself is unchanged, only the OUTER loop that used to keep discovering further
    breaths past this one is gone, because this segment's own end is now GIVEN, not
    found.

    Everything from the returned index to ``end`` is that segment's expiration,
    regardless of any further sign changes inside it — by construction, a segment
    produced by the override list is meant to be exactly ONE breath, so a residual
    flow wobble within it (the very artefact a ``join_s`` exists to absorb) must NOT
    trigger a second split here, or the override would simply reproduce the automatic
    over-segmentation it was asked to undo. A segment whose flow never leaves negative
    (i reaches ``end`` without a qualifying transition) reports the whole segment as
    inspiration with an empty expiration — a degenerate, boundary-notice-worthy
    breath, the same risk ``_make_breath``'s own ``is_boundary`` branch already
    handles for the ordinary auto-segmented case."""
    i = start
    while i < end and (flow[i] < 0 or np.mean(flow[i:min(end, i + bufferwidth)]) < 0):
        i += 1
    return i


def apply_segmentation_overrides(filename, auto_breaths, cut_s, join_s, timecol, flow, volume,
                                  poes, pgas, pdi, entropycolumns, emgcolumns, fs, bufferwidth,
                                  *, ignored_breaths, kinds, tolerance_s=None):
    """Repair the automatic flow-/volume-based breath segmentation using one file's
    manual ``cut_s``/``join_s`` (``SegmentationOverrideEntry``),
    applied AFTER the automatic segmentation (``auto_breaths`` — the un-overridden
    output of :func:`separateintobreathsbyflow`/:func:`separateintobreathsbyvolume`,
    read here ONLY for its breaths' own start times, never its numbering/ignore/kind,
    which describe the OLD, pre-override segmentation) and BEFORE breath numbering,
    ignore/kind assignment and phase re-splitting — so every downstream consumer
    (exclusions, types, references, ``t_onset_s`` anchors, boundary notices) sees ONE
    finished boundary list, exactly like the EMG-only ``separators`` method already
    gives its own consumers one finished list (see :func:`core.analysis.segments.
    separators`).

    ``cut_s`` inserts a new boundary at each named time, splitting whatever automatic
    breath currently spans it. ``join_s`` REMOVES the nearest AUTOMATIC boundary to
    each named time, within ``tolerance_s`` (default ``max(2/fs, 0.05 s)`` — the same
    tolerance the UI's click-to-remove uses for the EMG-only separators list). An
    out-of-range or already-occupied cut, or a join with no automatic boundary
    nearby, is reported as a soft :class:`SegmentationOverrideNotice` and otherwise
    ignored — one bad
    entry must not abort an otherwise-fine batch, the same posture
    :func:`trim_boundary_notices` already has.

    Each resulting segment gets its OWN inspiration/expiration split via
    :func:`_walk_insp_end`, bounded to that segment's own ``[start, end)`` — see that
    function's docstring for why a residual wobble inside a JOINED segment must not
    re-trigger a split. ``ignored_breaths``/``kinds`` (the same shape
    :func:`ignorebreaths`/:func:`breathkinds` already return) are then applied to the
    NEW numbering, exactly as the automatic segmenters apply them to their own.

    Returns ``(breaths, notices)``: ``breaths`` an ``OrderedDict`` in the exact shape
    :func:`_make_breath` builds (``has_phases=True``), numbered 1..N; ``notices`` a
    list of :class:`SegmentationOverrideNotice`. The caller (``core.pipeline.
    segment_file``) only invokes this function when at least one of ``cut_s``/
    ``join_s`` is non-empty — an analysis with no override entry for this file, or an
    entry whose lists are both empty, never reaches here at all, which is what makes
    the "empty overrides is byte-identical" acceptance criterion hold BY
    CONSTRUCTION rather than by this function happening to reproduce
    ``auto_breaths`` exactly."""
    timecol = np.atleast_1d(timecol)
    n = len(timecol)
    tol = tolerance_s if tolerance_s is not None else max(2.0 / fs, 0.05)
    notices: list[SegmentationOverrideNotice] = []

    # `timecol` is the TRIMMED window's own array, but its VALUES are still the
    # recording's absolute clock (compute.trim() slices timecolraw, an
    # arange(n_raw)/fs built before any trimming -- it never re-zeroes it): the same
    # convention `breath['time']`/`SeparatorEntry.times_s`/`cut_s`/`join_s` all use. So
    # the trimmed window's own first sample is NOT necessarily t=0.0 (trim() commonly
    # discards a leading partial expiration first) -- t0 below is that actual value,
    # used everywhere "the window's own start" would otherwise wrongly assume 0.0 (a
    # real bug this function shipped with once: on a file whose leading trim was
    # nonzero, breath 1's own start was treated as a second, spurious "boundary" a
    # metres away from true index 0, silently fabricating a near-empty first breath).
    t0 = float(timecol[0]) if n else 0.0

    def _index_of(t: float) -> int:
        return int(round((t - t0) * fs))

    auto_starts = sorted(
        float(np.asarray(b["time"]).reshape(-1)[0])
        for b in auto_breaths.values() if len(np.asarray(b["time"]).reshape(-1))
    )
    # boundaries strictly after the window's own start -- the first breath's own start
    # (t0, or as close to it as the first sample is) is never a JOINABLE boundary,
    # there is nothing analysed before it to merge into.
    removable = [t for t in auto_starts if t > t0 + 1e-9]

    for t in join_s:
        candidates = [b for b in removable if abs(b - t) <= tol]
        if not candidates:
            notices.append(SegmentationOverrideNotice(
                f"join at {t:g} s in {filename}: no automatic breath boundary within "
                f"{tol * 1000:.0f} ms — ignored."))
            continue
        removable.remove(min(candidates, key=lambda b: abs(b - t)))

    # Each boundary is tagged NATURAL (t0, a kept automatic breath start, or the
    # file's own end) or CUT (inserted by this override). separateintobreathsbyflow's
    # own `inend = i - 1` / `exend = min(i - 1, j)` drops exactly the sample AT a
    # transition it found -- a real property of every NATURAL boundary in this file,
    # reproduced below so a joined breath is byte-identical to the clean recording's
    # own. A CUT boundary is not such a transition (it is a bisection point chosen by
    # the user, not one the flow signal itself produced): giving it the same -1 would
    # invent an extra dropped sample the acceptance criterion's own Ti+Te-conservation
    # check exists to catch, so it does NOT get one -- the two segments either side of
    # a cut partition the original span exactly, with no sample belonging to neither.
    tagged = [(t0, True)] + [(t, True) for t in removable]
    for t in cut_s:
        ix = _index_of(t)
        if not (0 < ix < n):
            notices.append(SegmentationOverrideNotice(
                f"cut at {t:g} s in {filename} falls outside the recording — ignored."))
            continue
        if any(abs(b - t) <= tol for b, _natural in tagged):
            notices.append(SegmentationOverrideNotice(
                f"cut at {t:g} s in {filename} coincides with an existing boundary — "
                "ignored."))
            continue
        tagged.append((t, False))
    tagged.sort(key=lambda pair: pair[0])

    bound_ix = [0]
    natural_end_ix = {n}                    # the file's own end is always natural
    for t, natural in tagged[1:]:
        ix = _index_of(t)
        if ix > bound_ix[-1]:
            bound_ix.append(ix)
            if natural:
                natural_end_ix.add(ix)
    bound_ix.append(n)

    breaths = OrderedDict()
    breathno = 0
    n_segments = len(bound_ix) - 1
    for breathcnt, (start, end) in enumerate(zip(bound_ix[:-1], bound_ix[1:]), start=1):
        inend = _walk_insp_end(flow, start, end, bufferwidth)
        # `inend` gets the -1 UNLESS it is the degenerate "found no transition, ran
        # into this segment's own end" case (_walk_insp_end's own docstring) AND that
        # end is a CUT boundary rather than a natural one. A genuine transition
        # (`inend < end`) always gets it, same as `separateintobreathsbyflow`'s own
        # `inend = i - 1`. Reaching a NATURAL end without a transition (`inend ==
        # end in natural_end_ix`) ALSO gets it -- that reproduces the original
        # algorithm's own behaviour for a breath that runs to the file's end without
        # ever finding an insp->exp transition (`exend = min(i - 1, j)` drops that
        # same sample there too). Only reaching a CUT end without a transition must
        # keep the segment's real last sample intact -- a cut boundary is not a flow
        # transition, so inventing a drop there would silently lose exactly the
        # sample this segment's own Ti+Te-conservation guarantee promises to keep
        # (a real bug found by review: an earlier version of this function applied
        # the -1 here unconditionally, silently dropping one sample from any cut
        # placed mid-inspiration).
        insp_end = inend if (inend == end and end not in natural_end_ix) else inend - 1
        exend = end - 1 if end in natural_end_ix else end
        exp, insp, entcols, emgcols = _phase_dicts(
            (start, insp_end), (inend, exend), timecol, flow, volume, poes, pgas, pdi,
            entropycolumns, emgcolumns)
        is_boundary = (breathcnt == 1) or (breathcnt == n_segments)
        if breathcnt in ignored_breaths:
            ignored = True
        else:
            breathno += 1
            ignored = False
        breaths[breathcnt] = _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols,
                                          filename, is_boundary, kind=kinds.get(breathcnt))
    return breaths, notices


def _separator_times_for(curfile, settings):
    """Mirrors :func:`ignorebreaths`' file-keyed lookup, for the EMG-only ``separators``
    method's per-file boundary times (``settings.processing.mechanics.separators``, the
    ``[[file, [t…]], …]`` shape ``core._legacy_ns`` builds from ``SeparatorEntry``). A
    file with no entry has no separators -- the same, explicit "one segment" shape
    ``whole_file`` produces, just reached via this method instead."""
    d = dict(getattr(settings.processing.mechanics, "separators", []))
    return d.get(curfile, [])


def separateintobreaths(method, filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings,
                        detect_emg=None):
    """``detect_emg`` (``emg_burst`` only): the signal the bursts are found on, when that
    differs from ``emgcolumns`` -- the ECG-removed EMG before noise reduction. ``None``
    (every other caller) detects on ``emgcolumns`` itself."""
    method = str.lower(method)
    if method in ("whole_file", "separators", "fixed_windows", "emg_burst"):
        from respmech.core.analysis import segments as _segments
        ib = set(ignorebreaths(filename, settings))
        bk = breathkinds(filename, settings)
        fs = settings.input.format.samplingfrequency
        if method == "fixed_windows":
            es = settings.processing.mechanics.emgsegmentation
            return _segments.fixed_windows(
                filename, timecol, emgcolumns, entropycolumns, fs,
                window_s=es.window_s, hop_s=es.hop_s, ignored_breaths=ib, kinds=bk)
        if method == "emg_burst":
            es = settings.processing.mechanics.emgsegmentation
            return _segments.emg_burst(
                filename, timecol, emgcolumns, entropycolumns,
                emgcolumns if detect_emg is None else detect_emg, fs,
                threshold_frac=es.burst_threshold_frac, min_s=es.burst_min_s,
                smooth_s=es.burst_smooth_s, min_contrast=es.burst_min_contrast,
                ignored_breaths=ib, kinds=bk)
        if method == "whole_file":
            return _segments.whole_file(
                filename, timecol, emgcolumns, entropycolumns,
                settings.processing.emg.rms_s, fs, ignored_breaths=ib, kinds=bk)
        return _segments.separators(
            filename, _separator_times_for(filename, settings), timecol, emgcolumns,
            entropycolumns, fs, ignored_breaths=ib, kinds=bk)
    if method == "volume":
        return separateintobreathsbyvolume(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)
    return separateintobreathsbyflow(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)


def trim_boundary_notices(breaths, settings, *, min_relative_duration=None, min_other_breaths=None):
    """Per-file quality notice for a boundary breath likely truncated by ``trim`` (K-035).

    ``trim`` (above) discards only a leading partial expiration and a trailing partial
    inspiration; it never verifies that the breath it KEEPS at either boundary is
    itself complete. There is no reliable way to tell from a single boundary SAMPLE's
    sign alone: a first attempt at this check compared ``startix``/``endix`` against
    the raw array's own edges and false-flagged both the built-in sample recording and
    the committed golden synthetic inputs (``tests/golden/input``) — none of those are
    truncated, they simply end without a hair's-breadth of margin into the next phase,
    which is indistinguishable from real truncation at the single-sample level.

    Instead, this compares the FIRST breath's inspiratory duration, and the LAST
    breath's expiratory duration, against this file's own MEDIAN duration for that
    phase across every OTHER detected breath — a within-file, self-calibrating
    comparison that needs no assumption about the subject's breathing rate and
    tolerates ordinary breath-to-breath variability. A boundary phase shorter than
    ``min_relative_duration`` (default 80%) of that median is flagged.

    The threshold was measured, not guessed, against K-035's own reported reproduction
    (a recording cut 0.5 s into the built-in sample's first, 1.661 s inspiration, and a
    separate one cut 0.5 s into its last, 1.440 s expiration): replaying both cuts
    through this exact function gives ratios of 0.72 and 0.30 against the file's own
    median — the inspiratory case in particular is NOT "comfortably" below a lower
    threshold, it sits close to typical breath-to-breath variation. 0.8 sits roughly
    midway between that 0.72 "known truncated" case and 0.88, the tightest natural
    (non-truncated) ratio measured on the same built-in sample recording's own last
    breath (its synthetic generator varies each breath's period by design). A lower
    threshold like the 0.6 this function shipped with during development does NOT
    catch K-035's own motivating case at all — verified by replaying it after the fact
    — which is why 0.8 replaced it before this ticket closed.

    ``min_other_breaths`` (default 3) guards against an unstable median on a very
    short recording: with only 1-2 OTHER breaths to compare against, one atypically
    long or short breath (a sigh, an early arousal) can swing the median enough to
    flag a perfectly normal boundary breath, or hide a truncated one. Below that
    count, that side's check is skipped entirely rather than guessed at.

    Both defaults are read from ``settings.processing.mechanics.boundarynoticeminrelativeduration``
    / ``boundarynoticeminotherbreaths`` (``Settings.processing.segmentation.boundary_notice_*``
    in the typed model) when the caller does not pass an explicit override — see the
    follow-up investigation below for why this is a per-analysis SETTING and not a
    revised built-in statistic.

    **Follow-up investigation (06-09-2026, a review raised after this function
    shipped): is 0.8 too aggressive on ordinary, high-variability breathing?** A reviewer's
    Monte Carlo simulation (log-normal phase durations) found 15-39% false-positive rates
    at breath-to-breath coefficients of variation (CV) of 15-25% — a PER-FILE rate (either
    boundary check firing on a non-truncated file), not a per-check rate: independently
    reproducing the simulation gives ~8-19% for a single boundary check alone at the same
    CVs, which combines across the two independent checks (first inspiration, last
    expiration) to the cited 15-39% file-level range. Published measurements of
    real resting breathing (16 healthy subjects, 40 min quiet breathing, opto-electronic
    plethysmography: CV of fractional inspiratory time TI/TTOT = 17.9±6.5%, CV of
    respiratory frequency = 20.8±11.5%, and TI/TTOT is reported as LESS variable than the
    raw phase durations this function actually compares) confirm that range is physiologically
    realistic, not a pessimistic guess — the false-positive risk is real.

    The natural fix — replace the fixed ratio with a MAD-based robust z-score that adapts
    to each file's OWN measured variability — was built and Monte-Carlo-compared against
    the current ratio check at matching CVs and against this function's own two known
    reference cases (K-035's 0.72/0.30 truncated ratios; the built-in sample's 0.8775
    tightest natural ratio; both re-measured with the file's REAL median/MAD, not an
    assumed one). It does reduce false positives substantially (e.g. at CV 20%, ~14% down
    to ~1-5% depending on the z-threshold chosen) but at a cost that is NOT a wash: because
    a real truncation is a FIXED absolute cut (K-035's reproduction: 0.5 s), its size
    relative to the file's own natural spread shrinks as that spread grows, so the
    MAD-based check's sensitivity to the exact same truncation falls even faster than its
    false-positive rate does — from ~83% detection at CV 10% down to ~6-47% at CV 25-30%
    depending on the z-threshold, i.e. it becomes LEAST sensitive precisely on the more
    variable recordings where an ordinary-looking boundary breath is hardest for a human to
    catch by eye. A z-threshold picked to just span this function's own two known reference
    cases (~2.0, the midpoint of their measured z-scores -2.37 and -1.67, mirroring exactly
    how 0.8 was picked as the midpoint of 0.72 and 0.88) still trades meaningful detection
    power for a real but partial false-positive reduction, at every CV in the simulated
    range. Given this notice is advisory only (it never fails a file — missing a real
    truncation is the costlier failure mode of the two), replacing the statistic was
    rejected: it is a different trade-off, not a demonstrated improvement.

    The decision reached by this investigation (06-09-2026 — not yet reviewed or
    confirmed by Emil, unlike the earlier decisions elsewhere in this codebase that carry
    his name): keep 0.8/3 as the default, documented here as a known, accepted trade-off
    rather than a proven-safe value, and expose both numbers as a per-analysis setting (see
    above) so a study whose recordings are known to have unusually high natural variability
    can raise the threshold (e.g. to 0.6-0.7) deliberately, instead of the whole install
    silently trading detection power away for every user based on one un-validated
    system-wide guess. Real research recordings (``tests/golden/production``, unavailable
    in the sandbox this investigation ran in) would let a future session replace this
    entire trade-off analysis with a measured threshold instead — the preferred path, if
    that data becomes reachable, is to replace the simulated Monte Carlo comparison above
    with one measured directly against real recordings.

    A boundary breath that is ALREADY excluded (``processing.exclude_breaths`` /
    ``breaths[n]['ignored']``) still gets a notice ONLY when drift correction is on:
    excluding a breath removes it from the reported metrics, but ``correctdrift``
    anchors on the recording's raw first/last SAMPLE regardless of which breaths are
    excluded (see ``correctdrift`` above), so the volume baseline of every OTHER
    breath can still be tilted even after exclusion. The wording differs for this
    case (it does not claim the excluded breath is "analysed as if complete", which
    would now be false, and does not suggest excluding a breath that is already
    excluded) — with drift correction off, an already-excluded truncated boundary
    breath has no remaining consequence worth a notice, so none is raised.

    Returns a list of 0, 1 or 2 human-readable notice strings — the same shape as the
    ``FileResult.notices`` list the other per-file quality notices (ecg_auto_detect
    mismatch, cardiac-gated peak EMG) already populate, so this slots into the same
    report section and warning plumbing without a new mechanism.
    """
    if min_relative_duration is None:
        min_relative_duration = getattr(
            settings.processing.mechanics, "boundarynoticeminrelativeduration", 0.8)
    if min_other_breaths is None:
        min_other_breaths = getattr(
            settings.processing.mechanics, "boundarynoticeminotherbreaths", 3)

    numbers = sorted(breaths)
    if len(numbers) < 2:
        return []
    fs = float(settings.input.format.samplingfrequency)

    def _phase_seconds(bno, phase):
        return len(np.atleast_1d(breaths[bno][phase]["time"])) / fs

    drift_on = bool(settings.processing.mechanics.correctvolumedrift)

    def _notice(edge, phase, cur, median, ignored, direction, likely):
        if ignored:
            if not drift_on:
                return None
            return (
                f"the {edge} breath's {phase} ({cur:.2f} s) is much shorter than this "
                f"file's typical {phase} ({median:.2f} s) — it is already excluded "
                "from the analysis, but drift correction anchors on the recording's "
                "raw first and last sample regardless of which breaths are excluded, "
                "so the volume baseline of the OTHER breaths in this file may still "
                f"be tilted. Re-export the epoch so it {direction} to fix this at "
                "the source.")
        drift_tail = (" With drift correction on, this also tilts the volume baseline "
                      "of every breath in the file, not just this one." if drift_on else "")
        return (
            f"the {edge} breath's {phase} ({cur:.2f} s) is much shorter than this "
            f"file's typical {phase} ({median:.2f} s) — the recording likely {likely}, "
            f"so the {edge} breath is truncated and analysed as if it were complete."
            + drift_tail + f" Re-export the epoch so it {direction}, or exclude the "
            f"{edge} breath in Preview & QC.")

    notices = []
    first_no, last_no = numbers[0], numbers[-1]

    other_insp = [_phase_seconds(no, "inspiration") for no in numbers if no != first_no]
    if len(other_insp) >= min_other_breaths:
        median_insp = float(np.median(other_insp))
        first_insp = _phase_seconds(first_no, "inspiration")
        if median_insp > 0 and first_insp < min_relative_duration * median_insp:
            notice = _notice("first", "inspiration", first_insp, median_insp,
                             bool(breaths[first_no]["ignored"]),
                             "starts in expiration", "begins mid-inspiration")
            if notice:
                notices.append(notice)

    other_exp = [_phase_seconds(no, "expiration") for no in numbers if no != last_no]
    if len(other_exp) >= min_other_breaths:
        median_exp = float(np.median(other_exp))
        last_exp = _phase_seconds(last_no, "expiration")
        if median_exp > 0 and last_exp < min_relative_duration * median_exp:
            notice = _notice("last", "expiration", last_exp, median_exp,
                             bool(breaths[last_no]["ignored"]),
                             "ends in inspiration", "ends mid-expiration")
            if notice:
                notices.append(notice)

    return notices


# --- pressure-time product & integration -----------------------------------

def calcptp(pressure, bcnt, vefactor, samplingfreq, baseline_samples=1):
    # Pressure-time product: integrate the pressure relative to its end-expiratory
    # baseline (see docs/PTP_INVESTIGATION.md). The baseline is the mean over a short
    # window at the phase start (``baseline_samples``), which is robust to boundary
    # noise; baseline_samples=1 reproduces the single-sample behaviour.
    pressure = pressure.squeeze()
    n = int(max(1, min(baseline_samples, len(pressure))))
    baseline = np.mean(pressure[:n])
    pressure = pressure - baseline
    xval = np.linspace(0, len(pressure) / samplingfreq, len(pressure))
    integral = sp.integrate.simpson(pressure, x=xval)
    ptp = integral * bcnt * vefactor
    return ptp, integral


# --- work of breathing (Campbell diagram) ----------------------------------

def calculatewob(breath, bcnt, vefactor, settings):
    WOBUNITCHANGEFACTOR = 98.0638 / 1000  # cmH2O -> Joule; Pa = J / m3
    if settings.processing.wob.calcwobfrom == "average":
        volin = breath["inspiration"]["volumeavg"]
        volex = breath["expiration"]["volumeavg"]
        poesin = breath["inspiration"]["poesavg"]
        poesex = breath["expiration"]["poesavg"]
    else:
        volin = breath["inspiration"]["volume"]
        volex = breath["expiration"]["volume"]
        poesin = breath["inspiration"]["poes"]
        poesex = breath["expiration"]["poes"]

    eilv = [volin[len(poesin) - 1], poesin[len(poesin) - 1]]
    eelv = [volex[len(volex) - 1], poesex[len(volex) - 1]]

    # Inspiratory elastic WOB
    tbase = abs(eilv[0] - eelv[0])
    theight = abs(eilv[1] - eelv[1])
    wobinela = tbase * theight / 2 * WOBUNITCHANGEFACTOR

    # Inspiratory resistive WOB
    slope = (poesin[len(poesin) - 1] - poesin[0]) / (volin[len(volin) - 1] - volin[0])
    flyin = volin * slope + poesin[0]
    levelpoesin = (poesin * -1) - (flyin * -1)
    levelpoesin[np.where(levelpoesin < 0)] = 0
    wobinres = max(abs(sp.integrate.simpson(levelpoesin, x=volin)), 0) * WOBUNITCHANGEFACTOR

    # Expiratory WOB
    levelpoesex = poesex - poesex[len(poesex) - 1]
    levelpoesex[np.where(levelpoesex < 0)] = 0
    wobex = max(abs(sp.integrate.simpson(levelpoesex, x=volex)), 0) * WOBUNITCHANGEFACTOR

    wobin = wobinela + wobinres
    wobtotal = wobin + wobex
    return OrderedDict([
        ('wobtotal', wobtotal * bcnt * vefactor),
        ('wob_in_total', wobin * bcnt * vefactor),
        ('wob_ex_total', wobex * bcnt * vefactor),
        ('wob_in_ela', wobinela * bcnt * vefactor),
        ('wob_in_res', wobinres * bcnt * vefactor),
    ])


# --- averaging -------------------------------------------------------------

def resample(x, settings, kind='linear'):
    x = x.squeeze()
    n = settings.processing.wob.avgresamplingobs
    f = sp.interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))


def calculateaveragebreaths(breaths, settings):
    # boundarynotice-idiom (compute.py's own trim_boundary_notices): a hand-built
    # SimpleNamespace test double need not carry `capabilities` at all -> treat it as FULL.
    caps = getattr(settings, "capabilities", Capabilities.FULL)
    resamplingobs = settings.processing.wob.avgresamplingobs
    nobreaths = sum(1 for b in breaths.values() if not b["ignored"])
    volumein = np.empty([resamplingobs, nobreaths])
    volumeex = np.empty([resamplingobs, nobreaths])
    poesin = np.empty([resamplingobs, nobreaths]) if caps.poes else None
    poesex = np.empty([resamplingobs, nobreaths]) if caps.poes else None
    for breathno in breaths:
        breath = breaths[breathno]
        if not breath["ignored"]:
            nobreaths -= 1
            try:
                volumein[:, nobreaths] = resample(breath["inspiration"]["volume"], settings)
                volumeex[:, nobreaths] = resample(breath["expiration"]["volume"], settings)
                if caps.poes:
                    poesin[:, nobreaths] = resample(breath["inspiration"]["poes"], settings)
                    poesex[:, nobreaths] = resample(breath["expiration"]["poes"], settings)
            except Exception as e:
                raise ValueError(
                    "Could not resample breath #" + str(breath["number"]) +
                    ": it is too short to average. Check Preview & QC ▸ Mechanics ▸ "
                    "Advanced… ▸ Breath detection (peak thresholds / breath-separation "
                    "buffer), or exclude this breath in Preview & QC.") from e
    avgpoesin = np.mean(poesin, axis=1) if caps.poes else None
    avgpoesex = np.mean(poesex, axis=1) if caps.poes else None
    return (np.mean(volumein, axis=1), np.mean(volumeex, axis=1), avgpoesin, avgpoesex)


# --- entropy ---------------------------------------------------------------

def conditioned_entropy_columns(entropycolumns, volume, settings):
    """The entropy input matrix with the conditioned volume worked in (run before breath
    segmentation, so the whole-breath, inspiration and expiration windows all slice it).

    Entropy on a column that IS the volume column is taken on the volume RespMech itself
    analyses (zeroed, drift- and trend-corrected as configured), not on the raw file column
    (an EMG column that is also an entropy column is handled the same way, per breath, in
    ``calculateentropy``). Every signal in ``input.channels.entropy_derived`` (today only
    "volume") is appended as an extra column after the file columns: the same conditioned
    volume, for a volume that has no column of its own. Returns ``entropycolumns`` itself,
    untouched, when neither applies (so every other analysis is byte-identical).
    """
    data = settings.input.data
    vol_col = getattr(data, "column_volume", None)
    derived = list(getattr(data, "entropy_derived", []) or [])
    entcols = list(data.columns_entropy)
    emg = list(data.columns_emg)
    volume = np.asarray(volume, dtype=float)
    if volume.size == 0:
        return entropycolumns
    on_volume = [i for i, c in enumerate(entcols)
                 if vol_col is not None and not np.isnan(vol_col) and c == vol_col and c not in emg]
    if not on_volume and not derived:
        return entropycolumns
    base = np.asarray(entropycolumns, dtype=float)
    if base.size == 0:
        base = np.empty((volume.size, 0))
    out = np.column_stack([base] + [volume] * len(derived)) if derived else base.copy()
    for i in on_volume:
        out[:, i] = volume
    return out


def calculateentropy(breath, settings, phase=None, cancel_check=None):
    if phase is None:
        columns = breath["entcols"]
    else:
        columns = breath[phase]["entcols"]
    columns = np.array(columns, dtype=float)
    if columns.ndim == 1:
        columns = columns.reshape(-1, 1)

    # If EMG columns are also entropy columns, use the processed (not raw) data.
    if len(settings.input.data.columns_emg) > 0:
        emgcolnos = settings.input.data.columns_emg
        for entcolno in range(0, len(settings.input.data.columns_entropy)):
            entc = settings.input.data.columns_entropy[entcolno]
            if entc in emgcolnos:
                src = breath["emgcols"] if phase is None else breath[phase]["emgcols"]
                columns[:, entcolno] = np.asarray(src)[:, emgcolnos.index(entc)]

    epoch = settings.processing.entropy.entropy_epochs
    tolerancesd = settings.processing.entropy.entropy_tolerance
    sampen = np.zeros(columns.shape[1])
    for i in range(0, columns.shape[1]):
        std_ds = np.std(columns[:, i])
        se = entlib.sample_entropy(columns[:, i], epoch, tolerancesd * std_ds, cancel_check=cancel_check)
        sampen[i] = se[len(se) - 1]
    return sampen


# --- per-breath mechanics (the big one) ------------------------------------

def _add_gated_peaks(retbreath, breath, settings, peaks_s, detection_ok, detection_reason, phases=True):
    """Attach the opt-in cardiac-gated peak RMS for whole breath / inspiration / expiration.

    Does nothing at all unless processing.emg.robust_peak.enabled, so with the feature off no
    new keys appear and the result DataFrames are untouched. ``peaks_s`` are ABSOLUTE R-peak
    times in seconds; breath["time"] runs on the same absolute clock (compute.trim does not
    rebase it), so mapping to phase-local samples is a plain subtraction.

    ``phases=False`` (a phase-less segment, e.g. an EMG-only whole-file segment with no
    inspiration/expiration split) skips the computation entirely rather than indexing
    ``breath["inspiration"]``/``breath["expiration"]``, which do not exist for such a segment:
    all three keys go NaN-with-reason, uniformly, the same shape as the existing "no usable
    R-peaks" branch below.
    """
    rp = getattr(settings.processing.emg, "robust_peak", None)
    if rp is None or not rp.enabled:
        return
    fs = settings.input.format.samplingfrequency
    nch = len(settings.input.data.columns_emg)
    keys = ("rms_gated", "rms_gated_insp", "rms_gated_exp")

    if not phases:
        for key in keys:
            retbreath[key] = [float("nan")] * (nch + 2)
        retbreath["rms_gated_qc"] = {"ok": False, "reason": "segment has no phases"}
        return

    peaks = np.asarray(peaks_s, dtype=float) if peaks_s is not None else None

    if peaks is None or peaks.size == 0 or not detection_ok:
        reason = detection_reason or ("no R-peaks available — is processing.emg.remove_ecg on?"
                                      if peaks is None or peaks.size == 0 else "")
        for key in keys:
            retbreath[key] = [float("nan")] * (nch + 2)
        retbreath["rms_gated_qc"] = {"ok": False, "reason": reason}
        return

    qc_all = {"ok": True, "reason": ""}
    for key, seg in zip(keys, (breath, breath["inspiration"], breath["expiration"])):
        cols = seg["emgcols"]
        t0 = float(np.asarray(seg["time"], dtype=float)[0])
        local = (peaks - t0) * fs
        vals, qc = emglib.gated_peak_rms(
            cols, local, settings.processing.emg.rms_s, fs,
            gate_half_width_s=rp.gate_half_width_s, min_survival=rp.min_survival,
            min_island_s=rp.min_island_s)
        retbreath[key] = vals
        if not qc["ok"]:
            qc_all = {"ok": False, "reason": f"{key}: {qc['reason']}"}
    retbreath["rms_gated_qc"] = qc_all


def compute_segment_emg(retbreath, breath, settings, cancel_check, peaks_s, detection_ok, detection_reason,
                        phases=True):
    """EMG (RMS/integral/gated-peak) and sample-entropy features for one breath or segment.

    A verbatim extraction of what ``calculatemechanics`` used to compute inline, so it can
    be called per-segment (a segment that may not have an inspiration/expiration split —
    ``phases=False``, an EMG-only ``whole_file``/``separators`` segment from
    ``core.analysis.segments``) as well as per-breath (the call site in
    ``calculatemechanics``, always ``phases=True``). On the ``phases=True`` path every
    statement below runs exactly as it always did (golden-safe); ``phases=False`` computes
    only the WHOLE-segment quantities (R8: the segment's own ``rms``/``intemg``/``entropy``,
    identical in shape to a phase-having breath's own whole-breath values) and skips the
    inspiration/expiration-specific variants, which do not exist for such a segment."""
    if len(breath["emgcols"]) > 0:
        retbreath["rms"], retbreath["intemg"] = emglib.calculate_rms(breath["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
        if phases:
            retbreath["rms_insp"], retbreath["intemg_insp"] = emglib.calculate_rms(breath["inspiration"]["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
            retbreath["rms_exp"], retbreath["intemg_exp"] = emglib.calculate_rms(breath["expiration"]["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
        _add_gated_peaks(retbreath, breath, settings, peaks_s, detection_ok, detection_reason, phases=phases)

    n_file = len(settings.input.data.columns_entropy)
    n_derived = len(getattr(settings.input.data, "entropy_derived", []) or []) if phases else 0
    if n_file + n_derived > 0:
        # ``entcols`` carries the file columns first, then the derived signals
        # (``conditioned_entropy_columns``). Derived entropy is reported per column but kept
        # out of sample_entropy_max/min/mean, which stay a summary of the file columns only.
        def _split(entropy):
            entropy = np.asarray(entropy, dtype=float)
            return entropy[:n_file], entropy[n_file:]

        entropy, derived = _split(calculateentropy(breath, settings, cancel_check=cancel_check))
        retbreath["entropy"] = (np.append(entropy.T, [max(entropy.T), min(entropy.T), np.mean(entropy.T)])
                                if n_file > 0 else [])
        if n_derived:
            retbreath["entropy_derived"] = derived
        if phases:
            for phase, key in (("inspiration", "entropy_insp"), ("expiration", "entropy_exp")):
                entropy_p, derived_p = _split(calculateentropy(breath, settings, phase, cancel_check=cancel_check))
                if n_file > 0:
                    retbreath[key] = np.append(entropy_p.T, [max(entropy_p.T), min(entropy_p.T), np.mean(entropy_p.T)])
                if n_derived:
                    retbreath["entropy_derived_" + key.split("_")[1]] = derived_p
    else:
        retbreath["entropy"] = []


def calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings,
                       cancel_check=None, peaks_s=None, detection_ok=True, detection_reason=""):
    """Optional Poes/Pgas/Pdi (v2-only, docs/REVERSE_ENGINEERING.md §5.11): every value that
    reads one of those three channels is computed only when ``caps`` (from
    ``settings.capabilities``, defaulting to :data:`Capabilities.FULL` for a hand-built
    namespace) says the channel is present — the phase dicts carry an EMPTY array for an
    absent channel (core/io/loaders.py), so an unguarded ``max()``/index into it would raise.
    On the full-channel path every ``if caps.x:`` guard below is True and executes exactly
    the same statements as before this ticket, so golden output is unchanged. The final
    OrderedDict is assembled by walking ``LEGACY_MECHANICS_ORDER`` (core/analysis/registry.py)
    and taking each name that made it into ``values``, instead of a hardcoded literal — the
    key ORDER is still pinned by that same list (tests/unit/test_analysis_registry.py)."""
    check(cancel_check)   # per-breath abort point (no-op when cancel_check is None -> golden-safe)
    caps = getattr(settings, "capabilities", Capabilities.FULL)
    retbreath = breath
    retbreath["inspiration"]["volumeavg"] = avgvolumein
    retbreath["expiration"]["volumeavg"] = avgvolumeex
    retbreath["volumeavg"] = np.concatenate([avgvolumein, avgvolumeex])
    if caps.poes:
        retbreath["inspiration"]["poesavg"] = avgpoesin
        retbreath["expiration"]["poesavg"] = avgpoesex
        retbreath["poesavg"] = np.concatenate([avgpoesin, avgpoesex])

    insp = retbreath["inspiration"]
    exp = retbreath["expiration"]

    retbreath["eilv"] = [insp["volume"][-1], insp["poes"][-1] if caps.poes else np.nan]
    retbreath["eelv"] = [exp["volume"][-1], exp["poes"][-1] if caps.poes else np.nan]
    retbreath["eilvavg"] = [insp["volumeavg"][-1], insp["poesavg"][-1] if caps.poes else np.nan]
    retbreath["eelvavg"] = [exp["volumeavg"][-1], exp["poesavg"][-1] if caps.poes else np.nan]

    values = {}

    if caps.poes:
        values["poes_maxexp"] = max(exp["poes"])
        values["poes_endexp"] = exp["poes"][len(exp["poes"]) - 1]
    if caps.pdi:
        values["pdi_minexp"] = min(exp["pdi"])
        values["pdi_endexp"] = exp["pdi"][len(exp["pdi"]) - 1]
    if caps.pgas:
        values["pgas_endexp"] = exp["pgas"][len(exp["pgas"]) - 1]
        values["pgas_maxexp"] = max(exp["pgas"])
        values["pgas_minexp"] = min(exp["pgas"])

    midvolexp = min(exp["volume"]) + ((max(exp["volume"]) - min(exp["volume"])) / 2)
    midvolexpix = np.where(exp["volume"] <= midvolexp)[0][0]
    if caps.poes:
        values["poes_midvolexp"] = exp["poes"][midvolexpix]
    values["flow_midvolexp"] = -exp["flow"][midvolexpix]

    if caps.poes:
        values["poes_mininsp"] = min(insp["poes"])
        values["poes_endinsp"] = insp["poes"][len(insp["poes"]) - 1]
    if caps.pdi:
        values["pdi_maxinsp"] = max(insp["pdi"])
        values["pdi_endinsp"] = insp["pdi"][len(insp["pdi"]) - 1]
    if caps.pgas:
        values["pgas_endinsp"] = insp["pgas"][len(insp["pgas"]) - 1]

    if caps.poes:
        values["poes_tidal_swing"] = abs(max(retbreath["poes"]) - min(retbreath["poes"]))
    if caps.pgas:
        values["pgas_tidal_swing"] = abs(max(retbreath["pgas"]) - min(retbreath["pgas"]))
    if caps.pdi:
        values["pdi_tidal_swing"] = abs(max(retbreath["pdi"]) - min(retbreath["pdi"]))

    midvolinsp = min(insp["volume"]) + ((max(insp["volume"]) - min(insp["volume"])) / 2)
    midvolinspix = np.where(insp["volume"] >= midvolinsp)[0][0]
    if caps.poes:
        values["poes_midvolinsp"] = insp["poes"][midvolinspix]
    values["flow_midvolinsp"] = -insp["flow"][midvolinspix]

    values["vol_endinsp"] = insp["volume"][len(insp["volume"]) - 1]
    values["vol_endexp"] = exp["volume"][len(exp["volume"]) - 1]

    values["ti"] = len(insp["flow"]) / settings.input.format.samplingfrequency
    values["te"] = len(exp["flow"]) / settings.input.format.samplingfrequency
    values["ttot"] = len(retbreath["flow"]) / settings.input.format.samplingfrequency
    values["ti_ttot"] = values["ti"] / values["ttot"]

    values["vt"] = max(retbreath["volume"]) - min(retbreath["volume"])
    values["ve"] = values["vt"] * bcnt * vefactor

    if caps.poes and caps.pgas:
        vmrnumerator = (values["pgas_endinsp"] - values["pgas_endexp"])
        vmrdenominator = (values["poes_endinsp"] - values["poes_endexp"])
        # dtype=float is an extra safeguard, not the primary fix: the loader (core/io/loaders.py)
        # now casts every channel to float64 on load, so vmrnumerator/vmrdenominator should
        # already be float here. This keeps the division itself from raising even if some future
        # caller feeds compute_breath() an int array directly.
        values["vmr"] = np.divide(vmrnumerator, vmrdenominator,
                                   out=np.zeros_like(vmrnumerator, dtype=float), where=vmrdenominator != 0)

    if caps.poes:
        values["tlr_insp"] = abs(
            (values["poes_midvolexp"] - values["poes_midvolinsp"])
            / (values["flow_midvolexp"] - values["flow_midvolinsp"]))
    if caps.pdi:
        values["insp_pdi_rise"] = values["pdi_maxinsp"] - min(insp["pdi"])
    if caps.pgas:
        values["exp_pgas_rise"] = values["pgas_maxexp"] - min(exp["pgas"])

    # PTP is integrated relative to the end-expiratory baseline inside calcptp
    # (mean over a short window at the phase start). The former adjustforintegration
    # / "- min" pre-steps were redundant (integ(f - f[0]) is invariant to a constant
    # pre-shift — see docs/PTP_INVESTIGATION.md); the signed signals are passed
    # directly (Poes negated so inspiratory effort is positive).
    fs = settings.input.format.samplingfrequency
    ptp_bw = int(max(1, round(settings.processing.mechanics.ptp_baseline_window_s * fs)))
    if caps.poes:
        values["ptp_oesinsp"], values["int_oesinsp"] = calcptp(-insp["poes"], bcnt, vefactor, fs, ptp_bw)
    if caps.pdi:
        values["ptp_pdiinsp"], values["int_pdiinsp"] = calcptp(insp["pdi"], bcnt, vefactor, fs, ptp_bw)
    if caps.pgas:
        values["ptp_pgasexp"], values["int_pgasexp"] = calcptp(exp["pgas"], bcnt, vefactor, fs, ptp_bw)

    values["max_in_flow"] = min(insp["flow"]) * -1
    values["max_ex_flow"] = max(exp["flow"])
    values["in_flow_midvol"] = insp["flow"][midvolinspix] * -1
    values["ex_flow_midvol"] = exp["flow"][midvolexpix]

    values["bf"] = bcnt * vefactor

    if caps.poes:
        retbreath["wob"] = calculatewob(breath, bcnt, vefactor, settings)

    compute_segment_emg(retbreath, breath, settings, cancel_check, peaks_s, detection_ok, detection_reason)

    retbreath["mechanics"] = OrderedDict(
        (spec.name, values[spec.name]) for spec in LEGACY_MECHANICS_ORDER if spec.name in values
    )
    return retbreath

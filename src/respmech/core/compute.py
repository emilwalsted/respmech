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

from respmech.core import emg as emglib
from respmech.core import entropy as entlib
from respmech.core._cancel import check


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


class DegenerateBreathError(ValueError):
    """Raised when a detected breath's inspiration and expiration phases cannot be
    joined into one breath (a precondition failure, not a bug): typically an
    incomplete breath right at the start or end of a recording."""


def _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename):
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
        raise DegenerateBreathError(
            f"Breath #{breathcnt} in {filename} is degenerate (its inspiration and "
            f"expiration phases could not be joined into one breath) and cannot be "
            f"analysed. This usually happens at an incomplete breath right at the "
            f"start or end of a recording — exclude it in Preview & QC, or check the "
            f"breath-separation settings under Preview & QC ▸ Mechanics ▸ Advanced… ▸ "
            f"Breath detection.") from e
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
    breaths = OrderedDict()
    j = len(flow)
    bufferwidth = settings.processing.mechanics.breathseparationbuffer
    ib = ignorebreaths(filename, settings)
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
        exp, insp, entcols, emgcols = _phase_dicts(
            (instart, inend), (exstart, exend), timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns)
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
        breaths[breathcnt] = _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename)
    return breaths


def separateintobreathsbyvolume(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    from scipy import signal
    breaths = OrderedDict()
    ib = ignorebreaths(filename, settings)
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
        exp, insp, entcols, emgcols = _phase_dicts(
            (instart, inend), (exstart, exend), timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns)
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
        breaths[breathcnt] = _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename)
    return breaths


def separateintobreaths(method, filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    if str(str.lower(method)) == "volume":
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

    **Follow-up investigation (ticket 20260906-1307, a review raised after this function
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

    The decision reached by this investigation (ticket 20260906-1307 — not yet reviewed or
    confirmed by Emil, unlike the earlier decisions elsewhere in this codebase that carry
    his name): keep 0.8/3 as the default, documented here as a known, accepted trade-off
    rather than a proven-safe value, and expose both numbers as a per-analysis setting (see
    above) so a study whose recordings are known to have unusually high natural variability
    can raise the threshold (e.g. to 0.6-0.7) deliberately, instead of the whole install
    silently trading detection power away for every user based on one un-validated
    system-wide guess. Real research recordings (``tests/golden/production``, unavailable
    in the sandbox this investigation ran in) would let a future session replace this
    entire trade-off analysis with a measured threshold instead — see the ticket's own
    "Opgave" for the preferred path if that data becomes reachable.

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
    ``FileResult.notices`` list the K-192/K-224 quality notices already populate, so
    this slots into the same report section and warning plumbing without a new
    mechanism.
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
    resamplingobs = settings.processing.wob.avgresamplingobs
    nobreaths = sum(1 for b in breaths.values() if not b["ignored"])
    volumein = np.empty([resamplingobs, nobreaths])
    volumeex = np.empty([resamplingobs, nobreaths])
    poesin = np.empty([resamplingobs, nobreaths])
    poesex = np.empty([resamplingobs, nobreaths])
    for breathno in breaths:
        breath = breaths[breathno]
        if not breath["ignored"]:
            nobreaths -= 1
            try:
                volumein[:, nobreaths] = resample(breath["inspiration"]["volume"], settings)
                volumeex[:, nobreaths] = resample(breath["expiration"]["volume"], settings)
                poesin[:, nobreaths] = resample(breath["inspiration"]["poes"], settings)
                poesex[:, nobreaths] = resample(breath["expiration"]["poes"], settings)
            except Exception as e:
                raise ValueError(
                    "Could not resample breath #" + str(breath["number"]) +
                    ": it is too short to average. Check Preview & QC ▸ Mechanics ▸ "
                    "Advanced… ▸ Breath detection (peak thresholds / breath-separation "
                    "buffer), or exclude this breath in Preview & QC.") from e
    return (np.mean(volumein, axis=1), np.mean(volumeex, axis=1),
            np.mean(poesin, axis=1), np.mean(poesex, axis=1))


# --- entropy ---------------------------------------------------------------

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

def _add_gated_peaks(retbreath, breath, settings, peaks_s, detection_ok, detection_reason):
    """Attach the opt-in cardiac-gated peak RMS for whole breath / inspiration / expiration.

    Does nothing at all unless processing.emg.robust_peak.enabled, so with the feature off no
    new keys appear and the result DataFrames are untouched. ``peaks_s`` are ABSOLUTE R-peak
    times in seconds; breath["time"] runs on the same absolute clock (compute.trim does not
    rebase it), so mapping to phase-local samples is a plain subtraction.
    """
    rp = getattr(settings.processing.emg, "robust_peak", None)
    if rp is None or not rp.enabled:
        return
    fs = settings.input.format.samplingfrequency
    nch = len(settings.input.data.columns_emg)
    keys = ("rms_gated", "rms_gated_insp", "rms_gated_exp")
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


def calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings,
                       cancel_check=None, peaks_s=None, detection_ok=True, detection_reason=""):
    check(cancel_check)   # per-breath abort point (no-op when cancel_check is None -> golden-safe)
    retbreath = breath
    retbreath["inspiration"]["volumeavg"] = avgvolumein
    retbreath["expiration"]["volumeavg"] = avgvolumeex
    retbreath["volumeavg"] = np.concatenate([avgvolumein, avgvolumeex])
    retbreath["inspiration"]["poesavg"] = avgpoesin
    retbreath["expiration"]["poesavg"] = avgpoesex
    retbreath["poesavg"] = np.concatenate([avgpoesin, avgpoesex])

    retbreath["eilv"] = [retbreath["inspiration"]["volume"][-1], retbreath["inspiration"]["poes"][-1]]
    retbreath["eelv"] = [retbreath["expiration"]["volume"][-1], retbreath["expiration"]["poes"][-1]]
    retbreath["eilvavg"] = [retbreath["inspiration"]["volumeavg"][-1], retbreath["inspiration"]["poesavg"][-1]]
    retbreath["eelvavg"] = [retbreath["expiration"]["volumeavg"][-1], retbreath["expiration"]["poesavg"][-1]]

    exp = retbreath["expiration"]
    poes_maxexp = max(exp["poes"])
    poes_endexp = exp["poes"][len(exp["poes"]) - 1]
    pdi_minexp = min(exp["pdi"])
    pdi_endexp = exp["pdi"][len(exp["pdi"]) - 1]
    pgas_endexp = exp["pgas"][len(exp["pgas"]) - 1]
    pgas_maxexp = max(exp["pgas"])
    pgas_minexp = min(exp["pgas"])

    midvolexp = min(exp["volume"]) + ((max(exp["volume"]) - min(exp["volume"])) / 2)
    midvolexpix = np.where(exp["volume"] <= midvolexp)[0][0]
    poes_midvolexp = exp["poes"][midvolexpix]
    flow_midvolexp = -exp["flow"][midvolexpix]

    insp = retbreath["inspiration"]
    poes_mininsp = min(insp["poes"])
    poes_endinsp = insp["poes"][len(insp["poes"]) - 1]
    pdi_maxinsp = max(insp["pdi"])
    pdi_endinsp = insp["pdi"][len(insp["pdi"]) - 1]
    pgas_endinsp = insp["pgas"][len(insp["pgas"]) - 1]

    poes_tidal_swing = abs(max(retbreath["poes"]) - min(retbreath["poes"]))
    pgas_tidal_swing = abs(max(retbreath["pgas"]) - min(retbreath["pgas"]))
    pdi_tidal_swing = abs(max(retbreath["pdi"]) - min(retbreath["pdi"]))

    midvolinsp = min(insp["volume"]) + ((max(insp["volume"]) - min(insp["volume"])) / 2)
    midvolinspix = np.where(insp["volume"] >= midvolinsp)[0][0]
    poes_midvolinsp = insp["poes"][midvolinspix]
    flow_midvolinsp = -insp["flow"][midvolinspix]

    vol_endinsp = insp["volume"][len(insp["volume"]) - 1]
    vol_endexp = exp["volume"][len(exp["volume"]) - 1]

    ti = len(insp["flow"]) / settings.input.format.samplingfrequency
    te = len(exp["flow"]) / settings.input.format.samplingfrequency
    ttot = len(retbreath["flow"]) / settings.input.format.samplingfrequency
    ti_ttot = ti / ttot

    vt = max(retbreath["volume"]) - min(retbreath["volume"])
    ve = vt * bcnt * vefactor

    vmrnumerator = (pgas_endinsp - pgas_endexp)
    vmrdenominator = (poes_endinsp - poes_endexp)
    vmr = np.divide(vmrnumerator, vmrdenominator, out=np.zeros_like(vmrnumerator), where=vmrdenominator != 0)

    tlr_insp = abs((poes_midvolexp - poes_midvolinsp) / (flow_midvolexp - flow_midvolinsp))
    insp_pdi_rise = pdi_maxinsp - min(insp["pdi"])
    exp_pgas_rise = pgas_maxexp - min(exp["pgas"])

    # PTP is integrated relative to the end-expiratory baseline inside calcptp
    # (mean over a short window at the phase start). The former adjustforintegration
    # / "- min" pre-steps were redundant (integ(f - f[0]) is invariant to a constant
    # pre-shift — see docs/PTP_INVESTIGATION.md); the signed signals are passed
    # directly (Poes negated so inspiratory effort is positive).
    fs = settings.input.format.samplingfrequency
    ptp_bw = int(max(1, round(settings.processing.mechanics.ptp_baseline_window_s * fs)))
    ptp_oesinsp, int_oesinsp = calcptp(-insp["poes"], bcnt, vefactor, fs, ptp_bw)
    ptp_pdiinsp, int_pdiinsp = calcptp(insp["pdi"], bcnt, vefactor, fs, ptp_bw)
    ptp_pgasexp, int_pgasexp = calcptp(exp["pgas"], bcnt, vefactor, fs, ptp_bw)

    max_in_flow = min(insp["flow"]) * -1
    max_ex_flow = max(exp["flow"])
    inflowmidvol = insp["flow"][midvolinspix] * -1
    exflowmidvol = exp["flow"][midvolexpix]

    retbreath["wob"] = calculatewob(breath, bcnt, vefactor, settings)

    if len(breath["emgcols"]) > 0:
        retbreath["rms"], retbreath["intemg"] = emglib.calculate_rms(breath["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
        retbreath["rms_insp"], retbreath["intemg_insp"] = emglib.calculate_rms(breath["inspiration"]["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
        retbreath["rms_exp"], retbreath["intemg_exp"] = emglib.calculate_rms(breath["expiration"]["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
        _add_gated_peaks(retbreath, breath, settings, peaks_s, detection_ok, detection_reason)

    if len(settings.input.data.columns_entropy) > 0:
        entropy = calculateentropy(breath, settings, cancel_check=cancel_check)
        entropy_insp = calculateentropy(breath, settings, "inspiration", cancel_check=cancel_check)
        entropy_exp = calculateentropy(breath, settings, "expiration", cancel_check=cancel_check)
        retbreath["entropy"] = np.append(entropy.T, [max(entropy.T), min(entropy.T), np.mean(entropy.T)])
        retbreath["entropy_insp"] = np.append(entropy_insp.T, [max(entropy_insp.T), min(entropy_insp.T), np.mean(entropy_insp.T)])
        retbreath["entropy_exp"] = np.append(entropy_exp.T, [max(entropy_exp.T), min(entropy_exp.T), np.mean(entropy_exp.T)])
    else:
        retbreath["entropy"] = []

    retbreath["mechanics"] = OrderedDict([
        ('poes_maxexp', poes_maxexp), ('poes_mininsp', poes_mininsp),
        ('poes_endinsp', poes_endinsp), ('poes_endexp', poes_endexp),
        ('poes_midvolexp', poes_midvolexp), ('poes_midvolinsp', poes_midvolinsp),
        ('int_oesinsp', int_oesinsp), ('ptp_oesinsp', ptp_oesinsp),
        ('poes_tidal_swing', poes_tidal_swing),
        ('pgas_endinsp', pgas_endinsp), ('pgas_endexp', pgas_endexp),
        ('pgas_maxexp', pgas_maxexp), ('pgas_minexp', pgas_minexp),
        ('exp_pgas_rise', exp_pgas_rise), ('int_pgasexp', int_pgasexp),
        ('ptp_pgasexp', ptp_pgasexp), ('pgas_tidal_swing', pgas_tidal_swing),
        ('int_pdiinsp', int_pdiinsp), ('ptp_pdiinsp', ptp_pdiinsp),
        ('pdi_minexp', pdi_minexp), ('pdi_maxinsp', pdi_maxinsp),
        ('pdi_endinsp', pdi_endinsp), ('pdi_endexp', pdi_endexp),
        ('insp_pdi_rise', insp_pdi_rise), ('pdi_tidal_swing', pdi_tidal_swing),
        ('flow_midvolexp', flow_midvolexp), ('flow_midvolinsp', flow_midvolinsp),
        ('vol_endinsp', vol_endinsp), ('vol_endexp', vol_endexp),
        ('max_in_flow', max_in_flow), ('max_ex_flow', max_ex_flow),
        ('in_flow_midvol', inflowmidvol), ('ex_flow_midvol', exflowmidvol),
        ('ti', ti), ('te', te), ('ttot', ttot), ('ti_ttot', ti_ttot),
        ('vt', vt), ('bf', bcnt * vefactor), ('ve', ve),
        ('vmr', vmr), ('tlr_insp', tlr_insp),
    ])
    return retbreath

"""EMG signal processing — computation only (no plotting).

Faithful port of the validated legacy ``emg.py`` compute functions:
RMS + integrated EMG, ECG removal (averaged-template subtraction) and spectral
noise reduction. Plotting/sound export live elsewhere (``respmech.core.plots``).

One deliberate, documented change vs legacy ``master``:
* ``scipy.integrate.simpson`` is called with the keyword ``x=`` (positional ``x`` is
  removed in SciPy >= 1.14) — numerically identical (legacy bug #4).

The spectral noise-reduction helpers below (adapted, with permission, from code by
Tim Sainburg, https://timsainburg.com; require ``librosa``) are shared with the live
shared-profile path in ``core.noise``.
"""
import numpy as np
import scipy as sp
from scipy import signal

# librosa is only needed when noise reduction is used; the STFT helpers import it
# lazily so ECG removal / RMS work without it.
np.float = float  # compat shim for older third-party code paths


def calculate_rms(emgchannels, rms_s, samplingfrequency):
    rms = []
    intemg = []
    for emgch in emgchannels.T:
        # Root mean square (max sliding-window RMS over the breath)
        rollingwindowsize = int(rms_s * samplingfrequency)
        rollstart = 0
        rollend = rollingwindowsize - 1

        r = []
        try:
            for i in range(0 + int(rollingwindowsize / 2), len(emgch) - int(rollingwindowsize / 2)):
                r += [np.sqrt(np.mean(np.square(emgch[i + rollstart:i + rollend])))]
            rmsch = np.max(r)
        except Exception:
            # If breath is too small (i.e. invalid and should be ignored)
            rmsch = 0
        rms += [rmsch]

        # Integral of |EMG| over time
        intch = np.abs(emgch)
        xval = np.linspace(0, len(emgch) - 1, len(emgch)) / samplingfrequency
        intemg = np.append(intemg, [sp.integrate.simpson(intch, x=xval)])

    return rms + [np.max(rms), np.mean(rms)], list(intemg) + [np.max(intemg), np.mean(intemg)]


def timeshift(inputarray, shift):
    if (shift == 0) or (shift >= len(inputarray)):
        return inputarray
    ret = np.empty_like(inputarray)
    if shift >= 0:
        ret[:shift] = 0
        ret[shift:] = inputarray[:-shift]
    else:
        ret[shift:] = 0
        ret[:shift] = inputarray[-shift:]
    return ret


def timeshift_average(ecgdata, avgdata, tmin, tmax):
    lowest = np.inf
    tlowest = 0
    tavg = None
    for t in range(tmin, tmax + 1):
        shiftedavg = timeshift(avgdata, t)
        ecg = ecgdata if type(ecgdata) is np.ndarray else np.array(ecgdata)
        newl = np.sum(np.square(ecg - shiftedavg))
        if newl < lowest:
            lowest = newl
            tlowest = t
            tavg = shiftedavg
    return tlowest, tavg


def amplitude_average(ecgdata, avgdata, rangefactor, steps):
    lowest = np.inf
    alowest = 0
    aavg = None
    amplitudes = np.linspace(-1 * rangefactor, rangefactor, steps)
    for a in amplitudes:
        ampavg = avgdata * a
        newa = np.sum(np.square(ecgdata - ampavg))
        if newa < lowest:
            lowest = newa
            alowest = a
            aavg = ampavg
    return alowest, aavg


def subtractecg(ch, peaks, samplingfrequency, windowsize, cancel_check=None):
    from respmech.core._cancel import check as _ck
    emgecgchannels = list(np.array(ch).T)
    retch = []
    chno = 0
    for emgecgch in emgecgchannels:
        _ck(cancel_check)                 # abort a superseded preview job between channels
        chno += 1
        ecgwindows = []
        ecgwindowstart = int(windowsize * samplingfrequency / 2)
        ecgwindowend = int(windowsize * samplingfrequency / 2)
        shiftinterval = int(0.2 * samplingfrequency)

        # Create average ECG for the window
        for peak in peaks:
            if (peak - ecgwindowstart >= 0) and (peak + ecgwindowend <= len(emgecgch)):
                ecgwindow = np.array(emgecgch[peak - ecgwindowstart:peak + ecgwindowend])
                if len(ecgwindows) > 0:
                    _, ecgavgtime = timeshift_average(ecgwindows[0], ecgwindow, -shiftinterval, shiftinterval)
                    ecgwindows += [ecgavgtime]
                else:
                    ecgwindows += [ecgwindow]

        ecgavg = np.mean(ecgwindows, axis=0)

        # Subtract average ECG from EMG
        retwindows = []
        for peak in peaks:
            _ck(cancel_check)             # the 1000-step amplitude sweep per peak is the hot spot
            partial = False
            if (peak - ecgwindowstart < 0):
                ecgwindow = emgecgch[0:peak + ecgwindowend]
                emgecgch[0:peak + ecgwindowend] = list(np.array(ecgwindow) - np.array(ecgavg[len(ecgavg) - len(ecgwindow):len(ecgavg)]))
                retwindows += [[0, peak + ecgwindowend]]
                partial = True
            if (peak + ecgwindowend > len(emgecgch)):
                ecgwindow = emgecgch[peak - ecgwindowstart:len(emgecgch) - 1]
                emgecgch[peak - ecgwindowstart:len(emgecgch) - 1] = list(np.array(ecgwindow) - np.array(ecgavg[0:len(ecgwindow)]))
                retwindows += [[peak - ecgwindowstart, len(emgecgch) - 1]]
                partial = True
            if partial is False:
                ecgwindow = np.array(emgecgch[peak - ecgwindowstart:peak + ecgwindowend])
                retwindows += [[peak - ecgwindowstart, peak + ecgwindowend]]
                _, ecgavgtime = timeshift_average(ecgwindow, ecgavg, -shiftinterval, shiftinterval)
                _, fittedavg = amplitude_average(ecgwindow, ecgavgtime, 1.25, 1000)
                adjwindow = list(ecgwindow - fittedavg)
                emgecgch[peak - ecgwindowstart:peak + ecgwindowend] = adjwindow
        retch += [emgecgch]
    return list(np.array(retch).T), retwindows


def remove_ecg(emgecgchannels, peakch, samplingfrequency, ecgminheight, ecgmindistance, ecgminwidth, windowsize, cancel_check=None):
    peaks, _ = signal.find_peaks(peakch, height=ecgminheight, distance=ecgmindistance * samplingfrequency, width=ecgminwidth * samplingfrequency)
    processedchannels, ecgwindows = subtractecg(emgecgchannels, peaks, samplingfrequency, windowsize, cancel_check=cancel_check)
    return processedchannels, ecgwindows, peaks / samplingfrequency


def peak_window_rms(channel, peaks_samples, samplingfrequency, halfwidth_s=0.04):
    """RMS of the signal within +/- halfwidth around each R-peak — a measure of ECG
    contamination. Comparing before vs after ECG removal gives the suppression."""
    channel = np.asarray(channel, dtype=float)
    w = int(halfwidth_s * samplingfrequency)
    segs = [channel[max(0, p - w):p + w] for p in peaks_samples if 0 <= p < len(channel)]
    if not segs:
        return float("nan")
    return float(np.sqrt(np.mean(np.concatenate(segs) ** 2)))


# --- Cardiac-gated peak EMG (opt-in; default off -> golden-neutral) ---------

def rolling_rms(channel, rms_s, samplingfrequency):
    """The rolling RMS that :func:`calculate_rms` maximises, as (values, start_indices).

    Deliberately reproduces calculate_rms's exact grid rather than an idealised one, so the
    gated statistic is comparable with the shipped one sample for sample: the window is the
    99-sample slice ``channel[i:i+rws-1]`` evaluated for i in range(rws//2, len-rws//2), and
    the final windows are clipped by the array end. calculate_rms itself is pinned by the
    golden tests and is not touched; ``tests/unit/test_robust_peak.py`` asserts that the max
    of what this returns equals calculate_rms's value.
    """
    x = np.asarray(channel, dtype=float)
    n = len(x)
    rws = int(rms_s * samplingfrequency)
    lo, hi = rws // 2, n - rws // 2
    if rws < 2 or hi <= lo:
        return np.array([]), np.array([], dtype=int)
    idx = np.arange(lo, hi)
    ends = np.minimum(idx + rws - 1, n)
    csum = np.concatenate([[0.0], np.cumsum(np.square(x))])
    return np.sqrt((csum[ends] - csum[idx]) / np.maximum(ends - idx, 1)), idx


def detection_quality(peaks_s, ecg_min_distance_s, *, long_rr_factor=1.6,
                      max_long_rr_frac=0.02, hr_ceiling_margin=0.10):
    """Is this R-peak set complete enough to gate on? Returns a dict with ``ok`` + ``reason``.

    Two failure modes matter. A missed beat shows up as an RR interval near a multiple of the
    median, so an excess of long intervals means the set is incomplete. And a heart rate close
    to 60/ecg_min_distance_s means the detector's own refractory floor is about to start
    rejecting real beats, which is where misses begin.
    """
    peaks = np.asarray(peaks_s, dtype=float)
    ceiling_bpm = 60.0 / ecg_min_distance_s if ecg_min_distance_s > 0 else float("inf")
    out = {"n_peaks": int(peaks.size), "median_rr_s": float("nan"), "hr_bpm": float("nan"),
           "long_rr_frac": float("nan"), "ceiling_bpm": float(ceiling_bpm),
           "ok": False, "reason": ""}
    if peaks.size < 4:
        out["reason"] = f"only {peaks.size} R-peaks detected"
        return out
    rr = np.diff(np.sort(peaks))
    med = float(np.median(rr))
    out["median_rr_s"] = med
    out["hr_bpm"] = 60.0 / med if med > 0 else float("nan")
    out["long_rr_frac"] = float(np.mean(rr > long_rr_factor * med))
    if out["long_rr_frac"] > max_long_rr_frac:
        out["reason"] = (f"{100*out['long_rr_frac']:.1f}% of RR intervals exceed "
                         f"{long_rr_factor:g}x the median — beats are being missed")
        return out
    if out["hr_bpm"] >= ceiling_bpm * (1.0 - hr_ceiling_margin):
        out["reason"] = (f"heart rate {out['hr_bpm']:.0f} bpm is within "
                         f"{100*hr_ceiling_margin:.0f}% of the detector ceiling "
                         f"{ceiling_bpm:.0f} bpm set by ecg_min_distance_s")
        return out
    out["ok"] = True
    return out


def _longest_run(mask):
    """Length of the longest contiguous True run — the cardiac-free 'island'."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    edges = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def gated_peak_rms(emgchannels, peak_samples, rms_s, samplingfrequency, *,
                   gate_half_width_s=0.120, min_survival=0.40, min_island_s=0.20):
    """Per-channel max of the rolling RMS, ignoring samples within +/- gate of an R-peak.

    ``peak_samples`` are R-peak positions in samples relative to the START of this segment
    (they may fall outside it; those are simply ignored). Returns
    ``(values + [max, mean], qc)`` with the same shape calculate_rms returns, so the caller
    can build columns identically. When too little cardiac-free signal survives, every value
    is NaN and ``qc['ok']`` is False — an honest refusal beats a number nobody can trust.
    """
    cols = np.asarray(emgchannels, dtype=float)
    if cols.ndim == 1:
        cols = cols[:, None]
    nch = cols.shape[1]
    gate = int(round(gate_half_width_s * samplingfrequency))
    _, idx = rolling_rms(cols[:, 0], rms_s, samplingfrequency)
    qc = {"survival": 0.0, "island_s": 0.0, "ok": False, "reason": ""}
    if idx.size == 0:
        qc["reason"] = "segment shorter than the RMS window"
        return [float("nan")] * (nch + 2), qc

    clean = np.ones(cols.shape[0], dtype=bool)
    for p in np.asarray(peak_samples, dtype=float):
        if not np.isfinite(p):
            continue
        # Peaks outside this segment are normal — the caller passes the whole file's peak
        # train and lets each phase take the ones that reach it. Both bounds must be clamped
        # into range BEFORE slicing: a negative stop index means "from the end" in Python, so
        # a peak just before the segment would otherwise blank nearly all of it.
        lo = max(0, int(round(p)) - gate)
        hi = min(clean.size, int(round(p)) + gate + 1)
        if hi > lo:
            clean[lo:hi] = False
    keep = clean[idx]
    qc["survival"] = float(keep.mean())
    qc["island_s"] = _longest_run(keep) / float(samplingfrequency)
    if qc["survival"] < min_survival or qc["island_s"] < min_island_s:
        qc["reason"] = (f"only {100*qc['survival']:.0f}% of the phase survives the gate "
                        f"(longest cardiac-free island {1000*qc['island_s']:.0f} ms)")
        return [float("nan")] * (nch + 2), qc

    qc["ok"] = True
    vals = []
    for ci in range(nch):
        env, _ = rolling_rms(cols[:, ci], rms_s, samplingfrequency)
        vals.append(float(np.max(env[keep])))
    return vals + [float(np.max(vals)), float(np.mean(vals))], qc


# --- ECG auto-suggest (UI helper; NOT used by run_batch -> golden-neutral) ---
_ECG_DEFAULTS = {"ecg_min_height": 0.0005, "ecg_min_distance_s": 0.5,
                 "ecg_min_width_s": 0.001, "ecg_window_s": 0.4}


def _mad(x):
    """Median absolute deviation (robust scale, unnormalised)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))))


def suggest_ecg_settings(emg_matrix, fs, *, hr_min_bpm=40.0, hr_max_bpm=180.0, max_hr_bpm=240.0):
    """Suggest ECG-capture settings from raw EMG channels — a UI helper for the
    "EMG – ECG reduction" tab's *Auto-suggest* button. Pure numpy/scipy; NOT called by
    ``run_batch`` (so the golden output is unaffected).

    Picks the channel whose R-waves are clearest — most periodic *and* towering over the
    baseline — then derives ``find_peaks`` parameters (as ``remove_ecg`` consumes them) that
    capture those R-peaks and, as far as possible, only those. R-detection is on the
    DC-removed *positive* signal, matching ``remove_ecg``'s detector, so a suggestion is
    always directly usable. Returns a dict of the 5 settings fields plus a ``_diagnostics``
    entry (per-channel score, estimated bpm, confidence 'high'/'low').

    On ambiguous data (no clear ECG, too few beats, a single/flat channel) it falls back to
    the MIDDLE channel (``nch // 2`` — typically the weakest-EMG, clearest-ECG electrode) with
    conservative defaults and ``confidence='low'`` — it never fabricates a spurious capture.
    """
    X = np.asarray(emg_matrix, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, nch = (X.shape if X.ndim == 2 else (0, 0))
    fs = float(fs)

    rr_max = 60.0 / hr_min_bpm            # slowest plausible heartbeat spacing (s)
    rr_min = 60.0 / hr_max_bpm            # fastest plausible (s)
    rr_floor = 60.0 / max_hr_bpm          # absolute refractory floor (s)

    def _defaults(channel, conf="low", diag=None):
        out = dict(_ECG_DEFAULTS)
        out["detect_channel"] = int(channel)
        out["_diagnostics"] = {"confidence": conf, "scores": (diag or {}).get("scores", []),
                               "est_bpm": (diag or {}).get("est_bpm"), "n_peaks": (diag or {}).get("n_peaks", 0)}
        return out

    if nch == 0 or n < int(3 * fs):       # need ~3 beats' worth to say anything
        return _defaults(nch // 2 if nch else 0)

    # QRS-band bandpass for scoring only (lifts R-waves above broadband diaphragm EMG); guard
    # the Nyquist so a low sample rate falls back to the raw copy instead of raising.
    band_sos = None
    if fs > 62.0:
        try:
            band_sos = signal.butter(2, [8.0, 30.0], btype="band", fs=fs, output="sos")
        except Exception:                 # noqa: BLE001
            band_sos = None

    scores = np.full(nch, -np.inf)
    rr_ac = np.full(nch, np.nan)
    pvals = np.zeros(nch)                  # per-channel periodicity (max normalised autocorr in RR band)
    for c in range(nch):
        try:
            col = X[:, c]
            finite = np.isfinite(col)
            if finite.sum() < 0.5 * n:
                continue
            if not finite.all():          # interpolate the few gaps
                idx = np.arange(n)
                col = np.interp(idx, idx[finite], col[finite])
            xr = col - np.median(col)      # DC-removed; positive peaks (matches remove_ecg)
            if np.std(xr) <= 0:
                continue
            xb = signal.sosfiltfilt(band_sos, xr) if band_sos is not None else xr
            sigma = 1.4826 * _mad(xb) or (np.std(xb) + 1e-12)
            pk, props = signal.find_peaks(xb, distance=max(1, int(rr_floor * fs)),
                                          prominence=3.0 * sigma)
            # (A) prominence contrast: how many robust-sigmas the typical R towers over baseline
            C = 0.0
            if pk.size:
                C = float(np.clip(np.median(props["prominences"]) / sigma, 0.0, 30.0))
            # (B) periodicity of the |QRS| envelope via autocorrelation
            env = np.convolve(np.abs(xb), np.ones(max(1, int(0.05 * fs))) / max(1, int(0.05 * fs)), "same")
            e0 = env - env.mean()
            ac = signal.correlate(e0, e0, mode="full", method="fft")
            ac = ac[ac.size // 2:]                 # non-negative lags
            ac0 = ac[0] if ac[0] != 0 else 1.0
            lo, hi = int(rr_min * fs), min(int(rr_max * fs), ac.size - 1)
            P = 0.0
            if hi > lo:
                band = ac[lo:hi] / ac0
                P = float(np.clip(np.max(band), 0.0, 1.0))
                rr_ac[c] = (lo + int(np.argmax(band))) / fs
            pvals[c] = P
            # (C) R-R regularity from the detected beats
            R = 0.0
            if pk.size >= 3:
                rr = np.diff(pk) / fs
                rr = rr[(rr >= rr_min) & (rr <= rr_max)]
                if rr.size >= 2 and np.median(rr) > 0:
                    R = float(np.exp(-(_mad(rr) / np.median(rr)) / 0.25))
            scores[c] = P * R * np.log1p(C)
        except Exception:                  # noqa: BLE001 — a bad channel just scores -inf
            continue

    diag = {"scores": [None if not np.isfinite(v) else round(float(v), 4) for v in scores]}
    if not np.isfinite(scores).any():
        return _defaults(nch // 2, diag=diag)

    best = int(np.argmax(scores))
    # Derive params from the winner on the RAW (DC-removed) copy — the units find_peaks sees.
    col = X[:, best]
    finite = np.isfinite(col)
    if not finite.all():
        idx = np.arange(n)
        col = np.interp(idx, idx[finite], col[finite])
    xr = col - np.median(col)
    sigma_r = 1.4826 * _mad(xr) or (np.std(xr) + 1e-12)
    # Detect R-peaks at the beat-period spacing (from the winner's autocorrelation lag) so
    # find_peaks keeps the tall R per beat and skips baseline-noise maxima between beats —
    # otherwise the R-R and amplitude statistics (and thus the derived height) are corrupted.
    rr_est = float(rr_ac[best]) if (np.isfinite(rr_ac[best]) and rr_ac[best] > 0) else 0.8
    pk, _p = signal.find_peaks(xr, distance=max(1, int(0.6 * rr_est * fs)),
                               prominence=3.0 * sigma_r)

    # confidence: a clear (periodic), regular, sufficiently-sampled winner. A weak periodicity
    # peak (P < 0.30) means "no clear ECG" -> fall back to the MIDDLE channel + defaults.
    rr_intervals = np.diff(pk) / fs if pk.size >= 2 else np.array([])
    rr_intervals = rr_intervals[(rr_intervals >= rr_min) & (rr_intervals <= rr_max)]
    diag["n_peaks"] = int(pk.size)
    low_conf = ((not np.isfinite(scores[best])) or scores[best] <= 0
                or pvals[best] < 0.30 or rr_intervals.size < 4 or pk.size < 4)

    if low_conf:
        rr_med = float(np.median(rr_intervals)) if rr_intervals.size else np.nan
        diag["est_bpm"] = round(60.0 / rr_med, 1) if np.isfinite(rr_med) and rr_med > 0 else None
        return _defaults(nch // 2, diag=diag)     # ambiguous -> middle channel, never fabricate

    rr_med = float(np.median(np.diff(pk) / fs))
    if not np.isfinite(rr_med) or rr_med <= 0:
        rr_med = float(rr_ac[best]) if np.isfinite(rr_ac[best]) else 0.8
    diag["est_bpm"] = round(60.0 / rr_med, 1) if rr_med > 0 else None

    # (i) refractory: half the median R-R rejects T-waves + double counts, capped at ~240 bpm
    dist_s = float(np.clip(0.5 * rr_med, rr_floor, 0.6))
    # (ii) height: a log-midpoint split ABOVE the EMG baseline, BELOW the R amplitude
    r_med = float(np.median(xr[pk]))
    guard = np.ones(n, bool)
    half = max(1, int(0.5 * dist_s * fs))
    for p in pk:
        guard[max(0, p - half):min(n, p + half)] = False    # exclude near-R samples
    base_p = float(np.percentile(xr[guard], 95)) if guard.any() else float(np.median(xr))
    base_p = max(base_p, float(np.median(xr) + 3 * sigma_r))
    height = float(np.clip(np.sqrt(max(base_p, 1e-12) * max(r_med, base_p * 1.06)),
                           base_p * 1.05, 0.7 * r_med if r_med > 0 else base_p * 2))
    # (iii) min width: a floor WELL below the measured QRS so all R's pass
    try:
        widths = signal.peak_widths(xr, pk, rel_height=0.5)[0]
        meas_w = float(np.median(widths)) / fs if widths.size else 0.01
    except Exception:                      # noqa: BLE001
        meas_w = 0.01
    width_s = float(np.clip(0.4 * meas_w, 0.001, 0.02))
    # (iv) template window: QRS-T span, must not overlap neighbouring beats
    window_s = float(np.clip(min(0.4, 0.9 * rr_med), 0.2, 0.5))

    diag["confidence"] = "high"
    return {"detect_channel": best, "ecg_min_height": round(height, 8),
            "ecg_min_distance_s": round(dist_s, 4), "ecg_min_width_s": round(width_s, 5),
            "ecg_window_s": round(window_s, 4), "_diagnostics": diag}


# --- spectral noise reduction (adapted from Tim Sainburg) -------------------

def _stft(y, n_fft, hop_length, win_length):
    import librosa
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, n_fft, hop_length, win_length):
    import librosa
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _amp_to_db(x):
    import librosa
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x):
    import librosa
    return librosa.core.db_to_amplitude(x, ref=1.0)


# NOTE: the legacy per-call `removeNoise`/`reducenoise` spectral-gating pair (which tied
# `n_fft`/`win_length` to the noise-clip length — the `len(noise)**2` bug) has been removed.
# The live, shared-profile spectral path is `core.noise.NoiseProfile` (reuses the STFT
# helpers above). See docs/NOISE_ECG_OPTIMIZATION.md.

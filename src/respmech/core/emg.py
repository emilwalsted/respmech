"""EMG signal processing — computation only (no plotting).

Faithful port of the validated legacy ``emg.py`` compute functions:
RMS + integrated EMG, ECG removal (averaged-template subtraction) and spectral
noise reduction. Plotting/sound export live elsewhere (``respmech.plots``).

Two deliberate, documented changes vs legacy ``master``:
* ``reducenoise`` passes ``win_length=len(noise)**2`` to ``removeNoise`` — the newer
  noise-reduction variant from the ``resampling-options`` line (Emil confirmed the
  canonical newest EMG/noise baseline is master + resampling-options).
* ``scipy.integrate.simpson`` is called with the keyword ``x=`` (positional ``x`` is
  removed in SciPy >= 1.14) — numerically identical (legacy bug #4).

Noise reduction adapted, with permission, from code by Tim Sainburg
(https://timsainburg.com). Requires ``librosa``.
"""
import sys
import os
import numpy as np
import scipy as sp
from scipy import signal

from respmech.core._compat import complex_sign_v1 as _complex_sign_v1

# librosa is only needed when noise reduction is used; import lazily inside
# reducenoise so ECG removal / RMS work without it.
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


def removeNoise(audio_clip, noise_clip, n_grad_freq=0, n_grad_time=4, n_fft=2048,
                win_length=2048, hop_length=512, n_std_thresh=2, prop_decrease=1):
    """Spectral-gating noise reduction. Plotting/verbose paths from the legacy
    version are removed (they never ran in the compute path)."""
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    smoothing_filter = np.outer(
        np.concatenate([
            np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
            np.linspace(1, 0, n_grad_freq + 2),
        ])[1:-1],
        np.concatenate([
            np.linspace(0, 1, n_grad_time + 1, endpoint=False),
            np.linspace(1, 0, n_grad_time + 2),
        ])[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1], axis=0,
    ).T
    sig_mask = sig_stft_db < db_thresh
    sig_mask = sp.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * _complex_sign_v1(sig_stft)) + (1j * sig_imag_masked)
    recovered_signal = _istft(sig_stft_amp, n_fft, hop_length, win_length)
    return recovered_signal


def reducenoise(emgchannel, noiseprofile, noiseprofilecolumn, samplingfrequency):
    if len(noiseprofilecolumn) == 0:
        noiseprofilecolumn = emgchannel
    noise = np.array(noiseprofilecolumn[int(noiseprofile[0] * samplingfrequency):int(noiseprofile[1] * samplingfrequency)])
    # win_length=len(noise)**2: the newer noise-reduction variant (resampling-options).
    saved = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        output = removeNoise(audio_clip=emgchannel, noise_clip=noise,
                             n_fft=len(noise) ** 2, win_length=len(noise) ** 2)
    finally:
        sys.stdout.close()
        sys.stdout = saved
    return np.array(output)

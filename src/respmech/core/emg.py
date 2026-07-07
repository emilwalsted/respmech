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


def subtractecg(ch, peaks, samplingfrequency, windowsize):
    emgecgchannels = list(np.array(ch).T)
    retch = []
    chno = 0
    for emgecgch in emgecgchannels:
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


def remove_ecg(emgecgchannels, peakch, samplingfrequency, ecgminheight, ecgmindistance, ecgminwidth, windowsize):
    peaks, _ = signal.find_peaks(peakch, height=ecgminheight, distance=ecgmindistance * samplingfrequency, width=ecgminwidth * samplingfrequency)
    processedchannels, ecgwindows = subtractecg(emgecgchannels, peaks, samplingfrequency, windowsize)
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
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)
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

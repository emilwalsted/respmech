"""core.emg.suggest_ecg_settings — the Auto-suggest analysis for the EMG–ECG reduction tab.
Pure numpy/scipy, NOT called by run_batch, so it cannot affect the goldens. It must pick the
channel whose R-waves are clearest, derive a capture threshold that sits between the EMG
baseline and the R amplitude, and fall back to the middle channel on ambiguous data."""
import numpy as np

from respmech.core.emg import suggest_ecg_settings


def _synthetic(fs=2000, dur_s=20.0, rr_s=0.9, seed=0):
    """3 channels: outer 0/2 = respiration-modulated broadband EMG bursts (no ~1 Hz structure);
    middle 1 = the same weak EMG PLUS a periodic narrow QRS spike train (clearest ECG)."""
    rng = np.random.RandomState(seed)
    n = int(fs * dur_s)
    t = np.arange(n) / fs

    def emg_burst(amp):
        # broadband noise gated by a slow (respiration) envelope ~0.25 Hz
        env = 0.5 * (1 + np.sin(2 * np.pi * 0.25 * t))
        return amp * env * rng.standard_normal(n)

    ch0 = emg_burst(0.30)
    ch2 = emg_burst(0.30)
    # QRS spike train on the middle channel: narrow Gaussian R-waves every ~rr_s (± jitter)
    qrs = np.zeros(n)
    beat, sig = 0.0, 0.008 * fs           # 8 ms QRS width
    while beat < dur_s - 0.5:
        p = int((beat + rng.uniform(-0.02, 0.02)) * fs)
        lo, hi = max(0, p - int(6 * sig)), min(n, p + int(6 * sig))
        k = np.arange(lo, hi)
        qrs[lo:hi] += 1.0 * np.exp(-0.5 * ((k - p) / sig) ** 2)
        beat += rr_s
    ch1 = 0.10 * emg_burst(1.0) / 0.30 * 0.10 + qrs      # weak EMG + strong QRS
    ch1 = qrs + 0.06 * rng.standard_normal(n)
    return np.column_stack([ch0, ch1, ch2]), fs, qrs


def test_picks_the_clearest_ecg_channel_and_sane_params():
    X, fs, qrs = _synthetic()
    out = suggest_ecg_settings(X, fs)
    assert out["detect_channel"] == 1                       # the periodic-QRS (middle) channel
    assert out["_diagnostics"]["confidence"] == "high"
    # distance ~ half the 0.9 s R-R, within the physiological window
    assert 0.25 <= out["ecg_min_distance_s"] <= 0.6
    # height (derived from the WINNING/detect channel ch1) sits ABOVE ch1's EMG baseline but
    # BELOW the R amplitude (~1.0) — the ECG detector only ever runs on the detect channel.
    ch1 = X[:, 1] - np.median(X[:, 1])
    ch1_baseline = np.percentile(np.abs(ch1[ch1 < 0.5]), 95)   # non-R samples
    assert ch1_baseline < out["ecg_min_height"] < 1.0
    assert 0.001 <= out["ecg_min_width_s"] <= 0.02
    assert 0.2 <= out["ecg_window_s"] <= 0.5
    # the suggested params capture ~one peak per beat on the detect channel
    from scipy.signal import find_peaks
    pk, _ = find_peaks(ch1, height=out["ecg_min_height"],
                       distance=out["ecg_min_distance_s"] * fs,
                       width=out["ecg_min_width_s"] * fs)
    assert 18 <= len(pk) <= 24                               # ~22 beats over 20 s at 0.9 s R-R


def test_middle_default_and_low_confidence_on_ecg_free_data():
    rng = np.random.RandomState(1)
    X = rng.standard_normal((2000 * 20, 5)) * 0.3           # pure EMG-ish noise, no ECG
    out = suggest_ecg_settings(X, 2000)
    assert out["detect_channel"] == 2                        # middle of 5 channels
    assert out["_diagnostics"]["confidence"] == "low"
    for k, v in (("ecg_min_height", 0.0005), ("ecg_min_distance_s", 0.5),
                 ("ecg_min_width_s", 0.001), ("ecg_window_s", 0.4)):
        assert out[k] == v                                   # conservative defaults


def test_edge_cases_never_raise():
    # single channel
    s = suggest_ecg_settings(np.zeros((2000 * 5, 1)), 2000)
    assert s["detect_channel"] == 0
    # too short
    s = suggest_ecg_settings(np.random.RandomState(2).standard_normal((100, 3)), 2000)
    assert s["detect_channel"] == 1 and s["_diagnostics"]["confidence"] == "low"
    # flat + NaN channels
    X = np.zeros((2000 * 20, 3)); X[:, 0] = np.nan
    s = suggest_ecg_settings(X, 2000)
    assert s["detect_channel"] in (0, 1, 2)
    # empty
    s = suggest_ecg_settings(np.zeros((0, 0)), 2000)
    assert s["detect_channel"] == 0
    # low sampling rate (Nyquist guard: no bandpass, must not raise)
    fs = 50
    n = fs * 20
    t = np.arange(n) / fs
    x = np.zeros(n)
    for b in np.arange(0, 20, 0.9):
        x[int(b * fs)] = 1.0
    X = np.column_stack([0.2 * np.random.RandomState(3).standard_normal(n), x + 0.05, 0.2 * np.random.RandomState(4).standard_normal(n)])
    s = suggest_ecg_settings(X, fs)
    assert s["detect_channel"] in (0, 1, 2)                  # just must not raise

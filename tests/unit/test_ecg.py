"""ECG removal: suppression metric, per-test-consistent detection, and the
peak-window RMS helper. Uses a small synthetic ECG-contaminated signal (fast)."""
import numpy as np
import pytest

from respmech.core import emg as E


def _ecg_spike(w):
    """A short biphasic spike ~ an R-wave."""
    t = np.linspace(-1, 1, w)
    return np.exp(-(t ** 2) / 0.02) * np.sin(2 * np.pi * 1.5 * t)


def _make_signal(fs=2000, dur=12.0, hr_period=1.0, n_ch=3, seed=0, amp=1.0):
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    emg = rng.normal(0, 0.05, (n, n_ch))          # baseline EMG-like noise
    w = int(0.06 * fs)
    spike = _ecg_spike(w) * amp
    peaks = []
    p = int(0.5 * fs)
    while p + w < n:
        for c in range(n_ch):
            emg[p:p + w, c] += spike * (1.0 - 0.15 * c)   # ECG on every channel
        peaks.append(p + w // 2)
        p += int(hr_period * fs)
    return emg, np.array(peaks), fs


def test_peak_window_rms_helper():
    fs = 2000
    x = np.zeros(fs)
    x[100:140] = 1.0
    peaks = np.array([120])
    r = E.peak_window_rms(x, peaks, fs, halfwidth_s=0.04)
    assert r > 0
    assert np.isnan(E.peak_window_rms(x, np.array([], dtype=int), fs))


def test_ecg_removal_suppresses_contamination():
    emg, true_peaks, fs = _make_signal()
    detect = emg[:, 0]
    before = E.peak_window_rms(detect, true_peaks, fs)
    processed, _win, peaks_s = E.remove_ecg(
        emg.copy(), detect, samplingfrequency=fs,
        ecgminheight=0.2, ecgmindistance=0.5, ecgminwidth=0.001, windowsize=0.4)
    peaks_samp = (np.asarray(peaks_s) * fs).astype(int)
    after = E.peak_window_rms(np.asarray(processed)[:, 0], peaks_samp, fs)
    suppression = 1 - after / before
    assert len(peaks_samp) >= 8                   # detected most beats
    assert suppression > 0.3                      # meaningfully reduced ECG


def test_detection_is_deterministic_and_param_driven():
    # Same params + same signal -> identical detected peaks (consistent per test).
    emg, _tp, fs = _make_signal(seed=1)
    kw = dict(samplingfrequency=fs, ecgminheight=0.2, ecgmindistance=0.5,
              ecgminwidth=0.001, windowsize=0.4)
    _p1, _w1, peaks1 = E.remove_ecg(emg.copy(), emg[:, 0], **kw)
    _p2, _w2, peaks2 = E.remove_ecg(emg.copy(), emg[:, 0], **kw)
    assert np.array_equal(peaks1, peaks2)
    # A higher threshold detects fewer/equal peaks (parameter actually drives it).
    _p3, _w3, peaks3 = E.remove_ecg(emg.copy(), emg[:, 0], **{**kw, "ecgminheight": 5.0})
    assert len(peaks3) <= len(peaks1)


@pytest.mark.parametrize("import_ok", [True])
def test_pipeline_reports_ecg_suppression(import_ok, tmp_path):
    """The pipeline attaches ECG diagnostics to each FileResult when remove_ecg is on."""
    import os
    INPUT = os.path.join(os.path.dirname(__file__), "..", "golden", "input")
    if not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")):
        pytest.skip("synthetic input not present")
    # synthetic data has no ECG, but the diagnostic plumbing must still populate.
    from respmech.settingsio.migrate import migrate_dict
    from respmech.core.pipeline import run_batch
    legacy = {"input": {"inputfolder": os.path.abspath(INPUT), "files": "synth_case_A.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": True, "column_detect": 0,
                                     "minheight": 0.5, "mindistance": 0.3, "minwidth": 0.001}},
              "output": {"outputfolder": str(tmp_path), "data": {"savebreathbybreathdata": True}}}
    s, _ = migrate_dict(legacy)
    res = run_batch(s)
    fr = res.ok_files["synth_case_A.csv"]
    assert fr.ecg is not None
    assert "n_peaks" in fr.ecg and "suppression" in fr.ecg

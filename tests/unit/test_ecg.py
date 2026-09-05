"""ECG removal: suppression metric, per-test-consistent detection, and the
peak-window RMS helper. Uses a small synthetic ECG-contaminated signal (fast)."""
import numpy as np
import pytest

from respmech.core import emg as E


def _ecg_spike(w):
    """A short biphasic spike ~ an R-wave."""
    t = np.linspace(-1, 1, w)
    return np.exp(-(t ** 2) / 0.02) * np.sin(2 * np.pi * 1.5 * t)


def _make_signal(fs=2000, dur=12.0, hr_period=1.0, n_ch=3, seed=0, amp=1.0,
                 invert=False, dc_offset=0.0):
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    emg = rng.normal(0, 0.05, (n, n_ch)) + dc_offset   # baseline EMG-like noise
    w = int(0.06 * fs)
    spike = _ecg_spike(w) * amp * (-1.0 if invert else 1.0)
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


def test_inverted_r_wave_is_detected_too():
    """Ticket 5.7 / K-191: the detector used to find positive peaks only (find_peaks'
    height= convention), so a channel whose R-wave is inverted (negative-going) could
    not be used for detection at all. It must now find the beats regardless of
    polarity, and count the same as the upright case on the same recording."""
    kw = dict(samplingfrequency=2000, ecgminheight=0.2, ecgmindistance=0.5,
              ecgminwidth=0.001, windowsize=0.4)
    up, true_peaks, fs = _make_signal(seed=2, invert=False)
    down, _tp2, _fs2 = _make_signal(seed=2, invert=True)
    _pu, _wu, peaks_up = E.remove_ecg(up.copy(), up[:, 0], **kw)
    _pd, _wd, peaks_down = E.remove_ecg(down.copy(), down[:, 0], **kw)
    assert len(peaks_down) == len(peaks_up) == len(true_peaks)
    # detected instants line up between the two polarities (same underlying beats;
    # the exact sample can jitter a bit since the added noise differs by sign too)
    assert np.allclose(np.sort(peaks_up), np.sort(peaks_down), atol=15.0 / fs)


def test_detection_matches_dc_removed_signal_like_auto_suggest():
    """Ticket 5.7 / K-191: remove_ecg used to run find_peaks on the raw channel while
    suggest_ecg_settings derives ecgminheight from a median-subtracted copy, so a
    channel with a DC offset made a suggested height wrong for what remove_ecg
    actually saw. Detection must now be identical regardless of a constant DC offset
    added to the channel (median-removal cancels it exactly)."""
    kw = dict(samplingfrequency=2000, ecgminheight=0.2, ecgmindistance=0.5,
              ecgminwidth=0.001, windowsize=0.4)
    plain, _tp, fs = _make_signal(seed=4, dc_offset=0.0)
    offset, _tp2, _fs2 = _make_signal(seed=4, dc_offset=3.7)
    _p1, _w1, peaks_plain = E.remove_ecg(plain.copy(), plain[:, 0], **kw)
    _p2, _w2, peaks_offset = E.remove_ecg(offset.copy(), offset[:, 0], **kw)
    assert np.array_equal(peaks_plain, peaks_offset)


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


def test_stage_ecg_reduction_reports_suppression_matching_the_pipeline(tmp_path):
    """The 'EMG - ECG reduction' tab (ui.workers.stage_ecg_reduction) is a live TUNING
    surface, but until ticket 20260804-0922 it never computed the peak-window-RMS
    suppression core.pipeline._ecg_remove reports for a real run — so 'Min height'/'Min
    gap' were dialled in on nothing but eyeballing the traces. It must now report a
    positive suppression, on a syntetic signal with known R-peaks, of the same order of
    magnitude as the pipeline computes on the identically-loaded raw matrix, using the
    SAME core.emg.peak_window_rms helper (no new variant, no golden-affecting change)."""
    import pandas as pd
    from respmech.settingsio.migrate import migrate_dict
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core.io.loaders import load
    from respmech.core.pipeline import _ecg_remove
    from respmech.ui.workers import stage_ecg_reduction

    emg, _true_peaks, fs = _make_signal(seed=7)         # same synthetic ECG-on-EMG builder
    n = emg.shape[0]
    t = np.arange(n) / fs
    path = tmp_path / "ecg_contaminated.csv"
    zeros = np.zeros(n)
    pd.DataFrame({"time": t, "c2": emg[:, 0], "c3": emg[:, 1], "c4": emg[:, 2],
                 "flow": zeros, "volume": zeros,
                 "poes": zeros, "pgas": zeros, "pdi": zeros}).to_csv(path, index=False)

    legacy = {"input": {"inputfolder": str(tmp_path), "files": path.name,
                        "format": {"samplingfrequency": fs},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": True, "column_detect": 0,
                                     "minheight": 0.2, "mindistance": 0.5, "minwidth": 0.001,
                                     "windowsize": 0.4}},
              "output": {"outputfolder": str(tmp_path), "data": {}}}
    s, _ = migrate_dict(legacy)
    s.input.folder = str(tmp_path)

    data = stage_ecg_reduction(s, str(path))
    assert data["ecg_applied"] is True
    supp = data["suppression"]
    assert supp is not None and supp == supp            # not None, not NaN
    assert supp > 0.3                                    # meaningfully reduced (test_ecg_removal_* threshold)

    # Cross-check against what a REAL run computes, on the identically-loaded raw matrix,
    # via the same helper the ticket requires (core.emg.peak_window_rms, no new variant).
    ls = to_legacy_ns(s)
    _flow, _v, _p, _g, _d, _e, raw_emg = load(str(path), ls)
    _emgcols, diag = _ecg_remove(ls, raw_emg)
    assert diag is not None
    assert abs(supp - diag["suppression"]) < 0.05        # same order of magnitude, same detector

    # Removal OFF: nothing to suppress, so no number at all (None, not NaN) -- distinct
    # from "computed but degenerate", which stays NaN.
    s_off, _ = migrate_dict({**legacy, "processing": {**legacy["processing"],
                             "emg": {**legacy["processing"]["emg"], "remove_ecg": False}}})
    s_off.input.folder = str(tmp_path)
    data_off = stage_ecg_reduction(s_off, str(path))
    assert data_off["ecg_applied"] is False
    assert data_off["suppression"] is None

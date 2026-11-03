"""Characterisation of the EMG helpers in ``core.emg`` that the golden suite does not reach.

No golden scenario switches on cardiac gating, ECG auto-suggest or anything but the default
RMS window, so mutation testing found these behaviours unchecked: the fields and boundaries
of ``detection_quality`` and ``gated_peak_rms``, single-channel input, the short-input
guards of ``rolling_rms`` and ``peak_window_rms``, ``timeshift`` at the edges, the
degenerate path of ``calculate_rms`` and the numbers ``suggest_ecg_settings`` returns. The
expected values are written from the docstrings where they define them, and otherwise pin
the current output on fixed synthetic signals (see docs/MUTATION_TESTING.md).
"""
import math

import numpy as np
import pytest

from respmech.core import emg as emglib
from test_emg_suggest import _synthetic

FS = 1000


# --- detection_quality -------------------------------------------------------

def _train(rr, n=40, start=1.0):
    return start + rr * np.arange(n)


def test_detection_quality_fields_on_a_clean_train():
    out = emglib.detection_quality(_train(1.0), 0.5)
    assert out == {"n_peaks": 40, "median_rr_s": 1.0, "hr_bpm": 60.0, "long_rr_frac": 0.0,
                   "ceiling_bpm": 120.0, "ok": True, "reason": ""}


def test_detection_quality_too_few_peaks_keeps_the_defaults():
    out = emglib.detection_quality([1.0, 2.0, 3.0], 0.5)
    assert out["ok"] is False
    assert out["reason"] == "only 3 R-peaks detected"
    assert out["n_peaks"] == 3 and out["ceiling_bpm"] == 120.0
    for key in ("median_rr_s", "hr_bpm", "long_rr_frac"):
        assert math.isnan(out[key])
    # four peaks are enough
    assert emglib.detection_quality([1.0, 2.0, 3.0, 4.0], 0.5)["ok"] is True


def test_detection_quality_zero_refractory_floor_has_no_ceiling():
    out = emglib.detection_quality(_train(1.0), 0.0)
    assert out["ceiling_bpm"] == float("inf") and out["ok"] is True


def test_detection_quality_long_interval_boundary_and_message():
    # 50 intervals, one exactly 2x the median: not "longer than", so not counted
    peaks = np.concatenate([[0.0], np.cumsum([1.0] * 49 + [2.0])])
    assert emglib.detection_quality(peaks, 0.3, long_rr_factor=2.0)["long_rr_frac"] == 0.0
    # exactly 2 % long intervals is accepted, more is not
    peaks = np.concatenate([[0.0], np.cumsum([1.0] * 49 + [2.0])])
    out = emglib.detection_quality(peaks, 0.3)
    assert out["long_rr_frac"] == pytest.approx(0.02) and out["ok"] is True
    peaks = np.concatenate([[0.0], np.cumsum([1.0] * 48 + [2.0, 2.0])])
    out = emglib.detection_quality(peaks, 0.3)
    assert out["ok"] is False
    assert out["reason"] == "4.0% of RR intervals exceed 1.6x the median — beats are being missed"


def test_detection_quality_ceiling_boundary_and_message():
    # ceiling 120 bpm, margin 10 %: 108 bpm is refused, just below is accepted
    out = emglib.detection_quality(_train(60.0 / 108.0), 0.5)
    assert out["ok"] is False
    assert out["reason"] == ("heart rate 108 bpm is within 10% of the detector ceiling "
                             "120 bpm set by ecg_min_distance_s")
    assert emglib.detection_quality(_train(60.0 / 107.0), 0.5)["ok"] is True


def test_detection_quality_coincident_peaks_have_no_heart_rate():
    out = emglib.detection_quality([1.0, 1.0, 1.0, 1.0, 1.0], 0.5)
    assert out["median_rr_s"] == 0.0 and math.isnan(out["hr_bpm"])


# --- gated_peak_rms and its helpers ------------------------------------------

def _noise(n=6000, seed=0):
    return np.random.RandomState(seed).randn(n) * 0.01


def test_gated_peak_accepts_a_single_channel_vector():
    x = _noise()
    vec, qc_vec = emglib.gated_peak_rms(x, [3000], 0.05, FS)
    col, qc_col = emglib.gated_peak_rms(x[:, None], [3000], 0.05, FS)
    assert qc_vec == qc_col and qc_vec["ok"] is True
    assert vec == col and len(vec) == 3


def test_gated_peak_short_segment_reports_why():
    vals, qc = emglib.gated_peak_rms(np.zeros((40, 2)), [], 0.05, FS)
    assert len(vals) == 4 and all(math.isnan(v) for v in vals)
    assert qc == {"survival": 0.0, "island_s": 0.0, "ok": False,
                  "reason": "segment shorter than the RMS window"}


def test_gated_peak_qc_numbers_and_gate_width():
    x = _noise()
    x[2990:3011] = 5.0
    gate = 0.010                                   # +/- 10 samples, 21 blanked
    vals, qc = emglib.gated_peak_rms(np.column_stack([x, 2 * x]), [3000.0], 0.05, FS,
                                     gate_half_width_s=gate)
    idx = np.arange(25, 6000 - 25)
    clean = np.ones(6000, bool)
    clean[2990:3011] = False
    keep = clean[idx]
    assert qc["survival"] == pytest.approx(keep.mean(), abs=0)
    assert qc["island_s"] == pytest.approx((2990 - 25) / FS, abs=0)
    env0, _ = emglib.rolling_rms(x, 0.05, FS)
    env1, _ = emglib.rolling_rms(2 * x, 0.05, FS)
    expected = [float(np.max(env0[keep])), float(np.max(env1[keep]))]
    assert vals[:2] == expected
    assert vals[2] == max(expected) and vals[3] == pytest.approx(np.mean(expected), abs=0)
    assert qc["reason"] == "" and qc["ok"] is True


def test_gated_peak_skips_non_finite_peaks_and_keeps_going():
    x = _noise()
    x[3000] = 50.0
    a, _ = emglib.gated_peak_rms(x, [np.nan, 3000], 0.05, FS)
    b, _ = emglib.gated_peak_rms(x, [3000], 0.05, FS)
    assert a == b


def test_gated_peak_blanks_from_the_first_sample_and_up_to_the_gate():
    x = _noise()
    x[0:1] = 50.0
    x[146] = 50.0                                 # gate around sample 26 ends at 26+120
    vals, qc = emglib.gated_peak_rms(x, [26], 0.05, FS)
    ref, _ = emglib.rolling_rms(x, 0.05, FS)
    idx = np.arange(25, 6000 - 25)
    clean = np.ones(6000, bool)
    clean[0:147] = False
    assert vals[0] == float(np.max(ref[clean[idx]]))
    assert qc["survival"] == pytest.approx(clean[idx].mean(), abs=0)


def test_gated_peak_refusal_boundaries_and_message():
    x = _noise(n=1050)                            # 1000 RMS windows
    # blank the first 600 window starts: survival 0.40 exactly is enough
    vals, qc = emglib.gated_peak_rms(x, [324], 0.05, FS, gate_half_width_s=0.300,
                                     min_island_s=0.0)
    assert qc["survival"] == pytest.approx(0.4) and qc["ok"] is True
    # an island exactly the minimum is enough
    vals, qc = emglib.gated_peak_rms(x, [324], 0.05, FS, gate_half_width_s=0.300,
                                     min_survival=0.0, min_island_s=0.4)
    assert qc["island_s"] == pytest.approx(0.4) and qc["ok"] is True
    # either guard alone refuses
    vals, qc = emglib.gated_peak_rms(x, [324], 0.05, FS, gate_half_width_s=0.300,
                                     min_survival=0.5, min_island_s=0.0)
    assert qc["ok"] is False
    assert qc["reason"] == ("only 40% of the phase survives the gate "
                            "(longest cardiac-free island 400 ms)")
    vals, qc = emglib.gated_peak_rms(x, [324], 0.05, FS, gate_half_width_s=0.300,
                                     min_survival=0.0, min_island_s=0.5)
    assert qc["ok"] is False and len(vals) == 3 and all(math.isnan(v) for v in vals)


def test_gated_peak_default_settings():
    x = _noise(n=4000)
    # default gate 0.120 s, survival 0.40, island 0.20 s
    _, qc = emglib.gated_peak_rms(x, [1000, 2000, 3000], 0.05, FS)
    idx = np.arange(25, 3975)
    clean = np.ones(4000, bool)
    for p in (1000, 2000, 3000):
        clean[p - 120:p + 121] = False
    assert qc["survival"] == pytest.approx(clean[idx].mean(), abs=0)
    assert qc["island_s"] == pytest.approx(0.855) and qc["ok"] is True


def test_longest_run():
    assert emglib._longest_run([False, False]) == 0
    assert emglib._longest_run([1, 1, 0, 1, 1, 1, 0]) == 3
    assert emglib._longest_run([0, 1, 1, 1, 1]) == 4


def test_rolling_rms_guards_and_values():
    x = np.arange(1.0, 11.0)
    env, idx = emglib.rolling_rms(x, 0.004, FS)   # rws 4: windows of 3 samples
    assert list(idx) == [2, 3, 4, 5, 6, 7]
    expected = [np.sqrt(np.mean(np.square(x[i:min(i + 3, 10)]))) for i in idx]
    np.testing.assert_allclose(env, expected, rtol=1e-12)
    # three samples with a 2-sample window give one value, a 1-sample window none
    env, idx = emglib.rolling_rms(np.array([1.0, 2.0, 3.0]), 0.002, FS)
    assert list(idx) == [1] and env.size == 1
    env, idx = emglib.rolling_rms(np.array([1.0, 2.0, 3.0]), 0.001, FS)
    assert idx.size == 0 and idx.dtype.kind == "i"
    env, idx = emglib.rolling_rms(np.arange(9.0), 0.006, FS)   # rws 6: lo 3, hi 6
    assert list(idx) == [3, 4, 5]
    assert emglib.rolling_rms(np.arange(6.0), 0.006, FS)[1].size == 0
    env, idx = emglib.rolling_rms([1, 2, 3, 4, 5, 6, 7], 0.004, FS)
    assert env.dtype == float


def test_peak_window_rms_edges():
    ch = np.arange(10.0)
    assert emglib.peak_window_rms(ch, [0], 1000, halfwidth_s=0.002) == pytest.approx(
        np.sqrt(np.mean(np.square([0.0, 1.0]))))
    assert emglib.peak_window_rms(ch, [9], 1000, halfwidth_s=0.002) == pytest.approx(
        np.sqrt(np.mean(np.square([7.0, 8.0, 9.0]))))
    assert math.isnan(emglib.peak_window_rms(ch, [10, -1], 1000))
    assert emglib.peak_window_rms([1, 1, 1], [1], 1000, halfwidth_s=0.001) == 1.0


def test_timeshift_edges():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(emglib.timeshift(a, 0)) == [1, 2, 3, 4]
    assert list(emglib.timeshift(a, 4)) == [1, 2, 3, 4]
    assert list(emglib.timeshift(a, 3)) == [0, 0, 0, 1]
    assert list(emglib.timeshift(a, 1)) == [0, 1, 2, 3]
    assert list(emglib.timeshift(a, -1)) == [2, 3, 4, 0]


def test_calculate_rms_degenerate_window_and_too_short_breath():
    x = np.arange(1.0, 7.0)[:, None]
    # rms_s * fs = 1: rollend 0, every window empty -> NaN, as the original scalar path gave
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        rms, _ = emglib.calculate_rms(x, 0.001, FS)
    assert math.isnan(rms[0])
    # rms_s * fs = 2: one-sample windows, i in range(1, n - 1)
    rms, _ = emglib.calculate_rms(x, 0.002, FS)
    assert rms[0] == 5.0
    # a breath shorter than half a window gives 0
    rms, _ = emglib.calculate_rms(np.ones((3, 1)), 0.05, FS)
    assert rms == [0, 0, 0.0]


def test_calculate_rms_clipped_tail_windows():
    # rws 8: window starts 4..15, full 7-sample windows up to 13, then 6 and 5 samples
    x = np.zeros(20)
    x[18] = 3.0                                    # only in the 5-sample tail window
    rms, integ = emglib.calculate_rms(x[:, None], 0.008, FS)
    assert rms[0] == pytest.approx(np.sqrt(9.0 / 5.0), rel=1e-12)
    x = np.zeros(20)
    x[14] = 3.0                                    # best is the 6-sample window at 14
    rms, _ = emglib.calculate_rms(x[:, None], 0.008, FS)
    assert rms[0] == pytest.approx(np.sqrt(9.0 / 6.0), rel=1e-12)
    x = np.zeros(20)
    x[13] = 3.0                                    # last full window, 7 samples
    rms, _ = emglib.calculate_rms(x[:, None], 0.008, FS)
    assert rms[0] == pytest.approx(np.sqrt(9.0 / 7.0), rel=1e-12)
    x = np.zeros(20)
    x[5] = 3.0                                     # first full window starts at 4
    rms, _ = emglib.calculate_rms(x[:, None], 0.008, FS)
    assert rms[0] == pytest.approx(np.sqrt(9.0 / 7.0))


# --- suggest_ecg_settings ------------------------------------------------------

def test_suggest_pins_the_clear_ecg_case():
    X, fs, _ = _synthetic()
    assert emglib.suggest_ecg_settings(X, fs) == {
        "detect_channel": 1, "ecg_min_height": 0.45090903, "ecg_min_distance_s": 0.449,
        "ecg_min_width_s": 0.00805, "ecg_window_s": 0.4,
        "_diagnostics": {"scores": [0.8168, 0.9568, 0.889], "n_peaks": 23, "est_bpm": 66.8,
                         "confidence": "high"}}


def test_suggest_pins_an_inverted_fast_heart():
    X, fs, _ = _synthetic(rr_s=0.6, seed=3)
    assert emglib.suggest_ecg_settings(-X, fs) == {
        "detect_channel": 1, "ecg_min_height": 0.13212636, "ecg_min_distance_s": 0.25,
        "ecg_min_width_s": 0.02, "ecg_window_s": 0.4,
        "_diagnostics": {"scores": [0.8559, 2.2931, 0.8454], "n_peaks": 40, "est_bpm": 125.0,
                         "confidence": "high"}}


def test_suggest_pins_the_defaults_on_noise_and_on_silence():
    noise = np.random.RandomState(1).randn(20000, 3)
    assert emglib.suggest_ecg_settings(noise, 2000) == {
        "ecg_min_height": 0.0005, "ecg_min_distance_s": 0.5, "ecg_min_width_s": 0.001,
        "ecg_window_s": 0.4, "detect_channel": 1,
        "_diagnostics": {"confidence": "low", "scores": [0.1423, 0.1184, 0.0518],
                         "est_bpm": 51.3, "n_peaks": 10}}
    assert emglib.suggest_ecg_settings(np.zeros((100, 2)), 2000) == {
        "ecg_min_height": 0.0005, "ecg_min_distance_s": 0.5, "ecg_min_width_s": 0.001,
        "ecg_window_s": 0.4, "detect_channel": 1,
        "_diagnostics": {"confidence": "low", "scores": [], "est_bpm": None, "n_peaks": 0}}


def test_mad():
    assert emglib._mad([]) == 0.0
    assert emglib._mad([1, 2, 3, 4, 100]) == 1.0

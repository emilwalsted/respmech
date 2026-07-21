"""Cardiac-gated peak EMG (processing.emg.robust_peak).

The load-bearing properties, in order of how much damage getting them wrong would do:
  1. With the feature OFF nothing changes — no keys, no columns. The golden suite pins the
     existing output byte-for-byte, so this is what keeps the feature shippable.
  2. The gated statistic reads the SAME rolling RMS the shipped one maximises, so the two
     are comparable sample for sample.
  3. The gate actually excludes beat windows.
  4. The quality guards refuse rather than report a number nobody can trust.
"""
import numpy as np
import pytest

from respmech.core import emg as emglib
from respmech.core.settings import Settings


FS = 2000
RMS_S = 0.05


def _signal(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n) * 0.01


# --- 2. the gated path must agree with the pinned calculate_rms ---------------

def test_rolling_rms_max_matches_calculate_rms():
    """rolling_rms is a separate implementation (calculate_rms is golden-pinned and must not
    be touched); its maximum must nevertheless be the number calculate_rms reports."""
    for seed in range(4):
        x = _signal(seed=seed)
        cols = np.column_stack([x, x * 0.5])
        shipped, _ = emglib.calculate_rms(cols, RMS_S, FS)
        env, idx = emglib.rolling_rms(cols[:, 0], RMS_S, FS)
        assert idx[0] == int(RMS_S * FS) // 2
        assert float(np.max(env)) == pytest.approx(shipped[0], rel=1e-12, abs=1e-15)


def test_rolling_rms_empty_for_short_segment():
    env, idx = emglib.rolling_rms(np.zeros(50), RMS_S, FS)
    assert env.size == 0 and idx.size == 0


# --- 3. the gate excludes beat windows ---------------------------------------

def test_gated_peak_ignores_the_spike_it_is_told_about():
    x = _signal()
    x[10000:10050] += 5.0                      # a big transient at sample 10000
    cols = x[:, None]
    ungated, _ = emglib.calculate_rms(cols, RMS_S, FS)
    gated, qc = emglib.gated_peak_rms(cols, [10000], RMS_S, FS, gate_half_width_s=0.120)
    assert qc["ok"]
    assert gated[0] < ungated[0] / 10          # the spike no longer sets the number
    # and the trailing [max, mean] mirror calculate_rms's shape
    assert len(gated) == cols.shape[1] + 2
    assert gated[-2] == pytest.approx(max(gated[:-2]))


def test_gated_peak_survives_when_the_spike_is_elsewhere():
    """A gate placed away from the transient must leave the reported value alone."""
    x = _signal()
    x[10000:10050] += 5.0
    cols = x[:, None]
    ungated, _ = emglib.calculate_rms(cols, RMS_S, FS)
    gated, qc = emglib.gated_peak_rms(cols, [2000], RMS_S, FS, gate_half_width_s=0.120)
    assert qc["ok"]
    assert gated[0] == pytest.approx(ungated[0], rel=1e-12)


def test_peaks_outside_the_segment_are_ignored():
    """Regression: callers pass the whole file's peak train, so most peaks fall outside any
    one phase. A negative stop index means 'from the end' in Python, so a peak just before the
    segment used to blank nearly all of it — which showed up as an all-NaN column."""
    x = _signal(n=8000)
    cols = x[:, None]
    ungated, _ = emglib.calculate_rms(cols, RMS_S, FS)
    for stray in (-300, -18590, -1, 8100, 50000):
        vals, qc = emglib.gated_peak_rms(cols, [stray], RMS_S, FS, gate_half_width_s=0.120)
        assert qc["ok"], f"peak at {stray} wrongly blanked the segment: {qc['reason']}"
        assert qc["survival"] == pytest.approx(1.0, abs=0.06)
        assert vals[0] == pytest.approx(ungated[0], rel=1e-9)
    # a peak straddling the start blanks only the overlap, not the whole segment
    vals, qc = emglib.gated_peak_rms(cols, [-100], RMS_S, FS, gate_half_width_s=0.120)
    assert qc["ok"] and 0.90 < qc["survival"] < 1.0


def test_gated_peak_refuses_when_too_little_survives():
    x = _signal(n=6000)
    peaks = np.arange(0, 6000, 300)            # a beat every 150 ms -> nothing left
    vals, qc = emglib.gated_peak_rms(x[:, None], peaks, RMS_S, FS, gate_half_width_s=0.120)
    assert not qc["ok"]
    assert qc["survival"] < 0.40
    assert all(np.isnan(v) for v in vals)
    assert "survives the gate" in qc["reason"]


# --- 4. detection quality ----------------------------------------------------

def test_detection_quality_accepts_a_clean_peak_train():
    peaks = np.arange(0, 30, 0.72)             # ~83 bpm, perfectly regular
    dq = emglib.detection_quality(peaks, 0.3613)
    assert dq["ok"] and dq["reason"] == ""
    assert dq["hr_bpm"] == pytest.approx(83.3, abs=0.5)


def test_detection_quality_flags_missed_beats():
    peaks = np.arange(0, 30, 0.72)
    peaks = np.delete(peaks, [5, 12, 19, 26])  # drop ~10% -> doubled RR intervals
    dq = emglib.detection_quality(peaks, 0.3613)
    assert not dq["ok"]
    assert dq["long_rr_frac"] > 0.02
    assert "missed" in dq["reason"]


def test_detection_quality_flags_the_refractory_ceiling():
    """ecg_min_distance_s = 0.3613 is a 166 bpm ceiling; 162 bpm is inside the margin."""
    peaks = np.arange(0, 10, 60.0 / 162)
    dq = emglib.detection_quality(peaks, 0.3613)
    assert not dq["ok"]
    assert "ceiling" in dq["reason"]


def test_detection_quality_needs_enough_peaks():
    assert not emglib.detection_quality([1.0, 2.0], 0.5)["ok"]


# --- 1. off by default, and inert when off -----------------------------------

def test_robust_peak_is_off_by_default():
    s = Settings.from_dict({})
    assert s.processing.emg.robust_peak.enabled is False


def test_settings_round_trip():
    s = Settings.from_dict({"processing": {"emg": {"robust_peak": {
        "enabled": True, "gate_half_width_s": 0.2, "max_long_rr_frac": 0.05}}}})
    rp = s.processing.emg.robust_peak
    assert rp.enabled and rp.gate_half_width_s == 0.2 and rp.max_long_rr_frac == 0.05
    assert rp.min_survival == 0.40                     # untouched fields keep their defaults
    assert s.to_dict()["processing"]["emg"]["robust_peak"]["enabled"] is True


def test_calculatemechanics_adds_no_keys_when_disabled():
    """The golden guarantee: feature off -> retbreath gains nothing."""
    from respmech.core import compute
    s = Settings.from_dict({})
    ret = {}
    compute._add_gated_peaks(ret, {}, _legacy(s), np.array([1.0, 2.0]), True, "")
    assert ret == {}


def test_gated_keys_are_nan_when_detection_is_rejected():
    from respmech.core import compute
    s = Settings.from_dict({"input": {"channels": {"emg": [2, 3]}},
                            "processing": {"emg": {"robust_peak": {"enabled": True}}}})
    ret = {}
    compute._add_gated_peaks(ret, {}, _legacy(s), np.array([1.0, 2.0]), False, "beats missed")
    assert set(ret) == {"rms_gated", "rms_gated_insp", "rms_gated_exp", "rms_gated_qc"}
    assert all(np.isnan(v) for v in ret["rms_gated"])
    assert len(ret["rms_gated"]) == 4                  # 2 channels + max + mean
    assert ret["rms_gated_qc"]["reason"] == "beats missed"


def _legacy(s):
    from respmech.core._legacy_ns import to_legacy_ns
    return to_legacy_ns(s)

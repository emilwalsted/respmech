"""Tests for shared-profile EMG noise reduction (respmech.core.noise) and the
pipeline's 'identical transformation for every file in a test' guarantee."""
import os

import numpy as np
import pytest

from respmech.core import noise as N

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")


def _clip(seed=0, n=4000):
    return np.random.default_rng(seed).normal(0, 1, n)


def test_trapezoid_compat_numpy2():
    """np.trapz was removed in NumPy 2.0 (-> np.trapezoid); the compat shim must work
    on both and match a manual trapezoidal integral."""
    from respmech.core._compat import trapezoid
    x = np.linspace(0, 1, 11)
    y = x ** 2
    manual = np.sum((y[:-1] + y[1:]) / 2 * np.diff(x))
    assert float(trapezoid(y, x)) == pytest.approx(manual)


def test_from_clip_determinism_and_serialisation():
    prof = N.NoiseProfile.from_clip(_clip(), 2000)
    x = _clip(1, 6000)
    a, b = prof.apply(x, 0.6), prof.apply(x, 0.6)
    assert np.array_equal(a, b)                       # deterministic
    assert len(a) == len(x)                           # length preserved
    prof2 = N.NoiseProfile.from_dict(prof.to_dict())  # round-trip
    assert np.array_equal(prof2.apply(x, 0.6), a)


def test_fixed_nfft_not_tied_to_clip_length():
    # legacy bug was n_fft = len(noise)**2; here n_fft is fixed regardless of clip.
    p1 = N.NoiseProfile.from_clip(_clip(0, 4000), 2000, n_fft=256)
    p2 = N.NoiseProfile.from_clip(_clip(2, 9000), 2000, n_fft=256)
    assert p1.n_fft == p2.n_fft == 256
    assert p1.noise_thresh.shape == p2.noise_thresh.shape  # same freq resolution


def test_too_short_reference_raises():
    with pytest.raises(N.NoiseProfileError):
        N.NoiseProfile.from_clip(_clip(0, 100), 2000, n_fft=256)


def test_fidelity_metric():
    sr = 2000
    t = np.arange(4000) / sr
    x = np.sin(2 * np.pi * 100 * t)                   # 100 Hz, in-band
    assert N.fidelity(x, x, sr) == pytest.approx(1.0)         # identity
    assert N.fidelity(x, 0.5 * x, sr) == pytest.approx(0.25, rel=1e-6)  # half amp -> quarter power


def test_select_prop_picks_highest_above_target():
    profiles = N.build_profiles(np.column_stack([_clip(0), _clip(1)]), 2000)
    sr = 2000
    t = np.arange(6000) / sr
    active = np.column_stack([np.sin(2 * np.pi * 90 * t), np.sin(2 * np.pi * 120 * t)])
    quiet = np.column_stack([0.2 * _clip(3, 6000), 0.2 * _clip(4, 6000)])
    prop, report = N.select_prop_decrease(profiles, active, quiet, sr, target=0.8)
    # chosen prop's worst-channel fidelity must be >= target (or gentlest if none qualify)
    worst = min(r["fidelity"] for r in report["channels"])
    assert worst >= 0.8 or prop == min(report["frontier"])
    # frontier is monotone: higher prop_decrease -> lower (or equal) fidelity
    props = sorted(report["frontier"])
    for ch in range(2):
        fids = [report["frontier"][p][ch] for p in props]
        assert all(fids[i] >= fids[i + 1] - 1e-9 for i in range(len(fids) - 1))


def test_profile_set_applies_per_channel():
    profiles = N.build_profiles(np.column_stack([_clip(0), _clip(1), _clip(2)]), 2000)
    pset = N.NoiseProfileSet(profiles, 0.5)
    cols = np.column_stack([_clip(10, 6000), _clip(11, 6000), _clip(12, 6000)])
    out = pset.apply_columns(cols)
    assert out.shape == cols.shape
    for i, prof in enumerate(profiles):
        assert np.array_equal(out[:, i], prof.apply(cols[:, i], 0.5))


@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input not present")
def test_identical_transformation_regardless_of_batch_subset():
    """The core requirement: a file is denoised IDENTICALLY whether processed alone
    or within the full test (same shared profile + fixed prop)."""
    from respmech.settingsio.migrate import migrate_dict
    from respmech.core.pipeline import run_batch

    legacy = {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": 1000},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": []}},
        "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                     "avgresamplingobs": 300},
                       "emg": {"remove_ecg": False, "remove_noise": True,
                               "noise_profile": [["synth_case_B.csv", "synth_case_A.csv", []]]}},
        "output": {"outputfolder": "/tmp/rm_noise_test",
                   "data": {"saveaveragedata": False, "savebreathbybreathdata": True}},
    }
    s, _ = migrate_dict(legacy)
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.use_expiration = False
    s.processing.emg.noise.reference_intervals = [[1.0, 5.0]]   # explicit 4 s window
    s.processing.emg.noise.auto_prop = False        # fixed prop -> batch-independent
    s.processing.emg.noise.prop_decrease = 0.5

    full = run_batch(s)
    single = run_batch(s, only_files=["synth_case_B.csv"])
    tb_full = full.ok_files["synth_case_B.csv"].breaths_table.reset_index(drop=True)
    tb_single = single.ok_files["synth_case_B.csv"].breaths_table.reset_index(drop=True)
    rms_cols = [c for c in tb_full.columns if "rms" in str(c).lower()]
    assert rms_cols
    for c in rms_cols:
        assert np.allclose(tb_full[c].to_numpy(float), tb_single[c].to_numpy(float),
                           rtol=1e-12, atol=1e-15), f"{c} differs between batch and single-file run"

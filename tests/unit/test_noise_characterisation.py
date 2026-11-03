"""Characterisation of the spectral-gating noise reduction, number for number.

No golden scenario switches noise reduction on, and the behavioural tests in
``test_noise.py`` check directions (less noise, fidelity within bounds), so mutation
testing found that the threshold formula, the mask smoothing, the reconstruction and
the selection report could all change without any test noticing. These tests pin the
current output on one deterministic synthetic EMG-like signal. They are a tripwire,
not a specification: a deliberate change to the algorithm re-pins them in the same
commit, with the reason in the commit message.
"""
import numpy as np
import pytest

from respmech.core import noise as N

SR = 2000
RTOL = 1e-9


def _signals():
    rng = np.random.default_rng(42)
    t = np.arange(8000) / SR
    quiet = rng.normal(0, 0.2, t.size)
    burst = (np.sin(2 * np.pi * 1.5 * t) > 0.3) * np.sin(2 * np.pi * 90 * t)
    active = quiet + burst + 0.05 * np.sin(2 * np.pi * 250 * t)
    ref = rng.normal(0, 0.2, 6000)
    return ref, active, quiet


@pytest.fixture(scope="module")
def sig():
    return _signals()


@pytest.fixture(scope="module")
def profile(sig):
    ref, _, _ = sig
    return N.NoiseProfile.from_clip(ref, SR)


def test_noise_threshold_is_pinned(profile):
    assert float(np.sum(profile.noise_thresh)) == pytest.approx(1146.296267413893, rel=RTOL)
    assert float(profile.noise_thresh[0]) == pytest.approx(10.449795356192174, rel=RTOL)
    assert float(profile.noise_thresh[-1]) == pytest.approx(11.014050913265184, rel=RTOL)


@pytest.mark.parametrize("prop, energy, sample", [
    (0.0, 1897.5070745091302, 0.11540197375786289),
    (0.5, 1088.7903671374827, -0.010481306437286978),
    (1.0, 906.6471419261411, -0.008455409350980703),
])
def test_gated_signal_is_pinned(sig, profile, prop, energy, sample):
    _, active, _ = sig
    out = profile.apply(active, prop)
    assert len(out) == len(active)
    assert float(np.sum(out ** 2)) == pytest.approx(energy, rel=RTOL)
    assert float(out[1234]) == pytest.approx(sample, rel=RTOL)


def test_profile_survives_a_round_trip_through_its_dict(sig, profile):
    _, active, _ = sig
    again = N.NoiseProfile.from_dict(profile.to_dict())
    np.testing.assert_array_equal(again.apply(active, 0.5), profile.apply(active, 0.5))
    pset = N.NoiseProfileSet([profile, again], 0.5)
    back = N.NoiseProfileSet.from_dict(pset.to_dict())
    cols = np.column_stack([active, active])
    np.testing.assert_array_equal(back.apply_columns(cols), pset.apply_columns(cols))


def test_profile_set_refuses_a_channel_count_mismatch(sig, profile):
    _, active, _ = sig
    with pytest.raises(ValueError, match="channel count"):
        N.NoiseProfileSet([profile], 0.5).apply_columns(np.column_stack([active, active]))


def test_snr_and_fidelity_are_pinned(sig, profile):
    _, active, quiet = sig
    assert float(N.snr_db(active, quiet, SR)) == pytest.approx(13.716226321858453, rel=RTOL)
    assert float(N.fidelity(active, profile.apply(active, 0.7), SR)) == \
        pytest.approx(0.5793698865914899, rel=RTOL)


def test_selection_report_is_pinned(sig):
    ref, active, quiet = sig
    profs = N.build_profiles(np.column_stack([ref, ref[::-1]]), SR)
    prop, rep = N.select_prop_decrease(
        profs, np.column_stack([active, active]), np.column_stack([quiet, quiet]), SR)
    assert prop == 0.1
    assert sorted(rep) == ["channels", "fidelity_target", "frontier", "prop_decrease"]
    assert rep["prop_decrease"] == 0.1
    assert rep["fidelity_target"] == N.DEFAULT_FIDELITY_TARGET
    row = rep["channels"][0]
    assert sorted(row) == ["channel", "delta_snr_db", "fidelity"]
    assert row["channel"] == 0
    assert row["fidelity"] == pytest.approx(0.876523041871953, rel=RTOL)
    assert row["delta_snr_db"] == pytest.approx(4.248877521020852, rel=RTOL)
    frontier = {round(float(p), 1): f for p, f in rep["frontier"].items()}
    assert frontier[0.1] == [0.877, 0.867]
    assert frontier[0.5] == [0.638, 0.611]
    assert frontier[1.0] == [0.521, 0.488]

"""``core.analysis.normal_range``: the GLI 2022 reference set, the shape-scaled normal
flow-volume range, the registry seam, and how the MFVL figure draws it."""
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from respmech.core.analysis import normal_range as nr
from respmech.core.settings import Settings, SettingsError, SubjectEntry


# --------------------------------------------------------------------------- GLI 2022
def test_gli2022_pinned_values_for_a_40_year_old_man_of_180_cm():
    # Regression values; cross-checked in the implementing session against the independent
    # pyspiro implementation of the same tables (agreement within 0.0025 L over a grid of
    # ages and heights; the small gap is pyspiro rounding the age before the equation).
    p = nr.get_reference("gli_2022").predict("male", 40.0, 180.0)
    assert p.fvc == pytest.approx(5.074, abs=0.01)
    assert p.fev1 == pytest.approx(4.111, abs=0.01)
    assert p.fvc_lln == pytest.approx(3.907, abs=0.01)
    assert p.fev1_lln == pytest.approx(3.150, abs=0.01)


def test_gli2022_pinned_values_for_a_30_year_old_woman_of_168_cm():
    p = nr.Gli2022().predict("female", 30.0, 168.0)
    assert p.fvc == pytest.approx(3.862, abs=0.01)
    assert p.fev1 == pytest.approx(3.273, abs=0.01)


def test_limits_bracket_the_expected_value_and_sit_five_percent_apart_in_z():
    p = nr.Gli2022().predict("male", 55.0, 175.0)
    assert p.fvc_lln < p.fvc < p.fvc_uln
    assert p.fev1_lln < p.fev1 < p.fev1_uln


def test_taller_means_larger_and_older_means_smaller_in_adulthood():
    g = nr.Gli2022()
    assert g.predict("male", 40, 190).fvc > g.predict("male", 40, 170).fvc
    assert g.predict("male", 70, 180).fev1 < g.predict("male", 30, 180).fev1


@pytest.mark.parametrize("sex,age,ht", [
    ("male", 2.9, 170), ("male", 95.5, 170), ("male", 40, 60), ("female", 40, 260),
    ("other", 40, 170), (None, 40, 170), ("male", float("nan"), 170), ("male", "x", 170)])
def test_outside_the_validity_range_nothing_is_predicted(sex, age, ht):
    assert nr.Gli2022().predict(sex, age, ht) is None


def test_sex_spellings_are_normalised():
    assert nr.normalise_sex(" Male ") == "male"
    assert nr.normalise_sex("F") == "female"
    assert nr.normalise_sex("x") is None and nr.normalise_sex(3) is None


# --------------------------------------------------------------------------- registry
def test_registry_default_and_unknown_key():
    assert nr.get_reference().key == nr.DEFAULT_REFERENCE == "gli_2022"
    with pytest.raises(KeyError):
        nr.get_reference("no_such_reference")


def test_a_registered_reference_is_used_by_normal_band():
    class Flat:
        key, label, citation = "flat_test", "FLAT", "test"

        def predict(self, sex, age_years, height_cm):
            return nr.SpirometryPrediction(4.0, 3.0, 5.0, 3.2, 2.4, 4.0)

    nr.register(Flat())
    try:
        band = nr.normal_band("male", 40, 180, reference="flat_test")
        assert band.reference_label == "FLAT"
        assert band.v_expected[-1] == pytest.approx(4.0)
    finally:
        nr.REGISTRY.pop("flat_test")


# --------------------------------------------------------------------------- the curve
def _exhaled_after_one_second(v, f):
    t = np.concatenate([[0.0], np.cumsum(np.diff(v) / np.maximum(0.5 * (f[1:] + f[:-1]), 1e-9))])
    return float(np.interp(1.0, t, v))


@pytest.mark.parametrize("sex,age,ht", [("male", 40, 180), ("female", 30, 168),
                                         ("male", 70, 170), ("female", 20, 160)])
def test_expected_curve_reproduces_gli_fvc_and_fev1(sex, age, ht):
    band = nr.normal_band(sex, age, ht)
    assert band.v_expected[0] == 0.0 and band.flow_expected[0] == 0.0
    assert band.v_expected[-1] == pytest.approx(band.prediction.fvc)
    assert band.flow_expected[-1] == pytest.approx(0.0, abs=1e-9)
    assert _exhaled_after_one_second(band.v_expected, band.flow_expected) == pytest.approx(
        band.prediction.fev1, rel=2e-3)


def test_curve_is_a_plausible_adult_expiratory_limb():
    band = nr.normal_band("male", 40, 180)
    f = band.flow_expected
    assert 6.0 < f.max() < 12.0                      # peak flow in L/s
    peak_ix = int(np.argmax(f))
    assert band.v_expected[peak_ix] / band.prediction.fvc == pytest.approx(
        nr.PEF_VOLUME_FRACTION, abs=0.02)
    assert np.all(np.diff(f[peak_ix:]) <= 1e-9)      # falling after the peak
    assert np.all(np.isfinite(f)) and np.all(f >= 0)


def test_band_edges_are_ordered_and_the_expected_curve_lies_inside():
    band = nr.normal_band("female", 45, 165)
    assert np.all(band.flow_upper >= band.flow_lower)
    mid = np.interp(band.v_band, band.v_expected, band.flow_expected, right=0.0)
    assert np.all(mid <= band.flow_upper + 1e-6)
    assert np.all(mid >= band.flow_lower - 1e-6)


def test_missing_or_invalid_demographics_give_no_band():
    assert nr.normal_band(None, 40, 180) is None
    assert nr.normal_band("male", None, 180) is None
    assert nr.normal_band("male", 40, None) is None
    assert nr.normal_band("male", 120, 180) is None


def test_the_shape_clamps_to_the_eccs_range_instead_of_failing():
    # a 12-year-old is outside ECCS (18-70) but inside GLI: still drawable
    assert nr.normal_band("male", 12, 150) is not None
    assert nr.normal_band("female", 85, 150) is not None


# --------------------------------------------------------------------------- settings
def _settings_with(**kw):
    s = Settings()
    s.input.subjects.append(SubjectEntry(key="P01", **kw))
    return s


def test_normal_band_for_uses_the_subject_of_the_file():
    s = _settings_with(sex="male", age_years=40.0, height_cm=180.0)
    band = nr.normal_band_for(s, "P01_trial2.csv")
    assert band is not None and band.v_expected[-1] == pytest.approx(5.074, abs=0.01)
    assert nr.normal_band_for(s, "someone_else_01.csv") is None


def test_no_band_without_all_three_demographics():
    for kw in ({"sex": "male", "age_years": 40.0}, {"sex": "male", "height_cm": 180.0},
               {"age_years": 40.0, "height_cm": 180.0}, {}):
        assert nr.normal_band_for(_settings_with(**kw), "P01_trial2.csv") is None


@pytest.mark.parametrize("kw", [{"sex": "x"}, {"age_years": 2.0}, {"age_years": 99.0},
                                 {"height_cm": 50.0}, {"height_cm": 300.0}])
def test_settings_validation_rejects_implausible_demographics(kw):
    with pytest.raises(SettingsError):
        _settings_with(**kw).validate()


def test_demographics_round_trip_through_toml(tmp_path):
    from respmech.settingsio.toml_io import load_toml, save_toml
    s = _settings_with(sex="female", age_years=33.5, height_cm=171.0)
    path = tmp_path / "a.toml"
    save_toml(s, path)
    back = load_toml(path)
    sub = back.input.subjects[0]
    assert (sub.sex, sub.age_years, sub.height_cm) == ("female", 33.5, 171.0)


# --------------------------------------------------------------------------- drawing
def test_draw_normal_band_adds_a_band_a_curve_and_a_discreet_note():
    from respmech.core.plots import draw_normal_band
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    draw_normal_band(ax, nr.normal_band("male", 40, 180))
    assert len(ax.collections) == 1 and len(ax.lines) == 1
    note = ax.texts[0].get_text()
    assert "GLI 2022" in note and "ECCS 1993" in note
    assert ax.texts[0].get_fontsize() <= 7
    plt.close(fig)


def test_draw_normal_band_with_none_draws_nothing():
    from respmech.core.plots import draw_normal_band
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    draw_normal_band(ax, None)
    assert not ax.collections and not ax.lines and not ax.texts
    plt.close(fig)

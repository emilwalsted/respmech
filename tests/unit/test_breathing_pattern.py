"""core.analysis.breathing_pattern: opt-in flow/volume-only breathing-pattern columns.

The per-breath columns are pinned analytically on a synthetic sinusoidal breath whose
half-waves are an exact number of samples long, so every peak lands on a sample and the
expected values follow from the closed forms. The per-file coefficients of variation are
pinned on small hand-computable series, and the switches on a real ``run_batch`` over the
synthetic recording: with both flags off nothing is added, with them on the new columns are
present and agree with the ordinary timing columns of the same table.
"""
import math

import numpy as np
import pandas as pd
import pytest

from _helpers import assert_units, requires_synth, synth_settings

from respmech.core.analysis import breathing_pattern as bp
from respmech.core.analysis.registry import REGISTRY, resolve
from respmech.core.analysis.signals import Capabilities
from respmech.core.settings import BreathingPatternSettings, SettingsError

FS = 100.0
N_INSP, N_EXP = 150, 250           # Ti = 1.5 s, Te = 2.5 s
A_IN, A_EX = 0.6, 0.36             # peak flows, L/s


def _sine_breath(n_insp=N_INSP, n_exp=N_EXP, a_in=A_IN, a_ex=A_EX, fs=FS):
    """A raw breath dict: inspiratory flow negative (half-sine), expiratory positive, volume
    rising over inspiration and returning to zero over expiration."""
    ti, te = n_insp / fs, n_exp / fs
    ki = np.arange(n_insp)
    ke = np.arange(n_exp)
    flow_i = -a_in * np.sin(np.pi * ki / n_insp)
    vol_i = a_in * ti / np.pi * (1 - np.cos(np.pi * ki / n_insp))
    vmax = 2 * a_in * ti / np.pi
    flow_e = a_ex * np.sin(np.pi * ke / n_exp)
    vol_e = vmax * (1 + np.cos(np.pi * ke / n_exp)) / 2
    return {
        "inspiration": {"flow": flow_i, "volume": vol_i},
        "expiration": {"flow": flow_e, "volume": vol_e},
        "flow": np.concatenate([flow_i, flow_e]),
        "volume": np.concatenate([vol_i, vol_e]),
    }, vmax, ti, te


# --- per-breath columns, analytically ---------------------------------------------------

def test_extended_columns_on_a_sinusoidal_breath():
    breath, vmax, ti, te = _sine_breath()
    out = bp.breath_columns(breath, FS)

    assert list(out) == list(bp.EXTENDED_COLUMNS)
    # the volume moved in each phase is the analytic half-sine area (the last sample sits
    # one step short of the phase end, hence the small relative tolerance)
    assert out["vol_insp"] == pytest.approx(vmax, rel=1e-3)
    assert out["vol_exp"] == pytest.approx(vmax, rel=1e-3)
    vt = float(np.max(breath["volume"]) - np.min(breath["volume"]))
    assert out["mean_in_flow"] == pytest.approx(vt / ti, rel=1e-12)
    assert out["mean_ex_flow"] == pytest.approx(vt / te, rel=1e-12)
    # for a half-sine the mean flow is 2/pi of the peak flow
    assert out["mean_in_flow"] == pytest.approx(2 / math.pi * A_IN, rel=1e-3)
    assert out["mean_ex_flow"] == pytest.approx(2 / math.pi * A_EX, rel=1e-3)
    assert out["bf_inst"] == pytest.approx(60.0 / (ti + te), rel=1e-12)
    # both peaks fall exactly mid-phase
    assert out["t_peak_in_flow"] == pytest.approx(ti / 2, abs=1e-12)
    assert out["t_peak_ex_flow"] == pytest.approx(te / 2, abs=1e-12)
    assert out["t_peak_in_flow_frac"] == pytest.approx(0.5, abs=1e-12)


def test_peak_timing_follows_an_early_inspiratory_peak():
    """A peak at a quarter of inspiration is reported as such: the fraction is measured on
    the inspiratory time, not on the whole breath."""
    breath, _vmax, ti, _te = _sine_breath()
    flow = np.zeros(N_INSP)
    flow[N_INSP // 4] = -1.0
    flow[N_INSP // 4 + 1:] = -0.2
    breath["inspiration"]["flow"] = flow
    out = bp.breath_columns(breath, FS)
    assert out["t_peak_in_flow"] == pytest.approx(N_INSP // 4 / FS)
    assert out["t_peak_in_flow_frac"] == pytest.approx((N_INSP // 4) / N_INSP, abs=1e-12)


def test_an_empty_phase_gives_nan_rather_than_an_error():
    breath, *_ = _sine_breath()
    breath["expiration"] = {"flow": np.array([]), "volume": np.array([])}
    out = bp.breath_columns(breath, FS)
    assert math.isnan(out["mean_ex_flow"])
    assert math.isnan(out["vol_exp"])
    assert math.isnan(out["t_peak_ex_flow"])
    assert math.isfinite(out["mean_in_flow"])


def test_attach_stamps_the_ext_dict_and_isolates_a_fault():
    from types import SimpleNamespace
    settings = SimpleNamespace(input=SimpleNamespace(format=SimpleNamespace(samplingfrequency=FS)))

    breath, *_ = _sine_breath()
    assert bp.attach(breath, settings) is None
    assert set(breath["breathing_pattern_ext"]) == set(bp.EXTENDED_COLUMNS)

    broken = {"inspiration": {}}                    # no flow at all -> a KeyError inside
    note = bp.attach(broken, settings)
    assert note and "breathing pattern failed" in note
    assert all(math.isnan(v) for v in broken["breathing_pattern_ext"].values())
    assert set(broken["breathing_pattern_ext"]) == set(bp.EXTENDED_COLUMNS)


# --- per-file variability ---------------------------------------------------------------

def test_cv_pct_is_sample_sd_over_a_positive_mean():
    # [1, 2, 3]: mean 2, sample SD 1 -> 50 %
    assert bp.cv_pct([1.0, 2.0, 3.0]) == pytest.approx(50.0, rel=1e-12)
    # a constant series has no variability
    assert bp.cv_pct([2.0, 2.0, 2.0, 2.0]) == 0.0


def test_cv_pct_is_nan_below_three_breaths_and_for_a_non_positive_mean():
    assert math.isnan(bp.cv_pct([1.0, 2.0]))
    assert math.isnan(bp.cv_pct([]))
    assert math.isnan(bp.cv_pct([-1.0, -2.0, -3.0]))
    assert math.isnan(bp.cv_pct([-1.0, 0.0, 1.0]))          # zero mean
    # non-finite entries do not count towards the three
    assert math.isnan(bp.cv_pct([1.0, 2.0, float("nan")]))


def test_file_variability_columns_and_count():
    table = pd.DataFrame({
        "vt": [1.0, 2.0, 3.0], "ti": [1.0, 1.0, 1.0], "te": [2.0, 4.0, 6.0],
        "ttot": [3.0, 5.0, 7.0], "ti_ttot": [0.5, 0.5, 0.5],
    })
    out = bp.file_variability(table)
    assert list(out) == list(bp.VARIABILITY_COLUMNS)
    assert out["vt_cv"] == pytest.approx(50.0)
    assert out["ti_cv"] == 0.0
    assert out["te_cv"] == pytest.approx(50.0)
    assert out["ttot_cv"] == pytest.approx(100.0 * 2.0 / 5.0)
    assert out["ti_ttot_cv"] == 0.0
    assert out["n_breaths"] == 3


def test_file_variability_with_two_breaths_keeps_the_count_but_no_cv():
    table = pd.DataFrame({"vt": [1.0, 2.0], "ti": [1.0, 1.1], "te": [2.0, 2.2],
                          "ttot": [3.0, 3.3], "ti_ttot": [0.3, 0.3]})
    out = bp.file_variability(table)
    assert out["n_breaths"] == 2
    assert all(math.isnan(out[c]) for c in bp.VARIABILITY_COLUMNS if c != "n_breaths")


def test_attach_variability_writes_onto_the_average_row():
    table = pd.DataFrame({"vt": [1.0, 2.0, 3.0], "ti": [1.0, 1.0, 1.0], "te": [1.0, 1.0, 1.0],
                          "ttot": [2.0, 2.0, 2.0], "ti_ttot": [0.5, 0.5, 0.5]})
    row = pd.DataFrame({"file": ["x.csv"], "vt": [2.0]})
    bp.attach_variability(table, row)
    assert list(row.columns)[-len(bp.VARIABILITY_COLUMNS):] == list(bp.VARIABILITY_COLUMNS)
    assert row["vt_cv"].iloc[0] == pytest.approx(50.0)


# --- units, registry, settings ----------------------------------------------------------

def test_every_column_resolves_to_its_designed_unit():
    assert_units({
        "mean_in_flow": "L·s⁻¹", "mean_ex_flow": "L·s⁻¹",
        "vol_insp": "L", "vol_exp": "L",
        "bf_inst": "min⁻¹",
        "t_peak_in_flow": "s", "t_peak_ex_flow": "s", "t_peak_in_flow_frac": "—",
        "vt_cv": "%", "ti_cv": "%", "te_cv": "%", "ttot_cv": "%", "ti_ttot_cv": "%",
        "n_breaths": "",
    })


def test_registry_declares_every_column_and_names_the_module_for_a_flow_set():
    declared = {spec.name for spec in REGISTRY if spec.module == "breathing_pattern"}
    assert declared == set(bp.EXTENDED_COLUMNS) | set(bp.VARIABILITY_COLUMNS)
    assert all(spec.trigger == "breathing_pattern"
               for spec in REGISTRY if spec.module == "breathing_pattern")
    files = {spec.name for spec in REGISTRY
             if spec.module == "breathing_pattern" and spec.level == "file"}
    assert files == set(bp.VARIABILITY_COLUMNS)
    assert "breathing_pattern" in resolve(Capabilities.FULL)


def test_registry_does_not_offer_the_module_without_flow():
    caps = Capabilities(flow=False, volume=False, poes=False, pgas=False, pdi=False, emg=True,
                        entropy=False, declared=frozenset({"emg"}), mode="emg_only")
    assert "breathing_pattern" not in resolve(caps)


def test_settings_default_off_and_validate_rejects_non_bool(tmp_path):
    s = synth_settings(tmp_path)
    assert s.processing.breathing_pattern == BreathingPatternSettings(False, False)
    s.validate()
    s.processing.breathing_pattern.extended = "yes"
    with pytest.raises(SettingsError, match="breathing_pattern.extended"):
        s.validate()


def test_settings_round_trip_through_toml(tmp_path):
    from respmech.settingsio.toml_io import load_toml, save_toml

    s = synth_settings(tmp_path)
    s.processing.breathing_pattern.extended = True
    s.processing.breathing_pattern.variability = True
    path = tmp_path / "a.toml"
    save_toml(s, path)
    assert "[processing.breathing_pattern]" in path.read_text(encoding="utf-8")
    back = load_toml(path)
    assert back.processing.breathing_pattern == BreathingPatternSettings(True, True)


# --- end to end through run_batch -------------------------------------------------------

_FLOW_ONLY = {"poes": None, "pgas": None, "pdi": None, "emg": [], "entropy": []}


@requires_synth()
def test_switches_off_add_nothing_and_on_add_exactly_the_designed_columns(tmp_path):
    from respmech.core.pipeline import run_batch

    off = run_batch(synth_settings(tmp_path / "off", channels=_FLOW_ONLY),
                    only_files=["synth_case_A.csv"])
    fr_off = off.ok_files["synth_case_A.csv"]
    assert not (set(fr_off.breaths_table.columns) & set(bp.EXTENDED_COLUMNS))
    assert not (set(fr_off.average_row.columns) & set(bp.VARIABILITY_COLUMNS))

    s = synth_settings(tmp_path / "on", channels=_FLOW_ONLY)
    s.processing.breathing_pattern.extended = True
    s.processing.breathing_pattern.variability = True
    on = run_batch(s, only_files=["synth_case_A.csv"])
    assert on.failed_files == {}
    fr = on.ok_files["synth_case_A.csv"]

    # per-breath: exactly the eight extended columns, none of the per-file ones
    added = set(fr.breaths_table.columns) - set(fr_off.breaths_table.columns)
    assert added == set(bp.EXTENDED_COLUMNS)
    # per-file: the six variability columns on the average row, and the average row also
    # carries the mean of each per-breath column
    added_avg = set(fr.average_row.columns) - set(fr_off.average_row.columns)
    assert added_avg == set(bp.EXTENDED_COLUMNS) | set(bp.VARIABILITY_COLUMNS)

    # every column the switches did not add is unchanged
    common = [c for c in fr_off.breaths_table.columns]
    pd.testing.assert_frame_equal(fr.breaths_table[common], fr_off.breaths_table[common])

    t = fr.breaths_table
    # the new columns agree with the ordinary timing columns of the same rows
    np.testing.assert_allclose(t["mean_in_flow"], t["vt"] / t["ti"], rtol=1e-12)
    np.testing.assert_allclose(t["mean_ex_flow"], t["vt"] / t["te"], rtol=1e-12)
    np.testing.assert_allclose(t["bf_inst"], 60.0 / t["ttot"], rtol=1e-12)
    np.testing.assert_allclose(t["t_peak_in_flow_frac"], t["t_peak_in_flow"] / t["ti"], rtol=1e-12)
    assert (t["t_peak_in_flow"] <= t["ti"]).all() and (t["t_peak_ex_flow"] <= t["te"]).all()

    row = fr.average_row.iloc[0]
    assert row["n_breaths"] == len(t)
    if len(t) >= bp.MIN_BREATHS_FOR_CV:
        assert row["vt_cv"] == pytest.approx(bp.cv_pct(t["vt"]), rel=1e-12)
        assert row["ti_ttot_cv"] == pytest.approx(bp.cv_pct(t["ti_ttot"]), rel=1e-12)
    assert row["bf_inst"] == pytest.approx(float(t["bf_inst"].mean()), rel=1e-12)


@requires_synth()
def test_only_the_switched_on_group_is_added(tmp_path):
    from respmech.core.pipeline import run_batch

    s = synth_settings(tmp_path / "ext", channels=_FLOW_ONLY)
    s.processing.breathing_pattern.extended = True
    fr = run_batch(s, only_files=["synth_case_A.csv"]).ok_files["synth_case_A.csv"]
    assert set(bp.EXTENDED_COLUMNS) <= set(fr.breaths_table.columns)
    assert not (set(fr.average_row.columns) & set(bp.VARIABILITY_COLUMNS))

    s = synth_settings(tmp_path / "var", channels=_FLOW_ONLY)
    s.processing.breathing_pattern.variability = True
    fr = run_batch(s, only_files=["synth_case_A.csv"]).ok_files["synth_case_A.csv"]
    assert not (set(fr.breaths_table.columns) & set(bp.EXTENDED_COLUMNS))
    assert set(bp.VARIABILITY_COLUMNS) <= set(fr.average_row.columns)


@requires_synth()
def test_full_channel_run_gets_the_same_columns_as_flow_only(tmp_path):
    """The pattern columns read only flow and volume, so a full-channel run of the same
    recording gives the same values (rtol 1e-9)."""
    from respmech.core.pipeline import run_batch

    s_fo = synth_settings(tmp_path / "fo", channels=_FLOW_ONLY)
    s_full = synth_settings(tmp_path / "full", channels={"emg": [], "entropy": []})
    for s in (s_fo, s_full):
        s.processing.breathing_pattern.extended = True
        s.processing.breathing_pattern.variability = True
    fo = run_batch(s_fo, only_files=["synth_case_A.csv"]).ok_files["synth_case_A.csv"]
    full = run_batch(s_full, only_files=["synth_case_A.csv"]).ok_files["synth_case_A.csv"]
    cols = list(bp.EXTENDED_COLUMNS)
    np.testing.assert_allclose(fo.breaths_table[cols].to_numpy(dtype=float),
                               full.breaths_table[cols].to_numpy(dtype=float), rtol=1e-9)
    pcols = list(bp.VARIABILITY_COLUMNS)
    np.testing.assert_allclose(fo.average_row[pcols].to_numpy(dtype=float),
                               full.average_row[pcols].to_numpy(dtype=float), rtol=1e-9)


@requires_synth()
def test_provenance_names_the_pattern_columns_only_when_on(tmp_path):
    from respmech.core.io import writers
    import datetime as dt

    s = synth_settings(tmp_path, channels=_FLOW_ONLY)
    s.processing.breathing_pattern.extended = True
    rows = writers._provenance_rows(s, dt.datetime(2026, 1, 1))
    assert "Breathing pattern" in list(rows["Key"])

    s.processing.breathing_pattern.extended = False
    rows = writers._provenance_rows(s, dt.datetime(2026, 1, 1))
    assert "Breathing pattern" not in list(rows["Key"])


def test_a_breathing_pattern_edit_reruns_only_the_batch_pass():
    """The switches add columns in the post-mechanics pass and never touch a raw trace, so
    an edit must not fall through to the wide default that recomputes every panel."""
    from respmech.ui.screens.preview_screen import _kinds_for_settings_path

    for path in ("processing.breathing_pattern.extended",
                 "processing.breathing_pattern.variability"):
        assert _kinds_for_settings_path(path) == frozenset(("batch",))

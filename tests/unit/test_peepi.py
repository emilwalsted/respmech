"""core.analysis.pressure: opt-in PEEPi detection and the modified Campbell diagram's
threshold work.

The pinned case is a synthetic recording whose every breath but the first is preceded by an
end-expiratory pause of exactly zero flow (0.4 s), the last 0.25 s of which carries a linear
Poes fall of a known size ending on the last pause sample. The segmenter's own boundary lands
in the pause, up to ``buffer`` samples before flow starts, so the answer must not depend on
``breathseparationbuffer``: ``peepi_dyn`` is the analytical fall, to 1e-9.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "golden"))
import generate_data as gd  # noqa: E402  (tests/golden's own synthetic-data generator)

from respmech.core import compute  # noqa: E402
from respmech.core import quantities  # noqa: E402
from respmech.core.analysis import pressure  # noqa: E402
from respmech.core.analysis.signals import Capabilities  # noqa: E402
from respmech.core.settings import PeepiSettings, PressureSettings, Settings  # noqa: E402

FS = gd.FS
BCNT, VEFACTOR = 6, 10.0


def _caps(**over):
    kw = dict(flow=True, volume=True, poes=True, pgas=True, pdi=True, emg=False,
              entropy=False, declared=frozenset({"flow", "poes", "pgas", "pdi"}), mode="test")
    kw.update(over)
    return Capabilities(**kw)


def _settings(caps=None, **peepi):
    return SimpleNamespace(
        capabilities=caps or _caps(),
        input=SimpleNamespace(format=SimpleNamespace(samplingfrequency=FS)),
        processing=SimpleNamespace(pressure=PressureSettings(peepi=PeepiSettings(enabled=True, **peepi))),
    )


def _breaths(buffer=800, n_breaths=4, **wave):
    flow, vol, poes, pgas = gd.peepi_channels(n_breaths, **wave)
    n = len(flow)
    ns = SimpleNamespace(processing=SimpleNamespace(mechanics=SimpleNamespace(
        breathseparationbuffer=buffer, excludebreaths=[], breathtypes=[])))
    empty = np.array([])
    return compute.separateintobreathsbyflow(
        "synth.csv", np.arange(n) / FS, flow, vol, poes, pgas, pgas - poes, empty, empty, ns)


def _with_mechanics(breath):
    """The values calculatemechanics/calculatewob would have left, fixed so the arithmetic in
    attach() can be checked by hand."""
    breath["mechanics"] = {"vt": 0.8, "ti": 1.2, "int_oesinsp": 2.0, "int_pdiinsp": 3.0}
    breath["wob"] = {"wobtotal": 1.5, "wob_in_total": 1.0}
    return breath


def _run(breaths, settings, which):
    keys = list(breaths)
    i = keys.index(which)
    b = _with_mechanics(breaths[which])
    prev = breaths[keys[i - 1]] if i else None
    notice = pressure.attach(b, prev, BCNT, VEFACTOR, settings)
    return b["pressure_ext"], notice


# --- the true zero crossing ---------------------------------------------------------------

def test_true_flow_start_is_the_first_negative_sample_after_a_non_negative_one():
    flow = np.array([0.0, 0.0, 0.0, -0.1, -0.2, -0.1])
    assert pressure.true_flow_start(flow) == 3


def test_true_flow_start_falls_back_to_the_phase_start():
    assert pressure.true_flow_start(np.array([-0.1, -0.2, -0.1])) == 0     # already flowing
    assert pressure.true_flow_start(np.zeros(5)) == 0                       # never flows
    assert pressure.true_flow_start(np.array([])) == 0


@pytest.mark.parametrize("buffer", [500, 800, 1200])
def test_the_segment_boundary_is_not_the_zero_crossing(buffer):
    """The premise of the whole module: with an exact-zero pause the segmenter puts the breath
    boundary at the START of the pause (400 samples before flow does, whatever buffer exceeds
    the pause; a buffer of 400 or less is refused as flat flow), and t_flow finds the real
    start. NB the three buffers give the same segmentation: this pins that the boundary is
    not the zero crossing, not that buffer changes anything."""
    breaths = _breaths(buffer)
    insp = breaths[2]["inspiration"]
    # (the segmenter's phase slices are end-exclusive, so each phase loses its last sample)
    assert len(insp["flow"]) == gd.PEEPI_PAUSE_N + gd.PEEPI_INSP_N - 1
    assert pressure.true_flow_start(insp["flow"]) == gd.PEEPI_PAUSE_N


# --- the pinned pause case ------------------------------------------------------------------

@pytest.mark.parametrize("buffer", [500, 800, 1200])
def test_peepi_dyn_is_the_analytical_fall_with_the_boundary_400_samples_before_flow(buffer):
    breaths = _breaths(buffer)
    st = _settings()
    for which in (2, 3, 4):
        cols, notice = _run(breaths, st, which)
        assert notice is None
        assert cols["peepi_dyn"] == pytest.approx(gd.PEEPI_DROP, abs=1e-9)
        assert cols["peepi_pgas_drop"] == pytest.approx(gd.PEEPI_PGAS_DROP, abs=1e-9)
        assert cols["peepi_corr"] == pytest.approx(gd.PEEPI_DROP - gd.PEEPI_PGAS_DROP, abs=1e-9)


def test_lag_and_preflow_area_cover_the_deflection():
    cols, _ = _run(_breaths(800), _settings(), 2)
    # onset is found in the flat stretch just before the ramp (smoothing widens the walk by
    # a few samples), never inside the ramp and never absurdly early
    assert gd.PEEPI_RAMP_N / FS <= cols["peepi_lag"] <= (gd.PEEPI_RAMP_N + 50) / FS
    # area of a linear fall of `drop` over `ramp_n` samples, referenced to the onset value
    ramp_area = gd.PEEPI_DROP * (gd.PEEPI_RAMP_N / FS) / 2
    assert cols["int_oes_preflow"] == pytest.approx(ramp_area, rel=0.02)
    assert cols["ptp_oes_preflow"] == pytest.approx(cols["int_oes_preflow"] * BCNT * VEFACTOR)


def test_boundary_in_the_pause_adds_nothing_the_existing_columns_already_hold():
    """The segmenter's boundary lands in the pause BEFORE the deflection, so the existing
    PTP baseline and the Campbell polygon's references are already the pre-deflection level
    and contain the threshold: the ``*_peepi``/``*_thr`` columns equal the plain ones."""
    cols, _ = _run(_breaths(800), _settings(), 2)
    assert cols["wob_in_thr"] == 0.0
    assert cols["wob_in_total_thr"] == pytest.approx(1.0)
    assert cols["wobtotal_thr"] == pytest.approx(1.5)
    assert cols["int_oesinsp_peepi"] == pytest.approx(2.0, abs=1e-9)
    assert cols["ptp_oesinsp_peepi"] == pytest.approx(2.0 * BCNT * VEFACTOR, abs=1e-9)
    assert cols["int_pdiinsp_peepi"] == pytest.approx(3.0, abs=1e-9)
    assert cols["ptp_pdiinsp_peepi"] == pytest.approx(3.0 * BCNT * VEFACTOR, abs=1e-9)


def _late_boundary_pair(drop=3.0, pgas_drop=1.0, ramp_n=250, flat_n=500, insp_n=600):
    """Two hand-built neighbouring breaths where the boundary lands AFTER the deflection
    (the classic case: the fall happens under the previous breath's last expiratory
    samples, and the new inspiratory phase starts with flow already running)."""
    def phase(flow, poes, pgas, t0):
        n = len(flow)
        return {"time": (t0 + np.arange(n)) / FS, "flow": np.asarray(flow, float),
                "poes": np.asarray(poes, float), "pgas": np.asarray(pgas, float),
                "pdi": np.asarray(pgas, float) - np.asarray(poes, float)}
    # the ramp stops one step short of the first inspiratory sample, so an off-by-one at
    # t_flow would read a different pressure (and the linear start of inspiration keeps
    # falling, so t_flow+1 differs too)
    m = np.arange(1, ramp_n + 1) / (ramp_n + 1)
    exp = phase(np.full(flat_n + ramp_n, 0.01),
                np.concatenate([np.full(flat_n, -5.0), -5.0 - drop * m]),
                np.concatenate([np.full(flat_n, 9.0), 9.0 - pgas_drop * m]), 0)
    k = np.arange(insp_n) / insp_n
    insp = phase(-0.5 * np.ones(insp_n), -5.0 - drop - 6 * k,
                 8.0 + 1.5 * np.sin(np.pi * k) ** 2, len(exp["flow"]) + 1)
    return ({"has_phases": True, "expiration": exp, "inspiration": {"time": [0.0]}},
            {"has_phases": True, "expiration": exp, "inspiration": insp})


def test_boundary_after_the_deflection_adds_the_full_peepi():
    prev, this = _late_boundary_pair()
    this = _with_mechanics(this)
    notice = pressure.attach(this, prev, BCNT, VEFACTOR, _settings())
    assert notice is None
    cols = this["pressure_ext"]
    assert cols["peepi_dyn"] == pytest.approx(3.0, abs=1e-9)
    expected_thr = (3.0 - 1.0) * 0.8 * (98.0638 / 1000) * BCNT * VEFACTOR
    assert cols["wob_in_thr"] == pytest.approx(expected_thr, rel=1e-9)
    assert cols["wob_in_total_thr"] == pytest.approx(1.0 + expected_thr, rel=1e-9)
    assert cols["wobtotal_thr"] == pytest.approx(1.5 + expected_thr, rel=1e-9)
    assert cols["int_oesinsp_peepi"] == pytest.approx(2.0 + 3.0 * 1.2, abs=1e-9)
    assert cols["ptp_oesinsp_peepi"] == pytest.approx(cols["int_oesinsp_peepi"] * BCNT * VEFACTOR)
    assert cols["int_pdiinsp_peepi"] == pytest.approx(3.0 + 2.0 * 1.2, abs=1e-9)


def test_dynamic_source_uses_the_full_uncorrected_peepi_without_pgas():
    prev, this = _late_boundary_pair()
    caps = _caps(pgas=False, pdi=False, declared=frozenset({"flow", "poes"}))
    this = _with_mechanics(this)
    pressure.attach(this, prev, BCNT, VEFACTOR, _settings(caps))
    assert this["pressure_ext"]["wob_in_thr"] == pytest.approx(
        3.0 * 0.8 * (98.0638 / 1000) * BCNT * VEFACTOR, rel=1e-9)


def test_true_flow_start_ignores_flow_noise_in_the_pause():
    rng = np.random.default_rng(3)
    pause = rng.normal(0, 0.003, 400)                     # 3 mL/s of noise, sign flips freely
    insp = -np.sin(np.pi * (np.arange(1200) + 0.5) / 1200)
    flow = np.concatenate([pause, insp])
    t_flow = pressure.true_flow_start(flow)
    assert 395 <= t_flow <= 420                            # the real start, not sample 0 or a stray dip


def test_the_existing_wob_and_mechanics_are_left_alone():
    breaths = _breaths(800)
    b = _with_mechanics(breaths[2])
    before_mech, before_wob = dict(b["mechanics"]), dict(b["wob"])
    pressure.attach(b, breaths[1], BCNT, VEFACTOR, _settings())
    assert b["mechanics"] == before_mech and b["wob"] == before_wob


# --- breath #1, and other "cannot say" cases -------------------------------------------------

def test_first_breath_is_nan_with_a_notice():
    breaths = _breaths(800)
    cols, notice = _run(breaths, _settings(), 1)
    assert "no preceding breath" in notice
    assert list(cols) == list(_run(breaths, _settings(), 2)[0])       # same columns, all NaN
    assert all(np.isnan(v) for v in cols.values())


def test_a_non_adjacent_predecessor_is_refused():
    breaths = _breaths(800)
    keys = list(breaths)
    far = breaths[keys[0]]                     # breath #1 does not directly precede breath #3
    b = _with_mechanics(breaths[3])
    notice = pressure.attach(b, far, BCNT, VEFACTOR, _settings())
    assert "does not directly precede" in notice
    assert all(np.isnan(v) for v in b["pressure_ext"].values())


def test_a_non_finite_search_window_is_nan_with_a_notice():
    breaths = _breaths(800)
    breaths[1]["expiration"]["poes"][-10] = np.nan
    cols, notice = _run(breaths, _settings(), 2)
    assert "could not be located" in notice
    assert np.isnan(cols["peepi_dyn"])


# --- gastric correction --------------------------------------------------------------------

def test_the_gastric_correction_is_the_identity_when_pgas_is_constant():
    breaths = _breaths(800, pgas_drop=0.0)
    for b in breaths.values():
        b["pgas"][:] = 8.0
        b["inspiration"]["pgas"][:] = 8.0
        b["expiration"]["pgas"][:] = 8.0
    cols, _ = _run(breaths, _settings(), 2)
    assert cols["peepi_pgas_drop"] == 0.0
    assert cols["peepi_corr"] == pytest.approx(cols["peepi_dyn"], abs=1e-12)


def test_the_correction_never_goes_below_zero():
    # Pgas falls MORE than Poes does over the interval -> corrected PEEPi is clamped at 0
    cols, _ = _run(_breaths(800, pgas_drop=gd.PEEPI_DROP + 2.0), _settings(), 2)
    assert cols["peepi_corr"] == 0.0
    assert cols["peepi_dyn"] == pytest.approx(gd.PEEPI_DROP, abs=1e-9)


def test_without_pgas_the_dynamic_value_feeds_the_threshold_work():
    caps = _caps(pgas=False, pdi=False, declared=frozenset({"flow", "poes"}))
    cols, _ = _run(_breaths(800), _settings(caps), 2)
    assert "peepi_corr" not in cols and "peepi_pgas_drop" not in cols
    assert "int_pdiinsp_peepi" not in cols
    assert pressure.peepi_source(caps) == "dynamic"
    assert pressure.peepi_source(_caps()) == "corrected"


# --- thresholds ----------------------------------------------------------------------------

def test_a_deflection_below_min_deflection_is_reported_as_zero():
    cols, notice = _run(_breaths(800, drop=0.3, pgas_drop=0.0), _settings(), 2)
    assert notice is None
    assert cols["peepi_dyn"] == 0.0 and cols["peepi_lag"] == 0.0
    assert cols["int_oes_preflow"] == 0.0 and cols["wob_in_thr"] == 0.0
    assert cols["wobtotal_thr"] == pytest.approx(1.5)                    # the plain wobtotal


def test_no_deflection_at_all_is_zero_not_nan():
    cols, notice = _run(_breaths(800, drop=0.0, pgas_drop=0.0), _settings(), 2)
    assert notice is None and cols["peepi_dyn"] == 0.0


def test_min_deflection_is_configurable():
    cols, _ = _run(_breaths(800, drop=0.3, pgas_drop=0.0), _settings(min_deflection=0.1), 2)
    assert cols["peepi_dyn"] == pytest.approx(0.3, abs=1e-9)


def test_detect_peepi_onset_needs_a_usable_window():
    assert pressure.detect_peepi_onset(np.array([1.0, 2.0]), 1, 1, 0.1) is None
    assert pressure.detect_peepi_onset(np.array([1.0, np.nan, 2.0, 3.0]), 3, 1, 0.1) is None


# --- settings, units, registry -------------------------------------------------------------

def test_feature_is_off_by_default_and_validated():
    from respmech.core.settings import SettingsError
    from respmech.settingsio.toml_io import load_toml
    assert Settings().processing.pressure.peepi.enabled is False
    st = load_toml(os.path.join(HERE, "..", "golden", "scenarios", "flow_peepi_on.toml"))
    st.validate()
    st.processing.pressure.peepi.min_deflection = -1
    with pytest.raises(SettingsError, match="min_deflection"):
        st.validate()
    st.processing.pressure.peepi.min_deflection = 0.5
    st.processing.pressure.peepi.search_window_s = 0
    with pytest.raises(SettingsError, match="search_window_s"):
        st.validate()


def test_settings_round_trip_through_toml(tmp_path):
    from respmech.settingsio.toml_io import load_toml, save_toml
    st = Settings()
    st.processing.pressure.peepi.enabled = True
    st.processing.pressure.peepi.search_window_s = 0.8
    path = tmp_path / "s.toml"
    save_toml(st, str(path))
    back = load_toml(str(path))
    assert back.processing.pressure.peepi.enabled is True
    assert back.processing.pressure.peepi.search_window_s == 0.8


def test_units():
    from _helpers import assert_units
    assert_units({
        "peepi_dyn": "cmH₂O", "peepi_pgas_drop": "cmH₂O", "peepi_corr": "cmH₂O",
        "peepi_lag": "s",
        "int_oes_preflow": "cmH₂O·s", "int_oesinsp_peepi": "cmH₂O·s", "int_pdiinsp_peepi": "cmH₂O·s",
        "ptp_oes_preflow": "cmH₂O·s·min⁻¹", "ptp_oesinsp_peepi": "cmH₂O·s·min⁻¹",
        "ptp_pdiinsp_peepi": "cmH₂O·s·min⁻¹",
        "wob_in_thr": "J·min⁻¹", "wob_in_total_thr": "J·min⁻¹", "wobtotal_thr": "J·min⁻¹",
    })


def test_every_emitted_column_is_registered():
    from respmech.core.analysis.registry import REGISTRY
    registered = {s.name for s in REGISTRY if s.module == "peepi"}
    cols, _ = _run(_breaths(800), _settings(), 2)
    assert set(cols) == registered
    for name in cols:
        assert quantities.unit_for(name), name


# --- end to end ----------------------------------------------------------------------------

def _run_batch(tmp_path, *, enabled, signals=None):
    from respmech.core.pipeline import run_batch
    from respmech.settingsio.toml_io import load_toml
    st = load_toml(os.path.join(HERE, "..", "golden", "scenarios", "flow_peepi_on.toml"))
    st.input.folder = os.path.join(HERE, "..", "golden", "input")
    st.output.folder = str(tmp_path)
    st.processing.pressure.peepi.enabled = enabled
    if signals is not None:
        st.analysis.signals = signals
        st.input.channels.pgas = None
        st.input.channels.pdi = None
    return run_batch(st)


def test_pipeline_adds_the_columns_only_when_enabled(tmp_path):
    off = _run_batch(tmp_path / "off", enabled=False).ok_files["synth_peepi_A.csv"].breaths_table
    on = _run_batch(tmp_path / "on", enabled=True).ok_files["synth_peepi_A.csv"].breaths_table
    added = [c for c in on.columns if c not in off.columns]
    assert "peepi_dyn" in added and "wobtotal_thr" in added
    assert not [c for c in off.columns if c not in on.columns]
    for c in off.columns:                                                # nothing existing moved
        np.testing.assert_allclose(off[c].to_numpy(float), on[c].to_numpy(float),
                                   rtol=1e-12, equal_nan=True)
    assert np.isnan(on["peepi_dyn"].iloc[0])                             # breath #1
    np.testing.assert_allclose(on["peepi_dyn"].iloc[1:].to_numpy(float), gd.PEEPI_DROP, atol=1e-9)


def test_pipeline_gives_no_notice_for_the_first_breath_of_a_file(tmp_path):
    """Breath #1 never has a predecessor: blank by design, and a notice would repeat on
    every file."""
    fr = _run_batch(tmp_path, enabled=True).ok_files["synth_peepi_A.csv"]
    assert not [n for n in fr.notices if "PEEPi" in n]


def test_pipeline_without_poes_skips_with_a_notice(tmp_path):
    res = _run_batch(tmp_path, enabled=True, signals=["flow"])
    fr = res.ok_files["synth_peepi_A.csv"]
    assert "peepi_dyn" not in fr.breaths_table.columns
    assert any("lacks Flow or Poes" in n for n in fr.notices)


# --- measured on the built-in sample recording ---------------------------------------------
#
# The sample (core.sample) has NO PEEPi: its breaths have no end-expiratory pause, but its
# Poes carries a cardiac ripple and low-frequency wander of about 1-2 cmH2O. These are
# MEASUREMENTS of the starting thresholds against that signal, pinned so a change to the
# detector is visible -- not a claim that the values are right. They show the default
# min_deflection (0.5 cmH2O) is below the sample's own wander: three of eight breaths report a
# 1.2-2.1 cmH2O "deflection", and a wider smoothing window does not remove them (0.2 s and
# 0.4 s were also measured and report other breaths). Raising min_deflection to 2.5 reports
# zero everywhere. Whether 0.5 survives real recordings is exactly what the thresholds'
# provisional status is for (docs/beslutninger.md).

def _sample_peepi(tmp_path, **peepi):
    from respmech.core import sample
    from respmech.core.pipeline import run_batch
    desc = sample.write_sample_recording(str(tmp_path))
    st = sample.build_sample_settings(desc, str(tmp_path / "out"))
    st.processing.pressure.peepi.enabled = True
    for k, v in peepi.items():
        setattr(st.processing.pressure.peepi, k, v)
    return list(run_batch(st).ok_files.values())[0].breaths_table["peepi_dyn"].to_numpy(float)


def test_sample_recording_with_the_starting_thresholds(tmp_path):
    dyn = _sample_peepi(tmp_path)
    assert np.isnan(dyn[0])
    np.testing.assert_allclose(dyn[1:], [0, 0, 0, 1.99, 2.12, 0, 1.17, 0], atol=0.02)


def test_sample_recording_reports_zero_once_min_deflection_exceeds_its_wander(tmp_path):
    dyn = _sample_peepi(tmp_path, min_deflection=2.5)
    assert np.isnan(dyn[0])
    assert np.all(dyn[1:] == 0.0)


# --- validation, provenance ----------------------------------------------------------------

@pytest.mark.parametrize("name,value,match", [
    ("search_window_s", float("nan"), "finite"), ("smooth_s", float("inf"), "finite"),
    ("min_deflection", float("nan"), "finite"), ("onset_slope_frac", 0.0, "above 0"),
    ("onset_slope_frac", 1.5, "at most 1"),
])
def test_validate_rejects_unusable_thresholds(name, value, match):
    from respmech.core.settings import SettingsError
    from respmech.settingsio.toml_io import load_toml
    st = load_toml(os.path.join(HERE, "..", "golden", "scenarios", "flow_peepi_on.toml"))
    setattr(st.processing.pressure.peepi, name, value)
    with pytest.raises(SettingsError, match=match):
        st.validate()


def test_provenance_names_the_peepi_source():
    from respmech.core.io.writers import _provenance_rows
    from respmech.settingsio.toml_io import load_toml
    st = load_toml(os.path.join(HERE, "..", "golden", "scenarios", "flow_peepi_on.toml"))
    rows = dict(_provenance_rows(st, None).itertuples(index=False, name=None))
    assert "from corrected PEEPi" in rows["PEEPi threshold work"]
    st.processing.pressure.peepi.enabled = False
    assert "PEEPi threshold work" not in dict(_provenance_rows(st, None).itertuples(index=False, name=None))


def test_an_ignored_predecessor_is_still_used(tmp_path):
    from respmech.core.pipeline import run_batch
    from respmech.settingsio.toml_io import load_toml
    from respmech.core.settings import ExcludeEntry
    st = load_toml(os.path.join(HERE, "..", "golden", "scenarios", "flow_peepi_on.toml"))
    st.input.folder = os.path.join(HERE, "..", "golden", "input")
    st.output.folder = str(tmp_path)
    st.processing.exclude_breaths = [ExcludeEntry(file="synth_peepi_A.csv", breaths=[3])]
    t = run_batch(st).ok_files["synth_peepi_A.csv"].breaths_table
    assert list(t["breath_no"]) == [1, 2, 4, 5, 6]
    np.testing.assert_allclose(t["peepi_dyn"].to_numpy(float)[1:], gd.PEEPI_DROP, atol=1e-9)

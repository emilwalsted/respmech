"""core.analysis.mfvl (M-42): FVC/FEV1/PEF from one typed FVC breath, the MEFV
envelope it defines, and the per-tidal-breath expiratory-flow-limitation/VEcap/MVV
placement against it. Pure numpy, no file I/O -- same no-I/O style as
test_manoeuvres.py/test_lungvol.py."""
import numpy as np
import pytest

from respmech.core.analysis import mfvl as m
from respmech.core.settings import MfvlSettings, IcSettings, Settings
from respmech.core._legacy_ns import to_legacy_ns


FS = 1000.0


def _cfg(**kw):
    return MfvlSettings(**kw)


def _ns():
    """A real Settings()/legacy-namespace pair, minimally wired for attach()'s own
    input.subjects lookups (group_key needs input.format/channels; no subjects
    entries means _subject_fev1/_subject_mvv both correctly return None)."""
    st = Settings()
    st.input.format.sampling_frequency = FS
    st.input.channels.flow = 1
    st.input.channels.volume = 2
    return st, to_legacy_ns(st)


# --------------------------------------------------------------------------- #
# fvc_metrics: analytical FVC/FEV1/PEF case
# --------------------------------------------------------------------------- #

def _analytical_fvc_breath(*, v_tlc=4.0, pef=2.0):
    """A breath whose forced expiration is EXACTLY linear in volume (flow constant
    at ``pef``, ``volume(t) = v_tlc - pef*t``) -- so ``v(t) = pef*t`` exactly, the
    back-extrapolation tangent through ANY sample already IS the curve, and every
    downstream value (t0, BEV, FEV1) has a known, exact analytical answer, not
    merely a numerically-close one."""
    n = int(v_tlc / pef * FS) + 1
    t = np.arange(n) / FS
    volume = v_tlc - pef * t
    flow = np.full(n, pef)
    exp = {"time": t, "volume": volume, "flow": flow, "poes": np.zeros(n),
          "pgas": np.zeros(n), "pdi": np.zeros(n)}
    insp = {"time": np.array([0.0, 1.0]), "volume": np.array([0.0, v_tlc]),
           "flow": np.array([-1.0, -1.0]), "poes": np.zeros(2), "pgas": np.zeros(2),
           "pdi": np.zeros(2)}
    return {"number": 1, "expiration": exp, "inspiration": insp, "kind": "fvc"}


def test_analytical_fvc_fev1_pef_within_1e9():
    breath = _analytical_fvc_breath(v_tlc=4.0, pef=2.0)
    out = m.fvc_metrics(breath, tidal_breaths=[], fs=FS)
    assert out["fvc"] == pytest.approx(4.0, abs=1e-9)
    assert out["mfvl_peak_ex_flow"] == pytest.approx(2.0, abs=1e-9)
    assert out["fev1"] == pytest.approx(2.0, abs=1e-9)          # v(t0+1) = pef*1 = 2.0
    assert out["fev1_fvc"] == pytest.approx(0.5, abs=1e-9)
    assert out["fvc_bev"] == pytest.approx(0.0, abs=1e-9)       # v(t0)=pef*0=0, t0=0 exactly
    assert out["fvc_fet"] == pytest.approx(2.0, abs=1e-9)       # 4.0/2.0 - t0
    assert "BEV_HIGH" not in out["quality"]


def test_mefv_curve_is_forced_non_decreasing():
    """A tiny numerical dip in exp['volume'] must never make v() locally decrease
    (np.interp requires an increasing x)."""
    breath = {
        "inspiration": {"volume": np.array([0.0, 2.0])},
        "expiration": {"volume": np.array([2.0, 1.0, 1.05, 0.0]),   # 1.0 -> 1.05 is a dip
                       "flow": np.array([1.0, 1.0, 1.0, 1.0])},
    }
    v, _flow, v_tlc = m.mefv_curve(breath)
    assert v_tlc == 2.0
    assert np.all(np.diff(v) >= 0)


def test_fev1_clips_to_fvc_when_the_whole_manoeuvre_finishes_under_one_second():
    breath = _analytical_fvc_breath(v_tlc=1.0, pef=4.0)          # 0.25 s total
    out = m.fvc_metrics(breath, tidal_breaths=[], fs=FS)
    assert out["fev1"] == pytest.approx(out["fvc"])


def test_bev_high_flags_a_slow_ramp_up_to_pef():
    """A quadratic ramp-up to PEF (slope 0 at t=0, reaching slope=PEF exactly at
    t=t1), followed by a linear plateau at PEF -- v(t) and flow(t)=v'(t) are exact
    closed forms of each other here (unlike a hand-built breath whose flow/volume
    arrays are independently chosen), so the analytical back-extrapolation result
    (t0 = t1/2, BEV = PEF*t1/8) is known ahead of time: with t1=1.5s, PEF=3.0,
    BEV = 0.5625 L, comfortably over max(0.1, 5% of a 5.25 L FVC)."""
    pef, t1 = 3.0, 1.5
    n1 = int(t1 * FS) + 1
    t1_arr = np.arange(n1) / FS
    v1 = 0.5 * (pef / t1) * t1_arr ** 2
    flow1 = (pef / t1) * t1_arr
    n2 = int(1.0 * FS)
    t2_arr = t1 + (np.arange(1, n2 + 1) / FS)
    v2 = v1[-1] + pef * (t2_arr - t1)
    flow2 = np.full(n2, pef)
    t = np.concatenate([t1_arr, t2_arr])
    volume_from_tlc = np.concatenate([v1, v2])
    v_tlc = float(volume_from_tlc[-1])
    exp = {"time": t, "volume": v_tlc - volume_from_tlc,
          "flow": np.concatenate([flow1, flow2])}
    insp = {"volume": np.array([0.0, v_tlc]), "flow": np.array([-1.0, -1.0])}
    breath = {"number": 1, "expiration": exp, "inspiration": insp, "kind": "fvc"}
    out = m.fvc_metrics(breath, tidal_breaths=[], fs=FS)
    assert out["fvc_bev"] == pytest.approx(0.5625, abs=1e-3)
    assert "BEV_HIGH" in out["quality"]


def test_eofe_ok_true_when_fet_is_at_least_15s():
    breath = _analytical_fvc_breath(v_tlc=30.0, pef=2.0)   # 15 s exactly
    out = m.fvc_metrics(breath, tidal_breaths=[], fs=FS)
    assert out["fvc_eofe_ok"] is True


def test_eofe_ok_false_when_still_changing_fast_at_the_end():
    breath = _analytical_fvc_breath(v_tlc=4.0, pef=2.0)   # 2s, > 0.025 L change/s throughout
    out = m.fvc_metrics(breath, tidal_breaths=[], fs=FS)
    assert out["fvc_eofe_ok"] is False


def test_peak_in_flow_from_next_near_maximal_breath_only():
    fvc_breath = _analytical_fvc_breath(v_tlc=2.0, pef=2.0)
    fvc_breath["number"] = 5
    near_max = {"number": 6, "inspiration": {"volume": np.array([0.0, 1.9]),
                                            "flow": np.array([-5.0, -5.0])}}
    too_small = {"number": 6, "inspiration": {"volume": np.array([0.0, 0.5]),
                                             "flow": np.array([-5.0, -5.0])}}
    out_ok = m.fvc_metrics(fvc_breath, [near_max], fs=FS)
    assert out_ok["mfvl_peak_in_flow"] == pytest.approx(5.0)
    out_small = m.fvc_metrics(fvc_breath, [too_small], fs=FS)
    assert out_small["mfvl_peak_in_flow"] != out_small["mfvl_peak_in_flow"]   # NaN


# --------------------------------------------------------------------------- #
# apply_to_row / apply_tlc_consistency
# --------------------------------------------------------------------------- #

def test_apply_to_row_merges_and_extends_quality_without_duplicating():
    breath = _analytical_fvc_breath(v_tlc=4.0, pef=2.0)
    row = {"kind": "fvc", "quality": ["FVC_TOO_SHORT"]}
    m.apply_to_row(row, breath, tidal_breaths=[], fs=FS)
    assert row["fvc"] == pytest.approx(4.0)
    assert row["quality"] == ["FVC_TOO_SHORT"]        # BEV_HIGH did not fire here
    assert "_v_tlc" in row


def test_apply_to_row_is_a_noop_for_non_fvc_kinds():
    row = {"kind": "ic", "vol_ic": 3.0}
    m.apply_to_row(row, breath=None, tidal_breaths=[], fs=FS)
    assert row == {"kind": "ic", "vol_ic": 3.0}


def test_tlc_consistency_flags_a_mismatch_and_pops_the_internal_field():
    manoeuvres = {
        4: {"kind": "ic", "vol_ic": 3.0, "ic_eelv_pre": 0.0, "quality": []},
        7: {"kind": "fvc", "quality": [], "_v_tlc": 2.0},
    }
    m.apply_tlc_consistency(manoeuvres, tol=0.15)
    assert manoeuvres[7]["mfvl_tlc_consistency"] == pytest.approx(-1.0)
    assert "not_from_tlc" in manoeuvres[7]["quality"]
    assert "_v_tlc" not in manoeuvres[7]
    assert "_v_tlc" not in manoeuvres[4]


def test_tlc_consistency_does_not_flag_within_tolerance():
    manoeuvres = {
        4: {"kind": "ic", "vol_ic": 3.0, "ic_eelv_pre": 0.0, "quality": []},
        7: {"kind": "fvc", "quality": [], "_v_tlc": 3.1},
    }
    m.apply_tlc_consistency(manoeuvres, tol=0.15)
    assert manoeuvres[7]["mfvl_tlc_consistency"] == pytest.approx(0.1)
    assert "not_from_tlc" not in manoeuvres[7]["quality"]


def test_tlc_consistency_with_no_ic_breath_leaves_no_comparison():
    manoeuvres = {7: {"kind": "fvc", "quality": [], "_v_tlc": 3.1}}
    m.apply_tlc_consistency(manoeuvres)
    assert "mfvl_tlc_consistency" not in manoeuvres[7]
    assert "_v_tlc" not in manoeuvres[7]


# --------------------------------------------------------------------------- #
# tidal_mfvl_ext: the ticket's own EFL analytical case (triangle MEFV, IC 3 L,
# VT 1 L -> efl_pct ~ 40 +/- 0.5), VEcap closed form, coverage/no-IC handling.
# --------------------------------------------------------------------------- #

def _tidal_breath(*, vt=1.0, ex_flow=2.0, n=1000, ti_ttot=0.4, ve=10.0):
    t = np.arange(n) / FS
    exp_volume = np.linspace(vt, 0.0, n)         # EILV -> EELV
    exp = {"time": t, "volume": exp_volume, "flow": np.full(n, ex_flow)}
    insp = {"flow": np.array([-1.0, -1.0])}
    return {
        "expiration": exp, "inspiration": insp,
        "volume": np.concatenate([[0.0, vt], exp_volume]),
        "mechanics": {"vt": vt, "ti_ttot": ti_ttot, "ve": ve},
    }


def test_efl_pct_matches_the_tickets_own_triangle_case():
    """ic_op=3.0, vt=1.0 -> tidal breath sweeps v_below_tlc in [2.0, 3.0]. MEFV
    falls from 3.0 L/s at v=2.0 to 2.0 L/s at v=2.6, then stays flat at 2.0. The
    breath's own constant 2.0 L/s expiratory flow therefore reaches (>=) the
    envelope for exactly the LAST 40% of its own volume excursion (v in
    [2.6, 3.0])."""
    breath = _tidal_breath(vt=1.0, ex_flow=2.0, n=2000)
    mefv_v = np.array([2.0, 2.6, 3.0])
    mefv_flow = np.array([3.0, 2.0, 2.0])
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=3.0, pef=3.0,
        peak_in_flow=float("nan"), fev1_used=2.0, mfvl_cfg=_cfg())
    assert out["efl_pct"] == pytest.approx(40.0, abs=0.5)
    assert out["efl_present"] is True
    assert out["efl_coverage_pct"] == pytest.approx(100.0, abs=1e-6)


def test_efl_pct_zero_when_the_tidal_loop_never_touches_the_envelope():
    breath = _tidal_breath(vt=1.0, ex_flow=0.5, n=2000)    # well under the envelope everywhere
    mefv_v = np.array([2.0, 3.0])
    mefv_flow = np.array([5.0, 5.0])
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=3.0, pef=5.0,
        peak_in_flow=float("nan"), fev1_used=2.0, mfvl_cfg=_cfg())
    assert out["efl_pct"] == pytest.approx(0.0, abs=1e-6)
    assert out["efl_present"] is False


def test_ve_cap_closed_form_with_a_constant_mefv_envelope():
    """mefv is a CONSTANT 2.0 L/s across the whole range -> te_min = vt/mefv, a
    closed-form value this test checks directly, independent of the EFL test
    above's own (more complex) triangle envelope."""
    breath = _tidal_breath(vt=1.0, ex_flow=1.0, n=500, ti_ttot=0.4, ve=10.0)
    mefv_v = np.array([0.0, 5.0])
    mefv_flow = np.array([2.0, 2.0])
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=3.0, pef=2.0,
        peak_in_flow=float("nan"), fev1_used=None, mfvl_cfg=_cfg())
    assert out["te_min_mfvl"] == pytest.approx(0.5, abs=1e-6)         # 1.0 / 2.0
    assert out["ve_cap"] == pytest.approx(72.0, abs=1e-6)             # 60*1.0*(1-0.4)/0.5
    assert out["ve_pct_cap"] == pytest.approx(100.0 * 10.0 / 72.0, abs=1e-6)
    assert out["ve_reserve_pct"] == pytest.approx(100.0 * (72.0 - 10.0) / 72.0, abs=1e-6)


def test_te_min_nan_when_the_envelope_drops_below_the_floor():
    breath = _tidal_breath(vt=1.0, ex_flow=1.0, n=200)
    mefv_v = np.array([0.0, 3.0])
    mefv_flow = np.array([0.01, 0.01])       # below the 0.05 L/s floor everywhere
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=3.0, pef=0.01,
        peak_in_flow=float("nan"), fev1_used=None, mfvl_cfg=_cfg())
    assert out["te_min_mfvl"] != out["te_min_mfvl"]
    assert out["ve_cap"] != out["ve_cap"]


def test_without_ic_reference_only_the_two_peak_ratios_are_filled():
    breath = _tidal_breath(vt=1.0, ex_flow=1.5)
    out = m.tidal_mfvl_ext(
        breath, mefv_v=None, mefv_flow=None, ic_op=None, pef=3.0,
        peak_in_flow=5.0, fev1_used=2.0, mfvl_cfg=_cfg())
    assert out["max_ex_flow_pct_mfvl_peak"] == pytest.approx(50.0)   # 1.5/3.0
    for key in ("efl_pct", "efl_present", "ex_flow_pct_mfvl_max", "in_flow_pct_mfvl_max",
               "te_min_mfvl", "ve_cap", "ve_pct_cap", "ve_reserve_pct",
               "mvv_est", "ve_pct_mvv", "br_mvv_pct"):
        assert out[key] != out[key], f"{key} should be NaN without an IC reference"


def test_partial_coverage_nans_placement_columns_but_keeps_coverage_and_mvv():
    """ic_op places this breath's own operating range [1.0, 2.0] against an MEFV
    envelope whose own domain only covers [1.5, 2.0] -- half the breath's own
    range falls outside it."""
    breath = _tidal_breath(vt=1.0, ex_flow=1.0, n=2000)
    mefv_v = np.array([1.5, 2.0])
    mefv_flow = np.array([2.0, 2.0])
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=2.0, pef=2.0,
        peak_in_flow=float("nan"), fev1_used=2.0, mfvl_cfg=_cfg())
    assert 0.0 < out["efl_coverage_pct"] < 100.0
    assert out["efl_pct"] != out["efl_pct"]
    assert out["ve_cap"] != out["ve_cap"]
    assert out["mvv_est"] == out["mvv_est"]          # MVV does not need MEFV placement


def test_mvv_prefers_a_subject_override_over_the_fev1_multiplier():
    breath = _tidal_breath(vt=1.0, ex_flow=1.0, ve=20.0)
    mefv_v = np.array([0.0, 5.0])
    mefv_flow = np.array([2.0, 2.0])
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=3.0, pef=2.0,
        peak_in_flow=float("nan"), fev1_used=2.0, mvv_override=100.0, mfvl_cfg=_cfg())
    assert out["mvv_est"] == pytest.approx(100.0)
    assert out["ve_pct_mvv"] == pytest.approx(20.0)


def test_mvv_falls_back_to_fev1_times_multiplier_without_an_override():
    breath = _tidal_breath(vt=1.0, ex_flow=1.0, ve=20.0)
    mefv_v = np.array([0.0, 5.0])
    mefv_flow = np.array([2.0, 2.0])
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=3.0, pef=2.0,
        peak_in_flow=float("nan"), fev1_used=1.5, mfvl_cfg=_cfg(mvv_fev1_multiplier=40.0))
    assert out["mvv_est"] == pytest.approx(60.0)


def test_tidal_mfvl_ext_never_returns_a_fev1_source_key():
    """Self-review finding: fev1_source is a TEXT column, and joining it in via
    breath['mfvl_ext'] (build_breath_table's mechanics.mean() reduction, same
    join point as breath['wob']) broke that reduction for the WHOLE file. This
    function must never emit it; attach() writes it directly onto
    breaths_table/average_row AFTER build_breath_table has already run instead
    (see test_attach_resolves_fev1_source_spirometry_over_recorded below)."""
    breath = _tidal_breath(vt=1.0, ex_flow=1.0)
    out = m.tidal_mfvl_ext(
        breath, mefv_v=np.array([0.0, 5.0]), mefv_flow=np.array([2.0, 2.0]),
        ic_op=3.0, pef=2.0, peak_in_flow=float("nan"), fev1_used=2.0, mfvl_cfg=_cfg())
    assert "fev1_source" not in out
    out_no_ic = m.tidal_mfvl_ext(
        breath, mefv_v=None, mefv_flow=None, ic_op=None, pef=2.0,
        peak_in_flow=float("nan"), fev1_used=2.0, mfvl_cfg=_cfg())
    assert "fev1_source" not in out_no_ic


def test_attach_resolves_fev1_source_spirometry_over_recorded():
    from respmech.core.settings import SubjectEntry
    breath = _analytical_fvc_breath(v_tlc=4.0, pef=2.0)
    fr_manoeuvres = {1: {"kind": "fvc", "fvc": 4.0, "fev1": 1.5,
                        "mfvl_peak_ex_flow": 2.0, "mfvl_peak_in_flow": float("nan")}}
    breaths = {1: breath}
    tidal = [_tidal_breath(vt=1.0, ex_flow=1.0)]

    settings, s = _ns()
    settings.input.subjects.append(SubjectEntry(key="x.csv", fev1_l=3.5))
    _notice, fev1_source = m.attach(
        fr_manoeuvres=fr_manoeuvres, breaths=breaths, tidal_breaths=tidal,
        filename="x.csv", settings=settings, s=s)
    assert fev1_source == "spirometry"
    assert "fev1_source" not in tidal[0]["mfvl_ext"]     # never through this path

    # Without a subject FEV1, the derived value (from the FVC row itself) is used.
    settings2, s2 = _ns()
    tidal2 = [_tidal_breath(vt=1.0, ex_flow=1.0)]
    _notice2, fev1_source2 = m.attach(
        fr_manoeuvres=fr_manoeuvres, breaths=breaths, tidal_breaths=tidal2,
        filename="x.csv", settings=settings2, s=s2)
    assert fev1_source2 == "recorded"


# --------------------------------------------------------------------------- #
# resolve_same_file_curve: single (largest FVC) vs envelope (per-volume max)
# --------------------------------------------------------------------------- #

def test_single_source_picks_the_largest_fvc_attempt():
    small = _analytical_fvc_breath(v_tlc=2.0, pef=2.0)
    large = _analytical_fvc_breath(v_tlc=5.0, pef=2.0)
    fr_manoeuvres = {1: {"kind": "fvc", "fvc": 2.0}, 2: {"kind": "fvc", "fvc": 5.0}}
    breaths = {1: small, 2: large}
    v, _flow, v_tlc, curve_rows = m.resolve_same_file_curve(fr_manoeuvres, breaths, _cfg(source="single"))
    assert v_tlc == 5.0
    assert v[-1] == pytest.approx(5.0)
    assert curve_rows == [fr_manoeuvres[2]]


def test_envelope_source_takes_the_per_volume_maximum_across_attempts():
    a = _analytical_fvc_breath(v_tlc=3.0, pef=2.0)
    b = _analytical_fvc_breath(v_tlc=3.0, pef=4.0)     # higher flow, same fvc range
    fr_manoeuvres = {1: {"kind": "fvc", "fvc": 3.0}, 2: {"kind": "fvc", "fvc": 3.0}}
    breaths = {1: a, 2: b}
    v, flow, _v_tlc, curve_rows = m.resolve_same_file_curve(fr_manoeuvres, breaths, _cfg(source="envelope"))
    assert flow.max() == pytest.approx(4.0)
    assert len(curve_rows) == 2


def test_envelope_with_a_single_attempt_is_identical_to_single():
    a = _analytical_fvc_breath(v_tlc=3.0, pef=2.0)
    fr_manoeuvres = {1: {"kind": "fvc", "fvc": 3.0}}
    breaths = {1: a}
    v_env, flow_env, tlc_env, _rows_env = m.resolve_same_file_curve(
        fr_manoeuvres, breaths, _cfg(source="envelope"))
    v_single, flow_single, tlc_single, _rows_single = m.resolve_same_file_curve(
        fr_manoeuvres, breaths, _cfg(source="single"))
    assert tlc_env == tlc_single
    np.testing.assert_allclose(v_env, v_single)
    np.testing.assert_allclose(flow_env, flow_single)


def test_no_fvc_breath_returns_none():
    assert m.resolve_same_file_curve({1: {"kind": "ic"}}, {}, _cfg()) is None


# --------------------------------------------------------------------------- #
# resolve_same_file_ic_op
# --------------------------------------------------------------------------- #

def test_resolve_same_file_ic_op_averages_and_excludes_rejected():
    ic_cfg = IcSettings()
    fr_manoeuvres = {
        1: {"kind": "ic", "vol_ic": 3.0, "quality": []},
        2: {"kind": "ic", "vol_ic": 5.0, "quality": ["LOW_EFFORT"]},
    }
    assert m.resolve_same_file_ic_op(fr_manoeuvres, ic_cfg) == pytest.approx(3.0)


def test_resolve_same_file_ic_op_none_without_an_ic_breath():
    assert m.resolve_same_file_ic_op({1: {"kind": "fvc"}}, IcSettings()) is None


# --------------------------------------------------------------------------- #
# Self-review regression tests (found by two independent review passes)
# --------------------------------------------------------------------------- #

def test_attach_does_not_crash_on_a_half_built_fvc_row():
    """A row left behind by an upstream fvc_metrics failure (only 'kind'/'quality'
    set, no numeric fields at all) must never crash attach() via the `x == x`
    NaN idiom misreading a MISSING key's None as "not NaN"."""
    good = _analytical_fvc_breath(v_tlc=3.0, pef=2.0)
    good["number"] = 1
    fr_manoeuvres = {
        1: {"kind": "fvc", "quality": [], "fvc": 3.0, "fev1": 1.5,
           "mfvl_peak_ex_flow": 2.0, "mfvl_peak_in_flow": float("nan")},
        2: {"kind": "fvc", "quality": []},           # half-built: extraction failed
    }
    breaths = {1: good, 2: good}
    tidal = [_tidal_breath(vt=1.0, ex_flow=1.0)]

    settings, s = _ns()
    m.attach(fr_manoeuvres=fr_manoeuvres, breaths=breaths, tidal_breaths=tidal,
            filename="x.csv", settings=settings, s=s)


def test_attach_uses_the_smoothed_pef_not_the_raw_curve_maximum():
    """A single-sample spike interpolated into the composite curve must not raise
    the ratio-column denominator above the SMOOTHED PEF already on the Manoeuvres
    row (the value fvc_metrics computed and the sheet shows)."""
    breath = _analytical_fvc_breath(v_tlc=4.0, pef=2.0)
    fr_manoeuvres = {1: {"kind": "fvc", "fvc": 4.0, "mfvl_peak_ex_flow": 2.0,
                        "mfvl_peak_in_flow": float("nan")}}
    breaths = {1: breath}
    tidal = [_tidal_breath(vt=1.0, ex_flow=1.0)]

    settings, s = _ns()
    m.attach(fr_manoeuvres=fr_manoeuvres, breaths=breaths, tidal_breaths=tidal,
            filename="x.csv", settings=settings, s=s)
    # peak ex flow (1.0) against the row's own PEF (2.0), never against a raw spike.
    assert tidal[0]["mfvl_ext"]["max_ex_flow_pct_mfvl_peak"] == pytest.approx(50.0)


def test_tlc_consistency_excludes_a_reject_flagged_ic_breath():
    ic_cfg = IcSettings()      # default reject_flags = ["LOW_EFFORT"]
    manoeuvres = {
        4: {"kind": "ic", "vol_ic": 4.0, "ic_eelv_pre": 0.0, "quality": []},
        5: {"kind": "ic", "vol_ic": 2.5, "ic_eelv_pre": 0.0, "quality": ["LOW_EFFORT"]},
        7: {"kind": "fvc", "quality": [], "_v_tlc": 4.0},
    }
    m.apply_tlc_consistency(manoeuvres, ic_cfg, tol=0.15)
    assert manoeuvres[7]["mfvl_tlc_consistency"] == pytest.approx(0.0)   # only breath 4 counts
    assert "not_from_tlc" not in manoeuvres[7]["quality"]


def test_tlc_consistency_ignores_a_nan_ic_row_instead_of_poisoning_the_mean():
    ic_cfg = IcSettings()
    manoeuvres = {
        4: {"kind": "ic", "vol_ic": 3.0, "ic_eelv_pre": 0.0, "quality": []},
        5: {"kind": "ic", "vol_ic": float("nan"), "ic_eelv_pre": 0.0, "quality": []},
        7: {"kind": "fvc", "quality": [], "_v_tlc": 3.0},
    }
    m.apply_tlc_consistency(manoeuvres, ic_cfg, tol=0.15)
    assert manoeuvres[7]["mfvl_tlc_consistency"] == pytest.approx(0.0)   # not NaN


def test_efl_pct_never_exceeds_100_with_oscillating_expiratory_volume():
    """Sample-to-sample volume ripple (e.g. cardiogenic oscillation) adds PATH
    length without adding NET excursion -- efl_pct must stay bounded to a real
    percentage regardless."""
    n = 2000
    t = np.arange(n) / FS
    ramp = np.linspace(1.0, 0.0, n)
    ripple = 0.01 * np.sin(2 * np.pi * 20 * t)
    exp_volume = ramp + ripple
    breath = {
        "expiration": {"time": t, "volume": exp_volume, "flow": np.full(n, 5.0)},
        "inspiration": {"flow": np.array([-1.0, -1.0])},
        "volume": np.concatenate([[0.0, 1.0], exp_volume]),
        "mechanics": {"vt": 1.0, "ti_ttot": 0.4, "ve": 10.0},
    }
    mefv_v = np.array([-1.0, 4.0])       # wide margin: the small ripple never exits
    mefv_flow = np.array([4.0, 4.0])     # this domain, so coverage stays 100%
    out = m.tidal_mfvl_ext(
        breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=3.0, pef=4.0,
        peak_in_flow=float("nan"), fev1_used=2.0, mfvl_cfg=_cfg())
    assert out["efl_coverage_pct"] == pytest.approx(100.0)
    assert out["efl_pct"] <= 100.0 + 1e-9
    assert out["efl_pct"] == pytest.approx(100.0, abs=0.5)   # limited essentially everywhere


def test_te_min_nan_when_the_range_only_partly_exceeds_the_domain():
    """A range that is MOSTLY inside the curve's domain but pokes slightly outside
    it must NaN outright, not silently integrate over the clipped-down remainder."""
    mefv_v = np.array([1.0, 3.0])
    mefv_flow = np.array([2.0, 2.0])
    out_in_domain = m._te_min(mefv_v, mefv_flow, lo=1.0, hi=3.0)
    assert out_in_domain == out_in_domain    # a real number
    out_partly_outside = m._te_min(mefv_v, mefv_flow, lo=0.5, hi=3.0)
    assert out_partly_outside != out_partly_outside   # NaN, not a clipped number


def test_envelope_does_not_extrapolate_a_shorter_attempts_tail_flat():
    """A shorter attempt that is still flowing fast at ITS OWN end must not win
    the per-volume maximum over a volume range only the LONGER attempt reaches."""
    long_curve = (np.array([0.0, 5.0]), np.array([0.1, 0.1]), 5.0)     # near-zero flow near RV
    short_curve = (np.array([0.0, 2.0]), np.array([3.0, 3.0]), 2.0)    # stops early, still fast
    v, flow, v_tlc = m._envelope([long_curve, short_curve])
    # at v=5.0 (only the long attempt reaches there), the envelope must read the
    # LONG attempt's own low flow, never the short attempt's flat-held 3.0.
    at_5 = float(np.interp(5.0, v, flow))
    assert at_5 == pytest.approx(0.1, abs=1e-6)


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #

def test_units_of_every_new_column():
    from _helpers import assert_units
    assert_units({
        "fvc": "L", "fev1": "L", "fvc_bev": "L", "mfvl_tlc_consistency": "L",
        "fev1_fvc": "—", "fvc_fet": "s", "te_min_mfvl": "s", "fvc_eofe_ok": "",
        "efl_present": "", "fev1_source": "", "ve_cap": "L·min⁻¹", "mvv_est": "L·min⁻¹",
        "mfvl_peak_ex_flow": "L·s⁻¹", "mfvl_peak_in_flow": "L·s⁻¹",
        "efl_pct": "%", "efl_coverage_pct": "%", "ex_flow_pct_mfvl_max": "%",
        "in_flow_pct_mfvl_max": "%", "max_ex_flow_pct_mfvl_peak": "%",
        "max_in_flow_pct_mfvl_peak": "%", "ve_pct_cap": "%", "ve_reserve_pct": "%",
        "ve_pct_mvv": "%", "br_mvv_pct": "%",
    })

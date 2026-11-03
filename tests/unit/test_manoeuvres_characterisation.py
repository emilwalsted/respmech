"""Boundaries and branches of ``core.analysis.manoeuvres`` that the main manoeuvre tests
leave open (found with mutation testing): the IC volume with a non-zero end-expiratory
baseline, the Poes limb of the low-effort check, the exact EELV_UNSTABLE, FVC and
repeatability boundaries, the whole-breath sniff span, the per-channel EMG reference,
the median aggregate and the FVC suggestion's skip rules. Every expected value follows
from the documented definitions; none of these tests changes a number.
"""
from collections import OrderedDict

import numpy as np
import pytest

from respmech.core import emg as emglib
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.analysis import manoeuvres as m
from respmech.core.analysis.signals import Capabilities
from respmech.core.settings import IcSettings, Settings

from test_manoeuvres import (CAPS_FLOW_ONLY, CAPS_FULL, FS, _breath, _ic_breath,
                             _legacy_settings, _phase, _tidal_breath)

CAPS_POES = Capabilities(
    flow=True, volume=True, poes=True, pgas=False, pdi=False, emg=False, entropy=False,
    declared=frozenset({"flow", "poes"}), mode="poes_only",
)


# -- IC volume and its baseline --------------------------------------------------------

def test_vol_ic_subtracts_a_non_zero_preceding_eelv():
    ns = _legacy_settings()
    tidal = [_tidal_breath(no, eelv=1.0) for no in (1, 2, 3)]
    out = m.extract(_ic_breath(4, eelv=1.0, ic_vol=3.0), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert out["ic_eelv_pre"] == pytest.approx(1.0, abs=1e-12)
    assert out["vol_ic"] == pytest.approx(3.0, abs=1e-12)
    assert out["ic_eelv_pre_n"] == 3


def test_eelv_sd_is_computed_from_exactly_two_preceding_breaths():
    ns = _legacy_settings(ic=IcSettings(preceding_breaths=2, min_preceding_breaths=2))
    tidal = [_tidal_breath(1, eelv=0.9), _tidal_breath(2, eelv=1.1)]
    out = m.extract(_ic_breath(3, eelv=1.0), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert out["ic_eelv_pre_n"] == 2
    assert out["ic_eelv_pre_sd"] == pytest.approx(0.1, abs=1e-12)


@pytest.mark.parametrize("preceding", [0, 1])
def test_preceding_breaths_zero_or_one_take_that_many(preceding):
    ns = _legacy_settings(ic=IcSettings(preceding_breaths=preceding, min_preceding_breaths=1))
    tidal = [_tidal_breath(1, eelv=0.4), _tidal_breath(2, eelv=0.6)]
    out = m.extract(_ic_breath(3, eelv=2.0), "ic", tidal, CAPS_FLOW_ONLY, ns)
    if preceding == 0:      # no preceding breath asked for: falls back to the breath's own start
        assert out["ic_eelv_pre"] == pytest.approx(2.0, abs=1e-12)
    else:                   # the single immediately preceding breath
        assert out["ic_eelv_pre"] == pytest.approx(0.6, abs=1e-12)
    assert out["ic_eelv_pre_n"] == 1


# -- EELV_UNSTABLE ---------------------------------------------------------------------

def _flags(**kw):
    cfg = IcSettings()
    args = dict(vol_ic=3.0, ic_eelv_pre_sd=0.0, ic_eelv_pre_n=3, ic_peak_in_flow=5.0,
                ic_plateau_s=1.0, poes_ic_swing=None)
    args.update(kw)
    breath = {"number": 5}
    tidal = [{"number": 1, "inspiration": {"flow": np.array([-1.0])}},
             {"number": 9, "inspiration": {"flow": np.array([-1.0])}}]
    return m.ic_acceptance(breath, tidal, CAPS_FLOW_ONLY, cfg, **args)


@pytest.mark.parametrize("kw,flagged", [
    (dict(ic_eelv_pre_n=2, ic_eelv_pre_sd=1.0), True),       # two preceding breaths suffice
    (dict(ic_eelv_pre_n=1, ic_eelv_pre_sd=1.0), False),      # one is not a spread
    (dict(ic_eelv_pre_sd=1.0, vol_ic=0.0), False),           # no IC volume: not evaluable
    (dict(ic_eelv_pre_sd=0.2, vol_ic=0.5), True),            # 0.4 of a small IC
    (dict(ic_eelv_pre_sd=0.5, vol_ic=3.0), False),           # ratio 0.167, not the product
    (dict(ic_eelv_pre_sd=0.5, vol_ic=2.5), False),           # exactly the tolerance 0.2
])
def test_eelv_unstable_boundaries(kw, flagged):
    assert ("EELV_UNSTABLE" in _flags(**kw)) is flagged


# -- LOW_EFFORT ------------------------------------------------------------------------

def test_low_effort_returns_false_not_a_falsy_value():
    cfg = IcSettings()
    tidal = [_tidal_breath(1)]
    assert m._low_effort(5.0, None, tidal, CAPS_FLOW_ONLY, cfg) is False
    assert m._low_effort(0.1, None, tidal, CAPS_FLOW_ONLY, cfg) is True


@pytest.mark.parametrize("ic_peak,tidal_peak,flagged", [
    (0.5, 1.0, False),      # exactly half the tidal median is not "less than half"
    (0.2, 0.8, True),       # a tidal median below 1 L/s still counts
])
def test_low_effort_flow_boundaries(ic_peak, tidal_peak, flagged):
    cfg = IcSettings()
    tidal = [_tidal_breath(1, peak_flow=-tidal_peak)]
    assert m._low_effort(ic_peak, None, tidal, CAPS_FLOW_ONLY, cfg) is flagged


def test_low_effort_counts_a_one_sample_and_skips_an_empty_tidal_inspiration():
    cfg = IcSettings()
    one = {"number": 1, "inspiration": {"flow": np.array([-1.0]), "poes": np.array([0.0])}}
    empty = {"number": 2, "inspiration": {"flow": np.array([]), "poes": np.array([])}}
    assert m._low_effort(0.4, None, [one, empty], CAPS_FLOW_ONLY, cfg) is True


@pytest.mark.parametrize("ic_swing,flagged", [
    (2.49, True),           # below half the tidal median Poes swing of 5.0
    (2.5, False),           # exactly half
    (3.0, False),
])
def test_low_effort_poes_limb(ic_swing, flagged):
    ns = _legacy_settings()
    tidal = [_tidal_breath(no, poes_swing=5.0) for no in (1, 2, 3)]
    breath = _ic_breath(4, peak_flow=-2.0, poes_swing=ic_swing)
    out = m.extract(breath, "ic", tidal, CAPS_POES, ns)
    assert out["poes_ic_swing"] == pytest.approx(ic_swing, abs=1e-12)
    assert ("LOW_EFFORT" in out["quality"]) is flagged
    # the same breath without the Poes channel never looks at the swing
    out = m.extract(breath, "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "LOW_EFFORT" not in out["quality"]


def test_low_effort_poes_limb_with_a_small_tidal_swing():
    cfg = IcSettings()
    tidal = [_tidal_breath(1, poes_swing=0.8)]
    assert m._low_effort(5.0, 0.1, tidal, CAPS_POES, cfg) is True
    assert m._low_effort(5.0, 0.1, tidal, CAPS_FLOW_ONLY, cfg) is False


# -- pressures and plateau -------------------------------------------------------------

def test_pdi_ic_swing_is_measured_from_a_non_zero_baseline():
    ns = _legacy_settings()
    n = int(2.0 * FS)
    insp = _phase(n, flow=-2.0, volume_from=0.0, volume_to=3.0, pdi_from=2.0, pdi_to=30.0)
    exp = _phase(n, flow=1.0, volume_from=3.0, volume_to=0.0, pdi_from=30.0, pdi_to=2.0)
    out = m.extract(_breath(1, insp, exp, kind="ic"), "ic", [], CAPS_FULL, ns)
    assert out["pdi_ic_max"] == pytest.approx(30.0, abs=1e-12)
    assert out["pdi_ic_swing"] == pytest.approx(28.0, abs=1e-12)


def test_trailing_plateau_stops_at_a_sample_exactly_on_the_threshold():
    assert m._trailing_plateau_s(np.array([-1.0, 0.1, 0.05]), 10.0, 0.1) == pytest.approx(0.1)
    assert m._trailing_plateau_s(np.array([0.05, 0.05]), 10.0, 0.1) == pytest.approx(0.2)


# -- maximal effort --------------------------------------------------------------------

def _two_phase_pressure_breath():
    """Inspiration reaches Poes -20 / Pdi 15; the expiration of the same typed breath
    goes further (Poes -40 / Pdi 35), as a nasal sniff's arbitrary flow split can."""
    n = int(1.0 * FS)
    insp = _phase(n, flow=-1.0, volume_from=0.0, volume_to=1.0, poes_from=-2.0,
                  poes_to=-20.0, pdi_from=1.0, pdi_to=15.0)
    exp = _phase(n, flow=1.0, volume_from=1.0, volume_to=0.0, poes_from=-20.0,
                 poes_to=-40.0, pdi_from=15.0, pdi_to=35.0)
    return _breath(1, insp, exp)


def test_sniff_takes_its_swings_over_the_whole_typed_breath():
    ns = _legacy_settings()
    out = m.extract(_two_phase_pressure_breath(), "sniff", [], CAPS_FULL, ns)
    assert out["poes_max_ref"] == pytest.approx(38.0, abs=1e-12)    # -2 - (-40)
    assert out["pdi_max_ref"] == pytest.approx(34.0, abs=1e-12)     # 35 - 1


def test_max_insp_takes_its_swings_over_the_inspiration_only():
    ns = _legacy_settings()
    out = m.extract(_two_phase_pressure_breath(), "max_insp", [], CAPS_FULL, ns)
    assert out["poes_max_ref"] == pytest.approx(18.0, abs=1e-12)
    assert out["pdi_max_ref"] == pytest.approx(14.0, abs=1e-12)


def test_sniff_without_an_expiration_uses_the_inspiration():
    ns = _legacy_settings()
    breath = _two_phase_pressure_breath()
    del breath["expiration"]
    out = m.max_effort_from_breath(breath, CAPS_FULL, ns, "sniff")
    assert out["poes_max_ref"] == pytest.approx(18.0, abs=1e-12)


def test_rms_max_ref_is_reported_per_channel_and_skips_an_all_nan_channel():
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.flow = 1
    s.input.channels.volume = 2
    s.input.channels.emg = [6, 7, 8]
    ns = to_legacy_ns(s)
    labels = list(ns.input.data.columns_emg)
    assert len(labels) == 3
    breath = _ic_breath(1, kind="max_insp")
    rng = np.random.default_rng(5)
    n = len(breath["time"])
    cols = np.column_stack([rng.normal(0, 3.0, n), np.full(n, np.nan), rng.normal(0, 1.0, n)])
    breath["emgcols"] = cols
    caps = Capabilities(flow=True, volume=True, poes=False, pgas=False, pdi=False, emg=True,
                        entropy=False, declared=frozenset({"flow", "emg"}), mode="custom")
    out = m.extract(breath, "max_insp", [], caps, ns)
    rms_s = float(ns.processing.emg.rms_s)
    expected = [float(np.nanmax(emglib.rolling_rms(cols[:, ch], rms_s, FS)[0]))
                for ch in (0, 2)]
    assert out[f"rms_max_ref_col_{labels[0]}"] == pytest.approx(expected[0], rel=1e-12)
    assert f"rms_max_ref_col_{labels[1]}" not in out
    assert out[f"rms_max_ref_col_{labels[2]}"] == pytest.approx(expected[1], rel=1e-12)
    assert out["rms_max_ref"] == pytest.approx(max(expected), rel=1e-12)


# -- FVC -------------------------------------------------------------------------------

def _exp_with_times(times):
    times = np.asarray(times, dtype=float)
    return {"time": times, "flow": np.ones(times.size), "volume": np.zeros(times.size)}


@pytest.mark.parametrize("times,flags", [
    ([0.4, 1.2], ["FVC_TOO_SHORT"]),          # 0.8 s even though the end is past 1 s
    ([0.0, 0.5, 1.2], []),                    # duration from the first to the last sample
    ([0.0, 1.0], []),                         # exactly the 1 s minimum, two samples
    ([0.0, 0.99], ["FVC_TOO_SHORT"]),
    ([0.0], ["FVC_TOO_SHORT"]),
])
def test_validate_fvc_duration(times, flags):
    assert m.validate_fvc_manoeuvre({"expiration": _exp_with_times(times)}) == flags


def test_validate_fvc_without_an_expiration():
    assert m.validate_fvc_manoeuvre({}) == ["FVC_TOO_SHORT"]


def test_suggest_fvc_skip_rules_and_tie():
    def b(times, **extra):
        out = {"expiration": _exp_with_times(times) if times is not None else None}
        out.update(extra)
        return out
    breaths = OrderedDict([
        (1, b([0.0, 9.0], ignored=True)),
        (2, b([0.0, 8.0], has_phases=False)),
        (3, b(None)),
        (4, b([7.0])),
        (5, b([3.0, 5.0])),             # 2.0 s, starting at 3 s
        (6, b([10.0, 12.0])),           # also 2.0 s: the first one wins
        (7, b([0.0, 1.0], has_phases=True)),
    ])
    assert m.suggest_fvc(breaths) == 5


def test_suggest_fvc_takes_a_breath_without_a_has_phases_key():
    assert m.suggest_fvc({3: {"expiration": _exp_with_times([0.0, 0.5])}}) == 3


# -- repeatability ---------------------------------------------------------------------

def _ic_rows(vols, kind="ic"):
    return OrderedDict((i + 1, {"kind": kind, "vol_ic": v, "quality": []})
                       for i, v in enumerate(vols))


def test_two_disagreeing_ics_are_both_not_repeatable():
    rows = _ic_rows([3.0, 4.0])
    m.apply_repeatability(rows, IcSettings())
    assert [r["quality"] for r in rows.values()] == [["NOT_REPEATABLE"]] * 2


def test_median_aggregate_is_robust_to_one_outlier():
    rows = _ic_rows([3.0, 3.0, 3.1, 5.0])
    m.apply_repeatability(rows, IcSettings(aggregate="median"))
    assert [r["quality"] for r in rows.values()] == [[], [], [], ["NOT_REPEATABLE"]]
    rows = _ic_rows([3.0, 3.0, 3.1, 5.0])
    m.apply_repeatability(rows, IcSettings(aggregate="mean"))
    assert rows[1]["quality"] == ["NOT_REPEATABLE"]


def test_a_centre_of_exactly_one_is_compared_and_a_zero_centre_is_skipped():
    rows = _ic_rows([1.5, 1.0])
    m.apply_repeatability(rows, IcSettings())
    assert rows[1]["quality"] == ["NOT_REPEATABLE"]
    rows = _ic_rows([5.0, 0.0, 0.0])
    m.apply_repeatability(rows, IcSettings())
    assert rows[1]["quality"] == []            # its centre (the other two) is 0: skipped
    assert rows[2]["quality"] == ["NOT_REPEATABLE"]   # and the rows after it still run


def test_repeatability_exactly_at_the_tolerance_is_not_flagged_and_never_twice():
    rows = _ic_rows([1.25, 1.0])
    m.apply_repeatability(rows, IcSettings(repeatability_frac=0.25))
    assert [r["quality"] for r in rows.values()] == [[], []]
    rows = _ic_rows([3.0, 4.0])
    m.apply_repeatability(rows, IcSettings())
    m.apply_repeatability(rows, IcSettings())
    assert rows[1]["quality"] == ["NOT_REPEATABLE"]

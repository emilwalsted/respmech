"""core.analysis.manoeuvres (M-29): pure extraction of IC/FVC/max_insp/sniff values
from raw breath dicts — never from ``calculatemechanics`` output, which a typed breath
never reaches (M-19 unions every typed kind into ``excludebreaths`` on a flow-bearing
signal set)."""
from collections import OrderedDict

import numpy as np
import pytest

from respmech.core.analysis import manoeuvres as m
from respmech.core.analysis.signals import Capabilities
from respmech.core.settings import IcSettings, Settings
from respmech.core._legacy_ns import to_legacy_ns

FS = 200.0
CAPS_FULL = Capabilities(
    flow=True, volume=True, poes=True, pgas=True, pdi=True, emg=True, entropy=False,
    declared=frozenset({"flow", "poes", "pgas", "pdi", "emg"}), mode="full",
)
CAPS_FLOW_ONLY = Capabilities(
    flow=True, volume=True, poes=False, pgas=False, pdi=False, emg=False, entropy=False,
    declared=frozenset({"flow"}), mode="flow_only",
)


def _legacy_settings(*, ic: IcSettings | None = None):
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.flow = 1
    s.input.channels.volume = 2
    s.input.channels.poes = 3
    s.input.channels.pdi = 4
    s.input.channels.pgas = 5
    if ic is not None:
        s.processing.lung_volume.ic = ic
    return to_legacy_ns(s)


def _phase(n, *, flow=0.0, volume_from=0.0, volume_to=None, poes_from=0.0, poes_to=0.0,
          pdi_from=0.0, pdi_to=0.0, pgas_from=0.0, pgas_to=0.0, dt=1 / FS):
    volume_to = volume_from if volume_to is None else volume_to
    return {
        "time": np.arange(n) * dt,
        "flow": np.full(n, float(flow)),
        "volume": np.linspace(volume_from, volume_to, n),
        "poes": np.linspace(poes_from, poes_to, n),
        "pdi": np.linspace(pdi_from, pdi_to, n),
        "pgas": np.linspace(pgas_from, pgas_to, n),
    }


def _breath(no, insp, exp, *, kind=None, ignored=False):
    full = {}
    for key in ("time", "flow", "volume", "poes", "pdi", "pgas"):
        full[key] = np.concatenate([insp[key], exp[key]])
    return OrderedDict([
        ("number", no), ("name", f"Breath #{no}"), ("expiration", exp), ("inspiration", insp),
        ("time", full["time"]), ("flow", full["flow"]), ("volume", full["volume"]),
        ("poes", full["poes"]), ("pgas", full["pgas"]), ("pdi", full["pdi"]),
        ("breathcnt", no), ("ignored", ignored), ("kind", kind), ("has_phases", True),
        ("entcols", []), ("emgcols", []), ("filename", "rec.csv"),
    ])


def _tidal_breath(no, eelv=0.0, tidal_vol=0.5, poes_swing=5.0, peak_flow=-1.0):
    insp = _phase(int(1.0 * FS), flow=peak_flow, volume_from=eelv, volume_to=eelv + tidal_vol,
                 poes_from=0.0, poes_to=-poes_swing)
    exp = _phase(int(1.0 * FS), flow=1.0, volume_from=eelv + tidal_vol, volume_to=eelv,
                poes_from=-poes_swing, poes_to=0.0)
    return _breath(no, insp, exp, kind=None, ignored=False)


def _ic_breath(no, *, eelv=0.0, ic_vol=3.0, peak_flow=-2.0, plateau_s=0.0,
              poes_swing=None, pdi_swing=None, kind="ic"):
    n = int(2.0 * FS)
    insp = _phase(n, flow=peak_flow, volume_from=eelv, volume_to=eelv + ic_vol,
                 poes_from=0.0, poes_to=(-poes_swing if poes_swing is not None else 0.0),
                 pdi_from=0.0, pdi_to=(pdi_swing if pdi_swing is not None else 0.0))
    if plateau_s > 0:
        pn = int(plateau_s * FS)
        plateau = _phase(pn, flow=0.0, volume_from=eelv + ic_vol, volume_to=eelv + ic_vol,
                         poes_from=insp["poes"][-1], poes_to=insp["poes"][-1],
                         pdi_from=insp["pdi"][-1], pdi_to=insp["pdi"][-1])
        for key in insp:
            insp[key] = np.concatenate([insp[key], plateau[key]])
    exp = _phase(int(1.5 * FS), flow=1.0, volume_from=eelv + ic_vol, volume_to=eelv,
                poes_from=insp["poes"][-1], poes_to=0.0, pdi_from=insp["pdi"][-1], pdi_to=0.0)
    return _breath(no, insp, exp, kind=kind)


# --------------------------------------------------------------------------------- #
# Analytical IC case
# --------------------------------------------------------------------------------- #

def test_analytical_ic_from_eelv_zero_to_three_litres():
    ns = _legacy_settings()
    tidal = [_tidal_breath(i) for i in (1, 2, 3)]
    breath = _ic_breath(4, eelv=0.0, ic_vol=3.0, poes_swing=15.0)
    out = m.extract(breath, "ic", tidal, ns.capabilities, ns)
    assert out["vol_ic"] == pytest.approx(3.0, abs=1e-9)
    assert out["poes_ic_swing"] == pytest.approx(15.0, abs=1e-9)


def test_ic_eelv_pre_averages_up_to_preceding_breaths_tidal_ones():
    ns = _legacy_settings()
    tidal = [_tidal_breath(1, eelv=0.0), _tidal_breath(2, eelv=0.2), _tidal_breath(3, eelv=0.1)]
    breath = _ic_breath(4, eelv=0.1, ic_vol=3.0)
    out = m.extract(breath, "ic", tidal, ns.capabilities, ns)
    assert out["ic_eelv_pre_n"] == 3
    assert out["ic_eelv_pre"] == pytest.approx((0.0 + 0.2 + 0.1) / 3, abs=1e-9)
    assert out["ic_eelv_pre_sd"] == pytest.approx(np.std([0.0, 0.2, 0.1]), abs=1e-9)


def test_ic_eelv_pre_falls_back_to_own_insp_start_with_too_few_preceding_breaths():
    ns = _legacy_settings(ic=IcSettings(min_preceding_breaths=2))
    tidal = [_tidal_breath(1, eelv=0.05)]           # only one -> below min_preceding_breaths
    breath = _ic_breath(2, eelv=0.05, ic_vol=3.0)
    out = m.extract(breath, "ic", tidal, ns.capabilities, ns)
    assert out["ic_eelv_pre_n"] == 1
    assert out["ic_eelv_pre_sd"] == 0.0
    assert out["ic_eelv_pre"] == pytest.approx(0.05, abs=1e-9)   # this breath's own insp[0]


def test_ic_eelv_pre_only_uses_breaths_that_precede_this_one_in_time():
    """A tidal breath NUMBERED after the manoeuvre must never be averaged in."""
    ns = _legacy_settings()
    before = [_tidal_breath(1, eelv=0.0), _tidal_breath(2, eelv=0.0)]
    after = [_tidal_breath(6, eelv=999.0)]           # would wreck the mean if included
    breath = _ic_breath(4, eelv=0.0, ic_vol=3.0)
    out = m.extract(breath, "ic", before + after, ns.capabilities, ns)
    assert out["ic_eelv_pre"] == pytest.approx(0.0, abs=1e-9)
    assert out["ic_eelv_pre_n"] == 2


# --------------------------------------------------------------------------------- #
# Channel-conditioned columns
# --------------------------------------------------------------------------------- #

def test_ic_fields_are_channel_conditioned():
    ns = _legacy_settings()
    tidal = [_tidal_breath(i) for i in (1, 2, 3)]
    breath = _ic_breath(4, poes_swing=10.0, pdi_swing=8.0)

    full = m.extract(breath, "ic", tidal, CAPS_FULL, ns)
    for key in ("poes_ic_min", "poes_ic_eelv", "poes_ic_swing", "poes_ic_peakvol",
               "pdi_ic_max", "pdi_ic_swing", "pgas_ic_peakvol"):
        assert key in full

    flow_only = m.extract(breath, "ic", tidal, CAPS_FLOW_ONLY, ns)
    for key in ("poes_ic_min", "poes_ic_eelv", "poes_ic_swing", "poes_ic_peakvol",
               "pdi_ic_max", "pdi_ic_swing", "pgas_ic_peakvol"):
        assert key not in flow_only
    # the channel-independent core is unaffected either way
    assert flow_only["vol_ic"] == pytest.approx(full["vol_ic"], abs=1e-9)


# --------------------------------------------------------------------------------- #
# Quality flags
# --------------------------------------------------------------------------------- #

def test_boundary_flags_first_and_last_breath_only():
    ns = _legacy_settings()
    tidal = [_tidal_breath(i) for i in (1, 2, 3, 5, 6)]
    middle = m.extract(_ic_breath(4), "ic", tidal, ns.capabilities, ns)
    assert "BOUNDARY" not in middle["quality"]

    first = m.extract(_ic_breath(0), "ic", tidal, ns.capabilities, ns)
    assert "BOUNDARY" in first["quality"]

    last = m.extract(_ic_breath(7), "ic", tidal, ns.capabilities, ns)
    assert "BOUNDARY" in last["quality"]


def test_low_effort_flags_a_much_smaller_peak_flow_than_the_tidal_median():
    ns = _legacy_settings(ic=IcSettings(low_effort_frac=0.5))
    tidal = [_tidal_breath(i, peak_flow=-2.0) for i in (1, 2, 3, 5, 6)]
    weak = m.extract(_ic_breath(4, peak_flow=-0.5), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "LOW_EFFORT" in weak["quality"]

    strong = m.extract(_ic_breath(4, peak_flow=-5.0), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "LOW_EFFORT" not in strong["quality"]


def test_low_effort_is_not_evaluable_with_no_tidal_breaths_to_compare_against():
    ns = _legacy_settings()
    weak = m.extract(_ic_breath(1, peak_flow=-0.01), "ic", [], CAPS_FLOW_ONLY, ns)
    assert "LOW_EFFORT" not in weak["quality"]


def test_extract_is_pure_over_raw_breaths():
    """extract() never reads `breath['mechanics']`/`breath['wob']` (calculatemechanics'
    own output) — it must work identically whether or not those keys even exist, since
    a typed breath never reaches calculatemechanics in the real pipeline."""
    ns = _legacy_settings()
    tidal = [_tidal_breath(i) for i in (1, 2, 3)]
    breath = _ic_breath(4, ic_vol=1.234)
    assert "mechanics" not in breath and "wob" not in breath   # the fixture never sets them

    out_a = m.extract(breath, "ic", tidal, ns.capabilities, ns)
    out_b = m.extract(breath, "ic", tidal, ns.capabilities, ns)
    assert out_a == out_b                        # deterministic, no hidden state
    assert out_a["vol_ic"] == pytest.approx(1.234, abs=1e-9)

    breath["mechanics"] = {"poison": True}        # would only matter if extract() read it
    breath["wob"] = {"poison": True}
    out_c = m.extract(breath, "ic", tidal, ns.capabilities, ns)
    assert out_c["vol_ic"] == out_a["vol_ic"]


def test_extract_is_deterministic_given_the_same_tidal_breaths():
    """`extract` is a pure function of exactly its own five arguments -- calling it
    twice with the SAME (file-scoped) tidal-breath list gives the identical result,
    regardless of anything else. This is the unit-level half of the acceptance
    criterion "LOW_EFFORT is identical between a subset run and a full run"; the
    other half -- that `core.pipeline.run_batch` actually hands `extract` the SAME
    tidal breaths for a given file whether or not other files are also in the batch
    -- is exercised end to end in
    test_core_outputs.py::test_low_effort_flag_is_identical_between_a_subset_batch_and_a_full_batch."""
    ns = _legacy_settings()
    tidal = [_tidal_breath(i, peak_flow=-2.0) for i in (1, 2, 3)]
    breath = _ic_breath(4, peak_flow=-0.3)
    first = m.extract(breath, "ic", tidal, CAPS_FLOW_ONLY, ns)
    second = m.extract(breath, "ic", list(tidal), CAPS_FLOW_ONLY, ns)
    assert first["quality"] == second["quality"]


def test_no_plateau_flags_a_plateau_shorter_than_the_configured_minimum():
    ns = _legacy_settings(ic=IcSettings(min_plateau_s=0.3, plateau_flow_lps=0.05))
    tidal = [_tidal_breath(i) for i in (1, 2, 3)]
    short = m.extract(_ic_breath(4, plateau_s=0.1), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "NO_PLATEAU" in short["quality"]

    long_enough = m.extract(_ic_breath(4, plateau_s=0.5), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "NO_PLATEAU" not in long_enough["quality"]


def test_no_plateau_never_fires_with_the_default_zero_minimum():
    ns = _legacy_settings()                        # default min_plateau_s = 0.0
    tidal = [_tidal_breath(i) for i in (1, 2, 3)]
    out = m.extract(_ic_breath(4, plateau_s=0.0), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "NO_PLATEAU" not in out["quality"]


def test_eelv_unstable_flags_high_variance_preceding_eelvs():
    ns = _legacy_settings(ic=IcSettings(eelv_tolerance_frac=0.1))
    tidal = [_tidal_breath(1, eelv=0.5), _tidal_breath(2, eelv=1.5), _tidal_breath(3, eelv=0.5)]
    out = m.extract(_ic_breath(4, eelv=0.8), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "EELV_UNSTABLE" in out["quality"]


def test_eelv_unstable_does_not_fire_with_low_variance():
    ns = _legacy_settings(ic=IcSettings(eelv_tolerance_frac=0.2))
    tidal = [_tidal_breath(1, eelv=1.0), _tidal_breath(2, eelv=1.01), _tidal_breath(3, eelv=0.99)]
    out = m.extract(_ic_breath(4, eelv=1.0), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "EELV_UNSTABLE" not in out["quality"]


def test_eelv_unstable_is_scaled_by_vol_ic_not_by_a_near_zero_eelv_baseline():
    """Self-review finding: with the pipeline's default zero-referenced + drift-
    corrected volume signal, a real ic_eelv_pre routinely sits within a few mL of
    0 L. A relative tolerance measured against ic_eelv_pre ITSELF would explode on
    ordinary millimetre-scale breath-to-breath noise; scaled by vol_ic instead, the
    same noise is negligible against a real ~3 L manoeuvre and must NOT fire."""
    ns = _legacy_settings(ic=IcSettings(eelv_tolerance_frac=0.2))
    tidal = [_tidal_breath(1, eelv=0.001), _tidal_breath(2, eelv=-0.002),
             _tidal_breath(3, eelv=0.0015)]
    out = m.extract(_ic_breath(4, eelv=0.0, ic_vol=3.0), "ic", tidal, CAPS_FLOW_ONLY, ns)
    assert "EELV_UNSTABLE" not in out["quality"]


# --------------------------------------------------------------------------------- #
# Repeatability (post-processing pass, apply_repeatability)
# --------------------------------------------------------------------------------- #

def test_apply_repeatability_flags_an_outlier_among_repeat_ic_attempts():
    ic_cfg = IcSettings(repeatability_frac=0.10)
    manoeuvres = {
        1: {"kind": "ic", "vol_ic": 3.0, "quality": []},
        2: {"kind": "ic", "vol_ic": 3.05, "quality": []},
        3: {"kind": "ic", "vol_ic": 1.0, "quality": []},   # way off
    }
    m.apply_repeatability(manoeuvres, ic_cfg)
    assert "NOT_REPEATABLE" in manoeuvres[3]["quality"]


def test_apply_repeatability_does_not_flag_a_single_ic_with_no_sibling():
    ic_cfg = IcSettings()
    manoeuvres = {1: {"kind": "ic", "vol_ic": 3.0, "quality": []}}
    m.apply_repeatability(manoeuvres, ic_cfg)
    assert manoeuvres[1]["quality"] == []


def test_apply_repeatability_excludes_reject_flagged_entries_from_the_comparison_group():
    ic_cfg = IcSettings(repeatability_frac=0.10, reject_flags=["LOW_EFFORT"])
    manoeuvres = {
        1: {"kind": "ic", "vol_ic": 3.0, "quality": []},
        2: {"kind": "ic", "vol_ic": 3.05, "quality": []},
        3: {"kind": "ic", "vol_ic": 0.1, "quality": ["LOW_EFFORT"]},   # excluded from the group
    }
    m.apply_repeatability(manoeuvres, ic_cfg)
    assert manoeuvres[1]["quality"] == []
    assert manoeuvres[2]["quality"] == []
    assert "NOT_REPEATABLE" not in manoeuvres[3]["quality"]           # never touched at all


def test_apply_repeatability_ignores_non_ic_entries():
    ic_cfg = IcSettings(repeatability_frac=0.10)
    manoeuvres = {
        1: {"kind": "ic", "vol_ic": 3.0, "quality": []},
        2: {"kind": "fvc", "quality": []},
    }
    m.apply_repeatability(manoeuvres, ic_cfg)
    assert manoeuvres[2]["quality"] == []


# --------------------------------------------------------------------------------- #
# Measured threshold cases (M-41) — PLACEHOLDERS, not yet calibrated
# --------------------------------------------------------------------------------- #
#
# IcSettings' †-marked fields (eelv_tolerance_frac, repeatability_frac, among others)
# are the plan's STARTING values, not measured ones — see IcSettings' own docstring
# and docs/beslutninger.md: a threshold picked without a real recording in hand is a
# guess, not a fact (the K-035 lesson). This sandbox has no production IC recording to
# calibrate against, so the two cases below are PLACEHOLDERS: they exist so the shape
# — real numbers in, an assertion on the resulting flag out — is already written and
# reviewed, but they must never be trusted (or unskipped) without first being replaced
# by numbers actually read off a real recording.
#
# To fill these in:
#   1. Get RIU_H5_IC.txt locally (see tests/golden/README.md's typed_ic_h5 production
#      scenario) and run it through `respmech breaths` (or core.pipeline.segment_file
#      directly) plus a few of the same participant's tidal RIU_H5_*W.txt recordings.
#   2. Read off the ACTUAL preceding-EELV values / repeat-IC volumes (litres) for a
#      case that should, and a case that should not, flag — replace the PLACEHOLDER_*
#      rows below with those measured numbers (never invented ones).
#   3. Remove the `pytest.mark.skip` line once the case is confirmed against the real
#      recording, and record the calibrated threshold in docs/beslutninger.md (M-41's
#      own ticket names this as the exact deliverable to Emil).
#
# EELV_UNSTABLE cases: (case id, preceding EELVs [L], manoeuvre EELV [L], ic_vol [L],
# eelv_tolerance_frac, expect the flag to fire). The PLACEHOLDER values below are
# mechanically self-consistent (verified to actually fire/not-fire as claimed against
# the real formula) so the scaffold itself is proven correct — but they are still
# INVENTED, not measured, and must be replaced before this stops being skipped.
EELV_UNSTABLE_MEASURED_CASES = [
    pytest.param([0.0, 0.0, 0.0], 0.0, 3.0, 0.2, False, id="PLACEHOLDER_stable"),
    pytest.param([0.0, 0.0, 2.0], 0.0, 3.0, 0.2, True, id="PLACEHOLDER_unstable"),
]


@pytest.mark.skip(reason="måles lokalt — RIU_H5_IC.txt, se tests/golden/README.md")
@pytest.mark.parametrize(
    "preceding_eelvs,manoeuvre_eelv,ic_vol,tolerance,expect_flag",
    EELV_UNSTABLE_MEASURED_CASES)
def test_eelv_unstable_measured_case(preceding_eelvs, manoeuvre_eelv, ic_vol, tolerance,
                                     expect_flag):
    """Pins EELV_UNSTABLE against a REAL preceding-EELV spread measured on a real
    typed-IC recording (K-035: a threshold pinned without a real recording is a
    guess). See this section's own header comment for how to fill this in."""
    ns = _legacy_settings(ic=IcSettings(eelv_tolerance_frac=tolerance))
    tidal = [_tidal_breath(i, eelv=v) for i, v in enumerate(preceding_eelvs, start=1)]
    breath = _ic_breath(len(tidal) + 1, eelv=manoeuvre_eelv, ic_vol=ic_vol)
    out = m.extract(breath, "ic", tidal, CAPS_FLOW_ONLY, ns)
    if expect_flag:
        assert "EELV_UNSTABLE" in out["quality"]
    else:
        assert "EELV_UNSTABLE" not in out["quality"]


# NOT_REPEATABLE cases: (case id, repeat vol_ic values [L] in breath-number order,
# repeatability_frac, 1-based index of the expected outlier (None if none), expect the
# flag to fire on that outlier).
NOT_REPEATABLE_MEASURED_CASES = [
    pytest.param([3.0, 3.0, 3.0], 0.10, None, False, id="PLACEHOLDER_repeatable"),
    pytest.param([3.0, 3.0, 1.0], 0.10, 3, True, id="PLACEHOLDER_outlier"),
]


@pytest.mark.skip(reason="måles lokalt — RIU_H5_IC.txt, se tests/golden/README.md")
@pytest.mark.parametrize(
    "vol_ics,repeatability_frac,outlier_no,expect_flag",
    NOT_REPEATABLE_MEASURED_CASES)
def test_not_repeatable_measured_case(vol_ics, repeatability_frac, outlier_no, expect_flag):
    """Pins NOT_REPEATABLE against REAL repeat-IC volumes measured on a real typed-IC
    recording (K-035: a threshold pinned without a real recording is a guess). See
    this section's own header comment for how to fill this in."""
    ic_cfg = IcSettings(repeatability_frac=repeatability_frac)
    manoeuvres = {i: {"kind": "ic", "vol_ic": v, "quality": []}
                  for i, v in enumerate(vol_ics, start=1)}
    m.apply_repeatability(manoeuvres, ic_cfg)
    if expect_flag:
        assert "NOT_REPEATABLE" in manoeuvres[outlier_no]["quality"]
    else:
        assert all("NOT_REPEATABLE" not in e["quality"] for e in manoeuvres.values())


# --------------------------------------------------------------------------------- #
# max_insp / sniff
# --------------------------------------------------------------------------------- #

def test_max_effort_extracts_poes_and_pdi_references():
    ns = _legacy_settings()
    breath = _ic_breath(1, poes_swing=42.0, pdi_swing=30.0, kind="max_insp")
    out = m.extract(breath, "max_insp", [], CAPS_FULL, ns)
    assert out["kind"] == "max_insp"
    assert out["poes_max_ref"] == pytest.approx(42.0, abs=1e-9)
    assert out["pdi_max_ref"] == pytest.approx(30.0, abs=1e-9)
    assert out["quality"] == []


def test_max_effort_refs_are_swings_from_the_breaths_own_baseline_not_absolute():
    """Self-review finding: poes_max_ref/pdi_max_ref must be on the SAME
    baseline-subtracted footing as poes_ic_swing/pdi_ic_swing, not the raw absolute
    Poes/Pdi value -- a resting Poes baseline around -6 cmH2O (a realistic balloon
    zero-offset) would otherwise inflate the reference by that offset."""
    ns = _legacy_settings()
    n = int(1.0 * FS)
    insp = _phase(n, flow=-2.0, volume_from=0.0, volume_to=2.0,
                 poes_from=-6.0, poes_to=-50.0, pdi_from=2.0, pdi_to=25.0)
    exp = _phase(n, flow=1.0, volume_from=2.0, volume_to=0.0,
                poes_from=-50.0, poes_to=-6.0, pdi_from=25.0, pdi_to=2.0)
    breath = _breath(1, insp, exp, kind="max_insp")
    out = m.extract(breath, "max_insp", [], CAPS_FULL, ns)
    assert out["poes_max_ref"] == pytest.approx(44.0, abs=1e-9)   # -6.0 - (-50.0), not 50.0
    assert out["pdi_max_ref"] == pytest.approx(23.0, abs=1e-9)    # 25.0 - 2.0, not 25.0


def test_sniff_omits_pressure_refs_without_the_channel():
    ns = _legacy_settings()
    breath = _ic_breath(1, poes_swing=20.0, kind="sniff")
    out = m.extract(breath, "sniff", [], CAPS_FLOW_ONLY, ns)
    assert "poes_max_ref" not in out
    assert "pdi_max_ref" not in out


def test_max_effort_rms_max_ref_from_emg_columns():
    ns = _legacy_settings()
    n = int(1.0 * FS)
    breath = _ic_breath(1, kind="max_insp")
    rng = np.random.default_rng(0)
    breath["emgcols"] = rng.normal(0, 1, size=(len(breath["time"]), 2))
    caps = Capabilities(flow=True, volume=True, poes=False, pgas=False, pdi=False, emg=True,
                        entropy=False, declared=frozenset({"flow", "emg"}), mode="custom")
    out = m.extract(breath, "max_insp", [], caps, ns)
    assert "rms_max_ref" in out
    assert out["rms_max_ref"] > 0


# --------------------------------------------------------------------------------- #
# fvc / ic_fvc / other
# --------------------------------------------------------------------------------- #

def test_pure_fvc_reports_no_numeric_fields_only_quality():
    ns = _legacy_settings()
    breath = _ic_breath(1, kind="fvc")
    out = m.extract(breath, "fvc", [], CAPS_FULL, ns)
    assert out == {"kind": "fvc", "quality": []}


def test_fvc_too_short_is_flagged():
    ns = _legacy_settings()
    breath = _ic_breath(1, kind="fvc")
    breath["expiration"]["time"] = np.array([0.0, 0.01])     # 10 ms exp -- not a forced expiration
    out = m.extract(breath, "fvc", [], CAPS_FULL, ns)
    assert out["quality"] == ["FVC_TOO_SHORT"]


def test_ic_fvc_gets_the_full_ic_field_set_plus_fvc_quality():
    ns = _legacy_settings()
    tidal = [_tidal_breath(i) for i in (1, 2, 3)]
    breath = _ic_breath(4, ic_vol=2.0, kind="ic_fvc")
    breath["expiration"]["time"] = np.array([0.0, 0.01])     # too-short exp -> FVC_TOO_SHORT
    out = m.extract(breath, "ic_fvc", tidal, CAPS_FLOW_ONLY, ns)
    assert out["vol_ic"] == pytest.approx(2.0, abs=1e-9)
    assert "FVC_TOO_SHORT" in out["quality"]


def test_other_kind_reports_only_kind_and_empty_quality():
    ns = _legacy_settings()
    breath = _ic_breath(1, kind="other")
    out = m.extract(breath, "other", [], CAPS_FULL, ns)
    assert out == {"kind": "other", "quality": []}


# --------------------------------------------------------------------------------- #
# suggest_fvc / validate_fvc_manoeuvre (pure UI helpers, no pipeline call site yet)
# --------------------------------------------------------------------------------- #

def test_suggest_fvc_picks_the_longest_expiration_among_untyped_breaths():
    short = _tidal_breath(1)
    long_exp = _breath(2, _phase(50, flow=-1.0, volume_from=0.0, volume_to=0.5),
                       _phase(400, flow=1.0, volume_from=0.5, volume_to=0.0))
    typed_and_ignored = _ic_breath(3, kind="ic", ic_vol=5.0)
    typed_and_ignored["ignored"] = True
    breaths = {1: short, 2: long_exp, 3: typed_and_ignored}
    assert m.suggest_fvc(breaths) == 2


def test_suggest_fvc_returns_none_with_no_eligible_breath():
    typed_and_ignored = _ic_breath(1, kind="ic")
    typed_and_ignored["ignored"] = True
    assert m.suggest_fvc({1: typed_and_ignored}) is None


def test_validate_fvc_manoeuvre_accepts_a_long_enough_expiration():
    breath = _ic_breath(1, kind="fvc")
    assert m.validate_fvc_manoeuvre(breath) == []

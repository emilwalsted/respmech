"""Characterisation of ``core.compute.calculatemechanics`` and the gated peak EMG it attaches.

The golden suite pins the mechanics table, but not the extra fields a breath dict carries
for the plots, the loop CSV and the preview (``eilv``, ``eelv`` and their averaged
counterparts, ``volumeavg``/``poesavg``), and no golden scenario switches the
cardiac-gated peak EMG on with real R-peaks. These tests pin both on small hand-built
breaths whose values make every comparison in ``calculatemechanics`` decidable: the
mid-volume samples sit exactly on the mid-volume level, the oesophageal pressure swing
between end-inspiration and end-expiration is exactly 1, and the R-peaks land at known
times on an absolute clock that does not start at 0.

The expected gated values are computed by calling ``emg.gated_peak_rms`` directly with
the phase-local peak positions written out from the docstring of ``_add_gated_peaks``,
so the tests check the wiring, not the gating algorithm (``test_robust_peak.py`` and
``test_emg_characterisation.py`` cover that).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from respmech.core import compute
from respmech.core import emg as emglib
from respmech.core._cancel import Cancelled
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.settings import Settings


FS = 1000
T0 = 10.0                      # absolute start of the breath, deliberately not 0
NO_PEAKS_REASON = "no R-peaks available — is processing.emg.remove_ecg on?"


def _ns(*, emg=False, robust=False, **mech):
    channels = {"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9}
    if emg:
        channels["emg"] = [2, 3]
    d = {"input": {"channels": channels, "format": {"sampling_frequency": FS}}}
    if robust:
        d["processing"] = {"emg": {"robust_peak": {"enabled": True}}}
    ns = to_legacy_ns(Settings.from_dict(d))
    for k, v in mech.items():
        setattr(ns.processing.mechanics, k, v)
    return ns


def _phase(n, start, *, vol, flow, poes, pgas, pdi, emgcols=None):
    d = {"time": start + np.arange(n) / FS, "volume": np.asarray(vol, float),
         "flow": np.asarray(flow, float), "poes": np.asarray(poes, float),
         "pgas": np.asarray(pgas, float), "pdi": np.asarray(pdi, float)}
    if emgcols is not None:
        d["emgcols"] = emgcols
    return d


def _small_breath():
    """Five samples per phase, every value distinct so an index mutation shows."""
    insp = _phase(5, T0, vol=np.linspace(0.0, 1.0, 5), flow=[-0.1, -0.3, -0.7, -0.4, -0.2],
                  poes=[-5.0, -5.5, -6.5, -5.8, -4.0], pgas=[2.0, 2.2, 2.6, 2.4, 2.5],
                  pdi=[7.0, 7.7, 9.1, 8.2, 6.5])
    exp = _phase(5, T0 + 0.005, vol=np.linspace(1.0, 0.0, 5), flow=[0.15, 0.35, 0.65, 0.45, 0.05],
                 poes=[-4.2, -4.6, -4.9, -5.3, -5.0], pgas=[2.4, 2.1, 1.9, 2.3, 2.0],
                 pdi=[6.6, 6.7, 6.8, 7.6, 7.0])
    breath = {"inspiration": insp, "expiration": exp, "emgcols": [], "entcols": []}
    for k in ("time", "volume", "flow", "poes", "pgas", "pdi"):
        breath[k] = np.concatenate([insp[k], exp[k]])
    return breath


AVG_VIN = np.array([0.0, 0.3, 0.6, 0.9, 1.2])
AVG_VEX = np.array([1.1, 0.8, 0.5, 0.2, -0.1])
AVG_PIN = np.array([-5.1, -5.4, -6.0, -5.7, -4.3])
AVG_PEX = np.array([-4.4, -4.7, -4.8, -5.2, -5.05])


def _mech(breath, ns, **kw):
    return compute.calculatemechanics(breath, 12, 2.5, AVG_VIN, AVG_VEX, AVG_PIN, AVG_PEX, ns, **kw)


# --- the breath dict fields the plots, the loop CSV and the preview read --------------

def test_end_inspiratory_and_end_expiratory_points():
    b = _mech(_small_breath(), _ns())
    assert b["eilv"] == [1.0, -4.0]
    assert b["eelv"] == [0.0, -5.0]
    assert b["eilvavg"] == [1.2, -4.3]
    assert b["eelvavg"] == [-0.1, -5.05]


def test_average_breath_is_attached_to_the_breath_and_its_phases():
    b = _mech(_small_breath(), _ns())
    np.testing.assert_array_equal(b["volumeavg"], np.concatenate([AVG_VIN, AVG_VEX]))
    np.testing.assert_array_equal(b["poesavg"], np.concatenate([AVG_PIN, AVG_PEX]))
    assert b["inspiration"]["volumeavg"] is AVG_VIN
    assert b["expiration"]["volumeavg"] is AVG_VEX
    assert b["inspiration"]["poesavg"] is AVG_PIN
    assert b["expiration"]["poesavg"] is AVG_PEX


def test_mid_volume_index_takes_the_sample_exactly_on_the_level():
    """Both phases run 0, 0.25, 0.5, 0.75, 1 (exactly representable), so the mid-volume
    level 0.5 is hit exactly at index 2; the documented ``<=``/``>=`` take that sample."""
    m = _mech(_small_breath(), _ns())["mechanics"]
    assert m["flow_midvolinsp"] == 0.7
    assert m["flow_midvolexp"] == -0.65
    assert m["in_flow_midvol"] == 0.7
    assert m["ex_flow_midvol"] == 0.65
    assert m["poes_midvolinsp"] == -6.5
    assert m["poes_midvolexp"] == -4.9


def test_vmr_with_an_oesophageal_swing_of_exactly_one():
    """End-inspiratory minus end-expiratory Poes is -4 - (-5) = 1; only a zero
    denominator is replaced by 0, so VMR is the Pgas swing itself."""
    m = _mech(_small_breath(), _ns())["mechanics"]
    assert m["vmr"] == pytest.approx(0.5, abs=1e-12)


def test_ptp_baseline_window_shorter_than_one_sample_uses_one_sample():
    b = _small_breath()
    m = _mech(b, _ns(ptp_baseline_window_s=0.0001))["mechanics"]
    ptp, integ = compute.calcptp(-b["inspiration"]["poes"], 12, 2.5, FS, 1)
    assert m["ptp_oesinsp"] == ptp
    assert m["int_oesinsp"] == integ
    ptp2, _ = compute.calcptp(-b["inspiration"]["poes"], 12, 2.5, FS, 2)
    assert ptp2 != ptp                       # the window really matters on this breath


def test_settings_without_capabilities_default_to_the_full_channel_set():
    ns = _ns()
    bare = SimpleNamespace(**{k: v for k, v in vars(ns).items() if k != "capabilities"})
    full = _mech(_small_breath(), ns)
    hand = _mech(_small_breath(), bare)
    assert list(hand["mechanics"]) == list(full["mechanics"])
    assert hand["eilv"] == full["eilv"]
    assert hand["wob"] == full["wob"]


def test_cancel_check_is_honoured_before_any_work():
    b = _small_breath()
    with pytest.raises(Cancelled):
        _mech(b, _ns(), cancel_check=lambda: True)
    assert "mechanics" not in b and "eilv" not in b


# --- cardiac-gated peak EMG on a breath with R-peaks ----------------------------------

INSP_N, EXP_N = 1500, 2000
EXP_T0 = T0 + INSP_N / FS


def _emg_breath(seed=3):
    rng = np.random.RandomState(seed)
    n = INSP_N + EXP_N
    emg = rng.randn(n, 2) * 0.01
    for t in np.arange(T0 + 0.3, T0 + n / FS, 1.0):      # a large spike at every R-peak
        i = int(round((t - T0) * FS))
        emg[i - 5:i + 5, :] += 1.0
    vol_i = np.linspace(0.0, 1.0, INSP_N)
    vol_e = np.linspace(1.0, 0.0, EXP_N)
    insp = _phase(INSP_N, T0, vol=vol_i, flow=-np.sin(np.linspace(0.1, 3.0, INSP_N)),
                  poes=-5 - vol_i, pgas=2 + 0.1 * vol_i, pdi=7 + vol_i, emgcols=emg[:INSP_N])
    exp = _phase(EXP_N, EXP_T0, vol=vol_e, flow=np.sin(np.linspace(0.1, 3.0, EXP_N)),
                 poes=-5 - vol_e, pgas=2 + 0.1 * vol_e, pdi=7 + vol_e, emgcols=emg[INSP_N:])
    breath = {"inspiration": insp, "expiration": exp, "emgcols": emg, "entcols": []}
    for k in ("time", "volume", "flow", "poes", "pgas", "pdi"):
        breath[k] = np.concatenate([insp[k], exp[k]])
    return breath


PEAKS = np.arange(T0 + 0.3, T0 + 3.5, 1.0)              # absolute seconds


def _reference(breath, ns, peaks):
    rp = ns.processing.emg.robust_peak
    out = {}
    for key, seg in (("rms_gated", breath), ("rms_gated_insp", breath["inspiration"]),
                     ("rms_gated_exp", breath["expiration"])):
        local = (np.asarray(peaks, float) - float(seg["time"][0])) * FS
        out[key] = emglib.gated_peak_rms(
            seg["emgcols"], local, ns.processing.emg.rms_s, FS,
            gate_half_width_s=rp.gate_half_width_s, min_survival=rp.min_survival,
            min_island_s=rp.min_island_s)
    return out


def _assert_matches(ret, ref):
    for key, (vals, _qc) in ref.items():
        np.testing.assert_array_equal(np.asarray(ret[key], float), np.asarray(vals, float))


def test_gated_peak_on_each_phase_uses_phase_local_peak_positions():
    ns = _ns(emg=True, robust=True)
    b = _emg_breath()
    ret = {}
    compute._add_gated_peaks(ret, b, ns, PEAKS, True, "")
    ref = _reference(b, ns, PEAKS)
    assert all(qc["ok"] for _v, qc in ref.values())
    _assert_matches(ret, ref)
    assert all(np.isfinite(ret["rms_gated"]))
    assert len(ret["rms_gated"]) == 2 + 2
    assert ret["rms_gated_qc"] == {"ok": True, "reason": ""}
    # the gate really removes the spikes: the gated peak is far below the plain one
    plain = emglib.calculate_rms(b["emgcols"], ns.processing.emg.rms_s, FS)[0]
    assert ret["rms_gated"][0] < 0.5 * plain[0]


def test_gated_peak_with_a_single_r_peak_is_computed():
    ns = _ns(emg=True, robust=True)
    b = _emg_breath()
    peaks = np.array([T0 + 0.3])
    ret = {}
    compute._add_gated_peaks(ret, b, ns, peaks, True, "")
    _assert_matches(ret, _reference(b, ns, peaks))
    assert np.isfinite(ret["rms_gated_insp"][0])


def test_gated_peak_qc_names_the_phase_that_failed():
    ns = _ns(emg=True, robust=True)
    b = _emg_breath()
    peaks = np.arange(T0 + 0.05, T0 + 1.5, 0.2)          # beats packed into the inspiration
    ref = _reference(b, ns, peaks)
    assert ref["rms_gated"][1]["ok"] and ref["rms_gated_exp"][1]["ok"]
    assert not ref["rms_gated_insp"][1]["ok"]
    ret = {}
    compute._add_gated_peaks(ret, b, ns, peaks, True, "")
    _assert_matches(ret, ref)
    assert ret["rms_gated_qc"] == {
        "ok": False, "reason": "rms_gated_insp: " + ref["rms_gated_insp"][1]["reason"]}


@pytest.mark.parametrize("peaks", [None, np.array([])])
def test_no_r_peaks_gives_nan_and_the_documented_reason(peaks):
    ns = _ns(emg=True, robust=True)
    ret = {}
    compute._add_gated_peaks(ret, _emg_breath(), ns, peaks, True, "")
    for key in ("rms_gated", "rms_gated_insp", "rms_gated_exp"):
        assert len(ret[key]) == 4 and all(np.isnan(ret[key]))
    assert ret["rms_gated_qc"] == {"ok": False, "reason": NO_PEAKS_REASON}


def test_rejected_detection_without_a_reason_keeps_the_reason_empty():
    ns = _ns(emg=True, robust=True)
    ret = {}
    compute._add_gated_peaks(ret, _emg_breath(), ns, PEAKS, False, "")
    assert all(np.isnan(ret["rms_gated"]))
    assert ret["rms_gated_qc"] == {"ok": False, "reason": ""}


def test_calculatemechanics_passes_peaks_and_detection_through():
    ns = _ns(emg=True, robust=True)
    b = _emg_breath()
    ref = _reference(b, ns, PEAKS)
    ret = _mech(_emg_breath(), ns, peaks_s=PEAKS)       # defaults: detection accepted
    _assert_matches(ret, ref)
    assert ret["rms_gated_qc"] == {"ok": True, "reason": ""}

    ret = _mech(_emg_breath(), ns)                       # no peaks, default reason
    assert ret["rms_gated_qc"] == {"ok": False, "reason": NO_PEAKS_REASON}

    ret = _mech(_emg_breath(), ns, peaks_s=PEAKS, detection_ok=False,
                detection_reason="beats missed")
    assert ret["rms_gated_qc"] == {"ok": False, "reason": "beats missed"}
    assert all(np.isnan(ret["rms_gated_exp"]))


def test_calculatemechanics_passes_the_cancel_check_to_the_emg_features():
    """The breath-level checkpoint runs first; a checker that only trips on its second
    call can therefore only be caught by the entropy checkpoints further down."""
    d = {"input": {"channels": {"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9,
                                "entropy": [10]},
                   "format": {"sampling_frequency": FS}}}
    ns = to_legacy_ns(Settings.from_dict(d))
    b = _small_breath()
    rng = np.random.RandomState(0)
    ent = rng.randn(10, 1)
    b["entcols"] = ent
    b["inspiration"]["entcols"] = ent[:5]
    b["expiration"]["entcols"] = ent[5:]
    calls = []

    def checker():
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(Cancelled):
        _mech(b, ns, cancel_check=checker)

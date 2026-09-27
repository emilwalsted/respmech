"""``core.analysis.lungvol`` (M-36): the pure per-breath operating-lung-volume
formula, and ``attach()`` on hand-built ``FileResult``/``BatchResult`` objects — the
same no-file-I/O pattern ``test_reference_resolution.py``'s own ``attach()`` section
uses, since this module's ``attach`` only ever consumes what ``references.attach``
already put on the same objects."""
import pandas as pd
import pytest

from respmech.core.analysis.lungvol import attach, compute_breath_olv
from respmech.core.pipeline import BatchResult, FileResult
from respmech.core.settings import (
    BreathTypeEntry, GroupReferenceEntry, ReferenceEntry, BreathRef, Settings, SubjectEntry)


def _settings():
    s = Settings()
    s.input.format.sampling_frequency = 2000
    return s


# --------------------------------------------------------------------------- #
# compute_breath_olv: the pure per-breath formula
# --------------------------------------------------------------------------- #

def test_analytic_case_within_file_tracking_and_rv_anchored_vc():
    """The ticket's own pinned analytical case: IC_ref 3.0 L from EELV 0, VC 5.0 L, a
    tidal breath ending 0.3 L higher than the reference IC's own end-expiratory
    level -- ic_op = 2.7, vol_eelv = 2.3, vol_irv = 2.7 - vt, and vol_eelv abs. TLC
    tracks vol_endexp_b's own SIGN (it rose, so vol_eelv rose too, from what it would
    be with unchanged EELV)."""
    out = compute_breath_olv(
        vol_ic_ref=3.0, vt=0.5, vol_endexp=0.3, eelv_tracking="within_file",
        ic_eelv_pre=0.0, correct_trend=False, tlc=6.0, vc=5.0)
    assert out["d_eelv"] == pytest.approx(0.3)
    assert out["ic_op"] == pytest.approx(2.7)
    assert out["vol_irv"] == pytest.approx(2.7 - 0.5)
    assert out["vol_eelv"] == pytest.approx(2.3)
    assert out["vol_eelv_abs"] == pytest.approx(6.0 - 2.7)


def test_vol_eelv_moves_same_direction_as_vol_endexp_b():
    lower = compute_breath_olv(vol_ic_ref=3.0, vt=0.5, vol_endexp=0.0,
                               eelv_tracking="within_file", ic_eelv_pre=0.0, vc=5.0)
    higher = compute_breath_olv(vol_ic_ref=3.0, vt=0.5, vol_endexp=0.3,
                                eelv_tracking="within_file", ic_eelv_pre=0.0, vc=5.0)
    assert higher["vol_eelv"] > lower["vol_eelv"]


def test_default_tracking_none_holds_ic_op_constant_and_omits_d_eelv():
    out = compute_breath_olv(vol_ic_ref=3.0, vt=0.5, vol_endexp=0.9)
    assert "d_eelv" not in out
    assert out["ic_op"] == pytest.approx(3.0)


def test_correct_trend_forces_nan_not_zero_under_within_file_tracking():
    """A monotone EELV rise the trend-correction filter has ALREADY removed must not
    be silently reported as a flat, definite zero."""
    out = compute_breath_olv(vol_ic_ref=3.0, vt=0.5, vol_endexp=0.5,
                             eelv_tracking="within_file", ic_eelv_pre=0.0,
                             correct_trend=True)
    assert out["d_eelv"] != out["d_eelv"]                     # NaN
    assert out["ic_op"] != out["ic_op"]


def test_cross_file_reference_nans_via_missing_ic_eelv_pre():
    out = compute_breath_olv(vol_ic_ref=3.0, vt=0.5, vol_endexp=0.5,
                             eelv_tracking="within_file", ic_eelv_pre=None)
    assert out["d_eelv"] != out["d_eelv"]
    assert out["ic_op"] != out["ic_op"]


def test_without_tlc_but_with_vc_fills_rv_family_and_nans_tlc_family_only():
    out = compute_breath_olv(vol_ic_ref=3.0, vt=0.5, vol_endexp=0.0, vc=5.0, tlc=None)
    assert out["vol_eelv"] == pytest.approx(5.0 - 3.0)
    assert out["vol_eilv"] == pytest.approx(5.0 - 3.0 + 0.5)
    assert out["eelv_pct_vc"] == pytest.approx(100.0 * (5.0 - 3.0) / 5.0)
    assert out["vol_eelv_abs"] != out["vol_eelv_abs"]
    assert out["eelv_pct_tlc"] != out["eelv_pct_tlc"]


def test_without_either_tlc_or_vc_both_families_are_nan_but_ic_op_and_irv_are_not():
    out = compute_breath_olv(vol_ic_ref=3.0, vt=0.5, vol_endexp=0.0)
    assert out["ic_op"] == pytest.approx(3.0)
    assert out["vol_irv"] == pytest.approx(2.5)
    assert out["vol_eelv"] != out["vol_eelv"]
    assert out["vol_eelv_abs"] != out["vol_eelv_abs"]


def test_units_of_every_new_column():
    from _helpers import assert_units
    assert_units({
        "d_eelv": "L", "ic_op": "L", "delta_ic": "L", "delta_eelv": "L", "tlc": "L",
        "vc": "L", "vol_irv": "L", "vol_eelv": "L", "vol_eilv": "L",
        "vol_eelv_abs": "L", "vol_eilv_abs": "L", "delta_ic_pct": "%",
        "vt_pct_ic": "%", "eelv_pct_vc": "%", "eilv_pct_vc": "%", "irv_pct_vc": "%",
        "eelv_pct_tlc": "%", "eilv_pct_tlc": "%", "irv_pct_tlc": "%"})


# --------------------------------------------------------------------------- #
# attach(): pure, hand-built FileResult/BatchResult, no file I/O
# --------------------------------------------------------------------------- #

def _manoeuvre_row(vol_ic, ic_eelv_pre=0.0, quality=None):
    return {"kind": "ic", "vol_ic": vol_ic, "ic_eelv_pre": ic_eelv_pre,
           "quality": list(quality or [])}


def _breaths_table(rows):
    return pd.DataFrame(rows)


def _file_result(*, breaths_table, vol_ic_ref, ic_ref_n=1.0, ic_ref_source="A.csv",
                 manoeuvres=None, references_used=None, file="A.csv"):
    avg = pd.DataFrame([{c: breaths_table[c].mean() for c in breaths_table.columns}])
    avg["vol_ic_ref"] = vol_ic_ref
    avg["ic_ref_n"] = ic_ref_n
    avg["ic_ref_source"] = ic_ref_source
    bt = breaths_table.copy()
    bt["vol_ic_ref"] = vol_ic_ref
    bt["ic_ref_n"] = ic_ref_n
    bt["ic_ref_source"] = ic_ref_source
    fr = FileResult(file=file, role="tidal", breaths_table=bt, average_row=avg,
                    manoeuvres=manoeuvres or {})
    if references_used is not None:
        fr.references_used = references_used            # explicit, including {} on purpose
    else:
        fr.references_used = {
            "ic": {"source": ic_ref_source, "breaths": [4], "n": int(ic_ref_n),
                  "value": vol_ic_ref}}
    return fr


def _batch_result(files_dict, references=None):
    br = BatchResult()
    br.files = files_dict
    br.references = references or {}
    return br


def test_attach_is_a_no_op_without_a_resolvable_ic_reference():
    s = _settings()
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    a = FileResult(file="A.csv", role="tidal", breaths_table=bt,
                   average_row=pd.DataFrame([{"vt": 0.5}]))
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert "ic_op" not in a.breaths_table.columns


def test_attach_adds_ic_op_and_irv_with_default_tracking():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.1}, {"vt": 0.6, "vol_endexp": -0.1}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.0,
                     manoeuvres={4: _manoeuvre_row(3.0)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert list(a.breaths_table["ic_op"]) == pytest.approx([3.0, 3.0])
    assert list(a.breaths_table["vol_irv"]) == pytest.approx([2.5, 2.4])
    # column set is present even with no VC/TLC configured anywhere:
    for col in ("vol_eelv", "vol_eelv_abs", "eelv_pct_vc", "eelv_pct_tlc"):
        assert col in a.breaths_table.columns
        assert a.breaths_table[col].isna().all()


def test_attach_within_file_tracking_uses_the_same_file_ic_eelv_pre():
    s = _settings()
    s.processing.lung_volume.ic.eelv_tracking = "within_file"
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.3}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.0,
                     manoeuvres={4: _manoeuvre_row(3.0, ic_eelv_pre=0.0)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert a.breaths_table["d_eelv"].iloc[0] == pytest.approx(0.3)
    assert a.breaths_table["ic_op"].iloc[0] == pytest.approx(2.7)


def test_attach_nans_within_file_tracking_and_notices_on_a_cross_file_reference():
    s = _settings()
    s.processing.lung_volume.ic.eelv_tracking = "within_file"
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="A", ic=BreathRef(file="A_IC.csv", breaths=[4])))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.3}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.0, ic_ref_source="A_IC.csv",
                     file="A_1.csv",
                     references_used={"ic": {"source": "A_IC.csv", "breaths": [4],
                                            "n": 1, "value": 3.0}})
    result = _batch_result({"A_1.csv": a})
    attach(result, s, ["A_1.csv"])
    assert a.breaths_table["d_eelv"].isna().iloc[0]
    assert a.breaths_table["ic_op"].isna().iloc[0]
    assert any("cross-file" in n for n in a.notices)


def test_attach_uses_subject_vc_and_tlc():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A_1.csv", breath=4, kind="ic"))
    s.input.subjects.append(SubjectEntry(key="A", tlc_l=6.0, vc_l=5.0))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.0, manoeuvres={4: _manoeuvre_row(3.0)},
                     file="A_1.csv")
    result = _batch_result({"A_1.csv": a})
    attach(result, s, ["A_1.csv"])
    assert a.breaths_table["vc"].iloc[0] == pytest.approx(5.0)
    assert a.breaths_table["tlc"].iloc[0] == pytest.approx(6.0)
    assert a.breaths_table["vol_eelv"].iloc[0] == pytest.approx(2.0)
    assert a.breaths_table["vol_eelv_abs"].iloc[0] == pytest.approx(3.0)


def test_attach_computes_delta_ic_against_an_explicit_baseline_ic_link():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    s.processing.references.append(ReferenceEntry(
        file="A.csv", baseline_ic=BreathRef(file="A_rest.csv", breaths=[1])))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.5, manoeuvres={4: _manoeuvre_row(3.5)})
    result = _batch_result({"A.csv": a},
                           references={"A_rest.csv": {1: _manoeuvre_row(3.0)}})
    attach(result, s, ["A.csv"])
    assert a.breaths_table["delta_ic"].iloc[0] == pytest.approx(0.5)
    assert a.breaths_table["delta_eelv"].iloc[0] == pytest.approx(-0.5)
    assert a.breaths_table["delta_ic_pct"].iloc[0] == pytest.approx(100.0 * 0.5 / 3.0)


def test_attach_delta_ic_is_nan_without_notice_when_no_baseline_is_configured():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.5, manoeuvres={4: _manoeuvre_row(3.5)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert a.breaths_table["delta_ic"].isna().iloc[0]
    assert not any("baseline_ic" in n for n in a.notices)


def test_attach_notices_negative_operating_volumes_once_per_file():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A_1.csv", breath=4, kind="ic"))
    s.input.subjects.append(SubjectEntry(key="A", vc_l=1.0))
    # ic_op (3.0) exceeds vc (1.0) -> vol_eelv negative
    bt = _breaths_table([{"vt": 0.2, "vol_endexp": 0.0}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.0, manoeuvres={4: _manoeuvre_row(3.0)},
                     file="A_1.csv")
    result = _batch_result({"A_1.csv": a})
    attach(result, s, ["A_1.csv"])
    assert any("vol_eelv is negative" in n for n in a.notices)


def test_attach_skips_a_reference_only_file():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    a = FileResult(file="A_IC.csv", role="reference", manoeuvres={4: _manoeuvre_row(3.0)})
    result = _batch_result({"A_IC.csv": a})
    attach(result, s, ["A_IC.csv"])
    assert a.breaths_table is None                      # nothing to add a column to


def test_attach_records_the_family_and_active_files_in_analysis_plan():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.0, manoeuvres={4: _manoeuvre_row(3.0)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert result.analysis_plan["lung_volume"] == {"family": True, "active_files": ["A.csv"]}


def test_attach_records_no_active_files_when_family_absent():
    s = _settings()
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    a = FileResult(file="A.csv", role="tidal", breaths_table=bt,
                   average_row=pd.DataFrame([{"vt": 0.5}]))
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert result.analysis_plan["lung_volume"] == {"family": False, "active_files": []}


def test_attach_handles_an_empty_breaths_table_without_crashing():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    bt = pd.DataFrame(columns=["vt", "vol_endexp"])
    a = _file_result(breaths_table=bt, vol_ic_ref=3.0, manoeuvres={4: _manoeuvre_row(3.0)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])                          # must not raise
    for col in ("ic_op", "vol_irv", "vol_eelv", "tlc", "vc"):
        assert col in a.breaths_table.columns
    assert a.average_row["ic_op"].isna().iloc[0]
    assert "A.csv" in result.analysis_plan["lung_volume"]["active_files"]


def test_attach_degrades_a_single_files_unexpected_exception_to_a_notice():
    """A bug/unexpected exception while computing ONE file's operating lung volumes
    must not abort the whole batch -- the same per-file isolation run_batch's own
    main loop already gives every other failure mode."""
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="B.csv", breath=4, kind="ic"))
    bt_a = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    bt_b = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    a = _file_result(breaths_table=bt_a, vol_ic_ref=3.0, manoeuvres={4: _manoeuvre_row(3.0)},
                     file="A.csv")
    # Force a real exception on A.csv only: average_row missing the mean() target
    # column raises inside _attach_one_file's per-breath loop (a stand-in for "any
    # unanticipated bug"), while B.csv is an ordinary, healthy file.
    a.breaths_table = a.breaths_table.drop(columns=["vt"])
    b = _file_result(breaths_table=bt_b, vol_ic_ref=4.0, manoeuvres={4: _manoeuvre_row(4.0)},
                     file="B.csv")
    result = _batch_result({"A.csv": a, "B.csv": b})
    attach(result, s, ["A.csv", "B.csv"])                 # must not raise
    assert any("operating lung volumes could not be computed" in n for n in a.notices)
    assert "A.csv" not in result.analysis_plan["lung_volume"]["active_files"]
    assert "ic_op" in b.breaths_table.columns             # B.csv still processed normally
    assert "B.csv" in result.analysis_plan["lung_volume"]["active_files"]


def test_attach_within_file_tracking_does_not_say_cross_file_when_never_resolved():
    """When the IC reference never resolved at all (family present, this file's own
    link soft-failed), the notice must not claim it was resolved cross-file -- that
    would misdescribe why d_eelv/ic_op are NaN."""
    s = _settings()
    s.processing.lung_volume.ic.eelv_tracking = "within_file"
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    bt = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    # references_used has NO "ic" entry -- the soft-unresolved case (family present
    # elsewhere, this file's own link never resolved), as references.attach() itself
    # leaves it when nothing usable resolved.
    a = _file_result(breaths_table=bt, vol_ic_ref=float("nan"), ic_ref_n=float("nan"),
                     ic_ref_source=None, references_used={})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert not any("cross-file" in n for n in a.notices)
    assert a.breaths_table["ic_op"].isna().iloc[0]


def test_attach_baseline_pattern_fallback_matches_a_sibling_by_name():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A_1.csv", breath=4, kind="ic"))
    bt_target = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    target = _file_result(breaths_table=bt_target, vol_ic_ref=3.5,
                          manoeuvres={4: _manoeuvre_row(3.5)}, file="A_1.csv")
    bt_base = _breaths_table([{"vt": 0.4, "vol_endexp": 0.0}])
    baseline = _file_result(breaths_table=bt_base, vol_ic_ref=3.0, file="A_rest.csv")
    result = _batch_result({"A_1.csv": target, "A_rest.csv": baseline})
    attach(result, s, ["A_1.csv", "A_rest.csv"])
    assert target.breaths_table["delta_ic"].iloc[0] == pytest.approx(0.5)
    assert not any("baseline_ic" in n for n in target.notices)


def test_attach_baseline_pattern_notices_an_ambiguous_multiple_match():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A_1.csv", breath=4, kind="ic"))
    bt_target = _breaths_table([{"vt": 0.5, "vol_endexp": 0.0}])
    target = _file_result(breaths_table=bt_target, vol_ic_ref=3.5,
                          manoeuvres={4: _manoeuvre_row(3.5)}, file="A_1.csv")
    bt_base = _breaths_table([{"vt": 0.4, "vol_endexp": 0.0}])
    base_a = _file_result(breaths_table=bt_base, vol_ic_ref=2.0, file="A_rest_a.csv")
    base_b = _file_result(breaths_table=bt_base.copy(), vol_ic_ref=9.0, file="A_rest_b.csv")
    result = _batch_result(
        {"A_1.csv": target, "A_rest_a.csv": base_a, "A_rest_b.csv": base_b})
    attach(result, s, ["A_1.csv", "A_rest_a.csv", "A_rest_b.csv"])
    assert any("matches more than one sibling" in n for n in target.notices)
    # alphabetically first candidate (A_rest_a.csv, vol_ic_ref=2.0) wins:
    assert target.breaths_table["delta_ic"].iloc[0] == pytest.approx(1.5)


def test_resolve_vc_returns_none_gracefully_when_fvc_key_is_absent():
    """The linked-FVC fallback (pre-M-42, manoeuvres.extract never sets 'fvc') must
    degrade to None, not raise, so vol_eelv/vol_eilv/etc. simply stay NaN."""
    from respmech.core.settings import BreathRef, GroupReferenceEntry
    from respmech.core.analysis.lungvol import _resolve_vc

    s = _settings()
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="A", fvc=BreathRef(file="A_fvc.csv", breaths=[7])))
    a = _file_result(breaths_table=_breaths_table([{"vt": 0.5, "vol_endexp": 0.0}]),
                     vol_ic_ref=3.0, manoeuvres={7: {"kind": "fvc", "quality": []}},
                     file="A_1.csv")
    result = _batch_result({"A_1.csv": a})
    vc = _resolve_vc(result, s, "A_1.csv", None, s.processing.lung_volume.ic)
    assert vc is None

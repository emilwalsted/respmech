"""``core.analysis.references``: resolution order, ``check_links`` cautions, and (M-35)
the forepass/afterpass that wires cross-file references into ``core.pipeline.run_batch``.

``resolve_reference``/``check_links``/``external_reference_sources`` and the ``attach()``
tests built on hand-constructed ``FileResult``/``BatchResult`` objects are pure, Qt-free
and do no file I/O of their own (``core.pipeline`` is imported only for its plain
dataclasses — no ``segment_file``/``run_batch`` call in that section touches a real
recording). The forepass/subset-equivalence acceptance tests at the end of this file DO
run the real pipeline over the committed synthetic recordings (``requires_synth()``-
gated, same convention as ``test_core_outputs.py``) — see that section's own header.
"""
import pytest

from respmech.core.analysis.references import (
    REFERENCE_SLOTS, ReferenceLinkError, attach, check_links, external_reference_sources,
    resolve_reference)
from respmech.core.settings import (
    BreathRef, BreathTypeEntry, ExcludeEntry, GroupReferenceEntry, ReferenceEntry, Settings,
    SubjectEntry)


def _settings():
    s = Settings()
    s.input.format.sampling_frequency = 2000
    return s


# --------------------------------------------------------------------------- #
# resolve_reference: resolution order
# --------------------------------------------------------------------------- #

def test_explicit_reference_entry_wins_over_everything():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2, 3])))
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="P03", ic=BreathRef(file="P03_OTHER_IC.txt", breaths=[9])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_peak.txt", breath=1, kind="ic"))

    ref = resolve_reference("P03_peak.txt", "ic", s)
    assert ref == BreathRef(file="P03_IC.txt", breaths=[2, 3])


def test_group_default_wins_when_no_explicit_entry_for_this_slot():
    s = _settings()
    s.processing.references.append(ReferenceEntry(file="P03_peak.txt"))    # ic unset
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="P03", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_peak.txt", breath=1, kind="ic"))

    ref = resolve_reference("P03_peak.txt", "ic", s)
    assert ref == BreathRef(file="P03_IC.txt", breaths=[2])


def test_group_default_is_matched_via_group_key_leading_token():
    """No explicit ReferenceEntry at all -- the group is found via
    core.summary.group_key's default (the leading filename token)."""
    s = _settings()
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="P03", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    ref = resolve_reference("P03_120W.txt", "ic", s)
    assert ref == BreathRef(file="P03_IC.txt", breaths=[2])


def test_own_typed_breath_is_the_fallback_for_ic():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="P07_sniffs.txt", breath=1, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="P07_sniffs.txt", breath=3, kind="ic"))
    ref = resolve_reference("P07_sniffs.txt", "ic", s)
    assert ref == BreathRef(file="P07_sniffs.txt", breaths=[1, 3])


def test_own_typed_ic_fvc_counts_for_both_ic_and_fvc_slots():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=2, kind="ic_fvc"))
    assert resolve_reference("x.txt", "ic", s) == BreathRef(file="x.txt", breaths=[2])
    assert resolve_reference("x.txt", "fvc", s) == BreathRef(file="x.txt", breaths=[2])


def test_own_typed_sniff_counts_for_max_insp_slot():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=1, kind="sniff"))
    assert resolve_reference("x.txt", "max_insp", s) == BreathRef(file="x.txt", breaths=[1])


def test_baseline_ic_has_no_own_typed_fallback():
    """A baseline is by nature a DIFFERENT recording -- an own-typed IC breath must
    never be silently reused as the file's own baseline."""
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=1, kind="ic"))
    assert resolve_reference("x.txt", "baseline_ic", s) is None
    # the SAME breath still resolves normally for the "ic" slot:
    assert resolve_reference("x.txt", "ic", s) == BreathRef(file="x.txt", breaths=[1])


def test_no_link_at_all_resolves_to_none():
    s = _settings()
    assert resolve_reference("x.txt", "ic", s) is None


def test_unknown_slot_raises_value_error():
    s = _settings()
    with pytest.raises(ValueError, match="unknown reference slot"):
        resolve_reference("x.txt", "not_a_slot", s)


def test_all_four_slots_are_exposed():
    assert set(REFERENCE_SLOTS) == {"ic", "fvc", "baseline_ic", "max_insp"}


# --------------------------------------------------------------------------- #
# check_links: cautions, never an exception
# --------------------------------------------------------------------------- #

def test_check_links_returns_empty_for_a_fully_resolved_batch():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_IC.txt", breath=2, kind="ic"))
    assert check_links(s, ["P03_peak.txt", "P03_IC.txt"]) == []


def test_check_links_flags_a_source_not_among_the_analysed_files():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    cautions = check_links(s, ["P03_peak.txt"])          # P03_IC.txt not in the batch
    assert len(cautions) == 1
    assert "P03_IC.txt" in cautions[0] and "not among the analysed files" in cautions[0]


def test_check_links_uses_the_callers_filename_list_not_a_manifest_subset():
    """filenames is whatever run_batch's own match_input_files gives -- a caller may
    pass full paths (as match_input_files does); check_links compares basenames."""
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_IC.txt", breath=2, kind="ic"))
    cautions = check_links(s, ["/data/study/P03_peak.txt", "/data/study/P03_IC.txt"])
    assert cautions == []


def test_check_links_flags_a_linked_breath_not_typed_the_matching_kind():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[5])))
    # breath 5 of P03_IC.txt exists in the batch but was never typed 'ic'/'ic_fvc'
    cautions = check_links(s, ["P03_peak.txt", "P03_IC.txt"])
    assert len(cautions) == 1
    assert "linked as ic but not typed ic" in cautions[0]


def test_check_links_flags_a_linked_breath_that_is_excluded():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[5])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_IC.txt", breath=5, kind="ic"))
    s.processing.exclude_breaths.append(ExcludeEntry(file="P03_IC.txt", breaths=[5]))
    cautions = check_links(s, ["P03_peak.txt", "P03_IC.txt"])
    assert len(cautions) == 1
    assert "is excluded" in cautions[0]


def test_check_links_flags_a_group_default_key_matching_no_file():
    s = _settings()
    s.processing.reference_defaults.append(GroupReferenceEntry(group="P99"))
    cautions = check_links(s, ["P03_peak.txt"])
    assert len(cautions) == 1
    assert "P99" in cautions[0] and "matches no analysed file" in cautions[0]


def test_check_links_flags_a_subject_key_matching_no_file():
    s = _settings()
    s.input.subjects.append(SubjectEntry(key="P99"))
    cautions = check_links(s, ["P03_peak.txt"])
    assert len(cautions) == 1
    assert "P99" in cautions[0] and "matches no analysed file" in cautions[0]


def test_check_links_never_raises_on_a_batch_with_no_links_at_all():
    s = _settings()
    assert check_links(s, ["a.txt", "b.txt"]) == []


def test_check_links_baseline_ic_link_is_not_checked_for_own_typed_kind():
    """baseline_ic has no own-typed fallback (see resolve_reference), but an EXPLICIT
    baseline_ic link still resolves normally and is not flagged as 'not typed
    baseline_ic' -- there is no such breath kind to type it as (see _OWN_TYPED_KINDS)."""
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", baseline_ic=BreathRef(file="P03_rest.txt", breaths=[1])))
    # P03_rest.txt's breath 1 is never typed at all
    cautions = check_links(s, ["P03_peak.txt", "P03_rest.txt"])
    assert cautions == []


def test_check_links_flags_a_reference_naming_no_breaths_at_all():
    """A BreathRef whose source file IS in the batch but whose breaths list is empty
    names nothing usable -- silently accepting it would let a hand-edited or
    programmatically-built empty reference look fully resolved."""
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[])))
    cautions = check_links(s, ["P03_peak.txt", "P03_IC.txt"])
    assert len(cautions) == 1
    assert "P03_IC.txt" in cautions[0] and "no breaths" in cautions[0]


def test_check_links_reports_only_the_actually_mistyped_breath_in_a_mixed_breathref():
    """A multi-breath BreathRef where only SOME breaths are correctly typed must
    report exactly the mismatching one(s) -- not stop after the first breath, and not
    flag a breath that resolves cleanly."""
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2, 3])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_IC.txt", breath=2, kind="ic"))
    # breath 3 is never typed
    cautions = check_links(s, ["P03_peak.txt", "P03_IC.txt"])
    assert len(cautions) == 1
    assert "breath 3" in cautions[0] and "breath 2" not in cautions[0]


def test_check_links_reports_only_the_actually_excluded_breath_in_a_mixed_breathref():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2, 3])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_IC.txt", breath=2, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_IC.txt", breath=3, kind="ic"))
    s.processing.exclude_breaths.append(ExcludeEntry(file="P03_IC.txt", breaths=[3]))
    cautions = check_links(s, ["P03_peak.txt", "P03_IC.txt"])
    assert len(cautions) == 1
    assert "breath 3" in cautions[0] and "excluded" in cautions[0]


def test_check_links_full_path_filenames_resolve_group_keys_correctly():
    """Regression: check_links must basename EVERY filename before deriving group
    keys, not just before matching per-file reference sources -- otherwise a caller
    passing match_input_files' own full-path result (the documented convention) would
    always see reference_defaults/subjects group keys as mismatched."""
    s = _settings()
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="P03", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    s.processing.breath_types.append(BreathTypeEntry(file="P03_IC.txt", breath=2, kind="ic"))
    cautions = check_links(s, ["/data/study/P03_120W.txt", "/data/study/P03_IC.txt"])
    assert cautions == []


# --------------------------------------------------------------------------- #
# missing_reference_sources / ui.validation.path_problem's require_references gate
# --------------------------------------------------------------------------- #

def test_missing_reference_sources_lists_only_sources_not_in_the_batch():
    from respmech.core.analysis.references import missing_reference_sources

    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    assert missing_reference_sources(s, ["P03_peak.txt"]) == ["P03_IC.txt"]
    assert missing_reference_sources(s, ["P03_peak.txt", "P03_IC.txt"]) == []


def test_missing_reference_sources_is_empty_with_no_references_at_all():
    from respmech.core.analysis.references import missing_reference_sources
    assert missing_reference_sources(_settings(), ["a.txt"]) == []


def test_path_problem_is_a_soft_caution_by_default_when_a_reference_source_is_missing(
        tmp_path):
    from respmech.ui.validation import path_problem

    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "P03_peak.txt").write_text("x")
    out = tmp_path / "output"
    out.mkdir()
    s = _settings()
    s.input.folder = str(inp)
    s.input.files = "*.txt"
    s.output.folder = str(out)
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    assert path_problem(s) is None


def test_path_problem_blocks_on_a_missing_reference_source_once_required(tmp_path):
    from respmech.ui.validation import path_problem

    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "P03_peak.txt").write_text("x")
    out = tmp_path / "output"
    out.mkdir()
    s = _settings()
    s.input.folder = str(inp)
    s.input.files = "*.txt"
    s.output.folder = str(out)
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    s.processing.lung_volume.require_references = True
    msg = path_problem(s)
    assert msg is not None and "P03_IC.txt" in msg


# --------------------------------------------------------------------------- #
# references/subjects are never sent into core.compute
# --------------------------------------------------------------------------- #

def test_references_and_subjects_never_reach_to_legacy_ns():
    from respmech.core._legacy_ns import to_legacy_ns

    s = _settings()
    s.input.channels.flow = 1
    s.input.folder = "input"
    s.processing.references.append(ReferenceEntry(
        file="P03_peak.txt", ic=BreathRef(file="P03_IC.txt", breaths=[2])))
    s.processing.reference_defaults.append(GroupReferenceEntry(group="P03"))
    s.input.subjects.append(SubjectEntry(key="P03", tlc_l=6.0))

    ns = to_legacy_ns(s)
    ns_fields = vars(ns)
    for name in ("references", "reference_defaults", "subjects"):
        assert name not in ns_fields, (
            f"to_legacy_ns must never carry {name!r} -- referenced breaths are never "
            "sent into core.compute")


# --------------------------------------------------------------------------- #
# M-35: external_reference_sources (pure, no file I/O)
# --------------------------------------------------------------------------- #

def test_external_reference_sources_excludes_files_already_in_the_batch():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="B.csv", ic=BreathRef(file="A_ic.csv", breaths=[4])))
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="P03", fvc=BreathRef(file="P03_fvc.csv", breaths=[1])))
    assert external_reference_sources(s, ["B.csv", "A_ic.csv"]) == ["P03_fvc.csv"]


def test_external_reference_sources_is_sorted_and_deduplicated():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="B.csv", ic=BreathRef(file="shared.csv", breaths=[4]),
        fvc=BreathRef(file="shared.csv", breaths=[7])))
    assert external_reference_sources(s, ["B.csv"]) == ["shared.csv"]


def test_external_reference_sources_empty_when_nothing_configured():
    s = _settings()
    assert external_reference_sources(s, ["A.csv", "B.csv"]) == []


# --------------------------------------------------------------------------- #
# M-35: attach() -- pure, hand-built FileResult/BatchResult, no file I/O
# --------------------------------------------------------------------------- #

def _manoeuvre_row(vol_ic, quality=None):
    return {"kind": "ic", "vol_ic": vol_ic, "quality": list(quality or [])}


def _fake_table(n_rows, columns=("vt",)):
    import pandas as pd
    return pd.DataFrame([{c: 0.0 for c in columns} for _ in range(n_rows)])


def _file_result(*, role="tidal", manoeuvres=None, breaths_n=3, error=None):
    from respmech.core.pipeline import FileResult
    tidal_ok = role == "tidal" and error is None
    return FileResult(
        file="x", role=role, manoeuvres=manoeuvres or {}, error=error,
        breaths_table=_fake_table(breaths_n) if tidal_ok else None,
        average_row=_fake_table(1) if tidal_ok else None)


def _batch_result(files_dict, references=None, reference_errors=None):
    from respmech.core.pipeline import BatchResult
    br = BatchResult()
    br.files = files_dict
    br.references = references or {}
    br.reference_errors = reference_errors or {}
    return br


def test_attach_is_a_no_op_when_no_ic_reference_exists_anywhere():
    s = _settings()
    a = _file_result()
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert "vol_ic_ref" not in a.breaths_table.columns
    assert result.analysis_plan == {"ic": {"family": False, "resolved": {}, "unresolved": []}}


def test_attach_resolves_own_typed_ic_and_computes_mean_aggregate():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=9, kind="ic"))
    a = _file_result(manoeuvres={4: _manoeuvre_row(3.0), 9: _manoeuvre_row(5.0)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert list(a.breaths_table["vol_ic_ref"]) == [4.0, 4.0, 4.0]        # mean(3.0, 5.0)
    assert list(a.breaths_table["ic_ref_n"]) == [2.0, 2.0, 2.0]
    assert list(a.breaths_table["ic_ref_source"]) == ["A.csv"] * 3
    assert a.average_row["vol_ic_ref"].iloc[0] == pytest.approx(4.0)
    assert a.average_row["ic_ref_n"].iloc[0] == pytest.approx(2.0)
    assert a.average_row["ic_ref_source"].iloc[0] == "A.csv"
    assert a.references_used["ic"] == {
        "source": "A.csv", "breaths": [4, 9], "n": 2, "value": 4.0}
    assert result.analysis_plan["ic"]["resolved"]["A.csv"]["value"] == pytest.approx(4.0)
    # Columns are appended, never inserted before an existing one:
    assert list(a.breaths_table.columns)[:1] == ["vt"]


def test_attach_excludes_reject_flagged_breaths_from_the_aggregate():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=9, kind="ic"))
    a = _file_result(manoeuvres={
        4: _manoeuvre_row(3.0, quality=["LOW_EFFORT"]),
        9: _manoeuvre_row(5.0)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert a.average_row["vol_ic_ref"].iloc[0] == pytest.approx(5.0)   # LOW_EFFORT excluded
    assert a.average_row["ic_ref_n"].iloc[0] == pytest.approx(1.0)


def test_attach_uses_median_aggregate_when_configured():
    s = _settings()
    s.processing.lung_volume.ic.aggregate = "median"
    for b in (4, 9, 14):
        s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=b, kind="ic"))
    a = _file_result(manoeuvres={
        4: _manoeuvre_row(1.0), 9: _manoeuvre_row(2.0), 14: _manoeuvre_row(100.0)})
    result = _batch_result({"A.csv": a})
    attach(result, s, ["A.csv"])
    assert a.average_row["vol_ic_ref"].iloc[0] == pytest.approx(2.0)   # median, not mean


def test_attach_nans_and_notices_a_file_with_no_reference_when_family_present():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    a = _file_result(manoeuvres={4: _manoeuvre_row(3.0)})
    b = _file_result(manoeuvres={})
    result = _batch_result({"A.csv": a, "B.csv": b})
    attach(result, s, ["A.csv", "B.csv"])
    assert a.average_row["vol_ic_ref"].iloc[0] == pytest.approx(3.0)
    assert b.average_row["vol_ic_ref"].isna().iloc[0]
    assert b.average_row["ic_ref_n"].isna().iloc[0]
    assert b.notices and "no IC reference resolves" in b.notices[-1]
    assert b.error is None                          # soft: the file itself stays OK
    assert "B.csv" in result.analysis_plan["ic"]["unresolved"]


def test_attach_skips_a_reference_only_file():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    a = _file_result(manoeuvres={4: _manoeuvre_row(3.0)})
    ref_only = _file_result(role="reference", manoeuvres={1: _manoeuvre_row(9.0)})
    result = _batch_result({"A.csv": a, "R.csv": ref_only})
    attach(result, s, ["A.csv", "R.csv"])
    assert "R.csv" not in result.analysis_plan["ic"]["resolved"]
    assert "R.csv" not in result.analysis_plan["ic"]["unresolved"]


def test_attach_reads_an_external_forepass_source_via_result_references():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="A.csv", ic=BreathRef(file="EXT.csv", breaths=[4])))
    a = _file_result(manoeuvres={})
    result = _batch_result({"A.csv": a}, references={"EXT.csv": {4: _manoeuvre_row(7.5)}})
    attach(result, s, ["A.csv"])
    assert a.average_row["vol_ic_ref"].iloc[0] == pytest.approx(7.5)
    assert a.references_used["ic"]["source"] == "EXT.csv"


def test_attach_nans_and_notices_when_the_external_source_failed_entirely():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="A.csv", ic=BreathRef(file="EXT.csv", breaths=[4])))
    a = _file_result(manoeuvres={})
    result = _batch_result(
        {"A.csv": a}, references={}, reference_errors={("EXT.csv", None): "FileNotFoundError: x"})
    attach(result, s, ["A.csv"])
    assert a.average_row["vol_ic_ref"].isna().iloc[0]
    assert a.notices
    assert a.error is None                           # soft by default


def test_attach_ignores_manoeuvres_on_a_source_file_that_itself_failed():
    s = _settings()
    s.processing.references.append(ReferenceEntry(
        file="B.csv", ic=BreathRef(file="A.csv", breaths=[4])))
    a = _file_result(manoeuvres={4: _manoeuvre_row(3.0)}, error="TrimError: boom")
    a.error_kind = "TrimError"
    b = _file_result(manoeuvres={})
    result = _batch_result({"A.csv": a, "B.csv": b})
    attach(result, s, ["A.csv", "B.csv"])
    assert b.average_row["vol_ic_ref"].isna().iloc[0]


def test_attach_demotes_a_file_to_failed_when_require_references_is_set():
    s = _settings()
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=4, kind="ic"))
    s.processing.lung_volume.require_references = True
    a = _file_result(manoeuvres={4: _manoeuvre_row(3.0)})
    b = _file_result(manoeuvres={})
    result = _batch_result({"A.csv": a, "B.csv": b})
    attach(result, s, ["A.csv", "B.csv"])
    assert b.error is not None
    assert b.error_kind == "ReferenceLinkError"
    assert a.error is None
    assert "A.csv" in result.ok_files
    assert "B.csv" not in result.ok_files


def test_reference_link_error_is_importable_and_a_value_error():
    assert issubclass(ReferenceLinkError, ValueError)


# --------------------------------------------------------------------------- #
# M-35: forepass + subset/full-run equivalence -- real pipeline over the
# committed synthetic recordings (requires_synth(), same convention as
# test_core_outputs.py). Individually @requires_synth()-decorated rather than a
# file-wide pytestmark, since every test above this section needs no synthetic
# input at all and must keep running without it.
# --------------------------------------------------------------------------- #
from _helpers import requires_synth, synth_settings                        # noqa: E402


@requires_synth()
def test_forepass_loads_an_out_of_batch_reference_source_and_attach_uses_it(tmp_path):
    """synth_case_B.csv's IC reference names synth_manoeuvre_A.csv -- a DIFFERENT file
    synth_case_*.csv's own glob never matches -- so the forepass must load and segment
    it separately, and attach() must find its manoeuvre there. vol_ic == 3.0 is the
    same literal analytical constant the typed_ic_fvc_same_file golden scenario itself
    pins for this exact breath (see that scenario's own dedicated test)."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path))
    # The reference LINK alone does not type the breath -- typing and linking are
    # separate settings tables (core.settings.BreathTypeEntry vs. ReferenceEntry); the
    # source file's own breath 4 must be typed 'ic' for manoeuvres.extract() to
    # produce a vol_ic value for it at all, exactly as it would for an in-batch file.
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_manoeuvre_A.csv", breath=4, kind="ic"))
    s.processing.references.append(ReferenceEntry(
        file="synth_case_B.csv", ic=BreathRef(file="synth_manoeuvre_A.csv", breaths=[4])))
    s.validate()

    result = run_batch(s)

    assert "synth_manoeuvre_A.csv" in result.references
    assert 4 in result.references["synth_manoeuvre_A.csv"]
    b = result.files["synth_case_B.csv"]
    assert b.average_row["vol_ic_ref"].iloc[0] == pytest.approx(3.0)
    assert b.average_row["ic_ref_n"].iloc[0] == pytest.approx(1.0)
    assert b.references_used["ic"]["source"] == "synth_manoeuvre_A.csv"


@requires_synth()
def test_subset_run_reference_row_equals_full_run_row(tmp_path):
    """Acceptance criterion: a subset run whose IC source lies OUTSIDE the subset (but
    IS one of the batch's own matched files) gives the SAME reference row as a full
    run -- the forepass picks the source up regardless of only_files."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path))
    s.processing.breath_types.append(BreathTypeEntry(file="synth_case_A.csv", breath=4, kind="ic"))
    s.processing.references.append(ReferenceEntry(
        file="synth_case_B.csv", ic=BreathRef(file="synth_case_A.csv", breaths=[4])))
    s.validate()

    full = run_batch(s)
    subset = run_batch(s, only_files=["synth_case_B.csv"])

    b_full, b_subset = full.files["synth_case_B.csv"].average_row, subset.files["synth_case_B.csv"].average_row
    assert b_subset["vol_ic_ref"].iloc[0] == pytest.approx(b_full["vol_ic_ref"].iloc[0])
    assert b_subset["ic_ref_n"].iloc[0] == b_full["ic_ref_n"].iloc[0]
    assert b_subset["ic_ref_source"].iloc[0] == b_full["ic_ref_source"].iloc[0]
    # The forepass really did the work in the subset run (A is not in `files` there):
    assert "synth_case_A.csv" in subset.references


@requires_synth()
def test_subset_write_has_the_same_columns_as_the_full_run(tmp_path):
    """Acceptance criterion: the written column SET is identical between a subset and a
    full run -- the IC family is decided from settings across allfiles, never from
    what this particular run happened to resolve."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path))
    s.processing.breath_types.append(BreathTypeEntry(file="synth_case_A.csv", breath=4, kind="ic"))
    s.processing.references.append(ReferenceEntry(
        file="synth_case_B.csv", ic=BreathRef(file="synth_case_A.csv", breaths=[4])))
    s.validate()

    full = run_batch(s)
    subset = run_batch(s, only_files=["synth_case_B.csv"])

    full_cols = set(full.files["synth_case_B.csv"].average_row.columns)
    subset_cols = set(subset.files["synth_case_B.csv"].average_row.columns)
    assert full_cols == subset_cols
    assert {"vol_ic_ref", "ic_ref_n", "ic_ref_source"} <= full_cols


@requires_synth()
def test_a_failing_external_reference_source_is_soft_and_the_batch_continues(tmp_path):
    """A reference source that does not exist on disk at all registers a forepass
    error and NaNs its dependent's columns -- the batch itself must still complete
    (the whole point of the soft-by-default policy)."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path))
    s.processing.references.append(ReferenceEntry(
        file="synth_case_A.csv", ic=BreathRef(file="does_not_exist.csv", breaths=[1])))
    s.validate()

    result = run_batch(s)

    assert ("does_not_exist.csv", None) in result.reference_errors
    a = result.files["synth_case_A.csv"]
    assert a.error is None                            # soft: the file itself still succeeds
    assert a.average_row["vol_ic_ref"].isna().iloc[0]
    assert a.notices

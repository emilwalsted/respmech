"""``core.analysis.references`` (M-34): resolution order and ``check_links`` cautions.

Pure, Qt-free — no file I/O, no ``core.compute``. See that module's own docstring for
the exact resolution order and caution kinds this pins.
"""
import pytest

from respmech.core.analysis.references import REFERENCE_SLOTS, check_links, resolve_reference
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

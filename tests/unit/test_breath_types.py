"""BreathTypeEntry (M-19): a typed breath behaves exactly like an excluded one for
every existing consumer (check_breaths/calculateaveragebreaths/build_breath_table/
plots._breaths all filter on ``ignored`` alone, unchanged by this ticket), plus carries
a ``kind`` and, in ``results.build_processed_data``, an opt-in ``Breathkind`` column.
"""
import os
from collections import OrderedDict

import numpy as np
import pytest

from respmech.core import compute, results
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.settings import BreathTypeEntry, ExcludeEntry, Settings, SettingsError
from _helpers import requires_synth, synth_settings

FS = 200


def _base_settings():
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.flow = 2
    s.input.channels.volume = 3
    return s


# -- Settings.validate() -------------------------------------------------------------

def test_kind_must_be_a_known_value():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=1, kind="bogus"))
    with pytest.raises(SettingsError, match=r"breath_types\[0\]\.kind must be one of"):
        s.validate()


def test_breath_must_be_a_positive_integer():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=0, kind="ic"))
    with pytest.raises(SettingsError, match=r"breath_types\[0\]\.breath must be a positive integer"):
        s.validate()


def test_breath_must_be_an_integer_not_a_float():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=2.5, kind="ic"))
    with pytest.raises(SettingsError, match=r"breath_types\[0\]\.breath must be a positive integer"):
        s.validate()


def test_breath_must_be_positive_not_negative():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=-3, kind="ic"))
    with pytest.raises(SettingsError, match=r"breath_types\[0\]\.breath must be a positive integer"):
        s.validate()


def test_breath_true_the_bool_is_rejected_not_treated_as_1():
    """`isinstance(True, int)` is True in Python, so the guard must reject a bool
    explicitly rather than silently accept it as breath number 1."""
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=True, kind="ic"))
    with pytest.raises(SettingsError, match=r"breath_types\[0\]\.breath must be a positive integer"):
        s.validate()


def test_a_malformed_entry_that_is_not_even_a_breathtypeentry_is_rejected_cleanly():
    """Self-review finding: `_coerce` only builds a real `BreathTypeEntry` from a
    dict-shaped TOML table element; a hand-edited `breath_types = [1, 2]` (numbers, not
    tables) leaves the raw value in the list untouched. Without an explicit guard,
    `validate()`'s own new loop is the first code to read `.kind` off it and would raise
    a raw `AttributeError` instead of the `SettingsError` every caller assumes."""
    s = _base_settings()
    s.processing.breath_types.append(1)             # simulates the malformed-TOML shape
    with pytest.raises(SettingsError, match=r"breath_types\[0\] must be a table"):
        s.validate()


def test_a_valid_breath_type_entry_passes():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=4, kind="ic"))
    s.validate()                                                       # must not raise


def test_duplicate_breath_type_entries_for_the_same_breath_are_rejected():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=3, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=3, kind="fvc"))
    with pytest.raises(SettingsError, match=r"breath 3 of x\.txt is typed more than once"):
        s.validate()


def test_the_same_breath_number_in_two_different_files_is_not_a_duplicate():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="a.txt", breath=3, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="b.txt", breath=3, kind="fvc"))
    s.validate()                                                       # must not raise


def test_a_breath_both_typed_and_excluded_is_rejected():
    s = _base_settings()
    s.processing.exclude_breaths.append(ExcludeEntry(file="x.txt", breaths=[5]))
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=5, kind="ic"))
    with pytest.raises(SettingsError, match=r"breath 5 of x\.txt is both typed and excluded"):
        s.validate()


def test_breath_over_1_is_impossible_under_whole_file_segmentation():
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.emg = [2]
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "whole_file"
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=2, kind="rest"))
    with pytest.raises(SettingsError,
                        match=r"breath 2 > 1 is impossible under whole_file segmentation"):
        s.validate()


def test_breath_1_is_allowed_under_whole_file_segmentation():
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.emg = [2]
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "whole_file"
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=1, kind="rest"))
    s.validate()                                                       # must not raise


def test_breath_over_1_is_fine_under_separators_segmentation():
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.emg = [2]
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "separators"
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=3, kind="rest"))
    s.validate()                                                       # must not raise


# -- core._legacy_ns.to_legacy_ns: the union rule -------------------------------------

def test_typed_breaths_are_unioned_into_excludebreaths_for_a_flow_bearing_set():
    s = _base_settings()
    s.processing.exclude_breaths.append(ExcludeEntry(file="x.txt", breaths=[2]))
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=4, kind="ic"))
    ns = to_legacy_ns(s)
    assert dict(ns.processing.mechanics.excludebreaths)["x.txt"] == [2, 4]


def test_two_typed_breaths_in_the_same_file_merge_into_one_excludebreaths_entry():
    """The ticket's own stated risk: two entries for the same file must not silently
    lose one to a naive last-writer-wins merge."""
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=3, kind="ic"))
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=7, kind="fvc"))
    ns = to_legacy_ns(s)
    entries = ns.processing.mechanics.excludebreaths
    assert [f for f, _b in entries].count("x.txt") == 1
    assert dict(entries)["x.txt"] == [3, 7]


def test_emg_only_set_unions_only_rest_typed_breaths_not_other_kinds():
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.emg = [2]
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "separators"
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=1, kind="rest"))
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=2, kind="other"))
    ns = to_legacy_ns(s)
    assert dict(ns.processing.mechanics.excludebreaths)["x.txt"] == [1]


def test_flow_bearing_set_unions_every_kind_not_only_rest():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=6, kind="max_insp"))
    ns = to_legacy_ns(s)
    assert dict(ns.processing.mechanics.excludebreaths)["x.txt"] == [6]


def test_breathtypes_passthrough_groups_by_file_as_triples():
    s = _base_settings()
    s.processing.breath_types.append(
        BreathTypeEntry(file="x.txt", breath=4, kind="ic", t_onset_s=12.5))
    ns = to_legacy_ns(s)
    assert dict(ns.processing.mechanics.breathtypes)["x.txt"] == [[4, "ic", 12.5]]


# -- core.compute.breathkinds ---------------------------------------------------------

def test_breathkinds_mirrors_ignorebreaths_shape():
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="x.txt", breath=4, kind="ic"))
    ns = to_legacy_ns(s)
    assert compute.breathkinds("x.txt", ns) == {4: "ic"}
    assert compute.breathkinds("nope.txt", ns) == {}


def test_breathkinds_is_empty_with_no_typed_breaths():
    ns = to_legacy_ns(_base_settings())
    assert compute.breathkinds("x.txt", ns) == {}


# -- end-to-end through the segmenterers: kind lands on the breath dict --------------

def _sine_flow(seconds=10, fs=FS, hz=0.5):
    t = np.arange(0, seconds, 1 / fs)
    flow = np.sin(2 * np.pi * hz * t)
    return t, flow


def test_a_typed_breath_is_ignored_and_carries_its_kind_via_flow_segmentation():
    t, flow = _sine_flow()
    n = len(flow)
    zeros = np.zeros(n)
    s = _base_settings()
    s.processing.segmentation.buffer = 10          # small vs. the sine's own period
    s.processing.breath_types.append(BreathTypeEntry(file="rec.csv", breath=2, kind="ic"))
    ns = to_legacy_ns(s)
    breaths = compute.separateintobreathsbyflow(
        "rec.csv", t, flow, zeros, zeros, zeros, zeros, np.zeros((n, 0)), np.zeros((n, 0)), ns)
    assert len(breaths) >= 3
    assert breaths[2]["ignored"] is True
    assert breaths[2]["kind"] == "ic"
    assert breaths[1]["kind"] is None
    assert breaths[1]["ignored"] is False


def test_a_typed_breath_is_ignored_and_carries_its_kind_via_volume_segmentation():
    t, flow = _sine_flow()
    n = len(flow)
    volume = np.cumsum(flow) / FS
    zeros = np.zeros(n)
    s = _base_settings()
    s.processing.breath_types.append(BreathTypeEntry(file="rec.csv", breath=2, kind="fvc"))
    ns = to_legacy_ns(s)
    breaths = compute.separateintobreathsbyvolume(
        "rec.csv", t, flow, volume, zeros, zeros, zeros, np.zeros((n, 0)), np.zeros((n, 0)), ns)
    assert len(breaths) >= 3
    assert breaths[2]["ignored"] is True
    assert breaths[2]["kind"] == "fvc"
    assert breaths[1]["kind"] is None


# -- results.build_processed_data: the opt-in Breathkind column ----------------------

def _fake_breath(kind, ignored):
    return OrderedDict([
        ("time", np.array([0.0, 0.1, 0.2, 0.3])),
        ("flow", np.array([0.0, 0.1, 0.2, 0.3])),
        ("volume", np.array([0.0, 0.1, 0.2, 0.3])),
        ("poes", np.array([])),
        ("pgas", np.array([])),
        ("pdi", np.array([])),
        ("ignored", ignored),
        ("kind", kind),
        ("emgcols", []),
    ])


def test_build_processed_data_has_no_breathkind_column_without_any_typed_breath():
    ns = to_legacy_ns(_base_settings())
    ns.output.data.includeignoredbreaths = True
    breaths = OrderedDict({1: _fake_breath(kind=None, ignored=False)})
    result = results.build_processed_data(breaths, ns)
    assert "Breathkind" not in result.columns


def test_build_processed_data_omits_breathkind_when_the_only_typed_breath_is_never_emitted():
    """Self-review finding: with the default `include_ignored_breaths=False`, a typed
    breath on a flow-bearing set is ALSO `ignored=True` (unioned with exclude_breaths),
    so it is never written to this table at all. `has_typed` must be judged over the
    breaths that will actually appear, not every breath in the file, or the column
    would be added present-but-blank on every row -- worse than not adding it."""
    ns = to_legacy_ns(_base_settings())
    assert ns.output.data.includeignoredbreaths is False           # the default
    breaths = OrderedDict({
        1: _fake_breath(kind=None, ignored=False),
        2: _fake_breath(kind="ic", ignored=True),
    })
    result = results.build_processed_data(breaths, ns)
    assert "Breathkind" not in result.columns


def test_build_processed_data_includes_breathkind_for_an_emg_only_typed_breath_kept_visible():
    """The EMG-only union rule leaves a non-`rest` typed breath `ignored=False` (M-19's
    own asymmetric rule), so it IS emitted even under the default settings, and the
    column must reflect that."""
    ns = to_legacy_ns(_base_settings())
    breaths = OrderedDict({1: _fake_breath(kind="max_insp", ignored=False)})
    result = results.build_processed_data(breaths, ns)
    assert "Breathkind" in result.columns
    assert set(result["Breathkind"].unique()) == {"max_insp"}


def test_build_processed_data_adds_breathkind_column_when_a_breath_is_typed():
    ns = to_legacy_ns(_base_settings())
    ns.output.data.includeignoredbreaths = True
    breaths = OrderedDict({1: _fake_breath(kind="ic", ignored=True)})
    result = results.build_processed_data(breaths, ns)
    assert "Breathkind" in result.columns
    assert set(result["Breathkind"].unique()) == {"ic"}


def test_build_processed_data_breathkind_is_blank_not_nan_for_an_untyped_breath_in_a_typed_file():
    ns = to_legacy_ns(_base_settings())
    ns.output.data.includeignoredbreaths = True
    breaths = OrderedDict({
        1: _fake_breath(kind=None, ignored=False),
        2: _fake_breath(kind="ic", ignored=True),
    })
    result = results.build_processed_data(breaths, ns)
    assert "Breathkind" in result.columns
    by_breath = dict(result.groupby("Breathno")["Breathkind"].first())
    assert by_breath[1] == ""
    assert by_breath[2] == "ic"
    # a blank string must never have been routed through dropna() as NaN
    assert not result["Breathkind"].isna().any()


# -- M-27: BreathTypeEntry renumbering when manual separators are edited --------------
# (the Qt screen-level "_set_separators actually rewrites the entry" behaviour is
# covered end to end in test_separators.py; this is the pure, Qt-free half — the same
# remap_segment_number rule applied by hand to a BreathTypeEntry, matching how
# _SegmentsMixin._set_separators itself uses it.)

def test_a_typed_breath_is_renumbered_by_remap_segment_number_like_any_other_segment():
    from respmech.core.analysis.segments import remap_segment_number

    s = _base_settings()
    s.processing.breath_types.append(
        BreathTypeEntry(file="x.txt", breath=3, kind="rest", t_onset_s=2.0))
    old_bounds = [0.0, 1.0, 2.0, 3.5]           # 3 separators (segments 1..4)
    new_bounds = [0.0, 1.0, 1.5, 2.0, 3.5]      # a new separator inserted at 1.5

    entry = s.processing.breath_types[0]
    entry.breath = remap_segment_number(old_bounds, new_bounds, entry.breath)
    assert entry.breath == 4                    # follows the insertion, like ExcludeEntry


# --------------------------------------------------------------------------------- #
# M-30 — reference-only files (every breath in a file is typed, none tidal)
# --------------------------------------------------------------------------------- #

@requires_synth()
def test_a_file_with_every_breath_typed_runs_as_reference_only(tmp_path):
    """M-30's own acceptance criterion: a batch with a file where ALL breaths are
    typed IC runs without error, and the rest of the batch (a normal tidal file) is
    unaffected — its average table gets no row (and no NaN row) from the
    reference-only file."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path))
    # synth_case_B.csv has 6 breaths (1..6) -- type every one of them, so it has NO
    # tidal breathing at all; synth_case_A.csv (8 breaths) is left entirely alone.
    for n in range(1, 7):
        s.processing.breath_types.append(
            BreathTypeEntry(file="synth_case_B.csv", breath=n, kind="ic"))
    s.validate()
    result = run_batch(s)

    assert not result.failed_files
    fr_a = result.ok_files["synth_case_A.csv"]
    assert fr_a.role == "tidal"
    assert fr_a.breaths_table is not None and len(fr_a.breaths_table) > 0

    fr_b = result.ok_files["synth_case_B.csv"]
    assert fr_b.error is None
    assert fr_b.role == "reference"
    assert fr_b.breaths_table is None
    assert fr_b.average_row is None
    assert set(fr_b.manoeuvres) == set(range(1, 7))
    assert fr_b.manoeuvres_table is not None and len(fr_b.manoeuvres_table) == 6
    assert any("no tidal breaths" in n for n in fr_b.notices)


@requires_synth()
def test_reference_only_file_gets_no_spurious_cardiac_gated_nan_notice(tmp_path):
    """Self-review finding: the cardiac-gated peak EMG NaN notice is about the
    per-breath mechanics loop (compute.calculatemechanics/compute_segment_emg), which
    a reference-only file never reaches (it is entirely skipped for such a file) --
    with robust_peak enabled but remove_ecg off (guaranteeing 'no R-peaks available'),
    a reference-only file must not report a NaN gating failure for a computation it
    never attempted, while an ordinary tidal file in the SAME batch still does."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path))
    s.processing.emg.robust_peak.enabled = True
    s.processing.emg.remove_ecg = False           # -> "no R-peaks available" for every file
    for n in range(1, 7):
        s.processing.breath_types.append(
            BreathTypeEntry(file="synth_case_B.csv", breath=n, kind="ic"))
    s.validate()
    result = run_batch(s)

    fr_a = result.ok_files["synth_case_A.csv"]
    assert any("cardiac-gated peak EMG reported as NaN" in n for n in fr_a.notices)

    fr_b = result.ok_files["synth_case_B.csv"]
    assert fr_b.role == "reference"
    assert not any("cardiac-gated" in n for n in fr_b.notices)

    # the reference-only file contributes no row (not even a NaN one) to the average
    assert list(result.average_table["file"]) == ["synth_case_A.csv"]


@requires_synth()
def test_reference_only_file_writes_a_workbook_with_the_manoeuvres_table_as_data(tmp_path):
    """write_batch (M-30): a reference-only file's breathdata workbook has no tidal
    Data sheet to write, so the Manoeuvres table takes its place, with a Provenance
    row explaining why -- rather than crashing on a None DataFrame or writing an
    empty one with no explanation."""
    import openpyxl
    import pandas as pd

    from respmech.core.io.writers import write_batch
    from respmech.core.pipeline import run_batch

    s = synth_settings(str(tmp_path))
    for n in range(1, 7):
        s.processing.breath_types.append(
            BreathTypeEntry(file="synth_case_B.csv", breath=n, kind="ic"))
    s.validate()
    result = run_batch(s)
    write_batch(result, s, str(tmp_path))

    path = os.path.join(str(tmp_path), "data", "synth_case_B.csv.breathdata.xlsx")
    assert os.path.isfile(path)
    wb = openpyxl.load_workbook(path)
    assert "Manoeuvres" not in wb.sheetnames        # not duplicated as an extra sheet too
    data = pd.read_excel(path, sheet_name="Data")
    assert len(data) == 6
    assert "vol_ic" in data.columns and "kind" in data.columns

    prov = pd.read_excel(path, sheet_name="Provenance")
    note_row = prov.loc[prov["Key"] == "INCOMPLETE", "Value"]
    assert len(note_row) == 1
    assert "reference manoeuvres only" in note_row.iloc[0]

    # a run-report.txt is still written and names the file's typed breakdown
    report = open(os.path.join(str(tmp_path), "run-report.txt"), encoding="utf-8").read()
    assert "synth_case_B.csv" in report
    assert "6 typed: IC #1,#2,#3,#4,#5,#6" in report


@requires_synth()
def test_cli_dry_run_reports_reference_manoeuvres_not_zero_breaths(tmp_path, capsys):
    """CLI dry run (M-30): a reference-only file must not read as a bare, misleading
    '0 breaths' -- it says how many reference manoeuvres it actually has."""
    from respmech.cli.__main__ import main as cli_main
    from respmech.settingsio.toml_io import save_toml

    s = synth_settings(str(tmp_path))
    for n in range(1, 7):
        s.processing.breath_types.append(
            BreathTypeEntry(file="synth_case_B.csv", breath=n, kind="ic"))
    s.validate()
    toml = tmp_path / "s.toml"
    save_toml(s, toml)
    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "synth_case_B.csv: 0 tidal breaths, 6 reference manoeuvres" in out
    assert "synth_case_A.csv: 8 breaths" in out
    s.validate()                                # still a well-formed entry afterwards

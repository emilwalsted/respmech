"""``core.analysis.normalisation``: the inspiratory pressures and EMG expressed against a
maximal manoeuvre, and the tension-time indices built on it.

The pinned case is hand-built, so every number is analytic: a tidal breath with an
end-expiratory Poes baseline of -2 cmH2O and a minimum of -12 has a swing of 10; against a
maximal reference of 20 that is 50 %; with a pressure-time integral of 2.0 cmH2O.s, a
breath of 2 s and Ti = 0.5 s, the mean inspiratory pressure is 4 (20 %) and
``tt_es = 2 / (2 * 20) = 0.05``, all to 1e-9.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from respmech.core.analysis import manoeuvres as manoeuvreslib
from respmech.core.analysis import normalisation as norm
from respmech.core.pipeline import BatchResult, FileResult
from respmech.core.settings import (
    BreathRef, BreathTypeEntry, ExcludeEntry, GroupReferenceEntry, ReferenceEntry, Settings)

FS = 200.0
N_INSP = 100                     # Ti = 100 / 200 = 0.5 s


def _settings(*, poes=True, pdi=True, emg=False, enabled=True, ref_breath=9, ref_kind="max_insp"):
    s = Settings()
    s.input.format.sampling_frequency = int(FS)
    s.input.channels.flow = 1
    s.input.channels.volume = 2
    if poes:
        s.input.channels.poes = 3
    if pdi:
        s.input.channels.pgas = 4
        s.input.channels.pdi = 5
    if emg:
        s.input.channels.emg = [6]
    s.processing.pressure.normalization.enabled = enabled
    if ref_breath is not None:
        s.processing.breath_types.append(
            BreathTypeEntry(file="A.csv", breath=ref_breath, kind=ref_kind))
    return s


def _breath():
    poes = np.full(N_INSP, -12.0)
    poes[:10] = -2.0                              # baseline window (0.05 s * 200 Hz = 10 samples)
    pdi = np.full(N_INSP, 30.0)
    pdi[:10] = 5.0                                # baseline 5, max 30 -> swing 25
    return {"number": 1, "inspiration": {"flow": np.full(N_INSP, -1.0), "poes": poes, "pdi": pdi}}


def _table(**over):
    row = {"breath_no": 1, "ti": N_INSP / FS, "ttot": 2.0, "poes_mininsp": -12.0,
           "int_oesinsp": 2.0, "pdi_maxinsp": 30.0, "int_pdiinsp": 3.0,
           "rms_insp_max": 0.02, "bf": 10.0}
    row.update(over)
    return pd.DataFrame([row])


def _max_row(kind="max_insp", **values):
    row = {"kind": kind, "quality": []}
    row.update({"poes_max_ref": 20.0, "pdi_max_ref": 50.0, "rms_max_ref": 0.04,
                "rms_max_ref_col_6": 0.04})
    row.update(values)
    return row


def _result(settings, *, manoeuvres=None, table=None, breaths=None, references=None):
    fr = FileResult(file="A.csv", role="tidal",
                    breaths_table=table if table is not None else _table(),
                    average_row=pd.DataFrame([{"file": "A.csv"}]),
                    breaths=breaths if breaths is not None else {1: _breath()},
                    manoeuvres=manoeuvres if manoeuvres is not None else {9: _max_row()})
    br = BatchResult()
    br.files = {"A.csv": fr}
    br.references = references or {}
    return br, fr


# --------------------------------------------------------------------------- #
# the formulas
# --------------------------------------------------------------------------- #

def test_tension_time_is_the_pressure_integral_over_ttot_times_the_reference():
    assert norm.tension_time(2.0, 2.0, 20.0) == pytest.approx(0.05, abs=1e-9)
    # identical to (Pmean / Pmax) * (Ti / Ttot), the published form
    ti, ttot, integral, ref = 0.5, 2.0, 2.0, 20.0
    assert norm.tension_time(integral, ttot, ref) == pytest.approx(
        (integral / ti / ref) * (ti / ttot), abs=1e-12)


@pytest.mark.parametrize("ttot, ref", [(0.0, 20.0), (2.0, 0.0), (2.0, -1.0),
                                       (2.0, float("nan")), (float("nan"), 20.0)])
def test_tension_time_is_nan_without_a_positive_denominator(ttot, ref):
    assert np.isnan(norm.tension_time(2.0, ttot, ref))


def test_attach_reproduces_the_analytic_case_to_1e_9():
    s = _settings()
    result, fr = _result(s)
    norm.attach(result, s)
    row = fr.pressure_normalised.iloc[0]
    assert row["poes_max_ref"] == pytest.approx(20.0, abs=1e-9)
    assert row["poes_insp_swing"] == pytest.approx(10.0, abs=1e-9)
    assert row["poes_insp_swing_pct"] == pytest.approx(50.0, abs=1e-9)
    assert row["poes_mean_insp"] == pytest.approx(4.0, abs=1e-9)
    assert row["poes_mean_insp_pct"] == pytest.approx(20.0, abs=1e-9)
    assert row["tt_es"] == pytest.approx(0.05, abs=1e-9)
    assert row["pdi_max_ref"] == pytest.approx(50.0, abs=1e-9)
    assert row["pdi_insp_swing"] == pytest.approx(25.0, abs=1e-9)
    assert row["pdi_insp_swing_pct"] == pytest.approx(50.0, abs=1e-9)
    assert row["pdi_mean_insp"] == pytest.approx(6.0, abs=1e-9)
    assert row["pdi_mean_insp_pct"] == pytest.approx(12.0, abs=1e-9)
    assert row["tt_di"] == pytest.approx(3.0 / (2.0 * 50.0), abs=1e-9)


def test_nrdi_is_the_emg_percentage_of_maximum_times_the_breath_rate():
    s = _settings(emg=True)
    result, fr = _result(s)
    norm.attach(result, s)
    row = fr.pressure_normalised.iloc[0]
    assert row["rms_insp_max_pct"] == pytest.approx(50.0, abs=1e-9)      # 0.02 / 0.04
    assert row["nrdi"] == pytest.approx(500.0, abs=1e-9)                # 50 % * 10 min^-1


def test_the_baseline_window_follows_the_rate_the_breath_was_cut_at():
    """A resampled run cuts breaths at another rate than the file's own; the window is
    recovered from ti = len / fs, not read from the input format."""
    s = _settings()
    s.input.format.sampling_frequency = 1000                  # the file's own rate, unused
    result, fr = _result(s)
    norm.attach(result, s)
    assert fr.pressure_normalised.iloc[0]["poes_insp_swing"] == pytest.approx(10.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# off, absent reference, partial reference
# --------------------------------------------------------------------------- #

def test_off_by_default_does_nothing_and_says_so_in_the_plan():
    s = _settings(enabled=False)
    result, fr = _result(s)
    norm.attach(result, s)
    assert fr.pressure_normalised is None
    assert "max_insp" not in fr.references_used
    assert result.analysis_plan["pressure_normalisation"]["enabled"] is False
    assert Settings().processing.pressure.normalization.enabled is False


def test_without_a_reference_every_normalised_value_is_nan_and_one_notice_says_why():
    s = _settings(ref_breath=None)
    result, fr = _result(s, manoeuvres={})
    norm.attach(result, s)
    row = fr.pressure_normalised.iloc[0]
    for col in ("poes_max_ref", "poes_insp_swing_pct", "poes_mean_insp_pct", "tt_es",
                "pdi_max_ref", "pdi_insp_swing_pct", "pdi_mean_insp_pct", "tt_di"):
        assert np.isnan(row[col]), col
    # the un-normalised parts are still real numbers
    assert row["poes_insp_swing"] == pytest.approx(10.0, abs=1e-9)
    assert row["poes_mean_insp"] == pytest.approx(4.0, abs=1e-9)
    notices = [n for n in fr.notices if "no maximal-effort reference resolves" in n]
    assert len(notices) == 1
    assert "max_insp" not in fr.references_used
    assert result.analysis_plan["pressure_normalisation"]["unresolved"] == ["A.csv"]


def test_the_column_set_does_not_depend_on_whether_a_reference_resolved():
    s = _settings(emg=True)
    with_ref, fr1 = _result(s)
    without, fr2 = _result(_settings(emg=True, ref_breath=None), manoeuvres={})
    norm.attach(with_ref, s)
    norm.attach(without, _settings(emg=True, ref_breath=None))
    assert list(fr1.pressure_normalised.columns) == list(fr2.pressure_normalised.columns)


def test_a_reference_without_the_pdi_channel_leaves_only_the_pdi_columns_nan():
    s = _settings()
    row = _max_row()
    del row["pdi_max_ref"]
    result, fr = _result(s, manoeuvres={9: row})
    norm.attach(result, s)
    out = fr.pressure_normalised.iloc[0]
    assert out["tt_es"] == pytest.approx(0.05, abs=1e-9)
    assert np.isnan(out["tt_di"]) and np.isnan(out["pdi_insp_swing_pct"])
    assert any("pdi_max_ref" in n for n in fr.notices)


def test_a_non_positive_reference_gives_nan_not_infinity():
    s = _settings()
    result, fr = _result(s, manoeuvres={9: _max_row(poes_max_ref=0.0)})
    norm.attach(result, s)
    out = fr.pressure_normalised.iloc[0]
    assert np.isnan(out["tt_es"]) and np.isnan(out["poes_insp_swing_pct"])
    assert np.isfinite(out["tt_di"])
    assert any("poes_max_ref" in n for n in fr.notices)


def test_only_signal_set_columns_are_written():
    s = _settings(pdi=False)
    result, fr = _result(s, manoeuvres={9: _max_row()})
    norm.attach(result, s)
    cols = set(fr.pressure_normalised.columns)
    assert "tt_es" in cols
    assert not {c for c in cols if c.startswith("pdi") or c == "tt_di"}
    assert not {"nrdi", "rms_insp_max_pct"} & cols


def test_no_pressure_and_no_emg_produces_no_table():
    s = _settings(poes=False, pdi=False)
    result, fr = _result(s)
    norm.attach(result, s)
    assert fr.pressure_normalised is None


def test_a_reference_only_or_failed_file_is_skipped():
    s = _settings()
    result, fr = _result(s)
    fr.role = "reference"
    norm.attach(result, s)
    assert fr.pressure_normalised is None
    result, fr = _result(s)
    fr.error = "boom"
    norm.attach(result, s)
    assert fr.pressure_normalised is None


def test_an_unexpected_error_in_one_file_is_a_notice_not_a_crash():
    s = _settings()
    result, fr = _result(s, breaths={1: {"number": 1}})            # no "inspiration" key
    norm.attach(result, s)
    assert fr.pressure_normalised is None
    assert any("could not be computed" in n for n in fr.notices)


# --------------------------------------------------------------------------- #
# the reference itself
# --------------------------------------------------------------------------- #

def test_repeated_maximal_efforts_use_the_largest_value_of_each_quantity():
    s = _settings(ref_breath=None)
    for b in (7, 8, 9):
        s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=b, kind="max_insp"))
    manoeuvres = {7: _max_row(poes_max_ref=18.0, pdi_max_ref=55.0),
                  8: _max_row(poes_max_ref=22.0, pdi_max_ref=45.0),
                  9: _max_row(poes_max_ref=20.0, pdi_max_ref=50.0)}
    result, fr = _result(s, manoeuvres=manoeuvres)
    norm.attach(result, s)
    used = fr.references_used["max_insp"]
    assert used["poes_max_ref"] == 22.0 and used["pdi_max_ref"] == 55.0
    assert used["breaths"] == [7, 8, 9] and used["n"] == 3 and used["kinds"] == ["max_insp"]


def test_the_kind_behind_the_reference_is_recorded_and_mixing_kinds_is_flagged():
    s = _settings(ref_breath=None, poes=True, pdi=True)
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=8, kind="sniff"))
    s.processing.breath_types.append(BreathTypeEntry(file="A.csv", breath=9, kind="max_insp"))
    result, fr = _result(s, manoeuvres={8: _max_row(kind="sniff", pdi_max_ref=60.0), 9: _max_row()})
    norm.attach(result, s)
    assert fr.references_used["max_insp"]["kinds"] == ["max_insp", "sniff"]
    assert any("mixes max_insp and sniff" in n for n in fr.notices)
    result, fr = _result(_settings(ref_kind="sniff"), manoeuvres={9: _max_row(kind="sniff")})
    norm.attach(result, _settings(ref_kind="sniff"))
    assert fr.references_used["max_insp"]["kinds"] == ["sniff"]


def test_a_reference_breath_of_the_wrong_kind_is_not_used():
    s = _settings(ref_breath=None)
    s.processing.references.append(
        ReferenceEntry(file="A.csv", max_insp=BreathRef(file="A.csv", breaths=[9])))
    result, fr = _result(s, manoeuvres={9: {"kind": "ic", "quality": [], "vol_ic": 3.0}})
    norm.attach(result, s)
    assert np.isnan(fr.pressure_normalised.iloc[0]["tt_es"])
    assert any("is typed 'ic', not max_insp or sniff" in n for n in fr.notices)
    assert any("resolved to no usable breath" in n for n in fr.notices)


def test_an_explicit_reference_beats_the_group_default_beats_the_files_own_typed_breath():
    s = _settings()                                    # own typed breath 9 (poes_max_ref 20)
    s.processing.reference_defaults.append(GroupReferenceEntry(
        group="A.csv", max_insp=BreathRef(file="G.csv", breaths=[1])))
    ext = {"G.csv": {1: _max_row(poes_max_ref=30.0)}, "E.csv": {1: _max_row(poes_max_ref=40.0)}}
    result, fr = _result(s, references=ext)
    norm.attach(result, s)
    assert fr.references_used["max_insp"]["source"] == "G.csv"        # group default wins
    s.processing.references.append(
        ReferenceEntry(file="A.csv", max_insp=BreathRef(file="E.csv", breaths=[1])))
    result, fr = _result(s, references=ext)
    norm.attach(result, s)
    assert fr.references_used["max_insp"]["source"] == "E.csv"        # explicit wins
    assert fr.pressure_normalised.iloc[0]["poes_max_ref"] == 40.0


def test_a_source_outside_the_batch_is_read_from_the_forepass():
    s = _settings(ref_breath=None)
    s.processing.references.append(
        ReferenceEntry(file="A.csv", max_insp=BreathRef(file="X.csv", breaths=[2])))
    result, fr = _result(s, manoeuvres={}, references={"X.csv": {2: _max_row(poes_max_ref=25.0)}})
    norm.attach(result, s)
    assert fr.pressure_normalised.iloc[0]["poes_max_ref"] == 25.0
    assert fr.pressure_normalised.iloc[0]["poes_insp_swing_pct"] == pytest.approx(40.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# the EMG reference for the EMG-normalised sheet
# --------------------------------------------------------------------------- #

def _emg_batch(*, typed=True):
    s = Settings()
    s.processing.emg.normalization = "per_file_max"
    s.processing.emg.normalization_reference_file = "REF.csv"
    if typed:
        s.processing.breath_types.append(BreathTypeEntry(file="REF.csv", breath=3, kind="max_insp"))
    table = pd.DataFrame({"breath_no": [1, 2], "rms_col_6": [1.0, 2.0], "rms_col_7": [3.0, 4.0],
                          "rms_max": [3.0, 4.0], "rms_insp_max": [2.5, 3.5]})
    tidal = FileResult(file="T.csv", role="tidal", breaths_table=table,
                       average_row=pd.DataFrame([{"file": "T.csv"}]))
    ref = FileResult(file="REF.csv", role="reference", breaths_table=None, manoeuvres={
        3: {"kind": "max_insp", "quality": [], "rms_max_ref": 10.0,
            "rms_max_ref_col_6": 4.0, "rms_max_ref_col_7": 10.0}})
    br = BatchResult()
    br.files = {"T.csv": tidal, "REF.csv": ref}
    return s, br


def test_a_named_reference_file_with_a_typed_maximal_breath_is_read_at_that_breath():
    from respmech.core.summary import normalize_emg_table, resolve_emg_reference

    s, br = _emg_batch()
    values, notice = resolve_emg_reference(br, s)
    assert notice is None
    assert values["rms_col_6"] == 4.0 and values["rms_col_7"] == 10.0     # each channel's own peak
    assert values["rms_max"] == 10.0 and values["rms_insp_max"] == 10.0   # the largest channel
    out = normalize_emg_table(br.files["T.csv"].breaths_table, s, reference_values=values)
    assert out["rms_col_6_pct"].tolist() == pytest.approx([25.0, 50.0])
    assert out["rms_insp_max_pct"].tolist() == pytest.approx([25.0, 35.0])


def test_without_a_typed_maximal_breath_the_old_reference_path_is_unchanged():
    from respmech.core.summary import resolve_emg_reference

    s, br = _emg_batch(typed=False)
    values, notice = resolve_emg_reference(br, s)
    assert values is None                                    # reference-only file: no table
    assert notice and "no breath table" in notice
    # and a tidal reference file keeps reading its own per-column maximum
    br.files["REF.csv"] = FileResult(
        file="REF.csv", role="tidal", average_row=pd.DataFrame([{"file": "REF.csv"}]),
        breaths_table=pd.DataFrame({"breath_no": [1], "rms_col_6": [9.0]}))
    values, notice = resolve_emg_reference(br, s)
    assert values == {"rms_col_6": 9.0} and notice is None


# --------------------------------------------------------------------------- #
# what the extractor now provides, and the units
# --------------------------------------------------------------------------- #

def test_max_effort_extraction_adds_each_channels_own_peak_beside_the_largest():
    from types import SimpleNamespace
    from respmech.core.analysis.signals import Capabilities

    fs = 1000.0
    t = np.arange(int(fs)) / fs
    emg = np.column_stack([1.0 * np.sin(2 * np.pi * 50 * t), 3.0 * np.sin(2 * np.pi * 50 * t)])
    breath = {"number": 1, "kind": "max_insp", "emgcols": emg,
              "inspiration": {"poes": np.array([0.0, -5.0]), "pdi": np.array([0.0, 5.0])}}
    s = SimpleNamespace(
        input=SimpleNamespace(format=SimpleNamespace(samplingfrequency=fs),
                              data=SimpleNamespace(columns_emg=[6, 7])),
        processing=SimpleNamespace(emg=SimpleNamespace(rms_s=0.05)))
    caps = Capabilities(flow=True, volume=True, poes=False, pgas=False, pdi=False, emg=True,
                        entropy=False, declared=frozenset({"flow", "emg"}), mode="custom")
    out = manoeuvreslib.max_effort_from_breath(breath, caps, s)
    # the channels are the same waveform at 1x and 3x, so their peaks keep that ratio exactly
    assert out["rms_max_ref_col_7"] == pytest.approx(3.0 * out["rms_max_ref_col_6"], rel=1e-12)
    assert out["rms_max_ref_col_6"] > 0
    assert out["rms_max_ref"] == out["rms_max_ref_col_7"]


def test_units_of_every_normalised_column():
    from _helpers import assert_units
    assert_units({
        "poes_max_ref": "cmH₂O", "pdi_max_ref": "cmH₂O",
        "poes_insp_swing": "cmH₂O", "pdi_insp_swing": "cmH₂O",
        "poes_mean_insp": "cmH₂O", "pdi_mean_insp": "cmH₂O",
        "poes_insp_swing_pct": "%", "pdi_insp_swing_pct": "%",
        "poes_mean_insp_pct": "%", "pdi_mean_insp_pct": "%", "rms_insp_max_pct": "%",
        "tt_es": "—", "tt_di": "—",
        "rms_max_ref": "a.u.", "rms_max_ref_col_6": "a.u.",
        "nrdi": "",
    })


def test_every_column_attach_writes_has_a_unit_or_is_deliberately_blank():
    from respmech.core import quantities
    s = _settings(emg=True)
    result, fr = _result(s)
    norm.attach(result, s)
    for col in fr.pressure_normalised.columns:
        if col in ("breath_no", "nrdi"):
            continue
        assert quantities.unit_for(col), col


def test_the_setting_survives_a_toml_round_trip(tmp_path):
    from respmech.settingsio.toml_io import load_toml, save_toml
    s = Settings()
    s.processing.pressure.normalization.enabled = True
    path = str(tmp_path / "a.toml")
    save_toml(s, path)
    assert load_toml(path).processing.pressure.normalization.enabled is True


# --------------------------------------------------------------------------- #
# through the real pipeline and writer
# --------------------------------------------------------------------------- #

from _helpers import requires_synth, synth_settings                        # noqa: E402


def _synth(tmp_path, *, enabled):
    s = synth_settings(str(tmp_path))
    s.processing.breath_types.append(BreathTypeEntry(file="synth_case_A.csv", breath=4, kind="max_insp"))
    s.processing.pressure.normalization.enabled = enabled
    s.validate()
    return s


@requires_synth()
def test_end_to_end_tt_es_matches_the_tables_own_numbers(tmp_path):
    from respmech.core.pipeline import run_batch

    result = run_batch(_synth(tmp_path, enabled=True))
    fr = result.files["synth_case_A.csv"]
    ref = fr.manoeuvres[4]["poes_max_ref"]
    pn, table = fr.pressure_normalised, fr.breaths_table
    assert len(pn) == len(table) == 7                        # the typed breath is not tidal
    assert pn["breath_no"].tolist() == table["breath_no"].tolist()
    expected = table["int_oesinsp"].to_numpy() / (table["ttot"].to_numpy() * ref)
    assert pn["tt_es"].to_numpy() == pytest.approx(expected, abs=1e-9)
    assert fr.references_used["max_insp"]["poes_max_ref"] == pytest.approx(ref)
    # B has no reference of its own: NaN, one notice, the batch carries on
    b = result.files["synth_case_B.csv"]
    assert b.error is None and b.pressure_normalised["tt_es"].isna().all()
    assert sum("no maximal-effort reference" in n for n in b.notices) == 1


@requires_synth()
def test_switching_it_on_changes_no_existing_table(tmp_path):
    """Golden neutrality at the level the golden suite pins: the per-breath and average
    tables are byte-identical with the analysis on and off."""
    from respmech.core.pipeline import run_batch

    off = run_batch(_synth(tmp_path / "off", enabled=False))
    on = run_batch(_synth(tmp_path / "on", enabled=True))
    for name in off.files:
        pd.testing.assert_frame_equal(off.files[name].breaths_table, on.files[name].breaths_table)
        pd.testing.assert_frame_equal(off.files[name].average_row, on.files[name].average_row)
    pd.testing.assert_frame_equal(off.average_table, on.average_table)
    assert all(fr.pressure_normalised is None for fr in off.files.values())


@requires_synth()
def test_a_subset_run_gives_the_same_normalised_rows_as_a_full_run(tmp_path):
    from respmech.core.pipeline import run_batch

    s = _synth(tmp_path, enabled=True)
    s.processing.references.append(ReferenceEntry(
        file="synth_case_B.csv", max_insp=BreathRef(file="synth_case_A.csv", breaths=[4])))
    s.validate()
    full = run_batch(s)
    subset = run_batch(s, only_files=["synth_case_B.csv"])
    pd.testing.assert_frame_equal(full.files["synth_case_B.csv"].pressure_normalised,
                                  subset.files["synth_case_B.csv"].pressure_normalised)
    assert full.files["synth_case_B.csv"].pressure_normalised["tt_es"].notna().all()


@requires_synth()
def test_the_workbook_gets_a_pressure_normalised_sheet_and_a_provenance_row(tmp_path):
    from respmech.core.io.writers import write_batch
    from respmech.core.pipeline import run_batch

    s = _synth(tmp_path, enabled=True)
    result = run_batch(s)
    out = tmp_path / "out"
    out.mkdir()
    write_batch(result, s, str(out))
    book = next(p for p in out.rglob("synth_case_A.csv.breathdata.xlsx"))
    sheets = pd.read_excel(book, sheet_name=None)
    assert "Pressure normalised" in sheets
    assert list(sheets["Pressure normalised"].columns)[:2] == ["breath_no", "poes_max_ref"]
    prov = dict(zip(sheets["Provenance"]["Key"], sheets["Provenance"]["Value"]))
    assert "synth_case_A.csv #4 (max_insp)" in prov["Pressure normalisation"]
    assert "poes_max_ref" in prov["Pressure normalisation"]
    report = (out / "run-report.txt").read_text(encoding="utf-8")
    assert "PRESSURE NORMALISATION" in report
    b = next(p for p in out.rglob("synth_case_B.csv.breathdata.xlsx"))
    prov_b = pd.read_excel(b, sheet_name="Provenance")
    assert "no maximal-effort reference resolved" in " ".join(
        prov_b.loc[prov_b["Key"] == "Pressure normalisation", "Value"].astype(str))


@requires_synth()
def test_with_it_off_no_sheet_no_provenance_row_no_report_block(tmp_path):
    from respmech.core.io.writers import write_batch
    from respmech.core.pipeline import run_batch

    s = _synth(tmp_path, enabled=False)
    result = run_batch(s)
    out = tmp_path / "out"
    out.mkdir()
    write_batch(result, s, str(out))
    book = next(p for p in out.rglob("synth_case_A.csv.breathdata.xlsx"))
    sheets = pd.read_excel(book, sheet_name=None)
    assert "Pressure normalised" not in sheets
    assert "Pressure normalisation" not in set(sheets["Provenance"]["Key"])
    assert "PRESSURE NORMALISATION" not in (out / "run-report.txt").read_text(encoding="utf-8")


@requires_synth()
def test_the_emg_normalised_sheet_reads_a_named_files_typed_maximal_breath(tmp_path):
    from respmech.core.pipeline import run_batch
    from respmech.core.summary import normalize_emg_table, resolve_emg_reference

    s = _synth(tmp_path, enabled=False)
    s.processing.emg.normalization = "per_file_max"
    s.processing.emg.normalization_reference_file = "synth_case_A.csv"
    result = run_batch(s)
    values, notice = resolve_emg_reference(result, s)
    assert notice is None
    row = result.files["synth_case_A.csv"].manoeuvres[4]
    assert values["rms_col_2"] == pytest.approx(row["rms_max_ref_col_2"])
    assert values["rms_max"] == pytest.approx(row["rms_max_ref"])
    a = normalize_emg_table(result.files["synth_case_B.csv"].breaths_table, s, reference_values=values)
    b_table = result.files["synth_case_B.csv"].breaths_table
    assert a["rms_col_2_pct"].to_numpy() == pytest.approx(
        100.0 * b_table["rms_col_2"].to_numpy() / row["rms_max_ref_col_2"])


def test_a_mean_column_uses_the_mean_channel_peak_and_a_channel_without_a_peak_is_nan():
    s, br = _emg_batch()
    br.files["T.csv"].breaths_table["rms_mean"] = [1.0, 2.0]
    br.files["T.csv"].breaths_table["rms_col_9"] = [1.0, 2.0]
    values = norm.emg_reference_from_max_effort(br, s, "REF.csv")
    assert values["rms_mean"] == pytest.approx(7.0)            # mean of 4 and 10
    assert np.isnan(values["rms_col_9"])                       # no peak for channel 9


def test_per_file_mean_mode_keeps_the_earlier_reference_path():
    from respmech.core.summary import resolve_emg_reference
    s, br = _emg_batch()
    s.processing.emg.normalization = "per_file_mean"
    values, notice = resolve_emg_reference(br, s)
    assert values is None and "no breath table" in notice


def test_a_sniff_swing_is_taken_over_the_whole_typed_breath():
    from types import SimpleNamespace
    from respmech.core.analysis.signals import Capabilities
    caps = Capabilities(flow=True, volume=True, poes=True, pgas=False, pdi=False, emg=False,
                        entropy=False, declared=frozenset({"flow", "poes"}), mode="custom")
    breath = {"inspiration": {"poes": np.array([-2.0, -4.0])},
              "expiration": {"poes": np.array([-20.0, -3.0])}}
    s = SimpleNamespace()
    assert manoeuvreslib.max_effort_from_breath(breath, caps, s, "sniff")["poes_max_ref"] == pytest.approx(18.0)
    assert manoeuvreslib.max_effort_from_breath(breath, caps, s, "max_insp")["poes_max_ref"] == pytest.approx(2.0)

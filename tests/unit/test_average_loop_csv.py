"""``output.data.save_pv_average_csv``: the averaged Campbell / flow-volume loop as numbers.

The table must be the very arrays the average-Campbell figure is drawn from
(``volumeavg``/``poesavg``), written only on request, with the cohort file only for a
batch of more than one recording."""
import os
from datetime import datetime

import numpy as np
import pandas as pd

from tests.unit._helpers import requires_synth, synth_settings

LOOP = "synth_case_A.csv – Campbell (average).csv"
COHORT = "All files – Campbell (average).csv"


def _read(path):
    return pd.read_csv(path, float_precision="round_trip")


def _run_and_write(tmp_path, settings, **kw):
    from respmech.core.io.writers import write_batch
    from respmech.core.pipeline import run_batch
    result = run_batch(settings, only_files=kw.pop("only_files", None))
    written = write_batch(result, settings, str(tmp_path), when=datetime(2026, 10, 26), **kw)
    return result, written


def test_off_by_default():
    from respmech.core.settings import DataOutput
    assert DataOutput().save_pv_average_csv is False


@requires_synth()
def test_nothing_new_is_written_when_the_key_is_false(tmp_path):
    s = synth_settings(tmp_path)
    _result, written = _run_and_write(tmp_path, s)
    assert not [p for p in written if p.endswith("Campbell (average).csv")]
    assert not list((tmp_path / "data").glob("*Campbell (average).csv"))


@requires_synth()
def test_csv_matches_the_averaged_arrays_at_full_precision(tmp_path):
    s = synth_settings(tmp_path)
    s.output.data.save_pv_average_csv = True
    result, written = _run_and_write(tmp_path, s)
    path = tmp_path / "data" / LOOP
    assert str(path) in written
    df = _read(path)
    assert list(df.columns) == ["Phase", "Point", "Volume", "Poes"]
    b = next(b for b in result.ok_files["synth_case_A.csv"].breaths.values() if not b["ignored"])
    n_in, n_ex = len(b["inspiration"]["volumeavg"]), len(b["expiration"]["volumeavg"])
    assert n_in == n_ex == s.processing.wob.avg_resampling_obs
    assert (df["Phase"] == "inspiration").sum() == n_in
    assert (df["Phase"] == "expiration").sum() == n_ex
    # Point counts 1..N within each phase
    assert list(df["Point"][df["Phase"] == "inspiration"]) == list(range(1, n_in + 1))
    assert list(df["Point"][df["Phase"] == "expiration"]) == list(range(1, n_ex + 1))
    # exactly the arrays the figure plots, nothing recomputed
    assert np.array_equal(df["Volume"].to_numpy(), np.asarray(b["volumeavg"], dtype=float))
    assert np.array_equal(df["Poes"].to_numpy(), np.asarray(b["poesavg"], dtype=float))


@requires_synth()
def test_cohort_file_only_for_more_than_one_recording(tmp_path):
    s = synth_settings(tmp_path)
    s.output.data.save_pv_average_csv = True
    one = tmp_path / "one"; one.mkdir()
    _run_and_write(one, s, only_files=["synth_case_A.csv"])
    assert (one / "data" / LOOP).is_file()
    assert not (one / "data" / COHORT).exists()

    many = tmp_path / "many"; many.mkdir()
    result, _ = _run_and_write(many, s)
    assert len(result.ok_files) > 1
    cohort = _read(many / "data" / COHORT)
    assert list(cohort.columns) == ["File", "Phase", "Point", "Volume", "Poes"]
    assert set(cohort["File"]) == set(result.ok_files)
    for fname in result.ok_files:
        per_file = _read(many / "data" / f"{fname} – Campbell (average).csv")
        part = cohort[cohort["File"] == fname].drop(columns="File").reset_index(drop=True)
        pd.testing.assert_frame_equal(part, per_file)


@requires_synth()
def test_a_subset_rerun_never_rebuilds_the_cohort_file(tmp_path):
    s = synth_settings(tmp_path)
    s.output.data.save_pv_average_csv = True
    result, _ = _run_and_write(tmp_path, s, cohort_outputs=False)
    assert len(result.ok_files) > 1
    assert (tmp_path / "data" / LOOP).is_file()
    assert not (tmp_path / "data" / COHORT).exists()


@requires_synth()
def test_a_signal_set_without_poes_gets_no_poes_column(tmp_path):
    s = synth_settings(tmp_path, channels={"poes": None, "pgas": None, "pdi": None,
                                           "emg": [], "entropy": []})
    s.output.data.save_pv_average_csv = True
    result, _ = _run_and_write(tmp_path, s)
    df = _read(tmp_path / "data" / LOOP)
    assert list(df.columns) == ["Phase", "Point", "Volume"]
    b = next(b for b in result.ok_files["synth_case_A.csv"].breaths.values() if not b["ignored"])
    assert np.array_equal(df["Volume"].to_numpy(), np.asarray(b["volumeavg"], dtype=float))


def test_no_kept_breath_means_no_table():
    from respmech.core.io.loop_csv import average_loop_table
    assert average_loop_table([]) is None
    assert average_loop_table(None) is None
    assert average_loop_table([{"ignored": True, "inspiration": {}, "expiration": {}}]) is None


def test_the_first_kept_breath_supplies_the_average():
    from respmech.core.io.loop_csv import average_loop_table
    ins, exp = np.array([0.1, 0.2]), np.array([0.2, 0.1])
    kept = {"ignored": False, "inspiration": {"volumeavg": ins, "poesavg": -ins},
            "expiration": {"volumeavg": exp, "poesavg": -exp}}
    skipped = {"ignored": True, "inspiration": {"volumeavg": ins * 9}, "expiration": {"volumeavg": exp * 9}}
    t = average_loop_table([skipped, kept])
    assert list(t["Phase"]) == ["inspiration"] * 2 + ["expiration"] * 2
    assert list(t["Point"]) == [1, 2, 1, 2]
    assert list(t["Volume"]) == [0.1, 0.2, 0.2, 0.1]
    assert list(t["Poes"]) == [-0.1, -0.2, -0.2, -0.1]


@requires_synth()
def test_plan_lists_the_tables_only_when_asked_for(tmp_path):
    from respmech.core.io.plan import plan_outputs
    names = ["synth_case_A.csv", "synth_case_B.csv"]
    s = synth_settings(tmp_path)
    off = plan_outputs(s, names)
    assert not [g for g in off.groups if "Campbell loop" in g.category]
    s.output.data.save_pv_average_csv = True
    on = plan_outputs(s, names)
    paths = [p for g in on.groups for p in g.paths]
    assert "data/synth_case_A.csv – Campbell (average).csv" in paths
    assert "data/All files – Campbell (average).csv" in paths
    subset = plan_outputs(s, names, cohort_outputs=False)
    assert "data/All files – Campbell (average).csv" not in [p for g in subset.groups for p in g.paths]


@requires_synth()
def test_toml_round_trip_keeps_the_key(tmp_path):
    from respmech.settingsio.toml_io import load_toml, save_toml
    s = synth_settings(tmp_path)
    s.output.data.save_pv_average_csv = True
    p = tmp_path / "a.toml"
    save_toml(s, str(p))
    assert load_toml(str(p)).output.data.save_pv_average_csv is True


@requires_synth()
def test_preview_export_writes_the_same_table(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from respmech.core.pipeline import run_batch
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = synth_settings(str(tmp_path))
    result = run_batch(s, only_files=["synth_case_A.csv"])
    breaths = result.ok_files["synth_case_A.csv"].breaths
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._draw_campbell_or_loop(breaths)
    out = tmp_path / "preview.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "CSV table (*.csv)")))
    pv._export_campbell()
    pv_df = _read(out)

    s.output.data.save_pv_average_csv = True
    from respmech.core.io.writers import write_batch
    write_batch(result, s, str(tmp_path / "run"), when=datetime(2026, 10, 26))
    pd.testing.assert_frame_equal(pv_df, _read(tmp_path / "run" / "data" / LOOP))
    win.close()


@requires_synth()
def test_row_count_follows_avg_resampling_obs(tmp_path):
    s = synth_settings(tmp_path, avgresamplingobs=50)
    s.output.data.save_pv_average_csv = True
    _run_and_write(tmp_path, s, only_files=["synth_case_A.csv"])
    df = _read(tmp_path / "data" / LOOP)
    assert df.groupby("Phase").size().to_dict() == {"expiration": 50, "inspiration": 50}


def test_cohort_needs_two_recordings_with_an_average_breath():
    from respmech.core.io.loop_csv import average_loop_table, cohort_loop_table
    ins = np.array([0.1, 0.2])
    breath = {"ignored": False, "inspiration": {"volumeavg": ins}, "expiration": {"volumeavg": ins}}
    t = average_loop_table([breath])
    assert list(t.columns) == ["Phase", "Point", "Volume"]
    assert cohort_loop_table({"a": t, "b": None}).shape[0] == t.shape[0]
    assert cohort_loop_table({"a": None}) is None


@requires_synth()
def test_preview_export_branches(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from respmech.core.pipeline import run_batch
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = synth_settings(str(tmp_path), channels={"poes": None, "pgas": None, "pdi": None,
                                                "emg": [], "entropy": []})
    breaths = run_batch(s, only_files=["synth_case_A.csv"]).ok_files["synth_case_A.csv"].breaths
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._draw_campbell_or_loop(breaths)

    # CSV filter chosen but no extension typed: ".csv" is appended; no Poes column
    typed = tmp_path / "noext"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(typed), "CSV table (*.csv)")))
    pv._export_campbell()
    assert list(_read(str(typed) + ".csv").columns) == ["Phase", "Point", "Volume"]

    # extension typed, filter left on PNG: the extension decides
    ext = tmp_path / "x.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(ext), "PNG image (*.png)")))
    pv._export_campbell()
    assert ext.is_file()

    # nothing to export once the cache is forgotten: no file, a status line instead
    pv._forget_campbell()
    none = tmp_path / "none.csv"
    pv._export_campbell_csv(str(none), "Flow-volume loop")
    assert not none.exists()
    win.close()

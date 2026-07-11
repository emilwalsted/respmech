"""Wave 2 of the full-app review: transparency & deliverables.

P3 the result-changing signal transforms are surfaced on the Setup screen and bound to
the core settings (drift correction defaults ON, so it must be visible); P4 an "Output —
what to save" checklist with a live "you will get" read-out; P5 a dry-run pre-flight plan
(what will be read / written) plus an overwrite guard before a real run clobbers results;
P7 every run drops a reloadable ``analysis-used.toml`` manifest + a human ``run-report.txt``
and the Run screen offers "Open output folder".
"""
import os
from datetime import datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.ui.state import AppState  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")), reason="synthetic input absent")


@pytest.fixture(scope="module")
def qapp():
    from respmech.ui import theme
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    yield app


def _legacy_settings(out):
    from respmech.settingsio.migrate import migrate_dict
    legacy = {"input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4],
                                 "columns_entropy": [10, 11, 12]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": False, "remove_noise": False}},
              "output": {"outputfolder": str(out), "data": {}}}
    s, _ = migrate_dict(legacy)
    s.input.folder = INPUT
    s.output.folder = str(out)
    return s


# --------------------------------------------------------------------------- #
# P3 — the hidden, result-changing transforms are surfaced + bound
# --------------------------------------------------------------------------- #
def test_signal_transforms_surfaced_and_bound(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    # drift correction is ON by default — it silently changes results, so it must show it
    assert sc.correct_drift.isChecked()
    sc.correct_drift.setChecked(False)
    sc.inverse_flow.setChecked(True)
    sc.correct_trend.setChecked(True)
    sc.resample.setChecked(True); sc.resample_hz.setValue(500)
    sc._on_field_changed()
    v = sc.state.settings.processing.volume
    smp = sc.state.settings.processing.sampling
    assert v.correct_drift is False and v.inverse_flow is True and v.correct_trend is True
    assert smp.resample is True and smp.resample_to_frequency == 500
    # dependent fields only enable when their toggle is on
    assert sc.trend_method.isEnabled() and sc.resample_hz.isEnabled()
    sc.correct_trend.setChecked(False); sc.resample.setChecked(False); sc._on_field_changed()
    assert not sc.trend_method.isEnabled() and not sc.resample_hz.isEnabled()
    win.close()


# --------------------------------------------------------------------------- #
# P4 — "what to save" checklist + live read-out
# --------------------------------------------------------------------------- #
def test_output_checklist_binds_and_previews(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.save_processed.setChecked(True)
    sc.save_raw_fig.setChecked(True)
    sc._on_field_changed()
    d = sc.state.settings.output.data
    dg = sc.state.settings.output.diagnostics
    assert d.save_processed is True and dg.save_raw is True
    txt = sc.save_preview.text().lower()
    assert "processed csv" in txt          # the read-out reflects the ticked boxes
    win.close()


# --------------------------------------------------------------------------- #
# P5 — dry-run pre-flight plan + overwrite guard
# --------------------------------------------------------------------------- #
def test_dry_run_plan_lists_inputs_and_planned_outputs(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_legacy_settings(tmp_path))); rn = win.run_screen
    rn._append_plan(write=False)
    log = rn.log.toPlainText()
    assert "DRY RUN" in log and "nothing will be written" in log.lower()
    assert log.count("synth_case_") >= 2                      # both inputs listed
    assert "analysis-used.toml" in log and "run-report.txt" in log
    win.close()


def test_planned_outputs_follow_save_flags(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.validation import matching_files
    s = _legacy_settings(tmp_path)
    win = MainWindow(AppState(s)); rn = win.run_screen
    files = matching_files(s.input.folder, s.input.files)
    s.output.data.save_processed = False
    assert not any("Processed data" in n for n in rn._planned_outputs(files))
    s.output.data.save_processed = True
    assert any("Processed data" in n for n in rn._planned_outputs(files))
    win.close()


def test_overwrite_guard_detects_prior_results(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_legacy_settings(tmp_path))); rn = win.run_screen
    assert rn._existing_output() is None                     # empty folder — nothing to clobber
    (tmp_path / "analysis-used.toml").write_text("x")
    data = tmp_path / "data"; data.mkdir()
    (data / "Average breathdata.xlsx").write_text("y")
    ex = rn._existing_output()
    assert ex is not None and ex[1] == 2                      # 2 prior files detected
    win.close()


def test_open_output_button_starts_disabled(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_legacy_settings(tmp_path))); rn = win.run_screen
    assert not rn.btn_open.isEnabled()                        # enabled only after a real write
    win.close()


# --------------------------------------------------------------------------- #
# P7 — provenance manifest + run report
# --------------------------------------------------------------------------- #
def test_run_report_accounts_for_excluded_and_failed(tmp_path):
    """The run report must show per-file breath accounting (excluded → used) and any
    failures, without needing a full batch — driven by lightweight fakes."""
    from types import SimpleNamespace
    from respmech.core.io.writers import _write_run_report

    ok = SimpleNamespace(breaths={1: {"ignored": False}, 2: {"ignored": True},
                                  3: {"ignored": False}}, error=None)
    bad = SimpleNamespace(breaths=None, error="boom while loading")
    result = SimpleNamespace(
        ok_files={"good.csv": ok}, failed_files={"bad.csv": bad})
    s = _legacy_settings(tmp_path)
    path = _write_run_report(result, s, str(tmp_path), ["data/x.xlsx"], datetime(2026, 7, 11))
    report = open(path).read()
    assert "1 processed, 1 failed" in report
    assert "3 breaths (1 excluded → 2 used)" in report
    assert "[FAIL] bad.csv   ERROR: boom while loading" in report


def test_write_batch_emits_reloadable_manifest(tmp_path):
    from respmech.core.pipeline import run_batch
    from respmech.core.io.writers import write_batch
    from respmech.settingsio.toml_io import load_toml
    s = _legacy_settings(tmp_path)
    s.output.data.save_processed = True
    result = run_batch(s)
    written = write_batch(result, s, str(tmp_path), when=datetime(2026, 7, 11, 14, 0, 0))
    manifest = os.path.join(str(tmp_path), "analysis-used.toml")
    report = os.path.join(str(tmp_path), "run-report.txt")
    assert os.path.isfile(manifest) and os.path.isfile(report)
    assert manifest in written and report in written
    # the manifest is a real, reloadable settings file (round-trips the sampling rate)
    reloaded = load_toml(manifest)
    assert reloaded.input.format.sampling_frequency == 1000
    assert "2 processed, 0 failed" in open(report).read()

"""Headless tests for the error-surfacing UI (per-panel error cards + copyable
trace dialog, the batch error-log window), the startup splash, and maximise-on-start.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.settingsio.migrate import migrate_dict  # noqa: E402
from respmech.ui.state import AppState  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
    reason="synthetic input not present")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _settings(outdir):
    legacy = {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": 1000},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": [10, 11, 12]}},
        "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                     "avgresamplingobs": 300},
                       "emg": {"remove_ecg": True, "remove_noise": False}},
        "output": {"outputfolder": outdir,
                   "data": {"saveaveragedata": True, "savebreathbybreathdata": True}},
    }
    s, _ = migrate_dict(legacy)
    return s


# -- copyable text dialog + error summary -----------------------------------
def test_short_error_extracts_last_line():
    from respmech.ui.dialogs import short_error
    tb = "Traceback (most recent call last):\n  File \"x\", line 1\nValueError: boom"
    assert short_error(tb) == "ValueError: boom"
    assert short_error("") == "unknown error"


def test_text_viewer_dialog_is_copyable(qapp):
    from respmech.ui.dialogs import TextViewerDialog
    dlg = TextViewerDialog("Detail", "line one\nline two")
    assert dlg.text() == "line one\nline two"
    assert dlg.view.isReadOnly()
    dlg._copy()
    assert QApplication.clipboard().text() == "line one\nline two"


# -- splash -----------------------------------------------------------------
def test_build_splash_svg_has_brand_and_version():
    from respmech.ui.splash import build_splash_svg
    svg = build_splash_svg(780, 460, version="9.9.9")
    assert svg.lstrip().startswith("<svg")
    assert "RespMech" in svg and "9.9.9" in svg and "respmech.dk" in svg
    assert "03A9F4" in svg.upper()          # brand azure present
    assert "Human respiratory physiology analysis made easier" in svg   # site tagline


def test_logo_asset_bundled_and_loads(qapp):
    from respmech.ui.splash import _load_logo_pixmap
    pm = _load_logo_pixmap()
    assert pm is not None and not pm.isNull()   # the RM logo ships and loads


def test_make_splash_renders_a_pixmap(qapp):
    from respmech.ui.splash import make_splash
    sp = make_splash(qapp)
    assert sp is not None
    assert not sp.pixmap().isNull()


# -- per-panel error card + copyable trace ----------------------------------
def test_failed_job_shows_panel_error_with_copyable_detail(qapp, tmp_path):
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    # simulate a failed 'mech' job (unstarted QThread -> quit/wait are no-ops)
    job = _Job("mech", pv._tokens["mech"], QThread(), object())
    job.error = "Traceback (most recent call last):\n  ...\nValueError: bad channel"
    pv._jobs["mech"] = job
    pv._on_job_done(job, None)
    # both panels the 'mech' job feeds show the error and are no longer "busy"
    assert "bad channel" in (pv.panel_error("channels") or "")
    assert pv.panel_error("raw")
    assert pv.panel_busy("channels") is False
    # the round info button opens a copyable dialog carrying the full trace
    ov = pv._overlays["channels"]
    ov._open_detail()
    assert ov._detail_dialog is not None
    assert "bad channel" in ov._detail_dialog.text()
    # starting a new job clears the error card
    ov.start("Loading channels…")
    assert ov.error is None and ov.busy is True
    win.close()


def test_synchronous_preview_clears_stale_error_card(qapp, tmp_path):
    """A successful synchronous re-render must dismiss a lingering error card."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(0)
    job = _Job("mech", pv._tokens["mech"], QThread(), object())
    job.error = "ValueError: transient read error"
    pv._jobs["mech"] = job
    pv._on_job_done(job, None)
    assert pv.panel_error("channels")
    pv._preview()                                  # successful sync render
    assert pv.panel_error("channels") is None
    assert pv.panel_error("raw") is None
    win.close()


def test_late_superseded_job_does_not_wipe_error_card(qapp, tmp_path):
    """A stale superseded job finishing late must not erase the current job's
    error card."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._tokens["emg_detail"] = 2
    job_b = _Job("emg_detail", 2, QThread(), object())   # current owner, fails
    job_b.error = "RuntimeError: boom-B"
    pv._jobs["emg_detail"] = job_b
    pv._on_job_done(job_b, None)
    assert "boom-B" in (pv.panel_error("detail") or "")
    job_a = _Job("emg_detail", 1, QThread(), object())   # stale, finishes late
    pv._on_job_done(job_a, {"stale": True})
    assert "boom-B" in (pv.panel_error("detail") or "")  # card preserved
    win.close()


def test_reopening_error_detail_replaces_dialog(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    ov = pv._overlays["channels"]
    ov.show_error("failed", "ValueError: first")
    ov._open_detail(); first = ov._detail_dialog
    ov._open_detail(); second = ov._detail_dialog
    assert first is not None and first is not second   # replaced, not accumulated
    assert not first.isVisible()                        # the old one was closed
    win.close()


# -- batch error-log window (Run screen) ------------------------------------
def test_run_screen_error_report_and_window(qapp, tmp_path):
    from respmech.ui.screens.run_screen import RunScreen
    from respmech.core.pipeline import BatchResult, FileResult
    rs = RunScreen(AppState(_settings(str(tmp_path))))
    result = BatchResult(files={
        "good.csv": FileResult(file="good.csv"),
        "bad.csv": FileResult(file="bad.csv", error="ValueError: broken column 7"),
    })
    report = rs._error_report(result, result.failed_files)
    assert "bad.csv" in report and "broken column 7" in report
    rs._show_error_window(report)
    assert rs._error_dialog is not None
    assert "broken column 7" in rs._error_dialog.text()
    # fatal path (result is None) uses the stored fatal message
    rs._fatal_msg = "RuntimeError: kaboom"
    rep2 = rs._error_report(None, {})
    assert "did not complete" in rep2 and "kaboom" in rep2


# -- maximise on startup ----------------------------------------------------
def test_mainwindow_can_show_maximized(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    win.showMaximized()
    qapp.processEvents()
    assert win.isVisible()
    assert bool(win.windowState() & Qt.WindowMaximized)
    win.close()

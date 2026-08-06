"""Headless tests for the error-surfacing UI (per-panel error cards + copyable
trace dialog, the batch error-log window), the startup splash, and maximise-on-start.
"""
import os

import pytest




from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.ui.state import AppState  # noqa: E402


from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


pytestmark = requires_synth()




def _settings(outdir):
    return synth_settings(outdir, remove_ecg=True, data_out=_DATA_OUT)


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


def test_text_viewer_dialog_collapsed_detail_starts_hidden_behind_details(qapp):
    """Ticket D01: a caller with an already-diagnosed failure leads with the diagnosis
    (``intro``, shown in the normal — not muted — style, since it IS the message) and keeps
    the trace reachable but out of the way, instead of dumping it straight on screen.

    Checks visibility with ``isVisibleTo(dlg)`` rather than ``isVisible()``: the latter also
    asks whether the whole ancestor chain up to the screen is shown, which is False for an
    unshown top-level QDialog regardless of the child's own explicit setVisible() state — a
    property of the widget's OWN visibility flag, not of whether the dialog is on screen."""
    from respmech.ui.dialogs import TextViewerDialog
    dlg = TextViewerDialog("Channel setup — error", "Traceback (most recent call last):\n...",
                           intro="P05_60W.txt: row 4 has 9 values but the first row has 3.",
                           collapsed_detail=True)
    assert dlg.view.isVisibleTo(dlg) is False
    assert dlg.intro_label is not None
    assert dlg.intro_label.property("status") != "muted"
    assert dlg.details_btn is not None
    dlg.details_btn.setChecked(True)
    assert dlg.view.isVisibleTo(dlg) is True
    assert "Traceback" in dlg.text()        # still reachable — nothing was dropped
    dlg.details_btn.setChecked(False)
    assert dlg.view.isVisibleTo(dlg) is False    # toggles back off


def test_text_viewer_dialog_without_collapsed_detail_is_unchanged(qapp):
    """The default (existing) behaviour must be untouched: no Details button, the trace
    visible immediately, and the intro (when given) muted — the previous, plain error
    surface every OTHER caller of open_error_dialog still uses."""
    from respmech.ui.dialogs import TextViewerDialog
    dlg = TextViewerDialog("Open analysis — error", "Traceback...", intro="Open analysis failed.")
    assert dlg.details_btn is None
    assert dlg.view.isVisibleTo(dlg) is True
    assert dlg.intro_label.property("status") == "muted"


def test_text_viewer_dialog_collapsed_detail_without_an_intro_falls_back(qapp):
    """Self-review finding: no current caller passes ``collapsed_detail=True`` without an
    ``intro`` (there would be nothing to lead with), but a dialog that silently hid its
    only content behind a button labelled 'Details' would be a foot-gun for the next one —
    it must fall back to the ordinary, uncollapsed layout instead."""
    from respmech.ui.dialogs import TextViewerDialog
    dlg = TextViewerDialog("Channel setup — error", "Traceback...", collapsed_detail=True)
    assert dlg.details_btn is None
    assert dlg.view.isVisibleTo(dlg) is True


# -- splash -----------------------------------------------------------------
def test_build_splash_svg_has_brand_and_version():
    from respmech.ui.splash import build_splash_svg
    svg = build_splash_svg(780, 460, version="9.9.9")
    assert svg.lstrip().startswith("<svg")
    assert "RespMech" in svg and "9.9.9" in svg and "respmech.dk" in svg
    assert "03A9F4" in svg.upper()          # brand azure present
    assert "Human respiratory physiology analysis made easier" in svg   # site tagline


def test_logo_builds_renders_and_ships(qapp):
    import os
    from respmech.ui import splash
    from respmech.ui.logo import build_logo_svg, logo_pixmap, app_icon
    svg = build_logo_svg(512)
    assert svg.lstrip().startswith("<svg") and "FF9800" in svg.upper()   # orange EMG accent
    assert not logo_pixmap(128).isNull() and not logo_pixmap(32).isNull()
    icon = app_icon()
    assert icon is not None and not icon.isNull()
    # the shareable SVG asset ships in the package
    asset = os.path.join(os.path.dirname(splash.__file__), "assets", "respmech_logo.svg")
    assert os.path.isfile(asset)


def test_splash_uses_generated_logo(qapp):
    from respmech.ui.splash import _load_logo_pixmap
    pm = _load_logo_pixmap()
    assert pm is not None and not pm.isNull()


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


def test_error_card_summary_carries_the_diagnosis(qapp, tmp_path):
    """The error card's own visible summary now carries the exception's diagnosis (A03
    point 5), not only the generic '<job> failed' — previously the diagnosis was reachable
    only via the transient status line or behind the round 'i' info button."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    job = _Job("mech", pv._tokens["mech"], QThread(), object())
    job.error = "Traceback (most recent call last):\n  ...\nValueError: bad channel"
    pv._jobs["mech"] = job
    pv._on_job_done(job, None)
    ov = pv._overlays["channels"]
    assert "valueerror: bad channel" in ov._err_label.text().lower()
    win.close()


def test_synchronous_preview_clears_stale_error_card(qapp, tmp_path):
    """A successful synchronous re-render must dismiss a lingering error card."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_index(0)
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

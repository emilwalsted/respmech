"""The Mechanics test run on a full signal set that names EMG explicitly.

The test run behind Preview & QC's per-breath table and Campbell diagram drops the EMG
channels from its snapshot (the EMG work is shown on its own tabs). A complete analysis
that declares ``analysis.signals`` explicitly, as the built-in sample does, still listed
"emg" after that, and ``Settings.validate()`` refused the snapshot ("input.channels.emg
must name at least one column when 'emg' is in analysis.signals"): the two panels showed a
raw "Test run failed" card on the very first thing a new user opens.
"""
import os

from PySide6.QtCore import QEventLoop, QElapsedTimer, QTimer

from respmech.core.sample import write_sample_recording, build_sample_settings


def _pump_until(predicate, timeout=60.0):
    if predicate():
        return True
    loop = QEventLoop()
    clock = QElapsedTimer(); clock.start()
    state = {"ok": False}
    timer = QTimer(); timer.setInterval(10)

    def _tick():
        if predicate():
            state["ok"] = True
            loop.quit()
        elif clock.elapsed() > timeout * 1000:
            loop.quit()
    timer.timeout.connect(_tick)
    timer.start()
    loop.exec()
    timer.stop()
    return state["ok"] or predicate()


def test_the_mechanics_test_run_works_when_analysis_signals_names_emg(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    desc = write_sample_recording(os.path.join(str(tmp_path), "input"))
    s = build_sample_settings(desc, os.path.join(str(tmp_path), "output"))
    assert "emg" in s.analysis.signals and s.input.channels.emg   # the shape under test

    win = MainWindow(AppState(s))
    win.settings_screen.from_state()
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(desc["filename"])
    pv._schedule("batch")
    assert "batch" in pv._jobs
    assert _pump_until(lambda: not pv._jobs and not pv._draining)
    assert pv.panel_error("table") is None
    assert pv.panel_error("campbell") is None
    # the live analysis is untouched: only the test run's own snapshot drops EMG
    assert "emg" in s.analysis.signals and s.input.channels.emg == desc["mapping"]["emg"]
    win.close()

"""Headless tests for the interactive Preview & tuning graph features (A/B/C).

Constructed under QT_QPA_PLATFORM=offscreen; the mouse interaction itself is not
exercised — the drawing helpers and the toggle/apply/staging methods are called
directly, which is what must be testable without a display.
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


# -- Feature A: breath overlays + include/exclude ---------------------------
def test_breath_overlays_and_toggle(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(0)
    pv._preview()
    name = pv.file_combo.currentText()
    # overlays were drawn: one region per breath per channel plot + spans + labels
    assert pv._breath_spans, "no breath spans recorded"
    assert len(pv._channel_plots) == 5
    a_breath = next(iter(pv._breath_spans))
    assert len(pv._breath_regions[a_breath]) == 5   # one region on each channel plot
    # hit-testing maps a time inside the breath to its number
    t0, t1 = pv._breath_spans[a_breath]
    assert pv._breath_at((t0 + t1) / 2.0) == a_breath
    # toggle -> excluded, writes ExcludeEntry with the 1-based breath number
    assert pv._toggle_breath(a_breath) is True
    excl = win.state.settings.processing.exclude_breaths
    entry = next(e for e in excl if e.file == name)
    assert a_breath in entry.breaths
    assert "excluded" in pv.status.text()
    # toggle again -> included, entry removed
    assert pv._toggle_breath(a_breath) is False
    assert all(e.file != name for e in win.state.settings.processing.exclude_breaths)
    assert "included" in pv.status.text()


# -- Feature B: select a noise-profile region on the graph ------------------
def test_use_region_as_noise_writes_settings_and_mirrors(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv, sc = win.preview_screen, win.settings_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(0)
    name = pv.file_combo.currentText()
    pv._ensure_noise_region()
    pv._noise_region.setRegion((1.0, 5.0))
    out = pv._use_region_as_noise()
    assert out == (1.0, 5.0)
    n = win.state.settings.processing.emg.noise
    assert n.reference_file == name
    assert n.reference_intervals == [[1.0, 5.0]]
    assert n.use_expiration is False
    # the signal mirrored the choice into the Settings screen widgets
    assert sc.noise_ref.text() == name
    assert sc.noise_use_exp.isChecked() is False


# -- Feature C: preview EMG conditioning (staging + render) -----------------
def test_stage_emg_channel_and_render(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_channel
    s = _settings(str(tmp_path))
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.use_expiration = False
    s.processing.emg.noise.reference_intervals = [[1.0, 5.0]]
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    path = os.path.join(INPUT, "synth_case_A.csv")
    data = stage_emg_channel(s, path, 0)
    for k in ("t", "raw", "ecg", "noise", "freqs", "psd_raw", "psd_ecg", "psd_noise"):
        assert len(data[k]) > 0
    assert data["noise_applied"] is True
    assert len(data["raw"]) == len(data["ecg"]) == len(data["noise"])
    # render both stacks; PSD canvas always has exactly one axis
    pv.render_emg_time(data)
    pv.render_emg_psd(data)
    assert len(pv.emg_psd_canvas.figure.axes) == 1
    # the dedicated PSD canvas is independent of the fidelity canvas (locked contract)
    assert pv.emg_psd_canvas is not pv.fidelity_canvas


def test_emg_worker_runs_synchronously(qapp, tmp_path):
    from respmech.ui.workers import EmgConditioningWorker
    s = _settings(str(tmp_path))
    path = os.path.join(INPUT, "synth_case_A.csv")
    out = {}
    w = EmgConditioningWorker(s, path, 1)
    w.finished.connect(lambda d: out.setdefault("d", d))
    w.run()  # synchronous — verifies the worker logic without a thread
    assert out["d"] is not None
    assert out["d"]["channel"] == 1


def test_emg_preview_disabled_without_emg_channels(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    s.input.channels.emg = []
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(0)
    pv._update_actions()
    assert pv.btn_emg_preview.isEnabled() is False
    assert pv.emg_channel.count() == 0

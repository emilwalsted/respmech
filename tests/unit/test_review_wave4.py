"""Wave 4 of the full-app review: the Preview as an analysis & QC surface.

P12 the Campbell diagram gains the elastic-recoil line + shaded inspiratory work +
a WOB read-out; P13 the EMG detail overlays the RMS envelope it quantifies; P16 a
persistent batch-QC overview flags non-physiological breaths; P17 a crosshair read-out
+ Campbell figure export; P22 a zero-reference baseline on every mechanics channel.
All Preview-screen (pyqtgraph/matplotlib) work — no core, no golden exposure.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
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


def _settings():
    from respmech.settingsio.migrate import migrate_dict
    legacy = {"input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4],
                                 "columns_entropy": [10, 11, 12]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": False, "remove_noise": False}},
              "output": {"outputfolder": "", "data": {}}}
    s, _ = migrate_dict(legacy)
    s.input.folder = INPUT
    return s


def _render_mech(pv, s):
    from respmech.ui.workers import stage_mechanics_preview
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))


# --------------------------------------------------------------------------- #
# P12 — Campbell shows the work of breathing
# --------------------------------------------------------------------------- #
def test_campbell_has_recoil_line_and_shaded_work(qapp):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = _settings()
    win = MainWindow(AppState(s)); pv = win.preview_screen
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    pv._draw_campbell(fr.breaths)
    ax = pv.campbell.figure.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"average breath", "elastic recoil", "inspiratory work"} <= set(labels)
    assert len(ax.collections) >= 1                      # the shaded work polygon
    assert any("WOB" in t.get_text() for t in ax.texts)  # the read-out
    win.close()


# --------------------------------------------------------------------------- #
# P13 — RMS envelope
# --------------------------------------------------------------------------- #
def test_rms_envelope_helper():
    from respmech.ui.screens.preview_screen import _rms_envelope
    x = np.array([3.0, 4.0, 0.0, 0.0])
    env = _rms_envelope(x, 2)
    assert env.shape == x.shape and np.isfinite(env).all()
    assert env[1] == pytest.approx(np.sqrt((9 + 16) / 2))   # window over samples 0..1
    # a constant signal's RMS envelope equals its amplitude
    assert _rms_envelope(np.full(50, 2.0), 5) == pytest.approx(np.full(50, 2.0))


def test_emg_detail_overlays_rms_envelope(qapp):
    from respmech.ui.main_window import MainWindow
    s = _settings()
    win = MainWindow(AppState(s)); pv = win.preview_screen
    t = np.linspace(0, 1, 1000)
    raw = np.sin(2 * np.pi * 50 * t)
    pv.render_emg_time({"t": t, "raw": raw, "ecg": raw * 0.9, "noise": raw * 0.8,
                        "noise_applied": True})
    names = [it.name() for it in pv.emg_plots.getPlotItem().listDataItems() if it.name()]
    assert "RMS envelope" in names
    win.close()


# --------------------------------------------------------------------------- #
# P16 — batch QC overview
# --------------------------------------------------------------------------- #
def test_qc_overview_ok_and_warn(qapp):
    from types import SimpleNamespace
    import pandas as pd
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = _settings()
    win = MainWindow(AppState(s)); pv = win.preview_screen
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    pv._update_qc_overview(fr)
    assert pv.qc_overview.property("status") == "ok"
    assert "breaths used" in pv.qc_overview.text() and "no flags" in pv.qc_overview.text()
    # a non-physiological breath (Vt ≤ 0) trips the warn flag
    bad = SimpleNamespace(breaths=fr.breaths,
                          breaths_table=pd.DataFrame({"vt": [0.5, -0.1], "wobtotal": [1.0, 2.0]}))
    pv._update_qc_overview(bad)
    assert pv.qc_overview.property("status") == "warn"
    assert "vt≤0" in pv.qc_overview.text()
    win.close()


# --------------------------------------------------------------------------- #
# P17 — crosshair + export
# --------------------------------------------------------------------------- #
def test_crosshair_and_export(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = _settings()
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    assert len(pv._crosshair_lines) == len(pv._channel_plots)
    assert all(not ln.isVisible() for ln in pv._crosshair_lines)   # hidden until hover
    p0 = pv._channel_plots[0]
    pv._on_mech_mouse_moved([QPointF(p0.sceneBoundingRect().center())])
    assert pv._crosshair_lines[0].isVisible()
    assert "t = " in pv.crosshair_label.text()
    # export is gated until a Campbell exists, then writes a real file
    assert not pv.btn_export_fig.isEnabled()
    pv._draw_campbell(run_batch(s).ok_files["synth_case_A.csv"].breaths)
    assert pv.btn_export_fig.isEnabled()
    out = str(tmp_path / "campbell.png")
    pv.campbell.figure.savefig(out)
    assert os.path.getsize(out) > 0
    win.close()


# --------------------------------------------------------------------------- #
# P22 — zero-reference baselines
# --------------------------------------------------------------------------- #
def test_zero_baseline_on_every_channel(qapp):
    from respmech.ui.main_window import MainWindow
    s = _settings()
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    for p in pv._channel_plots:
        horiz = [it for it in p.items
                 if isinstance(it, pg.InfiniteLine) and float(it.angle) % 180 == 0]
        assert horiz, "channel missing its zero baseline"
    win.close()

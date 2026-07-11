"""Preview screen as an analysis & QC surface: the Campbell diagram's recoil line +
shaded work + WOB (P12), the EMG RMS envelope (P13), the batch-QC overview (P16), the
crosshair + Campbell export (P17) and zero-reference baselines (P22). Previously
test_review_wave4.py; the preview tests from waves 1 (P2) and 6 (P27) are appended below."""
import os

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


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
    s = synth_settings("")
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
    s = synth_settings("")
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
    s = synth_settings("")
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
    s = synth_settings("")
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
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    for p in pv._channel_plots:
        horiz = [it for it in p.items
                 if isinstance(it, pg.InfiniteLine) and float(it.angle) % 180 == 0]
        assert horiz, "channel missing its zero baseline"
    win.close()


# ---------------------------------------------------------------------------
# Keyboard file-stepping (P27) — from wave 6
# ---------------------------------------------------------------------------
def test_file_stepping(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); pv = win.preview_screen
    pv.file_combo.clear(); pv.file_combo.addItems(["f1.csv", "f2.csv", "f3.csv"])
    pv.file_combo.setCurrentIndex(0)
    pv._step_file(+1); assert pv.file_combo.currentText() == "f2.csv"
    pv._step_file(+1); pv._step_file(+1)                 # clamps at the end
    assert pv.file_combo.currentIndex() == 2
    pv._step_file(-1); assert pv.file_combo.currentText() == "f2.csv"
    win.close()


# --------------------------------------------------------------------------- #
# P28 — detected-format read-out
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input absent")


# ---------------------------------------------------------------------------
# Live recompute on breath exclusion (P2) — from wave 1
# ---------------------------------------------------------------------------
def test_toggle_breath_requests_recompute(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    assert pv._batch_recompute_pending is True            # the average is being recomputed…
    assert "recomputing" in pv.status.text().lower()      # …not deferred to a manual run
    win.close()

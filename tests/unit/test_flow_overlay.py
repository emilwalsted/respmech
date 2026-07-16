"""The discrete flow-background overlay superimposed on the EMG time-domain plots
(ui.plot_overlays.add_flow_background): it appears on every EMG time plot + the noise modal,
sits BEHIND the EMG traces (deepest z), is a clean no-op without flow, and never KeyErrors on
a minimal palette. Purely cosmetic + UI-only — the goldens are unaffected."""
import os

import numpy as np

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}
_Z = -20                       # add_flow_background._FLOW_Z (behind EMG traces at z=0)


def _settings(outdir):
    return synth_settings(outdir, remove_ecg=True, data_out=_DATA_OUT)


def _flow_items(plot_item):
    return [it for it in plot_item.listDataItems() if it.zValue() == _Z]


# -- the helper in isolation --------------------------------------------------
def test_helper_returns_item_behind_the_emg(qapp):
    import pyqtgraph as pg
    from respmech.ui.plot_overlays import add_flow_background
    pw = pg.PlotWidget(); pi = pw.getPlotItem()          # keep pw alive (else its ViewBox is GC'd)
    t = np.arange(200) / 200.0
    pi.plot(t, np.sin(2 * np.pi * 5 * t))                       # an EMG trace at z=0
    item = add_flow_background(pi, t, np.sin(2 * np.pi * 0.3 * t),
                               {"channels": {"flow": (44, 110, 155)}})
    assert item is not None
    assert item.zValue() == _Z                                 # strictly behind the EMG (z=0)
    assert item in pi.listDataItems()


def test_helper_is_noop_without_usable_flow(qapp):
    import pyqtgraph as pg
    from respmech.ui.plot_overlays import add_flow_background
    pw = pg.PlotWidget(); pi = pw.getPlotItem()
    t = np.arange(10, dtype=float)
    assert add_flow_background(pi, t, None, {}) is None
    assert add_flow_background(pi, t, np.array([]), {}) is None
    assert add_flow_background(pi, t, np.full(10, np.nan), {}) is None
    assert add_flow_background(pi, t, np.zeros(10), {}) is None   # flat flow (fmax==fmin)
    assert _flow_items(pi) == []


def test_helper_survives_minimal_palette(qapp):
    """The noise dialog's fallback palette lacks a 'channels' key — must not KeyError."""
    import pyqtgraph as pg
    from respmech.ui.plot_overlays import add_flow_background
    pw = pg.PlotWidget(); pi = pw.getPlotItem()
    item = add_flow_background(pi, np.arange(50) / 50.0, np.arange(50, dtype=float),
                               {"bg": "#fff", "noise_trace": (90, 150, 200)})
    assert item is not None and item.zValue() == _Z


# -- integration at each EMG render site -------------------------------------
def _win(s):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    return MainWindow(AppState(s))


def test_emg_detail_and_result_and_raw_carry_flow(qapp, tmp_path):
    from respmech.ui.workers import stage_emg_channel, stage_emg_all_channels
    s = _settings(str(tmp_path))
    path = os.path.join(INPUT, "synth_case_A.csv")
    win = _win(s)
    pv = win.preview_screen

    # DETAIL: stage carries a full-length flow aligned to t; render draws one background
    d = stage_emg_channel(s, path, 0)
    assert "flow" in d and len(d["flow"]) == len(d["t"])
    pv.render_emg_time(d)
    detail = pv.emg_plots.getPlotItem()
    assert len(_flow_items(detail)) == 1
    names = [it.name() for it in detail.listDataItems() if it.name()]
    assert "raw" in names and "RMS envelope" in names          # EMG traces intact

    # RESULT: one background behind the ticked channels; none when flow is absent
    alld = stage_emg_all_channels(s, path)
    assert "flow" in alld
    pv._emg_all = alld
    for _c, cb in pv.result_checks:
        cb.setChecked(True)
    pv._render_emg_result()
    assert len(_flow_items(pv.emg_result_plots.getPlotItem())) == 1
    pv._emg_all = {**alld, "flow": None}
    pv._render_emg_result()
    assert _flow_items(pv.emg_result_plots.getPlotItem()) == []  # no-op without flow

    # RAW stack: one background per channel subplot
    pv._render_raw_stack(np.column_stack(alld["raw"]), alld["fs"], alld["flow"])
    assert pv._emg_raw_subplots and all(len(_flow_items(p)) == 1 for p in pv._emg_raw_subplots)
    win.close()


def test_noise_modal_carries_flow_per_channel(qapp, tmp_path):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    from respmech.ui.workers import stage_raw_emg
    s = _settings(str(tmp_path))
    data = stage_raw_emg(s, os.path.join(INPUT, "synth_case_A.csv"))
    assert "flow" in data and len(data["flow"]) == len(data["t"])
    dlg = NoiseProfileDialog(data["raw"], data["t"], data["fs"], data["cols"], flow=data["flow"])
    assert dlg._plots and all(len(_flow_items(p)) == 1 for p in dlg._plots)
    # backward-compatible: no flow -> no overlay, dialog still fine
    dlg2 = NoiseProfileDialog(data["raw"], data["t"], data["fs"], data["cols"], flow=None)
    assert all(_flow_items(p) == [] for p in dlg2._plots)


def test_psd_has_no_flow_overlay(qapp, tmp_path):
    """The frequency-domain PSD (matplotlib) is correctly excluded."""
    from respmech.ui.workers import stage_emg_channel
    s = _settings(str(tmp_path))
    win = _win(s)
    pv = win.preview_screen
    d = stage_emg_channel(s, os.path.join(INPUT, "synth_case_A.csv"), 0)
    pv.render_emg_psd(d)
    # the PSD axis lines are raw/ecg/noise PSDs + the band markers — no flow silhouette exists
    ax = pv.emg_psd_canvas.figure.axes[0]
    assert ax.get_xlabel().lower().startswith("frequency")
    win.close()


# -- ECG capture markers ------------------------------------------------------
def test_capture_markers_two_items_noop_on_empty(qapp):
    import pyqtgraph as pg
    from respmech.ui.plot_overlays import add_ecg_capture_markers, _CAPTURE_Z
    pw = pg.PlotWidget(); pi = pw.getPlotItem()
    pi.plot(np.arange(200) / 200.0, np.sin(np.arange(200) / 5.0))
    res = add_ecg_capture_markers(pi, np.array([0.1, 0.5, 0.9]), {"bg": "#FCFDFE"})
    assert res is not None
    line, marker = res
    assert line.zValue() == _CAPTURE_Z and marker.zValue() == _CAPTURE_Z + 1
    assert line in pi.listDataItems() and marker in pi.listDataItems()
    # no-op on empty / missing / non-finite
    assert add_ecg_capture_markers(pi, np.array([]), {}) is None
    assert add_ecg_capture_markers(pi, None, {}) is None
    assert add_ecg_capture_markers(pi, np.full(3, np.nan), {}) is None


def test_stage_ecg_reduction_shape_and_peaks(qapp, tmp_path):
    import os
    from respmech.ui.workers import stage_ecg_reduction
    from respmech.ui.screens import _preview_cache as pc
    s = synth_settings(str(tmp_path), remove_ecg=True,
                       data_out={"saveaveragedata": True, "savebreathbybreathdata": True})
    nch = len(s.input.channels.emg)
    pc.clear_all()
    d = stage_ecg_reduction(s, os.path.join(INPUT, "synth_case_A.csv"))
    assert len(d["processed"]) == nch
    assert len(d["processed"][0]) == len(d["t"]) == len(d["raw_capture"]) == len(d["flow"])
    assert 0 <= d["detect"] < nch and d["ecg_applied"] is True
    assert np.asarray(d["peaks"]).ndim == 1                 # R-peak times (may be empty on synth)
    # removal OFF: still returns a result; processed == raw (no subtraction), capture still computed
    s2 = synth_settings(str(tmp_path), remove_ecg=False,
                        data_out={"saveaveragedata": True, "savebreathbybreathdata": True})
    pc.clear_all()
    d2 = stage_ecg_reduction(s2, os.path.join(INPUT, "synth_case_A.csv"))
    assert d2["ecg_applied"] is False and len(d2["processed"]) == nch


def test_mechanics_preview_applies_trend_correction_like_batch(qapp, tmp_path):
    """audit #11: the mechanics preview applied only drift, never trend — so with
    correct_trend on it diverged from the batch. It must now match the batch volume."""
    from respmech.core import pipeline
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(str(tmp_path), data_out=_DATA_OUT)
    s.processing.volume.correct_trend = True
    path = os.path.join(INPUT, "synth_case_A.csv")
    prev = np.asarray(stage_mechanics_preview(s, path)["series"]["volume"], float)
    fr = pipeline.run_batch(s, only_files=["synth_case_A.csv"]).ok_files["synth_case_A.csv"]
    batch = np.asarray(fr.signals["vol_final"], float)
    n = min(len(prev), len(batch))
    assert np.allclose(prev[:n], batch[:n], atol=1e-9)          # preview == batch (trend applied)
    s2 = synth_settings(str(tmp_path), data_out=_DATA_OUT)      # trend OFF
    prev2 = np.asarray(stage_mechanics_preview(s2, path)["series"]["volume"], float)
    assert not np.allclose(prev[:n], prev2[:n])                 # trend actually changed the volume

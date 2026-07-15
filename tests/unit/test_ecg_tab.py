"""The EMG–ECG reduction tab: reactive scoping of the 'ecg' kind, the capture-channel
middle-default seed, the render (flow background + R-peak capture markers on the raw capture
channel AND every processed channel), and the Auto-suggest button. UI-only — goldens untouched."""
import os

import numpy as np

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _win(s):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    return MainWindow(AppState(s))


def test_ecg_settings_paths_recompute_the_ecg_kind():
    from respmech.ui.screens.preview_screen import _kinds_for_settings_path
    for p in ("processing.emg.detect_channel", "processing.emg.remove_ecg",
              "processing.emg.ecg_min_height", "processing.emg.ecg_window_s"):
        assert "ecg" in _kinds_for_settings_path(p)          # tab recomputes on an ECG edit
        assert "emg_all" in _kinds_for_settings_path(p)      # ...and the downstream EMG panels
    assert "ecg" in _kinds_for_settings_path("input.channels.emg")
    assert "ecg" not in _kinds_for_settings_path("processing.wob.calc_from")   # unrelated


def test_capture_channel_defaults_to_middle_when_unconfigured(qapp, tmp_path):
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)   # 3 EMG cols, ECG unset
    pv = _win(s).preview_screen
    assert pv.ecg_capture_channel.currentIndex() == 1        # middle of 3
    assert s.processing.emg.detect_channel == 1              # seeded into the model
    # a configured analysis (removal already ON) keeps its stored channel (0), not the seed
    s2 = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    assert _win(s2).preview_screen.ecg_capture_channel.currentIndex() == 0


def test_ecg_render_draws_flow_and_capture_on_every_channel(qapp, tmp_path):
    from respmech.ui.plot_overlays import _FLOW_Z, _CAPTURE_Z
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    n, fs = 4000, 2000
    t = np.arange(n) / fs
    data = {"t": t, "fs": fs, "cols": [2, 3, 4], "detect": 1, "detect_col": 3,
            "raw_capture": np.sin(t), "processed": [np.sin(t), np.cos(t), np.sin(2 * t)],
            "peaks": np.array([0.5, 1.0, 1.5, 2.0]), "flow": np.sin(2 * np.pi * 0.3 * t),
            "ecg_applied": True, "ecg_error": None}
    pv._on_ecg_result(data)
    cap_z = [it.zValue() for it in pv.ecg_capture_plot.getPlotItem().listDataItems()]
    assert _FLOW_Z in cap_z and _CAPTURE_Z in cap_z          # flow behind + capture line/triangle
    assert len(pv._ecg_capture_subplots) == 3
    for p in pv._ecg_capture_subplots:                       # every processed channel too
        pz = [it.zValue() for it in p.listDataItems()]
        assert _FLOW_Z in pz and _CAPTURE_Z in pz
    # empty peaks -> no capture markers, but the trace + flow still render
    data2 = {**data, "peaks": np.array([])}
    pv._on_ecg_result(data2)
    assert _CAPTURE_Z not in [it.zValue() for it in pv.ecg_capture_plot.getPlotItem().listDataItems()]


def test_noise_strip_stays_compact_across_tab_switches(qapp, tmp_path):
    """Regression: with three sub-tabs the noise-strip aligner fires on every tab switch,
    including while the noise tab is not laid out. Keyed off the chip's live height it fed
    the button's min-height back into the strip row and let the chip balloon a little more
    each switch — once to ~247 px, ~55% of the tab, crushing the plots into a bottom band.
    Now keyed off sizeHint + a Maximum vertical policy, so it stays compact and idempotent."""
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    win = _win(s); pv = win.preview_screen
    win.resize(1400, 900); win.show(); qapp.processEvents()
    for _ in range(3):                                    # exercise the aligner repeatedly
        for i in range(pv.subtabs.count()):
            pv.subtabs.setCurrentIndex(i); qapp.processEvents()
    idx = pv.subtabs.indexOf(pv._emg_tab); pv.subtabs.setCurrentIndex(idx)
    for _ in range(4):
        qapp.processEvents()
    assert pv.btn_set_noise.minimumHeight() < 100         # compact — not the ~247 runaway
    assert pv.noise_opts.height() < 100
    win.close()


def test_autosuggest_writes_settings_and_selects_channel(qapp, tmp_path, monkeypatch):
    from respmech.core import emg as emglib
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)
    win = _win(s); pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentIndex(0)
    monkeypatch.setattr(emglib, "suggest_ecg_settings", lambda m, fs: {
        "detect_channel": 2, "ecg_min_height": 0.02, "ecg_min_distance_s": 0.45,
        "ecg_min_width_s": 0.005, "ecg_window_s": 0.4,
        "_diagnostics": {"confidence": "high", "est_bpm": 70.0}})
    pv._on_ecg_autosuggest()
    e = s.processing.emg
    assert e.detect_channel == 2 and abs(e.ecg_min_height - 0.02) < 1e-9
    assert abs(e.ecg_min_distance_s - 0.45) < 1e-9 and abs(e.ecg_window_s - 0.4) < 1e-9
    assert pv.ecg_capture_channel.currentIndex() == 2        # reflected into the widget
    win.close()

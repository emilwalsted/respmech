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


# -- Sub-tab structure: Mechanics always, EMG only with EMG channels --------
def test_emg_subtab_visibility_tracks_channels(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    # with EMG channels -> both sub-tabs present
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    titles = [pv.subtabs.tabText(i) for i in range(pv.subtabs.count())]
    assert titles == ["Mechanics", "EMG processing"]
    # remove the EMG channels and re-sync -> EMG sub-tab disappears, Mechanics stays
    win.state.settings.input.channels.emg = []
    pv.sync_from_settings()
    titles = [pv.subtabs.tabText(i) for i in range(pv.subtabs.count())]
    assert titles == ["Mechanics"]
    # the widgets still exist (just not shown), so nothing referencing them breaks
    assert pv.emg_channel.count() == 0


# -- Moved noise-window options live on the EMG tab, gated on a reference ----
def test_noise_options_gated_on_reference_file(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    s.processing.emg.noise.enabled = True
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentIndex(0); pv._update_actions()
    # noise on but no reference -> options disabled
    assert pv.noise_opts.isEnabled() is False
    # choose a rest region on the graph -> sets the reference -> options enable
    pv._ensure_noise_region(); pv._noise_region.setRegion((1.0, 5.0))
    pv._use_region_as_noise(); pv._update_actions()
    assert win.state.settings.processing.emg.noise.reference_file != ""
    assert pv.noise_opts.isEnabled() is True
    # editing a noise-window widget writes straight back to settings
    pv.noise_auto.setChecked(False)
    pv.noise_prop.setValue(0.35)
    assert win.state.settings.processing.emg.noise.auto_prop is False
    assert abs(win.state.settings.processing.emg.noise.prop_decrease - 0.35) < 1e-9


# -- Result view: stage all channels, pick which to show --------------------
def test_emg_result_view_selectable_channels(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_all_channels
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentIndex(0)
    path = os.path.join(INPUT, "synth_case_A.csv")
    data = stage_emg_all_channels(s, path)
    assert data["cols"] == [2, 3, 4]
    assert len(data["raw"]) == len(data["conditioned"]) == 3
    # a checkbox per EMG channel; the result view plots only the ticked ones
    assert len(pv.result_checks) == 3
    pv._emg_all = data
    for _c, cb in pv.result_checks:
        cb.setChecked(False)
    pv.result_checks[1][1].setChecked(True)   # triggers a re-render
    pv._render_emg_result()
    plotted = [it for it in pv.emg_result_plots.getPlotItem().listDataItems()]
    assert len(plotted) == 1


# -- breath numbering carried onto the EMG views ----------------------------
def _stage_mech(pv, s, name="synth_case_A.csv"):
    """Refresh + select `name`, then stage its mechanics preview synchronously."""
    from respmech.ui.workers import stage_mechanics_preview
    pv._refresh_files()
    pv.file_combo.setCurrentText(name)
    return stage_mechanics_preview(s, os.path.join(INPUT, name))


def _red_dominant(reg):
    c = reg.brush.color()
    return c.red() > c.blue()            # excluded brush is red, included is blue


def test_emg_raw_view_numbers_breaths_at_trim_offset(qapp, tmp_path):
    """Every mechanics breath is re-shaded + numbered on the raw EMG stack, shifted
    by the trim offset (startix/fs) so the untrimmed EMG time base lines up."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    data = _stage_mech(pv, s)
    pv._render_preview(data)                          # mechanics + raw EMG stack + numbers
    off = data["startix"] / data["fs"]
    assert pv._trim_offset_s == off and pv._breaths
    raw = pv._bov["raw"]
    for (num, t0, t1, _ig) in pv._breaths:
        a, b = raw["regions"][num][0].getRegion()     # region on the first raw subplot
        assert abs(a - (t0 + off)) < 1e-6 and abs(b - (t1 + off)) < 1e-6
        assert num in raw["texts"]                    # a visible number label per breath
    win.close()


def test_toggle_recolours_every_emg_view_and_writes_exclusion(qapp, tmp_path):
    """Toggling a breath recolours it in Mechanics AND every painted EMG view, and
    records the 1-based number in settings so the preview analyses exclude it."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_channel, stage_emg_all_channels
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    path = os.path.join(INPUT, "synth_case_A.csv")
    pv._render_preview(_stage_mech(pv, s))            # raw view painted
    pv._on_emg_detail_result(stage_emg_channel(s, path, 0))   # detail view painted
    pv._on_emg_all_result(stage_emg_all_channels(s, path))    # result view painted
    num = pv._breaths[0][0]
    for view in ("raw", "detail", "result"):
        assert num in pv._bov[view]["regions"], f"{view} missing breath {num}"
    assert pv._toggle_breath(num) is True             # now excluded
    excl = {b for e in s.processing.exclude_breaths
            if e.file == "synth_case_A.csv" for b in e.breaths}
    assert num in excl                                # excluded from the preview analyses
    assert _red_dominant(pv._breath_regions[num][0])  # Mechanics recoloured
    for view in ("raw", "detail", "result"):
        assert _red_dominant(pv._bov[view]["regions"][num][0]), f"{view} not recoloured"
    win.close()


def test_emg_detail_overlay_is_idempotent(qapp, tmp_path):
    """Re-rendering the EMG detail (e.g. a second staging job) must not accumulate
    overlay items — the per-view paint removes its previous items first."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_channel
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    path = os.path.join(INPUT, "synth_case_A.csv")
    pv._render_preview(_stage_mech(pv, s))
    d = stage_emg_channel(s, path, 0)
    pv._on_emg_detail_result(d)
    items, regions = len(pv._bov["detail"]["items"]), len(pv._bov["detail"]["regions"])
    pv._on_emg_detail_result(d)                       # re-render
    assert items > 0
    assert len(pv._bov["detail"]["items"]) == items
    assert len(pv._bov["detail"]["regions"]) == regions


def test_file_change_resets_breath_overlays(qapp, tmp_path):
    """Selecting a different file drops the previous file's breath overlays + spans
    so a new-file EMG job never numbers against the old breaths."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    assert pv._bov.get("raw", {}).get("regions")      # overlays present for A
    pv.file_combo.setCurrentText("synth_case_B.csv")  # -> _on_file_selected -> reset
    assert pv._bov == {} and pv._breaths == []
    win.close()


# -- review follow-ups: stale-state / wrong-file hardening ------------------
def test_file_change_clears_emg_all_cache(qapp, tmp_path):
    """A file switch must drop the previous file's cached all-channel result, so an
    always-live result-checkbox toggle can't re-plot the old curves and re-arm the
    result sentinel (which would paint the new file's numbers over old curves)."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_all_channels
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    path = os.path.join(INPUT, "synth_case_A.csv")
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    pv._on_emg_all_result(stage_emg_all_channels(s, path))
    assert pv._emg_all is not None                    # A's conditioned result cached
    pv.file_combo.setCurrentText("synth_case_B.csv")  # -> reset drops the cache
    assert pv._emg_all is None
    pv._render_emg_result()                           # a checkbox re-render now no-ops
    assert pv._result_label_y is None
    assert pv._bov.get("result", {}).get("regions", {}) == {}
    win.close()


def test_file_change_clears_previewed_file(qapp, tmp_path):
    """_previewed_file must reset on a switch so flipping back to the last-rendered
    file (while its jobs are still in flight) is treated as a real change, not a no-op."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    assert pv._previewed_file == "synth_case_A.csv"
    pv.file_combo.setCurrentText("synth_case_B.csv")
    assert pv._previewed_file is None
    win.close()


def test_file_switch_invalidates_inflight_tokens(qapp, tmp_path):
    """Every kind's token bumps synchronously on a switch, so an old-file job that
    finishes in the same event-loop iteration is dropped by the acceptance check."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    before = dict(pv._tokens)
    pv.file_combo.setCurrentText("synth_case_B.csv")  # -> _begin_file_switch
    assert all(pv._tokens[k] > before[k] for k in before)
    win.close()


def test_refresh_files_prev_vanished_resets_overlays(qapp, tmp_path):
    """refresh_files switches the current file silently (signals blocked) when the
    previous file no longer matches; that must still run the switch cleanup so stale
    overlays never linger clickable against the wrong file."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    assert pv._bov.get("raw", {}).get("regions")
    s.input.files = "synth_case_B.csv"                # narrow the glob: A no longer matches
    pv.refresh_files()
    assert pv.file_combo.currentText() == "synth_case_B.csv"
    assert pv._bov == {} and pv._breaths == [] and pv._previewed_file is None
    win.close()


def test_legend_click_does_not_toggle_breath(qapp, tmp_path):
    """A click landing on a plot's legend (which pyqtgraph leaves unaccepted) must
    not fall through and toggle the breath beneath it; a click that misses the legend
    still toggles. Fakes keep the guard branch deterministic without scene geometry."""
    from PySide6.QtCore import QPointF
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    num, t0, t1, _ig = pv._breaths[0]
    off = pv._trim_offset_s
    xc = (t0 + t1) / 2.0 + off                        # absolute-time centre of breath `num`

    class _Rect:
        def __init__(self, hit): self._hit = hit
        def contains(self, _p): return self._hit

    class _Legend:
        def __init__(self, hit): self._hit = hit
        def isVisible(self): return True
        def sceneBoundingRect(self): return _Rect(self._hit)

    class _View:
        def sceneBoundingRect(self): return _Rect(True)
        def mapSceneToView(self, _pos): return QPointF(xc, 0.0)

    class _Plot:
        def __init__(self, legend_hit): self.legend = _Legend(legend_hit)
        def getViewBox(self): return _View()

    class _Ev:
        def isAccepted(self): return False
        def scenePos(self): return QPointF(0.0, 0.0)

    def excluded():
        return {b for e in s.processing.exclude_breaths
                if e.file == "synth_case_A.csv" for b in e.breaths}

    pv._toggle_from_emg_click(_Ev(), [_Plot(legend_hit=True)], off)   # on the legend
    assert num not in excluded()                                     # -> not toggled
    pv._toggle_from_emg_click(_Ev(), [_Plot(legend_hit=False)], off)  # misses the legend
    assert num in excluded()                                         # -> toggled
    win.close()

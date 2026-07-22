"""Headless tests for the interactive Preview & tuning graph features (A/B/C).

Constructed under QT_QPA_PLATFORM=offscreen; the mouse interaction itself is not
exercised — the drawing helpers and the toggle/apply/staging methods are called
directly, which is what must be testable without a display.
"""
import os

import pytest





from respmech.ui.state import AppState  # noqa: E402


from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


pytestmark = requires_synth()




def _settings(outdir):
    return synth_settings(outdir, remove_ecg=True, data_out=_DATA_OUT)


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
    # ...and the Preview strip shows it read-only: the picker is the only writer of all three
    assert name in pv.noise_ref_readout.text()


# -- Feature C: preview EMG conditioning (staging + render) -----------------
def test_stage_emg_channel_and_render(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_channel
    s = _settings(str(tmp_path))
    s.processing.emg.noise.enabled = True          # the staging honours this gate, as run_batch does
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


def _noise_settings(outdir):
    """Settings with the noise stage fully configured AND enabled."""
    s = _settings(outdir)
    n = s.processing.emg.noise
    n.enabled = True
    n.reference_file = "synth_case_A.csv"
    n.use_expiration = False
    n.reference_intervals = [[1.0, 5.0]]
    return s


def test_preview_honours_the_noise_enabled_gate(tmp_path):
    """The staging must gate on noise.enabled exactly as run_batch does. Gating on
    reference_file alone showed a spectrally gated trace after the user switched noise
    reduction OFF — a signal the run would never produce."""
    import numpy as np
    from respmech.ui.workers import stage_emg_all_channels
    s = _noise_settings(str(tmp_path))
    path = os.path.join(INPUT, "synth_case_A.csv")
    on = stage_emg_all_channels(s, path)
    s.processing.emg.noise.enabled = False            # reference_file deliberately still set
    off = stage_emg_all_channels(s, path)
    assert on["noise_applied"] is True and off["noise_applied"] is False
    assert not all(np.array_equal(a, b)                # and the trace really changes
                   for a, b in zip(on["conditioned"], off["conditioned"]))


def test_preview_applies_the_auto_selected_suppression(tmp_path):
    """Under auto_prop the run applies the value the fidelity sweep chose, not the stored
    prop_decrease. The preview must apply the same one or it is tuned against a differently
    gated signal."""
    from respmech.ui.workers import stage_emg_all_channels, stage_noise_fidelity
    s = _noise_settings(str(tmp_path))
    s.processing.emg.noise.auto_prop = True
    s.processing.emg.noise.prop_decrease = 0.6        # a snapshot value that must NOT be used
    path = os.path.join(INPUT, "synth_case_A.csv")
    chosen = stage_noise_fidelity(s).get("prop_decrease")
    assert chosen is not None and abs(float(chosen) - 0.6) > 1e-9   # the sweep really differs
    assert stage_emg_all_channels(s, path)["prop_decrease"] == pytest.approx(float(chosen))


def test_noise_failure_leaves_no_channel_half_denoised(tmp_path):
    """Applying in place left channels 0..i-1 denoised when channel i raised, while the
    payload still said noise_applied=False — a mixed stack presented as ECG-only."""
    import numpy as np
    from respmech.core import noise as noiselib
    from respmech.ui.workers import stage_emg_all_channels, stage_emg_channel
    from respmech.ui.screens import _preview_cache as pc
    s = _noise_settings(str(tmp_path))
    s.processing.emg.noise.auto_prop = False
    path = os.path.join(INPUT, "synth_case_A.csv")
    orig, calls = noiselib.NoiseProfile.apply, {"n": 0}

    def boom(self, sig, prop):
        calls["n"] += 1
        if calls["n"] == 2:                            # fail on the SECOND channel
            raise RuntimeError("simulated per-channel failure")
        return orig(self, sig, prop)

    pc.clear_all()
    noiselib.NoiseProfile.apply = boom
    try:
        part = stage_emg_all_channels(s, path)
    finally:
        noiselib.NoiseProfile.apply = orig
    assert part["noise_applied"] is False
    ecg_only = stage_emg_channel(s, path, 0)["ecg"]    # channel 0 must be untouched too
    assert np.allclose(part["conditioned"][0], ecg_only)


def test_out_of_range_ecg_detect_channel_is_rejected(tmp_path):
    """The preview used to clamp an out-of-range capture channel to the last one while the
    core indexes it straight — so a misconfiguration silently captured off the wrong channel
    here and only failed in the run."""
    from respmech.ui.workers import stage_ecg_reduction
    s = _settings(str(tmp_path))
    s.processing.emg.detect_channel = 99
    with pytest.raises(ValueError, match="outside the"):
        stage_ecg_reduction(s, os.path.join(INPUT, "synth_case_A.csv"))


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


def test_emg_controls_disabled_without_emg_channels(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    s.input.channels.emg = []
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentIndex(0)
    pv._update_actions()
    assert pv.emg_channel.isEnabled() is False        # EMG-gated controls are off
    assert pv.btn_set_noise.isEnabled() is False
    assert pv.emg_channel.count() == 0


# -- Sub-tab structure: Mechanics always, the two EMG tabs only with EMG channels ----
def test_emg_subtab_visibility_tracks_channels(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    # with EMG channels -> all three sub-tabs, in the pipeline order (Mechanics › ECG › noise)
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    titles = [pv.subtabs.tabText(i) for i in range(pv.subtabs.count())]
    assert titles == ["Mechanics", "› EMG – ECG reduction", "› EMG – noise reduction"]
    # remove the EMG channels and re-sync -> both EMG sub-tabs disappear, Mechanics stays
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
    # count EMG channel traces by name; the discrete flow-background overlay is a separate item
    plotted = [it for it in pv.emg_result_plots.getPlotItem().listDataItems()
               if it.name() and it.name().startswith("col ")]
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
    """Each FILE-dependent kind's token bumps synchronously on a switch, so an old-file
    job that finishes in the same event-loop iteration is dropped by the acceptance
    check. The test-wide 'noise' token is deliberately preserved — its result does not
    depend on the selected file, so a running fidelity job must survive the switch."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    before = dict(pv._tokens)
    pv.file_combo.setCurrentText("synth_case_B.csv")  # -> _begin_file_switch
    assert all(pv._tokens[k] > before[k] for k in ("mech", "batch", "emg_all", "emg_detail"))
    assert pv._tokens["noise"] == before["noise"]     # file-independent -> preserved
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


# -- UI rework: toolbar, panel layout, breath labels, clearing --------------
def _under(widget, ancestor):
    w = widget
    while w is not None:
        if w is ancestor:
            return True
        w = w.parent()
    return False


def test_toolbar_slimmed_and_status_moved_to_bottom(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    # the obsolete buttons are gone; only 'Refresh' remains in the top bar (the test
    # run is automatic now, so its button + the empty-state CTA are gone too)
    for gone in ("btn_refresh", "btn_preview", "btn_noise", "btn_emg_preview",
                 "btn_emg_result", "btn_test", "_test_cta"):
        assert not hasattr(pv, gone), f"{gone} should be removed"
    assert pv.btn_refresh_all.text() == "Refresh"
    # the status label is not shown under the file selector (bottom bar only)
    assert pv.status.isHidden() is True
    win.close()


def test_conditioned_result_spans_the_width_above_the_diagnostics(qapp, tmp_path):
    """Conditioned result is EMG against time, so width is what makes it readable: it spans
    the tab and sits above the two compact diagnostics (fidelity frontier + PSD), which
    share the bottom row and gain nothing from the extra width."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    win.resize(1400, 900); win.show()
    win.tabs.setCurrentWidget(pv)
    for i in range(pv.subtabs.count()):                    # front the EMG noise sub-tab:
        if "noise" in pv.subtabs.tabText(i):               # unshown widgets keep their
            pv.subtabs.setCurrentIndex(i); break           # constructor sizes, which would
    qapp.processEvents()                                   # satisfy the ratio vacuously
    # the result panel is NOT in a shared row — it is a full-width band of its own
    assert not _under(pv.emg_result_plots, pv._emg_diag_row)
    assert _under(pv.fidelity_canvas, pv._emg_diag_row)
    assert _under(pv.emg_psd_canvas, pv._emg_diag_row)
    assert not _under(pv.emg_plots, pv._emg_diag_row)      # detail stays in the top row
    # ...and, laid out, it really spans the row the diagnostics split between them
    assert pv.emg_result_plots.width() > 1.5 * pv.fidelity_canvas.width()
    assert pv.emg_result_plots.width() > 1000               # full band at a 1400px window
    # the detail dropdown and the result picker live inside their panels (not a toolbar)
    assert _under(pv.emg_channel, pv._emg_tab)
    assert _under(pv.result_checks_holder, pv._emg_tab)
    win.close()


def test_noise_controls_are_one_compact_chip_strip(qapp, tmp_path):
    from PySide6.QtWidgets import QFrame
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    # the noise-reduction params live in a single rounded 'chip' (still the gated unit)
    assert isinstance(pv.noise_opts, QFrame) and pv.noise_opts.objectName() == "noiseChip"
    # the button and the chip are siblings on one strip (the EMG-tab content widget)
    assert pv.btn_set_noise.parent() is pv.noise_opts.parent()
    # the button height is aligned to the chip's natural (sizeHint) height
    pv._align_noise_strip()
    assert pv.btn_set_noise.minimumHeight() == pv.noise_opts.sizeHint().height()
    win.close()


def test_result_picker_check_is_inside_the_coloured_box(qapp, tmp_path):
    from PySide6.QtWidgets import QCheckBox
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    assert len(pv.result_checks) == 3
    # each channel is a QCheckBox whose indicator is the channel-coloured box with the
    # tick image inside it (no separate swatch); layout has generous spacing
    boxes = [pv.result_checks_layout.itemAt(i).widget()
             for i in range(pv.result_checks_layout.count())]
    assert len(boxes) == 3 and all(isinstance(b, QCheckBox) for b in boxes)
    for _c, cb in pv.result_checks:
        qss = cb.styleSheet()
        assert "::indicator" in qss and "background" in qss   # indicator IS the coloured box
        assert "respmech_check.png" in qss and "image:" in qss.replace(" ", "")  # tick inside it
    assert pv.result_checks_layout.spacing() >= 20            # more air between channels
    win.close()


def test_breath_labels_are_prefixed(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_channel
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    path = os.path.join(INPUT, "synth_case_A.csv")
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    pv._on_emg_detail_result(stage_emg_channel(s, path, 0))
    n0 = pv._breaths[0][0]
    # every view labels the breath — full 'Breath #N' or the compact '#N' when dense
    for texts in (pv._breath_texts, pv._bov["raw"]["texts"], pv._bov["detail"]["texts"]):
        assert texts[n0].textItem.toPlainText() in (f"Breath #{n0}", f"#{n0}")
    win.close()


def test_stage_raw_emg_returns_raw_channels(qapp, tmp_path):
    from respmech.ui.workers import stage_raw_emg
    s = _settings(str(tmp_path))
    d = stage_raw_emg(s, os.path.join(INPUT, "synth_case_A.csv"))
    assert d["cols"] == [2, 3, 4] and len(d["raw"]) == 3
    assert d["fs"] == 1000 and len(d["t"]) == len(d["raw"][0])


def test_set_noise_profile_dialog_applies_selection(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from respmech.ui.main_window import MainWindow
    from respmech.ui import noise_profile_dialog as npd
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")

    class _FakeDialog:                                   # stub: accept with a chosen span
        def __init__(self, *a, **k):
            from PySide6.QtWidgets import QCheckBox
            self.use_expiration = QCheckBox()            # the dialog's two-way choice
        def exec(self): return QDialog.Accepted
        def selected_region(self): return (0.4, 0.7)

    monkeypatch.setattr(npd, "NoiseProfileDialog", _FakeDialog)
    pv._open_noise_profile_dialog()
    n = s.processing.emg.noise
    assert n.reference_file == "synth_case_A.csv"
    assert n.reference_intervals == [[0.4, 0.7]]
    assert n.use_expiration is False
    win.close()


def test_set_noise_profile_dialog_cancel_leaves_settings(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from respmech.ui.main_window import MainWindow
    from respmech.ui import noise_profile_dialog as npd
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    before = s.processing.emg.noise.reference_file

    class _FakeDialog:
        def __init__(self, *a, **k):
            from PySide6.QtWidgets import QCheckBox
            self.use_expiration = QCheckBox()
        def exec(self): return QDialog.Rejected
        def selected_region(self): return (0.4, 0.7)

    monkeypatch.setattr(npd, "NoiseProfileDialog", _FakeDialog)
    pv._open_noise_profile_dialog()
    assert s.processing.emg.noise.reference_file == before   # Cancel changed nothing
    win.close()


def test_noise_without_reference_hint_points_to_the_modal(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = ""
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._update_actions()
    txt = pv.status.text()
    assert "Set noise profile" in txt                    # points at the new modal…
    assert "Use selection as noise profile" not in txt   # …not the removed button
    win.close()


def test_breath_labels_flush_autorange_for_fresh_plot(qapp, tmp_path):
    import numpy as _np
    import pyqtgraph as pg
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    glw = pg.GraphicsLayoutWidget()
    p = glw.addPlot(row=0, col=0)
    x = _np.arange(0, 300, 0.01)
    p.plot(x, _np.sin(x))                                 # a fresh, unpainted single plot
    p.getViewBox().updateAutoRange()                      # paints/flushes the deferred auto-range
    assert p.getViewBox().viewRange()[0][1] > 100         # range expanded to the data (~300 s)
    win.close()


def test_trim_error_returns_raw_channels(qapp, tmp_path, monkeypatch):
    from respmech.ui import workers
    from respmech.core import compute
    s = _settings(str(tmp_path))
    monkeypatch.setattr(compute, "trim",
                        lambda *a, **k: (_ for _ in ()).throw(compute.TrimError("flow never crosses zero")))
    d = workers.stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))
    assert d["trim_error"] and "crosses zero" in d["trim_error"]
    assert d["spans"] == [] and d["nbreaths"] == 0
    assert d["series"]["flow"].size > 0            # the raw channels are still returned


def test_render_preview_trim_error_shows_channels_not_error_card(qapp, tmp_path, monkeypatch):
    from respmech.ui.main_window import MainWindow
    from respmech.ui import workers
    from respmech.core import compute
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    monkeypatch.setattr(compute, "trim",
                        lambda *a, **k: (_ for _ in ()).throw(compute.TrimError("flow never crosses zero")))
    pv._render_preview(workers.stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    assert len(pv._channel_plots) == 5             # raw channels still drawn
    assert pv._breaths == []                       # no breaths detected
    assert pv.panel_error("channels") is None      # a soft status, NOT a hard error card
    assert "could not detect breaths" in pv.status.text().lower()
    win.close()


def test_batch_trim_error_is_soft_not_a_hard_card(qapp, tmp_path):
    """The now-automatic mechanics batch must treat a TrimError the SAME soft way the
    mech preview does — no hard 'Test run failed' card — since the raw channels + a
    'could not detect breaths' status already explain it."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import BatchResult, FileResult
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    result = BatchResult(files={"synth_case_A.csv": FileResult(
        file="synth_case_A.csv", error="TrimError: the flow signal never crosses zero")})
    pv._on_batch_result(result)                      # returns normally (no _FileRunError)
    assert pv.panel_error("campbell") is None        # no hard error card
    assert pv.table.rowCount() == 0                  # table left blank
    # a genuine (non-TrimError) analysis error still raises -> hard card
    bad = BatchResult(files={"synth_case_A.csv": FileResult(
        file="synth_case_A.csv", error="ValueError: bad column 7")})
    with pytest.raises(RuntimeError):
        pv._on_batch_result(bad)
    win.close()


def test_noise_fidelity_trim_error_returns_clean_error_dict(qapp, tmp_path, monkeypatch):
    """A misassigned/inverted flow channel means the noise reference has no segmentable
    breaths -> compute.trim raises TrimError deep in the reference-clip build. The noise
    stage must catch that precondition failure and return a clean {"error": msg} (a single,
    actionable line) instead of letting a raw traceback reach the fidelity panel."""
    from respmech.ui import workers
    from respmech.core import compute
    from respmech.ui.screens import _preview_cache as pc
    pc.clear_all()                                     # force a cache miss -> the trim runs
    s = synth_settings(str(tmp_path), remove_ecg=True, noise=True, data_out=_DATA_OUT)
    monkeypatch.setattr(compute, "trim",
                        lambda *a, **k: (_ for _ in ()).throw(compute.TrimError("flow never crosses zero")))
    report = workers.stage_noise_fidelity(s)
    assert isinstance(report, dict) and report.get("error")
    assert "\n" not in report["error"]                 # single line (short_error shows the last line)
    assert "flow channel" in report["error"].lower() and "reference intervals" in report["error"].lower()
    assert "traceback" not in report["error"].lower()


def test_on_noise_result_error_paints_a_clean_card_not_a_traceback(qapp, tmp_path):
    """_on_noise_result must raise _FileRunError for an {"error":...} report so _on_job_done
    paints a clean 'failed' card — NOT silently render a blank frontier (which would also mark
    the panel done and suppress the auto-retry)."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _FileRunError
    s = synth_settings(str(tmp_path), remove_ecg=True, noise=True, data_out=_DATA_OUT)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    with pytest.raises(_FileRunError) as ei:
        pv._on_noise_result({"error": "no usable breaths — check the flow channel"})
    assert "flow channel" in str(ei.value)
    assert pv._noise_has_result is False               # a failed noise job stays retry-eligible
    win.close()


def test_view_limits_bound_x_to_the_data(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    vb = pv._channel_plots[0].getViewBox()
    xmin, xmax = vb.state["limits"]["xLimits"]
    assert xmin is not None and xmax is not None and xmax > xmin   # bounded to the data
    (x0, x1), _yr = vb.viewRange()
    assert x0 >= xmin - 1e-6 and x1 <= xmax + 1e-6                 # view sits within the limits
    win.close()


def test_breath_label_headroom_above_the_signal(qapp, tmp_path):
    import numpy as _np
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    data = _stage_mech(pv, s, "synth_case_A.csv")
    pv._render_preview(data)
    flow_max = float(_np.nanmax(data["series"]["flow"]))
    _xr, (y0, y1) = pv._channel_plots[0].getViewBox().viewRange()
    assert y1 > flow_max                           # room above the signal for the breath labels
    win.close()


def test_label_headroom_does_not_compound_across_repaints(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_emg_channel
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    path = os.path.join(INPUT, "synth_case_A.csv")
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    pv._on_emg_detail_result(stage_emg_channel(s, path, 0))
    vb = pv.emg_plots.getPlotItem().getViewBox()
    top0 = vb.viewRange()[1][1]
    for _ in range(4):
        pv._repaint_view_breaths("detail")         # repeated repaints, no re-plot
    assert abs(vb.viewRange()[1][1] - top0) < 1e-6  # headroom stays fixed (no compounding)
    win.close()


def test_breath_label_is_always_compact():
    from respmech.ui.screens.preview_screen import PreviewScreen as P
    # 'breath' is dropped from per-breath numbering — the label is always the compact '#N'
    assert P._breath_label(7) == "#7"
    assert P._breath_label(12) == "#12"


def test_file_switch_clears_all_panels(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    pv._on_batch_result(run_batch(s, only_files=["synth_case_A.csv"]))   # fills table + Campbell
    assert pv.table.rowCount() > 0 and pv._channel_plots
    pv.file_combo.setCurrentText("synth_case_B.csv")   # -> clears every panel as at start
    assert pv.table.rowCount() == 0 and pv._channel_plots == []
    assert pv._bov == {} and pv._breaths == []
    win.close()


# -- review follow-ups: overlay lifecycle -----------------------------------
def test_stale_error_card_cleared_on_file_switch(qapp, tmp_path):
    """A stale panel error card (e.g. from a failed run) must be dismissed on a file
    switch, since _clear_all_panels blanks the screen to its start state."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._render_preview(_stage_mech(pv, s, "synth_case_A.csv"))
    pv._overlays["campbell"].show_error("Test run failed", "boom")
    assert pv.panel_error("campbell") is not None
    pv.file_combo.setCurrentText("synth_case_B.csv")   # -> _clear_all_panels dismisses it
    assert pv.panel_error("campbell") is None
    assert pv._overlays["campbell"].isHidden() is True
    win.close()


def test_detail_job_covers_both_time_and_psd_panels(qapp, tmp_path):
    """The emg_detail job renders the time plot AND the PSD, so both must carry the
    spinner/error overlay (the panel swap must not leave the PSD uncovered)."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _PANELS
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    assert _PANELS["emg_detail"] == ["detail", "detail_psd"]
    assert pv._overlays["detail_psd"].parent() is pv.emg_psd_canvas
    for p in _PANELS["emg_detail"]:
        pv._overlays[p].start("Staging detail…")
    assert {"detail", "detail_psd"} <= pv.busy_panels()
    win.close()


def test_file_strip_sizes_to_content_not_window(qapp, tmp_path):
    """The file selector sizes to the recording names instead of swallowing the row's
    stretch; the ◀/▶ glyphs keep side air and can never clip (width tracks sizeHint —
    the QSS min-width:0 makes any code-side fixed width a dead letter); Refresh sits
    just right of ▶ at its natural size, with the slack parked in a trailing stretch."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    win.resize(1400, 900); win.show()
    win.tabs.setCurrentWidget(pv); pv._refresh_files(); qapp.processEvents()
    assert pv.file_combo.count() > 0                     # the guard is meaningless when empty
    assert pv.file_combo.width() < win.width() / 3       # was 1222px of 1400 before the fix
    assert pv.file_combo.width() >= min(pv.file_combo.sizeHint().width(), 338)
    for b in (pv.btn_prev_file, pv.btn_next_file):
        assert b.width() >= b.sizeHint().width()         # the historical clipped-glyph mode
        assert b.width() > 20                            # air exists (13px before the fix)
    # Refresh keeps its natural size (the slack lives in the stretch, not the buttons)...
    assert pv.btn_refresh_all.width() <= 1.5 * pv.btn_refresh_all.sizeHint().width()
    # ...and sits just right of ▶ with a little air
    gap = pv.btn_refresh_all.x() - (pv.btn_next_file.x() + pv.btn_next_file.width())
    assert 8 <= gap <= 40
    win.close()


def test_campbell_legend_sits_bottom_left_in_the_preview_only(qapp):
    """The preview plots Poes on x and volume on y, which leaves the bottom-LEFT corner
    empty (the loop's EELV end is bottom-right). The output PDF transposes those axes, so
    its legend must NOT follow: "lower left" there lands on the loop's EILV tip. Its
    loc="best" already places it clear of the data."""
    import inspect
    from respmech.ui.screens import preview_screen as ps
    from respmech.core import plots
    assert 'loc="lower left"' in inspect.getsource(ps.PreviewScreen._overlay_campbell_work)
    # the PDF's Campbell keeps auto-placement — its axes are transposed
    assert 'loc="best"' in inspect.getsource(plots._pv_average)
    assert 'loc="lower left"' not in inspect.getsource(plots._pv_average)


def test_breath_labels_stay_pinned_to_the_view_top_under_zoom(qapp, tmp_path):
    """The breath numbers behave like the red capture marks: pinned to the TOP of the
    visible view instead of zooming/panning away with the data. They are added with
    ignoreBounds so the pinned position can never re-inflate autorange (runaway top)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    win.resize(1400, 900); win.show(); qapp.processEvents()
    pv._refresh_files(); pv.file_combo.setCurrentIndex(0)
    pv._preview(); qapp.processEvents()
    vb = pv._channel_plots[0].getViewBox()
    texts = list(pv._breath_texts.values())
    assert len(texts) >= 2

    def pinned():
        top = vb.viewRange()[1][1]
        return all(abs(t.pos().y() - top) < 1e-9 for t in texts)

    assert pinned()                                    # placed at the view top initially
    y0, y1 = vb.viewRange()[1]
    vb.setYRange(y0, y0 + (y1 - y0) * 0.4, padding=0); qapp.processEvents()
    assert pinned()                                    # zoomed in -> labels follow the new top
    vb.setYRange(y0 - (y1 - y0) * 0.3, y0 + (y1 - y0) * 0.1, padding=0); qapp.processEvents()
    assert pinned()                                    # panned -> still at the top
    # ...and the pinned labels never feed autorange (the runaway-top regression)
    vb.enableAutoRange(y=True); vb.updateAutoRange(); qapp.processEvents()
    tops = []
    for _ in range(5):
        vb.updateAutoRange(); qapp.processEvents()
        tops.append(round(vb.viewRange()[1][1], 6))
    assert len(set(tops)) == 1
    win.close()


def test_emg_legends_are_a_single_row_at_the_bottom(qapp):
    """The pipeline-stage / channel legends read as one line along the BOTTOM of the plot,
    clear of the breath numbers pinned at the top. The column count must be re-applied on
    EVERY render, not just at construction: clear() empties the legend and the entry count
    varies (the detail plot gains 'noise-reduced' only when noise conditioning ran; the
    result plot has one entry per ticked channel)."""
    import numpy as np
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    win.resize(1400, 900); win.show(); qapp.processEvents()
    t = np.linspace(0, 1, 500); raw = np.sin(2 * np.pi * 50 * t)

    def one_row(plot):
        p = plot.getPlotItem()
        leg = p.legend
        assert len(leg.items) >= 2                      # a 1-entry legend proves nothing
        assert leg.columnCount == len(leg.items)
        assert leg.layout.rowCount() == 1               # the real grid, not the bookkeeping attr
        vb = p.getViewBox().sceneBoundingRect()
        gap = vb.bottom() - leg.sceneBoundingRect().bottom()
        assert 0 <= gap <= 12, gap                       # tucked along the bottom edge, above it

    # detail plot, both entry counts (4 with noise reduction, 3 without)
    pv.render_emg_time({"t": t, "raw": raw, "ecg": raw * .9, "noise": raw * .8, "noise_applied": True})
    qapp.processEvents(); one_row(pv.emg_plots)
    pv.render_emg_time({"t": t, "raw": raw, "ecg": raw * .9, "noise_applied": False})
    qapp.processEvents(); one_row(pv.emg_plots)

    # result plot, and it must survive a channel being unticked and re-ticked
    cols = list(s.input.channels.emg)
    pv._emg_all = {"t": t, "cols": cols,
                   "conditioned": [raw * (1 - .1 * i) for i in range(len(cols))], "flow": None}
    pv._render_emg_result(); qapp.processEvents(); one_row(pv.emg_result_plots)
    pv.result_checks[0][1].setChecked(False); qapp.processEvents(); one_row(pv.emg_result_plots)
    pv.result_checks[0][1].setChecked(True); qapp.processEvents(); one_row(pv.emg_result_plots)
    win.close()


def test_titled_panels_keep_their_contents_off_the_edge(qapp):
    """The EMG panels used to run their plots and right-pinned selectors flush to the panel
    edge, so they butted against the neighbouring panel/nav. The panel's own margin is the
    single source of that air — it must stay non-zero on every side."""
    from PySide6.QtWidgets import QComboBox, QLabel
    from respmech.ui.screens.preview_screen import PreviewScreen
    box = PreviewScreen._titled("Raw EMG channels", QLabel("plot"), corner=QComboBox())
    m = box.layout().contentsMargins()
    assert min(m.left(), m.top(), m.right(), m.bottom()) >= 4
    # the header must not re-add its own right inset on top of the panel's
    header = box.layout().itemAt(0).layout()
    assert header.contentsMargins().right() == 0


def test_batch_spinner_sits_on_the_two_result_panels(qapp, tmp_path):
    """The batch job fills the per-breath table AND the Campbell diagram, so each must carry
    its own overlay. Parenting one overlay to the lower splitter instead painted a single
    'Running test…' band across the pair, which read as a third panel appearing mid-refresh."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _PANELS
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    assert _PANELS["batch"] == ["table", "campbell"]
    assert pv._overlays["table"].parent() is pv.table          # the real panels, not their splitter
    assert pv._overlays["campbell"].parent() is pv.campbell
    for p in _PANELS["batch"]:
        pv._overlays[p].start("Running test…")
    assert {"table", "campbell"} <= pv.busy_panels()
    win.close()


def test_mechanics_batch_snapshot_is_mechanics_only(qapp, tmp_path, monkeypatch):
    """The automatic mechanics test run must NOT do EMG/ECG/noise work: the batch worker
    is built from a snapshot with the EMG channels dropped."""
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))                     # has EMG cols [2,3,4] + remove_ecg
    s.processing.emg.noise.enabled = True
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    captured = {}
    monkeypatch.setattr(pv, "_launch", lambda kind, token, worker: captured.setdefault("w", worker))
    pv._schedule("batch")
    snap = captured["w"]._settings                   # the snapshot handed to BatchWorker
    assert list(snap.input.channels.emg) == []       # EMG dropped -> run_batch skips ECG/EMG/noise
    assert snap.processing.emg.remove_ecg is False
    assert snap.processing.emg.noise.enabled is False
    assert list(s.input.channels.emg) == [2, 3, 4]   # the real settings are untouched
    win.close()


def test_no_button_label_is_clipped_by_theme_padding(qapp, tmp_path):
    """Regression: the theme's QPushButton padding (7px 16px + borders = 34px) is WIDER
    than the app's fixed-size glyph buttons, so their labels were clipped to nothing —
    the ◀/▶ file steppers (30px) and the error-card 'i' (24px) both rendered blank on
    macOS. Any fixed-width button needing more than its fixed width hides its label."""
    from PySide6.QtWidgets import QPushButton
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    win.resize(1400, 900); win.show(); qapp.processEvents()
    clipped = []
    for b in win.findChildren(QPushButton):
        if not b.text():
            continue
        fixed = b.minimumWidth() == b.maximumWidth() and b.maximumWidth() < 16777215
        if fixed and b.sizeHint().width() > b.maximumWidth():
            clipped.append((b.text(), b.maximumWidth(), b.sizeHint().width()))
    assert clipped == [], f"buttons whose label cannot fit their fixed width: {clipped}"
    win.close()


def test_breath_labels_fit_inside_the_short_channel_plots(qapp, tmp_path):
    """The breath '#N' labels were clipped on the mechanics tab: the headroom was a fixed
    FRACTION of the data span (22%), but a TextItem is a fixed PIXEL height — on the five
    stacked channel plots (~46 px of data area each) 22% is less than the label. Headroom
    is now derived from the label's measured pixel height, and the label box itself is
    compact (QTextDocument's default 4 px margin, not the font, made it 23 px)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    win.resize(1360, 880); win.show(); qapp.processEvents()
    pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentIndex(0); pv._preview()
    for _ in range(10):
        qapp.processEvents()
    assert pv._breath_texts, "no breath labels drawn"
    import numpy as np
    vb = pv._channel_plots[0].getViewBox()
    (_x0, _x1), (y0, y1) = vb.viewRange()
    h_px = vb.height()
    assert h_px > 0 and y1 > y0
    dy_per_px = (y1 - y0) / h_px
    label_h = max(t.boundingRect().height() for t in pv._breath_texts.values())
    assert label_h <= 18, f"breath label box grew to {label_h}px (documentMargin/font regressed?)"
    # the labels are pinned AT the view top (hanging down into the view)...
    for t in pv._breath_texts.values():
        assert abs(t.pos().y() - y1) < 1e-9
    # ...so in the initial view the band _label_headroom reserves above the DATA must fit
    # the hanging label, or it would sit on the signal
    data_top = max(float(np.nanmax(c.yData))
                   for c in pv._channel_plots[0].listDataItems() if c.yData is not None)
    headroom_px = (y1 - data_top) / dy_per_px
    assert headroom_px >= label_h, f"label ({label_h}px) covers data: only {headroom_px:.1f}px of band"
    # ...and the signal must still get most of the plot, not be squashed to fit the label
    assert (data_top - y0) / dy_per_px >= 0.45 * h_px
    win.close()

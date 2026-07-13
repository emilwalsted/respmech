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


def test_detail_and_result_panels_swapped_with_inline_controls(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    # after the swap the mid splitter holds the Conditioned-result panel (not the detail)
    assert _under(pv.emg_result_plots, pv._emg_mid)
    assert not _under(pv.emg_plots, pv._emg_mid)     # detail moved up to the top row
    # the detail dropdown and the result picker live inside their panels (not a toolbar)
    assert _under(pv.emg_channel, pv._emg_tab)
    assert _under(pv.result_checks_holder, pv._emg_mid)   # picker sits in the result panel
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
    # the button height matches the chip band once the chip has a laid-out height
    pv.noise_opts.resize(420, 46)
    pv._align_noise_strip()
    assert pv.btn_set_noise.minimumHeight() == 46
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
        def __init__(self, *a, **k): pass
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
        def __init__(self, *a, **k): pass
        def exec(self): return QDialog.Rejected
        def selected_region(self): return (0.4, 0.7)

    monkeypatch.setattr(npd, "NoiseProfileDialog", _FakeDialog)
    pv._open_noise_profile_dialog()
    assert s.processing.emg.noise.reference_file == before   # Annullér changed nothing
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
    centers = list(_np.arange(3.0, 297.0, 3.0))          # dense breaths ~3 s apart
    pv._breath_labels_short(p, centers)                  # must flush the deferred auto-range
    assert p.getViewBox().viewRange()[0][1] > 100        # range expanded to the data (~300 s)
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


def test_breath_label_abbreviation_decision():
    from respmech.ui.screens.preview_screen import PreviewScreen as P
    # spread-out breaths (10 s apart) at 0.02 s/px -> full labels fit
    assert P._labels_would_overlap([0.0, 10.0, 20.0], 0.02) is False
    # tightly packed breaths (0.3 s apart) at the same scale -> would overlap -> abbreviate
    assert P._labels_would_overlap([0.0, 0.3, 0.6], 0.02) is True
    assert P._labels_would_overlap([0.0], 0.02) is False      # a single breath never overlaps
    assert P._breath_label(7, short=True) == "#7"
    assert P._breath_label(7, short=False) == "Breath #7"


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

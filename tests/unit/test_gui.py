"""Headless (offscreen) tests for the PySide6 GUI logic.

These verify construction and the settings/preview/run data flow without a display
(QT_QPA_PLATFORM=offscreen). Interactive look-and-feel still needs a real display,
but the wiring — settings round-trip, preview/test-run, and the worker running the
core and producing output — is locked here.
"""
import os

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401 (qapp from conftest)

pytestmark = requires_synth()

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _settings(outdir):
    return synth_settings(outdir, data_out=_DATA_OUT)


def test_mainwindow_constructs(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    # Setup, Preview & QC — Run & results has no tab of its own since B03 (it lives
    # inside Preview & QC as a drawer instead).
    assert win.tabs.count() == 2
    # settings form round-trips into shared state
    win.settings_screen.from_state()
    win.settings_screen.to_state()
    assert win.state.settings.input.folder == INPUT


def test_preview_and_batch_render(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    assert pv.file_rail.count() == 2
    pv.file_rail.select_index(0)
    pv._preview()
    assert "breaths" in pv.status.text()
    # the mechanics test run is automatic (async); its handler fills the table + Campbell
    pv._on_batch_result(run_batch(win.state.settings, only_files=["synth_case_A.csv"]))
    assert pv.table.model().rowCount() > 0 and pv.table.model().columnCount() > 0


def test_worker_runs_and_writes(qapp, tmp_path):
    from respmech.ui.workers import BatchWorker
    settings = _settings(str(tmp_path))
    results = {}
    w = BatchWorker(settings, write=True)
    w.finished.connect(lambda r: results.setdefault("r", r))
    w.run()  # synchronous — verifies the worker logic without a thread
    r = results["r"]
    assert r is not None
    assert set(r.ok_files) == {"synth_case_A.csv", "synth_case_B.csv"}
    assert os.path.exists(os.path.join(tmp_path, "data", "Average breathdata.xlsx"))


def test_preview_noise_fidelity_render(qapp, tmp_path):
    """The preview/tuning screen renders the per-test noise fidelity frontier."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = _settings(str(tmp_path))
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.use_expiration = False
    s.processing.emg.noise.reference_intervals = [[1.0, 5.0]]
    s.processing.emg.noise.auto_prop = True
    win = MainWindow(AppState(s))
    result = run_batch(s, only_files=["synth_case_B.csv"])
    assert result.noise_report is not None
    win.preview_screen.render_noise_report(result)
    assert "prop_decrease" in win.preview_screen.status.text()
    # the fidelity frontier figure has been drawn
    assert len(win.preview_screen.fidelity_canvas.figure.axes) == 1


def test_splash_resolves_fonts_to_installed_families(qapp):
    """The splash used CSS-style font stacks as SVG font-family, which Qt cannot
    resolve (the 'Populating font family aliases … missing font family' warning).
    make_splash must resolve them to a single installed family."""
    from PySide6.QtGui import QFontDatabase
    from respmech.ui import splash
    splash.make_splash(qapp)
    installed = set(QFontDatabase.families())
    # The real regression guard: the stacks are resolved to a SINGLE family, not passed to
    # Qt's SVG renderer as a comma-list (which triggered the "missing font family" warning).
    assert splash._MONO and "," not in splash._MONO
    assert splash._FONT and "," not in splash._FONT
    # A headless runner can expose NO fonts (Windows CI offscreen): only assert membership
    # when the font DB is actually populated.
    if installed:
        assert splash._MONO in installed and splash._FONT in installed


def test_theme_applies_light_and_dark(qapp, monkeypatch):
    from respmech.ui import theme
    assert theme.apply_theme(qapp) in ("light", "dark")
    monkeypatch.setenv("RESPMECH_THEME", "dark")
    assert theme.apply_theme(qapp) == "dark"
    assert len(qapp.styleSheet()) > 1000


def test_plot_palette_tracks_theme_and_has_a_complete_contract(qapp, monkeypatch):
    """The plot palettes drive the two plotting stacks (which don't inherit QSS).
    Dark mode must yield a genuinely dark plot ground, light must reproduce the
    historical near-white ground, and both tables must expose the SAME keys so no
    render path can KeyError depending on the theme."""
    from respmech.ui import theme

    monkeypatch.setenv("RESPMECH_THEME", "light")
    assert theme.apply_theme(qapp) == "light"
    light = theme.plot_palette()
    assert light["bg"] == "#FCFDFE" and light["mpl_bg"] == "#FFFFFF"   # unchanged light ground

    monkeypatch.setenv("RESPMECH_THEME", "dark")
    assert theme.apply_theme(qapp) == "dark"
    dark = theme.plot_palette()
    # a near-black plot ground for BOTH stacks -> the app actually reads as dark
    assert dark["bg"] != light["bg"] and dark["mpl_bg"] == dark["bg"]
    assert int(dark["bg"][1:3], 16) < 0x40                            # red channel < 64 -> very dark

    # identical key sets: every colour a render path may look up exists in both
    assert set(light) == set(dark)
    assert set(light["channels"]) == set(dark["channels"]) == {
        "flow", "volume", "poes", "pgas", "pdi"}
    assert len(light["emg_cycle"]) == len(dark["emg_cycle"]) >= 8

    monkeypatch.delenv("RESPMECH_THEME", raising=False)


def test_reactive_file_list_and_noise_gating(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    sc, pv = win.settings_screen, win.preview_screen
    assert pv.file_rail.count() == 2
    # changing the mask in Settings refreshes the Preview file list (reactive)
    sc.in_files.setText("synth_case_A.csv"); sc._on_inputs_changed()
    assert pv.file_rail.count() == 1
    sc.in_files.setText("synth_case_*.csv"); sc._on_inputs_changed()
    assert pv.file_rail.count() == 2
    # noise on but no reference -> the noise-window options are disabled + a hint pointing
    # at the one real way to set a reference: 'Set noise profile' on this file's own graph
    # (A03 dropped the old '(Settings)' alternative — there was never a picker there).
    # ECG removal has to be on first: it is the earlier prerequisite, and while it is off
    # the button that PICKS a reference is itself disabled, so asking for one would send
    # the user after something they cannot do.
    pv.noise_enabled.setChecked(True)             # noise enable is on the Preview strip now
    pv.state.settings.processing.emg.noise.reference_file = None
    pv.state.settings.processing.emg.remove_ecg = True
    pv.file_rail.select_index(0); pv._update_actions()
    assert pv.noise_opts.isEnabled() is False
    assert "set noise profile" in pv.status.text().lower()
    # ...and with ECG removal off, that prerequisite is what the user is told instead
    pv.state.settings.processing.emg.remove_ecg = False
    pv._update_actions()
    assert "remove ecg" in pv.status.text().lower()
    pv.state.settings.processing.emg.remove_ecg = True     # back to the prerequisite met
    # setting a reference enables them
    pv.state.settings.processing.emg.noise.reference_file = "synth_case_A.csv"
    pv._update_actions()
    assert pv.noise_opts.isEnabled() is True


def test_noise_enable_checkbox_binds_noise_enabled(qapp, tmp_path):
    """The noise on/off toggle moved from Setup to the Preview EMG-noise strip, model-direct."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    assert not hasattr(win.settings_screen, "remove_noise"), "the Setup checkbox is gone"
    pv.noise_enabled.setChecked(True)
    assert pv.state.settings.processing.emg.noise.enabled is True
    assert not hasattr(pv.state.settings.processing.emg, "remove_noise")   # legacy mirror gone
    pv.noise_enabled.setChecked(False)
    assert pv.state.settings.processing.emg.noise.enabled is False
    win.close()


def test_empty_input_folder_is_handled(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    s.input.folder = str(tmp_path / "does_not_exist")
    win = MainWindow(AppState(s))
    win.preview_screen.refresh_files()
    assert win.preview_screen.file_rail.count() == 0
    assert "not found" in win.preview_screen.status.text().lower()


def test_worker_cancellation(qapp, tmp_path):
    from respmech.ui.workers import BatchWorker
    w = BatchWorker(_settings(str(tmp_path)), write=False)
    w.cancel()  # cancel before running -> no files processed
    out = {}
    w.finished.connect(lambda r: out.setdefault("r", r))
    w.run()
    assert out["r"] is None  # cancelled -> None


def test_validate_checks_paths(qapp, tmp_path):
    """Validate reports filesystem problems the core (path-agnostic) validate misses."""
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(_settings(str(tmp_path))))
    # a good configuration passes the path checks
    assert sc._path_problem() is None
    # a non-existent input folder is caught
    sc.state.settings.input.folder = str(tmp_path / "nope")
    msg = sc._path_problem()
    assert msg and "input folder" in msg
    # restore folder; a missing noise reference file is caught when noise is on
    sc.state.settings.input.folder = INPUT
    sc.state.settings.processing.emg.noise.enabled = True
    sc.state.settings.processing.emg.noise.reference_file = "not_here.csv"
    msg = sc._path_problem()
    assert msg and "noise reference file" in msg


# --------------------------------------------------------------------------- #
# Status bar ownership (A03): the bar mirrors the ACTIVE tab's own status label,
# not whichever screen fired status_changed last — with one legitimate exception
# for a run's progress, which stays visible across tabs while it is in flight.
# --------------------------------------------------------------------------- #
def test_status_bar_shows_only_the_active_screens_message(qapp, tmp_path):
    """Run & results has no tab of its own since B03 — it lives inside Preview & QC as a
    drawer. A LIVE Run status update reaches the bar exactly when Preview & QC (the tab
    that hosts the drawer) is active, and not otherwise — the same per-tab ownership rule
    as before, just naming the tab that actually contains the drawer now. (Switching TO
    Preview & QC itself shows Preview's OWN message, since that tab's primary content is
    Preview, not Run — covered by test_refresh_files_skips_the_pick_one_line_when_already_drawn.)"""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    rn = win.run_screen
    assert win.tabs.currentWidget() is win.settings_screen    # Setup is the default active tab
    rn._set_status("RUN OWN MESSAGE")
    # Setup is the active tab -> Run's message must not reach the shared bar
    assert win.statusBar().currentMessage() != "RUN OWN MESSAGE"
    win.tabs.setCurrentWidget(win.preview_screen)              # switch to Preview & QC (hosts the drawer)
    rn._set_status("RUN OWN MESSAGE")                          # a fresh update, now that this tab is active
    assert win.statusBar().currentMessage() == "RUN OWN MESSAGE"
    win.tabs.setCurrentWidget(win.settings_screen)             # away again
    rn._set_status("STILL NOT SHOWN")
    assert win.statusBar().currentMessage() != "STILL NOT SHOWN"
    win.close()


def test_run_progress_shows_globally_while_a_batch_is_active(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    rn = win.run_screen
    assert win.tabs.currentWidget() is win.settings_screen    # looking at Setup, not Run
    win._on_run_started()
    rn._set_status("breath 3/10")
    assert win.statusBar().currentMessage() == "Run: breath 3/10"
    win._on_run_finished()
    # the run is over: the bar shows Run's own last message directly (its outcome is what
    # the user is looking at, regardless of which tab happens to be current — Run has no
    # tab of its own since B03), losing the "Run: " prefix that belonged to the exclusive
    # in-flight window rather than leaving the stale "Run: …" line sitting there
    assert win.statusBar().currentMessage() == "breath 3/10"
    rn._set_status("breath 4/10")            # Run isn't the active tab -> stays off the bar
    assert win.statusBar().currentMessage() != "breath 4/10"
    win.close()


def test_run_outcome_lines_are_not_double_prefixed(qapp, tmp_path):
    """Run's own outcome lines already start with the word 'Run' ('Run failed — …', 'Run
    cancelled — …') — the global "Run: " prefix must not stutter on top of them."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    rn = win.run_screen
    win._on_run_started()
    rn._set_status("Run failed — ValueError: boom")
    assert win.statusBar().currentMessage() == "Run failed — ValueError: boom"
    rn._set_status("Run cancelled — no output written.")
    assert win.statusBar().currentMessage() == "Run cancelled — no output written."
    # an ordinary progress line (not an outcome) still gets the prefix as normal
    rn._set_status("file 1/2: breath 3/40")
    assert win.statusBar().currentMessage() == "Run: file 1/2: breath 3/40"
    win.close()


def test_analysis_menu_actions_show_feedback_regardless_of_active_tab(qapp, tmp_path):
    """The Analysis menu (Save/Save as/New/Open) lives in the header and is reachable from
    any tab, but its confirmation ('Saved …', guided-flow entry, etc.) is emitted by Setup —
    which the per-tab ownership rule would otherwise swallow unless Setup happens to be the
    active tab. Regression for the ticket's self-review finding."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    win.tabs.setCurrentWidget(win.preview_screen)  # looking at Preview & QC (hosts the Run
                                                    # drawer since B03), not Setup
    win._new_analysis()
    # New analysis's own feedback moved to the guidance label (guided flow), so this is a
    # neutral fallback rather than a blank bar — the real regression check is that it is
    # NOT still showing whatever Run's own status happened to be.
    assert win.statusBar().currentMessage() == "Ready."
    win._on_noise_reference_changed("synth_case_A.csv", [[1.0, 2.0]], False)
    assert "noise reference set" in win.statusBar().currentMessage().lower()
    win.close()


def test_refresh_files_skips_the_pick_one_line_when_already_drawn(qapp, tmp_path):
    """A plain revisit of the Preview tab (main_window._on_tab_changed calls refresh_files
    on every entry, see test_reactive_file_list_and_noise_gating above and A03) must not
    reset the status line over an already-selected, already-drawn file's own result."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(
        win.state.settings, os.path.join(INPUT, "synth_case_A.csv")))
    pv._set_status("A CUSTOM RESULT LINE")
    pv.refresh_files()                       # simulate a plain tab revisit
    assert pv.status.text() == "A CUSTOM RESULT LINE"
    win.close()


def test_refresh_files_singular_and_plural_wording(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_settings(str(tmp_path))))
    pv = win.preview_screen
    pv.refresh_files()                        # 2 synthetic files -> plural, "pick one"
    assert pv.status.text() == "2 files — pick one; everything runs automatically."
    win.state.settings.input.files = "synth_case_A.csv"    # narrow the mask to exactly one
    pv.refresh_files()
    assert pv.status.text() == "1 file — everything runs automatically."
    win.close()


def test_checkboxes_show_a_box_in_both_states(qapp):
    """A deselected checkable option must read as an empty checkbox, not plain text:
    the theme styles QCheckBox::indicator (a visible box unchecked, accent + tick when
    checked) rather than leaving Fusion's checkmark-only default."""
    from respmech.ui import theme
    qss = qapp.styleSheet()
    assert "QCheckBox::indicator" in qss                 # the box is styled at all…
    assert "QCheckBox::indicator:checked" in qss         # …and the checked state
    assert "border" in qss.split("QCheckBox::indicator")[1][:120]   # unchecked draws a border
    assert theme._check_icon_path() != ""                # the tick image is available

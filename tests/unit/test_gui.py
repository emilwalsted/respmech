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
    assert win.tabs.count() == 3
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
    assert pv.file_combo.count() == 2
    pv.file_combo.setCurrentIndex(0)
    pv._preview()
    assert "breaths" in pv.status.text()
    # the mechanics test run is automatic (async); its handler fills the table + Campbell
    pv._on_batch_result(run_batch(win.state.settings, only_files=["synth_case_A.csv"]))
    assert pv.table.rowCount() > 0 and pv.table.columnCount() > 0


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
    assert splash._MONO in installed and splash._FONT in installed
    assert "," not in splash._MONO and "," not in splash._FONT   # single families, not stacks


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
    assert pv.file_combo.count() == 2
    # changing the mask in Settings refreshes the Preview file list (reactive)
    sc.in_files.setText("synth_case_A.csv"); sc._on_inputs_changed()
    assert pv.file_combo.count() == 1
    sc.in_files.setText("synth_case_*.csv"); sc._on_inputs_changed()
    assert pv.file_combo.count() == 2
    # noise on but no reference -> the noise-window options are disabled + a hint
    sc.remove_noise.setChecked(True); sc.noise_ref.setText(""); sc._on_field_changed()
    pv.file_combo.setCurrentIndex(0); pv._update_actions()
    assert pv.noise_opts.isEnabled() is False
    assert "reference" in pv.status.text().lower()
    # setting a reference enables them
    sc.noise_ref.setText("synth_case_A.csv"); sc._on_field_changed(); pv._update_actions()
    assert pv.noise_opts.isEnabled() is True


def test_remove_noise_checkbox_binds_noise_enabled(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(_settings(str(tmp_path))))
    sc.remove_noise.setChecked(True)
    sc.to_state()
    # the checkbox drives the real gate (noise.enabled) and the legacy mirror
    assert sc.state.settings.processing.emg.noise.enabled is True
    assert sc.state.settings.processing.emg.remove_noise is True


def test_empty_input_folder_is_handled(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _settings(str(tmp_path))
    s.input.folder = str(tmp_path / "does_not_exist")
    win = MainWindow(AppState(s))
    win.preview_screen.refresh_files()
    assert win.preview_screen.file_combo.count() == 0
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

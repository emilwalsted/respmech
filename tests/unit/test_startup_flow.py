"""The startup flow: the New/Open chooser + the guided progressive-disclosure of the
Settings screen and the downstream (Preview/Run) tabs.

New-analysis mode reveals the settings cards stage by stage (Input -> Output -> the
rest) and keeps Preview/Run hidden until everything validates; opening an existing
analysis reveals everything at once and validates. The guided flow is opt-in
(``MainWindow.begin_session`` / ``SettingsScreen.enter_new_mode``) so plain
construction keeps full access, which every other GUI test relies on.
"""
import os

import pytest




from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.ui.state import AppState  # noqa: E402


from _helpers import INPUT, requires_synth  # noqa: F401

pytestmark = requires_synth()




def _fill_valid_input(sc):
    sc.in_folder.setText(INPUT)
    sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()


def _fill_valid_output(sc, out_dir):
    sc.out_folder.setText(str(out_dir))
    sc._on_field_changed()


def _fill_valid_channels(sc):
    """Through the picker's own commit path — the channel fields are gone."""
    sc._apply_channel_mapping({"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9,
                               "emg": [2, 3, 4], "entropy": []})
    sc._on_field_changed()


def _pass_channel_step(sc, mapping=None):
    """Simulate the channel-assignment modal completing with a mapping (the real modal
    execs and can't run headless), mirroring _open_channel_setup_for_flow: mark the step
    done, apply the mapping, then let the disclosure reveal the rest of the cards."""
    sc._channel_modal_done = True
    sc._apply_channel_mapping(mapping or {"flow": 5, "volume": 6, "poes": 7, "pgas": 8,
                                          "pdi": 9, "emg": [2, 3, 4], "entropy": []})
    sc._update_disclosure()


def _shown(card):
    """The card's own visibility flag, independent of whether the (test) window is
    actually on screen — isVisible() would be False for every widget of an unshown
    top-level, so we ask whether it was explicitly hidden by the disclosure logic."""
    return not card.isHidden()


# --------------------------------------------------------------------------- #
# The chooser dialog
# --------------------------------------------------------------------------- #
def test_startup_dialog_new(qapp):
    from respmech.ui.startup_dialog import StartupDialog
    dlg = StartupDialog()
    dlg._choose_new()
    assert dlg.mode == "new" and dlg.path is None


def test_startup_dialog_open_uses_the_picked_file(qapp, monkeypatch, tmp_path):
    from respmech.ui import startup_dialog
    picked = str(tmp_path / "my_analysis.toml")
    monkeypatch.setattr(startup_dialog.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (picked, "")))
    dlg = startup_dialog.StartupDialog()
    dlg._choose_open()
    assert dlg.mode == "open" and dlg.path == picked


def test_startup_dialog_open_cancelled_keeps_chooser_open(qapp, monkeypatch):
    from respmech.ui import startup_dialog
    monkeypatch.setattr(startup_dialog.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    dlg = startup_dialog.StartupDialog()
    dlg._choose_open()
    assert dlg.mode == "new" and dlg.path is None      # unchanged; dialog would stay open


# --------------------------------------------------------------------------- #
# Terminology + default access
# --------------------------------------------------------------------------- #
def test_analysis_menu_uses_analysis_terminology_not_toml(qapp):
    """The analysis-file actions live in the header's Analysis menu — not under Setup — so
    they are reachable from every tab. "Analysis" is the user-facing name, never "TOML"."""
    from PySide6.QtWidgets import QPushButton
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    labels = [a.text() for a in win.analysis_btn.menu().actions()]
    assert "New analysis" in labels
    assert "Open analysis…" in labels
    assert "Save" in labels and "Save as…" in labels
    assert not any("TOML" in t or "legacy .py" in t for t in labels)
    # and they are gone from the Setup step's own buttons
    setup = [b.text() for b in win.settings_screen.findChildren(QPushButton)]
    assert not any(t in setup for t in ("New analysis", "Open analysis…", "Save", "Save as…", "Validate"))
    win.close()


def test_plain_construction_keeps_full_access(qapp):
    """No begin_session/enter_new_mode -> every tab visible (what all other GUI
    tests depend on)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    assert win.tabs.isTabEnabled(win._i_preview)
    assert win.tabs.isTabEnabled(win._i_run)
    win.close()


# --------------------------------------------------------------------------- #
# Guided new-analysis flow
# --------------------------------------------------------------------------- #
def test_new_mode_shows_only_input_and_hides_downstream(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    inp, out, rest = sc._stage_cards[0][0], sc._stage_cards[1][0], sc._stage_cards[2][0]
    assert _shown(inp) and not _shown(out) and not _shown(rest)
    assert not win.tabs.isTabEnabled(win._i_preview)
    assert not win.tabs.isTabEnabled(win._i_run)
    win.close()


def test_output_card_reordered_second(qapp):
    """The Output card is the second stage (right after Input), per the flow."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    assert sc._stage_cards[1][0].title() == "Output"
    win.close()


def test_new_analysis_defaults_the_file_mask_to_csv_and_txt(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert sc.in_files.text() == "*.csv; *.txt"
    assert sc.state.settings.input.files == "*.csv; *.txt"
    win.close()


def test_guided_status_points_at_mask_when_it_matches_nothing(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()                            # sets the *.txt default
    sc.in_folder.setText(str(tmp_path))            # a real folder with NO .txt files
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    assert "no files match" in sc.status.text().lower()   # not the generic prompt
    win.close()


def test_progressive_reveal_then_unlock_downstream(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    out, rest = sc._stage_cards[1][0], sc._stage_cards[2][0]
    sc.enter_new_mode()
    assert not _shown(out) and not _shown(rest)

    _fill_valid_input(sc)                     # Input valid -> Output appears
    assert _shown(out) and not _shown(rest)
    assert not win.tabs.isTabEnabled(win._i_preview)

    _fill_valid_output(sc, tmp_path)          # Output valid -> the channel modal gates the rest
    assert not _shown(rest)
    assert "assign your data channels" in sc.status.text().lower()
    assert not win.tabs.isTabEnabled(win._i_preview)

    _pass_channel_step(sc)                    # modal OK -> channels applied -> rest + unlock
    assert _shown(rest)
    assert sc._all_ok()
    assert win.tabs.isTabEnabled(win._i_preview)
    assert win.tabs.isTabEnabled(win._i_run)
    win.close()


def test_reveal_is_monotonic_but_downstream_relocks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    out, rest = sc._stage_cards[1][0], sc._stage_cards[2][0]
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _pass_channel_step(sc)
    assert win.tabs.isTabEnabled(win._i_preview)

    sc.in_folder.setText("/nonexistent-xyz-123"); sc._on_inputs_changed()
    # cards stay revealed (monotonic, no jarring retraction)...
    assert _shown(out) and _shown(rest)
    # ...but the downstream tabs re-lock because the settings are no longer valid
    assert not win.tabs.isTabEnabled(win._i_preview)
    assert not win.tabs.isTabEnabled(win._i_run)
    win.close()


# --------------------------------------------------------------------------- #
# Open an existing analysis
# --------------------------------------------------------------------------- #
def test_open_mode_reveals_everything(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert not win.tabs.isTabEnabled(win._i_preview)
    sc.enter_open_mode()
    for stage in sc._stage_cards:
        for card in stage:
            assert _shown(card)
    assert win.tabs.isTabEnabled(win._i_preview)
    assert win.tabs.isTabEnabled(win._i_run)
    win.close()


def test_open_analysis_from_saved_toml_roundtrips(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    # build a valid analysis and save it to a .toml
    win = MainWindow(AppState())
    sc = win.settings_screen
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _fill_valid_channels(sc)
    p = tmp_path / "saved.toml"
    sc.state.save_toml(str(p))
    win.close()

    # a fresh window opens it -> everything revealed and the fields are restored
    win2 = MainWindow(AppState())
    sc2 = win2.settings_screen
    sc2.enter_new_mode()
    sc2.open_analysis(str(p))
    assert sc2.in_folder.text() == INPUT
    assert [t for t in sc2.channel_summary.texts() if t.startswith("EMG  ·  Column")] != []
    assert win2.tabs.isTabEnabled(win2._i_preview)
    assert "valid" in sc2.status.text().lower()
    win2.close()


def test_open_analysis_routes_by_extension(qapp, monkeypatch):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    calls = []
    monkeypatch.setattr(sc, "_load", lambda path=None: calls.append(("toml", path)))
    monkeypatch.setattr(sc, "_import", lambda path=None: calls.append(("py", path)))
    sc.open_analysis("/some/where/a.toml")
    sc.open_analysis("/some/where/b.PY")            # case-insensitive
    assert calls == [("toml", "/some/where/a.toml"), ("py", "/some/where/b.PY")]
    win.close()


# --------------------------------------------------------------------------- #
# begin_session orchestration
# --------------------------------------------------------------------------- #
class _FakeChooser:
    def __init__(self, mode, path=None):
        self.mode, self.path = mode, path

    def exec(self):
        return 1


def test_begin_session_new_enters_guided_mode(qapp, monkeypatch):
    from respmech.ui import startup_dialog
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    monkeypatch.setattr(startup_dialog, "StartupDialog", lambda parent=None: _FakeChooser("new"))
    win.begin_session()
    assert win.settings_screen._mode == "new"
    assert not win.tabs.isTabEnabled(win._i_preview)
    win.close()


def test_begin_session_open_loads_and_reveals(qapp, monkeypatch, tmp_path):
    from respmech.ui import startup_dialog
    from respmech.ui.main_window import MainWindow
    # save a valid analysis to open
    seed = MainWindow(AppState())
    _fill_valid_input(seed.settings_screen)
    _fill_valid_output(seed.settings_screen, tmp_path)
    _fill_valid_channels(seed.settings_screen)
    p = tmp_path / "seed.toml"
    seed.settings_screen.state.save_toml(str(p))
    seed.close()

    win = MainWindow(AppState())
    monkeypatch.setattr(startup_dialog, "StartupDialog",
                        lambda parent=None: _FakeChooser("open", str(p)))
    win.begin_session()
    assert win.settings_screen._mode == "full"
    assert win.tabs.isTabEnabled(win._i_preview)
    win.close()


class _StubChannelDialog:
    """Stand-in for the real (exec-blocking) ChannelSetupDialog."""
    calls = 0

    def __init__(self, *a, accept=False, mapping=None, **k):
        self._accept = accept
        self._mapping = mapping or {"flow": 5, "volume": 6, "poes": 7, "pgas": 8,
                                    "pdi": 9, "emg": [2, 3, 4], "entropy": []}

    def exec(self):
        from PySide6.QtWidgets import QDialog
        type(self).calls += 1
        return QDialog.Accepted if self._accept else QDialog.Rejected

    def selected_mapping(self):
        return dict(self._mapping)


def test_channel_modal_auto_opens_exactly_once_after_output(qapp, tmp_path, monkeypatch):
    import respmech.ui.channel_setup_dialog as csd
    from respmech.ui.main_window import MainWindow
    _StubChannelDialog.calls = 0
    monkeypatch.setattr(csd, "ChannelSetupDialog",
                        lambda *a, **k: _StubChannelDialog(*a, accept=False, **k))
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc)
    _fill_valid_output(sc, tmp_path)               # Output valid -> schedule the one-shot
    assert sc._channel_modal_pending is True
    qapp.processEvents()                           # fire the QTimer.singleShot -> real open path
    assert _StubChannelDialog.calls == 1
    assert sc._channel_modal_done is True and sc._channel_modal_pending is False
    sc._update_disclosure(); qapp.processEvents()  # a further update must NOT re-open it
    assert _StubChannelDialog.calls == 1
    win.close()


def test_channel_modal_cancel_reveals_rest_but_keeps_downstream_locked(qapp, tmp_path, monkeypatch):
    import respmech.ui.channel_setup_dialog as csd
    from respmech.ui.main_window import MainWindow
    monkeypatch.setattr(csd, "ChannelSetupDialog",
                        lambda *a, **k: _StubChannelDialog(*a, accept=False, **k))
    win = MainWindow(AppState())
    sc = win.settings_screen
    rest = sc._stage_cards[2][0]
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path)
    qapp.processEvents()                           # auto-open -> cancelled
    assert _shown(rest)                            # the rest reveals (non-trapping)...
    assert not win.tabs.isTabEnabled(win._i_preview)   # ...but stays locked (channels unset)
    assert not win.tabs.isTabEnabled(win._i_run)
    win.close()


def test_completion_status_surfaces_a_science_note(qapp, tmp_path):
    """The soft science guardrails (EMG/pressure column overlap; low fs) should appear at the
    end of the guided flow, not only when the user clicks Validate."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path)
    _pass_channel_step(sc, {"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9,
                            "emg": [5], "entropy": []})     # EMG column 5 == the flow column
    assert sc._all_ok()
    assert "overlap" in sc.status.text().lower()            # surfaced without clicking Validate
    win.close()


def test_reentry_rearms_the_channel_gate(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _pass_channel_step(sc)
    assert sc._channel_modal_done is True
    sc.enter_new_mode()                            # start over -> gate must re-arm
    assert sc._channel_modal_done is False and sc._channel_modal_pending is False
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path)
    assert not _shown(sc._stage_cards[2][0])       # rest re-held for a fresh modal
    assert sc._channel_modal_pending is True
    win.close()


def test_begin_session_with_cli_path_opens_directly(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    win.settings_screen.enter_new_mode()
    win.begin_session(cli_path="/some/preloaded.toml")   # already loaded in __init__
    assert win.settings_screen._mode == "full"
    assert win.tabs.isTabEnabled(win._i_preview)
    win.close()


def test_cancelled_chooser_falls_through_to_new(qapp, monkeypatch):
    """Cancelling the REAL startup chooser (exec → reject, default mode 'new') must drop
    into the guided New flow — the genuine composed path, not a re-run of the 'new'
    branch (that is test_begin_session_new_enters_guided_mode)."""
    from respmech.ui import startup_dialog
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())

    class _Cancelled(startup_dialog.StartupDialog):
        def exec(self):
            self.reject()                       # the user cancels; mode stays the default "new"
            return 0
    monkeypatch.setattr(startup_dialog, "StartupDialog", _Cancelled)
    win.begin_session()
    assert win.settings_screen._mode == "new" and win.settings_screen.state.settings_path is None
    assert not win.tabs.isTabEnabled(win._i_preview)
    win.close()


# --------------------------------------------------------------------------- #
# Volume-optional (integrate from flow) path
# --------------------------------------------------------------------------- #
def test_integrate_from_flow_unlocks_without_a_volume_column(qapp, tmp_path):
    """Ticking 'Calculate volume from flow' makes the volume column optional, so the
    guided flow must still unlock Preview/Run with volume left at 0."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path)
    # complete the channel step with NO volume column assigned...
    _pass_channel_step(sc, {"flow": 5, "volume": None, "poes": 7, "pgas": 8,
                            "pdi": 9, "emg": [2, 3, 4], "entropy": []})
    assert not sc._all_ok()                # volume missing + integrate off -> not valid
    assert not win.tabs.isTabEnabled(win._i_preview)
    assert "volume" in sc.status.text().lower()
    # ...ticking 'Calculate volume from flow' makes the volume column optional -> unlock
    sc.integrate.setChecked(True); sc._on_field_changed()
    assert sc._all_ok()
    assert win.tabs.isTabEnabled(win._i_preview)
    win.close()


# --------------------------------------------------------------------------- #
# Guided-flow status wording
# --------------------------------------------------------------------------- #
def test_guided_status_messages_per_stage(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert sc.status.text() == "New analysis — fill in the Input card to begin."
    _fill_valid_input(sc)
    assert sc.status.text() == "Input set — now choose an Output folder."
    _fill_valid_output(sc, tmp_path)
    assert sc.status.text() == "Output set — assign your data channels to continue."
    _pass_channel_step(sc)
    assert sc.status.text() == "Settings complete — Preview and Run are now available."
    win.close()


# --------------------------------------------------------------------------- #
# Re-entering the guided flow
# --------------------------------------------------------------------------- #
def test_reenter_new_mode_after_completing_collapses_and_blanks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _pass_channel_step(sc)
    assert win.tabs.isTabEnabled(win._i_preview)          # completed once
    sc.enter_new_mode()                                   # start over
    assert _shown(sc._stage_cards[0][0])                  # Input shown again
    assert not _shown(sc._stage_cards[1][0]) and not _shown(sc._stage_cards[2][0])
    assert sc.in_folder.text() == "" and sc.out_folder.text() == ""
    assert not win.tabs.isTabEnabled(win._i_preview)
    assert not win.tabs.isTabEnabled(win._i_run)
    win.close()


# --------------------------------------------------------------------------- #
# Failure / rollback + invalid open
# --------------------------------------------------------------------------- #
def test_open_broken_toml_rolls_back_and_keeps_prior(qapp, tmp_path, monkeypatch):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    # a known-good prior analysis in memory
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _fill_valid_channels(sc)
    prior_settings = sc.state.settings
    # swallow the copyable error dialog so the test stays headless
    monkeypatch.setattr(sc, "_report_error", lambda *a, **k: None)
    bad = tmp_path / "broken.toml"
    bad.write_text("this is = not valid toml [[[\n")
    ok = sc.open_analysis(str(bad))
    assert ok is False                                     # signalled failure
    assert sc.state.settings is prior_settings            # rolled back, not clobbered
    assert sc.in_folder.text() == INPUT                   # form still reflects the prior
    win.close()


def test_open_invalid_analysis_reveals_all_but_warns(qapp, tmp_path):
    """A parseable analysis that fails validation still opens (all cards + tabs) but the
    status reports it as invalid."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import Settings
    from respmech.settingsio.toml_io import save_toml
    # volume None + integrate off -> parses but fails validate()
    s = Settings()
    s.input.folder = INPUT; s.input.files = "synth_case_*.csv"
    s.input.format.sampling_frequency = 1000
    s.input.channels.flow = 5; s.input.channels.poes = 7
    s.input.channels.pgas = 8; s.input.channels.pdi = 9
    s.input.channels.volume = None
    s.processing.volume.integrate_from_flow = False
    s.output.folder = str(tmp_path)
    p = tmp_path / "invalid.toml"
    save_toml(s, str(p))

    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert sc.open_analysis(str(p)) is True
    for stage in sc._stage_cards:
        for card in stage:
            assert _shown(card)
    assert win.tabs.isTabEnabled(win._i_preview)
    assert sc.status.text().lower().startswith("invalid")
    win.close()


def test_open_legacy_py_migrates_populates_and_reveals(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    legacy = tmp_path / "legacy_setup.py"
    legacy.write_text(
        "settings = {\n"
        f"  'input': {{'inputfolder': {INPUT!r}, 'files': 'synth_case_*.csv',\n"
        "    'format': {'samplingfrequency': 1000},\n"
        "    'data': {'column_poes':7,'column_pgas':8,'column_pdi':9,'column_volume':6,\n"
        "             'column_flow':5,'columns_emg':[2,3,4],'columns_entropy':[10,11,12]}},\n"
        "  'processing': {'mechanics': {'breathseparationbuffer':200,'separateby':'flow',\n"
        "                               'avgresamplingobs':300},\n"
        "                 'emg': {'remove_ecg': False, 'remove_noise': False}},\n"
        f"  'output': {{'outputfolder': {str(tmp_path)!r},\n"
        "    'data': {'saveaveragedata': True, 'savebreathbybreathdata': True}}}\n")
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert sc.open_analysis(str(legacy)) is True          # migrator ran + report shown
    assert sc.in_folder.text() == INPUT                   # migrated fields landed in the form
    assert sc.samp_freq.value() == 1000
    assert [t for t in sc.channel_summary.texts() if t.startswith("EMG  ·  Column")] != []
    for stage in sc._stage_cards:
        for card in stage:
            assert _shown(card)
    win.close()


def test_save_open_roundtrip_preserves_surfaced_fields(qapp, tmp_path):
    """A save -> open round-trip must restore every surfaced field, not just the ones the
    happy-path test checks (guards the ~20 to_state/from_state mappings)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _fill_valid_channels(sc)
    # set non-defaults across the groups
    sc._apply_channel_mapping({"flow": 5, "volume": 6, "poes": 7, "pgas": 8,
                               "pdi": 9, "emg": [2, 3, 4], "entropy": [10, 11, 12]})
    sc.seg_method.setCurrentText("Volume")        # display text; canonical token "volume" is userData
    sc.wob_from.setCurrentText("Individual")
    sc.emg_rms_window.setValue(0.1)
    sc.emg_outlier_sd.setValue(2.5)
    # ECG removal moved from the Settings screen to the Preview "› EMG – ECG reduction" tab;
    # set it on the shared model and verify it still round-trips through the TOML.
    e0 = sc.state.settings.processing.emg
    e0.remove_ecg = True; e0.ecg_window_s = 0.35; e0.ecg_min_distance_s = 0.6
    sc._on_field_changed()
    p = tmp_path / "full.toml"
    sc.state.save_toml(str(p))
    win.close()

    win2 = MainWindow(AppState())
    sc2 = win2.settings_screen
    assert sc2.open_analysis(str(p)) is True
    assert any("Entropy" in t and "Column 10" in t for t in sc2.channel_summary.texts())
    assert sc2.seg_method.currentData() == "volume"      # model token restored via userData
    assert sc2.wob_from.currentData() == "individual"
    assert abs(sc2.emg_rms_window.value() - 0.1) < 1e-9
    assert abs(sc2.emg_outlier_sd.value() - 2.5) < 1e-9
    e2 = sc2.state.settings.processing.emg          # ECG params persist at the model level
    assert e2.remove_ecg is True
    assert abs(e2.ecg_window_s - 0.35) < 1e-9
    assert abs(e2.ecg_min_distance_s - 0.6) < 1e-9
    win2.close()


# --------------------------------------------------------------------------- #
# Action-bar 'New analysis' confirmation
# --------------------------------------------------------------------------- #
def test_new_analysis_confirm_declined_is_noop_accepted_resets(qapp, tmp_path, monkeypatch):
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _fill_valid_channels(sc)
    kept = sc.in_folder.text()

    # cancel -> nothing changes (the guard offers Save/Discard/Cancel like every sibling)
    monkeypatch.setattr(ss.QMessageBox, "question",
                        staticmethod(lambda *a, **k: ss.QMessageBox.Cancel))
    sc.new_analysis()
    assert sc.in_folder.text() == kept and sc._mode == "full"

    # discard -> reset into a fresh guided flow with blanked folders
    monkeypatch.setattr(ss.QMessageBox, "question",
                        staticmethod(lambda *a, **k: ss.QMessageBox.Discard))
    sc.new_analysis()
    assert sc._mode == "new"
    assert sc.in_folder.text() == "" and sc.out_folder.text() == ""
    assert not win.tabs.isTabEnabled(win._i_preview)
    win.close()


# ---------------------------------------------------------------------------
# Startup doors + guided-flow gating (P23/P24/P25) — waves 1, 6
# ---------------------------------------------------------------------------
def test_cancelled_channel_modal_leaves_flow_gated(qapp, tmp_path, monkeypatch):
    """The cancel trap: cancelling the auto-opened modal must NOT unlock Preview/Run with
    every pressure defaulting to the time column."""
    import respmech.ui.channel_setup_dialog as csd
    from PySide6.QtWidgets import QDialog
    from respmech.ui.main_window import MainWindow

    class _Cancel:
        def __init__(self, *a, **k): pass
        def exec(self): return QDialog.Rejected
        def selected_mapping(self): return {}
    monkeypatch.setattr(csd, "ChannelSetupDialog", lambda *a, **k: _Cancel())
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000); sc._on_inputs_changed()
    sc.out_folder.setText(str(tmp_path)); sc._on_field_changed()
    qapp.processEvents()                                 # auto-open -> cancelled
    assert not sc._all_ok()                              # channels default to col 1 -> gated
    assert not win.tabs.isTabEnabled(win._i_preview)
    win.close()


def test_startup_sample_and_cancel(qapp, isolated_prefs):
    from respmech.ui.startup_dialog import StartupDialog
    dlg = StartupDialog()
    assert hasattr(dlg, "sample_btn")                    # the explore door (P23)
    dlg._choose_sample()
    assert dlg.mode == "sample"
    dlg2 = StartupDialog()
    dlg2.reject()                                        # Cancel leaves the safe default
    assert dlg2.mode == "new"


def test_downstream_tabs_locked_not_hidden(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()
    assert win.tabs.isTabVisible(win._i_preview)          # still VISIBLE (a stepper)…
    assert not win.tabs.isTabEnabled(win._i_preview)      # …but LOCKED until valid
    assert win.tabs.tabToolTip(win._i_preview)            # explains why
    assert sc.open_sample_analysis()
    assert win.tabs.isTabEnabled(win._i_preview) and win.tabs.isTabEnabled(win._i_run)
    win.close()


def test_new_from_last_rig(qapp, isolated_prefs):
    from respmech.core.settings import Settings
    from respmech.ui.main_window import MainWindow
    from respmech.ui.startup_dialog import StartupDialog
    s = Settings(); s.input.channels.poes = 7; s.input.channels.pdi = 9
    s.input.channels.emg = [2, 3, 4]; s.input.format.sampling_frequency = 800
    isolated_prefs.save_rig(s)
    dlg = StartupDialog()
    assert hasattr(dlg, "rig_btn")                        # offered because a rig exists
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode(use_last_rig=True)
    # The rig object is state.settings itself, so a model assertion here would pass
    # even with enter_new_mode's population and the whole summary dead.
    # a fresh AppState has no input folder, so there is no file to plot and the summary
    # uses its plain-list fallback
    rows = sc.channel_summary.texts()
    assert "Oesophageal pressure (Poes): Column #7" in rows
    assert "EMG: Columns #2, #3, #4" in rows
    assert sc.samp_freq.value() == 800
    assert not win.tabs.isTabEnabled(win._i_preview)      # still guided (folders blank)
    win.close()

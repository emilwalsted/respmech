"""The startup flow: the New/Open chooser + Setup's guided 'new analysis' mode.

Ticket B04 retired progressive disclosure: every Setup card (Input/Channels/Output) is
visible from the first frame in every mode, and the Preview & QC tab is never locked —
"it is the action that is closed, and the reason hangs on the action" (the Run drawer's
own commitment sheet, covered in test_run_screen.py). What survives from the old guided
flow: blanking the folders/mask for a fresh start, and a one-shot auto-open of the
channel-assignment modal once the input folder has matching files (no longer gated on the
Output card too). The guided flow is opt-in (``MainWindow.begin_session`` /
``SettingsScreen.enter_new_mode``) so plain construction keeps full access, which every
other GUI test relies on.
"""
import os

import pytest



from PySide6.QtWidgets import QApplication, QGroupBox  # noqa: E402

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
    execs and can't run headless), mirroring _open_channel_setup_for_flow's ACCEPT path:
    mark the step done and apply the mapping. No extra _update_disclosure() call here —
    _apply_channel_mapping already ran one (via _on_inputs_changed) before setting its
    own "Channels assigned…" status, and a redundant second call would immediately
    overwrite it with the generic validation status (see _open_channel_setup_for_flow's
    own docstring for why that only happens on cancel now)."""
    sc._channel_modal_done = True
    sc._apply_channel_mapping(mapping or {"flow": 5, "volume": 6, "poes": 7, "pgas": 8,
                                          "pdi": 9, "emg": [2, 3, 4], "entropy": []})


def _shown(card):
    """The card's own visibility flag, independent of whether the (test) window is
    actually on screen — isVisible() would be False for every widget of an unshown
    top-level, so we ask whether it was explicitly hidden."""
    return not card.isHidden()


def _card(sc, title):
    """A Setup card by its QGroupBox title — cards are no longer collected in any
    registry (B04 retired _stage_cards), so this is the one way left to find one."""
    return next(c for c in sc.findChildren(QGroupBox) if c.title() == title)


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
    """No begin_session/enter_new_mode -> every tab visible and enabled (B04: there is no
    locking mechanism left at all, so this holds trivially — kept as a regression name in
    case a future ticket reintroduces some form of gating)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    assert win.tabs.isTabEnabled(win._i_preview)
    win.close()


def test_every_tab_stays_enabled_over_an_empty_appstate(qapp):
    """Ticket B04 acceptance: from a new, empty analysis the user can click into Preview &
    QC (which hosts the Run drawer since B03) and get there — every tab is enabled in
    every state, guided or not."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    for enter in (lambda: None, sc.enter_new_mode, sc.enter_open_mode):
        enter()
        assert win.tabs.isTabEnabled(win._i_settings)
        assert win.tabs.isTabEnabled(win._i_preview)
    win.close()


# --------------------------------------------------------------------------- #
# Guided new-analysis flow
# --------------------------------------------------------------------------- #
def test_new_mode_shows_every_card_immediately(qapp):
    """B04: progressive disclosure is retired — Input, Channels and Output are all
    visible from the very first frame of a guided 'new analysis', not revealed stage by
    stage."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert _shown(_card(sc, "Input"))
    assert _shown(_card(sc, "Channels"))
    assert _shown(_card(sc, "Output"))
    win.close()


def test_setup_shows_the_assign_channels_button_from_the_first_frame(qapp):
    """Ticket B04 acceptance, verbatim: btn_assign_channels.isVisible() is true
    immediately after enter_new_mode() — needs a shown window, since isVisible() (unlike
    isHidden()) also depends on the whole ancestor chain actually being on screen."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    win.show()
    qapp.processEvents()
    assert sc.btn_assign_channels.isVisible()
    win.close()


def test_new_analysis_defaults_the_file_mask_to_csv_and_txt(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert sc.in_files.text() == "*.csv; *.txt"
    assert sc.state.settings.input.files == "*.csv; *.txt"
    win.close()


def test_status_points_at_mask_when_it_matches_nothing(qapp, tmp_path):
    """B04: the guided-flow guidance label is retired — the same live validation status
    Setup always had now reports this directly, sharing validation.path_problem's exact
    wording with the Run drawer's own commitment sheet. Channels are filled in FIRST so
    core validate() has nothing else to object to — otherwise (a fresh AppState has no
    channels assigned yet) validate() raises on the missing channels before ever reaching
    the path check, which is a real, separate fact and not what this test is about."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _pass_channel_step(sc)
    assert sc._all_ok()
    sc.in_files.setText("*.nomatch"); sc._on_inputs_changed()
    assert "no files match" in sc.status.text().lower()
    win.close()


def test_settings_becoming_valid_flips_flow_ready_and_tab_stays_enabled(qapp, tmp_path):
    """B04: flow_ready_changed still tracks the analysis's overall validity (used
    elsewhere, e.g. by a future screen), but nothing locks a tab on it any more — the
    Preview & QC tab is enabled throughout, before, during and after the guided flow, and
    stays enabled even if the settings go invalid again."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    assert win.tabs.isTabEnabled(win._i_preview)
    assert not sc._all_ok()

    _fill_valid_input(sc)
    assert win.tabs.isTabEnabled(win._i_preview)
    assert not sc._all_ok()                        # channels still unset

    _fill_valid_output(sc, tmp_path)
    assert win.tabs.isTabEnabled(win._i_preview)

    _pass_channel_step(sc)
    assert sc._all_ok()
    assert win.tabs.isTabEnabled(win._i_preview)

    # settings going invalid again must not re-lock anything — there is nothing left to lock
    sc.in_folder.setText("/nonexistent-xyz-123"); sc._on_inputs_changed()
    assert not sc._all_ok()
    assert win.tabs.isTabEnabled(win._i_preview)
    win.close()


# --------------------------------------------------------------------------- #
# Open an existing analysis
# --------------------------------------------------------------------------- #
def test_open_mode_shows_conditional_cards_that_apply(qapp):
    """Every staged card is unconditionally visible in every mode since B04 (see
    test_new_mode_shows_every_card_immediately) — the one thing 'open' still needs to get
    right is the RELEVANCE-based _cond_cards (Sample entropy), which a default AppState
    has no entropy channels for."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    sc.enter_open_mode()
    for card, predicate in sc._cond_cards:
        assert _shown(card) == predicate()
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

    # a fresh window opens it -> fields are restored and it validates
    win2 = MainWindow(AppState())
    sc2 = win2.settings_screen
    sc2.enter_new_mode()
    sc2.open_analysis(str(p))
    assert sc2.in_folder.text() == INPUT
    assert [t for t in sc2.channel_summary.texts() if t.startswith("EMG  ·  Column")] != []
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
    win.close()


class _StubChannelDialog:
    """Stand-in for the real (exec-blocking) ChannelSetupDialog."""
    calls = 0

    def __init__(self, *a, accept=False, mapping=None, integrate_from_flow=False, **k):
        self._accept = accept
        self._mapping = mapping or {"flow": 5, "volume": 6, "poes": 7, "pgas": 8,
                                    "pdi": 9, "emg": [2, 3, 4], "entropy": []}
        self._integrate_from_flow = integrate_from_flow

    def exec(self):
        from PySide6.QtWidgets import QDialog
        type(self).calls += 1
        return QDialog.Accepted if self._accept else QDialog.Rejected

    def selected_mapping(self):
        return dict(self._mapping)

    def integrate_from_flow(self):
        return self._integrate_from_flow


def test_channel_modal_auto_opens_exactly_once_after_input_matches_files(qapp, monkeypatch):
    """B04: the one-shot channel-modal auto-open now hangs on the input folder having
    matching files — NOT on the Output card too (the old disclosure-stage gate). Filling
    only Input, never Output, must still trigger it exactly once."""
    import respmech.ui.channel_setup_dialog as csd
    from respmech.ui.main_window import MainWindow
    _StubChannelDialog.calls = 0
    monkeypatch.setattr(csd, "ChannelSetupDialog",
                        lambda *a, **k: _StubChannelDialog(*a, accept=False, **k))
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc)                          # Output is left untouched on purpose
    assert sc._channel_modal_pending is True
    qapp.processEvents()                           # fire the QTimer.singleShot -> real open path
    assert _StubChannelDialog.calls == 1
    assert sc._channel_modal_done is True and sc._channel_modal_pending is False
    sc._update_disclosure(); qapp.processEvents()  # a further update must NOT re-open it
    assert _StubChannelDialog.calls == 1
    win.close()


def test_channel_modal_cancel_leaves_flow_not_ready_but_every_tab_reachable(qapp, tmp_path, monkeypatch):
    import respmech.ui.channel_setup_dialog as csd
    from respmech.ui.main_window import MainWindow
    monkeypatch.setattr(csd, "ChannelSetupDialog",
                        lambda *a, **k: _StubChannelDialog(*a, accept=False, **k))
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc)
    qapp.processEvents()                           # auto-open -> cancelled
    assert sc._channel_modal_done is True           # the one-shot never re-arms itself
    assert not sc._all_ok()                         # channels default to unset -> not ready
    assert win.tabs.isTabEnabled(win._i_preview)    # …but nothing was ever locked to begin with
    win.close()


def test_qc_strip_surfaces_a_science_note(qapp, tmp_path):
    """The soft science guardrails (EMG/pressure column overlap; low fs) live on the QC
    strip, not the status line — it is the ONE surface that always shows every current
    caution regardless of what just happened (the status line, by contrast, legitimately
    shows "Channels assigned…" right after the modal closes; the two must not fight over
    the same label, which is exactly why they are still two separate widgets)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path)
    _pass_channel_step(sc, {"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9,
                            "emg": [5], "entropy": []})     # EMG column 5 == the flow column
    assert sc._all_ok()
    assert "overlap" in sc.qc.text().lower()
    win.close()


def test_status_shows_channel_assignment_message_after_the_modal_completes(qapp, tmp_path):
    """Regression (the ticket B04 was built on top of): applying a channel mapping must
    leave ITS OWN 'Channels assigned…' message as the last word on the status line, not
    have it clobbered by the live-validation status _update_disclosure also sets — the two
    now share one label (the guidance label that used to keep them apart is retired), so
    the ordering inside _apply_channel_mapping (validate first, name the assignment last)
    is what keeps this honest."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path)
    _pass_channel_step(sc)
    assert "channels assigned" in sc.status.text().lower()
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
    _fill_valid_input(sc)                           # input alone is enough to re-trigger (B04)
    assert sc._channel_modal_pending is True
    sc._channel_modal_done = True                   # disarm the still-pending singleShot(0, ...)
                                                      # before it can fire against a REAL, unstubbed
                                                      # modal later in the run (hung the reviewer)
    win.close()


def test_begin_session_with_cli_path_opens_directly(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    win.settings_screen.enter_new_mode()
    win.begin_session(cli_path="/some/preloaded.toml")   # already loaded in __init__
    assert win.settings_screen._mode == "full"
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
    win.close()


# --------------------------------------------------------------------------- #
# Volume-optional (integrate from flow) path
# --------------------------------------------------------------------------- #
def test_integrate_from_flow_unlocks_without_a_volume_column(qapp, tmp_path):
    """Ticking 'Calculate volume from flow' makes the volume column optional, so the
    guided flow must still report the analysis ready with volume left at 0."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path)
    # complete the channel step with NO volume column assigned...
    _pass_channel_step(sc, {"flow": 5, "volume": None, "poes": 7, "pgas": 8,
                            "pdi": 9, "emg": [2, 3, 4], "entropy": []})
    assert not sc._all_ok()                # volume missing + integrate off -> not valid
    # the status line itself shows "Channels assigned…" right after the modal (a more
    # recent, more specific event) — check the underlying verdict directly instead of
    # the label, which is not what this test is about
    assert "volume" in sc._validation_status().lower()
    # ...ticking 'Calculate volume from flow' makes the volume column optional -> ready
    sc.state.settings.processing.volume.integrate_from_flow = True   # Preview-owned now
    sc._update_disclosure()
    assert sc._all_ok()
    win.close()


# --------------------------------------------------------------------------- #
# Re-entering the guided flow
# --------------------------------------------------------------------------- #
def test_reenter_new_mode_after_completing_blanks_the_folders(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.enter_new_mode()
    _fill_valid_input(sc); _fill_valid_output(sc, tmp_path); _pass_channel_step(sc)
    assert sc._all_ok()                                   # completed once
    sc.enter_new_mode()                                   # start over
    assert sc.in_folder.text() == "" and sc.out_folder.text() == ""
    assert not sc._all_ok()                               # blanked -> invalid again
    assert sc._channel_modal_done is False                # the gate re-armed too
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


def test_open_broken_toml_leads_with_a_plain_diagnosis(qapp, tmp_path):
    """Ticket D16: opening an invalid .toml used to pop the generic "Open analysis
    failed. The full detail below is copyable." over a 15-line traceback ending inside
    Python's own tomllib module — the user's mental model is "my analysis file", and the
    app never says TOML anywhere else in the interface. It must now lead with a
    plain-language diagnosis (naming the broken FILE, not the parser) and keep the
    traceback one click away, reusing the D01 collapsed-detail dialog pattern."""
    from respmech.ui.dialogs import TextViewerDialog
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    bad = tmp_path / "broken.toml"
    bad.write_text("this is = not valid toml [[[\n")
    ok = sc.open_analysis(str(bad))
    assert ok is False
    assert sc._err_dialog is not None
    intro = sc._err_dialog.intro_label.text()
    assert "broken.toml" in intro
    assert "not a valid analysis file" in intro
    assert "Your current analysis is unchanged" in intro
    assert "tomllib" not in intro and "Traceback" not in intro   # no raw parser leakage
    assert sc._err_dialog.details_btn is not None            # trace is one click away…
    assert sc._err_dialog.view.isVisibleTo(sc._err_dialog) is False   # …but starts hidden
    assert "Traceback" in sc._err_dialog.text()               # …and is still the FULL trace
    win.close()


def test_open_missing_toml_keeps_the_generic_surface(qapp, tmp_path):
    """A load failure that is NOT a malformed-TOML problem (here: the file does not
    exist — FileNotFoundError, not tomllib.TOMLDecodeError) must keep the ordinary,
    uncollapsed error surface: the diagnosis-first path is specific to "this file isn't
    valid TOML", not every possible way opening a file can fail."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    missing = str(tmp_path / "does_not_exist.toml")
    ok = sc.open_analysis(missing)
    assert ok is False
    assert sc._err_dialog is not None
    assert sc._err_dialog.details_btn is None                # ordinary, uncollapsed layout
    win.close()


def test_open_invalid_analysis_shows_cards_but_warns(qapp, tmp_path):
    """A parseable analysis that fails validation still opens (every card, per
    test_open_mode_shows_conditional_cards_that_apply) but the status reports it invalid."""
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
    for card, predicate in sc._cond_cards:     # shown iff they apply (this file has no EMG/entropy)
        assert _shown(card) == predicate()
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
    # mechanics + EMG conditioning are Preview-owned; set them on the shared model
    s0 = sc.state.settings.processing
    s0.segmentation.method = "volume"; s0.wob.calc_from = "individual"
    s0.emg.rms_window_s = 0.1; s0.emg.outlier_rms_sd_limit = 2.5
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
    p2 = sc2.state.settings.processing   # mechanics/EMG are Preview-owned; assert the model
    assert p2.segmentation.method == "volume" and p2.wob.calc_from == "individual"
    assert abs(p2.emg.rms_window_s - 0.1) < 1e-9 and abs(p2.emg.outlier_rms_sd_limit - 2.5) < 1e-9
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
    win.close()


# ---------------------------------------------------------------------------
# Startup doors + guided-flow gating (P23/P24/P25) — waves 1, 6
# ---------------------------------------------------------------------------
def test_cancelled_channel_modal_leaves_flow_gated(qapp, tmp_path, monkeypatch):
    """The cancel trap: cancelling the auto-opened modal must NOT unlock/ready Preview/Run
    with every pressure defaulting to the time column."""
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
    qapp.processEvents()                                 # auto-open -> cancelled
    sc.out_folder.setText(str(tmp_path)); sc._on_field_changed()
    assert not sc._all_ok()                              # channels default to col 1 -> gated
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
    assert not sc._all_ok()                               # still guided (folders blank)
    win.close()


# ---------------------------------------------------------------------------
# UI-overhaul ticket C03 — startup screen + the returning-user's stuck paths
# ---------------------------------------------------------------------------
def test_startup_dialog_skip_button_replaces_bare_cancel(qapp):
    """C03 point 1: the old bare "Cancel" was wired to reject(), which landed on the exact
    same outcome as the guided flow anyway (mode stays "new") — a user backing out with
    Esc to go find their file first was silently dropped into "New analysis" instead. The
    button is renamed to say what happens, and wired to the explicit choice rather than
    left as an implicit reject()."""
    from PySide6.QtWidgets import QDialog, QPushButton
    from respmech.ui.startup_dialog import StartupDialog
    dlg = StartupDialog()
    texts = [b.text() for b in dlg.findChildren(QPushButton)]
    assert "Skip — start a new analysis" in texts
    assert "Cancel" not in texts
    skip_btn = next(b for b in dlg.findChildren(QPushButton)
                    if b.text() == "Skip — start a new analysis")
    skip_btn.click()
    assert dlg.mode == "new" and dlg.path is None
    assert dlg.result() == QDialog.Accepted


def test_recent_label_is_shared_between_chooser_and_menu(qapp, isolated_prefs, tmp_path):
    """C03 point 3: the startup chooser's recent buttons and the header's Analysis menu
    must describe the SAME file the same way — both now go through prefs.recent_label
    (moved here from main_window._recent_label, which is gone)."""
    from PySide6.QtWidgets import QPushButton
    from respmech.ui import prefs
    from respmech.ui import main_window as mw
    from respmech.ui.startup_dialog import StartupDialog
    assert not hasattr(mw, "_recent_label")     # moved to prefs.recent_label, not duplicated
    p = tmp_path / "analysis.toml"; p.write_text("# analysis")
    isolated_prefs.add_recent_analysis(str(p))
    dlg = StartupDialog()
    expected = prefs.recent_label(str(p))
    recent_btn = next(b for b in dlg.findChildren(QPushButton) if b.text() == expected)
    assert recent_btn.toolTip() == str(p)
    assert recent_btn.property("recent") is True    # left-aligned via theme.py's QSS rule


def test_get_started_and_explore_sample_are_repeatable_from_the_menu(qapp, isolated_prefs):
    """C03 point 2: before this ticket, "Explore with sample data" and the startup chooser
    were reachable ONLY from the very first window of a session — open_sample_analysis had
    exactly one caller (begin_session), so a user who dismissed the sample could never see
    it again in the same process. Both are now real, repeatable menu actions."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    labels = [a.text() for a in win.analysis_btn.menu().actions()]
    assert "Get started…" in labels
    assert "Explore with sample data" in labels
    assert "Duplicate for another recordings folder…" in labels
    # the exact same QAction objects also sit in the File menu (C01 pattern) — never a
    # second implementation that could quietly disagree
    file_labels = [a.text() for a in win._file_menu.actions()]
    assert "Get started…" in file_labels
    assert win._act_sample.text() in file_labels
    assert win._act_get_started in win._file_menu.actions()
    # exercise the door itself: a fresh window has nothing to lose, so no guard fires
    win._explore_sample()
    assert win.settings_screen.state.is_sample is True
    # dismissible and reachable again — this is the whole point of the ticket
    win.settings_screen.new_analysis()
    assert win.settings_screen.state.is_sample is False
    win._explore_sample()
    assert win.settings_screen.state.is_sample is True
    win.close()


def test_open_analysis_dialog_and_import_start_in_the_open_analysis_folder(
        qapp, isolated_prefs, tmp_path, monkeypatch):
    """C03 point 4: before this ticket both started the file picker at a bare '.' — the
    PROCESS's cwd, unrelated to the user's data in a packaged .app/.msi. They now start in
    the CURRENTLY OPEN analysis's own folder (mirroring _browse's existing precedence),
    falling back to the sticky "analysis" folder when nothing is open."""
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    study_dir = tmp_path / "Study" / "S02"
    study_dir.mkdir(parents=True)
    analysis_path = study_dir / "analysis.toml"
    sc.state.settings_path = str(analysis_path)          # simulate an already-open analysis

    seen = {}
    def _fake_open(*a, **k):
        seen["start"] = a[2] if len(a) > 2 else k.get("dir")
        return ("", "")
    monkeypatch.setattr(ss.QFileDialog, "getOpenFileName", staticmethod(_fake_open))
    sc.open_analysis_dialog()
    assert seen["start"] == str(study_dir)

    seen.clear()
    sc._import()
    assert seen["start"] == str(study_dir)

    # nothing open -> falls back to the sticky "analysis" folder, never a bare "."
    sc.state.settings_path = None
    isolated_prefs.set_last_folder("analysis", str(tmp_path))
    seen.clear()
    sc.open_analysis_dialog()
    assert seen["start"] == str(tmp_path)
    win.close()


def test_legacy_import_marks_the_analysis_dirty(qapp, tmp_path):
    """C03 point 7: a just-converted legacy analysis used to be marked CLEAN — indistinct
    from an untouched save, so closing over it (or opening another analysis) asked
    nothing even though the conversion had never been written to a .toml of its own."""
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
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    assert sc.open_analysis(str(legacy)) is True
    assert sc.is_dirty() is True
    assert sc.state.display_name == "legacy_setup"
    assert sc.state.legacy_source_path == str(legacy)
    assert "legacy_setup" in win.windowTitle()
    assert "* (modified)" in win.windowTitle()
    win.close()


def test_sample_analysis_is_labelled_everywhere(qapp, isolated_prefs):
    """C03 point 8: nothing on screen used to distinguish the built-in sample from a real
    analysis, and it writes into a temp folder the OS may clean up. Checks the three
    on-screen tells: the Setup banner, the window title, and that a real save clears
    is_sample (a saved-and-reopened sample analysis is a plain analysis from then on)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    assert sc.open_sample_analysis() is True
    assert sc.state.is_sample is True
    assert _shown(sc.sample_banner)          # not isVisible(): the window is never shown here
    assert "Sample analysis (built-in)" in win.windowTitle()
    # a NEW analysis clears the flag and hides the banner
    sc.new_analysis()
    assert sc.state.is_sample is False
    assert not _shown(sc.sample_banner)
    win.close()


def test_open_sample_analysis_shows_a_busy_cursor_while_building(qapp, monkeypatch):
    """Ticket D28: building the sample recording is a synchronous ~1 s stall (mostly a
    lazy import, measured), previously with no status text or wait cursor, so the window
    appeared to freeze. Intercepts the synchronous build call to check the wait cursor and
    status text are up WHILE it runs, and the cursor is gone again once the call returns."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import respmech.core.sample as sample_mod
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    real_write = sample_mod.write_sample_recording
    seen = {}

    def _spy(*a, **k):
        cur = QApplication.overrideCursor()
        seen["shape"] = cur.shape() if cur is not None else None
        seen["status"] = sc.status.text()
        return real_write(*a, **k)
    monkeypatch.setattr(sample_mod, "write_sample_recording", _spy)
    assert sc.open_sample_analysis() is True
    assert seen["shape"] == Qt.WaitCursor
    assert seen["status"] == "Building the sample recording…"
    assert QApplication.overrideCursor() is None      # restored once the call returns
    win.close()


def test_open_sample_analysis_restores_the_cursor_even_when_building_fails(qapp, monkeypatch):
    """The except branch opens a non-modal error dialog — a wait cursor still hanging
    over the window behind it would be worse than the freeze this feature exists to
    fix, so the restore must happen in a finally, not only on the success path.

    Self-review finding (ticket D28): a version of this test that only checked the
    cursor was gone AFTER the call would pass just as well if the whole busy-cursor
    feature had been reverted (no cursor is ever set -> trivially None afterwards).
    Capturing the shape INSIDE the failing call proves the cursor really was up when
    the exception hit, so this genuinely exercises the finally path, not just its
    absence."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import respmech.core.sample as sample_mod
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    seen = {}

    def _boom(*a, **k):
        cur = QApplication.overrideCursor()
        seen["shape"] = cur.shape() if cur is not None else None
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(sample_mod, "write_sample_recording", _boom)
    assert sc.open_sample_analysis() is False
    assert seen["shape"] == Qt.WaitCursor      # the cursor really was up when it failed
    assert QApplication.overrideCursor() is None
    assert "Explore sample data failed" in sc.status.text()
    win.close()

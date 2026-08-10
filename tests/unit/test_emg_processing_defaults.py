"""EMG processing is off until asked for, and noise reduction waits for ECG removal.

Two requirements that pull in opposite directions, so they are tested together.

"Off by default" is a NEGATIVE requirement: the dataclass defaults already say False, so the
work is to make sure nothing quietly turns them on when EMG channels appear. The tempting
hook — _update_emg_tab_visibility, which runs on every settings change — would flip a saved
analysis's remove_ecg from true to false the moment it was opened, and persist that on the
next Save: a change to computed output the user never made.

The ECG dependency is the opposite: not a hard requirement (the core will denoise a raw
signal quite happily) but a scientific one. Noise reduction runs on whatever the ECG stage
produced and its reference clip is ECG-cleaned, so with the heartbeat still present the
profile models a large periodic artefact as steady background noise.
"""
import pytest

from respmech.core.settings import Settings
from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


# -- off by default, and nothing turns it on --------------------------------------
def test_the_dataclass_defaults_are_off():
    e = Settings().processing.emg
    assert e.remove_ecg is False
    assert e.noise.enabled is False


def test_assigning_emg_channels_does_not_enable_any_processing(qapp, tmp_path):
    """The whole point. Nothing may react to EMG channels appearing by switching processing
    on — not the tab visibility, not the sync, not the screen construction."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), data_out=_OUT)
    s.processing.emg.remove_ecg = False
    s.processing.emg.noise.enabled = False
    s.input.channels.emg = []
    win = MainWindow(AppState(s))
    sc, pv = win.settings_screen, win.preview_screen

    sc._apply_channel_mapping({"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9,
                               "emg": [2, 3, 4], "entropy": []})
    pv.sync_from_settings()
    qapp.processEvents()

    e = pv.state.settings.processing.emg
    assert e.remove_ecg is False, "assigning EMG channels switched ECG removal on"
    assert e.noise.enabled is False, "assigning EMG channels switched noise reduction on"
    win.close()


def test_opening_an_analysis_keeps_the_processing_it_was_saved_with(qapp, tmp_path):
    """The dangerous direction: a hook that forces the defaults would rewrite a saved
    analysis on open, and persist it on the next Save."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), data_out=_OUT)
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    win = MainWindow(AppState(s))
    win.preview_screen.sync_from_settings()
    qapp.processEvents()
    e = win.preview_screen.state.settings.processing.emg
    assert e.remove_ecg is True, "opening the analysis reset ECG removal"
    assert e.noise.enabled is True, "opening the analysis reset noise reduction"
    assert not win.settings_screen.is_dirty(), "merely opening it marked it modified"
    win.close()


# -- noise reduction waits for ECG removal ----------------------------------------
def _preview(qapp, tmp_path, ecg):
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    s.processing.emg.remove_ecg = ecg
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv._update_actions()
    qapp.processEvents()
    return pv


def test_the_noise_controls_are_inert_until_ecg_removal_is_on(qapp, tmp_path):
    from respmech.ui.screens.preview_screen import NEEDS_ECG_HINT
    pv = _preview(qapp, tmp_path, ecg=False)
    assert not pv.btn_set_noise.isEnabled()
    assert not pv.noise_opts.isEnabled()
    for w in (pv.btn_set_noise, pv.noise_opts):
        assert NEEDS_ECG_HINT in w.toolTip(), "disabled with no explanation"
    pv.shutdown()


def test_they_come_alive_the_moment_ecg_removal_is_switched_on(qapp, tmp_path):
    """Regression: the gate lives one strip away from the checkbox that drives it, and
    _on_ecg_param_changed did not refresh enablement — so the controls stayed greyed out
    immediately after the user had done the thing the tooltip asked for."""
    pv = _preview(qapp, tmp_path, ecg=False)
    assert not pv.btn_set_noise.isEnabled()
    pv.remove_ecg.setChecked(True)              # the real user gesture, on the ECG strip
    qapp.processEvents()
    assert pv.btn_set_noise.isEnabled(), "still disabled after Remove ECG was turned on"
    assert pv.noise_opts.isEnabled()
    assert pv.btn_set_noise.toolTip() == "", "the stale explanation was left behind"
    pv.shutdown()


def test_the_status_line_says_why_rather_than_only_the_tooltip(qapp, tmp_path):
    """A disabled control with the reason hidden behind a hover is not much of a reason."""
    from respmech.ui.screens.preview_screen import NEEDS_ECG_HINT
    pv = _preview(qapp, tmp_path, ecg=False)
    assert NEEDS_ECG_HINT in pv.status.text()
    pv.shutdown()


def test_the_gate_is_ui_policy_and_not_a_core_block(qapp, tmp_path):
    """The core denoises a raw signal without complaint — it gates only on noise.enabled and
    the presence of EMG channels. Pinning that keeps the UI rule honest about what it is: a
    scientific guard rail, not a technical impossibility."""
    from respmech.core.pipeline import run_batch
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    s.processing.emg.remove_ecg = False          # noise on, ECG removal off
    result = run_batch(s)
    assert result.failed_files == {}, result.failed_files
    assert result.ok_files


# -- one writer for the noise reference ------------------------------------------
# R3/R4. The reference used to be settable in two places that disagreed in two ways.
# Setup's Browse stored an ABSOLUTE path while the picker stores a bare file name; both
# "worked" only because the core resolves with os.path.join, which ignores its base when the
# second argument is absolute — so a Setup-written analysis embedded a machine-specific path
# and stopped being portable. And Setup's "from expiration" checkbox was silently unticked by
# the picker whenever a span was marked, then — re-ticked — silently made that span inert.

def test_setup_cannot_write_the_noise_reference(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(synth_settings(str(tmp_path), noise=True, data_out=_OUT)))
    assert not hasattr(sc, "noise_ref"), "the absolute-path writer is back"
    assert not hasattr(sc, "noise_use_exp"), "the contradicting checkbox is back"


def test_to_state_does_not_touch_the_reference_fields(qapp, tmp_path):
    """to_state runs on every tab change, so a field it rewrote from a widget it no longer
    has would be erased on the first switch away from Setup."""
    from respmech.ui.screens.settings_screen import SettingsScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    sc = SettingsScreen(AppState(s))
    for _ in range(3):
        n = sc.to_state().processing.emg.noise
    assert n.reference_file == "synth_case_A.csv"
    assert n.reference_intervals == [[1.0, 5.0]]
    assert n.use_expiration is False


def test_the_preview_strip_shows_which_part_of_the_reference_is_used(qapp, tmp_path):
    """Read the tooltip, not the rendered text: the read-out is ELIDED to a fixed pixel budget
    (flow_layout.elide), so how much of it survives depends on the platform's font metrics —
    on Windows even this short sample name came back as '· synth_ca….00–5.00 s' and turned a
    real assertion into a font measurement. The tooltip is the full string by contract, and
    test_window_fits_screen.py::test_the_noise_reference_readout_is_elided_not_unbounded is
    what guards the rendering side of it."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    pv = PreviewScreen(AppState(s))
    assert "synth_case_A.csv" in pv.noise_ref_readout.toolTip()
    assert pv.noise_ref_readout.text(), "the read-out must still show something"

    n = pv.state.settings.processing.emg.noise
    n.use_expiration, n.reference_intervals = True, []
    pv._refresh_noise_readout()
    assert "every expiration" in pv.noise_ref_readout.toolTip()

    n.reference_file = None
    pv._refresh_noise_readout()
    assert "rest reference: not set" in pv.noise_ref_readout.toolTip().lower()
    pv.shutdown()


def test_the_two_auto_checkboxes_are_named_distinctly(qapp, tmp_path):
    """Ticket D21 (UI-overhaul): the EMG – noise reduction sub-tab and the EMG – ECG
    reduction sub-tab each have their own 'Auto' checkbox, doing two unrelated things
    (auto-picking the noise-suppression strength for the current file, versus
    auto-detecting the ECG-removal parameters for the whole batch from a reference
    recording). They used to share the bare label 'Auto', with the noise one relying on a
    separate, easy-to-miss caption ('Noise') to say what it was Auto of. Neither label nor
    tooltip may collide now."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), remove_ecg=True, noise=True, data_out=_OUT)
    pv = PreviewScreen(AppState(s))
    assert pv.noise_auto.text() != pv.ecg_auto_batch.text()
    assert pv.noise_auto.text() != "Auto"
    assert pv.ecg_auto_batch.text() != "Auto"
    assert pv.noise_auto.toolTip() != pv.ecg_auto_batch.toolTip()
    assert "suppression" in pv.noise_auto.toolTip().lower()
    assert "batch" in pv.ecg_auto_batch.toolTip().lower()
    pv.shutdown()


def test_the_picker_writes_both_halves_together(qapp, tmp_path):
    """They are alternatives in the core (`use_expiration or not reference_intervals`), so
    leaving stale intervals behind would make the stored choice ambiguous on re-open."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")

    pv._apply_noise_expiration()
    n = s.processing.emg.noise
    assert n.use_expiration is True and n.reference_intervals == []

    pv._apply_noise_reference(2.0, 4.0)
    assert n.use_expiration is False and n.reference_intervals == [[2.0, 4.0]]
    win.close()


def test_setting_the_noise_reference_stamps_the_current_input_folder(qapp, tmp_path):
    """Ticket B06: NoiseSettings.reference_folder lets the carried-over banner tell a
    reference chosen against one recordings folder apart from one chosen against another
    that happens to share the reference filename."""
    from respmech.ui.main_window import MainWindow
    from _helpers import INPUT
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")

    pv._apply_noise_expiration()
    assert s.processing.emg.noise.reference_folder == INPUT

    s.processing.emg.noise.reference_folder = "/a/different/folder"   # simulate a stale tag
    pv._apply_noise_reference(2.0, 4.0)
    assert s.processing.emg.noise.reference_folder == INPUT           # restamped on re-apply
    win.close()


def test_the_noise_enable_checkbox_is_reachable_to_turn_noise_on(qapp, tmp_path):
    """Regression (change 5): the enable checkbox was placed inside the noise_opts chip,
    which _update_actions disables while noise is off — and a Qt child of a disabled parent
    cannot be re-enabled, so the control to turn noise ON greyed itself out exactly when it
    was needed, with no UI path to enable noise at all. It lives on the strip now.

    The suite missed this because setChecked() fires even on a disabled widget; this asserts
    isEnabled(), which is what a real click depends on."""
    pv = _preview(qapp, tmp_path, ecg=True)
    pv.state.settings.processing.emg.noise.enabled = False
    pv._update_actions()
    assert pv.noise_enabled.isEnabled(), "cannot turn noise on: the checkbox is disabled while off"
    # and it is not a one-way trap: turning it on then off leaves it reachable
    pv.noise_enabled.setChecked(True); pv._update_actions()
    pv.noise_enabled.setChecked(False); pv._update_actions()
    assert pv.noise_enabled.isEnabled(), "unticking disabled its own control (one-way trap)"
    pv.shutdown()


def test_the_noise_enable_checkbox_still_waits_for_ecg_removal(qapp, tmp_path):
    from respmech.ui.screens.preview_screen import NEEDS_ECG_HINT
    pv = _preview(qapp, tmp_path, ecg=False)
    assert not pv.noise_enabled.isEnabled()
    assert NEEDS_ECG_HINT in pv.noise_enabled.toolTip()
    pv.shutdown()


# -- D07 (UI-overhaul): the Detail plot's noise-reference band tells the truth ------
# Regression: _ensure_noise_region seeded the band ONCE, at construction, from whatever
# settings existed then (0.0-1.0 for a fresh AppState) and never moved it again except
# through an interactive "Set noise profile" approval. Opening or loading a saved
# analysis — the app's own declared workflow — left the band pointing at the first
# second of the recording regardless of where the real reference was.

def test_construction_does_not_show_another_files_reference(qapp, tmp_path):
    """Regression found in review of this ticket: PreviewScreen.__init__ used to call the
    OLD unconditional _ensure_noise_region() right after _load_noise_params(), which
    created/showed the band from reference_intervals[0] regardless of which file
    refresh_files() (later in the same constructor) was about to auto-select — a fresh
    construction could paint file B's span for one frame while file A was quietly
    adopted as the shown file, before the async detail render ever corrected it."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    s.processing.emg.noise.reference_file = "synth_case_B.csv"   # NOT the auto-selected file
    pv = PreviewScreen(AppState(s))
    assert pv.file_rail.current_filename() == "synth_case_A.csv", "test assumes A is auto-selected"
    assert pv._noise_region is not None, "a malformed/mismatched reference must not abort construction"
    assert pv._noise_region.isVisible() is False, "showed file B's reference while A was on screen"
    pv.shutdown()


def test_the_band_matches_the_saved_reference_on_its_own_file(qapp, tmp_path):
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)   # ref: synth_case_A.csv, [1.0, 5.0]
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._load_noise_params()
    assert pv._noise_region is not None
    assert pv._noise_region.isVisible() is True
    lo, hi = pv._noise_region.getRegion()
    assert (lo, hi) == pytest.approx((1.0, 5.0))
    assert pv._noise_label is not None and pv._noise_label.isVisible() is True
    pv.shutdown()


def test_opening_a_saved_analysis_refreshes_the_band_not_only_at_construction(qapp, tmp_path):
    """The exact bug: a fresh AppState seeds the band at 0.0-1.0 in the constructor: a
    later sync_from_settings — how a saved analysis actually arrives — must move it."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)   # noise=False here -> no reference yet
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    n = pv.state.settings.processing.emg.noise
    n.enabled = True
    n.reference_file = "synth_case_A.csv"
    n.reference_intervals = [[2.5, 6.5]]
    n.use_expiration = False
    pv.sync_from_settings()                           # the real path a file-open takes
    assert pv._noise_region.isVisible() is True
    assert pv._noise_region.getRegion() == pytest.approx((2.5, 6.5))
    pv.shutdown()


def test_the_band_hides_for_every_expiration(qapp, tmp_path):
    """No single span exists to point at, so showing the old one would misrepresent it."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is True         # sanity: shown before the switch
    n = pv.state.settings.processing.emg.noise
    n.use_expiration, n.reference_intervals = True, []
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is False
    assert pv._noise_label.isVisible() is False
    pv.shutdown()


def test_the_band_hides_when_the_shown_file_is_not_the_reference_file(qapp, tmp_path):
    """The test's one shared reference was built from a DIFFERENT recording — this file
    has no interval of its own to shade, so showing synth_case_A.csv's span while
    synth_case_B.csv is on screen would claim a reference this file does not have."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)   # ref: synth_case_A.csv
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is True
    pv.file_rail.select_filename("synth_case_B.csv")
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is False
    assert pv._noise_label.isVisible() is False
    # ...and switching back shows it again — this is about the CURRENT file, not a
    # one-way latch
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is True
    pv.shutdown()


def test_accepting_the_picker_on_a_different_file_moves_and_shows_the_band(qapp, tmp_path):
    """_apply_noise_reference's own docstring says it 'reflects the choice on the detail
    plot' — pin that down: after accepting on a NEW file, the band shows THAT file's span,
    not the old file's, and the old file no longer shows one at all."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)   # ref: synth_case_A.csv, [1.0, 5.0]
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_B.csv")
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is False        # B has no reference of its own yet
    pv._apply_noise_reference(3.0, 7.0)
    assert pv._noise_region.isVisible() is True
    assert pv._noise_region.getRegion() == pytest.approx((3.0, 7.0))
    n = pv.state.settings.processing.emg.noise
    assert n.reference_file == "synth_case_B.csv"
    pv.file_rail.select_filename("synth_case_A.csv")    # the file the OLD reference lived on
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is False, "still showing the reference it just lost"
    pv.shutdown()


def test_clearing_panels_reattaches_the_label_not_only_the_region(qapp, tmp_path):
    """Regression (secondary finding in review of this ticket): _clear_all_panels and
    _clear_file_panels called the OLD _ensure_noise_region() right after emg_plots.clear()
    wiped both the region AND the label — re-attaching only the region left a shaded band
    with no 'noise reference' caption until the next render happened to run."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), noise=True, data_out=_OUT)   # ref: synth_case_A.csv, [1.0, 5.0]
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._load_noise_params()
    assert pv._noise_region.isVisible() is True
    pv._clear_all_panels()
    assert pv._noise_region.isVisible() is True, "clearing lost the band for the still-current file"
    assert pv._noise_label is not None and pv._noise_label.isVisible() is True, \
        "region re-attached but its caption was not"
    pv.shutdown()

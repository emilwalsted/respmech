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

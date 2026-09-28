"""The UI half of the automatic EMG-only segmentations (``fixed_windows``/``emg_burst``):
the signal-set funnel that reconciles the noise reference with the chosen method, the
noise picker's third whole mode, the segments tab's 'Advanced…' slot, the preview
cache keys and the friendly settings messages. The core behaviour they drive is covered
in ``test_segments_emg.py`` / ``test_noise_reference_mode.py``.
"""
import pytest
from PySide6.QtWidgets import QDialog

from _helpers import requires_synth, synth_settings

from respmech.ui.state import AppState


def _emg_burst_settings(tmp_path, method="emg_burst"):
    s = synth_settings(tmp_path, channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})
    s.input.channels.entropy = []
    s.input.files = "synth_emgburst_A.csv"
    s.processing.segmentation.method = method
    return s


# -- the signal-set funnel ----------------------------------------------------------------

def test_choosing_emg_burst_selects_the_interburst_reference_and_keeps_auto_prop(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    noise = sc.state.settings.processing.emg.noise
    noise.auto_prop = True
    sc.apply_signal_set(["emg"], segmentation_method="emg_burst")
    assert sc.state.settings.processing.segmentation.method == "emg_burst"
    assert noise.reference_mode == "interburst"
    assert noise.auto_prop is True              # bursts against the periods between them exists
    assert noise.use_expiration is False
    win.close()


def test_leaving_emg_burst_retires_interburst_and_switches_auto_prop_off(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.apply_signal_set(["emg"], segmentation_method="emg_burst")
    sc.apply_signal_set(["emg"], segmentation_method="whole_file")
    noise = sc.state.settings.processing.emg.noise
    assert noise.reference_mode == "auto"
    assert noise.auto_prop is False
    win.close()


def test_gaining_flow_resets_an_emg_only_reference_mode(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.apply_signal_set(["emg"], segmentation_method="emg_burst")
    sc.apply_signal_set(["flow", "emg"])
    s = sc.state.settings
    assert s.processing.segmentation.method == "flow"
    assert s.processing.emg.noise.reference_mode == "auto"
    win.close()


# -- the noise picker's third whole mode --------------------------------------------------

def _dialog(modes):
    import numpy as np
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    fs = 1000
    t = np.arange(2000, dtype=float) / fs
    raw = [np.sin(2 * np.pi * 80 * t) * (i + 1) for i in range(3)]
    return NoiseProfileDialog(raw, t, fs, [2, 3, 4], modes_available=frozenset(modes))


def test_the_interburst_option_is_absent_unless_offered_and_returns_its_sentinel(qapp):
    from respmech.ui.noise_profile_dialog import INTERBURST
    absent = _dialog({"intervals"})
    assert absent.use_interburst.isHidden() is True
    absent.deleteLater()
    dlg = _dialog({"intervals", "interburst"})
    assert dlg.use_interburst.isHidden() is False
    assert dlg.selected_region() is None
    dlg.use_interburst.setChecked(True)
    assert dlg.selected_region() is INTERBURST
    assert dlg.btn_ok.isEnabled() is True
    dlg.deleteLater()


def test_interburst_and_the_other_whole_modes_retire_each_other(qapp):
    from respmech.ui.noise_profile_dialog import EXPIRATION, INTERBURST, REST_SEGMENTS
    dlg = _dialog({"intervals", "expiration", "rest_segments", "interburst"})
    dlg.use_interburst.setChecked(True)
    dlg.use_rest_segments.setChecked(True)
    assert dlg.selected_region() is REST_SEGMENTS and not dlg.use_interburst.isChecked()
    dlg.use_expiration.setChecked(True)
    assert dlg.selected_region() is EXPIRATION and not dlg.use_rest_segments.isChecked()
    dlg.use_interburst.setChecked(True)
    assert dlg.selected_region() is INTERBURST and not dlg.use_expiration.isChecked()
    dlg.use_interburst.setChecked(False)
    assert dlg.selected_region() is None and dlg.btn_ok.isEnabled() is False
    dlg.deleteLater()


@requires_synth()
def test_the_interburst_choice_is_written_explicitly_and_undone_by_another_choice(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = _emg_burst_settings(tmp_path)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_emgburst_A.csv")
    assert pv._apply_noise_interburst() is True
    n = s.processing.emg.noise
    assert (n.reference_file, n.reference_mode, n.reference_intervals) == (
        "synth_emgburst_A.csv", "interburst", [])
    assert "between bursts" in pv.noise_ref_readout.toolTip() or \
        "between bursts" in pv.noise_ref_readout.fullText()
    pv._apply_noise_reference(1.0, 2.0)                    # an explicit span replaces it
    assert n.reference_mode == "auto" and n.reference_intervals == [[1.0, 2.0]]
    pv._apply_noise_interburst()
    pv._apply_noise_rest_segments()
    assert n.reference_mode == "auto"
    win.close()


# -- the segments tab's Advanced… slot ----------------------------------------------------

@requires_synth()
@pytest.mark.parametrize("method,advanced,separators", [
    ("emg_burst", True, False), ("fixed_windows", True, False),
    ("separators", False, True), ("whole_file", False, True)])
def test_advanced_takes_the_place_of_place_separators_for_the_automatic_methods(
        qapp, tmp_path, method, advanced, separators):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(_emg_burst_settings(tmp_path, method)))
    win.show()
    pv = win.preview_screen
    pv._update_separators_button()
    assert pv.btn_segmentation_advanced.isVisibleTo(pv) is advanced
    assert pv.btn_place_separators.isVisibleTo(pv) is separators
    assert pv.btn_segmentation_advanced.isEnabled() is advanced
    win.close()


@requires_synth()
@pytest.mark.parametrize("method,edit,attr", [
    ("emg_burst", {"burst_threshold_frac": 0.4}, "burst_threshold_frac"),
    ("fixed_windows", {"window_s": 2.5}, "window_s")])
def test_advanced_writes_the_edited_parameter_and_marks_the_analysis_edited(
        qapp, tmp_path, monkeypatch, method, edit, attr):
    from respmech.ui.advanced_dialog import AdvancedDialog
    from respmech.ui.main_window import MainWindow
    s = _emg_burst_settings(tmp_path, method)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    seen = []
    pv.settings_edited.connect(lambda: seen.append(1))
    monkeypatch.setattr(AdvancedDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AdvancedDialog, "edited_values", lambda self: dict(edit))
    pv._open_segmentation_advanced()
    assert getattr(s.processing.segmentation.emg, attr) == edit[attr]
    assert seen
    monkeypatch.setattr(AdvancedDialog, "edited_values", lambda self: {})
    seen.clear()
    pv._open_segmentation_advanced()                       # OK without an edit: nothing happens
    assert not seen
    win.close()


@requires_synth()
def test_advanced_offers_only_the_current_methods_own_fields(qapp, tmp_path, monkeypatch):
    from respmech.ui.advanced_dialog import AdvancedDialog
    from respmech.ui.main_window import MainWindow
    keys = {}
    for method in ("emg_burst", "fixed_windows"):
        win = MainWindow(AppState(_emg_burst_settings(tmp_path, method)))
        monkeypatch.setattr(AdvancedDialog, "exec",
                            lambda self, m=method: keys.setdefault(m, set(self._widgets)) and
                            QDialog.Rejected)
        win.preview_screen._open_segmentation_advanced()
        win.close()
    assert keys["emg_burst"] == {"burst_threshold_frac", "burst_min_s", "burst_smooth_s",
                                 "burst_min_contrast"}
    assert keys["fixed_windows"] == {"window_s", "hop_s"}


# -- cache keys and messages --------------------------------------------------------------

def test_reference_clip_and_noise_report_keys_follow_the_burst_parameters(tmp_path):
    from respmech.ui.screens import _preview_cache as pc
    s = _emg_burst_settings(tmp_path)
    s.processing.emg.noise.reference_mode = "interburst"
    s.processing.emg.noise.reference_file = "synth_emgburst_A.csv"
    ref = f"{s.input.folder}/synth_emgburst_A.csv"
    before = pc.ref_clip_key(s, ref)
    assert before is not None
    s.processing.segmentation.emg.burst_min_contrast = 3.0
    assert pc.ref_clip_key(s, ref) != before
    s.processing.emg.noise.auto_prop = True
    files = [ref]
    k1 = pc.noise_report_key(s, ref, files)
    s.processing.segmentation.emg.burst_threshold_frac = 0.5
    assert pc.noise_report_key(s, ref, files) != k1


def test_the_new_settings_messages_are_translated_to_ui_controls():
    from respmech.core.settings import Settings, SettingsError
    from respmech.ui.validation import friendly_settings_error

    s = Settings()
    s.input.format.sampling_frequency = 1000
    s.analysis.signals = ["emg"]
    s.input.channels.emg = [2]
    s.processing.segmentation.method = "emg_burst"
    s.processing.segmentation.emg.burst_min_contrast = 1.0
    with pytest.raises(SettingsError) as ei:
        s.validate()
    assert "Advanced…" in friendly_settings_error(ei.value)
    s = Settings()
    s.input.format.sampling_frequency = 1000
    s.analysis.signals = ["emg"]
    s.input.channels.emg = [2]
    s.processing.segmentation.method = "whole_file"
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "r.csv"
    s.processing.emg.noise.reference_mode = "interburst"
    with pytest.raises(SettingsError) as ei:
        s.validate()
    friendly = friendly_settings_error(ei.value)
    assert friendly != str(ei.value) and "inter-burst" in friendly


def test_re_choosing_emg_burst_keeps_a_reference_the_user_picked(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.apply_signal_set(["emg"], segmentation_method="emg_burst")
    noise = sc.state.settings.processing.emg.noise
    noise.reference_mode = "auto"
    noise.reference_intervals = [[1.0, 2.0]]           # a hand-marked span
    sc.apply_signal_set(["emg"], segmentation_method="emg_burst")
    assert noise.reference_mode == "auto" and noise.reference_intervals == [[1.0, 2.0]]
    win.close()


def test_interburst_or_rest_segments_without_a_reference_file_is_blocked_up_front():
    """Otherwise the run starts and the whole batch aborts on the missing file."""
    from respmech.core.settings import Settings, SettingsError
    from respmech.ui.validation import friendly_settings_error
    for mode, method in (("interburst", "emg_burst"), ("rest_segments", "whole_file")):
        s = Settings()
        s.input.format.sampling_frequency = 1000
        s.analysis.signals = ["emg"]
        s.input.channels.emg = [2]
        s.processing.segmentation.method = method
        s.processing.emg.remove_ecg = True
        s.processing.emg.noise.enabled = True
        s.processing.emg.noise.auto_prop = False
        s.processing.emg.noise.reference_mode = mode
        with pytest.raises(SettingsError, match="no usable rest reference") as ei:
            s.validate()
        assert "rest reference" in friendly_settings_error(ei.value)


@requires_synth()
def test_advanced_asks_before_clearing_numbered_exclusions(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from respmech.core.settings import ExcludeEntry
    from respmech.ui.advanced_dialog import AdvancedDialog
    from respmech.ui.main_window import MainWindow
    s = _emg_burst_settings(tmp_path)
    s.processing.exclude_breaths = [ExcludeEntry(file="synth_emgburst_A.csv", breaths=[3])]
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    monkeypatch.setattr(AdvancedDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(AdvancedDialog, "edited_values",
                        lambda self: {"burst_threshold_frac": 0.4})
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    pv._open_segmentation_advanced()
    assert s.processing.segmentation.emg.burst_threshold_frac == 0.4
    assert s.processing.exclude_breaths                       # kept on No
    monkeypatch.setattr(AdvancedDialog, "edited_values",
                        lambda self: {"burst_threshold_frac": 0.5})
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    pv._open_segmentation_advanced()
    assert s.processing.exclude_breaths == []                 # cleared on Yes
    win.close()


@requires_synth()
def test_a_reference_without_bursts_is_a_clean_message_in_the_noise_panel(qapp, tmp_path):
    import numpy as np
    from respmech.ui.workers import stage_noise_fidelity
    rng = np.random.default_rng(0)
    n = 6000
    data = np.column_stack([np.arange(n) / 1000, rng.normal(0, 1, (n, 3))])
    np.savetxt(tmp_path / "noise.csv", data, delimiter=",", header="time,E1,E2,E3", comments="")
    s = _emg_burst_settings(tmp_path)
    s.input.folder = str(tmp_path)
    s.input.files = "noise.csv"
    s.input.channels.emg = [2, 3, 4]
    s.input.channels.entropy = []
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "noise.csv"
    s.processing.emg.noise.reference_mode = "interburst"
    out = stage_noise_fidelity(s)
    assert "error" in out and "No EMG bursts found" in out["error"]
    assert "Advanced" in out["error"]

"""The Setup-screen controls for the opt-in cardiac-gated peak EMG.

The core feature is covered by test_robust_peak.py; this is about the UI contract:
  1. the widgets round-trip to and from the Analysis file, including partial/older files;
  2. editing them marks the analysis modified (they must be wired into _wire_reactivity —
     the _row/_check_row helpers wire nothing themselves, which is easy to forget);
  3. the guards grey out when the feature is off, so they cannot be tuned into a vacuum;
  4. turning it on without ECG removal warns, because the gated columns would be blank;
  5. it changes NO preview panel — the classification must not fall through to the EMG
     conditioning bucket, which would re-run four panels on every tick of a checkbox.
"""
import numpy as np

from _helpers import requires_synth, synth_settings  # noqa: F401

from respmech.ui.state import AppState

pytestmark = requires_synth()


def _screen(tmp_path, **kw):
    from respmech.ui.screens.settings_screen import SettingsScreen
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True}, **kw)
    return SettingsScreen(AppState(s))


# -- 1. round-trip ------------------------------------------------------------

def test_widgets_load_the_defaults_and_write_back(qapp, tmp_path):
    sc = _screen(tmp_path)
    rp = sc.state.settings.processing.emg.robust_peak
    assert sc.emg_gated.isChecked() is False           # opt-in
    assert sc.emg_gate_width.value() == rp.gate_half_width_s

    sc.emg_gated.setChecked(True)
    sc.emg_gate_width.setValue(0.2)
    sc.rp_min_survival.setValue(0.55)
    sc.rp_min_island.setValue(0.3)
    sc.rp_long_rr.setValue(1.8)
    sc.rp_max_long_rr.setValue(0.05)
    sc.rp_hr_margin.setValue(0.2)
    out = sc.to_state().processing.emg.robust_peak
    assert out.enabled is True
    assert (out.gate_half_width_s, out.min_survival, out.min_island_s) == (0.2, 0.55, 0.3)
    assert (out.long_rr_factor, out.max_long_rr_frac, out.hr_ceiling_margin) == (1.8, 0.05, 0.2)


def test_from_state_survives_an_analysis_written_before_the_feature(qapp, tmp_path):
    """An older Analysis file simply has no robust_peak table; from_dict then supplies the
    dataclass defaults. The screen must load it without complaint."""
    from respmech.core.settings import Settings
    sc = _screen(tmp_path)
    d = sc.state.settings.to_dict()
    d["processing"]["emg"].pop("robust_peak", None)     # as an older file would be
    sc.state.settings = Settings.from_dict(d)
    sc.from_state()
    assert sc.emg_gated.isChecked() is False
    assert sc.emg_gate_width.value() == 0.120


# -- 2. editing marks the analysis modified -----------------------------------

def test_toggling_marks_dirty(qapp, tmp_path):
    """_row/_check_row attach tooltips only — every widget must be added by hand to
    _wire_reactivity, so this is the test that catches forgetting one."""
    for widget, new in (("emg_gated", True), ("emg_gate_width", 0.2),
                        ("rp_min_survival", 0.6), ("rp_min_island", 0.35),
                        ("rp_long_rr", 2.0), ("rp_max_long_rr", 0.04),
                        ("rp_hr_margin", 0.25)):
        sc = _screen(tmp_path)
        sc._mark_clean()
        assert not sc.is_dirty()
        w = getattr(sc, widget)
        w.setChecked(new) if isinstance(new, bool) else w.setValue(new)
        assert sc.is_dirty(), f"{widget} does not mark the analysis modified"


# -- 3. the guards follow the master switch -----------------------------------

def test_guards_are_disabled_until_the_feature_is_on(qapp, tmp_path):
    sc = _screen(tmp_path)
    guards = (sc.emg_gate_width, sc.rp_min_survival, sc.rp_min_island,
              sc.rp_long_rr, sc.rp_max_long_rr, sc.rp_hr_margin)
    assert not any(w.isEnabled() for w in guards)
    sc.emg_gated.setChecked(True)
    assert all(w.isEnabled() for w in guards)
    sc.emg_gated.setChecked(False)
    assert not any(w.isEnabled() for w in guards)


# -- 4. the ECG-removal dependency is surfaced --------------------------------

def test_enabling_without_ecg_removal_cautions(qapp, tmp_path):
    sc = _screen(tmp_path, remove_ecg=False)
    assert sc._science_note() == ""                      # nothing to say while it is off
    sc.emg_gated.setChecked(True)
    note = sc._science_note()
    assert "ECG removal" in note
    assert note in sc._all_cautions()                    # and it reaches the QC strip
    sc.state.settings.processing.emg.remove_ecg = True   # the Preview screen's own control
    assert sc._science_note() == ""


def test_the_prerequisite_is_not_hidden_behind_a_softer_caution(qapp, tmp_path):
    """Regression: the cautions were a first-match-wins chain with this one appended last, so
    any sub-1000 Hz EMG recording — a legitimate, merely-advisory condition — silently hid the
    one caution the user cannot diagnose from this screen."""
    sc = _screen(tmp_path, remove_ecg=False)
    sc.samp_freq.setValue(500)                           # earns the soft sampling-rate note
    sc.emg_gated.setChecked(True)
    cautions = sc._all_cautions()
    assert any("ECG removal" in c for c in cautions), cautions
    assert any("sampling frequency" in c for c in cautions), cautions
    assert "ECG removal" in sc._science_note()           # and it outranks the advisory one


def test_the_caution_clears_when_preview_turns_ecg_removal_on(qapp, tmp_path):
    """Regression: the warning names a control on ANOTHER screen, and nothing recomputed the
    strip when that screen wrote the field — so it stayed pinned after the user had followed
    its own instruction."""
    from respmech.ui.main_window import MainWindow
    from _helpers import synth_settings
    s = synth_settings(str(tmp_path), remove_ecg=False,
                       data_out={"saveaveragedata": True, "savebreathbybreathdata": True})
    win = MainWindow(AppState(s))
    sc = win.settings_screen
    sc.emg_gated.setChecked(True)
    assert "ECG removal" in sc.qc.text()

    # the Preview ECG tab owns remove_ecg; emulate its edit + the signal it emits
    sc.state.settings.processing.emg.remove_ecg = True
    win.preview_screen.settings_edited.emit()
    assert "ECG removal" not in sc.qc.text(), "stale after Preview fixed it"

    # and the reverse: switching it back off must bring the caution back on tab entry.
    # Leave Setup first — setCurrentIndex to the tab already shown emits nothing.
    sc.state.settings.processing.emg.remove_ecg = False
    win.tabs.setCurrentIndex(win._i_preview)
    win.tabs.setCurrentIndex(win._i_settings)
    assert "ECG removal" in sc.qc.text(), "stale after Preview broke it"
    win.close()


# -- 5. it must not invalidate any preview panel ------------------------------

def test_no_preview_panel_recomputes_for_a_gated_setting():
    """Falling through to the generic processing.emg rule would re-run ECG + both EMG panels
    + the noise frontier for a setting that changes nothing any of them draws."""
    from respmech.ui.screens.preview_screen import _kinds_for_settings_path
    for path in ("processing.emg.robust_peak.enabled",
                 "processing.emg.robust_peak.gate_half_width_s",
                 "processing.emg.robust_peak.min_survival",
                 "processing.emg.robust_peak.hr_ceiling_margin"):
        assert _kinds_for_settings_path(path) == frozenset(), path
    # a neighbouring EMG path must still recompute, i.e. the new rule is narrow
    assert _kinds_for_settings_path("processing.emg.rms_window_s")


def test_a_gated_edit_schedules_nothing_in_the_preview(qapp, tmp_path):
    """End to end through the real diffing path: change only robust_peak and no job runs."""
    from respmech.ui.screens.preview_screen import _changed_settings_paths
    sc = _screen(tmp_path)
    before = sc.to_state()
    import copy
    snapshot = copy.deepcopy(before)
    sc.emg_gated.setChecked(True)
    after = sc.to_state()
    paths = _changed_settings_paths(snapshot, after)
    assert paths == {"processing.emg.robust_peak.enabled"}, paths
    kinds = set()
    from respmech.ui.screens.preview_screen import _kinds_for_settings_path
    for p in paths:
        kinds |= _kinds_for_settings_path(p)
    assert kinds == set(), f"a gated edit would recompute {kinds}"

"""The cardiac-gated peak, now owned by the Preview EMG tab.

It moved off Setup because it is an EMG statistic and its prerequisite is ECG removal, which
lives one tab away rather than one screen away. That relocation is the point of these tests:
what used to be prose in three places plus a QC caution that went stale is now a plain
setEnabled with a tooltip, next to the EMG it changes.

What still matters, unchanged from when it lived on Setup:
  1. the values round-trip, including from an Analysis written before the feature existed;
  2. editing marks the analysis modified — nothing wires that automatically;
  3. the guards are reachable but out of the way, in the Advanced modal;
  4. it changes NO preview panel, so it must not schedule a recompute — the classification
     must not fall through to the EMG conditioning bucket and re-run four panels per click.
"""
import pytest
from PySide6.QtWidgets import QDialog

from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _preview(qapp, tmp_path, *, remove_ecg=True, **kw):
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), data_out=_OUT, **kw)
    s.processing.emg.remove_ecg = remove_ecg
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    pv._update_actions()
    qapp.processEvents()
    return pv


# -- 1. round-trip -------------------------------------------------------------
def test_the_strip_shows_the_stored_values(qapp, tmp_path):
    pv = _preview(qapp, tmp_path)
    rp = pv.state.settings.processing.emg.robust_peak
    assert pv.emg_gated.isChecked() is False            # opt-in
    assert pv.emg_gate_width.value() == rp.gate_half_width_s
    pv.shutdown()


def test_editing_the_strip_writes_the_model(qapp, tmp_path):
    pv = _preview(qapp, tmp_path)
    pv.emg_gated.setChecked(True)
    pv.emg_gate_width.setValue(0.2)
    rp = pv.state.settings.processing.emg.robust_peak
    assert rp.enabled is True and rp.gate_half_width_s == 0.2
    pv.shutdown()


def test_an_analysis_written_before_the_feature_still_loads(qapp, tmp_path):
    """An older Analysis file simply has no robust_peak table; from_dict then supplies the
    dataclass defaults."""
    from respmech.core.settings import Settings
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    d = s.to_dict()
    d["processing"]["emg"].pop("robust_peak", None)
    pv = PreviewScreen(AppState(Settings.from_dict(d)))
    pv._load_gated_params()
    assert pv.emg_gated.isChecked() is False
    assert pv.emg_gate_width.value() == 0.120
    pv.shutdown()


def test_setup_can_no_longer_clobber_it(qapp, tmp_path):
    """to_state runs on every tab change and rewrites everything Setup owns. If it still
    owned these, a gated-peak edit would be erased on the first switch to Setup."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), data_out=_OUT)
    win = MainWindow(AppState(s))
    win.preview_screen.emg_gated.setChecked(True)
    win.preview_screen.emg_gate_width.setValue(0.2)
    for _ in range(3):
        win.settings_screen.to_state()
    rp = s.processing.emg.robust_peak
    assert rp.enabled is True and rp.gate_half_width_s == 0.2
    win.close()


# -- 2. editing marks the analysis modified ------------------------------------
@pytest.mark.parametrize("widget, value", [("emg_gated", True), ("emg_gate_width", 0.2)])
def test_editing_marks_the_analysis_modified(qapp, tmp_path, widget, value):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    win.settings_screen._mark_clean()
    w = getattr(win.preview_screen, widget)
    w.setChecked(value) if isinstance(value, bool) else w.setValue(value)
    assert win.settings_screen.is_dirty(), f"{widget} does not mark the analysis modified"
    win.close()


# -- 3. the prerequisite is a gate now, not a caution --------------------------
def test_it_is_inert_until_ecg_removal_is_on(qapp, tmp_path):
    from respmech.ui.screens.preview_screen import NEEDS_ECG_GATE_HINT
    pv = _preview(qapp, tmp_path, remove_ecg=False)
    assert not pv.gate_opts.isEnabled()
    assert NEEDS_ECG_GATE_HINT in pv.gate_opts.toolTip(), "disabled with no explanation"
    pv.shutdown()


def test_it_comes_alive_the_moment_ecg_removal_is_switched_on(qapp, tmp_path):
    """The whole reason it moved: on Setup this could only ever be a caution, because the
    control it named lived on another screen — and that caution went stale."""
    pv = _preview(qapp, tmp_path, remove_ecg=False)
    assert not pv.gate_opts.isEnabled()
    pv.remove_ecg.setChecked(True)
    qapp.processEvents()
    assert pv.gate_opts.isEnabled()
    assert pv.gate_opts.toolTip() == ""
    pv.shutdown()


def test_setup_still_warns_that_the_columns_would_be_blank(qapp, tmp_path):
    """The QC strip reads the model, so it survives the move — and it is still worth having:
    it is the one place that summarises "this requested output will come back empty"."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_OUT)
    win = MainWindow(AppState(s))
    s.processing.emg.robust_peak.enabled = True
    win.settings_screen.refresh_qc()
    assert "ECG removal" in win.settings_screen.qc.text()
    win.close()


# -- 4. the guards are in the Advanced modal -----------------------------------
def test_the_guards_moved_off_the_strip(qapp, tmp_path):
    pv = _preview(qapp, tmp_path)
    for name in ("rp_min_survival", "rp_min_island", "rp_long_rr", "rp_max_long_rr",
                 "rp_hr_margin"):
        assert not hasattr(pv, name), f"{name} is back on the strip"
    pv.shutdown()


def test_the_guards_are_editable_in_the_modal(qapp, tmp_path, monkeypatch):
    import respmech.ui.advanced_dialog as ad
    pv = _preview(qapp, tmp_path)
    real = ad.AdvancedDialog

    class _Stub(real):
        def exec(self):
            self.widget("min_survival").setValue(0.55)
            self.widget("hr_ceiling_margin").setValue(0.2)
            return QDialog.Accepted

    monkeypatch.setattr(ad, "AdvancedDialog", _Stub)
    pv._open_emg_advanced()
    rp = pv.state.settings.processing.emg.robust_peak
    assert rp.min_survival == 0.55 and rp.hr_ceiling_margin == 0.2
    pv.shutdown()


# -- 5. it must not invalidate any preview panel -------------------------------
def test_no_preview_panel_recomputes_for_a_gated_setting():
    """Falling through to the generic processing.emg rule would re-run ECG + both EMG panels
    + the noise frontier for a setting that changes nothing any of them draws."""
    from respmech.ui.screens.preview_screen import _kinds_for_settings_path
    for path in ("processing.emg.robust_peak.enabled",
                 "processing.emg.robust_peak.gate_half_width_s",
                 "processing.emg.robust_peak.min_survival",
                 "processing.emg.robust_peak.hr_ceiling_margin"):
        assert _kinds_for_settings_path(path) == frozenset(), path
    assert _kinds_for_settings_path("processing.emg.rms_window_s")


def test_toggling_it_schedules_nothing(qapp, tmp_path):
    """End to end through the real handler: the gated columns are written by the batch and
    drawn by no panel, so a click must not queue any recompute."""
    pv = _preview(qapp, tmp_path)
    pv._pending_kinds = set()
    pv.emg_gated.setChecked(True)
    assert pv._pending_kinds == set(), f"a gated edit queued {pv._pending_kinds}"
    pv.shutdown()

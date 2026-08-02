"""The cardiac-gated peak, now owned entirely by the EMG Advanced modal.

It moved off Setup to the Preview strip because its prerequisite (ECG removal) lives one tab
away, and then off the strip into the Advanced modal (Emil, 02-08-2026): it is an output-only
add-on — extra 'gated' columns in the saved data, drawn by no preview panel — so it did not
earn permanent strip space that made the strip wrap on a laptop.

What still matters, unchanged through both moves:
  1. the values round-trip, including from an Analysis written before the feature existed;
  2. committing an edit marks the analysis modified — nothing wires that automatically;
  3. the ECG prerequisite is enforced where the control IS, with the same words;
  4. the quality guards live beside it in the modal;
  5. it changes NO preview panel, so committing it must not schedule a recompute — the
     classification must not fall through to the EMG conditioning bucket and re-run four
     panels for columns none of them show.
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


def _drive_modal(pv, monkeypatch, act):
    """Open the EMG Advanced modal with ``act(dlg)`` run in place of exec()."""
    import respmech.ui.advanced_dialog as ad
    real = ad.AdvancedDialog

    class _Stub(real):
        def exec(self):
            return act(self)

    monkeypatch.setattr(ad, "AdvancedDialog", _Stub)
    pv._open_emg_advanced()


# -- 1. round-trip -------------------------------------------------------------
def test_the_modal_shows_the_stored_values(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path)
    rp = pv.state.settings.processing.emg.robust_peak
    seen = {}

    def act(dlg):
        seen["enabled"] = dlg.widget("enabled").isChecked()
        seen["width"] = dlg.widget("gate_half_width_s").value()
        return QDialog.Rejected

    _drive_modal(pv, monkeypatch, act)
    assert seen["enabled"] is False                      # opt-in
    assert seen["width"] == rp.gate_half_width_s
    pv.shutdown()


def test_ok_writes_the_model_and_cancel_does_not(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path)
    rp = pv.state.settings.processing.emg.robust_peak

    def edit(dlg):
        dlg.widget("enabled").setChecked(True)
        dlg.widget("gate_half_width_s").setValue(0.2)
        return QDialog.Rejected

    _drive_modal(pv, monkeypatch, edit)
    assert rp.enabled is False and rp.gate_half_width_s == 0.120, "Cancel leaked an edit"

    def commit(dlg):
        dlg.widget("enabled").setChecked(True)
        dlg.widget("gate_half_width_s").setValue(0.2)
        return QDialog.Accepted

    _drive_modal(pv, monkeypatch, commit)
    assert rp.enabled is True and rp.gate_half_width_s == 0.2
    pv.shutdown()


def test_an_analysis_written_before_the_feature_still_loads(qapp, tmp_path, monkeypatch):
    """An older Analysis file simply has no robust_peak table; from_dict then supplies the
    dataclass defaults, and the modal shows them."""
    from respmech.core.settings import Settings
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    s.processing.emg.remove_ecg = True
    d = s.to_dict()
    d["processing"]["emg"].pop("robust_peak", None)
    pv = PreviewScreen(AppState(Settings.from_dict(d)))
    seen = {}

    def act(dlg):
        seen["enabled"] = dlg.widget("enabled").isChecked()
        seen["width"] = dlg.widget("gate_half_width_s").value()
        return QDialog.Rejected

    _drive_modal(pv, monkeypatch, act)
    assert seen["enabled"] is False
    assert seen["width"] == 0.120
    pv.shutdown()


# -- 2. committing marks the analysis modified ---------------------------------
def test_committing_marks_the_analysis_modified(qapp, tmp_path, monkeypatch):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    win.state.settings.processing.emg.remove_ecg = True
    win.settings_screen._mark_clean()

    def commit(dlg):
        dlg.widget("enabled").setChecked(True)
        return QDialog.Accepted

    _drive_modal(win.preview_screen, monkeypatch, commit)
    assert win.settings_screen.is_dirty(), "a gated-peak commit does not mark modified"
    win.close()


# -- 3. the ECG prerequisite follows the control into the modal ----------------
def test_it_opens_disabled_with_the_hint_until_ecg_removal_is_on(qapp, tmp_path, monkeypatch):
    from respmech.ui.screens.preview_screen import NEEDS_ECG_GATE_HINT
    pv = _preview(qapp, tmp_path, remove_ecg=False)
    seen = {}

    def act(dlg):
        w = dlg.widget("enabled")
        seen["enabled"] = w.isEnabled()
        seen["tip"] = w.toolTip()
        seen["width_enabled"] = dlg.widget("gate_half_width_s").isEnabled()
        return QDialog.Rejected

    _drive_modal(pv, monkeypatch, act)
    assert seen["enabled"] is False, "the gated peak is editable with no heartbeats to gate on"
    assert NEEDS_ECG_GATE_HINT in seen["tip"], "disabled with no explanation"
    assert seen["width_enabled"] is False
    pv.shutdown()


def test_it_is_editable_once_ecg_removal_is_on(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path, remove_ecg=True)
    seen = {}

    def act(dlg):
        seen["enabled"] = dlg.widget("enabled").isEnabled()
        return QDialog.Rejected

    _drive_modal(pv, monkeypatch, act)
    assert seen["enabled"] is True
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


# -- 4. the guards live beside it in the modal ---------------------------------
def test_the_gated_controls_left_the_strip(qapp, tmp_path):
    pv = _preview(qapp, tmp_path)
    for name in ("rp_min_survival", "rp_min_island", "rp_long_rr", "rp_max_long_rr",
                 "rp_hr_margin", "emg_gated", "emg_gate_width", "gate_opts"):
        assert not hasattr(pv, name), f"{name} is back on the strip"
    pv.shutdown()


def test_the_guards_are_editable_in_the_modal(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path)

    def commit(dlg):
        dlg.widget("min_survival").setValue(0.55)
        dlg.widget("hr_ceiling_margin").setValue(0.2)
        return QDialog.Accepted

    _drive_modal(pv, monkeypatch, commit)
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


def test_committing_only_gated_settings_schedules_nothing(qapp, tmp_path, monkeypatch):
    """End to end through the real commit: gated columns are written by the batch and drawn
    by no panel, so a robust_peak-only OK must not queue any recompute."""
    pv = _preview(qapp, tmp_path)
    pv._pending_kinds = set()

    def commit(dlg):
        dlg.widget("enabled").setChecked(True)
        dlg.widget("gate_half_width_s").setValue(0.2)
        return QDialog.Accepted

    _drive_modal(pv, monkeypatch, commit)
    assert pv._pending_kinds == set(), f"a gated-only commit queued {pv._pending_kinds}"
    pv.shutdown()


def test_a_mixed_commit_still_recomputes(qapp, tmp_path, monkeypatch):
    """The guard above must not overreach: an OK that ALSO changed a conditioning setting
    (here the RMS window) must recompute as it always did."""
    pv = _preview(qapp, tmp_path)
    pv._pending_kinds = set()

    def commit(dlg):
        dlg.widget("enabled").setChecked(True)
        dlg.widget("rms_window_s").setValue(0.08)
        return QDialog.Accepted

    _drive_modal(pv, monkeypatch, commit)
    assert pv._pending_kinds, "a conditioning edit no longer schedules a recompute"
    pv.shutdown()


# -- 6. an untouched OK must not fight the app ---------------------------------
def test_untouched_ok_does_not_revert_a_value_the_app_wrote(qapp, tmp_path, monkeypatch):
    """The modal captures its values when it OPENS. With 'Auto' on, a finished noise sweep
    writes its chosen prop_decrease into the model — so committing everything the dialog
    holds put the stale opening value back, marked the analysis modified and queued a
    five-panel recompute, all from an OK the user pressed without touching anything.

    Reproduced here by writing the model mid-dialog, which is exactly what the sweep does.
    """
    pv = _preview(qapp, tmp_path)
    n = pv.state.settings.processing.emg.noise
    n.auto_prop = True
    n.prop_decrease = 0.60
    pv._pending_kinds = set()
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))

    def app_writes_then_ok(dlg):
        n.prop_decrease = 0.35          # the sweep lands while the modal is open
        return QDialog.Accepted         # ...and the user just presses OK

    _drive_modal(pv, monkeypatch, app_writes_then_ok)

    assert n.prop_decrease == 0.35, (
        "an untouched OK reverted the suppression strength the run had chosen")
    assert not edits, "an untouched OK marked the analysis modified"
    assert pv._pending_kinds == set(), (
        f"an untouched OK queued a recompute: {pv._pending_kinds}")
    pv.shutdown()


def test_an_edit_still_wins_over_a_concurrent_app_write(qapp, tmp_path, monkeypatch):
    """The guard above must not overreach: if the user DID move the field, their value is
    the one that must land, even though the app also wrote one meanwhile."""
    pv = _preview(qapp, tmp_path)
    n = pv.state.settings.processing.emg.noise
    n.auto_prop = False
    n.prop_decrease = 0.60

    def user_edits_and_app_writes(dlg):
        dlg.widget("prop_decrease").setValue(0.25)   # the user's deliberate choice
        n.prop_decrease = 0.35                       # a concurrent app write
        return QDialog.Accepted

    _drive_modal(pv, monkeypatch, user_edits_and_app_writes)
    assert n.prop_decrease == 0.25, "the user's edit was lost"
    pv.shutdown()

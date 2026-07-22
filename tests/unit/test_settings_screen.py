"""Settings screen behaviours: the channel-mapping hard gate (P1), the dirty flag +
window title + guarded New (P6), the live QC caution strip (P9), the surfaced signal
transforms (P3), the output "what to save" checklist (P4) and the detected-format
read-out (P28). Previously split across test_review_wave1/2/6.py."""
import os

import pytest

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _assign(sc, **roles):
    """Assign channels the only way the app can now: through _apply_channel_mapping, which
    is what the picker calls. Writing widgets is no longer possible — there are none."""
    m = {"flow": None, "volume": None, "poes": None, "pgas": None, "pdi": None,
         "emg": [], "entropy": []}
    m.update(roles)
    sc._apply_channel_mapping(m)


def _valid(sc, tmp):
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    sc._on_field_changed()


def test_required_channel_on_the_time_column_blocks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    assert sc._all_ok()                                  # a good mapping is ready
    _assign(sc, flow=5, volume=6, poes=1, pgas=8, pdi=9, emg=[2, 3, 4])   # poes -> the time axis
    assert not sc._all_ok()
    assert "column 1" in sc._channel_collision().lower()
    win.close()


def test_two_required_channels_sharing_a_column_blocks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    _assign(sc, flow=5, volume=6, poes=7, pgas=7, pdi=9, emg=[2, 3, 4])   # pgas == poes column
    assert not sc._all_ok()
    assert "same column" in sc._channel_collision().lower()
    win.close()


def test_dirty_flag_and_window_title(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    assert not sc.is_dirty()
    assert "new analysis" in win.windowTitle().lower()
    sc.samp_freq.setValue(1234); sc._on_field_changed()  # a real edit
    assert sc.is_dirty()
    assert win.windowTitle().rstrip().endswith("* (modified)")
    sc._mark_clean()
    assert not sc.is_dirty() and "(modified)" not in win.windowTitle()
    win.close()


def test_new_analysis_only_confirms_when_dirty(qapp, monkeypatch):
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    calls = []
    monkeypatch.setattr(ss.QMessageBox, "question",
                        staticmethod(lambda *a, **k: calls.append(1) or ss.QMessageBox.Discard))
    sc.new_analysis()                                   # not dirty -> no confirmation
    assert calls == []
    sc.samp_freq.setValue(999); sc._on_field_changed()   # now dirty
    sc.new_analysis()
    assert calls == [1]                                  # confirmed exactly once
    win.close()


def test_qc_strip_flags_cautions_and_clears(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    assert sc.qc.property("status") == "ok" and "no warnings" in sc.qc.text().lower()
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[5, 2, 3])   # EMG col 5 == flow
    assert sc.qc.property("status") == "warn"
    assert "overlap" in sc.qc.text().lower()
    win.close()

def test_output_checklist_binds_and_previews(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.save_processed.setChecked(True)
    sc.save_raw_fig.setChecked(True)
    sc._on_field_changed()
    d = sc.state.settings.output.data
    dg = sc.state.settings.output.diagnostics
    assert d.save_processed is True and dg.save_raw is True
    txt = sc.save_preview.text().lower()
    assert "processed csv" in txt          # the read-out reflects the ticked boxes
    win.close()


def test_format_readout(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv"); sc._on_inputs_changed()
    assert "columns" in sc.format_readout.text() and "comma" in sc.format_readout.text()
    assert sc.format_readout.property("status") == "info"
    sc.in_files.setText("*.nope"); sc._on_inputs_changed()
    assert sc.format_readout.property("status") == "warn"
    win.close()


def test_advanced_panel_roundtrips_toml_only_knobs(qapp, tmp_path):
    """The Advanced panel surfaces knobs that were previously TOML-only (audit #16/17/22-27):
    they must reflect the model and write back through to_state."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import BreathCountEntry
    s = synth_settings(str(tmp_path))
    s.processing.segmentation.buffer = 640
    s.processing.segmentation.peak.height = 0.33
    s.processing.wob.avg_resampling_obs = 750
    s.processing.ptp.baseline_window_s = 0.08
    s.processing.entropy.epochs = 3
    s.processing.emg.noise.n_fft = 512
    s.input.format.matlab_variant = "windows"
    # a filename containing a comma must survive the round-trip (one entry per line)
    s.processing.breath_counts = [BreathCountEntry("a,b.txt", 12)]
    win = MainWindow(AppState(s)); sc = win.settings_screen
    # The mechanics/EMG knobs (seg_buffer, avg_resamp, n_fft, breath_counts) are Preview-owned
    # now; Setup's to_state must LEAVE them untouched — a round trip through Setup preserves
    # them rather than reverting to a widget default.
    out = sc.to_state()
    assert out.processing.segmentation.buffer == 640 and out.processing.wob.avg_resampling_obs == 750
    assert out.processing.emg.noise.n_fft == 512
    assert [(e.file, e.count) for e in out.processing.breath_counts] == [("a,b.txt", 12)]
    # matlab_variant + entropy STAY Setup-owned and bound
    assert sc.matlab_variant.currentData() == "windows"
    sc.ent_epochs.setValue(4)
    sc.matlab_variant.setCurrentIndex(sc.matlab_variant.findData("mac"))
    out2 = sc.to_state()
    assert out2.processing.entropy.epochs == 4
    assert out2.input.format.matlab_variant == "mac"
    win.close()

def test_form_fields_are_bounded_not_full_width(qapp, tmp_path):
    """Fusion's QFormLayout grows every uncapped field to the whole form width, so these
    combos/text areas stretched the entire window (1433px at a 1700px window) while their
    spin-box siblings sat in a tidy column. The fix caps them with an opt-in `formField`
    QSS property.

    This asserts the PORTABLE invariant — a capped field is bounded to a fraction of the
    window and carries the property that caps it — and deliberately does NOT assert exact
    pixel widths or text-fit: those depend on the platform's font and the offscreen CI
    runner's font substitution (which renders these fields quite differently from a real
    Mac/Windows), so pinning pixels made a correct layout flake on font trivia. Exact fit
    is verified by measurement on the platforms the app actually runs on. Asserting the
    property is also what guards the stylesheet: a QSS syntax error drops the whole sheet,
    the caps vanish, and the fields stretch — caught by the width bound below.
    """
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    win.resize(1700, 950); win.show(); qapp.processEvents()

    capped = {sc.in_files: "compact",
              sc.matlab_variant: "wide", sc.group_regex: "wide"}
    for w, kind in capped.items():
        assert w.property("formField") == kind, f"{w} lost its {kind} cap"
        assert w.width() < win.width() / 3, f"{w} stretched full width ({w.width()})"
    # the channel picker is a button, not a banner: it must not stretch the card's width
    assert sc.btn_assign_channels.property("compact") is True
    assert sc.btn_assign_channels.width() < win.width() / 3
    # the caps really order by band (compact < wide), independent of the absolute font size
    widest_compact = max(w.width() for w, k in capped.items() if k == "compact")
    narrowest_wide = min(w.width() for w, k in capped.items() if k == "wide")
    assert widest_compact <= narrowest_wide

    # the browse-row paths (in/out folder, noise reference) legitimately stay full width:
    # they hold absolute paths and share their row with the Browse button
    assert sc.in_folder.width() > win.width() / 2
    win.close()


def test_open_reconciles_clamped_form_values_into_state(qapp, tmp_path):
    """from_state() clamps values the widgets cannot represent (a 0/None sampling frequency
    shows as 2000, an unknown token falls back to the first entry). Opening an analysis must
    persist those clamps back into state.settings — the batch worker runs a deep copy of
    state.settings, so a divergence there means it runs values the form never showed."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    s.input.format.sampling_frequency = 0            # the spin box (min 1) cannot show 0
    win = MainWindow(AppState(s))
    sc = win.settings_screen
    assert sc.samp_freq.value() == 2000              # from_state clamped it on construction...
    assert sc.state.settings.input.format.sampling_frequency == 0   # ...but state still diverges
    sc.enter_open_mode()                             # the open path must reconcile the two
    assert sc.state.settings.input.format.sampling_frequency == 2000
    win.close()


def test_save_writes_back_to_the_opened_file_after_confirming(qapp, tmp_path, monkeypatch):
    """'Save' overwrites the analysis that was opened — no chooser — but only after the user
    confirms, because it replaces a file they may share. Declining must write nothing."""
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    p = tmp_path / "study.toml"
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    sc.state.save_toml(str(p)); before = p.read_text()
    sc.state.settings_path = str(p)
    sc.samp_freq.setValue(1234); sc._on_field_changed()       # an edit to save
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: (asked.append(a[1]), QMessageBox.No)[1])
    assert sc.save_analysis() is False                        # declined
    assert asked and p.read_text() == before                  # ...nothing written
    assert sc.is_dirty()                                      # ...and still dirty
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    assert sc.save_analysis() is True
    assert "1234" in p.read_text() and not sc.is_dirty()      # written to the SAME file
    win.close()


def test_save_is_refused_while_the_settings_are_invalid(qapp, tmp_path, monkeypatch):
    """Only valid settings may be saved. Validity here is the SETTINGS' own — a missing
    input folder is about the machine, not the analysis, and must not hold the file
    hostage (that would block saving an hour's work over an unmounted drive)."""
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    assert sc.can_save()
    _assign(sc, flow=5, volume=6, poes=5, pgas=8, pdi=9, emg=[2, 3, 4])   # one column, two signals
    sc._on_field_changed()
    assert not sc.can_save() and sc._save_blocker()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[1]))
    assert sc.save_analysis_as() is False and warned          # refused, with a reason
    # a path problem is NOT a save blocker: the analysis itself is still coherent
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])   # a coherent mapping again
    sc.state.settings.input.folder = str(tmp_path / "unmounted")
    assert sc.can_save()
    win.close()


def test_closing_with_unsaved_changes_offers_to_save(qapp, tmp_path, monkeypatch):
    """A dirty analysis must be rescued before the window goes. Cancel aborts the close and
    must leave the window fully live — the prompt therefore runs BEFORE worker teardown."""
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    win.show()                                                # the prompt is for USER closes:
    sc = win.settings_screen                                  # a hidden window never asks
    sc.samp_freq.setValue(1234); sc._on_field_changed()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
    ev = QCloseEvent(); win.closeEvent(ev)
    assert not ev.isAccepted()                                # close aborted
    assert win.preview_screen.isEnabled()                     # ...and nothing was torn down
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    ev = QCloseEvent(); win.closeEvent(ev)
    assert ev.isAccepted()
    # a clean analysis is never asked about
    win2 = MainWindow(AppState(synth_settings(str(tmp_path))))
    win2.show()
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: pytest.fail("clean analysis must not prompt"))
    ev = QCloseEvent(); win2.closeEvent(ev)
    assert ev.isAccepted()
    # a never-shown window's close is programmatic (scripting, headless tests): even a
    # DIRTY one must not pop a modal — there is nobody to answer it, so it would hang
    win3 = MainWindow(AppState(synth_settings(str(tmp_path))))
    win3.settings_screen.samp_freq.setValue(999); win3.settings_screen._on_field_changed()
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: pytest.fail("hidden window must not prompt"))
    ev = QCloseEvent(); win3.closeEvent(ev)
    assert ev.isAccepted()


def test_analysis_menu_lists_five_recents_and_guards_opening_them(qapp, tmp_path, isolated_prefs, monkeypatch):
    """The Analysis menu shows the 5 most recent analyses (of the 8 prefs stores), rebuilt
    on every drop-down; each carries the full path as tooltip; opening one over unsaved
    edits goes through the same Save/Discard/Cancel guard as closing. Save is offered only
    for a dirty analysis with a file to overwrite; Save as… only needs savable settings."""
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    paths = []
    for i in range(7):                                    # more than the menu shows
        p = tmp_path / f"study_{i}.toml"; p.write_text("# analysis"); paths.append(str(p))
        isolated_prefs.add_recent_analysis(str(p))
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    win._refresh_analysis_menu()                          # what aboutToShow triggers
    assert len(win._recent_actions) == 5
    assert win._recent_actions[0].toolTip() == paths[-1]  # most recent first, full path
    assert "study_6.toml" in win._recent_actions[0].text()
    assert win._recent_sep.isVisible()
    # rebuilding must not accumulate: same 5 actions, and the removed ones are deleted
    win._refresh_analysis_menu()
    assert len(win._recent_actions) == 5
    # Save/Save as enable matrix: Save is offered when the analysis is NEW or DIRTY
    sc = win.settings_screen
    assert win._act_save.isEnabled()                      # new (no file yet) -> Save works
    assert win._act_save_as.isEnabled()
    sc.samp_freq.setValue(1234); sc._on_field_changed()   # dirty, still no settings_path
    win._refresh_analysis_menu()
    assert win._act_save.isEnabled() and win._act_save_as.isEnabled()
    sc.state.settings_path = paths[0]
    win._refresh_analysis_menu()
    assert win._act_save.isEnabled()                      # dirty + a file to overwrite
    sc._mark_clean()
    win._refresh_analysis_menu()
    assert not win._act_save.isEnabled()                  # clean opened file: nothing to save
    assert win._act_save_as.isEnabled()
    # opening a recent over a DIRTY analysis honours the guard: Cancel aborts
    sc.samp_freq.setValue(777); sc._on_field_changed()    # dirty again for the guard test
    opened = []
    monkeypatch.setattr(sc, "open_analysis", lambda p: opened.append(p) or True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
    win._open_recent(paths[-1])
    assert opened == []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    win._open_recent(paths[-1])
    assert opened == [paths[-1]]
    win.close()


def test_save_as_writes_the_picked_file_and_a_failed_save_stays_dirty(qapp, tmp_path, monkeypatch):
    """The save flow's remaining paths: 'Save' with no associated file falls through to
    Save as…; Save as… writes the chooser-picked file and marks the analysis clean; a
    FAILED write must leave it dirty so the edits are not silently lost."""
    from PySide6.QtWidgets import QFileDialog
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    sc.samp_freq.setValue(1234); sc._on_field_changed()
    assert sc.state.settings_path is None
    picked = str(tmp_path / "picked.toml")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (picked, "")))
    assert sc.save_analysis() is True                     # no path -> falls through to Save as…
    assert os.path.exists(picked) and not sc.is_dirty()
    assert "picked.toml" in win.windowTitle()
    # a failed write keeps the dirty flag (the edits are still only in the form)
    sc.samp_freq.setValue(4321); sc._on_field_changed()
    monkeypatch.setattr(sc.state, "save_toml",
                        lambda p: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(sc, "_report_error", lambda *a: None)   # swallow the error dialog
    assert sc.save_analysis_as() is False
    assert sc.is_dirty()
    win.close()


def test_discard_guard_is_reentrant_safe_and_hides_save_when_unsavable(qapp, tmp_path, monkeypatch):
    """Two properties of the unsaved-changes guard that keep cocoa's modal sessions in
    order: a re-entrant call (Cmd+Q while another guard prompt is up) aborts the NEW
    action instead of stacking a second window-modal box, and Save is only offered while
    the settings can actually be saved — so _refuse_save's warning can never chain onto
    the prompt (and the choice matrix matches the Analysis menu's Save gating)."""
    from PySide6.QtWidgets import QMessageBox
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    sc.samp_freq.setValue(1234); sc._on_field_changed()   # dirty
    seen = []

    def fake_question(parent, title, text, buttons=None, default=None):
        seen.append((title, buttons))
        # re-enter while the prompt is "up": the nested call must refuse, not prompt
        if len(seen) == 1:
            assert sc.confirm_discard_changes("Close RespMech") is False
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    assert sc.confirm_discard_changes("Open analysis") is False
    assert len(seen) == 1                                 # exactly ONE box, no stacking
    assert seen[0][1] & QMessageBox.Save                  # savable settings offer Save
    # unsavable settings must not offer Save at all
    _assign(sc, flow=5, volume=6, poes=5, pgas=8, pdi=9, emg=[2, 3, 4])
    assert not sc.can_save()
    seen.clear()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda p, t, x, buttons=None, default=None:
                                     seen.append(buttons) or QMessageBox.Cancel))
    sc.confirm_discard_changes("Close RespMech")
    assert seen and not (seen[0] & QMessageBox.Save)
    assert seen[0] & QMessageBox.Discard and seen[0] & QMessageBox.Cancel
    win.close()


def test_preview_owned_settings_mark_the_analysis_dirty(qapp, tmp_path):
    """Noise/ECG params and breath exclusions are edited on the Preview screen but land in
    the saved .toml, so a user edit there must dirty the analysis exactly like a Setup
    edit — the title, the close guard and Save-gating all read the same flag. Programmatic
    fills must NOT dirty (loading an analysis would otherwise immediately mark it edited)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc, pv = win.settings_screen, win.preview_screen

    # programmatic fill: syncing widgets FROM settings is not an edit
    sc._mark_clean()
    pv._load_noise_params(); pv._load_ecg_params()
    assert not sc.is_dirty()

    # a user noise-param edit dirties (the spinbox slot writes into state.settings)
    pv.noise_prop.setValue(0.42); pv._on_noise_param_changed()
    assert sc.is_dirty()
    assert win.windowTitle().rstrip().endswith("* (modified)")

    # a user ECG-param edit dirties
    sc._mark_clean(); assert not sc.is_dirty()
    pv._on_ecg_param_changed()
    assert sc.is_dirty()

    # toggling a breath exclusion dirties (exclude_breaths is saved state)
    sc._mark_clean()
    pv.file_combo.blockSignals(True); pv.file_combo.addItem("synth_case_A.csv")
    pv.file_combo.setCurrentText("synth_case_A.csv"); pv.file_combo.blockSignals(False)
    pv._breath_spans = {1: (0.0, 1.0)}
    pv._toggle_breath(1)
    assert sc.is_dirty()
    win.close()


def test_every_edit_revalidates_into_the_status_bar(qapp, tmp_path):
    """There is no Validate button any more: each edit re-runs validation and reports the
    verdict, so a problem surfaces where the user made it rather than on a later click."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    sc.samp_freq.setValue(1234)                       # a valid edit
    assert sc.status.text() == "Settings valid ✓"
    sc.in_folder.setText(str(tmp_path / "nope"))      # a filesystem problem
    sc._on_inputs_changed()
    assert "Invalid:" in sc.status.text() and "input folder" in sc.status.text()
    win.close()


def test_run_lock_covers_the_header_analysis_menu(qapp, tmp_path):
    """The run lock exists so a running worker's settings can't be swapped out from under
    it. The Analysis menu (New/Open/Save) sits in the header, OUTSIDE the locked screen,
    so it has to be disabled alongside it."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    win._on_run_started()
    assert not win.settings_screen.isEnabled() and not win.analysis_btn.isEnabled()
    win._on_run_finished()
    assert win.settings_screen.isEnabled() and win.analysis_btn.isEnabled()
    win.close()


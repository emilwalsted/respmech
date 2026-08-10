"""Settings screen behaviours: the channel-mapping hard gate (P1), the dirty flag +
window title + guarded New (P6), the live QC caution strip (P9), the surfaced signal
transforms (P3), the output "what to save" checklist (P4) and the detected-format
read-out (P28). Previously split across test_review_wave1/2/6.py."""
import os

import pytest

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings, write_delim, write_xlsx

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
    assert "processed-signal csv" in txt   # the read-out reflects the ticked boxes
    win.close()


def test_format_readout(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    # B01: the synth fixtures are genuinely 1000 Hz — match it, or the manifest correctly
    # (and separately, see test_frequency_only_mismatch_marks_the_readout_as_warn_too)
    # flags the spinbox's arbitrary 2000 Hz default as a real mismatch and this becomes a
    # test of THAT caution rather than of the column/delimiter text this test is about.
    sc.samp_freq.setValue(1000)
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv"); sc._on_inputs_changed()
    assert "columns" in sc.format_readout.text() and "comma" in sc.format_readout.text()
    assert sc.format_readout.property("status") == "info"
    sc.in_files.setText("*.nope"); sc._on_inputs_changed()
    assert sc.format_readout.property("status") == "warn"
    win.close()


def test_format_readout_singular_and_plural_wording(qapp):
    """'N file(s) matched' reads as broken English at either count — A03 fixed it to a
    real singular/plural, matched here at n=1 (one exact file) and n>1 (the *.csv mask)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_A.csv"); sc._on_inputs_changed()
    assert sc.format_readout.text().startswith("1 file matched")
    sc.in_files.setText("synth_case_*.csv"); sc._on_inputs_changed()
    assert sc.format_readout.text().startswith("2 files matched")
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


def test_entropy_fields_are_named_and_explained_for_what_they_are(qapp, tmp_path):
    """D11 (UI-overhaul): the old label 'Embedding (m)' and tooltip 'Embedding dimension (m)
    ...; 2 by convention' were both wrong — core/entropy.py sets M = sample_length - 1, so the
    stored value (default 2) has always meant m = 1, never m = 2. Same story for tolerance:
    core/compute.py multiplies it by each column's own std, which neither old text said.
    Checked on the FULL toolTip(), not an elided text(), since that is the only place the
    variable-path + description actually live (see settings_screen.py's _row helper); the
    LABEL is found via QFormLayout.labelForField, the same accessor _row itself relies on to
    keep the label and the field showing the same tooltip."""
    from PySide6.QtWidgets import QFormLayout
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    form = sc.ent_epochs.parentWidget().layout()
    assert isinstance(form, QFormLayout)

    epochs_label = form.labelForField(sc.ent_epochs)
    assert epochs_label.text() == "Template length (m + 1)"
    tip = sc.ent_epochs.toolTip()
    assert epochs_label.toolTip() == tip, "label and field must share the same tooltip"
    assert "processing.entropy.epochs" in tip
    assert "one more than the embedding dimension" in tip
    assert "default 2 gives m = 1" in tip
    assert "m = 2" in tip and "conventional in the literature" in tip
    assert "2 by convention" not in tip, "the old, wrong claim must not survive verbatim"

    tol_label = form.labelForField(sc.ent_tol)
    assert tol_label.text() == "Tolerance (r), × SD"
    tol_tip = sc.ent_tol.toolTip()
    assert tol_label.toolTip() == tol_tip
    assert "processing.entropy.tolerance" in tol_tip
    assert "multiple of the per-column standard deviation" in tol_tip
    assert "0.1 × SD by default" in tol_tip and "0.2 × SD" in tol_tip
    win.close()


def test_entropy_readout_states_m_and_r_in_the_apps_own_words(qapp, tmp_path):
    """The live caption under the two entropy fields does the m = epochs - 1 arithmetic for
    the user, in the terms a methods section would use, and follows every edit."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    sc.ent_epochs.setValue(2)
    sc.ent_tol.setValue(0.1)
    assert sc.ent_caption.text() == "Computing SampEn with m = 1, r = 0.1 × SD."
    sc.ent_epochs.setValue(3)
    assert sc.ent_caption.text() == "Computing SampEn with m = 2, r = 0.1 × SD."
    sc.ent_tol.setValue(0.2)
    assert sc.ent_caption.text() == "Computing SampEn with m = 2, r = 0.2 × SD."
    win.close()


def test_decimal_separator_picker_reflects_and_writes_the_model(qapp, tmp_path):
    """Ticket D03: the manual override for CSV/text decimal separator, next to MATLAB file
    variant on the Input card. Like matlab_variant, it is Setup-owned and bound: from_state
    seeds it, to_state writes it back, and it round-trips through a loaded analysis."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    s.input.format.decimal = ","
    win = MainWindow(AppState(s)); sc = win.settings_screen
    assert sc.decimal_sep.currentData() == ","          # from_state seeded it from the model
    sc.decimal_sep.setCurrentIndex(sc.decimal_sep.findData("."))
    out = sc.to_state()
    assert out.input.format.decimal == "."              # to_state wrote the manual override back
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
              sc.matlab_variant: "wide", sc.decimal_sep: "wide", sc.group_regex: "wide"}
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

    # the browse-row paths (in/out folder, noise reference) legitimately stay full width —
    # of their own CARD, not the whole window (ticket B05: Setup splits into two columns,
    # so the Input card is now roughly half the window, and a field filling its card's
    # width no longer fills half the window's own width either)
    assert sc.in_folder.width() > sc._card_input.width() / 2
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
    sc.state.save_toml(str(p)); before = p.read_text(encoding="utf-8")
    sc.state.settings_path = str(p)
    sc.samp_freq.setValue(1234); sc._on_field_changed()       # an edit to save
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: (asked.append(a[1]), QMessageBox.No)[1])
    assert sc.save_analysis() is False                        # declined
    assert asked and p.read_text(encoding="utf-8") == before                  # ...nothing written
    assert sc.is_dirty()                                      # ...and still dirty
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    assert sc.save_analysis() is True
    assert "1234" in p.read_text(encoding="utf-8") and not sc.is_dirty()      # written to the SAME file
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

    # a user noise-param edit dirties (the slot writes into state.settings)
    pv.noise_auto.setChecked(not pv.noise_auto.isChecked()); pv._on_noise_param_changed()
    assert sc.is_dirty()
    assert win.windowTitle().rstrip().endswith("* (modified)")

    # a user ECG-param edit dirties
    sc._mark_clean(); assert not sc.is_dirty()
    pv._on_ecg_param_changed()
    assert sc.is_dirty()

    # toggling a breath exclusion dirties (exclude_breaths is saved state)
    sc._mark_clean()
    # select_filename sets the rail's identity directly — no row need exist for it
    # (unlike the old file_combo, which needed a fake item added first).
    pv.file_rail.select_filename("synth_case_A.csv")
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
    assert sc.status.text() == "Setup valid ✓"
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


# --------------------------------------------------------------------------- #
# B01: the batch manifest — the read-out and QC strip reflect the WHOLE folder, not
# just the first matching file
# --------------------------------------------------------------------------- #
def test_format_readout_flags_a_column_outlier_and_qc_is_not_green(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b", "c"):
        write_delim(tmp_path / f"{n}.csv", 9)
    write_delim(tmp_path / "d.csv", 8)             # the one outlier
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc._on_inputs_changed()
    text = sc.format_readout.text()
    assert "4 files matched" in text
    assert "d.csv" in text and "8 columns" not in text.split("d.csv")[0]  # named as the outlier
    assert sc.format_readout.property("status") == "warn"
    # the QC strip must ALSO carry the caveat — _update_qc must never say "no warnings"
    # about a batch it has not looked at file-by-file. A valid channel mapping + output
    # folder are needed here (ticket B07) so the strip's own hard-blocker check (unset
    # channels, unset output) does not mask the softer outlier caution this test is about.
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp_path / "out"))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    assert sc.qc.property("status") == "warn"
    assert "d.csv" in sc.qc.text()
    win.close()


def test_format_readout_flags_a_header_block_folder(qapp, tmp_path):
    """Ticket D01: a LabChart-style export whose header block (Interval=/ChannelTitle=/
    Range= lines) sniffs as a low, but MAJORITY-consistent, column count must show up as a
    warning on the Input card — not the previous 'info'/clean status the bare first-line
    sniff used to give it, since every file in the repro batch agrees on the same wrong
    shape and so wins the majority vote outright."""
    from respmech.ui.main_window import MainWindow
    header = "Interval=\t0.001 s\nChannelTitle=\tFlow\tPoes\tPgas\tPdi\n"
    rows = "\n".join(f"{i * 0.001:.3f}\t{i}\t{i + 1}\t{i + 2}\t{i + 3}" for i in range(10))
    for name in ("P05_60W.txt", "P06_60W.txt"):
        (tmp_path / name).write_text(header + rows + "\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.txt")
    sc._on_inputs_changed()
    text = sc.format_readout.text()
    assert "2 files matched" in text
    assert "header block" in text.lower()
    assert sc.format_readout.property("status") == "warn"
    win.close()


def test_qc_strip_also_flags_a_header_block_folder(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    header = "Interval=\t0.001 s\nChannelTitle=\tFlow\tPoes\tPgas\tPdi\n"
    rows = "\n".join(f"{i * 0.001:.3f}\t{i}\t{i + 1}\t{i + 2}\t{i + 3}" for i in range(10))
    for name in ("P05_60W.txt", "P06_60W.txt"):
        (tmp_path / name).write_text(header + rows + "\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.txt")
    sc._on_inputs_changed()
    # a valid channel mapping + output folder (ticket B07 pattern): otherwise the strip's
    # own hard blocker (unset channels/output) masks the softer header caution under test.
    # integrate_from_flow=True stands in for a volume column: the fixture's 5-column file
    # (time + 4 data columns) has no column left to spare for one.
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp_path / "out"))
    sc.state.settings.processing.volume.integrate_from_flow = True
    _assign(sc, flow=2, poes=3, pgas=4, pdi=5)
    assert sc.qc.property("status") == "warn"
    assert "header block" in sc.qc.text().lower()
    win.close()


def test_format_readout_no_longer_misreports_xlsx(qapp, tmp_path):
    pytest.importorskip("openpyxl")
    from respmech.ui.main_window import MainWindow
    write_xlsx(tmp_path / "a.xlsx", 9)
    write_xlsx(tmp_path / "b.xlsx", 9)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.xlsx")
    sc._on_inputs_changed()
    text = sc.format_readout.text()
    assert "whitespace-separated" not in text
    assert "1 columns" not in text
    assert "9 columns" in text and "Excel" in text
    win.close()


def test_qc_strip_flags_a_sampling_frequency_mismatch(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    write_delim(tmp_path / "at1000.csv", 9, fs=1000.0)
    write_delim(tmp_path / "at2000.csv", 9, fs=2000.0)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    # a valid channel mapping + output folder (ticket B07): otherwise the strip's own hard
    # blocker (unset channels/output) masks the softer frequency caution under test here.
    sc.out_folder.setText(str(tmp_path / "out"))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    assert sc.qc.property("status") == "warn"
    assert "2000 Hz" in sc.qc.text() and "at2000.csv" in sc.qc.text()
    # not a hard block — a frequency caveat must not by itself take down the guided flow
    assert "not the 1000 Hz" in sc.qc.text()
    win.close()


def test_mask_narrowing_is_reported_on_setup(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b", "c"):
        write_delim(tmp_path / f"{n}.csv", 9)
    write_delim(tmp_path / "d.txt", 9, sep="\t")
    win = MainWindow(AppState()); sc = win.settings_screen
    # valid channels + output FIRST (ticket B07): _assign's own _on_inputs_changed() would
    # otherwise re-narrow an ALREADY-single-pattern mask on a second pass and rebuild a
    # manifest with mask_narrowed_from reset to None, silently erasing the very caution
    # this test is about — narrow the mask exactly once, after everything else is valid.
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp_path / "out"))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv; *.txt")
    sc._on_inputs_changed()
    assert sc.in_files.text() == "*.csv"                   # still narrows, exactly as before
    assert "narrowed" in sc.format_readout.text()
    assert "*.csv" in sc.format_readout.text()
    assert sc.format_readout.property("status") == "warn"
    assert "narrowed" in sc.qc.text()
    win.close()


def test_manifest_cache_survives_across_edits_and_is_reused(qapp, tmp_path):
    """The same cache dict is reused for the screen's whole lifetime, so re-editing an
    unrelated field (which re-triggers nothing folder-related) never re-probes a file the
    scan has already seen for this exact folder+mask."""
    from respmech.ui.main_window import MainWindow
    write_delim(tmp_path / "a.csv", 9)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc._on_inputs_changed()
    assert len(sc._manifest_cache) > 0
    cache_before = dict(sc._manifest_cache)
    sc._on_inputs_changed()                                # same folder/mask again
    assert sc._manifest_cache == cache_before               # nothing new probed
    win.close()


def test_opening_a_saved_analysis_shows_a_fresh_qc_verdict_immediately(qapp, tmp_path):
    """Self-review regression (05-08-2026): _update_qc used to run BEFORE the manifest it
    now depends on was (re)built on the open-analysis path (from_state), and
    enter_open_mode's own manifest rebuild was never followed by a QC refresh at all — so
    opening a saved analysis whose folder had a real caveat could still show 'No warnings'
    until the user made an edit or switched tabs. The whole point of this ticket was to
    never show a stale all-clear; this is the open-a-file path, not just the live-edit one."""
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b", "c"):
        write_delim(tmp_path / f"{n}.csv", 9)
    write_delim(tmp_path / "d.csv", 8)             # the caveat that must survive Open
    src = MainWindow(AppState()); src_sc = src.settings_screen
    src_sc.in_folder.setText(str(tmp_path)); src_sc.in_files.setText("*.csv")
    src_sc.samp_freq.setValue(1000)
    src_sc._on_inputs_changed()
    _assign(src_sc, flow=5, volume=9, poes=6, pgas=7, pdi=8)   # a savable mapping over the 9-col files
    assert src_sc.can_save()                # sanity: a column-count outlier is a caution, not a save-blocker
    p = tmp_path / "analysis.toml"
    src_sc.state.save_toml(str(p))
    src.close()

    # a brand-new screen, as if the app had just started — _manifest is None until Open
    win = MainWindow(AppState()); sc = win.settings_screen
    assert sc._manifest is None
    assert sc.open_analysis(str(p)) is True
    # the caveat must be visible WITHOUT any further edit or tab switch
    assert sc._manifest is not None and sc._manifest.outliers
    assert sc.qc.property("status") == "warn"
    assert "d.csv" in sc.qc.text()
    win.close()


def test_correcting_the_sampling_frequency_clears_the_stale_caution(qapp, tmp_path):
    """Self-review regression (05-08-2026): the frequency-mismatch caution names the
    Settings value it compared against (Manifest.settings_fs, frozen at scan time). Fixing
    samp_freq to match the caution's own advice used to leave the OLD value quoted back at
    the user — 'not the 1000 Hz set here' — even after the field genuinely read 2000 Hz,
    because nothing rebuilt the manifest on a samp_freq-only edit."""
    from respmech.ui.main_window import MainWindow
    write_delim(tmp_path / "at1000.csv", 9, fs=1000.0)
    write_delim(tmp_path / "at2000.csv", 9, fs=2000.0)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    # valid channels + output (ticket B07), else the hard blocker masks the frequency caution
    sc.out_folder.setText(str(tmp_path / "out"))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    assert "not the 1000 Hz" in sc.qc.text()
    sc.samp_freq.setValue(2000)                    # the user does exactly what the caution said
    assert sc._manifest.settings_fs == 2000
    assert "not the 1000 Hz" not in sc.qc.text()
    # the OTHER file (at1000.csv) is now the mismatch instead
    assert "not the 2000 Hz" in sc.qc.text() and "at1000.csv" in sc.qc.text()
    win.close()


def test_frequency_only_mismatch_marks_the_readout_as_warn_too(qapp, tmp_path):
    """Self-review regression (05-08-2026): format_readout's status used to check
    outliers/narrowing/no-majority explicitly but never freq_mismatches, so a folder with
    consistent COLUMNS but a frequency disagreement showed 'info' on the read-out while the
    QC strip correctly said 'warn' for the exact same scan. Both must agree — that
    reconciliation is the entire point of this ticket."""
    from respmech.ui.main_window import MainWindow
    write_delim(tmp_path / "at1000.csv", 9, fs=1000.0)
    write_delim(tmp_path / "at2000.csv", 9, fs=2000.0)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    assert sc._manifest.outliers == ()              # columns agree — only frequency disagrees
    assert sc.format_readout.property("status") == "warn"
    # valid channels + output (ticket B07), else the hard blocker masks the frequency caution
    sc.out_folder.setText(str(tmp_path / "out"))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    assert sc.qc.property("status") == "warn"
    win.close()


def test_more_than_three_outliers_are_truncated_in_the_readout(qapp, tmp_path):
    """_named_list caps a caution at 3 names + '+N more' — asserted here at a scale (5
    outliers) no other test in this suite reaches, so a regression to an unbounded list
    over a large batch would actually be caught."""
    from respmech.ui.main_window import MainWindow
    for n in range(6):                              # majority: must outnumber the 5 outliers
        write_delim(tmp_path / f"rec{n}.csv", 9)
    for n in range(5):
        write_delim(tmp_path / f"odd{n}.csv", 8)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc._on_inputs_changed()
    assert sc._manifest.majority_columns == 9       # sanity: the 6 recN.csv files won the vote
    text = sc.format_readout.text()
    assert "5 files differ" in text
    assert "+2 more" in text
    assert text.count("odd") == 3                  # only the first 3 are named
    win.close()


def test_all_files_too_few_columns_gives_an_accurate_message(qapp, tmp_path):
    """Self-review regression (05-08-2026): a folder where every matched file has fewer
    than 2 columns (readable, just no signal beyond a time axis) used to say 'none of
    these files could be read' — true for an unreadable file, but wrong for one that reads
    fine and simply lacks columns. Both cases land in majority_columns is None (neither can
    win the >=2 majority floor), so the message must cover both rather than naming only
    one of the two possible reasons."""
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b"):
        write_delim(tmp_path / f"{n}.csv", 1)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc._on_inputs_changed()
    assert sc._manifest.majority_columns is None    # sanity: no file could win the >=2 floor
    text = sc.format_readout.text()
    assert "enough columns" in text and "unreadable" in text
    assert sc.format_readout.property("status") == "warn"
    win.close()


# --------------------------------------------------------------------------- #
# B07: the QC strip renders the app's own verdict instead of reaching a THIRD,
# independent one — see ``SettingsScreen._qc_verdict``.
# --------------------------------------------------------------------------- #
def test_qc_strip_is_muted_not_green_when_nothing_matches(qapp, tmp_path):
    """Acceptance (ticket B07): a folder containing only a non-matching file (a PDF, say)
    is 'nothing has been checked yet', not a clean bill of health — the strip must show
    neither the green checkmark nor an amber warning, and it must not contradict the
    Input card's own read-out, which already says the same thing in its own words."""
    from respmech.ui.main_window import MainWindow
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4\n")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc._on_inputs_changed()
    assert sc._manifest is not None and sc._manifest.files == ()
    assert sc.format_readout.property("status") == "warn"          # the Input card's own note
    assert "No files match" in sc.format_readout.text()
    assert sc.qc.property("status") == "muted"
    assert "✓" not in sc.qc.text() and "⚠" not in sc.qc.text()
    win.close()


def test_qc_strip_never_says_no_warnings_while_a_hard_blocker_stands(qapp, tmp_path):
    """Ticket D02 acceptance: an unset Volume channel (no 'derive from flow') is a HARD
    blocker (ui.validation.blockers) — the QC strip must show it as an error, never fall
    through to the green 'Ready to run — no warnings.' verdict while _all_ok() is False.
    This is the exact contradiction ticket D02's point 4 exists to rule out."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp_path))
    sc._on_inputs_changed()
    _assign(sc, flow=5, poes=7, pgas=8, pdi=9)          # every required role EXCEPT volume
    assert sc.state.settings.input.channels.volume is None
    assert sc.state.settings.processing.volume.integrate_from_flow is False
    assert sc._all_ok() is False
    assert sc.qc.property("status") == "error"
    assert "no warnings" not in sc.qc.text().lower()
    assert "volume" in sc.qc.text().lower()
    win.close()


def test_qc_strip_flags_unreadable_data_and_names_the_decimal_separator(qapp, tmp_path):
    """Acceptance (ticket B07), written before the fix per the ticket's own test
    discipline: a folder of semicolon-separated CSVs read under the default comma-decimal
    setting never wins the manifest's >=2-column majority vote (peek_columns splits every
    line on ',' and finds one field) — B01's ``majority_columns is None``. The strip must
    say so in error style and name the decimal separator as the thing to check, since
    ``path_problem`` alone cannot see this: the mask genuinely matched real files."""
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b"):
        write_delim(tmp_path / f"{n}.csv", 9, sep=";")
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp_path / "out"))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    assert sc._manifest.files != () and sc._manifest.majority_columns is None
    assert sc.qc.property("status") == "error"
    assert "could be read as data" in sc.qc.text()
    assert "decimal separator" in sc.qc.text()
    win.close()


def test_qc_strip_warns_about_the_effective_emg_rate_after_resampling(qapp, tmp_path):
    """Acceptance (ticket B07): with 'Resample before analysis' on and a 200 Hz target
    against EMG channels on a 1000 Hz recording, the strip must warn about the rate the
    batch will actually ANALYSE at (200 Hz), not the recorded rate (1000 Hz, which alone
    is not low for EMG) — ``workers.py`` documents that Preview never resamples, so this is
    the one place left that could silently certify a genuinely wrong EMG normalisation."""
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b"):
        write_delim(tmp_path / f"{n}.csv", 9, fs=1000.0)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp_path / "out"))
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    assert sc.qc.property("status") == "ok"                      # sanity: not warned yet
    assert "low for EMG" not in sc.qc.text()
    samp = sc.state.settings.processing.sampling
    samp.resample = True; samp.resample_to_frequency = 200
    sc.refresh_qc()
    assert sc.qc.property("status") == "warn"
    assert "analysis rate 200 Hz (resampled from 1000 Hz) is low for EMG" in sc.qc.text()
    assert "Mechanics ▸ Advanced" in sc.qc.text()
    win.close()


def test_green_checkmark_only_when_ready_and_at_least_one_file_was_read(qapp, tmp_path):
    """Acceptance (ticket B07): one test walking every non-green tier plus a genuinely
    valid batch, asserting the checkmark shows in exactly the last one — the strip's own
    former promise ('columns/sampling look consistent') covered more than it actually
    checked; this proves the new, narrower promise ('_all_ok() and >=1 file read') holds
    across all four non-ready states, not just whichever one a single test happens to try."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen

    # 1) nothing scanned yet
    assert sc.qc.property("status") == "muted"

    # 2) hard blocker: no channels assigned
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv")
    write_delim(tmp_path / "a.csv", 9)
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    assert sc.qc.property("status") == "error"

    # 3) hard blocker: unreadable data (decimal-separator mismatch)
    for f in tmp_path.glob("*.csv"):
        f.unlink()
    write_delim(tmp_path / "b.csv", 9, sep=";")
    sc._on_inputs_changed()
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    sc.out_folder.setText(str(tmp_path / "out")); sc._on_field_changed()
    assert sc._manifest.majority_columns is None
    assert sc.qc.property("status") == "error"

    # 4) soft caution: output folder equals the recordings folder
    for f in tmp_path.glob("*.csv"):
        f.unlink()
    write_delim(tmp_path / "c.csv", 9)
    sc._on_inputs_changed()
    sc.out_folder.setText(str(tmp_path)); sc._on_field_changed()
    assert sc.qc.property("status") == "warn"

    # 5) genuinely ready: valid channels, valid output, one real file actually read
    sc.out_folder.setText(str(tmp_path / "out")); sc._on_field_changed()
    assert sc._all_ok()
    assert sc._manifest.majority_columns is not None
    assert sc.qc.property("status") == "ok"
    assert sc.qc.text() == "✓  Ready to run — no warnings."
    win.close()


# --------------------------------------------------------------------------- #
# B05: Setup splits into two columns — the rig (Input + Channels) and the leverance
# (Output + Sample entropy) — and collapses to one below a threshold width.
# --------------------------------------------------------------------------- #
def test_setup_splits_into_two_columns_when_wide(qapp, tmp_path):
    """Derives its 'wide' width from the layout's OWN measured column target (the house
    convention — see test_section_flow.py::test_the_column_count_follows_the_width_and_
    never_oscillates), never a hand-picked pixel literal, and asserts unconditionally: a
    gated ``if columns_for(...) >= 2:`` around the one real assertion would let this test
    pass even if the split silently never triggered at any realistic width — which is
    exactly what happened here in self-review (06-08-2026): the Output card's FlowLayout
    checkbox rows inflated ``leverance.sizeHint()`` enough that the real two-column
    threshold sat past 1550 px, above a 1700 px-gated test's own comfort zone, but STILL
    inside where that old test's silent skip would have hidden it. See _FlowGroup for the
    fix (its sizeHint no longer votes with a full one-line width)."""
    from PySide6.QtCore import QRect
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    win.show()
    for _ in range(4):
        qapp.processEvents()
    cols = sc._columns_layout
    target, gap = cols.column_target(), cols._hgap
    wide = 2 * target + gap + 80                   # comfortably past the 2-column floor
    cols.setGeometry(QRect(0, 0, wide, 4000))
    assert cols.columns_for(wide) == 2, (
        f"two columns never afforded even at {wide}px (target={target}, gap={gap}) — "
        f"the split is not actually happening")
    assert sc._card_output.mapTo(sc, sc._card_output.rect().topLeft()).x() >= \
        sc._card_input.mapTo(sc, sc._card_input.rect().topRight()).x()
    win.close()


def test_setup_collapses_to_one_column_when_narrow(qapp, tmp_path):
    """Mirrors the wide test: the narrow width is derived from ``column_target()`` (one
    pixel under the floor a single column needs to stop being the only legal answer),
    never guessed, and the geometry assertion is unconditional."""
    from PySide6.QtCore import QRect
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    win.show()
    for _ in range(4):
        qapp.processEvents()
    cols = sc._columns_layout
    target = cols.column_target()
    narrow = max(1, target - 1)                     # too narrow to afford a second column
    cols.setGeometry(QRect(0, 0, narrow, 4000))
    assert cols.columns_for(narrow) == 1, (
        f"still more than one column at {narrow}px (target={target}) — the width chosen "
        f"for 'narrow' was not actually narrow enough")
    # a single column: rig and leverance share an x position (stacked, not side by side)
    assert sc._card_input.mapTo(sc, sc._card_input.rect().topLeft()).x() == \
        sc._card_output.mapTo(sc, sc._card_output.rect().topLeft()).x()
    win.close()


def test_setup_column_layout_never_demands_more_than_its_widest_card(qapp, tmp_path):
    """The load-bearing half of section_flow's contract (see test_section_flow.py): one
    column must always be a legal answer, so Setup can be squeezed to a laptop screen."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    win.show()
    for _ in range(4):
        qapp.processEvents()
    widest = max(sc._rig.minimumSizeHint().width(), sc._leverance.minimumSizeHint().width())
    assert sc._columns_layout.minimumSize().width() <= widest + 4
    assert sc._columns_layout.minimumSize().height() == 0, (
        "the columns layout must not demand a minimum height — the scroll area, not the "
        "window, absorbs the content")
    win.close()


def test_setup_no_max_width_cap_is_imposed(qapp, tmp_path):
    """Item 1's explicit instruction: unlike the Advanced modals (clamped to the screen by
    ``screen_fit`` as a top-level DIALOG — irrelevant here, Setup is a tab, not a window),
    nothing in this screen may narrow its own or its columns' width. QLayout.maximumSize()
    is Qt's constant sentinel regardless of any real cap (self-review finding, 06-08-2026:
    an earlier version of this test compared that sentinel to itself, which can never fail)
    — the actual constraint lives on the WIDGETS, so assert none of them were ever handed an
    explicit setMaximumWidth()/setMaximumSize() below Qt's own QWIDGETSIZE_MAX default
    (16777215 — not importable from PySide6, so read off a fresh, untouched QWidget)."""
    from PySide6.QtWidgets import QWidget
    from respmech.ui.main_window import MainWindow
    qwidgetsize_max = QWidget().maximumWidth()
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    for w in (sc, sc._columns_layout.parentWidget(), sc._rig, sc._leverance,
              sc._card_input, sc._card_output):
        assert w.maximumWidth() == qwidgetsize_max, f"{w} was given an explicit width cap"
    win.close()


def test_output_checkbox_groups_wrap_with_the_grouping_intact(qapp):
    """Tables and Diagnostic figures each reflow independently rather than one long stack —
    each FlowLayout still lists exactly its own checkboxes, in order, and no checkbox leaks
    from one group into the other's flow."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    tables = {sc._tables_flow.itemAt(i).widget() for i in range(sc._tables_flow.count())}
    diagnostics = {sc._diagnostics_flow.itemAt(i).widget()
                  for i in range(sc._diagnostics_flow.count())}
    assert tables == {sc.save_average, sc.save_bbb, sc.save_processed, sc.include_ignored}
    assert diagnostics == {sc.save_pv_avg, sc.save_pv_ind, sc.save_raw_fig,
                           sc.save_trimmed_fig, sc.save_drift_fig, sc.save_emg_fig}
    assert tables.isdisjoint(diagnostics)
    win.close()


def test_channels_card_is_the_compact_sparkline_not_the_full_dialog_view(qapp, tmp_path):
    """The Setup summary must actually be showing ColumnStack's sparkline mode, not the
    full tick-labelled panels the channel-assignment dialog uses — the whole point of
    'a compact opsummering' is that Setup no longer spends hundreds of vertical pixels on
    axis chrome nobody can interact with."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc._on_inputs_changed()
    sc._apply_channel_mapping({"flow": 5, "volume": None, "poes": 7, "pgas": 8, "pdi": 9,
                               "emg": [], "entropy": []})
    stack = sc.channel_summary.stack
    assert stack is not None
    from respmech.ui.channel_summary import SUMMARY_ROW_HEIGHT
    assert stack.plots[0].minimumHeight() == SUMMARY_ROW_HEIGHT
    assert stack.plots[0].getAxis("left").isVisible() is False
    win.close()


# --------------------------------------------------------------------------- #
# B05: the empty format-readout row takes no space; Output gets its own sticky-folder key
# and a suggested sibling; and a caution when Output points at the recordings folder.
# --------------------------------------------------------------------------- #
def test_the_empty_format_readout_row_is_hidden_not_just_blank(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    assert sc.format_readout.text() == ""
    assert sc._input_form.isRowVisible(sc.format_readout) is False
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc._on_inputs_changed()
    assert sc.format_readout.text() != ""
    assert sc._input_form.isRowVisible(sc.format_readout) is True
    sc.in_files.setText("*.nope"); sc._on_inputs_changed()
    # a "no files match" warning is still SOMETHING to say -> the row stays visible
    assert sc.format_readout.text() != ""
    assert sc._input_form.isRowVisible(sc.format_readout) is True
    # and it must re-hide, not just stay stuck visible once shown once
    sc.in_folder.setText(""); sc._on_inputs_changed()
    assert sc.format_readout.text() == ""
    assert sc._input_form.isRowVisible(sc.format_readout) is False
    win.close()


def test_output_browser_remembers_its_own_folder_not_the_input_ones(qapp, tmp_path, isolated_prefs, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from respmech.ui.main_window import MainWindow
    in_dir = tmp_path / "recordings"; in_dir.mkdir()
    out_dir = tmp_path / "results"; out_dir.mkdir()
    win = MainWindow(AppState())
    sc = win.settings_screen
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(in_dir)))
    sc._browse(sc.in_folder, folder=True)
    assert isolated_prefs.last_folder("browse", ".") == str(in_dir)
    assert isolated_prefs.last_folder("browse_output", ".") == "."   # untouched by the input pick
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(out_dir)))
    sc._browse(sc.out_folder, folder=True)
    assert isolated_prefs.last_folder("browse_output", ".") == str(out_dir)
    assert isolated_prefs.last_folder("browse", ".") == str(in_dir)   # untouched by the output pick
    win.close()


def test_choosing_the_input_folder_suggests_a_sibling_output_folder(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from respmech.ui.main_window import MainWindow
    in_dir = tmp_path / "study" / "recordings"; in_dir.mkdir(parents=True)
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.out_folder.setText("")                          # the guided ('New analysis') state
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(in_dir)))
    sc._browse(sc.in_folder, folder=True)
    assert sc.out_folder.text() == str(in_dir.parent / "respmech-output")
    assert sc.state.settings.output.folder == sc.out_folder.text()   # committed via to_state
    win.close()


def test_a_manually_entered_output_folder_is_never_overwritten_by_the_suggestion(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from respmech.ui.main_window import MainWindow
    in_dir = tmp_path / "recordings"; in_dir.mkdir()
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.out_folder.setText(str(tmp_path / "chosen-by-the-user"))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(in_dir)))
    sc._browse(sc.in_folder, folder=True)
    assert sc.out_folder.text() == str(tmp_path / "chosen-by-the-user")
    win.close()


def test_output_equal_to_input_is_cautioned(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    same = tmp_path / "shared"; same.mkdir()
    write_delim(same / "a.csv", 9)    # a real, readable file: an empty folder is 'muted',
                                       # not 'warn' (ticket B07), so this caution needs one
    assert sc._output_is_input_folder() is False       # both blank
    sc.in_folder.setText(str(same)); sc.in_files.setText("*.csv")
    sc.samp_freq.setValue(1000)
    sc._on_inputs_changed()
    sc.out_folder.setText(str(same))
    # valid channels (ticket B07), else the hard blocker masks the output==input caution
    _assign(sc, flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2, 3, 4])
    assert sc._output_is_input_folder() is True
    assert "recordings folder" in sc.qc.text()
    sc.out_folder.setText(str(tmp_path / "elsewhere")); sc._on_field_changed()
    assert sc._output_is_input_folder() is False
    assert "recordings folder" not in sc.qc.text()
    win.close()


def test_narrowing_three_or_more_extensions_keeps_every_dot(qapp, tmp_path):
    """Self-review regression (05-08-2026): joining stripped extensions with ', ' and
    prepending a single leading dot only decorated the FIRST one ('.txt, xlsx' — xlsx
    silently lost its dot). Needs >=2 DROPPED extensions to exercise the join at all (a
    single dropped extension never goes through ', '.join in a way that could lose a dot)."""
    from respmech.ui.main_window import MainWindow
    for n in ("a", "b", "c"):
        write_delim(tmp_path / f"{n}.csv", 9)
    write_delim(tmp_path / "d.txt", 9, sep="\t")
    write_xlsx(tmp_path / "e.xlsx", 9)
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(tmp_path)); sc.in_files.setText("*.csv; *.txt; *.xlsx")
    sc._on_inputs_changed()
    text = sc.format_readout.text()
    assert ".txt" in text and ".xlsx" in text
    import re
    assert not re.search(r"(?<!\.)\bxlsx\b", text)   # "xlsx" never appears without its dot
    win.close()


# -- carried-over exclusions/breath-counts/noise-reference banner (ticket B06) -----------
# The workflow the ticket exists to fix: point the same analysis at a DIFFERENT recordings
# folder that shares a filename with the old one, and the old exclusions must not just
# keep applying invisibly — see core.settings.carried_over_state.

def _entry_folder(s, filename):
    e = next(x for x in s.processing.exclude_breaths if x.file == filename)
    return e.folder


def test_switching_input_folder_shows_the_carried_over_banner(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import ExcludeEntry
    s01 = tmp_path / "S01"; s01.mkdir()
    s02 = tmp_path / "S02"; s02.mkdir()
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(s01)); sc._on_inputs_changed()
    sc.state.settings.processing.exclude_breaths.append(
        ExcludeEntry(file="sample_recording.csv", breaths=[2, 3], folder=str(s01)))
    assert sc.carried_banner.isHidden()          # still in the folder it was recorded for
    sc.in_folder.setText(str(s02)); sc._on_inputs_changed()
    assert not sc.carried_banner.isHidden()
    assert "sample_recording.csv" in sc.carried_label.text()
    win.close()


def test_editing_the_file_mask_alone_does_not_reopen_a_dismissed_banner(qapp, tmp_path):
    """_on_inputs_changed also fires from the FILES-MASK field, which must not re-check
    carried-over state on every keystroke commit — only a folder value that actually
    changed can make previously-fine state start naming the wrong folder. Otherwise a
    "Keep"-dismissed banner would pop back up on the next unrelated mask edit."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import ExcludeEntry
    s01 = tmp_path / "S01"; s01.mkdir()
    s02 = tmp_path / "S02"; s02.mkdir()
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(s01)); sc._on_inputs_changed()
    sc.state.settings.processing.exclude_breaths.append(
        ExcludeEntry(file="a.csv", breaths=[1], folder=str(s01)))
    sc.in_folder.setText(str(s02)); sc._on_inputs_changed()    # a REAL folder change
    assert not sc.carried_banner.isHidden()
    sc.btn_carried_keep.click()
    assert sc.carried_banner.isHidden()
    sc.in_files.setText("*.txt"); sc._on_inputs_changed()      # mask-only edit
    assert sc.carried_banner.isHidden()           # must NOT reopen on its own
    win.close()


def test_keep_dismisses_without_touching_settings(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import ExcludeEntry
    s01 = tmp_path / "S01"; s01.mkdir()
    s02 = tmp_path / "S02"; s02.mkdir()
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(s01)); sc._on_inputs_changed()
    sc.state.settings.processing.exclude_breaths.append(
        ExcludeEntry(file="a.csv", breaths=[2, 3], folder=str(s01)))
    sc.in_folder.setText(str(s02)); sc._on_inputs_changed()
    assert not sc.carried_banner.isHidden()
    sc.btn_carried_keep.click()
    assert sc.carried_banner.isHidden()
    assert [e.file for e in sc.state.settings.processing.exclude_breaths] == ["a.csv"]
    assert _entry_folder(sc.state.settings, "a.csv") == str(s01)   # never restamped
    win.close()


def test_clear_removes_only_the_carried_entries_and_hides_the_banner(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import ExcludeEntry
    s01 = tmp_path / "S01"; s01.mkdir()
    s02 = tmp_path / "S02"; s02.mkdir()
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.in_folder.setText(str(s02)); sc._on_inputs_changed()
    sc.state.settings.processing.exclude_breaths.append(
        ExcludeEntry(file="fresh.csv", breaths=[1], folder=str(s02)))     # matches -> kept
    sc.in_folder.setText(str(s01)); sc._on_inputs_changed()
    sc.state.settings.processing.exclude_breaths.append(
        ExcludeEntry(file="stale.csv", breaths=[2], folder=str(s01)))
    sc.in_folder.setText(str(s02)); sc._on_inputs_changed()
    assert not sc.carried_banner.isHidden()
    sc.btn_carried_clear.click()
    assert sc.carried_banner.isHidden()
    assert [e.file for e in sc.state.settings.processing.exclude_breaths] == ["fresh.csv"]
    assert sc._dirty is True                       # a real edit -> unsaved changes
    win.close()


def test_opening_an_analysis_written_before_this_field_existed_shows_the_banner(qapp, tmp_path):
    """An entry with no recorded folder (folder=None — every analysis written before this
    ticket) can never be proven to match the folder it's loaded into, so it is always shown
    as carried, the cautious default the ticket explicitly chose over guessing."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import ExcludeEntry, Settings
    s01 = tmp_path / "S01"; s01.mkdir()
    win = MainWindow(AppState()); sc = win.settings_screen
    s = Settings()
    s.input.format.sampling_frequency = 1000
    s.input.folder = str(s01)
    s.processing.exclude_breaths.append(ExcludeEntry(file="a.csv", breaths=[1], folder=None))
    sc.state.settings, sc.state.settings_path = s, None
    sc.from_state()
    assert not sc.carried_banner.isHidden()
    win.close()


def test_no_input_folder_yet_never_shows_the_banner(qapp, tmp_path):
    """A fresh guided analysis (no folder chosen) must not warn about carried-over state —
    there is no current folder for anything to mismatch against yet."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.settings import ExcludeEntry
    win = MainWindow(AppState()); sc = win.settings_screen
    sc.enter_new_mode()
    sc.state.settings.processing.exclude_breaths.append(
        ExcludeEntry(file="a.csv", breaths=[1], folder=str(tmp_path / "S01")))
    sc._update_carried_banner()
    assert sc.carried_banner.isHidden()
    win.close()


def test_the_sample_analysis_never_shows_a_false_carried_over_banner(qapp):
    """Self-review finding: build_sample_settings (core/sample.py) sets input.folder AND
    the noise reference in the same function — reference_folder must be stamped to match,
    or the very first "Explore sample data" would show a false carried-over banner (an
    unrecorded reference_folder always reads as carried, by design, since it can never be
    proven current)."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    assert sc.open_sample_analysis() is True
    assert sc.carried_banner.isHidden()
    win.close()


def test_a_preview_side_edit_refreshes_a_showing_setup_banner(qapp, tmp_path):
    """Self-review finding: confirming/clearing carried state from Preview & QC (a breath
    toggle, a breath-count-overrides commit) must not leave Setup's banner showing a
    stale, already-resolved warning — pv.settings_edited is wired to
    sc._update_carried_banner precisely for this (main_window.py)."""
    from respmech.core.settings import ExcludeEntry
    from respmech.ui.main_window import MainWindow
    from _helpers import INPUT, synth_settings
    s = synth_settings(str(tmp_path))
    s.processing.exclude_breaths.append(
        ExcludeEntry(file="synth_case_A.csv", breaths=[1], folder="/a/different/folder"))
    win = MainWindow(AppState(s))
    sc, pv = win.settings_screen, win.preview_screen
    sc._update_carried_banner()
    assert not sc.carried_banner.isHidden()
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    from respmech.ui.workers import stage_mechanics_preview
    import os as _os
    pv._render_preview(stage_mechanics_preview(s, _os.path.join(INPUT, "synth_case_A.csv")))
    pv._toggle_breath(1)                       # un-excludes the only (carried) breath
    assert sc.carried_banner.isHidden(), (
        "resolving the only carried entry from Preview must refresh Setup's banner")
    win.close()


def test_a_noise_reference_pick_refreshes_a_showing_setup_banner(qapp, tmp_path):
    """Same fix, the noise-reference write path — which goes through set_noise_reference,
    not the generic pv.settings_edited signal (see its docstring)."""
    from respmech.core.settings import ExcludeEntry
    from respmech.ui.main_window import MainWindow
    from _helpers import synth_settings
    s = synth_settings(str(tmp_path))
    # any carried exclusion is enough to show the banner; the noise pick itself will match
    # the current folder and so resolve it (nothing else is carried in this settings object)
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.reference_intervals = [[0.0, 1.0]]
    s.processing.emg.noise.reference_folder = "/a/different/folder"
    win = MainWindow(AppState(s))
    sc, pv = win.settings_screen, win.preview_screen
    sc._update_carried_banner()
    assert not sc.carried_banner.isHidden()
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    pv._apply_noise_expiration()
    assert sc.carried_banner.isHidden()
    win.close()


# ---------------------------------------------------------------------------
# UI-overhaul ticket C03 — missing recordings folder, 'Duplicate for another folder…'
# ---------------------------------------------------------------------------
def test_missing_recordings_folder_warns_instead_of_reading_as_unscanned(qapp, tmp_path):
    """C03 point 6: a folder that is SET but no longer exists (renamed/moved/unmounted
    since the analysis was saved) used to read as 'muted — nothing scanned yet', the same
    message as a genuinely empty Setup — indistinguishable from a folder nobody ever
    pointed the app at. It must instead say the folder is gone, and offer a way to fix it."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    gone = tmp_path / "moved-away"
    sc.in_folder.setText(str(gone)); sc._on_inputs_changed()
    status, text = sc._qc_verdict()
    assert status == "warn"
    assert str(gone) in text
    assert "Recordings folder not found" in text
    assert sc.format_readout.property("status") == "warn"
    assert "no longer exists" in sc.format_readout.text()
    assert not sc.btn_locate_folder.isHidden()
    win.close()


def test_a_folder_that_vanishes_mid_session_clears_the_stale_channel_preview(qapp, tmp_path):
    """C03 point 6, the 'warm' case: unlike a cold open of a dead path, this folder was
    VALID a moment ago, and _channel_view_signature's cache keys on the folder STRING (not
    the filesystem), so simply re-running the normal edit pipeline would not by itself
    notice anything changed and would keep the previous load's traces on screen next to a
    dead path."""
    from respmech.ui.main_window import MainWindow
    from _helpers import synth_settings
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    sc.from_state()
    assert sc.channel_summary.texts()          # a real mapping was rendered
    moved = str(tmp_path / "moved-recordings")
    sc.in_folder.setText(moved); sc._on_inputs_changed()   # folder now points nowhere
    assert not sc._valid_input_files()
    # the summary must have been asked to re-render with no data (matrix/names None) —
    # _refresh_channel_view(force=True) is what _update_format_readout calls for this case
    assert not sc.btn_locate_folder.isHidden()
    win.close()


def test_locate_folder_button_reuses_the_ordinary_input_edit_pipeline(qapp, tmp_path, monkeypatch):
    """C03 point 6: 'Locate folder…' must behave exactly like editing the field by hand —
    reusing _on_inputs_changed rather than a second, parallel folder-setting path."""
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    gone = tmp_path / "gone"
    sc.in_folder.setText(str(gone)); sc._on_inputs_changed()
    assert not sc.btn_locate_folder.isHidden()
    real = tmp_path / "real-recordings"; real.mkdir()
    monkeypatch.setattr(ss.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(real)))
    sc._locate_missing_folder()
    assert sc.in_folder.text() == str(real)
    assert sc.state.settings.input.folder == str(real)   # _on_inputs_changed committed it
    assert sc.btn_locate_folder.isHidden()
    win.close()


def test_deepest_existing_ancestor_walks_up_to_a_real_directory(tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    real = tmp_path / "real"; real.mkdir()
    dead = real / "subject" / "S01"
    assert SettingsScreen._deepest_existing_ancestor(str(dead)) == str(real)
    assert SettingsScreen._deepest_existing_ancestor(str(real)) == str(real)
    assert SettingsScreen._deepest_existing_ancestor("") == os.path.expanduser("~")


def test_duplicate_for_another_folder_derives_output_clears_ecg_reference_and_opens_save_as(
        qapp, tmp_path, monkeypatch):
    """C03 point 5, the end-to-end flow: pick a new recordings folder, confirm the
    suggested (sibling-derived) output folder, and land on Save as… pre-filled with the
    SAME analysis filename inside the NEW folder — never overwriting the template. The
    file-keyed exclude_breaths/breath_counts/noise-reference are left to B06's own
    Behold/Ryd banner (already exercised by the tests above this section); only
    ecg_reference_file (no such ask mechanism) is asserted cleared directly here."""
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    from _helpers import synth_settings

    study = tmp_path / "Study"
    old_input = study / "S01"; old_output = study / "S01-output"
    old_input.mkdir(parents=True); old_output.mkdir()
    new_input = study / "S02"; new_input.mkdir()

    s = synth_settings(str(old_output))
    s.input.folder = str(old_input)
    s.processing.emg.ecg_reference_file = "synth_case_A.csv"
    win = MainWindow(AppState(s))
    sc = win.settings_screen
    sc.state.settings_path = str(old_input / "analysis.toml")
    sc.from_state()

    monkeypatch.setattr(ss.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(new_input)))

    from respmech.ui import duplicate_dialog as dd
    captured = {}

    class _AutoAcceptDialog:
        def __init__(self, new_in, suggested_out, parent=None):
            captured["new_input"] = new_in
            captured["suggested_output"] = suggested_out
            self._out = suggested_out
        def exec(self):
            from PySide6.QtWidgets import QDialog
            return QDialog.Accepted
        def output_folder(self):
            return self._out
    monkeypatch.setattr(dd, "DuplicateFolderDialog", _AutoAcceptDialog)

    save_as_calls = []
    monkeypatch.setattr(ss.SettingsScreen, "save_analysis_as",
                        lambda self, suggested_path=None: save_as_calls.append(suggested_path))

    sc.duplicate_for_another_folder()

    assert captured["new_input"] == str(new_input)
    assert captured["suggested_output"] == str(study / "S02-output")   # derived, not asked
    assert sc.in_folder.text() == str(new_input)
    assert sc.out_folder.text() == str(study / "S02-output")
    assert sc.state.settings.processing.emg.ecg_reference_file is None
    assert sc.is_dirty()
    assert save_as_calls == [str(new_input / "analysis.toml")]
    win.close()


def test_duplicate_dialog_asks_when_output_is_not_a_sibling(qapp, tmp_path, monkeypatch):
    """C03 point 5: when derive_sibling_output can't guess (output nested inside input),
    the dialog opens with an EMPTY suggestion rather than a wrong guess."""
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    from _helpers import synth_settings

    old_input = tmp_path / "S01"; old_input.mkdir()
    old_output = old_input / "output"; old_output.mkdir()    # nested, not a sibling
    new_input = tmp_path / "S02"; new_input.mkdir()

    s = synth_settings(str(old_output)); s.input.folder = str(old_input)
    win = MainWindow(AppState(s)); sc = win.settings_screen
    sc.from_state()
    monkeypatch.setattr(ss.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(new_input)))

    from respmech.ui import duplicate_dialog as dd
    captured = {}

    class _CancelDialog:
        def __init__(self, new_in, suggested_out, parent=None):
            captured["suggested_output"] = suggested_out
        def exec(self):
            from PySide6.QtWidgets import QDialog
            return QDialog.Rejected
    monkeypatch.setattr(dd, "DuplicateFolderDialog", _CancelDialog)

    kept_input = sc.in_folder.text()
    sc.duplicate_for_another_folder()
    assert captured["suggested_output"] == ""       # nothing guessable -> ask, don't assume
    assert sc.in_folder.text() == kept_input         # cancelled -> nothing applied
    win.close()


def test_science_note_navigation_uses_exactly_one_arrow(qapp, tmp_path):
    """Ticket D21 (UI-overhaul): a screen-navigation hint like 'Preview & QC ▸ EMG – ECG
    reduction' must carry exactly one arrow. This used to render with two ('Preview & QC ▸
    › EMG – ECG reduction') because the sentence concatenated its own '▸ ' separator with a
    sub-tab name that already carries a leading '›' internally (that leading arrow is
    correct when the name is used alone, e.g. in a tooltip naming just the sub-tab — it is
    only wrong once another arrow is glued in front of it here).

    Also asserts run_screen's _FIX_HINTS uses the SAME arrow character as this screen: the
    two used to disagree ('→' vs '▸'), describing the same kind of route in two different
    symbols."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.run_screen import _FIX_HINTS
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    s = sc.state.settings
    s.processing.emg.robust_peak.enabled = True
    s.processing.emg.remove_ecg = False
    notes = sc._science_notes()
    gated_peak_notes = [n for n in notes if "cardiac-gated peak" in n]
    assert len(gated_peak_notes) == 1
    note = gated_peak_notes[0]
    assert note.count("▸") == 1, f"expected exactly one ▸ in: {note!r}"
    assert "›" not in note, f"a sub-tab's own leading arrow leaked into the sentence: {note!r}"
    for hint in _FIX_HINTS.values():
        assert "→" not in hint, f"_FIX_HINTS still uses '→' instead of '▸': {hint!r}"
        assert "▸" in hint
    win.close()


"""Settings screen behaviours: the channel-mapping hard gate (P1), the dirty flag +
window title + guarded New (P6), the live QC caution strip (P9), the surfaced signal
transforms (P3), the output "what to save" checklist (P4) and the detected-format
read-out (P28). Previously split across test_review_wave1/2/6.py."""
import pytest

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _valid(sc, tmp):
    sc.in_folder.setText(INPUT); sc.in_files.setText("synth_case_*.csv")
    sc.samp_freq.setValue(1000); sc.out_folder.setText(str(tmp))
    sc.col_flow.setValue(5); sc.col_volume.setValue(6); sc.col_poes.setValue(7)
    sc.col_pgas.setValue(8); sc.col_pdi.setValue(9); sc.cols_emg.setText("2,3,4")
    sc._on_field_changed()


def test_required_channel_on_the_time_column_blocks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    assert sc._all_ok()                                  # a good mapping is ready
    sc.col_poes.setValue(1); sc._on_field_changed()      # poes -> column 1 (the time axis)
    assert not sc._all_ok()
    assert "column 1" in sc._channel_collision().lower()
    win.close()


def test_two_required_channels_sharing_a_column_blocks(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    _valid(sc, tmp_path)
    sc.col_pgas.setValue(7); sc._on_field_changed()      # pgas == poes column
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
    assert win.windowTitle().rstrip().endswith("•")      # dirty bullet
    sc._mark_clean()
    assert not sc.is_dirty() and not win.windowTitle().rstrip().endswith("•")
    win.close()


def test_new_analysis_only_confirms_when_dirty(qapp, monkeypatch):
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    calls = []
    monkeypatch.setattr(ss.QMessageBox, "question",
                        staticmethod(lambda *a, **k: calls.append(1) or ss.QMessageBox.Yes))
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
    sc.cols_emg.setText("5,2,3"); sc._on_field_changed()  # EMG col 5 == the flow column
    assert sc.qc.property("status") == "warn"
    assert "overlap" in sc.qc.text().lower()
    win.close()


def test_signal_transforms_surfaced_and_bound(qapp):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState()); sc = win.settings_screen
    # drift correction is ON by default — it silently changes results, so it must show it
    assert sc.correct_drift.isChecked()
    sc.correct_drift.setChecked(False)
    sc.inverse_flow.setChecked(True)
    sc.correct_trend.setChecked(True)
    sc.resample.setChecked(True); sc.resample_hz.setValue(500)
    sc._on_field_changed()
    v = sc.state.settings.processing.volume
    smp = sc.state.settings.processing.sampling
    assert v.correct_drift is False and v.inverse_flow is True and v.correct_trend is True
    assert smp.resample is True and smp.resample_to_frequency == 500
    # dependent fields only enable when their toggle is on
    assert sc.trend_method.isEnabled() and sc.resample_hz.isEnabled()
    sc.correct_trend.setChecked(False); sc.resample.setChecked(False); sc._on_field_changed()
    assert not sc.trend_method.isEnabled() and not sc.resample_hz.isEnabled()
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
    assert sc.seg_buffer.value() == 640 and sc.avg_resamp.value() == 750
    assert sc.noise_nfft.value() == 512 and sc.matlab_variant.currentData() == "windows"
    assert sc.breath_counts_edit.toPlainText() == "a,b.txt = 12"
    assert [(e.file, e.count) for e in sc.to_state().processing.breath_counts] == [("a,b.txt", 12)]
    # edit + write back
    sc.seg_buffer.setValue(900); sc.ent_epochs.setValue(4)
    sc.matlab_variant.setCurrentIndex(sc.matlab_variant.findData("mac"))
    sc.breath_counts_edit.setPlainText("x.txt = 5\ny.txt = 9")
    out = sc.to_state()
    assert out.processing.segmentation.buffer == 900 and out.processing.entropy.epochs == 4
    assert out.input.format.matlab_variant == "mac"
    assert [(e.file, e.count) for e in out.processing.breath_counts] == [("x.txt", 5), ("y.txt", 9)]
    win.close()


def test_form_fields_are_bounded_and_never_elide(qapp, tmp_path):
    """Fusion's QFormLayout grows every uncapped field to the whole form width, so these
    combos/text areas stretched the entire window (1433px at a 1700px window) while their
    spin-box siblings sat in a tidy column. Each must stay bounded AND still fit its own
    longest content — a cap that elides the text trades one bug for another.

    This also guards the stylesheet itself: a QSS syntax error makes Qt silently drop the
    WHOLE sheet, which un-caps every field here.
    """
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path))))
    sc = win.settings_screen
    win.resize(1700, 950); win.show(); qapp.processEvents()
    column = sc.seg_buffer.width()                       # the QSS-capped numeric column
    for w in (sc.seg_method, sc.wob_from, sc.trend_method, sc.matlab_variant,
              sc.emg_norm, sc.breath_counts_edit):
        assert w.width() < win.width() / 3, f"{w} stretched to {w.width()}"
        assert w.width() >= w.sizeHint().width(), f"{w} elides its own text"
    # word-length choices share the spin column exactly; sentence-length ones get more room
    for w in (sc.seg_method, sc.wob_from, sc.trend_method, sc.matlab_variant):
        assert w.width() == column
    for w in (sc.emg_norm, sc.breath_counts_edit):
        assert column < w.width() <= 2.5 * column
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

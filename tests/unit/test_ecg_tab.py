"""The EMG–ECG reduction tab: reactive scoping of the 'ecg' kind, the capture-channel
middle-default seed, the render (flow background + R-peak capture markers on the raw capture
channel AND every processed channel), and the Auto-suggest button. UI-only — goldens untouched."""
import os

import numpy as np

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _win(s):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    return MainWindow(AppState(s))


def test_ecg_settings_paths_recompute_the_ecg_kind():
    from respmech.ui.screens.preview_screen import _kinds_for_settings_path
    for p in ("processing.emg.detect_channel", "processing.emg.remove_ecg",
              "processing.emg.ecg_min_height", "processing.emg.ecg_window_s"):
        assert "ecg" in _kinds_for_settings_path(p)          # tab recomputes on an ECG edit
        assert "emg_all" in _kinds_for_settings_path(p)      # ...and the downstream EMG panels
    assert "ecg" in _kinds_for_settings_path("input.channels.emg")
    assert "ecg" not in _kinds_for_settings_path("processing.wob.calc_from")   # unrelated


def test_capture_channel_defaults_to_middle_when_unconfigured(qapp, tmp_path):
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)   # 3 EMG cols, ECG unset
    pv = _win(s).preview_screen
    assert pv.ecg_capture_channel.currentIndex() == 1        # middle of 3
    assert s.processing.emg.detect_channel == 1              # seeded into the model
    # a configured analysis (removal already ON) keeps its stored channel (0), not the seed
    s2 = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    assert _win(s2).preview_screen.ecg_capture_channel.currentIndex() == 0


def test_ecg_render_draws_flow_and_capture_on_every_channel(qapp, tmp_path):
    from respmech.ui.plot_overlays import _FLOW_Z, _CAPTURE_Z
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    n, fs = 4000, 2000
    t = np.arange(n) / fs
    data = {"t": t, "fs": fs, "cols": [2, 3, 4], "detect": 1, "detect_col": 3,
            "raw_capture": np.sin(t), "processed": [np.sin(t), np.cos(t), np.sin(2 * t)],
            "peaks": np.array([0.5, 1.0, 1.5, 2.0]), "flow": np.sin(2 * np.pi * 0.3 * t),
            "ecg_applied": True, "ecg_error": None}
    pv._on_ecg_result(data)
    cap_z = [it.zValue() for it in pv.ecg_capture_plot.getPlotItem().listDataItems()]
    assert _FLOW_Z in cap_z and _CAPTURE_Z in cap_z          # flow behind + capture line/triangle
    assert len(pv._ecg_capture_subplots) == 3
    for p in pv._ecg_capture_subplots:                       # every processed channel too
        pz = [it.zValue() for it in p.listDataItems()]
        assert _FLOW_Z in pz and _CAPTURE_Z in pz
    # empty peaks -> no capture markers, but the trace + flow still render
    data2 = {**data, "peaks": np.array([])}
    pv._on_ecg_result(data2)
    assert _CAPTURE_Z not in [it.zValue() for it in pv.ecg_capture_plot.getPlotItem().listDataItems()]


def test_noise_strip_stays_compact_across_tab_switches(qapp, tmp_path):
    """Regression: with three sub-tabs the noise-strip aligner fires on every tab switch,
    including while the noise tab is not laid out. Keyed off the chip's live height it fed
    the button's min-height back into the strip row and let the chip balloon a little more
    each switch — once to ~247 px, ~55% of the tab, crushing the plots into a bottom band.
    Now keyed off sizeHint + a Maximum vertical policy, so it stays compact and idempotent."""
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    win = _win(s); pv = win.preview_screen
    win.resize(1400, 900); win.show(); qapp.processEvents()
    for _ in range(3):                                    # exercise the aligner repeatedly
        for i in range(pv.subtabs.count()):
            pv.subtabs.setCurrentIndex(i); qapp.processEvents()
    idx = pv.subtabs.indexOf(pv._emg_tab); pv.subtabs.setCurrentIndex(idx)
    for _ in range(4):
        qapp.processEvents()
    assert pv.btn_set_noise.minimumHeight() < 100         # compact — not the ~247 runaway
    assert pv.noise_opts.height() < 100
    win.close()


def test_ecg_auto_batch_checkbox_reflects_and_writes_settings(qapp, tmp_path):
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    s.processing.emg.ecg_auto_detect = True
    pv = _win(s).preview_screen
    assert pv.ecg_auto_batch.isChecked() is True                # loaded from settings
    pv.ecg_auto_batch.setChecked(False)
    assert s.processing.emg.ecg_auto_detect is False             # written straight back
    pv.ecg_auto_batch.setChecked(True)
    assert s.processing.emg.ecg_auto_detect is True


def test_ecg_auto_batch_greys_out_the_fields_it_overrides(qapp, tmp_path):
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    manual = (pv.ecg_capture_channel, pv.ecg_min_height, pv.ecg_min_distance, pv.btn_ecg_advanced)
    assert all(w.isEnabled() for w in manual)                    # auto off -> editable
    pv.ecg_auto_batch.setChecked(True)
    assert all(not w.isEnabled() for w in manual)                # auto on -> greyed out
    pv.ecg_auto_batch.setChecked(False)
    assert all(w.isEnabled() for w in manual)                    # back off -> editable again


def test_ecg_auto_batch_blanks_the_stale_numeric_fields(qapp, tmp_path):
    """Ticket 20260804-0923: greyed out is not enough — the min height/min gap fields kept
    showing the last manual/Auto-suggest numbers even though a real run ignores them
    entirely and auto-detects its own from the reference file. They must show a dash
    while auto-batch governs them, and the REAL number again the moment it is unticked —
    without the underlying setting ever changing because of the display switch."""
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    s.processing.emg.ecg_min_height = 0.2309
    s.processing.emg.ecg_min_distance_s = 0.405        # 3 decimals — the field's own precision
    pv = _win(s).preview_screen
    fields = (pv.ecg_min_height, pv.ecg_min_distance)
    for w in fields:
        assert w.lineEdit().text() != "—"
    pv.ecg_auto_batch.setChecked(True)
    for w in fields:
        assert w.lineEdit().text() == "—"
    # blanking is display-only: the model must still hold the real numbers
    assert abs(s.processing.emg.ecg_min_height - 0.2309) < 1e-9
    assert abs(s.processing.emg.ecg_min_distance_s - 0.405) < 1e-9
    pv.ecg_auto_batch.setChecked(False)
    for w in fields:
        assert w.lineEdit().text() != "—"
    assert abs(pv.ecg_min_height.value() - 0.2309) < 1e-9
    assert abs(pv.ecg_min_distance.value() - 0.405) < 1e-9


def test_ecg_auto_batch_caption_survives_a_later_status_overwrite(qapp, tmp_path):
    """The one sentence explaining 'these values are stale, the run auto-detects its own'
    used to live only in the shared status line, where an EMG or noise job finishing a
    moment later silently erased it. It must now live in the ECG strip's own persistent
    caption instead, which nothing else on the tab can overwrite."""
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    assert pv.ecg_caption.fullText() == ""                       # nothing to say yet
    pv.ecg_auto_batch.setChecked(True)
    caption = pv.ecg_caption.fullText()
    assert "Auto-detect for the batch" in caption
    assert "first matched file" in caption                       # no reference file set
    # ticket D21 (UI-overhaul): the sentence must say ONE parameter set for the WHOLE
    # batch, never "for every file" in the sense of a per-file re-detection — an earlier
    # status-line wording ("auto-detects its own parameters ... for every file") read as
    # the opposite of what the pipeline actually does (one set, derived once, applied to
    # all). Checked on fullText(), never the elided text(), same as elsewhere in this file.
    assert "applied to every file" in caption
    # simulate an EMG/noise job finishing a moment later and overwriting the status line
    pv._set_status("EMG result: conditioned (ECG + noise).")
    assert pv.ecg_caption.fullText() == caption                  # untouched
    assert pv.status.text() != caption
    pv.ecg_auto_batch.setChecked(False)
    assert pv.ecg_caption.fullText() == ""                       # cleared once auto is off


def test_ecg_auto_batch_caption_names_the_reference_file(qapp, tmp_path):
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    s.processing.emg.ecg_reference_file = "S03.csv"
    pv = _win(s).preview_screen
    pv.ecg_auto_batch.setChecked(True)
    assert "'S03.csv'" in pv.ecg_caption.fullText()


def test_ecg_auto_batch_disabled_without_remove_ecg(qapp, tmp_path):
    # Mirrors Settings.validate()'s requirement: auto-detect needs Remove ECG on, so the
    # checkbox that would enable it is itself disabled until Remove ECG is ticked.
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    assert pv.ecg_auto_batch.isEnabled() is False
    pv.remove_ecg.setChecked(True)
    assert pv.ecg_auto_batch.isEnabled() is True
    pv.remove_ecg.setChecked(False)
    assert pv.ecg_auto_batch.isEnabled() is False


def test_unchecking_remove_ecg_clears_stuck_auto_batch(qapp, tmp_path):
    # Regression: turning Remove ECG off while Auto (whole batch) is on used to leave
    # ecg_auto_detect=True stuck (an invalid combination Settings.validate rejects) with the
    # checkbox itself disabled-but-checked and no way to untick it short of re-enabling
    # Remove ECG first.
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    pv.ecg_auto_batch.setChecked(True)
    assert s.processing.emg.ecg_auto_detect is True
    pv.remove_ecg.setChecked(False)
    assert s.processing.emg.remove_ecg is False
    assert s.processing.emg.ecg_auto_detect is False         # cleared, not left stuck
    assert pv.ecg_auto_batch.isChecked() is False             # widget reflects the clear
    s.validate()                                              # the resulting settings are valid
    # ticket 20260804-0923: the forced uncheck must also un-blank the fields and clear the
    # caption — a stuck-open dash or a stale "Auto (whole batch) is on" sentence here would
    # be its own, quieter version of the bug this ticket fixes.
    assert pv.ecg_min_height.lineEdit().text() != "—"
    assert pv.ecg_min_distance.lineEdit().text() != "—"
    assert pv.ecg_caption.fullText() == ""


def test_ecg_auto_batch_also_gates_autosuggest(qapp, tmp_path):
    # Regression: Auto-suggest writes the same 5 fields the batch auto-detect will overwrite
    # anyway, so it must grey out alongside them — otherwise a click looks like it did
    # something a real run silently discards.
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    assert pv.btn_ecg_autosuggest.isEnabled() is True
    pv.ecg_auto_batch.setChecked(True)
    assert pv.btn_ecg_autosuggest.isEnabled() is False
    assert pv.btn_ecg_autosuggest.toolTip() != ""              # explains why, not just blank
    pv.ecg_auto_batch.setChecked(False)
    assert pv.btn_ecg_autosuggest.isEnabled() is True
    assert "clearest ECG" in pv.btn_ecg_autosuggest.toolTip()  # original help text restored


def test_autosuggest_writes_settings_and_selects_channel(qapp, tmp_path, monkeypatch):
    from respmech.core import emg as emglib
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)
    win = _win(s); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_index(0)
    monkeypatch.setattr(emglib, "suggest_ecg_settings", lambda m, fs: {
        "detect_channel": 2, "ecg_min_height": 0.02, "ecg_min_distance_s": 0.45,
        "ecg_min_width_s": 0.005, "ecg_window_s": 0.4,
        "_diagnostics": {"confidence": "high", "est_bpm": 70.0}})
    pv._on_ecg_autosuggest()
    e = s.processing.emg
    assert e.detect_channel == 2 and abs(e.ecg_min_height - 0.02) < 1e-9
    assert abs(e.ecg_min_distance_s - 0.45) < 1e-9 and abs(e.ecg_window_s - 0.4) < 1e-9
    assert pv.ecg_capture_channel.currentIndex() == 2        # reflected into the widget
    win.close()


def test_loading_an_analysis_with_a_stuck_auto_batch_repairs_it(qapp, tmp_path):
    """Loading is the path the interactive guard above does not cover.

    Emil's S07.toml, 30-07-2026: it carried ``ecg_auto_detect = true`` with
    ``remove_ecg = false``. Measured before this fix — the checkbox came up ticked AND
    disabled, ``_settings_ok`` reported the analysis invalid so every preview was gated out
    with "Settings incomplete", the mechanics test run died on ``SettingsError:
    processing.emg.ecg_auto_detect requires processing.emg.remove_ecg to be enabled`` behind
    a raw traceback, and there was no way to untick the box without re-enabling Remove ECG
    first. The interactive path had a guard (test above); the load path had none.

    Repairing is the right way round here: remove_ecg is the switch the user sees, and
    auto-detect is a sub-option of it. The CLI still raises on such a file, deliberately —
    a hand-written config with that pair is a mistake worth reporting, not silently altering.
    """
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)
    s.processing.emg.ecg_auto_detect = True          # the invalid pair, as stored in the file
    s.processing.emg.remove_ecg = False

    pv = _win(s).preview_screen

    assert s.processing.emg.ecg_auto_detect is False, "the stored pair must be repaired, not just the widget"
    assert pv.ecg_auto_batch.isChecked() is False
    assert pv.ecg_auto_batch.isEnabled() is False     # still gated on Remove ECG
    # And the analysis must now be runnable at all, which is the whole point.
    ok, why = pv._settings_ok()
    assert ok is True, why
    s.validate()                                      # would raise before the fix


def test_the_repair_is_announced_and_marks_the_analysis_dirty(qapp, tmp_path):
    """A silent repair would change what the file asked for without saying so."""
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)
    s.processing.emg.ecg_auto_detect = True
    pv = _win(s).preview_screen

    seen = []
    pv.settings_edited.connect(lambda *_: seen.append(True))
    pv._announce_ecg_auto_repair()                    # what the deferred timer calls
    assert seen, "the repair changes the analysis, so it must mark it dirty"
    # ...and only once: a second call must not re-announce or re-dirty
    seen.clear()
    pv._announce_ecg_auto_repair()
    assert not seen


def test_a_consistent_analysis_is_left_alone(qapp, tmp_path):
    """The repair must not touch a legitimately enabled auto-detect."""
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    s.processing.emg.ecg_auto_detect = True
    pv = _win(s).preview_screen
    assert s.processing.emg.ecg_auto_detect is True
    assert pv.ecg_auto_batch.isChecked() is True
    assert pv._ecg_auto_repaired is False


def test_the_processed_panel_title_reflects_whether_removal_is_actually_on(qapp, tmp_path):
    """Ticket 20260804-0922: the panel over the processed EMG stack used to say
    'ECG-processed EMG channels' even with Remove ECG off, while it showed the RAW
    channels underneath (R-takker in full swing) -- reading as 'the removal is broken'
    rather than 'the removal is off'. It must now name which is true, and carry the
    peak-window-RMS suppression when removal is actually on. Asserted on fullText()
    (the ElidingLabel's untruncated string), never a rendered/elided text()."""
    from respmech.ui.screens.preview._ecg import _ECG_PROCESSED_TITLE_ON, _ECG_PROCESSED_TITLE_OFF
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    n, fs = 4000, 2000
    t = np.arange(n) / fs
    base = {"t": t, "fs": fs, "cols": [2, 3, 4], "detect": 1, "detect_col": 3,
            "raw_capture": np.sin(t), "processed": [np.sin(t), np.cos(t), np.sin(2 * t)],
            "peaks": np.array([0.5, 1.0, 1.5, 2.0]), "flow": np.sin(2 * np.pi * 0.3 * t),
            "ecg_error": None}

    pv._on_ecg_result({**base, "ecg_applied": True, "suppression": 0.78})
    title_on = pv._ecg_processed_box._title_label.fullText()
    assert title_on.startswith(_ECG_PROCESSED_TITLE_ON)
    assert "78%" in title_on                    # the suppression number reaches the title band
    assert "78%" in pv.status.text()             # ...and the status line

    pv._on_ecg_result({**base, "ecg_applied": False, "suppression": None})
    title_off = pv._ecg_processed_box._title_label.fullText()
    assert title_off == _ECG_PROCESSED_TITLE_OFF
    assert "%" not in title_off                  # nothing was suppressed -- no number to show
    assert "ECG-processed" not in title_off      # must not claim removal happened
    assert "%" not in pv.status.text()

    # Blanking the panel (file switch) resets it to the CURRENT remove_ecg setting's base
    # title (here: ON, since the settings object has remove_ecg=True) -- see the sibling
    # test below for the off-setting case. The next _on_ecg_result corrects it either way.
    pv._set_ecg_processed_title()
    assert pv._ecg_processed_box._title_label.fullText() == _ECG_PROCESSED_TITLE_ON


def test_the_processed_panel_title_reset_reflects_the_current_setting_not_a_hardcoded_on(qapp, tmp_path):
    """Ticket 20260804-0922 follow-up (self-review finding): the capture panel's reset
    text is neutral ("Raw capture channel..."), but the processed panel's ON title is
    itself a claim -- so a reset that always hard-codes ON would show "ECG-processed EMG
    channels" on a freshly blanked panel even when remove_ecg is off, exactly the
    misleading verdict this ticket exists to fix, just a beat early (e.g. while the
    settings-incomplete gate in screen.py keeps _on_ecg_result from firing at all)."""
    from respmech.ui.screens.preview._ecg import _ECG_PROCESSED_TITLE_ON, _ECG_PROCESSED_TITLE_OFF
    s = synth_settings(str(tmp_path), remove_ecg=False, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    pv._set_ecg_processed_title()
    assert pv._ecg_processed_box._title_label.fullText() == _ECG_PROCESSED_TITLE_OFF

    s.processing.emg.remove_ecg = True
    pv._set_ecg_processed_title()
    assert pv._ecg_processed_box._title_label.fullText() == _ECG_PROCESSED_TITLE_ON


def test_a_failed_detection_never_claims_ecg_processed_over_the_raw_fallback(qapp, tmp_path):
    """Ticket 20260804-0922 follow-up (self-review finding): stage_ecg_reduction falls
    back to the RAW channels whenever detection/removal raises, regardless of the
    remove_ecg setting (data['ecg_applied'] stays True). The error-branch title must
    therefore say OFF, or it reproduces the exact bug this ticket exists to fix --
    'ECG-processed' printed over a stack that is, right now, unprocessed."""
    from respmech.ui.screens.preview._ecg import _ECG_PROCESSED_TITLE_OFF
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    n, fs = 2000, 2000
    t = np.arange(n) / fs
    data = {"t": t, "fs": fs, "cols": [2, 3, 4], "detect": 1, "detect_col": 3,
            "raw_capture": np.sin(t), "processed": [np.sin(t), np.cos(t), np.sin(2 * t)],
            "peaks": np.array([]), "flow": np.sin(2 * np.pi * 0.3 * t),
            "ecg_applied": True, "ecg_error": "boom", "suppression": None}
    pv._on_ecg_result(data)
    title = pv._ecg_processed_box._title_label.fullText()
    assert title.startswith(_ECG_PROCESSED_TITLE_OFF)
    assert "ECG-processed" not in title


def test_emg_all_result_label_composes_ecg_and_noise_independently(qapp, tmp_path):
    """Ticket 20260804-0922: the EMG result status line hard-coded '(ECG + noise)'
    whenever noise_applied was true, regardless of whether ECG removal actually ran, so
    turning ECG removal off while noise reduction stayed on kept claiming ECG removal
    had happened. All four combinations must be named independently and correctly."""
    s = synth_settings(str(tmp_path), remove_ecg=True, data_out=_DATA_OUT)
    pv = _win(s).preview_screen
    n, fs = 200, 1000
    t = np.arange(n) / fs
    base = {"t": t, "fs": fs, "cols": [2, 3, 4],
            "raw": [np.zeros(n)] * 3, "conditioned": [np.zeros(n)] * 3,
            "flow": np.zeros(n), "ecg_error": None, "noise_error": None}

    cases = [
        (True, True, "conditioned (ECG + noise)"),
        (True, False, "ECG-removed"),
        (False, True, "noise-reduced"),
        (False, False, "raw"),
    ]
    for ecg_applied, noise_applied, expected in cases:
        pv._on_emg_all_result({**base, "ecg_applied": ecg_applied, "noise_applied": noise_applied})
        assert f"EMG result: {expected}." in pv.status.text(), (ecg_applied, noise_applied, pv.status.text())

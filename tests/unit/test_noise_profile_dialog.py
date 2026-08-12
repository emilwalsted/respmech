"""Tests for the modal noise-profile picker (NoiseProfileDialog).

The mouse interaction itself needs a real event loop; the selection STATE
(_set_selection / _clear_selection / _maybe_warn / selected_region) is factored out
of the handlers so it is exercised headless here.
"""
import numpy as np  # qapp fixture comes from conftest
import pytest


def _data(nch=3, n=2000, fs=1000):
    t = np.arange(n, dtype=float) / fs
    raw = [np.sin(2 * np.pi * 80 * t) * (i + 1) for i in range(nch)]
    return raw, t, fs, [2, 3, 4][:nch]


def test_dialog_constructs_with_a_plot_per_channel(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    assert len(dlg._plots) == 3 and len(dlg._vlines) == 3 and len(dlg._regions) == 3
    assert dlg.btn_ok.text() == "Set noise profile" and dlg.btn_ok.isEnabled() is False
    assert dlg.btn_cancel.text() == "Cancel"      # the app is English throughout
    assert dlg.selected_region() is None
    assert dlg.isModal() is True
    dlg.deleteLater()


def test_selection_enables_ok_and_marks_every_channel(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg._set_selection(0.20, 1.00)
    assert dlg.selected_region() == (0.20, 1.00)
    assert dlg.btn_ok.isEnabled() is True
    assert all(r.isVisible() for r in dlg._regions)      # shaded on every channel
    # 0.80 s at 1000 Hz -> 9 STFT frames >= 8 -> stable, so no warning (D09: the
    # old fixed-width rule was silent at 0.30 s too, but 0.30 s is only 1 frame
    # and deserved the warning it now gets)
    assert dlg.warn.isHidden() is True
    # a plain click clears it (afværge)
    dlg._clear_selection()
    assert dlg.selected_region() is None and dlg.btn_ok.isEnabled() is False
    assert not any(r.isVisible() for r in dlg._regions)
    dlg.deleteLater()


def test_short_selection_warns_in_frames_and_says_how_far_to_drag(qapp):
    """D09: the warning is driven by the SAME frame arithmetic as the tab, live
    while dragging, and gives the target in seconds. At 1000 Hz with the default
    256/64 STFT, stability needs 0.704 s — so the ticket's 0.45 s must warn (it
    yields 4 frames) and 0.71 s must not. The old fixed-width rule pointed the
    exact opposite way on both."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    from respmech.ui.stft_frames import stft_frame_count

    raw, t, fs, cols = _data(n=3000, fs=1000)
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg._set_selection(1.00, 1.45)                       # 0.45 s -> 4 frames
    assert dlg.warn.isHidden() is False
    txt = dlg.warn.text()
    # the number shown is the number the tab computes for the same span — one statement
    frames = stft_frame_count(int(round(0.45 * fs)), 256, 64)
    assert str(frames) in txt and "frame" in txt.lower()
    assert "0.70" in txt, f"the warning must give the target in seconds, got: {txt}"
    assert "processing time" not in txt.lower(), (
        "the compute-time wording is gone: the reference's width never drove it")
    # dragging past the threshold turns the label neutral, live
    dlg._set_selection(1.00, 2.71)                       # 1.71 s -> well past 0.704 s
    assert dlg.warn.isHidden() is True
    dlg.deleteLater()


def test_no_warning_for_the_tickets_real_2000_hz_analysis(qapp):
    """The counterexample that proved the fixed rule frequency-blind: 0.4797 s at
    2000 Hz yields 11 frames and is fine, though it is narrower than the old 0.5 s
    line ever allowed for 1000 Hz recordings."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog

    raw, t, fs, cols = _data(n=16000, fs=2000)
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg._set_selection(5.1799, 5.6595)
    assert dlg.warn.isHidden() is True
    dlg.deleteLater()


def test_win_and_hop_are_threaded_not_assumed(qapp):
    """The dialog cannot know the frequency-dependent threshold without the STFT
    geometry; hardcoding it would quietly assume 1000 Hz. With a coarser hop the
    same span yields fewer frames, so the SAME 1.0 s that is fine at 256/64 must
    warn at 512/256."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog

    raw, t, fs, cols = _data(n=4000, fs=1000)
    fin = NoiseProfileDialog(raw, t, fs, cols)                    # defaults 256/64
    fin._set_selection(1.00, 2.00)                                # 1.0 s -> 12 frames
    assert fin.warn.isHidden() is True
    grov = NoiseProfileDialog(raw, t, fs, cols,
                              win_length=512, hop_length=256)
    grov._set_selection(1.00, 2.00)                               # 1.0 s -> 2 frames
    assert grov.warn.isHidden() is False
    assert "2" in grov.warn.text()
    fin.deleteLater(); grov.deleteLater()


def test_reversed_drag_is_normalised(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg._set_selection(0.8, 0.3)                          # dragged right-to-left
    assert dlg.selected_region() == (0.3, 0.8)
    dlg.deleteLater()


def test_dialog_opens_showing_the_whole_recording(qapp):
    """The first view is the full recording with the pan bar disabled. pyqtgraph's padded
    first auto-range gets clamped against the xMax limit and, via the x-link chain, locks
    a rightward drift in — the dialog then opened with the start cropped off-screen and
    the scrollbar already at max. The x view is therefore explicit, never auto-ranged."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    for nch in (1, 2, 3, 5):
        raw, t, fs, cols = _data(nch=nch, n=4000)
        dlg = NoiseProfileDialog(raw, t, fs, cols)
        dlg.resize(940, 580); dlg.show(); qapp.processEvents()
        lo, hi = dlg._plots[0].getViewBox().viewRange()[0]
        assert lo <= float(t[0]) + 1e-6, f"nch={nch}: start cropped, view begins at {lo:.3f}"
        assert hi >= float(t[-1]) - 1e-6, f"nch={nch}: end cropped at {hi:.3f}"
        assert not dlg.scroll.isEnabled()                 # nothing hidden -> nothing to pan
        assert dlg.scroll.value() == 0
        dlg.close(); dlg.deleteLater()


def test_zoom_cannot_escape_the_dataset(qapp):
    """The view is bounded to the recording. The limit must be on EVERY ViewBox, not just
    the first: the plots are x-linked in a chain, so a wheel over channel 3 originates the
    range change at ITS ViewBox — an unlimited one scales freely and only the far end of
    the chain clamps, which desyncs the stack instead of stopping the zoom."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg.resize(900, 500); dlg.show(); qapp.processEvents()
    t0, t1 = float(t[0]), float(t[-1])
    for i in range(len(dlg._plots)):                  # drive it from each channel in turn
        dlg._plots[i].getViewBox().setXRange(-50, 50, padding=0)
        qapp.processEvents()
        lo, hi = dlg._plots[0].getViewBox().viewRange()[0]
        assert lo >= t0 - 1e-6 and hi <= t1 + 1e-6, f"escaped via channel {i}: {lo}..{hi}"
        spans = {tuple(round(v, 3) for v in p.getViewBox().viewRange()[0]) for p in dlg._plots}
        assert len(spans) == 1, f"channels desynced via channel {i}: {spans}"
    dlg.close(); dlg.deleteLater()


def test_scrollbar_pans_every_channel_and_keeps_the_zoom(qapp):
    """Left-drag belongs to the region picker, so the scrollbar is the only way to move
    through a zoomed recording. It pans all channels (they are x-linked) without changing
    the zoom width, and is disabled while the whole recording already fits."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg.resize(900, 500); dlg.show(); qapp.processEvents()
    t0, t1 = float(t[0]), float(t[-1])
    dlg._plots[0].getViewBox().setXRange(t0, t1, padding=0); qapp.processEvents()
    assert not dlg.scroll.isEnabled()                 # nothing hidden -> nothing to scroll
    dlg._plots[0].getViewBox().setXRange(0.40, 0.60, padding=0); qapp.processEvents()
    assert dlg.scroll.isEnabled()
    assert dlg.scroll.maximum() == pytest.approx(round((t1 - t0 - 0.20) * 1000), abs=2)
    dlg.scroll.setValue(1000)                         # drag to t = 1.000 s
    qapp.processEvents()
    lo, hi = dlg._plots[0].getViewBox().viewRange()[0]
    assert lo == pytest.approx(1.0, abs=1e-3) and (hi - lo) == pytest.approx(0.20, abs=1e-3)
    spans = {tuple(round(v, 3) for v in p.getViewBox().viewRange()[0]) for p in dlg._plots}
    assert len(spans) == 1                            # every channel followed
    assert dlg.scroll.value() == 1000                 # the two-way wiring did not oscillate
    dlg.close(); dlg.deleteLater()


def test_double_click_resets_zoom_but_keeps_the_marked_region(qapp):
    """Qt delivers a double-click as Press-Release-DblClick-Release; the leading Release runs
    the click-to-clear path. Resetting the zoom must NOT cost the user the rest region they
    marked (and must leave 'Set noise profile' enabled), so the double-click restores it."""
    from PySide6.QtCore import QPointF, QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg.resize(800, 400); dlg.show(); qapp.processEvents()

    dlg._set_selection(0.20, 0.50)                       # user marks a rest region
    dlg._plots[0].getViewBox().setXRange(0.30, 0.45, padding=0)   # ...then zooms in
    dlg._zoomed = True

    vp = dlg.glw.viewport()
    pt = QPointF(vp.width() / 2, vp.height() / 2)        # one stationary point for the whole gesture
    seq = (QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
           QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease)
    for etype in seq:
        buttons = Qt.LeftButton if etype in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick) else Qt.NoButton
        dlg.eventFilter(vp, QMouseEvent(etype, pt, Qt.LeftButton, buttons, Qt.NoModifier))

    assert dlg.selected_region() == (0.20, 0.50)         # region survived the double-click
    assert dlg.btn_ok.isEnabled() is True                # ...and OK is still usable
    lo, hi = dlg._plots[0].getViewBox().viewRange()[0]   # ...and the zoom is back to the whole recording
    assert lo <= float(t[0]) + 1e-6 and hi >= float(t[-1]) - 1e-6
    dlg.close(); dlg.deleteLater()


def test_click_with_jitter_clears_but_a_real_drag_selects(qapp):
    """A 'click to dismiss' must survive a pixel of pointer jitter (pixel-space test),
    while a genuine drag past the drag threshold marks a region."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg.resize(800, 400); dlg.show(); qapp.processEvents()
    dlg._set_selection(0.4, 0.7)                          # a real selection exists
    # a click with 1 px of jitter -> NOT a drag -> clears the selection
    dlg._dragging = True; dlg._moved = False
    dlg._press_px = QPoint(400, 150); dlg._press_x = 0.5
    dlg._on_move(QPoint(401, 150))
    assert dlg._moved is False
    dlg._on_release(QPoint(401, 150))
    assert dlg.selected_region() is None
    # a genuine drag past the threshold -> marks a region
    far = 400 + QApplication.startDragDistance() + 8
    dlg._dragging = True; dlg._moved = False
    dlg._press_px = QPoint(400, 150); dlg._press_x = 0.5
    dlg._on_move(QPoint(far, 150))
    assert dlg._moved is True
    dlg._on_release(QPoint(far, 150))
    assert dlg.selected_region() is not None
    dlg.close(); dlg.deleteLater()


# -- the reference is defined in ONE place ------------------------------------------
# Regression: the two ways to define it — a marked span, or every expiration — were split
# across two screens. This dialog silently unticked the Setup checkbox whenever a span was
# marked, and re-ticking it silently made the marked span inert. Neither screen showed what
# the other had done, and the core resolves them as `use_expiration or not intervals`, so
# whichever the user looked at last was not necessarily what would run.

def test_the_whole_expiration_option_is_offered_here(qapp):
    from respmech.ui.noise_profile_dialog import EXPIRATION, NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    assert dlg.selected_region() is None                  # nothing chosen yet
    dlg.use_expiration.setChecked(True)
    assert dlg.selected_region() is EXPIRATION
    assert dlg.btn_ok.isEnabled(), "a valid choice must be acceptable"


def test_the_two_options_visibly_retire_each_other(qapp):
    from respmech.ui.noise_profile_dialog import EXPIRATION, NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg._set_selection(1.0, 2.0)
    assert dlg.selected_region() == (1.0, 2.0)
    dlg.use_expiration.setChecked(True)
    assert dlg.selected_region() is EXPIRATION            # expiration takes over
    assert not dlg.glw.isEnabled(), "the span could still be edited while inert"
    dlg.use_expiration.setChecked(False)
    assert dlg.selected_region() == (1.0, 2.0)            # ...and the span comes back
    assert dlg.glw.isEnabled()


def test_marking_a_span_while_expiration_is_chosen_is_not_possible(qapp):
    """The plot is disabled, so the user cannot leave the dialog believing a span they
    dragged will be used when the core would ignore it."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg.use_expiration.setChecked(True)
    assert not dlg.glw.isEnabled()


# -- D07 (UI-overhaul): the picker shows the reference already in force -------------
# Regression: _open_noise_profile_dialog never called _set_selection with the saved
# reference, so re-opening the picker on a re-visited analysis showed an empty picker —
# a marked span could be inspected, confirmed, or fine-tuned only by dragging a brand
# new one over it, i.e. by overwriting it to look at it.

def test_seeding_shows_the_existing_reference_and_enables_ok(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data(n=3000)
    dlg = NoiseProfileDialog(raw, t, fs, cols, file_name="synth_case_B.csv",
                             reference_file="synth_case_B.csv")
    assert dlg.selected_region() is None                   # nothing seeded until asked
    dlg._seed_reference(12.00, 12.80)
    assert dlg.selected_region() == (12.00, 12.80)
    assert dlg.btn_ok.isEnabled() is True                   # "keep as is" is a valid accept
    assert all(r.isVisible() for r in dlg._regions)
    assert "Current reference" in dlg.info.text()
    assert "synth_case_B.csv" in dlg.info.text()
    assert "12.00" in dlg.info.text() and "12.80" in dlg.info.text()
    dlg.deleteLater()


def test_a_fresh_drag_after_seeding_replaces_the_seeded_wording(qapp):
    """Once the user actually marks something new, the info line must read as a NEW pick,
    not still claim to describe the saved reference."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data(n=3000)
    dlg = NoiseProfileDialog(raw, t, fs, cols, file_name="synth_case_A.csv",
                             reference_file="synth_case_A.csv")
    dlg._seed_reference(1.0, 2.0)
    assert "Current reference" in dlg.info.text()
    dlg._set_selection(3.0, 4.0)                            # a real drag would call this
    assert dlg.selected_region() == (3.0, 4.0)
    assert "Current reference" not in dlg.info.text()
    assert "Rest region" in dlg.info.text()
    dlg.deleteLater()


def test_seeding_does_not_affect_the_whole_expiration_option(qapp):
    """'every expiration' must keep behaving exactly as before seeding was added."""
    from respmech.ui.noise_profile_dialog import EXPIRATION, NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols, file_name="synth_case_A.csv",
                             reference_file="synth_case_A.csv")
    dlg.use_expiration.setChecked(True)
    assert dlg.selected_region() is EXPIRATION
    assert dlg.btn_ok.isEnabled()
    assert not dlg.glw.isEnabled()
    dlg.deleteLater()


def test_a_different_reference_file_warns_and_relabels_ok(qapp):
    """Accepting here would move the WHOLE test's reference to the file the picker is
    open on — a fact that used to live only in the status bar, overwritten by the very
    autorun the acceptance triggers a moment later."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols, file_name="synth_case_B.csv",
                             reference_file="synth_case_A.csv")
    assert dlg.file_warn.isHidden() is False              # not shown -> isVisible() needs a real window
    assert "synth_case_A.csv" in dlg.file_warn.text()
    assert "synth_case_B.csv" in dlg.file_warn.text()
    assert dlg.btn_ok.text() == "Replace rest reference"
    dlg.deleteLater()


def test_the_same_reference_file_shows_no_warning(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols, file_name="synth_case_A.csv",
                             reference_file="synth_case_A.csv")
    assert dlg.file_warn.isHidden() is True
    assert dlg.btn_ok.text() == "Set noise profile"
    dlg.deleteLater()


def test_no_reference_yet_shows_no_warning(qapp):
    """Nothing to 'replace' on a test's first-ever reference — the default text/behaviour
    must be unchanged from before D07."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols, file_name="synth_case_A.csv", reference_file="")
    assert dlg.file_warn.isHidden() is True
    assert dlg.btn_ok.text() == "Set noise profile"
    dlg.deleteLater()


# -- D08 (UI-overhaul): the picker's hint matches the signal it actually shows -----
# Regression: the hint asked the user to find a "quiet (EMG-free)" span while the
# picker was fed RAW EMG, which on real data is dominated by heartbeats the profile is
# never built from (it is built from the ECG-reduced matrix). The caller now resolves
# which signal to pass and whether ECG removal ran; the dialog just reflects that.

def _capture_items(plot_item):
    from respmech.ui.plot_overlays import _CAPTURE_Z as z
    return [it for it in plot_item.listDataItems() if it.zValue() in (z, z + 1)]


def test_ecg_applied_hint_says_heartbeats_are_removed(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols, ecg_applied=True)
    assert "already been removed" in dlg.hint.text()
    assert "quiet (EMG-free)" not in dlg.hint.text()
    dlg.deleteLater()


def test_ecg_not_applied_hint_says_heartbeats_remain(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data()
    dlg = NoiseProfileDialog(raw, t, fs, cols, ecg_applied=False)
    assert "still in this signal" in dlg.hint.text()
    assert "already been removed" not in dlg.hint.text()
    dlg.deleteLater()


def test_peak_times_draw_capture_markers_on_every_channel(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data(nch=3)
    dlg = NoiseProfileDialog(raw, t, fs, cols, peak_times=[0.2, 0.5, 0.8])
    assert dlg._plots and all(len(_capture_items(p)) == 2 for p in dlg._plots), \
        "expected a line + a ▼ marker item on every channel"
    dlg.deleteLater()


def test_no_peak_times_draws_no_markers(qapp):
    """Backward-compatible default: a caller that does not pass peak_times (or an
    accepted-but-peakless capture) gets the picker exactly as before D08."""
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data(nch=3)
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    assert all(_capture_items(p) == [] for p in dlg._plots)
    dlg.deleteLater()


# ---------------------------------------------------------------- D09: the caption --
# The core's own stability warning ("Noise reference gives only N STFT frames…")
# went through warnings.warn to a stderr no packaged-app user ever sees, while the
# fidelity verdict for the same short reference read as a pass. The fidelity panel
# now carries a persistent caption, driven by the SAME frame arithmetic in the UI
# (never warnings.catch_warnings: process-global, and the worker thread runs
# beside other jobs).

def _preview(qapp, tmp_path):
    from respmech.core.settings import Settings
    from respmech.ui.state import AppState
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = Settings()
    s.output.folder = str(tmp_path)
    pv = PreviewScreen(AppState(s))
    qapp.processEvents()
    return pv


def test_fidelity_caption_is_visible_below_the_stability_threshold(qapp, tmp_path):
    pv = _preview(qapp, tmp_path)
    pv._update_fidelity_caption({"noise_clip_frames": 4})
    assert pv.fidelity_caption.isHidden() is False
    txt = pv.fidelity_caption.fullText() if hasattr(pv.fidelity_caption, "fullText") else pv.fidelity_caption.text()
    assert "4" in txt and "frame" in txt.lower()
    assert pv.fidelity_caption.property("status") == "warn"
    pv.deleteLater()


def test_fidelity_caption_is_absent_at_or_above_the_threshold(qapp, tmp_path):
    pv = _preview(qapp, tmp_path)
    pv._update_fidelity_caption({"noise_clip_frames": 11})
    assert pv.fidelity_caption.isHidden() is True
    # and a report WITHOUT the key (older cache entries, the batch path) hides
    # rather than guesses — showing a stale count would be worse than nothing
    pv._update_fidelity_caption({"noise_clip_frames": 3})
    pv._update_fidelity_caption({})
    assert pv.fidelity_caption.isHidden() is True
    pv._update_fidelity_caption({"noise_clip_frames": 3})
    pv._update_fidelity_caption(None)
    assert pv.fidelity_caption.isHidden() is True
    pv.deleteLater()


def test_the_tab_and_dialog_quote_the_same_frame_count(qapp, tmp_path):
    """The ticket's acceptance in one line: for the same span, the figure the tab
    computes is the figure the dialog shows. Both call ui.stft_frames, so this is
    structural; here it is also measured."""
    import re as _re
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    from respmech.ui.stft_frames import stft_frame_count

    raw, t, fs, cols = _data(n=2000, fs=1000)
    dlg = NoiseProfileDialog(raw, t, fs, cols, win_length=256, hop_length=64)
    dlg._set_selection(0.20, 0.65)                        # 0.45 s -> 4 frames
    vist = _re.search(r"gives (\d+) STFT", dlg.warn.text())
    assert vist is not None
    assert int(vist.group(1)) == stft_frame_count(int(round(0.45 * fs)), 256, 64)
    dlg.deleteLater()

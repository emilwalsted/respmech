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
    dlg._set_selection(0.20, 0.50)
    assert dlg.selected_region() == (0.20, 0.50)
    assert dlg.btn_ok.isEnabled() is True
    assert all(r.isVisible() for r in dlg._regions)      # shaded on every channel
    assert dlg.warn.isHidden() is True                   # 0.30 s <= 0.5 s -> no warning
    # a plain click clears it (afværge)
    dlg._clear_selection()
    assert dlg.selected_region() is None and dlg.btn_ok.isEnabled() is False
    assert not any(r.isVisible() for r in dlg._regions)
    dlg.deleteLater()


def test_wide_selection_warns_about_processing_time(qapp):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    raw, t, fs, cols = _data(n=3000)
    dlg = NoiseProfileDialog(raw, t, fs, cols)
    dlg._set_selection(0.10, 1.20)                       # 1.10 s > 0.5 s
    assert dlg.warn.isHidden() is False
    assert "processing time" in dlg.warn.text().lower()
    # narrowing back under the threshold hides the warning again
    dlg._set_selection(0.10, 0.30)
    assert dlg.warn.isHidden() is True
    dlg.deleteLater()


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

"""The main window must be able to fit on a normal laptop screen.

Emil's report on the v2.3.1 dmg: choosing **Preview & QC** made the window wider than the
screen and it ran off the right edge on a Mac. Measured cause: the EMG control strips were
single non-wrapping ``QHBoxLayout`` rows, so each row's minimum width was the SUM of its chips
(noise 1770 px, ECG 1039 px). Qt propagates a layout's minimum up to the window
(``QLayout.SetDefaultConstraint``), which pinned the window's minimum width at 1800 px — and a
window can never be resized below its minimum, so both ``showMaximized()`` and
``MainWindow._fit_to_screen()`` were powerless.

The guard: the window's minimum width must stay comfortably under the narrowest screen we care
about. The smallest current MacBook is 1280 logical px wide, so 1280 is the ceiling; anything
approaching it means a strip has stopped wrapping again.

That guard then measured 1005 px here and passed for three days while Windows CI measured 1516
and failed: the chips are the same widgets, but the platform's font metrics decide how wide
they are, and macOS's are the narrowest we build for. "Comfortably" was doing real work in that
sentence and nothing was checking it.

So the two guards added afterwards are RATIOS, not pixel figures, and every precondition is
stated relative to something measured in the same run. A pixel literal here is a measurement of
this machine's fonts: the first cut of both tests hard-coded numbers read off macOS and went
red on Windows for that exact reason, which is the mistake this file is about.

To reproduce the Windows runner locally, widen the font's HORIZONTAL advance and leave its
height alone -- ``QFont.setStretch(145)`` on the application font puts this machine within a
few percent of the runner (verified against CI: unfixed window minimum 1506-1538 px modelled
vs 1516 px measured). Do NOT scale the point size instead: that inflates row heights as well,
so it understates the width problem and invents height failures that Windows never has.
"""
import pytest

from respmech.ui.state import AppState  # noqa: E402

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

# The narrowest screen the app must fit on (a 13" MacBook is 1280 logical px).
_NARROWEST_SCREEN = 1280

# A wrapping chip's minimum is its widest single cluster, so it sits well below the chip's
# natural one-line width; a chip that cannot wrap has minimum == natural, i.e. exactly 1.0.
# 0.65 rather than 0.5 because the ratio is not scale-free: the gated-peak chip's widest
# cluster is a long caption, which grows with the font while the spin boxes beside it barely
# do, so the ratio drifts up on wider metrics (measured 0.475 here, 0.546 on the Windows
# runner). Anything under 0.65 still separates "wraps" from "cannot wrap" by a wide margin.
_WRAPPED = 0.65

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _window(qapp, tmp_path, **kw):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), data_out=_DATA_OUT, **kw)
    win = MainWindow(AppState(s))
    win.resize(1200, 800)
    win.show()
    for _ in range(6):
        qapp.processEvents()
    return win


@pytest.mark.parametrize("tab", [0, 1, 2])
def test_window_minimum_width_fits_a_laptop_screen(qapp, tmp_path, tab):
    """On every screen — Setup, Preview & QC, Run — the window must be able to be narrow."""
    win = _window(qapp, tmp_path, remove_ecg=True, noise=True)
    win.tabs.setCurrentIndex(tab)
    for _ in range(6):
        qapp.processEvents()
    got = win.minimumSizeHint().width()
    assert got < _NARROWEST_SCREEN, (
        f"tab {tab}: the window demands {got} px minimum width, which does not fit a "
        f"{_NARROWEST_SCREEN} px screen — a control strip has stopped wrapping")
    win.close()


def test_the_emg_strips_wrap_rather_than_summing_their_chips(qapp, tmp_path):
    """The specific regression: the EMG subtab pages must not demand the sum of their chips.

    Measured before the fix: noise 1770 px, ECG 1039 px. Each page's minimum should now be
    about its widest single chip, so it stays well under a laptop screen.
    """
    win = _window(qapp, tmp_path, remove_ecg=True, noise=True)
    win.tabs.setCurrentIndex(1)
    pv = win.preview_screen
    pv.refresh_files()
    for _ in range(8):
        qapp.processEvents()

    pages = {pv.subtabs.tabText(i): pv.subtabs.widget(i).minimumSizeHint().width()
             for i in range(pv.subtabs.count())}
    assert pages, "no preview subtabs were built — the assertion would be vacuous"
    for name, width in pages.items():
        assert width < _NARROWEST_SCREEN, f"subtab {name!r} demands {width} px"
    win.close()


def test_every_control_chip_can_be_squeezed_far_below_its_natural_width(qapp, tmp_path):
    """The invariant behind the pixel ceiling, stated without pixels.

    A chip laid out as one plain QHBoxLayout has minimum == natural width: it cannot be
    squeezed at all, and whatever it sums to is forced on the window. That is exactly what
    the ECG chip was, and adding one more checkbox to it (the "Auto (whole batch)" tickbox)
    was enough to push the window past a laptop screen on Windows. A chip whose clusters wrap
    asks for its natural width but can fall back to its widest single cluster, so the ratio
    is a small fraction — and being a ratio, it says the same thing on every platform's fonts.
    """
    win = _window(qapp, tmp_path, remove_ecg=True, noise=True)
    pv = win.preview_screen
    for name in ("ecg_opts", "noise_opts", "gate_opts"):
        chip = getattr(pv, name)
        natural, floor = chip.sizeHint().width(), chip.minimumSizeHint().width()
        assert natural > 200, (
            f"{name} reports a {natural} px natural width — it is empty or was never laid out, "
            f"and the ratio below would pass on nothing")
        assert floor < natural * _WRAPPED, (
            f"{name} demands {floor} px of its {natural} px natural width — it cannot wrap, so "
            f"its whole width is forced on the window")
    win.close()


def test_a_squeezed_chip_wraps_inside_itself_rather_than_off_the_edge(qapp, tmp_path):
    """Fitting the screen is only half of it: the controls must still be ON it.

    Qt does not clip a child that is laid out too wide, it just draws it past the right edge —
    so a chip placed at its natural width inside a narrower strip lets the window honour its
    minimum while the knobs walk off the window. The mirror of that is vertical: a chip whose
    size policy caps it at its one-line height wraps its second row correctly and then paints
    it below its own bottom edge. Both were true of a first cut at this fix.
    """
    win = _window(qapp, tmp_path, remove_ecg=True, noise=True)
    win.tabs.setCurrentIndex(1)              # Preview & QC, or nothing below is ever laid out
    pv = win.preview_screen
    pv.refresh_files()
    for _ in range(6):
        qapp.processEvents()
    ecg = next(i for i in range(pv.subtabs.count()) if "ECG" in pv.subtabs.tabText(i))
    pv.subtabs.setCurrentIndex(ecg)
    # Squeeze to the window's OWN minimum rather than a pixel figure read off this machine:
    # that minimum is ~700 px here and ~1230 px on the Windows runner, so a literal would be
    # measuring the platform's fonts instead of the layout. Whatever it is, the chip's natural
    # width must exceed it, which is the precondition the assertions below need.
    win.resize(win.minimumSizeHint().width(), 850)
    for _ in range(12):
        qapp.processEvents()

    assert pv.ecg_opts.isVisible(), (
        "the ECG chip was never shown, so its geometry is the unlaid-out default and every "
        "assertion below would be measuring nothing")
    assert win.width() < pv.ecg_opts.sizeHint().width(), (
        f"the window ({win.width()} px) is still wider than the ECG chip wants to be "
        f"({pv.ecg_opts.sizeHint().width()} px), so the chip is never asked to wrap and the "
        f"rest of this test proves nothing")
    chip, page = pv.ecg_opts, pv.subtabs.widget(ecg)
    assert chip.x() + chip.width() <= page.width() + 1, (
        f"the ECG chip reaches {chip.x() + chip.width()} px on a {page.width()} px page — it is "
        f"drawn past the right edge instead of wrapping")
    for w in (pv.ecg_capture_channel, pv.remove_ecg, pv.ecg_auto_batch,
              pv.ecg_min_height, pv.ecg_min_distance):
        assert w.x() + w.width() <= chip.width() + 1, f"{w.objectName() or w} runs off the chip"
        assert w.y() + w.height() <= chip.height() + 1, (
            f"a wrapped row is painted below the chip's own bottom edge "
            f"({w.y() + w.height()} px vs {chip.height()} px) — the chip did not grow a line")
    win.close()


def test_the_noise_reference_readout_is_elided_not_unbounded(qapp, tmp_path):
    """A real recording's filename is long; un-elided it would push the strip (and the window)
    arbitrarily wide, which is how the 1800 px minimum crept in unnoticed."""
    win = _window(qapp, tmp_path, remove_ecg=True, noise=True)
    pv = win.preview_screen
    long_name = "SUBJ07_" + "very_long_recording_name_" * 6 + ".csv"
    n = pv.state.settings.processing.emg.noise
    n.reference_file, n.use_expiration, n.reference_intervals = long_name, False, [[0.1, 0.9]]
    pv._refresh_noise_readout()
    for _ in range(4):
        qapp.processEvents()

    label = pv.noise_ref_readout
    assert label.minimumSizeHint().width() < 400, (
        f"the read-out is {label.minimumSizeHint().width()} px wide for a long filename")
    assert long_name in label.toolTip(), "the full reference must stay available on hover"
    assert label.text(), "the read-out collapsed to nothing — it must stay visible"
    win.close()

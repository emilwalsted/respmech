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
"""
import pytest

from respmech.ui.state import AppState  # noqa: E402

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

# The narrowest screen the app must fit on (a 13" MacBook is 1280 logical px).
_NARROWEST_SCREEN = 1280

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

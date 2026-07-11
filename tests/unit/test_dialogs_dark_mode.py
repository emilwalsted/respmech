"""Every dialog must render in both light and dark mode.

Dark mode is OS-auto-detected (theme._prefers_dark). Qt widgets follow the applied
palette/QSS automatically, but the pyqtgraph/matplotlib plots inside the channel-setup
and noise-profile dialogs do NOT inherit Qt styling — they must pull the dark plot
palette via theme.plot_palette(). These tests force dark mode, build every dialog, and
assert the theme is dark and the plot surfaces are dark (not the light fallback).
"""
import numpy as np
from PySide6.QtWidgets import QApplication

from _helpers import is_dark_hex as _is_dark_hex  # noqa: F401 (dark_app fixture from conftest)


def test_theme_switches_to_dark(dark_app):
    _app, theme = dark_app
    assert theme.is_dark()
    assert _is_dark_hex(theme.plot_palette()["bg"])       # dark plot surface, not the light fallback


def test_startup_dialog_dark(dark_app):
    from respmech.ui.startup_dialog import StartupDialog
    from PySide6.QtWidgets import QLabel
    dlg = StartupDialog(); dlg.show(); QApplication.processEvents()
    heading = next(l for l in dlg.findChildren(QLabel) if l.property("role") == "heading")
    fg = heading.palette().color(heading.foregroundRole())
    assert fg.lightness() > 150                           # heading text is light on the dark ground
    dlg.close()


def test_channel_setup_dialog_dark(dark_app):
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    n = 1200
    mat = np.column_stack([np.arange(n) / 500.0, 0.02 * np.sin(np.linspace(0, 40, n)),
                           -80 + 30 * np.sin(np.linspace(0, 15, n)), np.linspace(0, 1, n)])
    dlg = ChannelSetupDialog(["demo.csv"], 500, loader=lambda p: (mat, ["t", "e", "p", "v"]))
    dlg.show(); QApplication.processEvents()
    assert _is_dark_hex(dlg._pal["bg"])                   # the previews use the dark plot palette
    assert dlg._plots                                     # and they were built without error
    dlg.close()


def test_noise_profile_dialog_dark(dark_app):
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog, _plot_pal
    t = np.arange(2500) / 500.0
    raw = [0.02 * np.sin(2 * np.pi * 60 * t) * np.clip(np.sin(np.pi * t / 1.5), 0, None)
           + 0.004 * np.random.default_rng(i).standard_normal(2500) for i in range(3)]
    dlg = NoiseProfileDialog(raw, t, 500, [2, 3, 4], file_name="demo.csv")
    dlg.show(); QApplication.processEvents()
    assert _is_dark_hex(_plot_pal()["bg"])
    assert len(dlg._plots) == 3
    dlg.close()


def test_text_viewer_dialog_dark(dark_app):
    from respmech.ui.dialogs import TextViewerDialog
    dlg = TextViewerDialog("Error log", "some error\n(traceback)", intro="A file failed.")
    dlg.show(); QApplication.processEvents()
    # the window inherits the dark application palette
    assert dlg.palette().color(dlg.backgroundRole()).lightness() < 120
    dlg.close()

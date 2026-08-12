"""Stacked channel graphs must share one left margin so their x-axes line up.

Without a pinned left-axis width, a panel whose y-ticks read "-100" sits further right
than one reading "0", and the stacked channels visibly step in and out. theme.align_left_axis
pins every panel to the same width; these tests assert the panels end up x-aligned.
"""
import os

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from _helpers import INPUT  # noqa: F401  (qapp fixture comes from conftest)


def _viewbox_lefts(plots):
    QApplication.processEvents()
    QApplication.processEvents()
    return [p.getViewBox().sceneBoundingRect().left() for p in plots]


def test_channel_setup_previews_are_x_aligned(qapp):
    """The column previews use separate PlotWidgets (no shared grid) — the case that
    actually misaligned. Columns of very different magnitude must still line up."""
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    n = 1500
    mat = np.column_stack([
        np.arange(n) / 500.0,                       # time: 0..3   (narrow ticks)
        0.02 * np.sin(np.linspace(0, 40, n)),       # EMG: ±0.02
        -80 + 30 * np.sin(np.linspace(0, 15, n)),   # Poes: -60..-110  (wide ticks)
        np.linspace(0, 1, n)])
    dlg = ChannelSetupDialog(["demo.csv"], 500, loader=lambda p: (mat, ["t", "e", "p", "v"]))
    dlg.resize(760, 560); dlg.show()
    widths = [p.getAxis("left").width() for p in dlg._plots]
    assert len(set(round(w) for w in widths)) == 1          # all one width
    lefts = _viewbox_lefts(dlg._plots)
    assert max(lefts) - min(lefts) < 1.0                    # plotting areas start together
    dlg.close()


@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input absent")
def test_mechanics_channel_stack_is_x_aligned(qapp):
    from respmech.ui.workers import stage_mechanics_preview
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.screens.preview_screen import _CHANNELS
    from _helpers import synth_settings
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    pv.plots.setFixedSize(640, 540)
    lefts = _viewbox_lefts(pv._channel_plots)               # processes events → layout settles
    widths = [p.getAxis("left").width() for p in pv._channel_plots]
    assert len(set(round(w) for w in widths)) == 1          # every panel one width → aligned
    assert max(lefts) - min(lefts) < 1.0                    # plotting areas start together
    # each channel must name itself on its own axis — this is the screen whose job is to
    # let the user confirm the channel assignment they just made (regression: 5b4e33b
    # swapped in set_channel_label without the showLabel()/labelText that setLabel() used
    # to do for free, so the axes carried the right HTML but never displayed it)
    for p, (_key, expected_label, _colour) in zip(pv._channel_plots, _CHANNELS):
        expected_name = expected_label.partition(" (")[0]
        axis = p.getAxis("left")
        assert axis.label.isVisible()
        assert axis.labelText == expected_name
    win.close()


@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input absent")
def test_mechanics_crosshair_names_the_channel(qapp):
    """The crosshair readout is the tool that turns the picture into a number — it must say
    which channel and unit that number belongs to, not the generic 'value' (regression:
    _on_mech_mouse_moved reads getAxis("left").labelText, which set_channel_label never set)."""
    from respmech.ui.workers import stage_mechanics_preview
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from _helpers import synth_settings
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    QApplication.processEvents(); QApplication.processEvents()   # let the scene layout settle
    p2 = pv._channel_plots[2]                                # Poes — third stacked curve
    pv._on_mech_mouse_moved([QPointF(p2.sceneBoundingRect().center())])
    text = pv.crosshair_label.text()
    assert "Poes = " in text
    assert "cmH" in text                                     # names the unit too, not just the channel
    assert "value" not in text
    win.close()


def test_align_helper_is_safe_and_fixed_width(qapp):
    import pyqtgraph as pg
    from respmech.ui import theme
    p = pg.PlotWidget(); p.resize(320, 200); p.show()
    theme.align_left_axis(p)
    QApplication.processEvents(); QApplication.processEvents()
    assert abs(p.getAxis("left").width() - theme.PLOT_AXIS_WIDTH) <= 1   # pinned to the fixed width
    theme.align_left_axis(None)                              # never raises on a bad arg
    p.close()

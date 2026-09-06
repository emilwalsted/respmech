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


def test_scitaxis_label_returns_after_a_panel_shrinks_then_grows_back(qapp):
    """A SciAxis label that hid itself because a panel got too short to hold "name + unit"
    (regression: found in this ticket's own self-review) must come back once the panel is
    tall enough again — the same panel can shrink and grow repeatedly as a splitter is
    dragged, and a permanently blank axis after one such round-trip is a real, user-visible
    defect, not a one-off cosmetic glitch."""
    from respmech.ui.screens.preview._plot_helpers import SciAxis
    ax = SciAxis(orientation="left")
    ax.set_channel_label("Oesophageal pressure", "cmH2O")
    assert ax.label.isVisible()
    ax.resize(20, 30)                                          # too short for any wording
    assert not ax.label.isVisible()
    ax.resize(20, 400)                                         # grown back — plenty of room
    assert ax.label.isVisible(), "label must reappear once the panel is tall enough again"


def _scitaxis(name, unit):
    """A bare, laid-out-nowhere SciAxis with its label set, plus the widths its two
    base-size wordings need — measured, so no test below carries a pixel literal."""
    from respmech.ui.screens.preview._plot_helpers import SciAxis
    ax = SciAxis(orientation="left")
    ax.set_channel_label(name, unit)
    widths = {}
    for include_unit in (True, False):
        ax.label.setHtml(ax._label_html(include_unit))
        widths[include_unit] = ax.label.boundingRect().width()
    ax.label.setHtml(ax.labelString())
    return ax, widths[True], widths[False]


def _label_fits(ax):
    return ax.label.isVisible() and ax.label.boundingRect().width() <= ax.height()


@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input absent")
def test_mechanics_channel_stack_names_every_channel_in_windows_metrics(windows_metrics):
    """The same stack as test_mechanics_channel_stack_is_x_aligned, in the Windows
    runner's ~1.5x wider font. That is where "Volume" alone outgrew the 76 px axis the
    96 px row floor leaves, and where the name+unit / name / nothing picker went blank
    (Windows CI red, 06-09-2026, while macOS and Linux showed the label). Every channel
    must still name itself, and no label may be wider than the axis it is rotated into —
    a wider one runs into the row above or below (the defect 14.3 exists for)."""
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
    _viewbox_lefts(pv._channel_plots)                        # processes events → layout settles
    for p, (_key, expected_label, _colour) in zip(pv._channel_plots, _CHANNELS):
        axis = p.getAxis("left")
        name = expected_label.partition(" (")[0]
        assert axis.labelText == name
        assert axis.label.isVisible(), f"{name}: axis left blank in Windows metrics"
        need, have = axis.label.boundingRect().width(), axis.height()
        assert need <= have, f"{name}: label needs {need:.0f} px of a {have:.0f} px axis"
        assert name in axis.label.toPlainText(), f"{name}: the label no longer names it"
    win.close()


def test_scitaxis_pick_survives_pyqtgraphs_own_relabel(qapp):
    """The shortened wording must outlive pyqtgraph re-rendering the label from
    labelString() — which it does on every range change and inside showLabel(True).
    Regression: 14.3's labelString() always returned name + unit, so its "name alone"
    pick was overwritten within the very showLabel(True) that confirmed it (measured:
    a 94 px "Poes (cmH₂O)" back on a 76 px axis, the overrun the picker exists to stop)."""
    ax, full, name_only = _scitaxis("Poes", "cmH₂O")
    assert name_only < full
    ax.resize(20, int((full + name_only) / 2))                 # room for the name, not the unit
    assert _label_fits(ax), "the name alone fits this axis and must be what is shown"
    assert "cmH₂O" not in ax.label.toPlainText()
    ax.setRange(-40.0, 10.0)                                   # → updateAutoSIPrefix → _updateLabel
    assert _label_fits(ax), "pyqtgraph's own relabel put the unit line back"
    assert "cmH₂O" not in ax.label.toPlainText()
    ax.showLabel(True)
    assert _label_fits(ax) and "cmH₂O" not in ax.label.toPlainText()


def test_scitaxis_shrinks_the_name_before_hiding_it(qapp):
    """When even the name alone is too long at the base size, the axis shows it smaller
    rather than blank — down to _MIN_LABEL_SCALE of the base, which is what keeps "Volume"
    on a 76 px axis in the Windows runner's font. And the shrink is a response to the
    height, not a permanent downgrade: grown back, the base size and the unit return."""
    ax, full, name_only = _scitaxis("Volume", "L")
    ax.resize(20, int(name_only * 0.85))                       # too short for the base size
    assert _label_fits(ax), "a name that fits at a smaller font must not be hidden"
    assert "Volume" in ax.label.toPlainText()
    size, _unit = ax._label_size
    base, _ = ax._base_font_size()
    assert ax._MIN_LABEL_SCALE * base <= size < base
    ax.resize(20, int(full * 3))                               # grown back — plenty of room
    assert _label_fits(ax)
    assert ax._label_size is None and "L" in ax.label.toPlainText(), (
        "the base size and the unit line must come back once there is room")
    ax.resize(20, int(name_only * ax._MIN_LABEL_SCALE * 0.5)) # too short even at the floor
    assert not ax.label.isVisible(), "below the floor the label hides rather than overruns"


def test_scitaxis_keeps_the_scale_annotation_when_it_drops_the_unit(qapp):
    """The ·10ⁿ annotation is the only thing on screen saying a tick reading "500" means
    0.5 L, so a wording that drops the unit line must keep it while the ticks stay
    scaled — the shortening may never leave the ticks silently misstated."""
    ax, _full, _name_only = _scitaxis("Volume", "L")
    ax.resize(20, 400)
    ax.setRange(0.0, 0.5)                                      # 0..0.5 L → ticks in 10⁻³ L
    assert ax.autoSIPrefixScale == 1000.0
    assert "·10⁻³" in ax.label.toPlainText() and "L" in ax.label.toPlainText()
    annotated = ax.label.boundingRect().width()
    ax.resize(20, int(annotated) - 1)                          # too short for the full wording
    assert ax.autoSIPrefixScale == 1000.0, "the ticks are still scaled…"
    assert _label_fits(ax)
    assert "·10⁻³" in ax.label.toPlainText(), "…so the label must still say so"

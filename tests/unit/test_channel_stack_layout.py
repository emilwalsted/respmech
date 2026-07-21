"""The stacked channel views on the three Preview sub-tabs share one styling helper, and
the linking contract differs by tab in a way that is easy to break later:

  - x is always linked (pan/zoom the time axis once, every channel follows);
  - y is linked on the EMG stacks (raw, ECG-processed) — same units, so a shared scale lets
    you compare electrodes and one zoom zooms all;
  - y is NOT linked on the mechanics stack — flow, volume and the pressures are different
    units, and a shared range would flatten most of them;
  - the shared EMG y-range is the UNION of every channel, so synchronising never clips a
    louder electrode down to a quieter first one;
  - x tick VALUES show only on the bottom channel, so the stack reads as one block.

The link CONTRACT is checked via ViewBox.linkedView rather than by measuring ranges: without
a spinning event loop pyqtgraph's autorange has not run, so viewRange() is unreliable, but
the link topology is set synchronously by the helper.
"""
import numpy as np
import pyqtgraph as pg

from respmech.ui.screens.preview_screen import PreviewScreen

_X, _Y = pg.ViewBox.XAxis, pg.ViewBox.YAxis


def _stack(amps, *, link_y):
    """Returns (glw, plots). The caller MUST keep glw — if it is collected the PlotItems go
    with it and pyqtgraph raises 'Signal source has been deleted' mid-test."""
    glw = pg.GraphicsLayoutWidget()
    t = np.linspace(0, 1, 400)
    plots = []
    for i, a in enumerate(amps):
        p = glw.addPlot(row=i, col=0)
        p.plot(t, a * np.sin(2 * np.pi * 6 * t))
        plots.append(p)
    PreviewScreen._style_channel_stack(glw, plots, link_y=link_y)
    return glw, plots


def test_x_is_always_linked(qapp):
    for link_y in (True, False):
        glw, plots = _stack([1.0, 1.0, 1.0], link_y=link_y)
        for p in plots[1:]:
            assert p.getViewBox().linkedView(_X) is plots[0].getViewBox()


def test_emg_stacks_link_y(qapp):
    glw, plots = _stack([1.0, 1.0, 1.0], link_y=True)
    for p in plots[1:]:
        assert p.getViewBox().linkedView(_Y) is plots[0].getViewBox()


def test_the_shared_emg_range_is_the_union_and_never_clips(qapp):
    """The loudest channel need not be the first — the shared range must still cover it."""
    glw, plots = _stack([0.2, 1.0, 5.0], link_y=True)
    ranges = [p.getViewBox().viewRange()[1] for p in plots]
    assert all(abs(r[0] - ranges[0][0]) < 1e-6 for r in ranges), "y not synchronised"
    assert ranges[0][1] >= 5.0, "the shared range clipped the loudest channel"


def test_mechanics_stack_does_not_link_y(qapp):
    glw, plots = _stack([0.2, 1.0, 5.0], link_y=False)
    for p in plots[1:]:
        assert p.getViewBox().linkedView(_Y) is None, "mechanics channels share a y-range"


def test_x_values_show_only_on_the_bottom_channel(qapp):
    glw, plots = _stack([1.0, 1.0, 1.0], link_y=True)
    assert plots[0].getAxis("bottom").style["showValues"] is False
    assert plots[1].getAxis("bottom").style["showValues"] is False
    assert plots[-1].getAxis("bottom").style["showValues"] is True


def test_a_single_channel_stack_is_handled(qapp):
    glw, plots = _stack([1.0], link_y=True)          # one electrode: nothing to link
    assert plots[0].getAxis("bottom").style["showValues"] is True
    assert plots[0].getViewBox().linkedView(_Y) is None


def test_empty_is_a_noop(qapp):
    PreviewScreen._style_channel_stack(pg.GraphicsLayoutWidget(), [], link_y=True)

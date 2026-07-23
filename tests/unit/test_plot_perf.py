"""Preview plot decimation (Wave 3.1) — display-only, and switchable off at runtime.

``ui/plot_perf.py`` turns on pyqtgraph's own decimation and view clipping so a repaint does
not hand every sample of a multi-million-point recording to Qt. These tests pin the three
properties that make it safe to leave on: it reaches the curves, it changes nothing about
the underlying data, and it can be disabled without a rebuild.
"""
import importlib

import numpy as np
import pyqtgraph as pg
import pytest

from respmech.ui import plot_perf


@pytest.fixture()
def plot_widget(qapp):
    w = pg.PlotWidget()
    yield w
    w.deleteLater()


def test_tune_enables_decimation_and_clipping(plot_widget):
    item = plot_widget.getPlotItem()
    assert item.downsampleMode() == (1, False, "peak")   # pyqtgraph default: auto off
    assert item.clipToViewMode() is False

    plot_perf.tune(item)

    ds, auto, mode = item.downsampleMode()
    assert auto is True and mode == "peak"
    assert item.clipToViewMode() is True


def test_curves_added_later_inherit_the_settings(plot_widget):
    """The whole design rests on this: PlotItem.addItem propagates to each PlotDataItem,
    so the ~8 .plot() call sites in preview_screen.py never have to be touched."""
    plot_perf.tune(plot_widget.getPlotItem())
    curve = plot_widget.plot([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert curve.opts["autoDownsample"] is True
    assert curve.opts["downsampleMethod"] == "peak"
    assert curve.opts["clipToView"] is True


def test_decimation_is_display_only(plot_widget):
    """Decimation must not touch the data behind the curve — only what gets rasterised."""
    plot_perf.tune(plot_widget.getPlotItem())
    t = np.arange(200_000) / 1000.0
    y = np.sin(2 * np.pi * 3 * t)
    curve = plot_widget.plot(t, y)
    plot_widget.getPlotItem().vb.setXRange(0.0, 30.0)

    assert curve.xData.size == t.size and curve.yData.size == y.size
    assert np.array_equal(curve.xData, t)
    assert np.array_equal(curve.yData, y)


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_env_kill_switch_disables_it(monkeypatch, plot_widget, value):
    """RESPMECH_PLOT_DOWNSAMPLE=0 must make this a no-op, so a good/bad comparison on real
    data is one relaunch apart rather than a rebuild."""
    monkeypatch.setenv("RESPMECH_PLOT_DOWNSAMPLE", value)
    importlib.reload(plot_perf)
    try:
        assert plot_perf.enabled() is False
        item = plot_widget.getPlotItem()
        plot_perf.tune(item)
        assert item.downsampleMode() == (1, False, "peak")
        assert item.clipToViewMode() is False
    finally:
        monkeypatch.delenv("RESPMECH_PLOT_DOWNSAMPLE", raising=False)
        importlib.reload(plot_perf)


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("RESPMECH_PLOT_DOWNSAMPLE", raising=False)
    importlib.reload(plot_perf)
    assert plot_perf.enabled() is True


def test_scatter_markers_do_not_break_downsampling(plot_widget):
    """Regression: pyqtgraph 0.14's PlotItem.updateDownsampling() iterates self.curves and
    calls setDownsampling()/setClipToView() on every entry, but self.curves also holds
    ScatterPlotItems (the ECG R-peak markers), which implement neither. Before the shim in
    plot_perf, enabling decimation on a plot carrying markers spewed

        AttributeError: 'ScatterPlotItem' object has no attribute 'setDownsampling'

    on every update. Reported from a native run — the offscreen suite had not touched a
    plot that carried both a curve and markers.
    """
    item = plot_widget.getPlotItem()
    plot_perf.tune(item)
    plot_widget.plot(np.arange(1000) / 100.0, np.random.default_rng(0).standard_normal(1000))
    item.addItem(pg.ScatterPlotItem(x=np.array([1.0, 2.0, 3.0]), y=np.array([0.0, 0.0, 0.0])))

    item.updateDownsampling()          # raised AttributeError before the fix
    item.vb.setXRange(0.0, 5.0)


def test_tune_never_raises():
    """A slow plot is a nuisance; a plot that fails to render is a bug. Tuning must not be
    able to turn the first into the second."""
    plot_perf.tune(None)
    plot_perf.tune(object())
    plot_perf.tune_widget(None)
    plot_perf.tune_widget(object())

"""Rendering performance for the pyqtgraph preview plots.

Preview curves are drawn at full resolution: a 30-minute recording at 2 kHz is ~3.6 M
points per channel, and pyqtgraph hands every one of them to Qt on each repaint even
though the widget is only a few hundred pixels wide. Turning on pyqtgraph's own decimation
and view clipping cuts a repaint from ~19 ms to ~1.5 ms.

Two settings, applied once per ``PlotItem``. ``PlotItem.addItem`` propagates both to every
``PlotDataItem`` it holds — including curves added later — so the call sites never have to
touch the individual ``.plot()`` calls:

* ``setDownsampling(auto=True, mode="peak")`` — draw a min/max envelope when zoomed out.
  For EMG this is arguably *more* faithful than the status quo, where the rasteriser thins
  the trace arbitrarily and can drop the peak of a burst outright. It is nonetheless a
  visible change and wants a look on real data before it is taken for granted.
* ``setClipToView(True)`` — build the path only for the visible x-range. This requires
  monotonically increasing x; every preview curve is plotted against the time vector, so
  that holds. Anything plotted against something else must not be tuned here.

NB ``PlotItem`` spells the mode ``mode=`` while ``PlotDataItem`` spells it ``method=``.

TURNING IT OFF
--------------
Set ``RESPMECH_PLOT_DOWNSAMPLE=0`` in the environment to disable the whole thing at
runtime, so an A/B comparison is one relaunch apart and needs no rebuild::

    RESPMECH_PLOT_DOWNSAMPLE=0 respmech-gui

To remove it from the code entirely, revert the single commit that added this module — it
is deliberately self-contained, and nothing outside the display path depends on it.

CAVEAT
------
The ~19 ms -> ~1.5 ms figure was measured under ``QT_QPA_PLATFORM=offscreen``. Offscreen Qt
has repeatedly mis-stated GUI behaviour on this project, so treat it as indicative until it
has been confirmed on a real native window (and on Windows).

This module changes **display only** — no computed value, table or exported figure is
affected by anything here.
"""
from __future__ import annotations

import os

_ENV = "RESPMECH_PLOT_DOWNSAMPLE"


def enabled() -> bool:
    """Whether plot decimation is active. On by default; ``RESPMECH_PLOT_DOWNSAMPLE=0`` off."""
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


_scatter_shim_applied = False


def _shim_scatter_items() -> None:
    """Work around a pyqtgraph 0.14 bug that only bites once decimation is enabled.

    ``PlotItem.updateDownsampling()`` iterates ``self.curves`` and calls
    ``setDownsampling()`` / ``setClipToView()`` on every entry — but ``self.curves`` also
    holds ``ScatterPlotItem``s (our R-peak markers, ``plot_overlays.add_ecg_capture_markers``),
    which implement neither. ``PlotItem.addItem`` guards exactly these two calls with
    ``isinstance(item, PlotDataItem)``; ``updateDownsampling`` does not. So a PlotItem that
    carries markers raises ``AttributeError: 'ScatterPlotItem' object has no attribute
    'setDownsampling'`` on every update — harmless to the render, but it floods the console.

    Adding explicit no-ops is safe: neither method exists on ``ScatterPlotItem``, so nothing
    is shadowed, and a scatter has no decimation semantics to preserve — pyqtgraph's own
    intent (per ``addItem``) is precisely to skip non-``PlotDataItem`` curves.

    Applied from :func:`tune` rather than at import, so that with the kill-switch off this
    module leaves pyqtgraph completely untouched.
    """
    global _scatter_shim_applied
    if _scatter_shim_applied:
        return
    try:
        from pyqtgraph import ScatterPlotItem
        if not hasattr(ScatterPlotItem, "setDownsampling"):
            ScatterPlotItem.setDownsampling = lambda self, *a, **k: None
        if not hasattr(ScatterPlotItem, "setClipToView"):
            ScatterPlotItem.setClipToView = lambda self, *a, **k: None
        _scatter_shim_applied = True
    except Exception:               # noqa: BLE001 — see tune()
        pass


def tune(plot_item) -> None:
    """Enable decimation + view clipping on one pyqtgraph ``PlotItem``.

    Never raises: a plot that renders slowly is a nuisance, a plot that fails to render is
    a bug, so this must not be able to turn the former into the latter.
    """
    if plot_item is None or not enabled():
        return
    _shim_scatter_items()
    try:
        plot_item.setDownsampling(auto=True, mode="peak")
        plot_item.setClipToView(True)
    except Exception:               # noqa: BLE001 — display tuning, never fatal
        pass


def tune_widget(plot_widget) -> None:
    """:func:`tune` for a ``PlotWidget``, which wraps a single ``PlotItem``."""
    if plot_widget is None:
        return
    try:
        tune(plot_widget.getPlotItem())
    except Exception:               # noqa: BLE001 — see tune()
        pass

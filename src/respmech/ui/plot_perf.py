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


def close_plots(container) -> None:
    """Release every ``PlotItem`` a plot container holds, via pyqtgraph's OWN
    ``PlotItem.close()``/``ViewBox.setMenuEnabled(False)`` — the library's documented
    cleanup path ("Most of this crap is needed to avoid PySide trouble", see
    ``PlotItem.close()``'s own comment), not a hand-rolled sweep of already-closed
    windows from outside. Handles a bare ``PlotWidget`` (one ``PlotItem`` via
    ``getPlotItem()``) and a ``GraphicsLayoutWidget``/``GraphicsLayout`` (0..N
    ``PlotItem``s in ``.ci.items``).

    Why this exists: each ``PlotItem`` unconditionally builds its own context menu (one
    ``QMenu`` plus six submenus) and each ``ViewBox`` its own ``ViewBoxMenu`` — eagerly,
    at construction, whether or not the app ever shows them. Preview & QC's GUI tests
    never delete a closed ``MainWindow`` (deleting one segfaults on Python 3.11 — see
    ``tests/unit/conftest.py``), so those menus accumulated for the life of the test
    session: ~130 parentless QMenus per closed window, measured 6,137 QMenus after just
    57 GUI-heavy tests (RESPMECH_NET_CENSUS, point 6). An earlier attempt to fix this by
    reaping already-closed windows from OUTSIDE (a generic ``deleteLater`` +
    ``sendPostedEvents(DeferredDelete)`` sweep over ``app.topLevelWidgets()``) segfaulted
    nondeterministically on py3.11 and was never committed. This function instead closes
    each plot's OWN internals, in order, at the moment ITS OWNER is shutting down — no
    sweep over unrelated, possibly half-torn-down objects. Verified on py3.11.15
    offscreen: 360 open/close cycles, zero errors, zero surviving QMenus (see the ticket
    for the measured before/after).

    Order matters: ``ViewBox.setMenuEnabled(False)`` must run BEFORE ``PlotItem.close()``
    (which drops the item's own reference to its view box), and both must run BEFORE the
    container's own ``.close()`` (``GraphicsView.close()`` calls ``scene().clear()``,
    which invalidates any ``PlotItem``/``ViewBox`` still attached to the scene — closing
    the container first raises ``RuntimeError: Signal source has been deleted`` on the
    now-dead C++ objects underneath).

    Call this ONCE, when the container's owning screen is genuinely shutting down
    (``MainWindow.closeEvent``), never mid-session: afterwards every ``PlotItem`` in the
    container is closed (``ctrlMenu``/axes/view box gone) and the container itself is
    unusable. Never raises: a plot that fails to release its menus is a memory-growth
    nuisance, not a reason to abort the rest of teardown.

    Deliberately does NOT call ``ViewBox.close()`` (which additionally calls
    ``unregister()``, dropping the view box from pyqtgraph's class-level
    ``ViewBox.AllViews``/``NamedViews`` registries): both are ``weakref.WeakKeyDictionary``/
    ``WeakValueDictionary``, so a view box left registered there is not kept alive by it —
    a bookkeeping leftover, not a memory leak, and not worth the extra call.
    """
    if container is None:
        return
    plot_items = []
    try:
        get_plot_item = getattr(container, "getPlotItem", None)
        if callable(get_plot_item):
            pi = get_plot_item()
            if pi is not None:
                plot_items.append(pi)
        else:
            ci = getattr(container, "ci", None)
            items = getattr(ci, "items", None) if ci is not None else None
            if items:
                plot_items.extend(list(items.keys()))
    except Exception:               # noqa: BLE001 — see docstring
        pass
    for p in plot_items:
        try:
            vb = p.getViewBox()
            if vb is not None:
                vb.setMenuEnabled(False)   # drops the ViewBoxMenu (ViewBox._applyMenuEnabled)
        except Exception:               # noqa: BLE001 — see docstring
            pass
        try:
            p.close()                      # drops ctrlMenu + submenus, closes axes, drops vb ref
        except Exception:               # noqa: BLE001 — see docstring
            pass
    try:
        container.close()                  # GraphicsView.close(): scene().clear() etc.
    except Exception:               # noqa: BLE001 — see docstring
        pass

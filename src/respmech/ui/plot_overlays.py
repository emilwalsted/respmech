"""Shared pyqtgraph overlay helpers for the EMG views.

``add_flow_background`` superimposes a discrete, low-alpha flow silhouette BEHIND the EMG
traces so the user can visually relate EMG activity to the respiration cycle. It is purely
cosmetic: it never expands the EMG y-axis (added with ``ignoreBounds=True``), sits at the
deepest z-order, and ANY failure is a silent no-op — a background reference must never break
a render. Colours are read defensively from the plot palette (the minimal fallback palettes
used when theming is unavailable lack the channel keys), so it cannot KeyError.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# Behind the EMG traces (z=0) and the breath shading (z=-10) — the deepest background layer.
_FLOW_Z = -20
# Capture markers sit ABOVE the EMG traces (z=0), flow (-20) and breath shading (±10).
_CAPTURE_Z = 50
_FALLBACK_FLOW_HUE = (60, 130, 200)   # a neutral flow-blue if the palette has no 'flow' colour


def _flow_colours(pal):
    """(fill_rgba, line_rgba) derived from the palette's flow hue, muted + low-alpha so the
    trace reads as a background reference in both light and dark themes."""
    hue = None
    try:
        channels = pal.get("channels") if hasattr(pal, "get") else None
        if channels:
            hue = channels.get("flow")
    except Exception:              # noqa: BLE001 — any odd palette shape -> fallback
        hue = None
    if not hue or len(hue) < 3:
        hue = _FALLBACK_FLOW_HUE
    r, g, b = int(hue[0]), int(hue[1]), int(hue[2])
    return (r, g, b, 26), (r, g, b, 90)


def add_flow_background(plot_item, t, flow, pal, *, frac=0.8):
    """Draw ``flow`` as a faint, filled respiration silhouette behind the EMG traces on a
    pyqtgraph ``PlotItem``.

    ``t`` and ``flow`` are the FULL-LENGTH, untrimmed signals aligned sample-for-sample with
    the EMG the plot already holds (no trim offset — unlike breath overlays). The flow is
    scaled into ``frac`` of the plot's current EMG y-band and drawn about ZERO, so its SHAPE
    (the respiration cycle), not its amplitude, is what shows, and zero flow always coincides
    with the EMG's own zero line. Returns the created item, or None on any no-op
    (missing/empty/degenerate flow, or an unexpected error)."""
    try:
        t = np.asarray(t, dtype=float)
        flow = np.asarray(flow, dtype=float)
        n = int(min(t.size, flow.size))
        if n < 2:
            return None
        t = t[:n]
        flow = flow[:n]
        finite = np.isfinite(flow)
        if not finite.any():
            return None
        fmin = float(np.min(flow[finite]))
        fmax = float(np.max(flow[finite]))
        if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
            return None
        # Scale into a fraction of the EMG y-band. childrenBounds() reflects the EMG traces
        # already plotted; guard the (common) degenerate/empty case with a symmetric fallback
        # so a zero-trace panel still gets a sensible background scale.
        ylo, yhi = -1.0, 1.0
        try:
            ybounds = plot_item.getViewBox().childrenBounds()[1]
            if (ybounds is not None and np.isfinite(ybounds[0]) and np.isfinite(ybounds[1])
                    and ybounds[1] > ybounds[0]):
                ylo, yhi = float(ybounds[0]), float(ybounds[1])
        except Exception:                              # noqa: BLE001
            pass

        # Both the band and the flow are referenced to ZERO, deliberately.
        #
        # Using the MIDPOINT of the EMG's own min/max instead would only equal zero when the
        # traces happen to be symmetric, and one startup transient is enough to break that:
        # measured on real recordings the midpoint sat 20 % of the half-band BELOW zero on one
        # file and 16 % ABOVE on another, so the same respiration silhouette was drawn at a
        # different height in each -- and the two files could not be compared by eye.
        #
        # Min-max normalising the flow has the same flaw one level down: it puts zero flow at
        # (0-fmin)/(fmax-fmin), i.e. wherever the larger of the inspiratory/expiratory peaks
        # happens to be (0.54 and 0.52 on those two files, not 0.5). Scaling by the larger
        # absolute peak keeps the shape and pins zero flow to the zero line, so inspiration
        # and expiration read the right way round against the EMG.
        half = max(abs(ylo), abs(yhi)) * float(frac)
        if not np.isfinite(half) or half <= 0:
            half = 1.0
        scale = max(abs(fmin), abs(fmax))              # > 0: fmax > fmin was checked above
        y = np.where(finite, flow, 0.0) / scale * half  # NaNs -> the zero line, drawn flat

        fill_rgba, line_rgba = _flow_colours(pal)
        item = pg.PlotDataItem(t, y, pen=pg.mkPen(line_rgba, width=1),
                               fillLevel=0.0, fillBrush=pg.mkBrush(fill_rgba))
        item.setZValue(_FLOW_Z)
        # ignoreBounds=True: the flow must never widen the EMG autorange.
        plot_item.addItem(item, ignoreBounds=True)
        return item
    except Exception:                                  # noqa: BLE001 — cosmetic; never break a render
        return None


def _is_dark_bg(bg):
    try:
        if isinstance(bg, str) and bg.startswith("#"):
            h = bg.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        elif isinstance(bg, (tuple, list)) and len(bg) >= 3:
            r, g, b = bg[0], bg[1], bg[2]
        else:
            return False
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 128
    except Exception:              # noqa: BLE001
        return False


def _capture_red(pal):
    """A distinct, saturated red for the R-peak capture markers, brightened on a dark ground.
    NOT reused from any data-trace colour so it never collides with an EMG signal."""
    bg = None
    try:
        bg = pal.get("bg") if hasattr(pal, "get") else None
    except Exception:              # noqa: BLE001
        bg = None
    return (255, 96, 96) if _is_dark_bg(bg) else (200, 32, 32)


def add_ecg_capture_markers(plot_item, peak_times, pal, *, size=11, y_inset_frac=0.03, width=1):
    """Draw the ECG "capture" markers on a pyqtgraph ``PlotItem``: at each detected R-peak, a
    full-height red vertical line through the data + a red DOWNWARD (inverted) triangle pinned
    to the top of the view — like the pre-UI respmech PDF outputs.

    ``peak_times`` are R-peak times in SECONDS (the third return of ``core.emg.remove_ecg``),
    already on the same axis as each plot's ``t = arange(n)/fs``, so the SAME array is passed to
    the raw-capture plot AND every processed-channel plot. Two pyqtgraph items total regardless
    of beat count. The lines/triangle re-pin to the visible band on zoom via a self-disconnecting
    y-range slot. Mouse-transparent (never intercepts breath-toggle clicks) and ``ignoreBounds``
    (never widens the EMG autorange). A silent no-op on empty/missing peaks or any error."""
    try:
        pk = np.asarray(peak_times, dtype=float)
        pk = pk[np.isfinite(pk)]
        if pk.size == 0:
            return None
        vb = plot_item.getViewBox()
        if vb is None:
            return None
        red = _capture_red(pal)

        def _lines_xy(y0, y1):
            return np.repeat(pk, 3), np.tile([y0, y1, np.nan], pk.size)

        y0, y1 = vb.viewRange()[1]
        lx, ly = _lines_xy(y0, y1)
        line = pg.PlotDataItem(lx, ly, connect="finite", pen=pg.mkPen(red, width=width))
        line.setZValue(_CAPTURE_Z)
        marker = pg.ScatterPlotItem(x=pk, y=np.full(pk.size, y1 - (y1 - y0) * y_inset_frac),
                                    symbol="t", size=size, pxMode=True,
                                    pen=pg.mkPen(red), brush=pg.mkBrush(red))
        marker.setZValue(_CAPTURE_Z + 1)
        for it in (line, marker):
            try:
                it.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
                it.setAcceptHoverEvents(False)
            except Exception:      # noqa: BLE001 — pyqtgraph version drift
                pass
        plot_item.addItem(line, ignoreBounds=True)
        plot_item.addItem(marker, ignoreBounds=True)

        def _reposition(*_):
            try:
                if marker.getViewBox() is None:       # cleared -> self-remove the slot
                    vb.sigYRangeChanged.disconnect(_reposition)
                    return
                a, b = vb.viewRange()[1]
                nx, ny = _lines_xy(a, b)
                line.setData(x=nx, y=ny, connect="finite")
                marker.setData(x=pk, y=np.full(pk.size, b - (b - a) * y_inset_frac))
            except Exception:      # noqa: BLE001
                try:
                    vb.sigYRangeChanged.disconnect(_reposition)
                except Exception:  # noqa: BLE001
                    pass

        vb.sigYRangeChanged.connect(_reposition)
        return line, marker
    except Exception:              # noqa: BLE001 — cosmetic; never break a render
        return None

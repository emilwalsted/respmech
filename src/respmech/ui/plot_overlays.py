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

# Behind the EMG traces (z=0) and the breath shading (z=-10) — the deepest background layer.
_FLOW_Z = -20
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
    normalised into ``frac`` of the plot's current EMG y-band and centred, so its SHAPE (the
    respiration cycle), not its amplitude, is what shows. Returns the created item, or None on
    any no-op (missing/empty/degenerate flow, or an unexpected error)."""
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
        norm = (flow - fmin) / (fmax - fmin)          # -> [0, 1]
        norm[~finite] = 0.5                            # NaNs -> mid-band, drawn flat

        # Map into a centred fraction of the EMG y-band. childrenBounds() reflects the EMG
        # traces already plotted; guard the (common) degenerate/empty case with a symmetric
        # fallback so a zero-trace panel still gets a sensible background scale.
        ylo, yhi = -1.0, 1.0
        try:
            ybounds = plot_item.getViewBox().childrenBounds()[1]
            if (ybounds is not None and np.isfinite(ybounds[0]) and np.isfinite(ybounds[1])
                    and ybounds[1] > ybounds[0]):
                ylo, yhi = float(ybounds[0]), float(ybounds[1])
        except Exception:                              # noqa: BLE001
            pass
        mid = 0.5 * (ylo + yhi)
        half = 0.5 * (yhi - ylo) * float(frac)
        if not np.isfinite(half) or half <= 0:
            half = 1.0
        y = mid + (2.0 * norm - 1.0) * half

        fill_rgba, line_rgba = _flow_colours(pal)
        item = pg.PlotDataItem(t, y, pen=pg.mkPen(line_rgba, width=1),
                               fillLevel=mid, fillBrush=pg.mkBrush(fill_rgba))
        item.setZValue(_FLOW_Z)
        # ignoreBounds=True: the flow must never widen the EMG autorange.
        plot_item.addItem(item, ignoreBounds=True)
        return item
    except Exception:                                  # noqa: BLE001 — cosmetic; never break a render
        return None

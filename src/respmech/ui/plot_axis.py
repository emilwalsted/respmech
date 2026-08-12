"""A pyqtgraph axis whose major ticks thin out when the plot gets short.

WHY. pyqtgraph picks tick spacing off a ladder of "nice" numbers and then draws the top
level UNCONDITIONALLY -- ``generateDrawSpecs`` applies its crowding limits only from the
second level down (``if i > 0: ## always draw top level``). So a short plot prints its
major labels straight through one another. Measured on Preview's Detail channel at 43 px
of data area: six 13 px labels drawn 8.6 px apart, i.e. every pair overlapping by 4.4 px.

The two obvious levers do not work. ``setTickDensity`` is inert here because the interval
count is floored at ``max(2.25, ...)``; and even 2.25 intervals is not what comes out,
because the ladder step below 0.5 is 0.2 -- so a request for 2.25 intervals is served with
5. ``textFillLimits`` never applies, being a second-level-and-down rule.

Overriding ``tickSpacing`` is pyqtgraph's own documented hook for this. Here it takes the
spacing pyqtgraph chose and walks it UP the same ladder until the labels it implies have
room to sit side by side. It is re-evaluated on every draw with the current ``size``, so a
panel dragged taller gets its finer ticks straight back.

Extracted from Preview & QC's ``_FitAxis`` (ticket B05): ``ColumnStack``'s own rows are
short too (74 px in the channel-assignment dialog) and needed exactly this thinning,
without ``_FitAxis``'s other behaviour (SI-prefix suppression, the multi-wording axis
label), which is Preview-specific. ``_FitAxis`` now subclasses this base, unchanged in
behaviour, so Preview comes out the other side of the extraction identical.
"""
from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtGui import QFontMetrics


class MinPitchAxis(pg.AxisItem):
    """A value axis whose major ticks never sit closer than one label's own height."""

    #: extra pixels demanded between two labels, on top of the text height itself
    AIR = 3

    def tickSpacing(self, minVal, maxVal, size):
        levels = super().tickSpacing(minVal, maxVal, size)
        span = abs(maxVal - minVal)
        if not levels or self.orientation not in ("left", "right") or span <= 0 or size <= 0:
            return levels
        pitch = self._min_pitch()
        if pitch <= 0:
            return levels
        need = pitch * span / size          # the label pitch expressed in view units
        major = levels[0][0]
        # Bounded, and it stops if the ladder cannot climb. _next_nice is exact only in the
        # normal double range: below ~3.3e-311 its decade term underflows to 0.0, so it
        # returns a value no larger than its input (and divides by zero at 5e-324). No real
        # signal has a span that small — but an unbounded `while` on a function that can stop
        # increasing is a hang, not a wrong tick, so it is bounded rather than reasoned about.
        for _ in range(64):
            if major >= need or major >= span:
                break
            nxt = _next_nice(major)
            if not nxt > major:
                break
            major = nxt
        if major == levels[0][0]:
            return levels
        # keep only the finer levels that are still finer than the new major
        return [(major, levels[0][1])] + [lv for lv in levels[1:] if lv[0] < major]

    def _min_pitch(self):
        """Height one tick label needs, in pixels.

        pyqtgraph measures its own strings with ``boundingRect(...)`` and then shrinks the
        result by 0.8, so that same factor is applied here — otherwise this asks for more
        room than the renderer will actually use and the axis thins one step too eagerly.
        """
        font = self.style.get("tickFont") or self.font()
        try:
            return QFontMetrics(font).height() * 0.8 + self.AIR
        except Exception:                    # pragma: no cover - falls back to no thinning
            return 0


def _next_nice(v):
    """The next value up pyqtgraph's 1 / 2 / 5 ladder (0.2 -> 0.5 -> 1 -> 2 -> 5 -> 10)."""
    if v <= 0:
        return v
    dec = 10.0 ** math.floor(math.log10(v))
    if dec <= 0:                 # subnormal input: the decade underflowed to 0.0
        return v
    m = v / dec
    if m < 1.999:
        return 2.0 * dec
    if m < 4.999:
        return 5.0 * dec
    return 10.0 * dec

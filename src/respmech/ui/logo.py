"""The RespMech logo — a generated "signal tile" mark: an azure graph-paper tile
with a white respiratory flow curve and an orange diaphragm-EMG burst. It nods to
the site's data/graphs aesthetic, uses the respmech.dk palette (azure #03A9F4 /
orange #FF9800), and scales crisply from a 16 px favicon to the splash.

Used for the splash badge, the application/window icon, and written to
``assets/respmech_logo.svg`` as a shareable asset.
"""
from __future__ import annotations

import math

_AZ_HI = "#29B6F6"      # tile gradient (top-left)
_AZ_LO = "#0277BD"      # tile gradient (bottom-right)
_ORANGE = "#FF9800"     # EMG burst
_WHITE = "#FFFFFF"


def _wave(x0, x1, y0, amp, cyc, ph, n=150):
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        y = y0 + amp * math.sin(2.0 * math.pi * cyc * i / n + ph)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _burst(x0, x1, y0, amp, cyc, n=110):
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        env = math.sin(math.pi * i / n)
        y = y0 + amp * env * math.sin(2.0 * math.pi * cyc * i / n)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def build_logo_svg(size: int = 512) -> str:
    """Return the logo mark as an SVG string (viewBox 0..512, scaled to ``size``)."""
    grid = []
    for k in range(1, 6):
        v = 32 + 448 * k / 6
        grid.append(f'<line x1="{v:.0f}" y1="64" x2="{v:.0f}" y2="448" '
                    f'stroke="#FFFFFF" stroke-opacity="0.11" stroke-width="2"/>')
        grid.append(f'<line x1="64" y1="{v:.0f}" x2="448" y2="{v:.0f}" '
                    f'stroke="#FFFFFF" stroke-opacity="0.11" stroke-width="2"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="{size}" height="{size}">
  <defs>
    <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{_AZ_HI}"/>
      <stop offset="1" stop-color="{_AZ_LO}"/>
    </linearGradient>
    <linearGradient id="gloss" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.18"/>
      <stop offset="0.5" stop-color="#FFFFFF" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect x="32" y="32" width="448" height="448" rx="112" fill="url(#tile)"/>
  <g>{''.join(grid)}</g>
  <line x1="64" y1="256" x2="448" y2="256" stroke="#FFFFFF" stroke-opacity="0.20" stroke-width="2.5"/>
  <polyline points="{_wave(76, 436, 256, 66, 1.6, 0.5)}" fill="none" stroke="{_WHITE}"
            stroke-width="19" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline points="{_burst(200, 328, 256, 46, 6.5)}" fill="none" stroke="{_ORANGE}"
            stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="432" cy="{256 + 66 * math.sin(2 * math.pi * 1.6 + 0.5):.0f}" r="12" fill="{_ORANGE}"/>
  <rect x="32" y="32" width="448" height="448" rx="112" fill="url(#gloss)"/>
</svg>'''


def logo_pixmap(size: int = 512):
    """Render the logo to a transparent QPixmap, or None if Qt SVG is unavailable."""
    try:
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtGui import QPixmap, QPainter
        from PySide6.QtCore import QByteArray, QRectF, Qt
    except Exception:                                    # pragma: no cover
        return None
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    QSvgRenderer(QByteArray(build_logo_svg(size).encode("utf-8"))).render(
        painter, QRectF(0, 0, size, size))
    painter.end()
    return pm


def app_icon():
    """A multi-resolution QIcon for the window/app, or None if unavailable."""
    try:
        from PySide6.QtGui import QIcon
    except Exception:                                    # pragma: no cover
        return None
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256, 512):
        pm = logo_pixmap(s)
        if pm is not None:
            icon.addPixmap(pm)
    return icon if not icon.isNull() else None

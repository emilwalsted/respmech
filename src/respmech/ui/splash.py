"""Startup splash screen.

Brand-matched to www.respmech.dk: the site's azure (#03A9F4) and orange (#FF9800)
accents over a dark slate hero, the site's tagline, and the real RespMech "RM"
monogram (bundled at ``ui/assets/respmech_logo.png``) on a white badge. The
respiratory-signal motif with a red EMG burst nods to the site's lab-hardware hero.

The base graphic is a generated SVG (crisp at any DPI); ``make_splash(app)`` renders
it to a pixmap, composites the logo, and returns a ``QSplashScreen`` (or ``None`` if
Qt SVG support is unavailable), which the entry point shows while the window builds.
"""
from __future__ import annotations

import math

# Palette lifted from the respmech.dk theme + the logo.
_AZURE = "#03A9F4"
_AZURE_SOFT = "#5FC7FB"
_ORANGE = "#FF9800"
_RED = "#E30613"          # the logo's "M" red
_SLATE_TOP = "#22303F"
_SLATE_BOT = "#151C24"
_INK = "#0B0E13"
_TEXT = "#F1F6FB"
_MUTED = "#93A4B7"

# font stack — Muli/Mulish is the site face; Avenir Next is a close macOS match
_FONT = "Mulish, Muli, Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif"

_TAGLINE = "Human respiratory physiology analysis made easier"
_SUBTAGLINE = "Respiratory mechanics · work of breathing · diaphragm EMG"

# white logo badge geometry (SVG/viewBox coords) — shared with make_splash
_BADGE_X, _BADGE_Y, _BADGE_S = 48, 92, 120
_LOGO_PAD = 15


def _wave(width, y0, amp, cycles, phase, n=260):
    pts = []
    for i in range(n + 1):
        x = width * i / n
        y = y0 + amp * math.sin(2.0 * math.pi * cycles * i / n + phase)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _emg_burst(x0, x1, y0, amp, cycles, n=200):
    pts = []
    span = x1 - x0
    for i in range(n + 1):
        x = x0 + span * i / n
        env = math.sin(math.pi * i / n)
        y = y0 + amp * env * math.sin(2.0 * math.pi * cycles * i / n) \
            * (0.6 + 0.4 * math.sin(11.0 * i / n))
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def build_splash_svg(width: int = 780, height: int = 460, version: str | None = None) -> str:
    """Return the splash background as an SVG string (everything except the logo,
    which ``make_splash`` composites onto the white badge)."""
    if version is None:
        try:
            from respmech import __version__ as version
        except Exception:                                # pragma: no cover
            version = ""
    w, h = width, height
    bx, by, bs = _BADGE_X, _BADGE_Y, _BADGE_S

    # respiratory-signal motif: faint azure layers + a bright flow curve + a red EMG burst
    waves = []
    for y0, amp, cyc, ph, op, sw in [
        (318, 30, 3.1, 0.4, 0.10, 2.0),
        (344, 40, 2.3, 1.7, 0.13, 2.0),
        (306, 22, 4.2, 3.0, 0.09, 1.6),
        (366, 26, 2.8, 0.9, 0.11, 1.6),
    ]:
        waves.append(
            f'<polyline points="{_wave(w, y0, amp, cyc, ph)}" fill="none" '
            f'stroke="{_AZURE}" stroke-width="{sw}" stroke-opacity="{op}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')
    waves.append(                                        # subtle orange lane
        f'<polyline points="{_wave(w, 388, 16, 3.6, 2.1)}" fill="none" '
        f'stroke="{_ORANGE}" stroke-width="1.5" stroke-opacity="0.12" '
        f'stroke-linecap="round" stroke-linejoin="round"/>')
    waves.append(                                        # bright flow curve
        f'<polyline points="{_wave(w, 342, 34, 2.3, 1.7)}" fill="none" '
        f'stroke="{_AZURE_SOFT}" stroke-width="2.6" stroke-opacity="0.55" '
        f'stroke-linecap="round" stroke-linejoin="round"/>')
    waves.append(                                        # red EMG burst (echoes the logo red)
        f'<polyline points="{_emg_burst(0.36 * w, 0.60 * w, 306, 30, 34)}" fill="none" '
        f'stroke="{_RED}" stroke-width="1.6" stroke-opacity="0.72" '
        f'stroke-linecap="round" stroke-linejoin="round"/>')

    footer_y = h - 34
    credit = f"v{version} · Emil Walsted" if version else "Emil Walsted"

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{_SLATE_TOP}"/>
      <stop offset="1" stop-color="{_SLATE_BOT}"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.2" cy="0.3" r="0.7">
      <stop offset="0" stop-color="{_AZURE}" stop-opacity="0.20"/>
      <stop offset="1" stop-color="{_AZURE}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="veil" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{_INK}" stop-opacity="0"/>
      <stop offset="0.55" stop-color="{_INK}" stop-opacity="0.32"/>
      <stop offset="1" stop-color="{_INK}" stop-opacity="0.72"/>
    </linearGradient>
  </defs>

  <rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#bg)"/>
  <rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#glow)"/>
  <g>{"".join(waves)}</g>
  <rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#veil)"/>

  <!-- white badge for the RM logo (logo composited on top by make_splash) -->
  <rect x="{bx + 3}" y="{by + 5}" width="{bs}" height="{bs}" rx="20" fill="{_INK}" fill-opacity="0.35"/>
  <rect x="{bx}" y="{by}" width="{bs}" height="{bs}" rx="20" fill="#FFFFFF"/>
  <rect x="{bx}" y="{by}" width="{bs}" height="{bs}" rx="20" fill="none"
        stroke="{_AZURE}" stroke-opacity="0.35" stroke-width="1.5"/>
  <text x="{bx + bs / 2}" y="{by + bs / 2 + 14}" font-family="{_FONT}" font-size="44"
        font-weight="800" fill="#0E1116" text-anchor="middle" opacity="0.12">RM</text>

  <text x="196" y="150" font-family="{_FONT}" font-size="60" font-weight="800"
        letter-spacing="-1.0" fill="{_TEXT}">RespMech</text>
  <rect x="198" y="166" width="56" height="5" rx="2.5" fill="{_ORANGE}"/>
  <text x="198" y="200" font-family="{_FONT}" font-size="19" font-weight="600"
        letter-spacing="0.2" fill="{_AZURE}">{_TAGLINE}</text>
  <text x="198" y="226" font-family="{_FONT}" font-size="13.5" letter-spacing="0.3"
        fill="{_MUTED}">{_SUBTAGLINE}</text>

  <line x1="42" y1="{footer_y - 16}" x2="{w - 42}" y2="{footer_y - 16}"
        stroke="{_AZURE}" stroke-opacity="0.18" stroke-width="1"/>
  <text x="42" y="{footer_y + 4}" font-family="{_FONT}" font-size="13"
        fill="{_MUTED}">{credit}</text>
  <text x="{w - 42}" y="{footer_y + 4}" font-family="{_FONT}" font-size="13"
        fill="{_AZURE_SOFT}" text-anchor="end">www.respmech.dk</text>
</svg>'''


def _load_logo_pixmap():
    """Load the bundled RM logo as a QPixmap, or None if unavailable."""
    try:
        from importlib.resources import files
        from PySide6.QtGui import QPixmap
        data = (files("respmech.ui") / "assets" / "respmech_logo.png").read_bytes()
        pm = QPixmap()
        pm.loadFromData(data)
        return pm if not pm.isNull() else None
    except Exception:                                    # pragma: no cover
        return None


def make_splash(app=None, width: int = 780, height: int = 460):
    """Render the splash SVG to a high-DPI pixmap, composite the RM logo onto the
    badge, and return a ``QSplashScreen``. Returns ``None`` if Qt SVG support is
    unavailable (the app then starts without a splash rather than failing)."""
    try:
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtGui import QPixmap, QPainter
        from PySide6.QtCore import QByteArray, QRectF, Qt
        from PySide6.QtWidgets import QSplashScreen
    except Exception:                                    # pragma: no cover
        return None

    svg = build_splash_svg(width, height)
    dpr = 2.0
    try:
        screen = app.primaryScreen() if app is not None else None
        if screen is not None and screen.devicePixelRatio() > 0:
            dpr = max(1.0, min(4.0, float(screen.devicePixelRatio())))
    except Exception:                                    # pragma: no cover
        pass

    pm = QPixmap(int(width * dpr), int(height * dpr))
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.scale(dpr, dpr)                              # draw in viewBox coordinates
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter, QRectF(0, 0, width, height))

    logo = _load_logo_pixmap()
    if logo is not None:
        d = _BADGE_S - 2 * _LOGO_PAD
        target = QRectF(_BADGE_X + _LOGO_PAD, _BADGE_Y + _LOGO_PAD, d, d)
        painter.drawPixmap(target, logo, QRectF(logo.rect()))
    painter.end()
    pm.setDevicePixelRatio(dpr)

    splash = QSplashScreen(pm)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    return splash

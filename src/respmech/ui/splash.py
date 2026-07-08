"""Startup splash screen.

Brand-matched to www.respmech.dk: the site's azure (#03A9F4) / orange (#FF9800)
accents over a dark slate hero, the site's tagline, the new RespMech "signal-tile"
logo, and — echoing the site's most representative background — a softly **blurred
backdrop of graphs, code and calculations** (faint plots, a Bland-Altman-style
scatter, and scattered analysis/code tokens).

The graphic is layered in ``make_splash(app)``: slate base → blurred data texture →
legibility veil + text → the logo tile. It returns a ``QSplashScreen`` (or ``None``
if Qt SVG support is unavailable), shown while the main window is built.
"""
from __future__ import annotations

import math

_AZURE = "#03A9F4"
_AZURE_SOFT = "#5FC7FB"
_ORANGE = "#FF9800"
_SLATE_TOP = "#22303F"
_SLATE_BOT = "#151C24"
_INK = "#0B0E13"
_TEXT = "#F1F6FB"
_MUTED = "#93A4B7"
_DATA = "#8FB6D6"        # faint data-texture ink

_FONT = "Mulish, Muli, Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif"
_MONO = "SFMono-Regular, Menlo, Consolas, monospace"

_TAGLINE = "Human respiratory physiology analysis made easier"
_SUBTAGLINE = "Respiratory mechanics · work of breathing · diaphragm EMG"

# logo-tile placement (SVG/viewBox coords) — the tile is self-contained
_LOGO_X, _LOGO_Y, _LOGO_S = 48, 96, 128

# scattered analysis/code tokens for the blurred backdrop ("graphs, code, calculations")
_TOKENS = [
    "∫ P·dV", "WOB_insp", "prop_decrease = 0.5", "Poes", "Pgas", "Pdi",
    "def compute_breath():", "EMG_rms", "SampEn", "0.94  [0.90, 0.96]",
    "cmH₂O", "Campbell", "fidelity ≥ 0.80", "−0.47", "np.trapezoid(P, V)",
    "breath[i].ignored", "Bland–Altman", "n_fft = 256", "ΔSNR", "1.00",
]


def _wave(x0, x1, y0, amp, cyc, ph, n=220):
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        y = y0 + amp * math.sin(2.0 * math.pi * cyc * i / n + ph)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _emg_di(x0, x1, mid, amp, cyc, ph, carrier, n=720):
    """A clean diaphragm-EMG trace whose baseline sits on the flow's 0-flow line
    (``mid`` — halfway between the flow peak and nadir). It is flat on that line,
    and each burst spans exactly from a flow PEAK to the following NADIR — the
    descending half where ``-cos(2π·cyc·x + ph)`` is positive (0 at the peak,
    maximal at the mid-descent, 0 at the nadir) — oscillating symmetrically about
    the 0-flow line. Assumes the flow is ``mid - A·sin(2π·cyc·x + ph)`` (peak at
    the top), so the burst starts at the peak and ends at the nadir."""
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        theta = 2.0 * math.pi * cyc * i / n + ph
        env = max(0.0, -math.cos(theta)) ** 0.9         # peak -> nadir gate
        y = mid - amp * env * math.sin(2.0 * math.pi * carrier * i / n)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _base_svg(w, h):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{_SLATE_TOP}"/><stop offset="1" stop-color="{_SLATE_BOT}"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.2" cy="0.3" r="0.75">
      <stop offset="0" stop-color="{_AZURE}" stop-opacity="0.20"/>
      <stop offset="1" stop-color="{_AZURE}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#bg)"/>
  <rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#glow)"/>
</svg>'''


def _texture_svg(w, h):
    """A dense 'graphs + code + calculations' layer (blurred by make_splash)."""
    parts = []
    # faint plot curves
    for y0, amp, cyc, ph, op in [(150, 40, 2.2, 0.3, 0.36), (250, 60, 1.6, 1.4, 0.32),
                                 (350, 34, 3.1, 2.6, 0.30), (410, 22, 4.0, 0.8, 0.26)]:
        parts.append(f'<polyline points="{_wave(0, w, y0, amp, cyc, ph)}" fill="none" '
                     f'stroke="{_DATA}" stroke-width="3" stroke-opacity="{op}"/>')
    # a Bland-Altman-style scatter band (deterministic pseudo-scatter)
    for i in range(70):
        x = 40 + (w - 80) * ((i * 97) % 100) / 100.0
        y = 300 + 60 * math.sin(i * 1.7) * (0.4 + 0.6 * (((i * 53) % 100) / 100.0))
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="2.6" '
                     f'fill="{_DATA}" fill-opacity="0.36"/>')
    parts.append(f'<line x1="40" y1="300" x2="{w-40}" y2="300" stroke="{_DATA}" '
                 f'stroke-opacity="0.32" stroke-width="1.5"/>')
    # scattered analysis / code tokens
    for i, tok in enumerate(_TOKENS):
        x = 30 + ((i * 149) % (w - 120))
        y = 70 + ((i * 83) % (h - 120))
        rot = -8 + (i * 37) % 16
        parts.append(f'<text x="{x}" y="{y}" font-family="{_MONO}" font-size="16" '
                     f'fill="{_DATA}" fill-opacity="0.46" '
                     f'transform="rotate({rot} {x} {y})">{tok}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">{"".join(parts)}</svg>')


def build_splash_svg(width: int = 780, height: int = 460, version: str | None = None) -> str:
    """The foreground content layer (transparent background): a legibility veil, a
    bright foreground flow curve, and the wordmark/tagline/footer. The base, the
    blurred data texture and the logo tile are composited around it in make_splash."""
    if version is None:
        try:
            from respmech import __version__ as version
        except Exception:                                # pragma: no cover
            version = ""
    w, h = width, height
    footer_y = h - 34
    credit = f"v{version} · Emil Walsted" if version else "Emil Walsted"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <linearGradient id="veil" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{_INK}" stop-opacity="0.20"/>
      <stop offset="0.5" stop-color="{_INK}" stop-opacity="0.36"/>
      <stop offset="1" stop-color="{_INK}" stop-opacity="0.72"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#veil)"/>
  <!-- wavy ventilation / flow curve (blue): peak at top, nadir at bottom -->
  <polyline points="{_wave(0, w, 352, -32, 2.3, 1.6)}" fill="none" stroke="{_AZURE_SOFT}"
            stroke-width="3" stroke-opacity="0.7" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- diaphragm EMG (orange): baseline on the 0-flow line, burst spans peak->nadir -->
  <polyline points="{_emg_di(0, w, 352, 22, 2.3, 1.6, 44)}" fill="none" stroke="{_ORANGE}"
            stroke-width="1.6" stroke-opacity="0.9" stroke-linecap="round" stroke-linejoin="round"/>

  <text x="204" y="152" font-family="{_FONT}" font-size="60" font-weight="800"
        letter-spacing="-1.0" fill="{_TEXT}">RespMech</text>
  <rect x="206" y="168" width="56" height="5" rx="2.5" fill="{_ORANGE}"/>
  <text x="206" y="202" font-family="{_FONT}" font-size="19" font-weight="600"
        letter-spacing="0.2" fill="{_AZURE}">{_TAGLINE}</text>
  <text x="206" y="228" font-family="{_FONT}" font-size="13.5" letter-spacing="0.3"
        fill="{_MUTED}">{_SUBTAGLINE}</text>

  <line x1="42" y1="{footer_y - 16}" x2="{w - 42}" y2="{footer_y - 16}"
        stroke="{_AZURE}" stroke-opacity="0.18" stroke-width="1"/>
  <text x="42" y="{footer_y + 4}" font-family="{_FONT}" font-size="13" fill="{_MUTED}">{credit}</text>
  <text x="{w - 42}" y="{footer_y + 4}" font-family="{_FONT}" font-size="13"
        fill="{_AZURE_SOFT}" text-anchor="end">www.respmech.dk</text>
</svg>'''


def _blur(pixmap, w, h, factor=6):
    from PySide6.QtCore import Qt
    sw, sh = max(1, w // factor), max(1, h // factor)
    small = pixmap.scaled(sw, sh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    return small.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)


def _load_logo_pixmap(size=256):
    """The RespMech logo tile as a QPixmap (generated), or None if unavailable."""
    try:
        from respmech.ui.logo import logo_pixmap
        return logo_pixmap(size)
    except Exception:                                    # pragma: no cover
        return None


def make_splash(app=None, width: int = 780, height: int = 460):
    """Render the layered splash and return a ``QSplashScreen`` (or ``None`` if Qt
    SVG support is unavailable)."""
    try:
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtGui import QPixmap, QPainter
        from PySide6.QtCore import QByteArray, QRectF, Qt
        from PySide6.QtWidgets import QSplashScreen
    except Exception:                                    # pragma: no cover
        return None

    dpr = 2.0
    try:
        screen = app.primaryScreen() if app is not None else None
        if screen is not None and screen.devicePixelRatio() > 0:
            dpr = max(1.0, min(4.0, float(screen.devicePixelRatio())))
    except Exception:                                    # pragma: no cover
        pass

    def _render(svg):
        p = QPixmap(width, height)
        p.fill(Qt.transparent)
        pt = QPainter(p)
        pt.setRenderHint(QPainter.Antialiasing, True)
        QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(pt, QRectF(0, 0, width, height))
        pt.end()
        return p

    texture = _blur(_render(_texture_svg(width, height)), width, height)

    pm = QPixmap(int(width * dpr), int(height * dpr))
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.scale(dpr, dpr)
    QSvgRenderer(QByteArray(_base_svg(width, height).encode())).render(painter, QRectF(0, 0, width, height))
    painter.setOpacity(0.72)
    painter.drawPixmap(QRectF(0, 0, width, height), texture, QRectF(texture.rect()))
    painter.setOpacity(1.0)
    QSvgRenderer(QByteArray(build_splash_svg(width, height).encode())).render(painter, QRectF(0, 0, width, height))

    logo = _load_logo_pixmap(256)
    if logo is not None:
        painter.drawPixmap(QRectF(_LOGO_X, _LOGO_Y, _LOGO_S, _LOGO_S), logo, QRectF(logo.rect()))
    painter.end()
    pm.setDevicePixelRatio(dpr)

    splash = QSplashScreen(pm)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    return splash

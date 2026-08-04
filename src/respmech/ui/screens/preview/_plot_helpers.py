"""Axis and plot styling helpers shared by every Preview & QC sub-tab: the
SI-prefix axis, the compact fit-axis, pen/palette helpers and the check-icon cache.
Split out of preview_screen.py (ticket A02); moved verbatim, not rewritten."""

from __future__ import annotations

import copy
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QObject, QSize, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui import plot_perf
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers
from respmech.ui import wheel as _wheel
from respmech.ui.flow_layout import (FlowLayout, cluster as _cluster,
                                     elide as _elide, install_flow as _install_flow)
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


class SciAxis(pg.AxisItem):
    """A left axis that centres its (two-line) channel label and, when pyqtgraph auto-scales
    the ticks, annotates the scale as ``·10⁻³`` rather than pyqtgraph's default ``(x0.001)``.

    The name and unit are set separately via ``set_channel_label`` so the two can stack on
    two centred lines and so the scale annotation can be inserted next to the unit.
    """

    def __init__(self, *a, **k):
        self._name = ""
        self._unit = ""             # set before super(): its __init__ calls labelString()
        super().__init__(*a, **k)

    def set_channel_label(self, name, unit=""):
        self._name, self._unit = name, unit
        self.labelUnits = unit          # keep pyqtgraph's own SI-scaling of the tick values
        self.label.setHtml(self.labelString())
        # labelText is ufarlig at sætte her: labelString() (over) bygger sin HTML af
        # _name/_unit og ignorerer labelText, så dette ændrer intet i selve aksen. Men
        # pyqtgraph's egen setLabel() ville have sat den, og krydshårets aflæsning
        # (_on_mech_mouse_moved) læser netop labelText for at navngive kanalen.
        self.labelText = name
        # pyqtgraph's AxisItem.__init__ efterlader label skjult; almindelig setLabel()
        # kalder showLabel() internt, men den kalder vi bevidst ikke (den ville overskrive
        # labelString()'ens to-linjers HTML og ·10⁻³-annotation). Et tomt navn skal
        # stadig skjule etiketten.
        self.showLabel(bool(name))
        self._adjustSize() if hasattr(self, "_adjustSize") else None

    def _scale_annotation(self):
        scale = getattr(self, "autoSIPrefixScale", 1.0)
        if not self.autoSIPrefix or scale == 1.0:
            return ""
        import math
        exp = int(round(math.log10(1.0 / scale)))   # displayed = value·10^exp
        return "·10" + str(exp).translate(_SUP)

    def labelString(self):
        scale = self._scale_annotation()
        if scale and self._unit:
            second = f"({scale} {self._unit})"
        elif scale:
            second = f"({scale})"
        elif self._unit:
            second = f"({self._unit})"
        else:
            second = ""
        inner = self._name + (f"<br>{second}" if second else "")
        style = ";".join(f"{k}: {v}" for k, v in self.labelStyle.items())
        return f"<span style='{style}'><div style='text-align:center'>{inner}</div></span>"

# channel -> (axis label with units, pen colour by physiological meaning)
_CHANNELS = [
    ("flow", "Flow (L/s)", (44, 110, 155)),
    ("volume", "Volume (L)", (31, 122, 77)),
    ("poes", "Poes (cmH₂O)", (180, 50, 42)),
    ("pgas", "Pgas (cmH₂O)", (183, 121, 31)),
    ("pdi", "Pdi (cmH₂O)", (125, 91, 166)),
]
# distinct pens for EMG channels in the raw / result views
_EMG_PENS = [(44, 110, 155), (180, 50, 42), (31, 122, 77), (183, 121, 31),
             (125, 91, 166), (14, 124, 123), (180, 80, 122), (92, 107, 122)]


def _pen(colour, width=1):
    return pg.mkPen(colour, width=width)


def _rms_envelope(x, win):
    """Sliding-window RMS envelope of a 1-D signal (the quantity each breath's EMG
    amplitude is taken from), same length as ``x``. Front-padded so it aligns in time."""
    x = np.asarray(x, dtype=float)
    win = max(1, int(win))
    if x.size == 0 or win >= x.size:
        return np.sqrt(np.maximum(np.mean(x * x) if x.size else 0.0, 0.0)) * np.ones_like(x)
    c = np.cumsum(np.insert(x * x, 0, 0.0))
    ma = (c[win:] - c[:-win]) / win
    env = np.sqrt(np.maximum(ma, 0.0))
    return np.concatenate([np.full(x.size - env.size, env[0]), env])


# Fallback plot palette used only if the theme module failed to import (it never
# raises in practice); mirrors theme._PLOT_LIGHT so light behaviour is unchanged.
_FALLBACK_PAL = {
    "bg": "#FCFDFE", "fg": "#33404D", "grid_alpha": 0.15,
    "channels": {k: c for k, _l, c in _CHANNELS},
    "emg_cycle": list(_EMG_PENS),
    "breath_incl_brush": (44, 110, 155, 32), "breath_excl_brush": (180, 50, 42, 70),
    "breath_incl_label": (90, 107, 122), "breath_excl_label": (180, 50, 42),
    "separator": (150, 165, 180), "noise_region": (44, 110, 155, 45),
    "raw_trace": (150, 165, 180), "noise_trace": (90, 150, 200),
    "legend_bg": (255, 255, 255, 0),
    "mpl_bg": "#FFFFFF",
    "mpl_accent": "#2C6E9B", "mpl_ok": "#1F7A4D", "mpl_warn": "#B7791F",
    "mpl_error": "#B4322A", "mpl_muted": "0.6",
    "mpl_loop": "0.55", "mpl_zeroline": "0.85", "mpl_target": "0.35",
}


def _plot_pal():
    """The active theme's plot colour table (light before any theme is applied)."""
    if _theme is not None:
        try:
            return _theme.plot_palette()
        except Exception:  # pragma: no cover - defensive
            pass
    return _FALLBACK_PAL


_CHECK_ICON_PATH = None


def _check_icon_url(colour: str = "white"):
    """A checkmark PNG (generated once per colour, cached in the temp dir) used as the
    ``:checked`` image of the colour-filled result-channel checkboxes, so the tick sits
    inside the channel's coloured box. Qt QSS ``url()`` needs a real file, not a data URI.
    See :func:`_tick_colour` for why the colour is decided per channel, not per theme."""
    global _CHECK_ICON_PATH
    if _CHECK_ICON_PATH is None:
        _CHECK_ICON_PATH = {}
    if colour not in _CHECK_ICON_PATH:
        import tempfile
        from PySide6.QtGui import QImage, QPainter, QPen, QColor
        from PySide6.QtCore import QPointF
        img = QImage(24, 24, QImage.Format_ARGB32); img.fill(Qt.transparent)
        p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(colour)); pen.setWidth(3)
        pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); p.setPen(pen)
        p.drawPolyline([QPointF(5, 12), QPointF(10, 18), QPointF(19, 6)]); p.end()
        path = os.path.join(tempfile.gettempdir(),
                            f"respmech_check_{colour.lstrip('#')}.png")
        img.save(path)
        _CHECK_ICON_PATH[colour] = path.replace("\\", "/")
    return _CHECK_ICON_PATH[colour]


def _tick_colour(rgb) -> str:
    """Black or white — whichever actually contrasts better against the fill ``rgb``.

    The tick is painted INSIDE the channel's own colour box, and the dark theme brightens
    every channel colour (flow goes (44,110,155) -> (96,172,226)), so a fixed white tick that
    reads cleanly in light mode turns into a smear on the dark theme's pale fills.

    It picks by measuring WCAG contrast both ways rather than by thresholding brightness: a
    threshold is a guess about where the crossover sits, and the first one tried here (0.6 of
    perceived brightness) left the dark theme's salmon EMG colour at 2.90:1 — under the 3.0
    floor for a graphical element. Measuring cannot be wrong by construction, and it keeps
    choosing correctly if the palette is ever retuned."""
    def _rel_lum(c):
        out = []
        for v in c[:3]:
            s = float(v) / 255.0
            out.append(s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4)
        return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]

    def _contrast(a, b):
        la, lb = _rel_lum(a) + 0.05, _rel_lum(b) + 0.05
        return max(la, lb) / min(la, lb)

    dark, light = (12, 17, 22), (255, 255, 255)
    return "#0C1116" if _contrast(dark, rgb) >= _contrast(light, rgb) else "white"


class _FitAxis(pg.AxisItem):
    """A value axis whose major ticks thin out when the plot gets short.

    pyqtgraph picks tick spacing off a ladder of "nice" numbers and then draws the top
    level UNCONDITIONALLY — ``generateDrawSpecs`` applies its crowding limits only from the
    second level down (``if i > 0: ## always draw top level``). So a short plot prints its
    major labels straight through one another. Measured on the Detail channel at 43 px of
    data area: six 13 px labels drawn 8.6 px apart, i.e. every pair overlapping by 4.4 px.

    The two obvious levers do not work. ``setTickDensity`` is inert here because the
    interval count is floored at ``max(2.25, ...)``; and even 2.25 intervals is not what
    comes out, because the ladder step below 0.5 is 0.2 — so a request for 2.25 intervals
    is served with 5. ``textFillLimits`` never applies, being a second-level-and-down rule.

    Overriding ``tickSpacing`` is pyqtgraph's own documented hook for this. Here it takes
    the spacing pyqtgraph chose and walks it UP the same ladder until the labels it implies
    have room to sit side by side. It is re-evaluated on every draw with the current
    ``size``, so a panel dragged taller gets its finer ticks straight back.
    """

    #: extra pixels demanded between two labels, on top of the text height itself
    AIR = 3

    # class-level, not set in __init__: AxisItem's own constructor calls setRange, which
    # reaches resizeEvent below long before an __init__ body would have run
    _variants = ()
    _picking = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # No SI multiplier on these axes. pyqtgraph states the factor INSIDE the axis label
        # ("EMG (a.u.) (x0.001)"), which is the single biggest reason the label would not fit
        # a short panel — the "(x0.001)" alone measures ~55 px of the 63 px available. The
        # multiplier buys nothing here either: this EMG range is -0.43..0.64, so unscaled
        # ticks read -0.4 / 0.1 / 0.6, every bit as short as the scaled -400 / 100 / 600.
        # Disabling it up front also keeps autoSIPrefixScale pinned at 1.0 for good, since
        # setRange only recomputes the scale while the prefix is enabled.
        self.enableAutoSIPrefix(False)

    def set_label_variants(self, *variants):
        """Offer several wordings of the axis label, longest first.

        The axis label is drawn rotated inside the axis' own rect, so on a short plot it is
        simply clipped — measured here at 81 px of "EMG (a.u.)" (the SI multiplier pyqtgraph
        appends is part of it) trying to fit 63 px, which ate the leading E. Both EMG views
        were clipped; only one showed it, because the other's overflow happened to fall into
        a taller x-axis before reaching the widget edge. Rather than pick a shorter label
        permanently and lose the unit on big screens, the longest wording that FITS is used,
        re-chosen whenever the axis is resized.

        A variant of ``None`` means "no label at all" — the last resort for a panel too short
        for any rotated text at all. It is only safe because these axes carry no SI
        multiplier (see __init__): the ×N factor lives IN the label, so hiding the label
        while the ticks stayed scaled would have left them silently misstated — "500" for a
        value of 0.5, with nothing on screen saying so.
        """
        self._variants = tuple(variants)
        self._pick_label()

    def resizeEvent(self, ev=None):
        super().resizeEvent(ev)
        self._pick_label()

    def _pick_label(self):
        if not self._variants or self._picking:
            return                       # setLabel re-enters here via _updateLabel
        avail = self.height() if self.orientation in ("left", "right") else self.width()
        if avail <= 0:
            return
        self._picking = True
        try:
            chosen = self._variants[-1]
            for text in self._variants:
                if text is None:
                    chosen = None
                    break
                self._apply_label(text)
                if self.label.boundingRect().width() <= avail:
                    chosen = text
                    break
            self._apply_label(chosen)
            # pyqtgraph positions the rotated label in AxisItem.resizeEvent from the label's
            # CURRENT bounding rect. The wording is swapped AFTER that has run, and changing
            # it alters no geometry (the axis width derives from the label's HEIGHT, one line
            # either way), so no second resizeEvent is ever queued and the label keeps the
            # placement computed for the OLD, longer text -- measured 20 px below centre and
            # running past the bottom of a 63 px axis. Re-derive it for what is now there.
            pg.AxisItem.resizeEvent(self, None)
        except Exception:                # pragma: no cover - the label is cosmetic
            pass
        finally:
            self._picking = False

    def _apply_label(self, text):
        self.setLabel(text or None)
        self.showLabel(text is not None)

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


def _restrict_body_wheel_to_x(vb):
    """Make a channel plot's scroll wheel zoom the TIME axis only, never y.

    pyqtgraph's ViewBox zooms both axes on a body wheel; the mechanics channels have
    different units, so a stray scroll re-scaled whichever graph the cursor was over and left
    the rest alone — jarring. The AxisItem's own wheel (over the y-axis) still passes an
    explicit axis and so keeps working, which is the one place y-zoom should happen.
    """
    _orig = vb.wheelEvent

    def wheelEvent(ev, axis=None):
        _orig(ev, axis=0 if axis is None else axis)   # body wheel (axis=None) -> x only

    vb.wheelEvent = wheelEvent

"""Axis and plot styling helpers shared by every Preview & QC sub-tab: the
SI-prefix axis, the compact fit-axis, pen/palette helpers and the check-icon cache.
Split out of preview_screen.py (ticket A02); moved verbatim, not rewritten."""

from __future__ import annotations

import copy
import math
import os
import re
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QObject, QRectF, QSize, QThread, QTimer, Signal
from PySide6.QtGui import QFont

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui import plot_perf
from respmech.ui.plot_axis import MinPitchAxis, _next_nice
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

    On a short stacked panel, "Poes (cmH₂O)" on two lines is taller than the row and the
    rotated label ran into its neighbour (measured: the unit lines of adjacent channels
    merged into one string, e.g. "(cmH₂O)(cmH₂O)"). ``_pick_label`` re-derives, longest
    first and re-measured on every resize (the same approach ``_FitAxis._pick_label``
    uses below), the wording that fits the axis: name + unit, then the name alone, then
    the name alone at a smaller font down to ``_MIN_LABEL_SCALE`` of the base size, and
    only past that nothing at all. The font step exists because this is the screen whose
    job is to let the user confirm the channel assignment, so a blank axis is a defect,
    not a graceful fallback: with the stack at its 96 px row floor the axis is 76 px, and
    "Volume" alone measures ~92 px in the Windows runner's font (~75 px on the macOS
    runner, ~60 px on Linux). 14.3's two-step picker (name + unit, name, nothing) hid it
    on Windows alone, which is what turned the Windows CI red (06-09-2026).

    The picked wording has to survive pyqtgraph's own re-rendering. ``labelString()`` is
    what ``AxisItem._updateLabel()`` re-renders from, and pyqtgraph calls that on every
    range change (``setRange`` → ``updateAutoSIPrefix``), on ``showLabel(True)``, and
    from ``setLabel``/``enableAutoSIPrefix``, so it must return the CURRENT pick, not
    always the full wording. 14.3 returned the full wording, and its "name alone" pick
    was overwritten inside the very ``showLabel(True)`` that confirmed it (measured: a
    94 px "Poes (cmH₂O)" back on a 76 px axis). ``_updateLabel`` is overridden to re-pick
    instead of re-render, since the SI scale it may just have changed is part of the
    wording's width; the re-entrancy guard keeps the nested calls on pyqtgraph's plain path.

    A shortened wording keeps the ``·10ⁿ`` annotation whenever the ticks are scaled:
    the annotation is the only thing on screen saying "500" means 0.5 L, so it may be
    dropped only together with the whole label, and pyqtgraph itself pins the scale at
    1.0 while the label is hidden (``updateAutoSIPrefix`` reads ``label.isVisible()``).
    """

    _label_picking = False
    # Both are read by labelString() from inside AxisItem.__init__ (setRange →
    # updateAutoSIPrefix → _updateLabel), long before an __init__ body of ours has run.
    _include_unit = True        # current pick: unit line on the second line?
    _label_size = None          # current pick: None = base font, else (size, "pt"|"px")
    _MIN_LABEL_SCALE = 0.7      # smallest font the name is shown at, as a share of the base

    def __init__(self, *a, **k):
        self._name = ""
        self._unit = ""             # set before super(): its __init__ calls labelString()
        super().__init__(*a, **k)

    def set_channel_label(self, name, unit=""):
        self._name, self._unit = name, unit
        self._include_unit, self._label_size = True, None
        self.labelUnits = unit          # keep pyqtgraph's own SI-scaling of the tick values
        # labelText is ufarlig at sætte her: labelString() (nedenfor) bygger sin HTML af
        # _name/_unit og ignorerer labelText, så dette ændrer intet i selve aksen. Men
        # pyqtgraph's egen setLabel() ville have sat den, og krydshårets aflæsning
        # (_on_mech_mouse_moved) læser netop labelText for at navngive kanalen.
        self.labelText = name
        # pyqtgraph's AxisItem.__init__ efterlader label skjult; almindelig setLabel()
        # kalder showLabel() internt, men den kalder vi bevidst ikke (den ville overskrive
        # labelString()'ens to-linjers HTML og ·10⁻³-annotation). Et tomt navn skal
        # stadig skjule etiketten.
        self.showLabel(bool(name))
        if name:
            self._pick_label()
        self._adjustSize() if hasattr(self, "_adjustSize") else None

    def resizeEvent(self, ev=None):
        super().resizeEvent(ev)
        if self._name:
            self._pick_label()

    def _updateLabel(self):
        # pyqtgraph re-renders the label from labelString() here on every range change,
        # SI-scale change, showLabel() and setLabel(). Any of those can change the
        # wording's width (a "·10⁻³" line appears or goes), so re-pick rather than trust
        # the last pick. Nested calls made BY the pick take pyqtgraph's own path.
        if self._label_picking or not self._name:
            super()._updateLabel()
        else:
            self._pick_label()

    def _pick_label(self):
        if self._label_picking or getattr(self, "label", None) is None:
            return
        avail = self.height() if self.orientation in ("left", "right") else self.width()
        self._label_picking = True
        try:
            if avail <= 0:
                # geometry not resolved yet (fresh axis, not yet laid out): show the full
                # wording for now, resizeEvent re-picks once a real size is known.
                self._include_unit, self._label_size = True, None
                self.label.setHtml(self.labelString())
                return
            was_visible = self.label.isVisible()
            if not was_visible and self.autoSIPrefix:
                # measure with the SI scale the ticks will carry once the label shows:
                # pyqtgraph pins the scale at 1.0 for as long as the label is hidden.
                self.label.setVisible(True)
                self.updateAutoSIPrefix()
            fits = False
            width = 0.0
            for include_unit in (True, False):
                self._include_unit, self._label_size = include_unit, None
                self.label.setHtml(self.labelString())
                width = self.label.boundingRect().width()
                if width <= avail:
                    fits = True
                    break
            if not fits and width > 0:
                # Name alone is still too long: shrink the font towards the floor. The
                # first size is the linear estimate (text scales with the font, the
                # document's fixed margins do not), rounded DOWN to a half-point; each
                # miss steps down another half-point until the floor is reached.
                base, floor, unit = self._label_sizes()
                margin = 2.0 * self.label.document().documentMargin()
                size = base * max(0.0, avail - margin) / max(1e-6, width - margin)
                size = min(base, math.floor(size * 2.0) / 2.0)
                while size >= floor:
                    self._label_size = (size, unit)
                    self.label.setHtml(self.labelString())
                    if self.label.boundingRect().width() <= avail:
                        fits = True
                        break
                    size -= 0.5
            if fits:
                if not was_visible:
                    self.showLabel(True)
            else:
                # nothing fits even at the smallest font: hide rather than overrun.
                self._include_unit, self._label_size = True, None
                self.showLabel(False)
            pg.AxisItem.resizeEvent(self, None)   # re-centre for what is now there
            self.picture = None                   # ticks may carry a changed SI scale
            self.update()
        except Exception:                # pragma: no cover - the label is cosmetic
            pass
        finally:
            self._label_picking = False

    def _base_font_size(self):
        """The label's font size before any shrinking, as ``(size, "pt" | "px")``: an
        explicit ``font-size`` in ``labelStyle`` wins, else the label's own font."""
        m = re.match(r"\s*([0-9.]+)\s*(pt|px)", str(self.labelStyle.get("font-size", "")))
        if m:
            return float(m.group(1)), m.group(2)
        f = self.label.font()
        if f.pointSizeF() > 0:
            return f.pointSizeF(), "pt"
        return float(max(1, f.pixelSize())), "px"

    def _label_sizes(self):
        """``(base, smallest, unit)``: the base font size, the smallest size the picker
        will show the name at (``_MIN_LABEL_SCALE`` of the base, rounded UP to the
        half-point step the shrink loop walks in, so it is a size the loop actually
        tries), and the CSS unit both are in. A test proving a label was hidden for
        cause measures the name at ``smallest``: wider than the axis means no wording
        this axis is allowed to show would have fitted."""
        base, unit = self._base_font_size()
        return base, math.ceil(base * self._MIN_LABEL_SCALE * 2.0) / 2.0, unit

    def _scale_annotation(self):
        scale = getattr(self, "autoSIPrefixScale", 1.0)
        if not self.autoSIPrefix or scale == 1.0:
            return ""
        exp = int(round(math.log10(1.0 / scale)))   # displayed = value·10^exp
        return "·10" + str(exp).translate(_SUP)

    def _label_html(self, include_unit=True, size=None):
        scale = self._scale_annotation()             # kept in every wording but "hidden"
        unit = self._unit if include_unit else ""
        if scale and unit:
            second = f"({scale} {unit})"
        elif scale:
            second = f"({scale})"
        elif unit:
            second = f"({unit})"
        else:
            second = ""
        inner = self._name + (f"<br>{second}" if second else "")
        style = dict(self.labelStyle)
        if size is not None:
            style["font-size"] = f"{size[0]:g}{size[1]}"
        style = ";".join(f"{k}: {v}" for k, v in style.items())
        return f"<span style='{style}'><div style='text-align:center'>{inner}</div></span>"

    def labelString(self):
        return self._label_html(include_unit=self._include_unit, size=self._label_size)

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


class BreathSpansItem(pg.GraphicsObject):
    """One filled-region item per PLOT that paints every breath span on that plot,
    replacing one ``pg.LinearRegionItem`` (plus a now-removed boundary line — see
    below) per breath. A six-minute recording can carry 100+ breaths; on a 5-channel
    mechanics stack that was 11 QGraphicsItems x breaths x channels = over a thousand
    items, and the per-item redraw cost of that is what froze the GUI for seconds on
    every file step (ticket D15). One aggregate item per plot — a handful total,
    however many breaths it carries — replaces that with a fixed item count.

    The boundary line pg.LinearRegionItem's neighbour used to draw is dropped
    entirely, here and at every other caller of this class: it sat exactly on the
    region's own left edge and drew nothing the region didn't already show.

    Y-extent tracks the current view exactly like LinearRegionItem's own default
    ``span=(0, 1)`` does (``self.viewRect()``, refreshed by the base class on every
    view-transform change) — full plot height regardless of y-zoom, without this
    item having to know the plotted data's y-range at all. ``dataBounds`` mirrors
    LinearRegionItem's own axis restriction for the same reason LinearRegionItem
    has it: the item's y-extent is DERIVED from the view, so it must never feed
    back into that view's own y-autorange calculation.

    Purely a painter: mouse interaction (breath click-to-toggle) is resolved
    elsewhere from scene coordinates against the breath-span data, never from an
    item the mouse actually hit, so this item needs no hover/click handling of its
    own to keep that behaviour. One consequence, accepted for this ticket: a
    per-breath hover tooltip ("carried over from a previous folder…") that used to
    sit on the individual LinearRegionItem cannot be reproduced on a single shared
    item without new hover-tracking machinery outside this ticket's scope — the
    same fact is already shown, always-on, by the hatched brush this item still
    paints for a carried breath (see ``_breath_brush``) and by the QC line, so nothing
    that was the ONLY way to learn something is lost, only a redundant hover on top."""

    def __init__(self):
        super().__init__()
        self._spans = []      # [(t0, t1, QBrush), ...] — order is paint (== z) order
        self._x0 = self._x1 = 0.0

    def set_spans(self, spans):
        """Replace the full set of spans. ``spans``: iterable of ``(t0, t1, brush)``.
        Called once per file preview — not per breath, that is the whole point."""
        self._spans = list(spans)
        if self._spans:
            self._x0 = min(t0 for t0, _t1, _b in self._spans)
            self._x1 = max(t1 for _t0, t1, _b in self._spans)
        else:
            self._x0 = self._x1 = 0.0
        self.prepareGeometryChange()
        self.update()

    def set_brush(self, index, brush):
        """Recolour ONE span by its position in the list ``set_spans`` was given —
        the include/exclude toggle repaint path. Geometry is untouched, so this is
        just a paint, not a prepareGeometryChange."""
        if 0 <= index < len(self._spans):
            t0, t1, _old = self._spans[index]
            self._spans[index] = (t0, t1, brush)
            self.update()

    def boundingRect(self):
        vr = self.viewRect()
        br = QRectF(vr) if vr is not None else QRectF()
        if self._spans:
            br.setLeft(self._x0)
            br.setRight(self._x1)
        return br

    def paint(self, p, *args):
        if not self._spans:
            return
        vr = self.viewRect()
        full = QRectF(vr) if vr is not None else self.boundingRect()
        top, height = full.top(), full.height()
        p.setPen(pg.mkPen(None))
        for t0, t1, brush in self._spans:
            p.setBrush(brush)
            p.drawRect(QRectF(t0, top, t1 - t0, height))

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == 0:
            return (self._x0, self._x1) if self._spans else None
        return None


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


class _FitAxis(MinPitchAxis):
    """A :class:`~respmech.ui.plot_axis.MinPitchAxis` with two Preview-specific extras: no
    SI multiplier, and an axis label that re-picks its own wording as the panel is resized.

    The tick-thinning behaviour itself (measured on the Detail channel at 43 px of data
    area: six 13 px labels drawn 8.6 px apart, every pair overlapping by 4.4 px) now lives
    in the base class — extracted (ticket B05) so ``ColumnStack``'s own short rows could
    have it too, without dragging in the SI-suppression/multi-wording behaviour below,
    which is Preview-specific. This subclass is otherwise unchanged.
    """

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

    # tickSpacing / _min_pitch / _next_nice now live on MinPitchAxis (respmech.ui.plot_axis),
    # imported above and re-exported here (_next_nice) for the modules that already do
    # ``from ._plot_helpers import _next_nice``.


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

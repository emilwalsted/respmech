"""Compact-figure fitting for the matplotlib panels (Campbell diagram, fidelity
frontier): label/legend placement that shrinks to fit a small panel instead of
clipping. Split out of preview_screen.py (ticket A02); moved verbatim."""

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



def _pick_xlabel(canvas, ax, variants, renderer=None):
    """Set the longest x-label wording that is drawn entirely inside the canvas.

    matplotlib centres the x label on the AXES, not on the figure, and the axes are inset by
    the y-axis furniture — so "does the text fit the canvas width?" is the wrong question and
    answers yes while the label runs off the edge. Measured on the fidelity panel at its
    default 345 px: the full wording is 342 px wide (so it "fits") but is drawn from x=26 to
    x=368 and loses its last four characters. The extent is what gets measured here.

    The room is measured with the SHORTEST wording in place, never with the current one.
    tight_layout shrinks the axes to accommodate a label that overhangs them, so the room a
    long label is judged against is a room that long label already shrank — which makes the
    choice bistable. Measured: the same 400 px panel kept the full wording when it had been
    wider and dropped two rungs when it had been narrower, and a 460 px panel chose LESS text
    than a 400 px one. Judging every candidate against one layout that no candidate perturbed
    makes the answer a function of the width alone.
    """
    if not variants:
        return None
    try:
        ax.set_xlabel(variants[-1])
        if renderer is None:                # settle a layout the shortest label cannot skew
            canvas.draw()
        # renderer fetched AFTER the resize+draw: draw() builds a new one for the new size,
        # and measuring against the old one reported extents in the previous figure's
        # coordinates — which granted room that did not exist and chose a wording 24 px too
        # wide for the panel it was drawn in.
        r = renderer or canvas.get_renderer()
        ref = ax.xaxis.label.get_window_extent(renderer=r)
        centre = (ref.x0 + ref.x1) / 2.0    # == the axes centre; the label is centred on it
        # Measured in the FIGURE's own pixels, never the widget's. An earlier version forced
        # the figure to the widget size first, so that both agreed — and that shipped a
        # visible defect: a figure smaller than its canvas is painted into the widget's
        # top-left corner and the rest of the widget keeps whatever was there before, so the
        # panel showed a small plot composited over the previous, larger one. The figure size
        # belongs to Qt. If it is briefly behind the widget, the wording is re-picked when
        # the resize arrives (see _CompactFigureFitter) rather than forced now.
        width = float(canvas.figure.bbox.width)
        room = 2.0 * min(centre, width - centre)
        chosen = variants[-1]
        for text in variants:
            ax.set_xlabel(text)
            # a Text artist's WIDTH is pure font metrics — unlike its position, it does not
            # depend on the layout, so this comparison is stable
            if ax.xaxis.label.get_window_extent(renderer=r).width <= room:
                chosen = text
                break
    except Exception:                       # pragma: no cover - measurement is cosmetic
        chosen = variants[-1]
    ax.set_xlabel(chosen)
    return chosen


def _pick_ylabel(canvas, ax, variants, renderer=None):
    """Set the longest y-label wording that fits the axes height.

    A y label is rotated, so its length runs along the axes HEIGHT and it clips exactly as
    the x label clips against the width — measured on the Campbell panel at 130 px, where
    "Lung volume above end-expiration (L)" came out as "olume above end-ex".
    """
    if not variants:
        return None
    try:
        ax.set_ylabel(variants[-1])
        if renderer is None:
            canvas.draw()
        r = renderer or canvas.get_renderer()
        room = float(ax.get_window_extent(renderer=r).height)
        chosen = variants[-1]
        for text in variants:
            ax.set_ylabel(text)
            if ax.yaxis.label.get_window_extent(renderer=r).height <= room:
                chosen = text
                break
    except Exception:                       # pragma: no cover - measurement is cosmetic
        chosen = variants[-1]
    ax.set_ylabel(chosen)
    return chosen


def _fit_compact_figure(canvas, ax, *, title=None, legend_kw=None, xlabel_variants=None,
                        ylabel_variants=None):
    """Shed figure furniture that does not fit the canvas it has been given.

    These two diagnostics share the compact reference row, so their canvas can be ~100 px
    tall. matplotlib does not drop anything to make room — it lays the title, the legend and
    both axis labels out anyway, and on a real Windows runner they printed straight through
    each other and through the data (Emil's 02-08-2026 screenshots). Ordered by what a reader
    can most afford to lose: the title first (the panel header already names the figure), then
    the legend, then the x label.

    Every decision here is SYMMETRIC and is re-taken from the same stashed "full" furniture
    on every call, so the fit follows the panel in both directions. It has to: the reference
    row lives in a QSplitter, and matplotlib — unlike a pyqtgraph axis — never re-lays a
    figure out on resize. A one-way version dropped the legend when the panel was drawn short
    and could never bring it back once the panel was dragged tall again.
    """
    full = getattr(ax, "_rm_full", None)
    if full is None:
        full = ax._rm_full = {"title": title or "", "xlabel": ax.get_xlabel(),
                              "legend_kw": legend_kw, "xlabel_variants": xlabel_variants,
                              "ylabel": ax.get_ylabel(), "ylabel_variants": ylabel_variants}
    h = canvas.height()
    # Font sizes FIRST. The x-label wording is chosen by measuring the text, so it has to be
    # measured in the size it will be drawn in — setting the size afterwards made the choice
    # against the PREVIOUS fit's metrics, and a label measured at 8 pt (277 px) then drawn at
    # 10 pt (344 px) ran 24 px off the panel, which is the truncation this whole ladder exists
    # to prevent.
    small = h < 170
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_fontsize(8 if small else 10)
    ax.tick_params(labelsize=7 if small else 9)
    ax.set_title(full["title"] if h >= 150 else "")
    if full["legend_kw"] is not None:
        # only when something is actually labelled: matplotlib warns and draws an EMPTY
        # legend box otherwise, which is reachable here whenever a breath is missing the
        # average/recoil fields that carry the labels
        if h >= 190 and ax.get_legend_handles_labels()[1]:
            ax.legend(**full["legend_kw"])
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    # ONE settling draw for both ladders. Each used to draw for itself, which put two full
    # matplotlib renders into every fit — and a fit runs on every splitter drag, for three
    # canvases. Measured on the unit suite: 21 min -> 33 min, and Windows CI is already
    # hours. Both ladders shorten to their smallest wording first, so a single draw settles
    # a layout neither of them can skew, and both then measure against that one renderer.
    xvars = full["xlabel_variants"] if h >= 120 else None
    yvars = full.get("ylabel_variants")
    r = None
    if xvars or yvars:
        if xvars:
            ax.set_xlabel(xvars[-1])
        if yvars:
            ax.set_ylabel(yvars[-1])
        canvas.draw()
        try:
            r = canvas.get_renderer()
        except Exception:                   # pragma: no cover - falls back to per-pick draws
            r = None
    if h < 120:
        ax.set_xlabel("")
    elif xvars:
        # re-picked per fit, so the wording grows back when the panel is widened again
        full["xlabel"] = _pick_xlabel(canvas, ax, xvars, renderer=r)
    else:
        ax.set_xlabel(full["xlabel"])
    if yvars:
        full["ylabel"] = _pick_ylabel(canvas, ax, yvars, renderer=r)
    try:
        # The "tight" LAYOUT ENGINE, not a tight_layout() call. tight_layout() shrinks the
        # axes to clear the furniture, measured against the margins currently in force, so
        # repeating it compounds — and this now runs on every resize, which collapsed both
        # diagnostics into a corner of their panel after a few splitter drags. The engine
        # re-solves from the figure's original geometry at each draw instead, so it is
        # idempotent however many times the panel is resized.
        canvas.figure.set_layout_engine("tight")
    except Exception:                       # pragma: no cover - layout is cosmetic
        pass


def refit_compact_figure(canvas):
    """Re-take :func:`_fit_compact_figure`'s decisions for the canvas' CURRENT size.

    Cheap to call repeatedly and safe to call often: every decision is re-derived from the
    canvas' current size, and _pick_xlabel settles its own reference layout, so the outcome
    does not depend on what the panel was before.

    A re-fit at a SIZE already fitted is skipped rather than repeated: _fit_compact_figure's
    result is a pure function of (ax, canvas width, canvas height) once ax._rm_full is stashed
    (width feeds _pick_xlabel's room measurement, height feeds font sizing and title/legend
    visibility) — so redoing it at an unchanged size can only reproduce the same decision,
    except matplotlib's tight layout re-measures text extents against the axes' OWN current
    position each time, and on a real Windows runner that measurement is not bit-for-bit
    repeatable call to call (this sandbox's Agg renderer reproduces the exact same axes
    position over 20 repeated calls at a fixed size; Windows measured a 0.0131 drift). A fixed
    point beats a finer tolerance: never giving the loop the chance to re-measure removes the
    drift outright rather than chasing it.
    """
    axes = [a for a in canvas.figure.get_axes() if getattr(a, "_rm_full", None) is not None]
    if not axes:
        return
    size = (canvas.width(), canvas.height())
    axes = [a for a in axes if getattr(a, "_rm_last_fit_size", None) != size]
    if not axes:
        return
    for ax in axes:
        _fit_compact_figure(canvas, ax)
        ax._rm_last_fit_size = size
    canvas.draw_idle()


class _CompactFigureFitter(QObject):
    """Keep a compact matplotlib panel's furniture in step with its height.

    The two diagnostics sit in a splitter, so their canvas is resized by every splitter drag
    and every window resize — none of which redraws the figure. Without this, the furniture
    stays as it was at the last render: a legend shed on a short panel never returns, and one
    kept on a tall panel goes straight back to overprinting when the panel is dragged short.

    The refit is deferred by one event-loop turn and coalesced, so a drag that emits a
    hundred resize events does ONE fit — against the size that actually stuck — rather than a
    hundred, each of which costs a full figure draw. Correctness does not rest on the
    deferral: _pick_xlabel forces the figure to the widget's size before it measures anything,
    which is what makes the result independent of when this runs. (An event filter does see
    the event before its target, so the figure genuinely is a size behind here.)
    """

    def __init__(self, canvas):
        super().__init__(canvas)
        self._canvas = canvas
        self._pending = False
        canvas.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self._canvas and ev.type() == QEvent.Resize and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._run)
        return False

    def _run(self):
        self._pending = False
        refit_compact_figure(self._canvas)


class _PlotTitleOverlay(QObject):
    """Float a panel's title — and its corner control — INSIDE the plot's top band.

    The Detail and Conditioned panels used to spend a full layout row each on
    "title … picker" above the graph; on a laptop that was the difference between the strips
    wrapping and not (Emil, 02-08-2026: "kan flyttes ned så de ligger ovenpå grafen"). The
    label and the control are separate small children of the plot widget — deliberately NOT
    one full-width overlay row, whose body would eat the clicks meant for the plot between
    them. ``BAND`` is also applied as the plot item's top content margin, which does two
    jobs: the plotted data and the breath numbers stay clear of the floating controls, and
    the y axis' top tick label stops being clipped at the widget edge (the other half of the
    same report).
    """

    #: air above and below the floating row, inside the reserved band
    PAD = 5

    def __init__(self, plot, title, corner=None, extra=None):
        super().__init__(plot)
        self._plot = plot
        self._band = 0
        self._label = QLabel(title, plot)
        self._label.setProperty("status", "muted")
        # the label is decoration: clicks through it belong to the plot underneath
        self._label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._corner = corner
        # ``extra`` rides in the band between the title and the corner — it is where the
        # trace key goes, so naming the traces costs no plot height at all (the band is
        # already as tall as the corner control, and a one-line key is shorter than that).
        self._extra = extra
        if corner is not None:
            corner.setParent(plot)
        if extra is not None:
            extra.setParent(plot)
            extra.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        plot.installEventFilter(self)
        self._label.show()
        if corner is not None:
            corner.show()
        if extra is not None:
            extra.show()
        self.fit()

    def eventFilter(self, obj, ev):
        if obj is self._plot and ev.type() in (QEvent.Resize, QEvent.Show):
            self.fit()
        return False

    #: gap kept between the title and the corner control
    TITLE_GAP = 16

    #: narrowest the wrapping ``extra`` may be squeezed to before it is hidden instead
    EXTRA_MIN_WIDTH = 140

    def fit(self):
        """Re-pin the pieces and reserve exactly the band they need.

        The band is MEASURED, not a constant: the corner control is a combo box on one panel
        and a row of channel checkboxes on the other, and a fixed 28 px guess let both sit on
        top of the axis' top tick label and the multiplier caption. Reserving
        max(label, corner) + padding as the plot item's top margin is what keeps the floating
        row and the plotted content from ever sharing pixels.
        """
        self._label.adjustSize()
        band = self._label.height()
        if self._corner is not None:
            self._corner.adjustSize()
            # Bound it to the room actually left beside the title, and let a wrapping corner
            # take the height it then needs. Without this the corner kept its natural width
            # and was simply clipped off the right edge (12 EMG channels), and at 8+ it also
            # printed straight over the title.
            # measured against the title, not a constant: a fixed reserve was narrower than
            # the title itself on Windows metrics, so the corner still printed over it
            room = max(60, self._plot.width() - self._label.width()
                       - 12 - 14 - self.TITLE_GAP)
            if self._corner.width() > room:
                need = (self._corner.heightForWidth(room)
                        if self._corner.hasHeightForWidth() else self._corner.height())
                self._corner.resize(room, max(need, self._corner.height()))
            band = max(band, self._corner.height())
        # where the corner will sit; the extra gets what is left between title and corner
        corner_left = self._plot.width()
        if self._corner is not None:
            corner_left = max(0, self._plot.width() - self._corner.width() - 14)
        extra_left = 12 + self._label.width() + self.TITLE_GAP
        extra_room = corner_left - self.TITLE_GAP - extra_left
        if self._extra is not None:
            # Wrap rather than vanish. A one-line key measured 816 px against 815 px of room
            # at Windows font metrics in a 1280 px window — the narrowest screen this project
            # targets — and hiding it left the detail traces with nothing naming them at all,
            # the legend it replaced having been removed.
            #
            # Word wrap is turned on ONLY when it is needed, and off again when it is not: a
            # QLabel that is permanently wrappable reports a sizeHint from its own aspect-
            # ratio heuristic rather than its one-line width, which broke the key into four
            # lines and pushed this band from 45 px to 78 px — spending more plot height than
            # the whole re-proportioning had won back.
            self._extra.setWordWrap(False)
            self._extra.adjustSize()
            if self._extra.width() > extra_room >= self.EXTRA_MIN_WIDTH:
                self._extra.setWordWrap(True)
                # heightForWidth ONLY. A wrapped QLabel's sizeHint is that same aspect-ratio
                # heuristic, so folding it in with max() re-inflated the height the wrap was
                # meant to bound — 102 px where two lines needed 34.
                need = (self._extra.heightForWidth(extra_room)
                        if self._extra.hasHeightForWidth() else self._extra.height())
                self._extra.resize(extra_room, need)
            band = max(band, self._extra.height())
        band += 2 * self.PAD
        if band != self._band:
            self._band = band
            try:
                self._plot.getPlotItem().setContentsMargins(0, band, 0, 0)
            except Exception:                   # pragma: no cover - margin is cosmetic
                pass
        inner = band - 2 * self.PAD
        self._label.move(12, self.PAD + (inner - self._label.height()) // 2)
        self._label.raise_()
        if self._corner is not None:
            self._corner.move(corner_left,
                              self.PAD + (inner - self._corner.height()) // 2)
            self._corner.raise_()
        if self._extra is not None:
            # Hidden only when even a wrapped key has no usable room left — below that a
            # half-drawn key naming two of four traces would be worse than none.
            fits = extra_room >= min(self._extra.width(), self.EXTRA_MIN_WIDTH)
            self._extra.setVisible(fits)
            if fits:
                self._extra.move(extra_left,
                                 self.PAD + (inner - self._extra.height()) // 2)
                self._extra.raise_()

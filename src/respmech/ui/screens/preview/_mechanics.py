"""PreviewScreen's Mechanics sub-tab: the channel/flow-volume-pressure stack,
the Campbell diagram and per-breath table, and the click-to-exclude breath
overlays. Split out of preview_screen.py (ticket A02); moved verbatim."""

from __future__ import annotations

import copy
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QMenu, QProgressBar, QPushButton,
                               QAbstractItemView, QScrollArea, QSplitter, QTableView,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QObject, QSize, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QCursor, QFont, QFontMetrics, QFontMetricsF

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.analysis.signals import Capabilities
from respmech.core.settings import (BreathRef, BreathTypeEntry, ExcludeEntry,
                                    GroupReferenceEntry, ReferenceEntry,
                                    SegmentationOverrideEntry)
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui import plot_perf
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers
from respmech.ui.validation import matching_files
from respmech.ui import wheel as _wheel
from respmech.ui.flow_layout import (ElidingLabel, FlowLayout, cluster as _cluster,
                                     elide as _elide, install_flow as _install_flow)
from respmech.ui.result_table import (ResultTableModel, configure_result_table,
                                      resize_result_table)
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)
from respmech.ui import prefs as _prefs
from respmech.ui.theme import SELECTED_BREATH_RGB, SELECTED_BREATH_HEX

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

from ._figure_fit import _CompactFigureFitter, _fit_compact_figure
from ._jobs import _FileRunError, _KIND_LABEL, _PANELS
from ._plot_helpers import (BreathSpansItem, SciAxis, SeparatorLinesItem, _CHANNELS,
                            _pen, _plot_pal, _restrict_body_wheel_to_x)


#: A breath span's/BreathTypeEntry's kind -> the palette suffix _breath_brush/
#: _breath_label_color reads (``breath_<suffix>_brush``/``_label``). 'excluded' is a
#: UI-only pseudo-kind (processing.exclude_breaths, never a BREATH_KINDS member); any
#: real BREATH_KINDS member without its own dedicated entry here (ic_fvc/max_insp/sniff/
#: other — M-20's minimal menu never offered them, and M-31's fuller one deliberately
#: keeps them visually grouped with 'other' rather than growing the palette further)
#: falls back to 'other' rather than raising on an unknown kind.
_KIND_PALETTE_SUFFIX = {"excluded": "excl", "ic": "ic", "fvc": "fvc", "rest": "rest"}

#: The full type menu (M-31): every respmech.core.settings.BREATH_KINDS member, plus
#: the two pseudo-kinds 'tidal'/'excluded' _set_breath_type also accepts. 'rest' is
#: filtered out by _handle_type_requested for a flow-bearing (non-EMG-only) file —
#: it names a noise-reference segment, which only makes sense on the EMG-only
#: 'EMG – segments' tab this same menu-building code also serves (see the module's
#: 'kunder' comment on _handle_type_requested). Order matches the ticket's own
#: ordering, which is also the order BREATH_KINDS is declared in.
_TYPE_MENU_KINDS = ("tidal", "excluded", "ic", "fvc", "ic_fvc", "max_insp", "sniff", "rest", "other")
_TYPE_MENU_LABELS = {
    "tidal": "Tidal",
    "excluded": "Excluded",
    "ic": "IC manoeuvre",
    "fvc": "FVC manoeuvre",
    "ic_fvc": "IC + FVC",
    "max_insp": "Maximal inspiratory effort",
    "sniff": "Sniff",
    "rest": "Rest",
    "other": "Other…",
}

#: statusTip shown while hovering each type-menu action (M-31) — the same
#: bold-path/description shape ui.help_text.tooltip() uses for settings controls,
#: but plain text (QAction.statusTip is not rich text) since there is no settings
#: path to show for a menu choice.
_TYPE_MENU_STATUS_TIPS = {
    "tidal": "Ordinary tidal breathing — included in the average like any other breath.",
    "excluded": "Exclude this breath from the tidal average without typing it.",
    "ic": "Mark as an inspiratory capacity manoeuvre: reports its own volume, timing and "
          "pressure swings on the Manoeuvres sheet.",
    "fvc": "Mark as a forced vital capacity manoeuvre (a deliberately prolonged forced "
           "exhalation).",
    "ic_fvc": "Mark as a combined inspiratory-capacity-then-forced-exhalation manoeuvre.",
    "max_insp": "Mark as a maximal inspiratory effort — reports a peak-effort reference value.",
    "sniff": "Mark as a maximal sniff manoeuvre — reports a peak-effort reference value.",
    "rest": "Mark as a quiet-breathing reference segment for noise-profile estimation.",
    "other": "Mark as typed without extracting any manoeuvre values.",
}

#: The second line of a breath's label (ticket: show the breath type as text under the
#: breath number). Plain tidal breathing has none. 'excluded' reads "(Excluded)" as
#: Emil asked; the typed kinds use the SHORT names the Manoeuvres sheet uses, not the
#: menu's longer sentences, so the line stays about as wide as a breath span usually is.
_KIND_TAG = {
    "excluded": "(Excluded)", "ic": "IC", "fvc": "FVC", "ic_fvc": "IC+FVC",
    "max_insp": "Max insp", "sniff": "Sniff", "rest": "Rest", "other": "Other",
}
#: The compact stand-in used instead of the text when the view is zoomed out so far that
#: the text would run into its neighbours: one or two characters, still in the kind's own
#: colour, so the type stays readable (initials, with '×' for an exclusion and '…' for
#: 'other') without needing a symbol font.
_KIND_ICON = {
    "excluded": "×", "ic": "I", "fvc": "F", "ic_fvc": "IF",
    "max_insp": "M", "sniff": "S", "rest": "R", "other": "…",
}
#: Spare pixels required between a label's text and the edge of its own breath span
#: before the full text is judged to fit (labels are centred on their span, so a label
#: wider than its span is what collides with its neighbour).
_LABEL_FIT_PAD_PX = 4.0

#: slot -> the BreathTypeEntry.kind values that count as "this breath is already typed
#: as slot" for the quick 'Use as IC reference for' submenu (M-37) -- mirrors
#: core.analysis.references._OWN_TYPED_KINDS['ic'] exactly (only that one slot has a
#: one-click shortcut here; the other three slots are only reachable through the full
#: 'Reference manoeuvres…' picker).
_IC_REFERENCE_KINDS = frozenset({"ic", "ic_fvc"})


def _set_chip_text(chip, text, *, fixed_tooltip=None):
    """``QLabel.setText``, but routed through ``ElidingLabel.setFullText`` when ``chip``
    is one (``qc_overview``, M-37) so the FULL text -- not just whatever happens to be
    displayed right now -- survives into the tooltip and can be re-elided on resize.
    ``_update_qc_overview``/``_qc_overview_not_assessed``/``_reset_qc_overview`` share
    this so a caller never has to know which chip type it was handed (``chip`` can also
    be ``segments_qc_overview``, still a plain QLabel -- M-26, where ``fixed_tooltip`` is
    silently ignored).

    ``fixed_tooltip``: ``ElidingLabel.setFullText`` always overwrites the tooltip with
    the text just set -- right for ``mech_window_label`` (the tooltip only ever needs to
    recover the elided text itself), wrong here: ``qc_overview``'s tooltip is a STATIC
    explanation of what the chip even measures ("the CURRENTLY PREVIEWED file's most
    recent test run"), which the short QC verdict text alone does not say. Passing it
    re-applies that fixed wording after ``setFullText``'s own overwrite, the same
    two-step ``_set_analysis_window`` already does for ``mech_window_label``."""
    if hasattr(chip, "setFullText"):
        chip.setFullText(text)
        if fixed_tooltip is not None:
            chip.setToolTip(fixed_tooltip)
    else:
        chip.setText(text)


def _short_boundary_note(notices):
    """Compress ``compute.trim_boundary_notices()``'s full sentences (K-035) to a
    short status-bar-safe form. The status line is a single-line ``QStatusBar``
    message that does not wrap, so the full ~300-character explanation risks being
    clipped exactly on the screen it points the user to — see
    ``_render_preview_stage3``'s caller for the full reasoning. The full text is
    unabridged everywhere else (Run log, terminal, ``run-report.txt``)."""
    parts = []
    for n in notices:
        edge = "first" if "first breath" in n else "last" if "last breath" in n else "boundary"
        if "already excluded" in n:
            parts.append(f"⚠ {edge} breath excluded, but drift baseline may still be "
                         "tilted — re-export to fix.")
        else:
            parts.append(f"⚠ {edge} breath may be truncated — re-export or exclude it.")
    return " ".join(parts)


def _parse_breath_counts(text, entry_cls, folder=None):
    """Parse the 'filename = count' lines of the breath-count override box into entries.
    Splits on the LAST '=' so a filename may itself contain one; skips blank/malformed lines.

    ``folder`` (the CURRENT ``input.folder``) is stamped onto every entry the box
    produces. This dialog holds one flat list — the file the box currently names, current
    input folder included — so every line it commits belongs to that folder by
    construction; an untouched OK never gets here at all (see the caller, which only
    re-parses when the user actually edited this field), so this cannot relabel an entry
    the user never looked at."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        fpart, _, cpart = line.rpartition("=")
        try:
            cnt = int(cpart.strip())
        except ValueError:
            continue
        if fpart.strip():
            out.append(entry_cls(fpart.strip(), cnt, folder))
    return out

def _analysis_rate(resample, resample_to_frequency, native_frequency):
    """The sampling rate segmentation actually runs at: the pre-analysis resample target
    when it is on, otherwise the recording's own rate. The SAME rule as ``fs_eff`` in
    ``Settings.validate()`` (settings.py) and the resample applied in ``pipeline.py`` —
    reproduced here rather than imported, since both of those operate on a committed
    ``Settings``/dataframe while this reads live, uncommitted dialog values."""
    return resample_to_frequency if (resample and resample_to_frequency > 0) else native_frequency


def _buffer_debounce_seconds(buffer_samples, resample, resample_to_frequency,
                             native_frequency):
    """The analysis-rate duration a segmentation debounce of ``buffer_samples`` samples
    corresponds to. Returns None if the analysis rate is not known yet (e.g. a dialog
    opened before Setup ▸ Input's sampling frequency is filled in)."""
    fs_eff = _analysis_rate(resample, resample_to_frequency, native_frequency)
    if not fs_eff:
        return None
    return buffer_samples / fs_eff


def _buffer_debounce_hint(buffer_samples, resample, resample_to_frequency, native_frequency):
    """The 'N samples ≈ X.XX s at Y Hz' line shown beside the Breath-separation debounce
    field (D29/UI-overhaul) — the debounce is stored and edited in samples (unaffected by
    a later resample), but samples alone hide how long it actually holds the boundary at
    the rate the run analyses at, which is the ONE thing about this setting that changes
    with the recording."""
    secs = _buffer_debounce_seconds(buffer_samples, resample, resample_to_frequency,
                                    native_frequency)
    if secs is None:
        return f"{buffer_samples} samples (sampling rate not set yet)."
    fs_eff = _analysis_rate(resample, resample_to_frequency, native_frequency)
    return f"{buffer_samples} samples ≈ {secs:.2f} s at {fs_eff:g} Hz."


# Per-file errors that mean "this recording cannot support this step", not "something
# broke". The mech preview lands them softly and explains them in the status line, so the
# batch panel must not also paint a hard 'Test run failed' card over the same thing.
# EmgSegmentationError: a bad separator placement on an EMG-only set -- the
# 'batch' test run reaches this via FileResult.error_kind exactly like the other three;
# the 'segments' preview job never raises it at all (stage_emg_segments_preview catches
# it itself, see _segments.py's _render_segments_preview's own 'Not processed' status line).
# ReferenceLinkError: a file's cross-file IC reference could not be resolved and
# processing.lung_volume.require_references is set -- see
# core.analysis.references.ReferenceLinkError's own docstring for when this actually
# escapes a file (off by default: an unresolved reference is normally just a caution
# plus NaN, never a file error).
_SOFT_FILE_ERRORS = ("TrimError", "VolumeTrendError", "NoBreathsError", "EmgSegmentationError",
                     "ReferenceLinkError")

# The Mechanics-advanced fields that change the volume the trend detector sees. The live
# trough count is only valid while these still match the rendered preview it was taken
# from (see _trend_hint).
_TREND_PROBE_KEYS = ("integrate_from_flow", "correct_drift", "inverse_flow",
                     "inverse_volume", "resample", "resample_to_frequency")

# Campbell diagram axis-label ladders (D12/UI-overhaul). The volume axis must ALWAYS
# name its datum — EELV or "end-expiration" — never bottom out at the ambiguous
# "Volume (L)", which on a Campbell diagram reads as absolute lung volume, a different
# quantity. Poes gets its own ladder now that the axis swap below moves it onto the
# height-constrained axis the volume label used to occupy.
_CAMPBELL_XLABEL_VARIANTS = ("Lung volume above end-expiration (L)",
                            "Volume above EELV (L)", "V−EELV (L)")
_CAMPBELL_YLABEL_VARIANTS = ("Oesophageal pressure  Poes (cmH₂O)", "Poes (cmH₂O)", "Poes")

#: M-17 (R7): the Campbell panel's stand-in for a Poes-less (Flow only) signal set — a
#: tidal flow-volume loop, same axis-label ladder mechanics as the Campbell diagram above.
_FV_XLABEL_VARIANTS = ("Lung volume (L)", "Volume (L)", "V (L)")
_FV_YLABEL_VARIANTS = ("Flow (L/s)", "Flow")
#: The same panel when the tidal loops sit inside the file's MFVL — TLC on the left.
_MFVL_XLABEL_VARIANTS = ("Volume below TLC (L)", "Below TLC (L)", "V (L)")


def _mech_channel_count(settings) -> int:
    """How many rows the Mechanics stack draws for this signal set (M-17, R7).

    ``_render_preview_stage1`` filters ``_CHANNELS`` (flow, volume, poes, pgas, pdi) down to
    whichever keys ``series`` actually carries for the file just previewed — a crash guard
    against a reduced signal set, added before this ticket. This mirrors that same count from
    ``Capabilities`` alone, so the stack's floor (``_update_mech_stack_floor``) is right even
    before any file has ever been previewed, and on every resize in between — the two counts
    agree by construction, because both ultimately trace back to the same declared/assigned
    channels: Capabilities' five flow/pressure fields are named and ordered exactly like
    ``_CHANNELS``' five keys. Never zero: an analysis with no flow/pressure channel at all
    (unreachable from the UI today, see ``subtab_plan``) falls back to the full five rather
    than flooring a stack for none, which ``theme.set_stack_floor`` treats as at least one
    anyway (``max(1, rows)``). Uses ``from_settings_or_none``: ``_update_mech_stack_floor``
    (the sole caller) is invoked from ``_MechStackFloorFitter``'s deferred resize/show
    callback, which fires on real ``MainWindow`` startup exactly like the sub-tab bar and
    Campbell title this same ticket's fix already covers — ``None`` (nothing safe to
    derive) falls back the same way the zero-signal case already does, to the full five,
    rather than raising out of that Qt callback."""
    caps = Capabilities.from_settings_or_none(settings)
    if caps is None:
        return len(_CHANNELS)
    n = sum((caps.flow, caps.volume, caps.poes, caps.pgas, caps.pdi))
    return n or len(_CHANNELS)

#: QSettings key for the persisted vertical splitter (channel stack / table+Campbell) — D14.
_MECH_VSPLIT_PREF_KEY = "preview.mech.vsplit"

#: How long to wait after the last ``splitterMoved`` before writing it to QSettings — a drag
#: fires many of these, and only the position it settles on is worth persisting (D14).
_SPLIT_SAVE_DEBOUNCE_MS = 400


class _MechStackFloorFitter(QObject):
    """Keep the Mechanics channel stack's floor eftergivende for the viewport it is actually
    shown in (D14, UI-overhaul).

    ``theme.set_stack_floor`` used to floor the five-channel stack at a flat
    ``rows * PLOT_ROW_MIN_HEIGHT`` (480 px) regardless of the window. On a 1280x720 screen
    that pinned the vertical splitter's lower half at ITS OWN floor with zero pixels of
    travel — the splitter reported movable but a drag could not move it, so scrolling was
    the only way to reach the table/Campbell half at all, and while scrolling the graphs the
    "#1…#9" breath strip a user needs to click scrolled out of view with them. This installs
    on the Mechanics scroll area's viewport and recomputes the floor (via
    ``PreviewScreen._update_mech_stack_floor``) on every resize, so the floor gives way
    exactly where it previously had no room to; a tall window is unaffected because the cap
    only bites once the viewport is shorter than the full floor (see ``set_stack_floor``).

    Deferred by one event-loop turn and coalesced — the same pattern
    ``_CompactFigureFitter``/``_PlotTitleOverlay`` already use in this package: a resize drag
    fires many Resize events, and shrinking the floor itself triggers a LayoutRequest the
    FitScrollArea reacts to by resizing the page, which is not itself a viewport resize but
    is cheap to just recompute against again next turn rather than special-case away.
    ``QEvent.Show`` is included because a scroll area's viewport can hold its final size
    while the Mechanics sub-tab is not yet the current one — GUI construction runs before
    anything is shown — so the first live measurement often arrives as a Show, not a Resize.
    """

    def __init__(self, screen, viewport, update_fn=None):
        super().__init__(viewport)
        self._screen = screen
        self._viewport = viewport
        self._pending = False
        # M-26: bound per stack — the segments tab's own GraphicsLayoutWidget reuses this
        # exact eftergivende-floor mechanism via its own update_fn (_update_segments_stack_
        # floor) instead of the Mechanics-specific _update_mech_stack_floor every prior
        # instance defaulted to.
        self._update_fn = update_fn if update_fn is not None else screen._update_mech_stack_floor
        viewport.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self._viewport and ev.type() in (QEvent.Resize, QEvent.Show) and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._run)
        return False

    def _run(self):
        self._pending = False
        self._update_fn()


class _MechanicsMixin:

    def _build_mech_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)
        # Cursor read-out (time, value at the pointer) — above the graphs, right-aligned, and
        # never wrapping: it updates on every mouse move, so a wrapping label made the whole
        # stack jump a row up and down. A fixed single line on top stays put.
        self.crosshair_label = QLabel("")
        self.crosshair_label.setProperty("status", "muted")
        self.crosshair_label.setWordWrap(False)
        self.crosshair_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # All the mechanics settings live behind here now (they left the Setup screen); this
        # is the tab they shape, so it is where they belong.
        self.btn_mech_advanced = QPushButton("Advanced… (breath detection, volume, WOB)")
        self.btn_mech_advanced.setProperty("compact", True)
        self.btn_mech_advanced.setToolTip(
            "Breath segmentation, work-of-breathing source, volume/drift corrections, "
            "resampling and per-file breath-count overrides.")
        self.btn_mech_advanced.clicked.connect(self._open_mech_advanced)
        # A persistent one-line caption — breath count, exclusion count, the click-to-
        # exclude instruction — kept current by _render_preview and _toggle_breath (see
        # _set_mech_caption). It used to live only in the transient status bar, where an
        # EMG job finishing a second later would silently erase it.
        self.mech_caption = ElidingLabel("")
        self.mech_caption.setProperty("status", "muted")
        _xh = QHBoxLayout(); _xh.setContentsMargins(6, 0, 6, 0)
        _xh.addWidget(self.btn_mech_advanced)
        _xh.addSpacing(10)
        _xh.addWidget(self.mech_caption, 1)
        _xh.addWidget(self.crosshair_label, 0, Qt.AlignRight)
        v.addLayout(_xh)
        split = QSplitter(Qt.Vertical)
        self._mech_vsplit = split
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setAccessibleName("Mechanics signals")   # C02: named for QAccessible/screen readers
        _theme.set_plot_floor(self.plots)
        self.plots.setBackground(_plot_pal()["bg"])
        self.plots.scene().sigMouseClicked.connect(self._on_plot_clicked)
        # P17: a crosshair that tracks the cursor across the X-linked channel stack and
        # reads out (time, value); built once, the per-channel lines are (re)made per render.
        self._crosshair_lines = []
        self._crosshair_proxy = pg.SignalProxy(
            self.plots.scene().sigMouseMoved, rateLimit=60, slot=self._on_mech_mouse_moved)
        split.addWidget(self.plots)
        lower = QSplitter(Qt.Horizontal)
        self.table = QTableView()
        self.table.setAccessibleName("Per-breath results")   # C02: named for QAccessible/screen readers
        self._table_model = ResultTableModel()
        self.table.setModel(self._table_model)
        # breath_no is already a table COLUMN; the row header numbered rows 1..N
        # regardless, which after an exclusion disagreed with breath_no's own
        # numbering (1, 3, 4, 5 …) — hide the row header rather than carry two
        # numbering schemes that can say different things about the same row.
        self.table.verticalHeader().setVisible(False)
        configure_result_table(self.table)
        # a table squeezed to one visible row is as useless as a flattened graph
        _theme.set_plot_floor(self.table)
        # M-31: the Manoeuvres sheet, stacked BELOW the per-breath table in the SAME
        # panel (no third action-band row — _build_mech_action_band stays a single fixed
        # row) — its own small ResultTableModel, hidden until a test run reports at least
        # one typed breath (_fill_manoeuvres_table), so an untyped file's Mechanics panel
        # looks exactly as it did before this ticket.
        self.manoeuvres_table = QTableView()
        self.manoeuvres_table.setAccessibleName("Manoeuvres")   # C02
        self._manoeuvres_model = ResultTableModel()
        self.manoeuvres_table.setModel(self._manoeuvres_model)
        self.manoeuvres_table.verticalHeader().setVisible(False)
        configure_result_table(self.manoeuvres_table)
        _theme.set_plot_floor(self.manoeuvres_table)
        self._manoeuvres_label = ElidingLabel("Manoeuvres")
        self._manoeuvres_label.setProperty("status", "muted")
        self._manoeuvres_section = QWidget()
        _mv_lay = QVBoxLayout(self._manoeuvres_section)
        _mv_lay.setContentsMargins(0, 6, 0, 0)
        _mv_lay.setSpacing(2)
        _mv_lay.addWidget(self._manoeuvres_label)
        _mv_lay.addWidget(self.manoeuvres_table, 1)
        self._manoeuvres_section.setVisible(False)
        _tables_box = QWidget()
        _tables_lay = QVBoxLayout(_tables_box)
        _tables_lay.setContentsMargins(0, 0, 0, 0)
        _tables_lay.setSpacing(6)
        _tables_lay.addWidget(self.table, 2)
        _tables_lay.addWidget(self._manoeuvres_section, 1)
        # D22 (UI-overhaul): titled, like the Campbell panel beside it, so the header can
        # name the work-of-breathing source (see _set_wob_table_note) — otherwise nothing
        # on screen says that the five wob* columns are one whole-file value repeated on
        # every row when "Work of breathing from" is Average (the default).
        self._table_panel = self._titled("Per-breath results", _tables_box)
        lower.addWidget(self._table_panel)
        self.campbell = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        self.campbell.setAccessibleName("Campbell diagram")   # C02: named for QAccessible/screen readers
        # A matplotlib canvas reports a 10 px minimum, so it was the one Preview graph with
        # no floor at all — measured 90 px on the target screen and 70 px at the window's own
        # minimum, which for the panel a reader takes the recoil line and work fill off is a
        # box, not a diagram.
        _theme.set_plot_floor(self.campbell)
        # A titled panel, not a bare canvas. It used to butt flush against the table and the
        # panel edge (Emil, 02-08-2026), which a hand-rolled box with _titled's margins fixed
        # — but the figure's own title was still the only thing naming the panel, and at the
        # height this panel actually gets (130 px measured) matplotlib cannot fit a title:
        # it was drawn with its top 10 px outside the figure, so it read as a half-cut
        # "Campbell diagram". The header names it at any height and costs the same room the
        # clipped title did; the figure keeps its title only for the export (_export_campbell).
        self._campbell_fitter = _CompactFigureFitter(self.campbell)
        # title_floor_chars: this header is the ONLY thing naming the diagram (the
        # figure's own title is deliberately screen-only, see above) — see
        # titled_panel()'s docstring for why it must not be allowed to collapse to a
        # bare ellipsis the way the general-purpose default floor allows.
        # Stored (not just added) so its header text can follow the signal set (M-17): a
        # Poes-less analysis draws a flow-volume loop here instead of a Campbell diagram —
        # see _update_campbell_panel_title/_draw_campbell_or_loop.
        self._campbell_panel = self._titled("Campbell diagram", self.campbell,
                                            title_floor_chars=10)
        lower.addWidget(self._campbell_panel)
        # Non-collapsible: a QSplitter's children are collapsible by default, and a
        # collapsible child can be dragged — or arrive from a restored/derived layout —
        # BELOW its minimumSizeHint, all the way to nothing. That is the one mechanism by
        # which a floored header can still render as a bare '…': the Windows runner showed
        # exactly that (test_the_campbell_panel_is_readable_at_the_height_it_gets, red
        # #192→#216) while every macOS/Linux run kept the pane wide enough. With collapsing
        # off, the splitter honours each pane's minimum — the table keeps its scroll-based
        # small minimum, and the Campbell pane can never be squeezed below the header floor
        # + margins that keep its name readable.
        lower.setChildrenCollapsible(False)
        # D12: the diagram is what the user came for, not the number dump beside it — at
        # 3:1 (720/240) a maximised window gave the table 1265 px and the diagram 406x249,
        # a 3x weight difference in favour of the numbers. ~58/42 (matches the stretch
        # factor below) gives the diagram room to be legible without starving the table.
        lower.setStretchFactor(0, 7); lower.setStretchFactor(1, 5)
        lower.setSizes([560, 400])
        split.addWidget(lower)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        # D14: restore a previously-chosen split BEFORE it is shown, so a user who has found
        # a division that works keeps it session to session — falls back to the stretch
        # factors above (unchanged default) the first time, when nothing is saved yet.
        _prefs.restore_splitter_state(_MECH_VSPLIT_PREF_KEY, split)
        self._mech_split_save_pending = False
        split.splitterMoved.connect(self._schedule_mech_split_save)
        v.addWidget(split)
        return w

    def _schedule_mech_split_save(self, *_args):
        """Debounced save of the vertical splitter's handle position (D14): a drag fires
        many ``splitterMoved`` signals, and only where it settles is worth persisting."""
        if self._mech_split_save_pending:
            return
        self._mech_split_save_pending = True
        QTimer.singleShot(_SPLIT_SAVE_DEBOUNCE_MS, self._flush_mech_split_save)

    def _flush_mech_split_save(self):
        self._mech_split_save_pending = False
        _prefs.save_splitter_state(_MECH_VSPLIT_PREF_KEY, self._mech_vsplit)

    def _update_mech_stack_floor(self):
        """Recompute the Mechanics channel stack's floor (D14) for the viewport it is
        actually shown in right now. The floor gives way once that viewport genuinely
        cannot afford the full ``rows * PLOT_ROW_MIN_HEIGHT`` — see
        ``theme.set_stack_floor`` for why and by how much. Called both after every render
        (``_render_preview``, which used to set the viewport-blind floor directly) and on
        every resize/show of the Mechanics scroll area (``_MechStackFloorFitter``), so the
        floor is correct regardless of which happens first — a render before the tab has
        ever been shown (viewport height 0, so the floor stays the original ``rows *
        row_height``), or a resize of an already-rendered tab.

        Reached from a deferred ``QTimer.singleShot(0, ...)`` (``_MechStackFloorFitter``),
        so the widgets it touches can in principle have been C++-deleted by the time it
        fires (window closed mid-resize); guarded the same way the codebase already guards
        every other cosmetic, deferred Qt call (see ``theme.set_plot_floor``) — never worth
        crashing a build over a floor that no longer has anywhere to apply."""
        if _theme is None:
            return
        try:
            area = getattr(self, "_mech_tab", None)
            vp_h = area.viewport().height() if area is not None else 0
            _theme.set_stack_floor(self.plots, _mech_channel_count(self.state.settings),
                                   viewport_height=vp_h if vp_h > 0 else None)
        except RuntimeError:                  # pragma: no cover - deleted C++ widget
            pass

    def _update_mech_action_band_visibility(self):
        """Show each tab's own action band only while ITS sub-tab is the current one —
        the ECG/noise sub-tabs have no equivalent band, and a band would otherwise sit
        fixed under whichever page happens to be showing. Compares the TAB WRAPPER
        (``_mech_tab``/``_segments_tab``), not the page, because that is what
        ``self.subtabs`` actually holds (see the comment in ``PreviewScreen._build`` on
        why the wrapper is the tab item).

        M-26: now toggles the segments tab's own band too — 'accepts both tabs' rather
        than being Mechanics-only, since Mechanics and the segments tab are mutually
        exclusive (never both present in the same ``subtab_plan``, see ``_schedule``'s
        'mech'/'segments' gate), so exactly one of the two bands, or neither (ECG/noise),
        is ever visible at once."""
        cur = self.subtabs.currentWidget()
        band = getattr(self, "_mech_action_band", None)
        if band is not None:
            band.setVisible(cur is self._mech_tab)
        seg_band = getattr(self, "_segments_action_band", None)
        if seg_band is not None:
            seg_band.setVisible(cur is getattr(self, "_segments_tab", None))

    def _build_mech_action_band(self):
        """The Mechanics QC verdict + its two per-file actions (P16/P17), built as a small
        standalone widget so ``PreviewScreen._build`` can pin it in the ROOT layout under
        ``self.subtabs`` (D14, UI-overhaul) — fixed OUTSIDE the scrolling page, the same
        pattern ``settings_screen.py`` uses for its own QC strip (``self.qc``).

        It used to sit inside ``_build_mech_tab``'s scrolled page, at the very bottom. On a
        short screen the page scrolls to reach it, and while scrolled the verdict that gives
        the Mechanics tab its name — and the only two actions the tab offers — were the ONE
        thing guaranteed to be off screen: the tab opened on the channel stack, never on its
        own judgement. Visibility is toggled by ``PreviewScreen`` to only the Mechanics
        sub-tab (``_update_mech_action_band_visibility``); the ECG/EMG sub-tabs have no
        equivalent band."""
        band = QWidget()
        band.setMinimumHeight(36)
        band.setMaximumHeight(40)
        bar = QHBoxLayout(band)
        bar.setContentsMargins(6, 0, 6, 0)
        # ElidingLabel (M-37, not a plain QLabel): the QC verdict text grows with the
        # number of flags found (_update_qc_overview's "⚠ ..." suffix), and this chip
        # sits in a fixed-height action band next to mech_window_label, which already
        # needed the same fix for the same reason (see its own comment below).
        self.qc_overview = ElidingLabel("")
        self.qc_overview.setProperty("banner", True)   # box baked at first polish (theme.py)
        self.qc_overview.setProperty("status", "muted")
        self.qc_overview.setToolTip(self._QC_OVERVIEW_TOOLTIP)
        # D23 (UI-overhaul): the analysis window, at a PERSISTENT spot beside the QC chip —
        # not the shared status line (main_window.py connects every screen's status_changed
        # to one bar, so a Setup/EMG message lands a moment later and silently overwrites
        # whatever Mechanics last said there). The mechanics channel stack's time axis reads
        # time SINCE the analysis window starts (zero-based), while the EMG tabs read the
        # file's own untrimmed clock — this line names the window so both are readable.
        # ElidingLabel (not a plain QLabel), given the stretch this band's addStretch(1)
        # used to hold: a plain, non-stretching QHBoxLayout item hands its sizeHint straight
        # to the window's minimum width, and this text's length varies with the recording
        # (self-review of this ticket flagged the same narrow-window overflow class this
        # codebase already hit once with the header subtitle, see flow_layout.ElidingLabel).
        self.mech_window_label = ElidingLabel("")
        self.mech_window_label.setProperty("status", "muted")
        self.mech_window_label.setToolTip(self._MECH_WINDOW_TOOLTIP)
        self.btn_export_fig = QPushButton("Export Campbell…")
        self.btn_export_fig.setEnabled(False)          # enabled once a diagram is drawn
        self.btn_export_fig.setToolTip("Save the Campbell diagram as a PNG or PDF.")
        self.btn_export_fig.clicked.connect(self._export_campbell)
        # Mechanics' own 'Place separators' — repairs the AUTOMATIC flow-/
        # volume-based segmentation (processing.segmentation.overrides) by cut/join,
        # reusing the exact button label and checkable-is-armed pattern _segments.py's
        # own button already established for the EMG-only separators list. Checkable,
        # so its own pressed state IS the armed flag _on_plot_clicked reads, same
        # reasoning as the segments tab's own button.
        self.btn_place_overrides = QPushButton("Place separators")
        self.btn_place_overrides.setCheckable(True)
        self.btn_place_overrides.toggled.connect(self._on_place_overrides_toggled)
        # P19: process AND write just this file, so a tuned file can be produced without
        # re-running the whole batch (reuses the Run screen's write machinery).
        self.btn_process_file = QPushButton("Process && write this file")
        self.btn_process_file.setEnabled(False)
        self.btn_process_file.setToolTip("Run and write output for the previewed file only.")
        self.btn_process_file.clicked.connect(self._process_this_file)
        bar.addWidget(self.qc_overview)
        bar.addSpacing(10)
        bar.addWidget(self.mech_window_label, 1)   # takes the stretch the plain addStretch(1) used to
        bar.addWidget(self.btn_place_overrides)
        bar.addWidget(self.btn_process_file); bar.addWidget(self.btn_export_fig)
        self._update_overrides_button()
        return band

    _MECH_WINDOW_TOOLTIP = (
        "The window of the recording actually analysed, after trimming to whole "
        "breaths. The Mechanics channel stack's time axis is zero at the start of "
        "this window; the EMG tabs show the file's own untrimmed clock.")

    _QC_OVERVIEW_TOOLTIP = (
        "Quality overview of the CURRENTLY PREVIEWED file's most recent test run — "
        "not a batch summary. See the file rail for every file's exclusion count.")

    def _process_this_file(self):
        name = self._previewed_file or self._selected_filename()
        if name:
            self.process_file_requested.emit(os.path.basename(name))

    def _reset_qc_overview(self):
        """Neutral QC chip + disabled 'Process & write' — the same nulstillingsdiscipline
        ``_forget_campbell`` already applies to the export button, extended to this chip
        and this button: clearing what a panel shows and clearing the widget that judges
        it must be the same act. Called from ``_clear_file_panels`` (a file switch/blank);
        the mechanics render (``_render_preview``) is what re-enables the button for a
        file that actually loaded.

        M-26: resets BOTH the Mechanics chip and the segments tab's own chip
        unconditionally — this runs on every file switch regardless of which shape/tab
        is currently active, so the one that is not showing must not be left stale
        either (it becomes visible again the moment the signal set changes back)."""
        for chip in (self.qc_overview, getattr(self, "segments_qc_overview", None)):
            if chip is None:
                continue
            _set_chip_text(chip, "QC:  —", fixed_tooltip=self._QC_OVERVIEW_TOOLTIP)
            chip.setProperty("status", "muted")
            chip.style().unpolish(chip)
            chip.style().polish(chip)
        self._mech_window_base_text = ""
        self.mech_window_label.setFullText("")
        self.mech_window_label.setToolTip(self._MECH_WINDOW_TOOLTIP)
        self._process_ready = False
        self.btn_process_file.setEnabled(False)
        if hasattr(self, "btn_process_segments_file"):
            self.btn_process_segments_file.setEnabled(False)
        # Same discipline for 'Place separators' (M-27): unarm and disable it here too,
        # not just re-sync it later on a SUCCESSFUL 'segments' render
        # (_render_segments_preview's own _update_separators_button call). Without this,
        # a file switch whose 'segments' job then fails (a bad separator, a crashed
        # worker) left the button exactly as the PREVIOUS file's render had set it —
        # enabled and possibly still CHECKED — while every plot it would hit-test
        # against had already been torn down by this same method a few lines up; a click
        # during that window would resolve the NEW file's name (_selected_filename)
        # against the OLD file's now-stale geometry. Unarming here closes the window
        # unconditionally, the same way the process button already is.
        if hasattr(self, "btn_place_separators"):
            if self.btn_place_separators.isChecked():
                self.btn_place_separators.setChecked(False)   # also clears _separators_armed
            self.btn_place_separators.setEnabled(False)
        # Same discipline for Mechanics' own 'Place separators' (segmentation
        # overrides) -- unarm and disable it on every file switch, for the identical
        # stale-geometry reason the comment above documents for that button.
        if hasattr(self, "btn_place_overrides"):
            if self.btn_place_overrides.isChecked():
                self.btn_place_overrides.setChecked(False)   # also clears _overrides_armed
            self.btn_place_overrides.setEnabled(False)

    def _qc_overview_not_assessed(self, detail, chip=None):
        """The chip's honest state while the test run itself failed or was skipped —
        called from ``_on_batch_result``'s error branches and from ``_on_job_done``'s
        'batch' failure branch. Purely presentational: no computed value changes.
        ``chip`` (M-26): the segments tab's own QC chip for an EMG-only 'batch' run,
        defaulting to the Mechanics chip unchanged."""
        chip = chip if chip is not None else self.qc_overview
        _set_chip_text(chip, f"QC:  not assessed — {short_error(str(detail))}",
                      fixed_tooltip=self._QC_OVERVIEW_TOOLTIP)
        chip.setProperty("status", "warn")
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    def _update_qc_overview(self, fr, chip=None):
        """P16: a persistent, at-a-glance quality summary of the current test run —
        breaths used/excluded plus a conservative flag for any non-physiological
        per-breath value (so a suspect run is obvious without reading the table).
        ``chip`` (M-26): see ``_qc_overview_not_assessed``."""
        chip = chip if chip is not None else self.qc_overview
        if getattr(fr, "role", "tidal") == "reference":
            # M-30: every breath here was deliberately TYPED as a manoeuvre, not
            # excluded — self-review finding: the ordinary "N used, M excluded" framing
            # below would misreport a fully successful reference-only file as if QC had
            # quietly dropped every breath, directly contradicting the success status
            # line shown alongside it.
            n_typed = len(getattr(fr, "manoeuvres", None) or {})
            _set_chip_text(chip, f"QC:  {n_typed} typed manoeuvre breath"
                          f"{'s' if n_typed != 1 else ''}, no tidal breathing",
                          fixed_tooltip=self._QC_OVERVIEW_TOOLTIP)
            chip.setProperty("status", "ok")
            chip.style().unpolish(chip)
            chip.style().polish(chip)
            return
        bt = getattr(fr, "breaths_table", None)
        total = len(fr.breaths) if getattr(fr, "breaths", None) else (len(bt) if bt is not None else 0)
        used = len(bt) if bt is not None else 0
        excl = max(0, total - used)
        flags = []
        if bt is not None and len(bt):
            for c in ("vt", "ti", "te", "ttot"):
                if c in bt.columns and (bt[c] <= 0).any():
                    flags.append(f"{c}≤0")
            for c in ("wobtotal", "vt", "ve"):          # core metrics must be finite
                if c in bt.columns and not np.isfinite(bt[c].to_numpy(dtype=float)).all():
                    flags.append(f"{c} NaN")
        excl_note = ""
        # fr.file, NOT self._selected_filename(): _on_batch_result's own fallback
        # (`result.files.get(cur) or next(iter(result.files.values()), None)`) can hand this
        # a DIFFERENT file's FileResult than the one currently selected (e.g. a stale batch
        # result racing a file switch, or cur missing from the result set) — checking the
        # selected name against fr's own exclusion count would then blame/clear the wrong
        # file. fr.file is a real FileResult field; getattr defends the same "fr may be a
        # plain SimpleNamespace" case _qc_overview's other fields already guard against.
        if excl and self._exclusion_carried_for(getattr(fr, "file", None)):
            excl_note = " (carried over from a previous recordings folder)"
        msg = f"QC:  {used} breaths used" + (f",  {excl} excluded{excl_note}" if excl else "")
        status = "ok"
        if flags:
            msg += "   ⚠ " + "; ".join(dict.fromkeys(flags))
            status = "warn"
        else:
            msg += "   ·  no flags"
        _set_chip_text(chip, msg, fixed_tooltip=self._QC_OVERVIEW_TOOLTIP)
        chip.setProperty("status", status)
        chip.style().unpolish(chip)
        chip.style().polish(chip)

    def _set_analysis_window(self, data):
        """D23 (UI-overhaul): word the trim window at the persistent spot beside the QC
        chip, from the same ``stage_mechanics_preview`` dict the mechanics stack itself is
        drawn from — ``startix``/``endix`` are indices into the file's own untrimmed
        arrays (``core.compute.trim``), and ``emg_flow`` is that untrimmed flow, so its
        length is the file's total duration regardless of whether trimming succeeded.
        Purely presentational: no computed value is read or changed here.

        M-37: the trim-window sentence is only HALF of what this label now shows —
        ``_refresh_reference_chip`` appends the file's own resolved IC reference (if
        any) inside the SAME eliding slot, so ``_mech_window_base_text`` (this method's
        own half) is stashed for that method to read rather than recomputed there."""
        fs = data.get("fs")
        if not fs:
            self._mech_window_base_text = ""
            self._refresh_reference_chip()
            return
        total_s = len(data.get("emg_flow", ())) / fs
        if data.get("trim_error"):
            # No breath could be segmented (see stage_mechanics_preview's TrimError
            # fallback, which hands back startix=0/endix=len(flow) — the WHOLE file, not
            # a real window). Self-review of this ticket flagged that feeding those into
            # the normal formula below reads as a successful trim to 0.00-<total> s,
            # while the QC chip is simultaneously reporting the failure right next to it.
            text = (f"Showing the raw, untrimmed file ({total_s:.2f} s) — "
                    "could not trim to whole breaths")
        else:
            start_s = data["startix"] / fs
            end_s = data["endix"] / fs
            trimmed_s = max(0.0, total_s - (end_s - start_s))
            text = f"Analysis window {start_s:.2f}–{end_s:.2f} s of {total_s:.2f} s"
            if trimmed_s > 0.005:                       # hide a rounding-only "0.00 s trimmed"
                text += f" ({trimmed_s:.2f} s trimmed)"
        self._mech_window_base_text = text
        self._refresh_reference_chip()

    def _reference_chip_text(self, name):
        """The short 'IC ref: …' fragment appended to ``mech_window_label`` (M-37), or
        ``''`` when ``name`` has no resolved IC reference at all — a file that simply
        does not use one shows no chip, rather than an unconditional 'IC ref: none'
        cluttering every ordinary analysis. Reads ``resolve_reference`` (M-34), so this
        already reports a group-default or the file's own typed breath, not only an
        explicit ``processing.references`` entry — exactly like the chip's job
        ('viser den') requires after ANY of the three ways a reference can resolve."""
        from respmech.core.analysis.references import resolve_reference  # noqa: PLC0415

        if not name:
            return ""
        ref = resolve_reference(name, "ic", self.state.settings)
        if ref is None:
            return ""
        where = "this file" if ref.file == name else ref.file
        if not ref.breaths:
            return f"IC ref: {where}"
        n = len(ref.breaths)
        breath_txt = f"breath {ref.breaths[0]}" if n == 1 else f"{n} breaths"
        return f"IC ref: {where} ({breath_txt})"

    def _refresh_reference_chip(self):
        """Recompose ``mech_window_label`` from its stashed trim-window half
        (``_mech_window_base_text``) plus a freshly-resolved reference chip for the
        CURRENTLY SELECTED file — called both from a real render (``_set_analysis_
        window``) and from every write path that can change what the reference
        resolves to (``_set_reference``, ``_open_reference_picker``) without waiting
        for the next full re-render. The combined text is stashed as the tooltip too
        (appended after the fixed explanation, mirroring ``_set_chip_text``'s
        ``fixed_tooltip`` two-step for ``qc_overview``) so the full, un-elided sentence
        stays one hover away exactly like every other use of ElidingLabel here."""
        base = getattr(self, "_mech_window_base_text", "")
        ref_text = self._reference_chip_text(self._selected_filename())
        text = f"{base}  ·  {ref_text}" if (base and ref_text) else (ref_text or base)
        self.mech_window_label.setFullText(text)
        tip = self._MECH_WINDOW_TOOLTIP
        if text:
            tip += "\n\n" + text
        self.mech_window_label.setToolTip(tip)

    # -- P17 crosshair + figure export -------------------------------------
    def _on_mech_mouse_moved(self, evt):
        if not self._channel_plots or not self._crosshair_lines:
            return
        pos = evt[0]
        for p, ln in zip(self._channel_plots, self._crosshair_lines):
            if p.sceneBoundingRect().contains(pos):
                mp = p.getViewBox().mapSceneToView(pos)
                for l in self._crosshair_lines:
                    l.setPos(mp.x()); l.show()
                axis = p.getAxis("left")
                lab = axis.labelText or "value"
                # only a named axis has a real unit to report; "value" is the no-name fallback
                unit = getattr(axis, "_unit", "") if lab != "value" else ""
                suffix = f" {unit}" if unit else ""
                self.crosshair_label.setText(f"t = {mp.x():.3f} s     {lab} = {mp.y():.3g}{suffix}")
                return

    def _export_campbell(self):
        from PySide6.QtWidgets import QFileDialog       # noqa: PLC0415
        # M-17 (R7): the export follows whichever diagram the panel is actually showing —
        # "Campbell" with Poes declared, "Flow-volume loop" without it.
        is_campbell = Capabilities.from_settings(self.state.settings).poes
        panel_title = "Campbell diagram" if is_campbell else "Flow-volume loop"
        base = (self._previewed_file or "campbell").rsplit(".", 1)[0]
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {panel_title}", f"{base} – {panel_title}.png",
            "PNG image (*.png);;PDF document (*.pdf)")
        if not path:
            return
        # Export LIGHT, whatever the app is wearing. The on-screen figure follows the theme,
        # so in dark mode saving the live facecolor produced a near-black PNG/PDF — which is
        # a figure destined for a report or a paper. The batch writers already pin their
        # output light (core/plot_style) for the same reason; this is the interactive export
        # catching up. Re-rendered rather than recoloured: the dark palette's brightened
        # traces score barely 2:1 on white, so only a genuine light redraw is legible.
        breaths = getattr(self, "_campbell_breaths", None)
        redrawn = False
        try:
            if breaths is not None and ((_theme is not None and _theme.is_dark())
                                         or self._selected_breath is not None):
                # (a marked breath also forces a redraw: its green loop is screen-only)
                # set BEFORE the call it guards: _draw_campbell_or_loop begins by clearing
                # the figure, so a failure part-way through still leaves the on-screen
                # diagram destroyed and the finally-branch below is the only thing that puts
                # it back
                redrawn = True
                self._loop_export = True
                dark = _theme is not None and _theme.is_dark()
                self._draw_campbell_or_loop(breaths, pal=_theme._PLOT_LIGHT if dark else None)
            # The export is a stand-alone figure with no panel header around it, so it gets
            # the title back that the on-screen panel leaves to its header — and then loses
            # it again straight away. Leaving it set showed the title twice on screen, once
            # in the header and once in the figure, on every light-mode export (dark mode
            # hid the leak, because it redraws the figure in the finally-branch below).
            fig = self.campbell.figure
            try:
                for _ax in fig.axes:
                    _ax.set_title(panel_title)
                fig.savefig(path, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            finally:
                for _ax in fig.axes:
                    _ax.set_title("")
                self.campbell.draw_idle()
            self._set_status(f"Saved {panel_title} → {path}")
        except Exception as e:                          # noqa: BLE001
            self._set_status(f"Could not save figure: {short_error(str(e))}")
        finally:
            self._loop_export = False
            if redrawn:                                 # put the on-screen figure back
                try:
                    self._draw_campbell_or_loop(breaths)
                except Exception:                       # pragma: no cover - defensive
                    pass

    def _open_mech_advanced(self):
        """Every mechanics setting: breath segmentation, work-of-breathing source, the volume
        and drift corrections, resampling, and the per-file breath counts. They left the Setup
        screen and live here, beside the Mechanics preview they shape, editing the model
        directly (like the ECG params) so Setup never reverts them on a tab change."""
        from respmech.core.settings import BreathCountEntry
        from respmech.ui.advanced_dialog import AdvancedDialog, Field, apply_values
        s = self.state.settings
        seg, peak = s.processing.segmentation, s.processing.segmentation.peak
        vol, samp, wob, ptp = (s.processing.volume, s.processing.sampling,
                               s.processing.wob, s.processing.ptp)
        fields = [
            ("seg", Field("method", "Signal used to split breaths", "choice",
                          "processing.segmentation.method",
                          "Which signal marks where each breath begins and ends.",
                          options=[("Flow", "flow"), ("Volume", "volume")])),
            ("wob", Field("calc_from", "Work of breathing from", "choice",
                          "processing.wob.calc_from",
                          "One averaged breath (default), or each breath then averaged.",
                          options=[("Average", "average"), ("Individual", "individual")])),
            ("vol", Field("integrate_from_flow", "Calculate volume from flow", "bool",
                          "processing.volume.integrate_from_flow",
                          "Derive volume by integrating flow instead of a separate channel.")),
            ("vol", Field("correct_drift", "Correct volume drift", "bool",
                          "processing.volume.correct_drift",
                          "Remove slow baseline drift from the volume trace; ON by default.")),
            ("vol", Field("correct_trend", "Correct end-expiratory trend", "bool",
                          "processing.volume.correct_trend",
                          "Remove a between-breath trend in end-expiratory lung volume.")),
            ("vol", Field("trend_method", "Trend interpolation", "choice",
                          "processing.volume.trend_method",
                          "How the end-expiratory trend is interpolated between breaths.",
                          options=[("Linear", "linear"), ("Nearest", "nearest"),
                                   ("Cubic", "cubic"), ("Quadratic", "quadratic"),
                                   ("Previous", "previous"), ("Next", "next")],
                          depends_on="correct_trend")),
            ("vol", Field("trend_peak_min_prominence_frac",
                          "Trend anchor — minimum breath depth", "float",
                          "processing.volume.trend_peak_min_prominence_frac",
                          "How deep a trough must be, as a fraction of the recording's own "
                          "volume range, to count as end-expiratory.",
                          lo=0.001, hi=0.999, step=0.01, decimals=3,
                          depends_on="correct_trend",
                          note="Scales with each recording, so it works at any tidal volume "
                               "and in any volume unit. Lower it only if breaths are missed "
                               "on a recording that also contains a large manoeuvre.")),
            ("vol", Field("trend_peak_min_distance_s", "Trend anchor — minimum spacing",
                          "float", "processing.volume.trend_peak_min_distance_s",
                          "Minimum time between two detected end-expiratory troughs.",
                          lo=0.001, hi=60.0, step=0.05, decimals=4, suffix=" s",
                          depends_on="correct_trend")),
            ("vol", Field("trend_peak_min_height",
                          "Trend anchor — absolute threshold (legacy)", "float",
                          "processing.volume.trend_peak_min_height",
                          "Fixed depth below the recording's HIGHEST volume that a trough "
                          "must reach. Overrides the breath-depth rule above. Set to Auto "
                          "(the lowest value) to use 'Trend anchor — minimum breath depth' "
                          "instead, which is the default.",
                          # the sentinel sits just BELOW zero because 0 is itself a legal
                          # legacy value ("no absolute gate"); sharing the minimum would
                          # silently rewrite such an analysis to Auto on the next OK. With
                          # decimals=3 the only reachable negative IS the sentinel.
                          lo=-0.001, hi=1_000_000.0, step=0.05, decimals=3,
                          # Just "Auto": the old "Auto — use minimum breath depth" is a
                          # sentence inside a numeric field and needed 473 px on Windows
                          # font metrics against the ~340 px a column can give it, so it
                          # was permanently clipped mid-word. What it said now lives in the
                          # tooltip above, which has room for it.
                          auto_text="Auto",
                          depends_on="correct_trend",
                          note="Only for reproducing an analysis made before v2.3.3. It is "
                               "measured against the whole recording, so a value larger than "
                               "the recording's volume range finds no troughs at all.")),
            ("vol", Field("inverse_flow", "Invert the flow signal", "bool",
                          "processing.volume.inverse_flow",
                          "Flip the flow sign if inspiration reads positive.")),
            ("vol", Field("inverse_volume", "Invert the volume signal", "bool",
                          "processing.volume.inverse_volume", "Flip the volume sign.")),
            ("samp", Field("resample", "Resample before analysis", "bool",
                           "processing.sampling.resample",
                           "Resample every recording to a common rate first; off by default.")),
            ("samp", Field("resample_to_frequency", "Resample to", "int",
                           "processing.sampling.resample_to_frequency",
                           "Target rate (Hz). Keep ≥ ~1000 Hz with EMG channels present.",
                           lo=1, hi=1_000_000, step=100, suffix=" Hz",
                           depends_on="resample")),
            ("seg", Field("buffer", "Breath-separation debounce", "int",
                          "processing.segmentation.buffer",
                          "How long the flow must stay reversed before a phase boundary "
                          "is accepted, suppresses spurious boundaries from noise near "
                          "zero flow.",
                          lo=0, hi=100_000, step=10,
                          # Stored and edited in samples (an existing analysis's TOML is
                          # unaffected), but samples alone hide how long that debounce
                          # actually holds at the rate the run analyses at — the ONE thing
                          # about this setting that changes with the recording (D29). The
                          # note is this dialog's *opening* value; _refresh_buffer_hint
                          # below keeps it live as buffer/resample/resample_to_frequency
                          # are edited.
                          note=_buffer_debounce_hint(
                              seg.buffer, samp.resample, samp.resample_to_frequency,
                              s.input.format.sampling_frequency))),
            ("peak", Field("height", "Breath peak — minimum height", "float",
                           "processing.segmentation.peak.height",
                           "Minimum peak height for breath detection (volume-based).",
                           lo=0.0, hi=1_000_000.0, step=0.01, decimals=4)),
            ("peak", Field("distance_s", "Breath peak — minimum distance", "float",
                           "processing.segmentation.peak.distance_s",
                           "Minimum time between detected breath peaks.",
                           lo=0.0, hi=60.0, step=0.05, decimals=4, suffix=" s")),
            ("peak", Field("width_s", "Breath peak — minimum width", "float",
                           "processing.segmentation.peak.width_s",
                           "Minimum width of a detected breath peak.",
                           lo=0.0, hi=60.0, step=0.05, decimals=4, suffix=" s")),
            ("wob", Field("avg_resampling_obs", "Average-breath resampling points", "int",
                          "processing.wob.avg_resampling_obs",
                          "Points each breath is resampled to for the average breath / WOB.",
                          lo=10, hi=100_000, step=10)),
            ("ptp", Field("baseline_window_s", "PTP baseline window", "float",
                          "processing.ptp.baseline_window_s",
                          "End-expiratory window whose mean is the PTP baseline.",
                          lo=0.0, hi=1.0, step=0.01, decimals=4, suffix=" s",
                          # Only while the opt-in PEEPi analysis is on (Setup ▸ Intrinsic PEEP):
                          # the one-baseline rule means the pre-flow area stays OUT of every PTP
                          # column, and a reader comparing them should be told where it went.
                          note=("The pre-flow area of the PEEPi deflection (int_oes_preflow) is "
                                "reported in its own column and is never added to PTP."
                                if s.processing.pressure.peepi.enabled else None))),
            # M-37: only the four fields the ticket names -- the rest of LungVolumeSettings/
            # IcSettings (baseline_pattern, the EELV_UNSTABLE/NOT_REPEATABLE/LOW_EFFORT
            # thresholds, reject_flags) stay TOML-only for now, same "surface the most-used
            # fields, preserve the rest on round-trip" rule the whole dialog already follows.
            ("lv", Field("require_references", "Require linked references", "bool",
                        "processing.lung_volume.require_references",
                        "Make a reference source named in Setup that is missing from the "
                        "analysed files a hard error instead of a caution.")),
            ("ic", Field("eelv_tracking", "EELV tracking", "choice",
                        "processing.lung_volume.ic.eelv_tracking",
                        "Whether each tidal breath's operating IC follows its own "
                        "end-expiratory drift, same-file reference only.",
                        options=[("None — reference IC held fixed", "none"),
                                 ("Within this file", "within_file")])),
            ("ic", Field("aggregate", "Repeat-IC aggregate", "choice",
                        "processing.lung_volume.ic.aggregate",
                        "How repeated IC manoeuvres in the same reference are combined.",
                        options=[("Mean", "mean"), ("Median", "median")])),
            ("ic", Field("preceding_breaths", "Preceding breaths for EELV baseline", "int",
                        "processing.lung_volume.ic.preceding_breaths",
                        "How many tidal breaths right before the manoeuvre set its "
                        "end-expiratory baseline.",
                        lo=0, hi=1000, step=1)),
            # The four MFVL fields most often changed; efl_present_min_pct stays TOML-only.
            ("mfvl", Field("source", "MFVL source", "choice",
                          "processing.mfvl.source",
                          "Which forced expiration the tidal breaths are placed against: the "
                          "single attempt with the largest FVC, or the per-volume maximum flow "
                          "across every attempt in the file.",
                          options=[("Largest-FVC attempt", "single"),
                                   ("Envelope of all attempts", "envelope")])),
            ("mfvl", Field("efl_rel_tol", "EFL tolerance (relative)", "float",
                          "processing.mfvl.efl_rel_tol",
                          "A tidal sample counts as flow-limited when its flow reaches the MFVL "
                          "flow at that volume minus this fraction (0 = it must reach the "
                          "curve).",
                          lo=0.0, hi=1.0, step=0.01, decimals=3)),
            ("mfvl", Field("efl_abs_tol_lps", "EFL tolerance (absolute)", "float",
                          "processing.mfvl.efl_abs_tol_lps",
                          "Extra flow, in L/s, a tidal sample may fall short of the MFVL and "
                          "still count as flow-limited.",
                          lo=0.0, hi=10.0, step=0.01, decimals=3, suffix=" L/s")),
            ("mfvl", Field("mvv_fev1_multiplier", "MVV = FEV1 ×", "float",
                          "processing.mfvl.mvv_fev1_multiplier",
                          "Multiplier that estimates maximal voluntary ventilation from FEV1 "
                          "when no measured MVV is given (ATS/ACCP 2003 uses 40).",
                          lo=0.0, hi=100.0, step=1.0, decimals=1)),
            # Breathing pattern: two switches, both opt-in and both flow/volume only.
            ("bp", Field("extended", "Extended per-breath pattern columns", "bool",
                        "processing.breathing_pattern.extended",
                        "Add mean inspiratory and expiratory flow, the volume moved in each "
                        "phase, the instantaneous breathing rate and the timing of the peak "
                        "flows to every breath.")),
            ("bp", Field("variability", "Per-file variability (CV)", "bool",
                        "processing.breathing_pattern.variability",
                        "Add the coefficient of variation of VT, Ti, Te, Ttot and Ti/Ttot over "
                        "the included breaths, and the number of breaths, to each file's "
                        "average row (needs at least 3 breaths).")),
        ]
        # M-17 (R7): "Work of breathing" and "Pressure–time product" are meaningless without
        # a Poes trace — WOB/PTP are never computed for a Poes-less analysis (M-14's
        # compute-guards). Filtering the FLAT fields list here, rather than the sections
        # table below, drops both cards automatically (their keys leave `remaining` empty,
        # so the "if picked" check skips them — see below) instead of dumping their fields
        # onto "Other". Nothing is lost: unlike a value the user typed, these two cards'
        # settings simply are not offered while the shape cannot use them, and return the
        # moment Poes is added back to the signal set (Setup ▸ Signals) — same "hidden
        # follows the set, never workflow progress" rule as every other R7 surface.
        if not Capabilities.from_settings(s).poes:
            _hidden_without_poes = {"calc_from", "avg_resampling_obs", "baseline_window_s"}
            fields = [(grp, f) for grp, f in fields if f.key not in _hidden_without_poes]
        # Breathing pattern is flow and volume only, so it needs Flow -- absent from an
        # EMG-only analysis, where the card would offer switches with nothing to switch on.
        if not Capabilities.from_settings(s).flow:
            fields = [(grp, f) for grp, f in fields if grp != "bp"]
        owner = {"seg": seg, "peak": peak, "vol": vol, "samp": samp, "wob": wob, "ptp": ptp,
                "lv": s.processing.lung_volume, "ic": s.processing.lung_volume.ic,
                "mfvl": s.processing.mfvl, "bp": s.processing.breathing_pattern}
        values = {f.key: getattr(owner[grp], f.key) for grp, f in fields}
        # breath counts round-trip as one 'file = count' line each, edited as text
        bc_field = Field("breath_counts", "Breath-count overrides", "text",
                         "processing.breath_counts",
                         "Per-file override of the breath count used for per-minute scaling — "
                         "one 'filename = count' per line. Blank = each file's detected count.",
                         placeholder="one 'filename = count' per line",
                         note="Excluded breaths are NOT set here — right-click a breath in "
                              "the channel plots above and choose Excluded or Tidal; the file rail "
                              "shows each file's exclusion count.")
        bc_text = "\n".join(f"{e.file} = {e.count}" for e in s.processing.breath_counts)
        # Which card each setting is shown on, in display order. Grouping only: the flat
        # ``fields`` list above still drives building, reading and committing, so a setting
        # cannot change meaning by being moved between cards. A key that gates others
        # (``correct_trend``, ``resample``) must come first WITHIN its own card, because
        # ``depends_on`` resolves against widgets already built.
        sections = [
            ("Breath detection", ["method", "buffer", "height", "distance_s", "width_s"]),
            ("Work of breathing", ["calc_from", "avg_resampling_obs"]),
            ("Volume", ["integrate_from_flow", "correct_drift",
                        "inverse_flow", "inverse_volume"]),
            ("End-expiratory trend", ["correct_trend", "trend_method",
                                      "trend_peak_min_prominence_frac",
                                      "trend_peak_min_distance_s", "trend_peak_min_height"]),
            ("Sampling", ["resample", "resample_to_frequency"]),
            ("Pressure–time product", ["baseline_window_s"]),
            ("Lung volumes", ["require_references", "eelv_tracking", "aggregate",
                              "preceding_breaths", "source", "efl_rel_tol",
                              "efl_abs_tol_lps", "mvv_fev1_multiplier"]),
            ("Breathing pattern", ["extended", "variability"]),
            # not "Breath-count overrides" — the one field on this card is already called
            # that, and a card whose title repeats its only row reads like a mistake
            ("Per-file overrides", ["breath_counts"]),
        ]
        by_key = {f.key: f for _g, f in fields}
        by_key[bc_field.key] = bc_field
        # Cards are filled by CONSUMING from a working copy, which makes the table's three
        # failure modes harmless instead of silent:
        #   * a key listed twice picks the field once — listing it again finds nothing, so
        #     there is no second widget to shadow the first in ``_widgets`` and swallow its
        #     edits (measured before this: 21 rows built against 20 widgets);
        #   * a mistyped key matches nothing and its real field simply stays unplaced;
        #   * anything left unplaced lands on "Other" rather than vanishing from the dialog
        #     while still being committed on OK — the worst possible outcome for a
        #     scientific parameter.
        # An empty card is dropped, so a typo shows up as a setting in the wrong place
        # rather than as a mysterious empty box.
        remaining = dict(by_key)
        all_fields = []
        for title, keys in sections:
            picked = [remaining.pop(k) for k in keys if k in remaining]
            if picked:
                all_fields.append((title, picked))
        if remaining:
            all_fields.append(("Other", list(remaining.values())))
        vals = dict(values); vals["breath_counts"] = bc_text

        def _commit(staged):
            """Write ``staged`` (an ``edited_values()`` dict) onto the model and, if
            anything actually changed, mark the analysis modified and recompute. Shared by
            Apply (mid-dialog) and OK (on close) so the two commit through exactly the same
            path — Apply is not a second, parallel way to write these settings."""
            changed = False
            for grp, f in fields:
                if f.key in staged and apply_values(owner[grp], {f.key: staged[f.key]}):
                    changed = True
            # only when the user actually edited the box — ``staged`` carries edited keys
            # only, so an untouched commit must not re-parse (and re-write) the entries
            if "breath_counts" in staged:
                parsed = _parse_breath_counts(staged["breath_counts"], BreathCountEntry,
                                              folder=s.input.folder)
                if parsed != s.processing.breath_counts:
                    s.processing.breath_counts = parsed
                    changed = True
            if not changed:
                return False              # commit without an edit: no dirty flag, no recompute
            self.settings_edited.emit()
            self._request_autorun()      # mechanics feed mech + batch (+ the noise clip)
            return True

        def _trend_hint(v):
            """The live line under the thresholds: end-expiratory trough count while trend
            correction is on (the number that decides whether the run succeeds and cannot
            be read off the settings alone), or a live BREATH count while it is off (see
            _advanced_live_breath_count) — the number this dialog otherwise gives no
            feedback on at all until the dialog is closed and the preview redraws."""
            probe = self._trend_probe
            # The probe is the drift-corrected volume from the LAST rendered preview. This
            # same dialog also stages the settings that BUILD that volume, so once any of
            # them is touched the probe describes a signal the run will not use — and a
            # confident count about the wrong signal is worse than no count. Shared by both
            # branches below.
            stale = ({k: v.get(k) for k in _TREND_PROBE_KEYS} != self._trend_probe_shape)
            if not v.get("correct_trend"):
                if probe is None or not len(probe):
                    return "No previewed file to count breaths in yet."
                if stale:
                    return ("Volume conditioning changed — press OK, then reopen this "
                            "dialog for a breath count.")
                n = self._advanced_live_breath_count(probe, v)
                if n is None:
                    return "Could not count breaths with these breath-detection thresholds."
                return (f"In {self._trend_probe_file}: {n} breath{'s' if n != 1 else ''} "
                        "found with these thresholds.")
            from respmech.core import compute
            need = compute._TREND_MIN_ANCHORS.get(v["trend_method"], 2)
            if probe is None or not len(probe):
                return f"Needs at least {need} end-expiratory troughs in every file."
            if stale:
                return ("Volume conditioning changed — press OK, then reopen this dialog "
                        f"for a trough count. Needs at least {need}.")
            n = compute.trend_anchors(
                probe, s.input.format.sampling_frequency,
                min_height=v["trend_peak_min_height"],
                min_prominence_frac=v["trend_peak_min_prominence_frac"],
                min_distance_s=v["trend_peak_min_distance_s"]).size
            return (f"In {self._trend_probe_file}: volume range {float(np.ptp(probe)):.2f}, "
                    f"{n} trough(s) found — "
                    + ("OK." if n >= need
                       else f"NOT ENOUGH (needs {need}); the run will fail this file."))

        # Not modal (see AdvancedDialog.exec): the channel stack behind this dialog is
        # exactly what the fields on it shape, so Apply has to be visible while it redraws.
        # derived_debounce_ms=250: the live breath count runs a real peak search, not
        # trivial arithmetic like the ECG/EMG modals' derived lines, so it must not fire on
        # every keystroke.
        dlg = AdvancedDialog("Mechanics — advanced", all_fields, vals, parent=self,
                             intro="How breaths are detected, how work of breathing is "
                                   "computed, and the volume/drift corrections. The defaults "
                                   "suit ordinary recordings.",
                             derived=_trend_hint, modal=False, on_apply=_commit,
                             derived_debounce_ms=250)

        def _refresh_buffer_hint(*_args):
            """Keep the debounce note (built above from this dialog's OPENING values)
            reading the analysis rate the user is actually staging, not the one the dialog
            opened with — the whole reason D29 asked for a live line rather than a static
            one. Trivial arithmetic, so unlike _trend_hint this runs synchronously on
            every edit, no debounce timer needed."""
            lbl = dlg.note("buffer")
            if lbl is None:
                return
            v = dlg.values()
            lbl.setText(_buffer_debounce_hint(
                v["buffer"], v["resample"], v["resample_to_frequency"],
                s.input.format.sampling_frequency))

        for _key in ("buffer", "resample", "resample_to_frequency"):
            _sig = getattr(dlg.widget(_key), "valueChanged", None) \
                or getattr(dlg.widget(_key), "toggled", None)
            if _sig is not None:
                _sig.connect(_refresh_buffer_hint)

        if dlg.exec() != QDialog.Accepted:
            return
        _commit(dlg.edited_values())          # see AdvancedDialog.edited_values

    # -- channel preview (Mechanics), synchronous render -------------------
    def _preview(self):
        path = self._current_file()
        if not path:
            return
        try:
            data = stage_mechanics_preview(self.state.settings, path)
            self._render_preview(data)               # render also inside the guard
        except Exception:                            # noqa: BLE001 — copyable error card
            detail = traceback.format_exc()
            self._set_status(f"Channel preview failed — {short_error(detail)}")
            for p in ("channels", "raw"):
                self._overlays[p].show_error("Channel preview failed", detail)

    def _set_mech_caption(self, total, excluded):
        """Update the Mechanics tab's persistent caption (breath count, exclusion count,
        the click-to-exclude instruction) — see self.mech_caption in _build_mech_tab."""
        n = f"{total} breath{'s' if total != 1 else ''}"
        excl = f", {excluded} excluded" if excluded else ""
        self.mech_caption.setFullText(
            f"{n}{excl} — click a shaded breath to mark it; right-click to exclude it or set its "
            f"type (red = excluded).")

    def _render_preview(self, data):
        """Draw the mechanics channel stack + breath overlays and the raw EMG
        stack from staged arrays, fully synchronously. GUI-thread only; this is
        the contract _preview() (and every existing test that calls it) relies
        on — one call in, everything on screen when it returns.

        The live reactive job dispatch does NOT call this: _RENDER['mech'] is
        _render_preview_async, which runs the same three stages (below) but
        yields the event loop between the two expensive ones — see its
        docstring and ticket D15."""
        self._render_preview_stage1(data)
        self._render_preview_stage2(data)
        self._render_preview_stage3(data)

    def _render_preview_async(self, data):
        """The reactive 'mech' job's render entry point (_RENDER['mech'] in
        screen.py). Same three stages as _render_preview, but stage2 (breath
        overlays) and stage3 (raw EMG stack + detail/result repaint + status)
        are each posted with ``QTimer.singleShot(0, …)`` instead of run back to
        back in one Python call.

        Those two stages are what measured 3.4-4.3s of unbroken GUI-thread block
        on a real six-minute recording, on top of an 8.7-11.7s deferred-redraw
        stall once control finally returned to Qt — a total ~12s freeze on a
        single ▶ file-step (ticket D15). Splitting them lets Qt's event loop turn
        in between: the busy overlay's indeterminate QProgressBar can animate,
        and a click or scroll during the gap is not simply queued up behind a
        frozen thread. The panel-level item-count reduction (BreathSpansItem, see
        _plot_helpers.py) is the primary fix for the total time; this split is
        what keeps that remaining time from reading as a hang.

        _mech_render_gen is bumped by stage1 (and by _reset_breath_state) so a
        stale continuation — the user stepped to another file before this one's
        deferred stage fired — recognises it no longer owns the panels and
        abandons without touching them; see _render_preview_async_stage2/3.
        This chain owns stopping the 'channels'/'raw' busy overlays itself
        (_PANELS['mech']) once stage3 completes — screen.py's _on_job_done skips
        its own generic post-render stop for kind == 'mech' for exactly that
        reason."""
        self._render_preview_stage1(data)
        gen = self._mech_render_gen
        QTimer.singleShot(0, lambda: self._render_preview_async_stage2(gen, data))

    def _render_preview_async_stage2(self, gen, data):
        if gen != self._mech_render_gen:
            return                      # superseded — the newer render owns the panels now
        try:
            self._render_preview_stage2(data)
        except Exception:               # noqa: BLE001 — same surface a sync failure gets
            self._fail_async_mech_render(traceback.format_exc())
            return
        QTimer.singleShot(0, lambda: self._render_preview_async_stage3(gen, data))

    def _render_preview_async_stage3(self, gen, data):
        if gen != self._mech_render_gen:
            return
        try:
            self._render_preview_stage3(data)
        except Exception:               # noqa: BLE001
            self._fail_async_mech_render(traceback.format_exc())
            return
        for p in _PANELS["mech"]:
            self._overlays[p].stop()

    def _fail_async_mech_render(self, detail):
        """Mirror _on_job_done's own 'display error' card + status line for a
        failure that happens in a DEFERRED stage — outside _on_job_done's own
        try/except, since that only wraps the synchronous stage1 call."""
        label = _KIND_LABEL["mech"]
        msg = f"{label} — display error: {short_error(detail)}"
        self._set_status(msg)
        for p in _PANELS["mech"]:
            self._overlays[p].show_error(msg, detail)
        self._update_actions(status=False)

    def _render_preview_stage1(self, data):
        """Clear stale panel state and draw the channel curves (flow/volume always;
        poes/pgas/pdi only when the signal set declares them present, see
        ``present_channels`` below) + crosshairs. Kept cheap and synchronous even on
        the async path: curve-plotting is not the cost this ticket addresses
        (decimation/clipToView already bound it, see plot_perf.tune) — the breath
        overlays and the raw EMG stack are."""
        self._mech_render_gen += 1
        t = data["t"]
        series = data["series"]
        self._trim_offset_s = data["startix"] / data["fs"]  # trimmed span -> absolute EMG time
        self._update_overrides_button()   # re-sync for the CURRENT method/run state
        self._set_analysis_window(data)                      # D23: persistent trim-window label
        self._breaths = list(data["spans"])
        # M-31: the type menu's 'Suggested: FVC' hint — .get() so a stale suggestion from
        # the previously-viewed file/tab can never survive (stage_emg_segments_preview's
        # own dict carries no such key, since a segment has no expiration to judge).
        self._suggested_fvc = data.get("suggested_fvc")
        # Drop the click-to-toggle maps NOW, not just on a file switch (_reset_breath_state):
        # a same-file settings edit re-dispatches 'mech' WITHOUT going through a file switch,
        # so without this, a click landing in the stage1->stage2 gap (the async path's
        # QTimer.singleShot(0, ...) turn) would hit-test against the PRE-edit breath spans —
        # still non-empty, so _on_plot_clicked's guard would not catch it — and could toggle
        # an exclusion for a breath number that no longer matches what stage2 is about to
        # draw. Found in self-review of this ticket. Clearing them makes that gap read as
        # "no breaths known yet" (the same guard _reset_breath_state already relies on),
        # exactly like the very first render of a file.
        self._breath_spans = {}
        self._breath_regions = {}
        self._breath_texts = {}
        # error-only, not stop(): a legitimately in-flight async render's busy overlay
        # (see _render_preview_async) must survive this — stopping it here was found,
        # in this ticket, to be the reason the busy spinner "froze" instead of animating:
        # Qt has no chance to repaint the resulting hide() until the render's own block
        # ends, so the LAST painted frame just sits there for the whole thing.
        self._clear_panel_errors("channels", "raw")
        self.plots.clear()
        self._channel_plots = []
        self._crosshair_lines = []                       # cleared with the plots; rebuilt below
        pal = _plot_pal()
        # Per ROW, not for the stack as a whole, and eftergivende for the viewport this stack
        # is actually shown in (D14) — see _update_mech_stack_floor/theme.set_stack_floor.
        # Five channels sharing a flat 130 px container is 10 px of trace each.
        self._update_mech_stack_floor()
        # A reduced signal set (Flow only / Flow + Poes, already reachable today via
        # channel_setup_dialog.py's declared-roles gating) omits the absent pressure
        # keys from `series` entirely (core/pipeline.py/ui/workers.py's "None means
        # absent" contract) — filter the row list to what is actually present, rather
        # than indexing `series[key]` unconditionally and raising KeyError. This is a
        # crash guard, not the full relevance-driven layout a later ticket still owns
        # (the stack floor above still sizes for all five rows regardless of how many
        # are actually drawn — a cosmetic gap, not a correctness issue).
        present_channels = [c for c in _CHANNELS if c[0] in series]
        for i, (key, label, colour) in enumerate(present_channels):
            p = self.plots.addPlot(row=i, col=0, axisItems={"left": SciAxis(orientation="left")})
            p.showGrid(x=True, y=True, alpha=pal["grid_alpha"])
            # name over unit on two centred lines — "Poes (cmH₂O)" on one line is taller than
            # the short stacked plot and the rotated label overran its neighbour
            name, _, unit = label.partition(" (")
            p.getAxis("left").set_channel_label(name, unit.rstrip(")"))
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            # scroll over the graph zooms time only; y-zoom is reserved for the y-axis itself
            _restrict_body_wheel_to_x(p.getViewBox())
            y = series[key]
            p.plot(t[:len(y)], y, pen=_pen(pal["channels"].get(key, colour)))
            # P22: a subtle zero-reference baseline on every channel, so the sign and
            # excursion of each signal read at a glance.
            p.addLine(y=0, pen=pg.mkPen(pal["separator"], width=1, style=Qt.DashLine))
            self._limit_x(p, t[:len(y)])          # can't zoom/pan past the data
            self._channel_plots.append(p)
        # x-values on the bottom channel only; y-zoom is NOT shared — flow, volume and the
        # pressures are different units, so a common y-range would flatten most of them.
        # D23 (UI-overhaul): labelled distinctly from the EMG/ECG stacks' "Time (s)" — this
        # axis is zero-based on the TRIMMED analysis window (``t = data["t"]`` above), while
        # the EMG tabs plot the file's own untrimmed clock (see ``_render_raw_stack``'s
        # ``np.arange(emg.shape[0]) / fs``). Same label on two different clocks let a user
        # read a time on this stack, switch tabs, and land on the wrong breath.
        self._style_channel_stack(self.plots, self._channel_plots, link_y=False,
                                  time_label="Time from analysis start (s)")
        # P17: one crosshair line per channel, hidden until the cursor enters the stack
        self._crosshair_lines = []
        for p in self._channel_plots:
            ln = p.addLine(x=0, pen=pg.mkPen(pal["separator"], width=1, style=Qt.DashLine))
            ln.hide()
            self._crosshair_lines.append(ln)
        # D24: gated jointly with _run_active — a file switch mid-run (allowed; only the
        # WRITE actions are locked, see set_run_active) reaches this same render path and
        # must not silently re-enable the button a running batch has locked.
        self._process_ready = True
        self.btn_process_file.setEnabled(not self._run_active)

    def _render_preview_stage2(self, data):
        """Breath overlays on the mechanics stack — the single most expensive
        piece measured in ticket D15 (1.6-2.4s of the ~3.4-4.3s synchronous
        block), now a handful of BreathSpansItem paints instead of up to 1210
        individual QGraphicsItems."""
        self._draw_breath_overlays(data["spans"], data["label_y"],
                                   carried=self._exclusion_carried_for(data["name"]))
        self._update_override_lines(data["name"])   # cut markers, own pen

    def _render_preview_stage3(self, data):
        """The raw EMG stack, the detail/result overlay repaint, and the status
        line/caption/trend-probe bookkeeping that used to close out
        _render_preview in one go."""
        fs = data["fs"]
        spans = data["spans"]
        self._render_raw_stack(data["emg"], fs, data.get("emg_flow"))
        # breaths are now known -> (re)number any EMG detail/result already rendered
        self._repaint_view_breaths("detail")
        self._repaint_view_breaths("result")

        self._previewed_file = data["name"]
        # what the trend detector sees for THIS file — feeds the live anchor count in
        # Mechanics — advanced… (see _trend_hint), together with the volume-conditioning
        # settings it was produced under, so the count can be withdrawn once they change
        self._trend_probe = data.get("vol_drift")
        self._trend_probe_file = data["name"]
        self._trend_probe_shape = self._trend_probe_settings()
        # K-035: a boundary breath trim keeps but cannot verify as complete — appended to
        # whichever status line below applies (it is independent of trend_error, and does
        # not arise at all in the trim_error branch, where no window was even computed).
        # SHORT form only: the status bar is a single-line QStatusBar message (no wrap),
        # and the full notice from compute.trim_boundary_notices() can run past 300
        # characters — long enough to be clipped on the very screen it points the user
        # to. The full text is unabridged in the Run log, the terminal and
        # run-report.txt, none of which are width-constrained the same way.
        boundary_note = _short_boundary_note(data.get("boundary_notices") or [])
        if data.get("trim_error"):
            self._set_status(f"{data['name']}: showing raw channels — could not detect "
                             f"breaths. {data['trim_error']}")
            # no spans in this branch (see stage_mechanics_preview's TrimError fallback) ->
            # nothing to click, so the persistent caption must not keep asserting the
            # previous file's breath count over a panel with none.
            self.mech_caption.setFullText("")
        elif data.get("trend_error"):
            # Deliberately showing something the batch will NOT produce, so say so.
            self._set_status(f"{data['name']}: {data['nbreaths']} breaths, showing the "
                             f"DRIFT-corrected volume — the end-expiratory trend correction "
                             f"was skipped here and will fail this file in a run. "
                             f"{data['trend_error']}"
                             + (f" {boundary_note}" if boundary_note else ""))
            # breaths ARE detected and clickable here (only the trend correction failed),
            # so the caption still applies — same as the plain success branch below.
            nign = sum(1 for _n, _a, _b, ig in spans if ig)
            self._set_mech_caption(data['nbreaths'], nign)
        else:
            nign = sum(1 for _n, _a, _b, ig in spans if ig)
            self._set_status(
                f"{data['name']}: {data['nbreaths']} breaths"
                + (f" ({nign} excluded)" if nign else "")
                + f", trimmed to {data['startix'] / fs:.2f}–{data['endix'] / fs:.2f} s. "
                "Click a shaded breath to mark it; right-click to exclude it or set its type "
                "(red = excluded)."
                + (f" {boundary_note}" if boundary_note else ""))
            self._set_mech_caption(data['nbreaths'], nign)

    def _render_raw_stack(self, emg, fs, flow=None):
        """Draw the stacked raw EMG channels and keep the noise region alive. A discrete
        full-length flow silhouette is superimposed behind each channel so EMG activity can
        be read against the respiration cycle (``flow`` is untrimmed, aligned to the emg)."""
        emg = np.asarray(emg, dtype=float)
        if emg.ndim == 1:
            emg = emg[:, None]
        self._emg_raw_subplots = []
        self._raw_label_y = None
        self.emg_raw_plots.clear()
        if emg.size == 0 or emg.ndim != 2 or emg.shape[1] == 0:
            self._paint_breaths("raw", [], self._trim_offset_s, 0.0)   # drop stale overlays
            self._ensure_noise_region()
            return
        cols = list(self.state.settings.input.channels.emg)
        t = np.arange(emg.shape[0]) / fs
        cycle = _plot_pal()["emg_cycle"]
        _theme.set_stack_floor(self.emg_raw_plots, emg.shape[1])   # per channel, not per stack
        for i in range(emg.shape[1]):
            p = self.emg_raw_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            # This is now a compact reference panel (bottom-row third), so the channels are
            # short. Drop pyqtgraph's "(×0.001)" SI suffix: rotated, it made the label longer
            # than the channel is tall, and the three labels overran each other.
            p.getAxis("left").enableAutoSIPrefix(False)
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            p.plot(t, emg[:, i], pen=_pen(cycle[i % len(cycle)]))
            add_flow_background(p, t, flow, _plot_pal())   # discrete respiration reference, behind
            self._limit_x(p, t)
            self._emg_raw_subplots.append(p)
        # x-values on the bottom channel only; y-zoom shared across the raw EMG channels
        self._style_channel_stack(self.emg_raw_plots, self._emg_raw_subplots, link_y=True)
        self._ensure_noise_region()
        self._raw_label_y = self._safe_top(emg[:, 0])
        self._repaint_view_breaths("raw")

    # M-25's provisional 'segments' renderer (_render_emg_segments_preview) lived here —
    # drawing into the raw EMG stack with no click surface at all, since no dedicated tab
    # existed yet. M-26 replaces it with the real, interactive one:
    # _segments.py::_SegmentsMixin._render_segments_preview.

    # -- feature A: breath overlays + include/exclude/type ------------------
    @staticmethod
    def _breath_brush(kind, carried=False):
        """``kind``: ``None``/falsy for plain tidal breathing, the pseudo-kind
        ``'excluded'`` for a manual exclusion, or a ``respmech.core.settings.
        BREATH_KINDS`` member for a typed breath (M-20) — see ``_KIND_PALETTE_SUFFIX``.
        ``carried``: this entry was recorded against a different (or unrecorded)
        recordings folder than the one now loaded — see ``_exclusion_carried_for``. Drawn
        HATCHED instead of solid, so an inherited exclusion reads differently from one made
        in the folder actually on screen without needing a second colour (which would
        collide with the included/excluded palette already in use elsewhere)."""
        pal = _plot_pal()
        if not kind:
            return pg.mkBrush(*pal["breath_incl_brush"])
        suffix = _KIND_PALETTE_SUFFIX.get(kind, "other")
        colour = pg.mkColor(*pal[f"breath_{suffix}_brush"])
        if carried:
            return QBrush(colour, Qt.BDiagPattern)
        return QBrush(colour, Qt.SolidPattern)

    @staticmethod
    def _breath_label_color(kind):
        pal = _plot_pal()
        if not kind:
            return pg.mkColor(*pal["breath_incl_label"])
        suffix = _KIND_PALETTE_SUFFIX.get(kind, "other")
        return pg.mkColor(*pal[f"breath_{suffix}_label"])

    @staticmethod
    def _limit_x(plot, t):
        """Bound a plot's x view to the data's time extent so it cannot zoom/pan past
        the recording (req: no zooming out beyond the data)."""
        try:
            t = np.asarray(t, dtype=float)
            t = t[np.isfinite(t)]
            if t.size < 2:
                return
            x0, x1 = float(t.min()), float(t.max())
            if x1 > x0:
                vb = plot.getViewBox()
                vb.setLimits(xMin=x0, xMax=x1)
                vb.setXRange(x0, x1, padding=0)
        except Exception:                        # noqa: BLE001 — cosmetic
            pass

    def _breath_text(self, num, kind, span=None):
        """A breath-number TextItem, sized for the SHORT stacked mechanics channel plots
        (~46 px of data area each): at the 13 pt app font the box is 23 px, so the headroom
        needed to show it swallows half the plot.

        Both steps matter, and the second is the one that counts: the default box is 23 px
        almost entirely because QTextDocument adds a 4 px margin on every side — setting a
        smaller font ALONE leaves it at 23 px. Font 9 pt + documentMargin 0 gives ~15 px."""
        txt = pg.TextItem(self._breath_label(num, kind),
                          color=self._breath_label_color(kind), anchor=(0.5, 0.0))
        f = QFont(); f.setPointSizeF(9.0)
        txt.setFont(f)
        doc = txt.textItem.document()
        doc.setDocumentMargin(0)
        opt = doc.defaultTextOption(); opt.setAlignment(Qt.AlignHCenter)
        doc.setDefaultTextOption(opt)            # the type line is centred under '#n'
        txt._rm_label = {"num": num, "kind": kind, "span": span, "compact": False,
                         "full_px": self._label_full_width_px(txt, num, kind) if kind else 0.0}
        txt.updateTextPos()
        return txt

    @staticmethod
    def _label_px(texts, default=25.0, margin=4.0):
        """The tallest breath label's rendered height in pixels (+ a little breathing room),
        measured rather than assumed — a TextItem is taller than its font size."""
        try:
            h = max(t.boundingRect().height() for t in (texts.values() if hasattr(texts, "values") else texts))
            return float(h) + margin if h > 0 else default + margin
        except Exception:                        # noqa: BLE001 — cosmetic
            return default + margin

    def _label_headroom(self, plot, label_px=25.0, frac=0.22):
        """Expand the label-carrying plot's y range upward so the breath labels — pinned
        at the view top, hanging down into the view — start over an empty band instead of
        sitting on the signal. Re-fits y to the DATA first so repeated repaints of a
        persistent plot (detail / result) don't compound the headroom.

        The headroom is sized in PIXELS, not as a fraction of the data span: a TextItem is
        a fixed pixel height whatever the scale, so on a short plot a fixed fraction leaves
        less band than the label needs (five stacked mechanics channels leave each ~70 px,
        where 22% ≈ 15 px — less than the label). Solving
        ``new_span = span + label_px * new_span / height_px`` for the exact expansion makes
        the label occupy ``label_px`` at the top at any height; ``frac`` is only the
        fallback when the plot has no laid-out height yet."""
        try:
            vb = plot.getViewBox()
            vb.enableAutoRange(y=True)           # snap y back to the data extent…
            vb.updateAutoRange()                 # …(not the previous, already-expanded view)
            (x0, x1), (y0, y1) = vb.viewRange()
            span = y1 - y0
            if span <= 0:
                return
            h_px = float(vb.height() or 0.0)
            extra = (span * (label_px / (h_px - label_px))
                     if h_px > label_px + 6 else frac * span)
            vb.setYRange(y0, y1 + extra, padding=0)   # (this disables y auto again)
        except Exception:                        # noqa: BLE001 — cosmetic
            pass

    def _pin_breath_labels(self, plot, txt_map):
        """Keep the breath-number labels pinned to the TOP of the visible view under
        y-zoom/pan — the same behaviour as the red capture marks: a sigYRangeChanged
        slot that only setPos()es the existing TextItems (wheel-zoom fires this
        continuously, so nothing is re-created) and self-disconnects once its labels
        leave the view. The labels MUST be added with ignoreBounds=True: a view-pinned
        item that still fed childrenBounds would re-inflate every autorange pass.
        Returns an unpin callable for eager teardown on repaint."""
        vb = plot.getViewBox()
        texts = list(txt_map.values())
        if vb is None or not texts:
            return lambda: None

        def _unpin():
            for sig, slot in ((vb.sigYRangeChanged, _reposition), (vb.sigXRangeChanged, _recompact),
                         (vb.sigResized, _recompact)):
                try:
                    sig.disconnect(slot)
                except Exception:                      # noqa: BLE001
                    pass

        def _recompact(*_):
            # x zoom changes how many pixels a breath span gets: swap each typed label
            # between its full type text and the compact marker (see _refresh_breath_label)
            for t in texts:
                self._refresh_breath_label(t)

        def _reposition(*_):
            try:
                if texts[0].getViewBox() is None:      # repainted/cleared -> self-remove
                    _unpin()
                    return
                top = vb.viewRange()[1][1]
                for t in texts:
                    t.setPos(t.pos().x(), top)
            except Exception:                          # noqa: BLE001 — cosmetic
                _unpin()

        vb.sigYRangeChanged.connect(_reposition)
        vb.sigXRangeChanged.connect(_recompact)
        vb.sigResized.connect(_recompact)          # a resize changes px/s without changing x range
        _reposition()                                  # place at the CURRENT view top now
        _recompact()
        return _unpin

    @staticmethod
    def _breath_label(num, kind=None, compact=False):
        """The per-breath label: the number, e.g. '#3', and for a typed or excluded breath
        its type on a second line (``_KIND_TAG``), e.g. '#3\nFVC' or '#3\n(Excluded)'.
        ``compact`` swaps that second line for the kind's one-or-two-character stand-in
        (``_KIND_ICON``), used when the full text would collide with a neighbour. The word
        'breath' is intentionally omitted from per-breath numbering — it is implicit from
        context on every graph. Plain tidal breathing (``kind`` falsy) stays a bare '#3'."""
        if not kind:
            return f"#{num}"
        second = (_KIND_ICON if compact else _KIND_TAG).get(kind, _KIND_ICON["other"] if compact else _KIND_TAG["other"])
        return f"#{num}\n{second}"

    @staticmethod
    def _label_full_width_px(txt, num, kind):
        """The rendered width, in pixels, of ``txt``'s FULL (non-compact) label."""
        fm = QFontMetricsF(txt.textItem.font())
        return max(fm.horizontalAdvance(line)
                   for line in _MechanicsMixin._breath_label(num, kind).split("\n"))

    def _refresh_breath_label(self, txt):
        """Pick, for one breath-number TextItem, between the full type text and the compact
        marker, from the zoom the label's own view currently has: the full text when the
        breath's span is wide enough on screen to hold it, the marker otherwise. A no-op
        for an untyped breath (nothing to shorten) and for a label whose view has no laid-out
        width yet (it keeps the full text until a real range arrives)."""
        st = getattr(txt, "_rm_label", None)
        if not st or not st["kind"]:
            return
        compact = False
        try:
            vb = txt.getViewBox()
            span = st.get("span")
            if vb is not None and span is not None and vb.width() > 0:
                (x0, x1) = vb.viewRange()[0]
                if x1 > x0:
                    span_px = (span[1] - span[0]) / (x1 - x0) * float(vb.width())
                    compact = span_px < st["full_px"] + _LABEL_FIT_PAD_PX
        except Exception:                        # noqa: BLE001 — cosmetic
            return
        if compact != st["compact"]:
            st["compact"] = compact
            txt.setText(self._breath_label(st["num"], st["kind"], compact))

    def _draw_breath_overlays(self, spans, label_y=0.0, carried=False, plots=None):
        """Shade every breath + a number label on ``plots`` (default: the Mechanics
        channel stack, ``self._channel_plots``). One BreathSpansItem PER PLOT carries
        every breath's region (D15) — the old per-breath pg.LinearRegionItem (plus its
        now-dropped redundant boundary line, see BreathSpansItem's docstring) is gone;
        only the label stays a per-breath TextItem, same as before. ``spans``:
        ``(n, t0, t1, kind)`` — see ``_breath_brush``'s docstring for what ``kind`` may be.

        ``plots`` (M-26): the segments tab's own stack passes its subplot list here
        instead, reusing this exact click/type-menu-bearing overlay machinery — safe
        because it writes into the SAME shared ``_breath_spans``/``_breath_regions``/
        ``_breath_texts``/``_mech_unpin`` state ``_toggle_breath``/``_set_breath_type``
        already read generically, and Mechanics and the segments tab are never both
        rendering breaths at once (a signal set is either flow-bearing or EMG-only,
        never both — see ``_schedule``'s 'mech'/'segments' gate)."""
        if plots is None:
            plots = self._channel_plots
        self._mech_unpin()               # the old labels are torn down with their pin slot
        self._breath_spans = {n: (t0, t1) for (n, t0, t1, _k) in spans}
        self._breath_regions = {n: [] for (n, _0, _1, _k) in spans}   # n -> [(item, index), ...]
        self._breath_texts = {}
        brushes_by_index = [self._breath_brush(kind, carried=carried) for _n, _t0, _t1, kind in spans]
        for plot in plots:
            item = BreathSpansItem()
            item.set_spans([(t0, t1, brushes_by_index[i], n)
                            for i, (n, t0, t1, _k) in enumerate(spans)])
            item.setZValue(-10)
            item.typeRequested.connect(self._handle_type_requested)
            plot.addItem(item)
            for i, (n, _t0, _t1, _k) in enumerate(spans):
                self._breath_regions[n].append((item, i))
        if plots:
            for n, t0, t1, kind in spans:
                txt = self._breath_text(n, kind, span=(t0, t1))
                txt.setPos((t0 + t1) / 2.0, label_y)
                plots[0].addItem(txt, ignoreBounds=True)
                self._breath_texts[n] = txt
        if plots and self._breath_texts:
            # size the headroom from the label's REAL rendered height, not a guess
            self._label_headroom(plots[0], label_px=self._label_px(self._breath_texts))
            self._mech_unpin = self._pin_breath_labels(plots[0], self._breath_texts)
        # a mark survives a re-render only while its breath still exists in this render
        if self._selected_breath is not None and self._selected_breath not in self._breath_spans:
            self._clear_breath_selection()
        self._apply_breath_selection(redraw_loop=False)

    def _breath_at(self, t):
        for n, (t0, t1) in self._breath_spans.items():
            if t0 <= t <= t1:
                return n
        return None

    def _on_plot_clicked(self, ev):
        # While 'Place separators' (Mechanics' own manual segmentation-repair
        # toggle) is armed, EVERY click on this stack places or removes an override
        # instead of toggling a breath — checked FIRST, mirroring _emg_noise.py's
        # _toggle_from_emg_click (the same armed-click precedence used for the
        # EMG-only segments tab). _place_or_remove_override does its own
        # accepted/button checks.
        if self._overrides_armed:
            self._place_or_remove_override(ev, self._channel_plots)
            return
        # M-20: a right-click/Ctrl+left-click that landed on a breath is handled at
        # item level (BreathSpansItem.mouseClickEvent -> typeRequested) and accepted
        # there — this scene-level handler is for the plain left-click toggle only, and
        # must ignore anything already accepted (including a right-click ViewBox itself
        # accepted to raise its own menu, when the click missed every span).
        if ev.isAccepted() or ev.button() != Qt.LeftButton:
            return
        if not self._breath_spans:
            return
        try:
            pos = ev.scenePos()
        except Exception:                              # noqa: BLE001
            return
        for plot in self._channel_plots:
            vb = plot.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(pos):
                bno = self._breath_at(vb.mapSceneToView(pos).x())
                if bno is not None:
                    self._select_breath(bno)
                return

    # -- Marked (selected) breath -------------------------------------------
    # A plain left click MARKS a breath instead of toggling its exclusion (exclusion
    # now lives in the right-click menu, _build_type_menu's Tidal/Excluded). The mark is
    # one number, self._selected_breath, painted in the SAME green everywhere that
    # breath shows: its span on every plot, its number label, its row in the result
    # tables (scrolled into view) and its loop plus legend entry in the Campbell /
    # flow-volume diagram. Clicking the marked breath again clears it; clicking another
    # moves it. Purely a view state: it never touches settings, so it is allowed while a
    # run is in progress and never emits settings_edited.
    def _select_breath(self, breath_no):
        """Mark ``breath_no``, or clear the mark if it is already the marked one."""
        if breath_no not in self._breath_spans:
            return
        self._selected_breath = None if self._selected_breath == breath_no else breath_no
        name = self._selected_filename()
        if self._selected_breath is None:
            self._set_status(f"{name}: breath {breath_no} unmarked.")
        else:
            self._set_status(f"{name}: breath {breath_no} marked. Click it again to unmark; "
                             f"right-click it to exclude it or set its type.")
        self._apply_breath_selection()

    def _clear_breath_selection(self):
        """Drop the mark without repainting (file change, stale breath numbers)."""
        self._selected_breath = None
        for model in (getattr(self, "_table_model", None), getattr(self, "_manoeuvres_model", None),
                      getattr(self, "_segtable_model", None)):
            if model is not None:
                model.set_highlight_breath(None)
        self._update_breath_bar()

    def _apply_breath_selection(self, redraw_loop=True):
        """Paint ``self._selected_breath`` (or its absence) on every surface that shows
        breaths: the span items of the Mechanics stack and of the EMG views, the number
        labels, the result tables and (``redraw_loop``) the Campbell / flow-volume diagram."""
        sel = self._selected_breath
        if sel is not None and sel not in self._breath_spans \
                and sel not in {b[0] for b in self._breaths}:
            sel = self._selected_breath = None          # a stale number: nothing to mark
        self._update_breath_bar()
        seen = set()
        items = [it for lst in self._breath_regions.values() for it, _i in lst]
        for rec in self._bov.values():
            items.extend(it for _p, it in rec.get("items", []) if isinstance(it, BreathSpansItem))
            for num, txt in rec.get("texts", {}).items():
                self._style_breath_label(txt, num == sel)
        for it in items:
            if id(it) not in seen:
                seen.add(id(it))
                it.set_selected(sel)
        for num, txt in self._breath_texts.items():
            self._style_breath_label(txt, num == sel)
        for model, view in ((self._table_model, self.table),
                            (self._manoeuvres_model, self.manoeuvres_table),
                            (self._segtable_model, self.segtable)):
            row = model.set_highlight_breath(sel)
            if row is not None:
                view.scrollTo(model.index(row, 0), QAbstractItemView.EnsureVisible)
        if redraw_loop and not getattr(self, "_loop_no_mark", False):
            breaths = getattr(self, "_campbell_breaths", None)
            if breaths is not None:
                try:
                    self._draw_campbell_or_loop(breaths)
                except Exception:                      # noqa: BLE001 — cosmetic
                    pass

    def _style_breath_label(self, txt, selected):
        """Number label in the selection green while marked (colour only: a bold face would be
        wider than the width the compact-label logic measured), back to its kind's
        own colour when not."""
        st = getattr(txt, "_rm_label", None)
        if st is None:
            return
        if selected:
            txt.setColor(pg.mkColor(*SELECTED_BREATH_RGB))
        else:
            txt.setColor(self._breath_label_color(st["kind"]))

    def _set_breath_type(self, breath_no, kind):
        """Single funnel for every breath-classification write: the plain include/
        exclude toggle (``_toggle_breath``) and the type menu's Tidal/Excluded/Rest
        (``_handle_type_requested``) all end here, so the run-lock guard, the
        folder-stamp-only-on-creation rule and the one ``settings_edited`` emission
        exist in exactly one place (D24's original reasoning for ``_toggle_breath``,
        now shared). ``kind`` is one of:

        - ``'tidal'`` — clear both exclusion and typing (plain tidal breathing);
        - ``'excluded'`` — manually excluded, ``processing.exclude_breaths``;
        - a ``respmech.core.settings.BREATH_KINDS`` member — typed,
          ``processing.breath_types`` (M-19), storing ``t_onset_s``.

        A breath is at all times in exactly one of these three states: setting one
        clears whichever of the other two it may have carried, so the settings.
        validate() invariants ('typed more than once', 'both typed and excluded')
        can never actually be reached from this funnel.

        Returns ``kind`` again on success (never ``None`` — 'tidal' is returned as the
        literal string, so a caller can tell "applied, now tidal" apart from "did not
        apply" unambiguously), or ``None`` if the write was blocked (a run in
        progress) or the target is invalid (no file selected, or a breath number this
        file's current render does not know about — e.g. a stale click, same guard
        ``_toggle_breath`` has always had).

        Does NOT itself write a status-bar message or the Mechanics caption: those
        differ enough between the toggle's "N/M excluded" wording and the menu's
        "set to <kind>" wording that each caller composes its own, after checking
        the return value is not ``None``."""
        if self._run_active:
            # _set_status alone would be invisible here: MainWindow suppresses every
            # non-run_screen status while a run is active (see write_action_blocked's own
            # docstring in screen.py for why), which is exactly the window this guard fires
            # in. write_action_blocked is MainWindow's forced-onto-the-bar escape hatch.
            msg = "Breath selection is locked while a run is in progress."
            self._set_status(msg)
            self.write_action_blocked.emit(msg)
            return None
        name = self._selected_filename()
        if not name or breath_no not in self._breath_spans:
            return None
        proc = self.state.settings.processing
        excl = proc.exclude_breaths
        types = proc.breath_types
        excl_entry = next((e for e in excl if e.file == name), None)
        type_entry = next((t for t in types if t.file == name and t.breath == breath_no), None)

        if kind == "tidal":
            if excl_entry is not None and breath_no in excl_entry.breaths:
                excl_entry.breaths = [b for b in excl_entry.breaths if b != breath_no]
                if not excl_entry.breaths:
                    excl.remove(excl_entry)
            if type_entry is not None:
                types.remove(type_entry)
            paint_kind = None
        elif kind == "excluded":
            if type_entry is not None:
                types.remove(type_entry)
            if excl_entry is None:
                # ONLY a brand-new entry gets stamped with the current folder here. An
                # EXISTING entry's folder is deliberately left untouched by a plain
                # toggle, even when the user is un-excluding one of ITS OWN breaths:
                # entry.folder is one tag for the WHOLE file, but excl.breaths can hold
                # a MIX of breaths the user just decided on and others still carried
                # from a different folder that this click never looked at. Restamping
                # on every touch (an earlier version of this fix did) would silently
                # "confirm" those untouched breaths too — exactly the invisible
                # application B06 exists to stop, just moved one click later. An entry
                # only stops reading as carried via a fresh creation here or via the
                # Setup banner's "Clear" (core.settings.clear_carried_over); "Keep" is a
                # pure dismiss and — like this — never restamps either, matching "as
                # long as the user has chosen to keep them, a carried exclusion is
                # still drawn hatched", not "keeping it once makes it look native from
                # then on". Known accepted imprecision: a genuinely NEW breath added to
                # an EXISTING carried entry still reads as carried until the whole
                # entry is cleared — ExcludeEntry.folder is one tag per FILE, not per
                # breath, by B06's own design.
                excl_entry = ExcludeEntry(file=name, breaths=[],
                                          folder=self.state.settings.input.folder)
                excl.append(excl_entry)
            excl_entry.breaths = sorted(set(excl_entry.breaths) | {breath_no})
            paint_kind = "excluded"
        else:
            if excl_entry is not None and breath_no in excl_entry.breaths:
                excl_entry.breaths = [b for b in excl_entry.breaths if b != breath_no]
                if not excl_entry.breaths:
                    excl.remove(excl_entry)
            # self._breath_spans is zero-based at the TRIMMED window's own start
            # (stage_mechanics_preview's cum/fs); + _trim_offset_s recovers the
            # recording's own clock, matching both breath['time'][0] (core) and the
            # absolute time base the EMG views already align their spans to
            # (_paint_breaths' own `t0 + offset`) — self-review finding: an earlier
            # version stored the trimmed-window-relative t0 instead, which would have
            # silently drifted from the recording's own clock whenever the trim
            # settings changed.
            t0_abs = self._breath_spans[breath_no][0] + self._trim_offset_s
            if type_entry is None:
                # same folder-stamp-only-on-creation rule as the exclude branch above.
                type_entry = BreathTypeEntry(file=name, breath=breath_no, kind=kind,
                                             t_onset_s=t0_abs,
                                             folder=self.state.settings.input.folder)
                types.append(type_entry)
            else:
                type_entry.kind = kind
                type_entry.t_onset_s = t0_abs
            paint_kind = kind

        self.settings_edited.emit()      # exclude_breaths/breath_types land in the .toml
        # M-31: manoeuvres.suggest_fvc's own contract is "untyped, non-ignored" — the
        # instant THIS write types/excludes the breath the hint was pointing at, that
        # contract is broken until the next full re-stage (self._suggested_fvc is only
        # ever recomputed there). Rather than showing a now-wrong hint on an already-
        # decided breath until some LATER unrelated re-stage happens to correct it,
        # drop it immediately — advertising nothing is honest; a stale suggestion is not.
        if breath_no == self._suggested_fvc:
            self._suggested_fvc = None
        self._repaint_breath(breath_no, paint_kind)
        # M-32: the wide, all-files sync (exclusion + typed + segment badges) — a
        # narrower single-file version (the old _sync_excluded_badge) no longer exists,
        # same reasoning _toggle_separator_at (_segments.py) already had for using this
        # one directly.
        self._sync_rail_breath_state()
        # a type/exclude change must update the AVERAGED result in lockstep with the
        # overlay — otherwise the Campbell loop + per-breath table stay stale and the
        # user tunes blind. Recompute the (mechanics-only) test run, debounced.
        self._request_batch_recompute()
        return kind

    def _repaint_breath(self, breath_no, paint_kind):
        """Recolour one breath's overlay everywhere it is currently painted (the
        mechanics stack + every EMG view that has it) to ``paint_kind`` — the shared
        repaint step both ``_set_breath_type`` and (indirectly, via it) ``_toggle_breath``
        use. ``paint_kind``: ``None``/``'excluded'``/a BREATH_KINDS member, see
        ``_breath_brush``. Never threads ``carried`` through: a live single-breath
        repaint has never reflected it (only a full re-render via
        ``_draw_breath_overlays``/``_paint_breaths`` does), unchanged by this ticket."""
        # each entry is (BreathSpansItem, index-into-that-item's-span-list) — one PAIR per
        # plot the breath is drawn on (5 for the mechanics stack), not one item per plot.
        for item, idx in self._breath_regions.get(breath_no, []):
            item.set_brush(idx, self._breath_brush(paint_kind))
        txt = self._breath_texts.get(breath_no)
        if txt is not None:
            self._retag_breath_label(txt, breath_no, paint_kind, self._breath_texts)
        for view in ("raw", "detail", "result"):
            rec = self._bov.get(view)
            if not rec:
                continue
            for item, idx in rec["regions"].get(breath_no, []):
                try:
                    item.set_brush(idx, self._breath_brush(paint_kind))
                except Exception:                      # noqa: BLE001
                    pass
            t = rec["texts"].get(breath_no)
            if t is not None:
                try:
                    self._retag_breath_label(t, breath_no, paint_kind, rec["texts"])
                except Exception:                      # noqa: BLE001
                    pass

    def _retag_breath_label(self, txt, breath_no, paint_kind, txt_map):
        """Live counterpart of ``_breath_text``: give an already-drawn label its new colour
        AND its new type line (``_breath_label``) after a type/exclusion change, then let
        ``_refresh_breath_label`` choose text vs. marker for the current zoom. A label that
        has just grown a second line may be taller than the headroom reserved above the
        signal when the view was drawn (an all-tidal file reserves one line), so the
        headroom is re-fitted once, and only when it has actually grown."""
        txt.setColor(pg.mkColor(*SELECTED_BREATH_RGB) if breath_no == self._selected_breath
                     else self._breath_label_color(paint_kind))
        st = getattr(txt, "_rm_label", None)
        if st is None:
            return
        before = txt.boundingRect().height()
        st["kind"] = paint_kind
        st["compact"] = False
        st["full_px"] = self._label_full_width_px(txt, breath_no, paint_kind) if paint_kind else 0.0
        txt.setText(self._breath_label(breath_no, paint_kind))
        self._refresh_breath_label(txt)             # may swap to the marker for the current zoom
        if txt.boundingRect().height() > before + 1.0:
            vb = txt.getViewBox()
            plot = getattr(vb, "parentItem", lambda: None)() if vb is not None else None
            if plot is not None:
                self._label_headroom(plot, label_px=self._label_px(txt_map))

    def _toggle_breath(self, breath_no):
        # D24: the single funnel both the Mechanics-stack click (_on_plot_clicked above)
        # and every EMG plot's click handler (_emg_noise._toggle_from_emg_click) call
        # through — one guard here covers both. A click during a run used to silently
        # rewrite exclude_breaths without ever touching the batch that is already reading
        # a frozen deepcopy of the settings taken at _start() (run_screen.py) — the click
        # LOOKED like it worked (the overlay recoloured immediately) while the running
        # batch, and the results it was about to write, never saw it. (M-20: reimplemented
        # on top of the shared _set_breath_type funnel; the include/exclude toggle is a
        # two-state special case of the three-state kind model, plain 'excluded' vs
        # 'tidal', with its own status wording preserved exactly.)
        name = self._selected_filename()
        excl_entry = None
        if name:
            excl_entry = next((e for e in self.state.settings.processing.exclude_breaths
                               if e.file == name), None)
        was_excluded = excl_entry is not None and breath_no in excl_entry.breaths
        result = self._set_breath_type(breath_no, "tidal" if was_excluded else "excluded")
        if result is None:
            return None
        now_excluded = result == "excluded"
        nexcl = len({b for e in self.state.settings.processing.exclude_breaths if e.file == name
                    for b in e.breaths} & set(self._breath_spans))
        self._set_status(
            f"{name}: breath {breath_no} {'excluded' if now_excluded else 'included'} "
            f"({nexcl}/{len(self._breath_spans)} excluded). Recomputing the average…")
        self._set_mech_caption(len(self._breath_spans), nexcl)
        return now_excluded

    # -- Manual segmentation overrides (cut/join), Mechanics tab -----------

    def _update_overrides_button(self):
        """Enable/disable + tooltip Mechanics' own 'Place separators' for the
        CURRENT segmentation method and run state — mirrors ``_segments.py``'s
        ``_update_separators_button`` exactly, but for
        ``processing.segmentation.overrides``: only 'flow'/'volume' segmentation has
        an automatic detector this repairs (``whole_file``/``separators`` are
        EMG-only and already have their own manual-boundary mechanism,
        ``SeparatorEntry``, consulted only for THOSE methods — see
        ``SegmentationSettings.overrides``' own comment)."""
        method = self.state.settings.processing.segmentation.method
        if method not in ("flow", "volume"):
            if self.btn_place_overrides.isChecked():
                self.btn_place_overrides.setChecked(False)   # also clears _overrides_armed
            self.btn_place_overrides.setEnabled(False)
            self.btn_place_overrides.setToolTip(
                "Segmentation repair applies to flow-/volume-based breath detection "
                "only.")
        else:
            self.btn_place_overrides.setEnabled(not self._run_active)
            self.btn_place_overrides.setToolTip(
                "Click a channel trace to cut a new breath boundary there, or click "
                "an existing cut (within a few pixels) to remove it. Click near an "
                "automatic breath boundary to join the two breaths either side of it.")

    def _on_place_overrides_toggled(self, checked):
        self._overrides_armed = checked

    def _update_override_lines(self, filename):
        """(Re)draw the cut markers on every current Mechanics channel-stack plot
        from ``processing.segmentation.overrides``' ``cut_s`` — mirrors
        ``_segments.py``'s own ``_update_separator_lines``, with this feature's own
        ``"segmentation_override"`` pen so the two never look alike. ``join_s`` names
        an AUTOMATIC boundary to REMOVE, so it has no marker of its own to draw:
        joining one leaves nothing at that instant for a marker to point at."""
        entry = next((e for e in self.state.settings.processing.segmentation.overrides
                     if e.file == filename), None)
        cut_times = list(entry.cut_s) if entry is not None else []
        self._override_items = []
        for p in self._channel_plots:
            item = SeparatorLinesItem(pen_key="segmentation_override")
            item.set_times(cut_times)
            item.setZValue(-5)          # above BreathSpansItem's fill (-10), below the trace
            p.addItem(item)
            self._override_items.append(item)

    def _set_segmentation_overrides(self, file, cut_s, join_s, old_bounds, new_bounds):
        """Rewrite ``file``'s ``cut_s``/``join_s`` and renumber every existing
        ``ExcludeEntry``/``BreathTypeEntry`` for it in lockstep — the counterpart
        of ``_segments.py``'s ``_set_separators``, reusing the SAME
        ``remap_segment_number`` rule: it is already generic over WHERE a boundary
        list came from (an explicit ``separators`` list there, an
        auto-plus-override-adjusted one here).

        ``cut_s``/``join_s``: the FULL new lists (absolute recording-clock seconds,
        the same convention ``SeparatorEntry.times_s``/``ExcludeEntry``/
        ``BreathTypeEntry`` already use). ``old_bounds``/``new_bounds``: the ACTUAL
        breath-start boundary lists in effect before/after this one edit, supplied by
        the caller (``_toggle_override_at``) — NOT derived from ``cut_s`` alone here,
        because a JOIN changes which breaths merge (and therefore every later
        breath's number) WITHOUT touching ``cut_s`` at all; only the caller, which
        already knows the file's currently rendered breath spans, can name the real
        before/after boundaries. Folder is stamped ONLY when a brand-new
        ``SegmentationOverrideEntry`` is created here, never on an edit of an
        existing one — the same carried-over-state rule every other tagged kind
        already follows (see ``_set_breath_type``)."""
        self._clear_breath_selection()           # the breaths are renumbered: a mark would drift
        from respmech.core.analysis.segments import remap_segment_number  # noqa: PLC0415

        proc = self.state.settings.processing
        seg = proc.segmentation
        entry = next((e for e in seg.overrides if e.file == file), None)
        new_cut = sorted(cut_s)

        def remap(old_number):
            return remap_segment_number(old_bounds, new_bounds, old_number)

        excl_entry = next((e for e in proc.exclude_breaths if e.file == file), None)
        if excl_entry is not None and excl_entry.breaths:
            excl_entry.breaths = sorted({remap(b) for b in excl_entry.breaths})

        for t in proc.breath_types:
            if t.file != file:
                continue
            t.breath = remap(t.breath)
            idx = min(max(t.breath - 1, 0), len(new_bounds) - 1)
            t.t_onset_s = new_bounds[idx]
        # A cut can fold two previously distinct typed breaths onto the SAME new
        # number (same collision ``_set_separators`` already guards against) —
        # keep the first one this file's own list still holds, drop the rest.
        seen = set()
        deduped = []
        for t in proc.breath_types:
            key = (t.file, t.breath)
            if t.file == file:
                if key in seen:
                    continue
                seen.add(key)
            deduped.append(t)
        proc.breath_types[:] = deduped
        # …and the exclusion/typed cross-kind collision that function also guards
        # against: typed wins over a plain exclusion on the same new number.
        if excl_entry is not None and excl_entry.breaths:
            typed_here = {t.breath for t in proc.breath_types if t.file == file}
            excl_entry.breaths = sorted(set(excl_entry.breaths) - typed_here)
            if not excl_entry.breaths:
                proc.exclude_breaths.remove(excl_entry)

        if entry is None:
            seg.overrides.append(SegmentationOverrideEntry(
                file=file, cut_s=new_cut, join_s=sorted(join_s),
                folder=self.state.settings.input.folder))
        else:
            entry.cut_s = new_cut
            entry.join_s = sorted(join_s)

    def _toggle_override_at(self, name, t, vb):
        """Decide join/cut/remove-cut for an armed click at absolute-recording-time
        ``t`` on ``name`` — the counterpart of ``_segments.py``'s
        ``_toggle_separator_at``. Three outcomes:

        - ``t`` is within tolerance of an EXISTING cut this override already placed
          -> remove that cut (undo).
        - ``t`` is within tolerance of one of the file's CURRENTLY RENDERED breath
          boundaries (an untouched automatic one — a breath's own start in
          ``self._breath_spans``) and not already named in ``join_s`` -> request a
          JOIN there (remove that automatic boundary).
        - otherwise -> add a new CUT at ``t``.

        Tolerance uses ``vb``'s own current pixel scale, same as the EMG-only
        separators list's own tolerance (a fixed-seconds tolerance would feel wildly
        different zoomed in vs out). Never validates ``t`` against the recording's
        own duration: ``compute.apply_segmentation_overrides`` already turns an
        out-of-range cut into a soft per-file notice, exactly like
        ``segments.separators()`` does for its own boundaries."""
        proc = self.state.settings.processing
        entry = next((e for e in proc.segmentation.overrides if e.file == name), None)
        existing_cuts = list(entry.cut_s) if entry is not None else []
        existing_joins = list(entry.join_s) if entry is not None else []
        tol = abs(vb.viewPixelSize()[0]) * 6
        # The boundary list ACTUALLY in effect right now (whatever the last
        # successful segment_file computed for this file — auto detection, minus
        # any joined-away boundaries, plus any cuts already placed), read directly
        # off the rendered breath spans (window-relative + the trim offset ->
        # absolute, same conversion _set_breath_type already uses for t_onset_s).
        # This is what a JOIN actually changes for renumbering purposes, which
        # cut_s/join_s individually do NOT capture (a join changes numbering
        # without touching cut_s at all).
        old_bounds = sorted(t0 + self._trim_offset_s for (t0, _t1) in self._breath_spans.values())

        nearest_cut = min(existing_cuts, key=lambda s: abs(s - t)) if existing_cuts else None
        if nearest_cut is not None and abs(nearest_cut - t) <= tol:
            new_cuts = [s for s in existing_cuts if s != nearest_cut]
            new_bounds = [b for b in old_bounds if abs(b - nearest_cut) > tol]
            self._set_segmentation_overrides(name, new_cuts, existing_joins, old_bounds, new_bounds)
            verb, at = "cut removed at", nearest_cut
        else:
            nearest_boundary = (min(old_bounds, key=lambda s: abs(s - t))
                                if old_bounds else None)
            if (nearest_boundary is not None and abs(nearest_boundary - t) <= tol
                    and not any(abs(nearest_boundary - j) <= tol for j in existing_joins)):
                new_joins = sorted(existing_joins + [nearest_boundary])
                new_bounds = [b for b in old_bounds if abs(b - nearest_boundary) > tol]
                self._set_segmentation_overrides(name, existing_cuts, new_joins, old_bounds, new_bounds)
                verb, at = "breaths joined at", nearest_boundary
            else:
                new_cuts = sorted(existing_cuts + [t])
                new_bounds = sorted(old_bounds + [t])
                self._set_segmentation_overrides(name, new_cuts, existing_joins, old_bounds, new_bounds)
                verb, at = "cut placed at", t
        self.settings_edited.emit()
        # Wide, all-files sync — a segmentation-override edit can change this file's
        # own exclusion/typed/segment counts exactly like a separator edit can
        # (the same reasoning ``_toggle_separator_at`` has for using this instead of
        # a narrower, single-file version, which does not exist).
        self._sync_rail_breath_state()
        # Wide, not just {"mech", "batch"}: _kinds_for_settings_path treats every
        # processing.segmentation.* field as needing the full _AUTO_KINDS recompute
        # (same rule ``_toggle_separator_at`` itself relies on) — an override can
        # renumber a rest-typed BreathTypeEntry too, on an EMG-only-adjacent mixed
        # signal set, and feeds segment_file/the noise reference exactly like buffer
        # does.
        self._request_autorun()
        self._set_status(f"Segmentation {verb} {at:.2f} s in {name}.")

    def _place_or_remove_override(self, ev, plot_items):
        """The armed counterpart of the plain breath-exclude click — mirrors
        ``_segments.py``'s ``_place_or_remove_separator`` exactly (same
        best-effort button check, same scene-position lookup, same per-plot ViewBox
        hit test), differing only in what the click then does
        (``_toggle_override_at``) and in reading the click position via
        ``self._trim_offset_s`` directly rather than a passed-in offset: unlike the
        EMG-only views the shared funnel there serves (all effectively untrimmed,
        offset always 0.0), the Mechanics stack's own x-axis is deliberately
        zero-based AT the trimmed window's start (``_MECH_WINDOW_TOOLTIP``), so a
        click there must be shifted FORWARD by the trim offset to reach the
        recording's absolute clock, the same conversion ``_set_breath_type`` already
        performs for ``t_onset_s``."""
        if self._run_active:
            msg = "Segmentation repair is locked while a run is in progress."
            self._set_status(msg)
            self.write_action_blocked.emit(msg)
            return
        name = self._selected_filename()
        if not name:
            return
        try:
            if ev.isAccepted():
                return
            button = getattr(ev, "button", None)
            if button is not None and button() != Qt.LeftButton:
                return
            pos = ev.scenePos()
        except Exception:                              # noqa: BLE001
            return
        for p in plot_items:
            vb = p.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(pos):
                t = vb.mapSceneToView(pos).x() + self._trim_offset_s
                self._toggle_override_at(name, t, vb)
                return

    def _build_type_menu(self, breath_no, kinds):
        """The breath-type context menu (M-20's minimal Tidal/Excluded/Rest, extended by
        M-31 to the full manoeuvre set). Parented to ``self.plots`` so
        ``_lone_ampersands``'s QMenu scan (``tests/unit/_helpers.py``) reaches it like
        every other menu in the window — a menu with no such parent is invisible to that
        scan — and set to delete itself on close so a right-click doesn't leak one QMenu
        per use.

        ``kinds`` is the caller's choice of which real kinds to OFFER (``_handle_type_
        requested`` filters ``'rest'`` in/out by capability) — the disabled hint and the
        M-37 reference actions below are unconditional, since they name a property of
        the BREATH's current type (already decided, not something ``kinds`` should ever
        need to suppress)."""
        from respmech.core.summary import group_key  # noqa: PLC0415

        menu = QMenu(self.plots)
        menu.setAttribute(Qt.WA_DeleteOnClose)
        # M-31: a plain, deterministic hint (manoeuvres.suggest_fvc) — the untyped
        # breath with the longest expiration in the file is the most plausible untyped
        # FVC candidate. Purely advisory: shown disabled, next to whichever breath it
        # names, so it never competes with (or is mistaken for) an actual menu choice.
        if getattr(self, "_suggested_fvc", None) == breath_no:
            hint = menu.addAction("Suggested: FVC")
            hint.setEnabled(False)
            menu.addSeparator()
        for kind in kinds:
            action = menu.addAction(_TYPE_MENU_LABELS.get(kind, kind.capitalize()))
            tip = _TYPE_MENU_STATUS_TIPS.get(kind)
            if tip:
                action.setStatusTip(tip)
            action.triggered.connect(
                lambda _checked=False, k=kind: self._apply_breath_type_choice(breath_no, k))
        # M-37: reference actions. 'Use as IC reference for' is a real submenu (Qt draws
        # its own arrow — no manual '▸' needed), enabled only once THIS breath is already
        # typed 'ic'/'ic_fvc' (_IC_REFERENCE_KINDS): pointing another file's reference at
        # a breath that is not itself an IC manoeuvre would silently create a reference
        # to nothing useful. 'Reference manoeuvres…' has no such precondition — the full
        # picker lets any file's/breath's typed manoeuvres be linked regardless of what
        # (if anything) this particular breath is typed as — so it stays enabled here;
        # both funnels re-check the run lock themselves (same "gate the action, not the
        # tab" split _set_breath_type already uses).
        menu.addSeparator()
        name = self._selected_filename()
        current_kind = self._current_breath_kind(name, breath_no) if name else None
        can_ref = current_kind in _IC_REFERENCE_KINDS
        ref_for = menu.addMenu("Use as IC reference for")
        ref_for.setEnabled(can_ref and bool(name))
        ref_for.menuAction().setStatusTip(
            "Point another file's (or this group's) IC reference at this manoeuvre."
            if can_ref else
            "Type this breath as an IC manoeuvre (or IC + FVC) first.")
        if name:
            group = group_key(name, self.state.settings)
            a_file = ref_for.addAction("This file")
            a_file.triggered.connect(
                lambda _checked=False: self._set_reference(name, breath_no, "file"))
            a_group = ref_for.addAction(f"All files of {group}")
            a_group.triggered.connect(
                lambda _checked=False: self._set_reference(name, breath_no, "group"))
            a_all = ref_for.addAction("All files")
            a_all.triggered.connect(
                lambda _checked=False: self._set_reference(name, breath_no, "all"))
        ref_manoeuvres = menu.addAction("Reference manoeuvres…")
        ref_manoeuvres.setStatusTip(
            "Browse and link IC/FVC/baseline/maximal-effort manoeuvres across files.")
        ref_manoeuvres.triggered.connect(
            lambda _checked=False: self._open_reference_picker(name))
        return menu

    def _current_breath_kind(self, file, breath_no):
        """The ``BreathTypeEntry.kind`` already recorded for ``file``'s ``breath_no``,
        or ``None`` if it is untyped (plain tidal, or excluded — neither is a
        ``BreathTypeEntry``, see ``_set_breath_type``'s own three-state contract)."""
        entry = next((t for t in self.state.settings.processing.breath_types
                     if t.file == file and t.breath == breath_no), None)
        return entry.kind if entry is not None else None

    def _set_reference(self, file, breath_no, scope):
        """M-37 funnel for the quick 'Use as IC reference for' submenu: point ``scope``
        (``'file'`` | ``'group'`` | ``'all'``) at ``file``'s ``breath_no`` — already
        typed ``'ic'``/``'ic_fvc'``, see ``_IC_REFERENCE_KINDS`` — as its IC reference.
        Same run-lock / folder-stamp-only-on-creation / one ``settings_edited`` contract
        as ``_set_breath_type``: an EXISTING ``ReferenceEntry``/``GroupReferenceEntry``'s
        ``folder`` is never rewritten by this, only a brand-new one gets stamped. The
        full slot/file/breath picker (``_open_reference_picker``) is the OTHER way to
        write ``processing.references``/``reference_defaults``; this is the one-click
        shortcut for the single most common case (this breath's own file IS the IC
        source).

        Returns ``scope`` again on success, or ``None`` if the write was blocked (a run
        in progress) or ``scope`` is not one of the three recognised values."""
        from respmech.core.summary import group_key  # noqa: PLC0415

        if self._run_active:
            msg = "Reference editing is locked while a run is in progress."
            self._set_status(msg)
            self.write_action_blocked.emit(msg)
            return None
        if scope not in ("file", "group", "all"):
            return None
        proc = self.state.settings.processing
        folder = self.state.settings.input.folder

        def _point_file_at(target):
            # A fresh BreathRef PER entry, never one shared instance handed to every
            # target (self-review finding): a scope='all' write used to alias the SAME
            # BreathRef — and its mutable .breaths LIST — across every file's
            # ReferenceEntry.ic, so mutating one file's reference later (e.g. a future
            # multi-breath edit) silently corrupted every other file's reference too.
            entry = next((e for e in proc.references if e.file == target), None)
            if entry is None:
                entry = ReferenceEntry(file=target, folder=folder)
                proc.references.append(entry)
            entry.ic = BreathRef(file=file, breaths=[breath_no])

        if scope == "file":
            _point_file_at(file)
            where = "this file"
        elif scope == "group":
            group = group_key(file, self.state.settings)
            entry = next((e for e in proc.reference_defaults if e.group == group), None)
            if entry is None:
                entry = GroupReferenceEntry(group=group, folder=folder)
                proc.reference_defaults.append(entry)
            entry.ic = BreathRef(file=file, breaths=[breath_no])
            where = f"all files of {group}"
        else:                                              # "all"
            targets = [os.path.basename(f) for f in matching_files(
                self.state.settings.input.folder, self.state.settings.input.files)]
            for target in targets:
                _point_file_at(target)
            where = "all files"

        self.settings_edited.emit()
        self._sync_rail_breath_state()
        self._refresh_reference_chip()
        self._request_batch_recompute()
        self._set_status(f"{file}: breath {breath_no} set as the IC reference for {where}.")
        return scope

    def _open_reference_picker(self, filename=None):
        """M-37: the full picker (``ReferencePickerDialog``) for all four reference
        slots at once, opened from the type menu's 'Reference manoeuvres…' (the
        currently previewed file) or from the file rail's row context menu
        (``FileRail.referencesRequested``, a specific ``filename``). Same run-lock
        contract as ``_set_reference``; writes nothing if the dialog is cancelled or
        nothing was actually touched.

        Only writes the slots ``dlg.touched_slots()`` reports (self-review finding: an
        earlier version wrote every slot ``staged()`` returned whenever ANY slot
        differed from the existing explicit entry, which silently truncated a
        legitimate MULTI-breath own-typed-breath reference to the picker's
        single-breath selection the instant the user touched a completely different
        slot and clicked OK — see ``ReferencePickerDialog.staged()``'s own docstring)."""
        if self._run_active:
            msg = "Reference editing is locked while a run is in progress."
            self._set_status(msg)
            self.write_action_blocked.emit(msg)
            return
        name = filename or self._selected_filename()
        if not name:
            return
        from respmech.ui.reference_picker_dialog import ReferencePickerDialog
        s = self.state.settings
        files = sorted({os.path.basename(f) for f in matching_files(
            s.input.folder, s.input.files)} | {name})
        dlg = ReferencePickerDialog(name, s, files, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        touched = dlg.touched_slots()
        if not touched:
            return
        staged = dlg.staged()
        proc = s.processing
        entry = next((e for e in proc.references if e.file == name), None)
        if entry is None:
            entry = ReferenceEntry(file=name, folder=s.input.folder)
            proc.references.append(entry)
        for slot in touched:
            setattr(entry, slot, staged[slot])
        self.settings_edited.emit()
        self._sync_rail_breath_state()
        self._refresh_reference_chip()
        self._request_batch_recompute()
        self._set_status(f"{name}: reference manoeuvres updated.")

    def _apply_breath_type_choice(self, breath_no, kind):
        result = self._set_breath_type(breath_no, kind)
        self._update_breath_bar()           # also resets the selector if the write was refused
        if result is None:
            return
        name = self._selected_filename()
        # .get(), mirroring _build_type_menu's own fallback: a future menu offering a
        # kind not in _TYPE_MENU_LABELS (M-31) must not KeyError here AFTER the
        # settings write above has already happened (self-review finding).
        label = _TYPE_MENU_LABELS.get(kind, kind.capitalize())
        self._set_status(f"{name}: breath {breath_no} set to {label.lower()}.")

    def _handle_type_requested(self, breath_no, scene_pos):
        """Slot for ``BreathSpansItem.typeRequested`` (right-click/Ctrl+left-click on a
        span): resolve the emitting item's own scene to a global point (the item does
        not know which widget it is embedded in — it can be any plot on the mechanics
        stack or any EMG view) and pop the type menu up there.

        M-31: the offered kinds differ by signal set — this one slot serves BOTH the
        flow-bearing Mechanics stack and the EMG-only 'EMG – segments' tab (see
        ``_draw_breath_overlays``'s own docstring on why the two share this machinery),
        and the two shapes support genuinely different manoeuvre kinds:

        - 'Rest' names a quiet-breathing noise-reference SEGMENT, which only exists as a
          concept where segmentation (not inspiration/expiration) is the unit — offering
          it on a flow-bearing file would type a real tidal breath as a noise reference
          no resolver (``resolve_noise_reference_mode``) ever reads for that shape.
        - IC/FVC/IC+FVC/max_insp/sniff all need ``core.pipeline.run_batch``'s per-breath
          extraction loop, which is unconditionally skipped for an EMG-only set
          (``if not emg_only:`` around the ``manoeuvreslib.extract`` call — a segment has
          no inspiration/expiration split for it to read, see ``core.analysis.segments``'
          own module docstring). Offering them on the 'EMG – segments' tab would type a
          segment as, say, 'ic' with NO Manoeuvres row ever appearing for it — the kind
          is silently a dead end there today (self-review finding: this ticket's first
          cut offered the full set on both tabs, which is what the guard below fixes)."""
        item = self.sender()
        view = None
        if item is not None:
            sc = item.scene()
            if sc is not None and sc.views():
                view = sc.views()[0]
        if view is not None:
            # QGraphicsView.mapFromScene(QPointF) returns a QPoint in PySide6, which
            # has no .toPoint() (only QPointF does) — a real right-click crashed this
            # slot every time (self-review finding), silently: PySide6 prints the
            # AttributeError to stderr from inside the signal emission and swallows
            # it, so the only visible symptom was "nothing happens".
            global_pos = view.mapToGlobal(view.mapFromScene(scene_pos))
        else:                                          # pragma: no cover — defensive only
            global_pos = QCursor.pos()
        menu = self._build_type_menu(breath_no, self._type_menu_kinds())
        menu.popup(global_pos)

    def _type_menu_kinds(self):
        """The breath kinds on offer for the current signal set (see
        ``_handle_type_requested`` for why they differ). Shared by the right-click menu
        and the breath action bar, so the two can never offer different choices."""
        caps = Capabilities.from_settings_or_none(self.state.settings)
        emg_only = bool(caps is not None and caps.mode == "emg_only")
        if emg_only:
            return tuple(k for k in _TYPE_MENU_KINDS if k in ("tidal", "excluded", "rest", "other"))
        return tuple(k for k in _TYPE_MENU_KINDS if k != "rest")

    # -- Breath action bar ---------------------------------------------------
    # The right-click menu's choices as controls on the toolbar row next to Refresh: a type
    # selector, the 'Use as IC reference for' menu and 'Reference manoeuvres…'. They act on
    # the MARKED breath (_selected_breath), reuse the menu's own funnels
    # (_apply_breath_type_choice / _set_reference / _open_reference_picker) and are
    # disabled until a breath is marked. _update_breath_bar is the one place that decides
    # their state.
    def _build_breath_bar(self, bar):
        bar.addSpacing(12)
        self.breath_bar_label = QLabel("Breath:")
        self.breath_type_combo = QComboBox()
        self.breath_type_combo.setAccessibleName("Type of the marked breath")
        self.breath_type_combo.setPlaceholderText("Type")
        self.breath_type_combo.activated.connect(self._on_breath_type_combo)
        self.btn_breath_ic_ref = QPushButton("IC reference for")
        self.btn_breath_ic_ref.setAccessibleName("Use the marked breath as IC reference")
        self.btn_breath_ic_ref.setMenu(QMenu(self.btn_breath_ic_ref))
        self.btn_breath_refs = QPushButton("Reference manoeuvres…")
        self.btn_breath_refs.clicked.connect(lambda _c=False: self._open_reference_picker())
        self._breath_combo_kinds = ()
        # any write to the exclusions/types (from whichever surface) re-syncs the bar
        self.settings_edited.connect(self._update_breath_bar)
        bar.addWidget(self.breath_bar_label)
        for w in (self.breath_type_combo, self.btn_breath_ic_ref, self.btn_breath_refs):
            w.setEnabled(False)          # nothing marked yet; the file rail does not exist
            bar.addWidget(w)             # at this point, so no _update_breath_bar() here

    def _on_breath_type_combo(self, index):
        sel = self._selected_breath
        kind = self.breath_type_combo.itemData(index)
        if sel is None or kind is None:
            return
        if kind == (self._breath_kind_now(sel) or "tidal"):
            return                      # same choice again: nothing to write or recompute
        self._apply_breath_type_choice(sel, kind)

    def _update_breath_bar(self):
        """Sync the action bar with the marked breath, its kind, the signal set and the run
        lock. Safe to call at any time (before the first render, with nothing marked)."""
        combo = getattr(self, "breath_type_combo", None)
        if combo is None:
            return
        from respmech.core.summary import group_key  # noqa: PLC0415

        name = self._selected_filename()
        sel = self._selected_breath
        marked = sel is not None and bool(name)
        locked = bool(self._run_active)
        usable = marked and not locked
        if not marked:
            why = "Mark a breath first: click it in a plot."
        elif locked:
            why = "Locked while a run is in progress."
        else:
            why = ""
        self.breath_bar_label.setText(f"Breath {sel}:" if marked else "Breath:")
        kinds = self._type_menu_kinds()
        combo.blockSignals(True)
        try:
            if kinds != self._breath_combo_kinds:
                combo.clear()
                for k in kinds:
                    combo.addItem(_TYPE_MENU_LABELS.get(k, k.capitalize()).rstrip("…"), k)
                self._breath_combo_kinds = kinds
            kind = (self._breath_kind_now(sel) or "tidal") if marked else None
            idx = combo.findData(kind) if kind is not None else -1
            combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
        combo.setEnabled(usable)
        combo.setToolTip(why or "Set the type of the marked breath (same choices as the right-click menu).")
        # 'IC reference for' needs the marked breath to be an IC manoeuvre already, exactly
        # as in the right-click menu.
        typed = self._current_breath_kind(name, sel) if marked else None
        can_ref = usable and typed in _IC_REFERENCE_KINDS
        menu = self.btn_breath_ic_ref.menu()
        menu.clear()
        if marked:
            group = group_key(name, self.state.settings)
            menu.addAction("This file").triggered.connect(
                lambda _c=False, n=name, b=sel: self._set_reference(n, b, "file"))
            menu.addAction(f"All files of {group}").triggered.connect(
                lambda _c=False, n=name, b=sel: self._set_reference(n, b, "group"))
            menu.addAction("All files").triggered.connect(
                lambda _c=False, n=name, b=sel: self._set_reference(n, b, "all"))
        self.btn_breath_ic_ref.setEnabled(can_ref)
        if not usable:
            self.btn_breath_ic_ref.setToolTip(why)
        elif not can_ref:
            self.btn_breath_ic_ref.setToolTip("Type this breath as an IC manoeuvre (or IC + FVC) first.")
        else:
            self.btn_breath_ic_ref.setToolTip("Point another file's (or this group's) IC reference at this manoeuvre.")
        self.btn_breath_refs.setEnabled(usable)
        self.btn_breath_refs.setToolTip(
            why or "Browse and link IC/FVC/baseline/maximal-effort manoeuvres across files.")

    def _request_batch_recompute(self):
        """Debounced recompute of the mechanics test run (Campbell + per-breath table) after
        a breath is toggled, so the averaged result tracks the overlay live. Inert headless
        (never spins the loop), like _request_autorun."""
        if getattr(self, "_batch_recompute_pending", False):
            return
        self._batch_recompute_pending = True
        QTimer.singleShot(0, self._run_batch_recompute)

    def _run_batch_recompute(self):
        self._batch_recompute_pending = False
        self._schedule("batch")

    # -- breath overlays on the EMG views ----------------------------------
    def _breath_kind_now(self, breath_no):
        """Live 3-way kind for one breath of the CURRENT file — the single source of
        truth for the EMG views' overlay colour (mirrors ``core.compute``'s own
        ignorebreaths+breathkinds union, see M-19's ``compute.breathkinds`` docstring).
        A BREATH_KINDS member if typed (``processing.breath_types``), the pseudo-kind
        ``'excluded'`` if only manually excluded (``processing.exclude_breaths``), else
        ``None`` (plain tidal breathing). Used by ``_paint_breaths``, which redraws
        from ``self._breaths`` rather than a fresh ``stage_mechanics_preview`` call and
        so cannot read the ``kind`` that call already baked into the mechanics stack's
        own spans."""
        name = self._selected_filename()
        proc = self.state.settings.processing
        type_entry = next((t for t in proc.breath_types
                           if t.file == name and t.breath == breath_no), None)
        if type_entry is not None:
            return type_entry.kind
        excl_entry = next((e for e in proc.exclude_breaths if e.file == name), None)
        if excl_entry is not None and breath_no in excl_entry.breaths:
            return "excluded"
        return None

    def _exclusion_carried_for(self, name):
        """True iff ``name``'s exclusion entry was recorded against a DIFFERENT (or
        unrecorded) recordings folder than the one currently loaded — the file-scoped
        counterpart of ``core.settings.carried_over_state``, used to hatch the overlay
        (``_breath_brush``) and word the QC line (``_update_qc_overview``) for exactly the
        file on screen, without recomputing the whole-analysis state on every repaint."""
        from respmech.core.settings import is_carried_folder
        entry = next((e for e in self.state.settings.processing.exclude_breaths if e.file == name), None)
        if entry is None or not entry.breaths:
            return False
        return is_carried_folder(entry.folder, self.state.settings.input.folder)

    def _paint_breaths(self, view, plot_items, offset, label_y):
        """Idempotent per-view overlay: remove the view's previous items (guarded),
        then shade every breath on each plot and a number on the first plot.
        Colour is taken from the LIVE exclusion set.

        One BreathSpansItem PER PLOT carries every breath's region (same D15
        item-count fix as _draw_breath_overlays — the raw EMG view can have as
        many plots as EMG channels, so this was the same 11-items-per-breath
        blow-up, just with a channel count instead of 5). The redundant boundary
        line each region used to get is dropped here too, for the same reason."""
        rec = self._bov.get(view)
        if rec:
            rec.get("unpin", lambda: None)()           # detach the old pin slot eagerly
            for plot, item in rec.get("items", []):
                try:
                    plot.removeItem(item)
                except Exception:                      # noqa: BLE001
                    pass
        self._bov[view] = {"items": [], "regions": {}, "texts": {}, "unpin": lambda: None}
        if not plot_items or not self._breaths:
            return
        carried = self._exclusion_carried_for(self._selected_filename())
        reg_map = self._bov[view]["regions"]; txt_map = self._bov[view]["texts"]
        items = self._bov[view]["items"]
        spans = [(num, t0 + offset, t1 + offset, self._breath_kind_now(num))
                for (num, t0, t1, _k) in self._breaths]
        for p in plot_items:
            item = BreathSpansItem()
            item.set_spans([(a, b, self._breath_brush(kind, carried=carried), num)
                            for num, a, b, kind in spans])
            item.setZValue(-10)
            item.typeRequested.connect(self._handle_type_requested)
            p.addItem(item)
            items.append((p, item))
            for i, (num, _a, _b, _k) in enumerate(spans):
                reg_map.setdefault(num, []).append((item, i))
        for num, a, b, kind in spans:
            txt = self._breath_text(num, kind, span=(a, b))
            txt.setPos((a + b) / 2.0, label_y)
            plot_items[0].addItem(txt, ignoreBounds=True)
            items.append((plot_items[0], txt)); txt_map[num] = txt
        self._label_headroom(plot_items[0], label_px=self._label_px(txt_map))   # labels above the signal
        self._bov[view]["unpin"] = self._pin_breath_labels(plot_items[0], txt_map)
        for it in (i for _p, i in items if isinstance(i, BreathSpansItem)):
            it.set_selected(self._selected_breath)
        for num, txt in txt_map.items():
            self._style_breath_label(txt, num == self._selected_breath)

    def _repaint_view_breaths(self, view):
        """Repaint one EMG view IF it has rendered real data (label_y sentinel set)."""
        if view == "raw":
            plot_items, label_y = self._emg_raw_subplots, self._raw_label_y
        elif view == "detail":
            plot_items, label_y = [self.emg_plots.getPlotItem()], self._detail_label_y
        elif view == "result":
            plot_items, label_y = [self.emg_result_plots.getPlotItem()], self._result_label_y
        else:
            return
        if label_y is None or not plot_items:
            return
        try:
            self._paint_breaths(view, plot_items, self._trim_offset_s, label_y)
        except Exception:                              # noqa: BLE001 — overlay is cosmetic
            pass

    def _reset_breath_state(self):
        """On a file change, drop stale overlays + spans so a new-file EMG job that
        finishes before the new mech never shows the previous file's numbers."""
        # invalidate any deferred _render_preview_async continuation still pending for
        # the file being left — see _render_preview_stage1/_render_preview_async's
        # docstring. self.plots itself is cleared by the caller (_begin_file_switch),
        # so there is nothing of the mechanics stack's own overlays left to remove here.
        self._mech_render_gen += 1
        self._mech_unpin()
        self._mech_unpin = lambda: None
        for rec in self._bov.values():
            rec.get("unpin", lambda: None)()
            for plot, item in rec.get("items", []):
                try:
                    plot.removeItem(item)
                except Exception:                      # noqa: BLE001
                    pass
        self._bov = {}
        self._breaths = []
        self._suggested_fvc = None
        self._breath_spans = {}
        self._breath_regions = {}
        self._breath_texts = {}
        self._clear_breath_selection()          # breath numbers belong to the file being left
        self._campbell_breaths = None           # and so do the cached loops a mark would redraw
        self._emg_raw_subplots = []
        self._raw_label_y = self._detail_label_y = self._result_label_y = None
        self._trim_offset_s = 0.0
        # a result-checkbox toggle re-renders self._emg_all synchronously (no token
        # gate); drop the previous file's staged result so it can't repaint the old
        # curves and re-arm the result sentinel with them
        self._emg_all = None
        self.emg_result_plots.clear()
        # _previewed_file only advances on a COMPLETED mech render, so leaving it set
        # here would make a flip back to the last-rendered file look like "no change"
        # and skip this very reset while the new file's jobs are still in flight
        self._previewed_file = None
        # the persistent caption is a claim about breaths on screen — drop it here too, or
        # it would keep asserting the PREVIOUS file's breath/exclusion count over an emptied
        # Mechanics tab until (or unless) the new file's render lands.
        self.mech_caption.setFullText("")

    def _trend_probe_settings(self):
        """The volume-conditioning settings behind the current trend probe, in the same
        shape the advanced dialog stages them (see _TREND_PROBE_KEYS / _trend_hint)."""
        vol, samp = self.state.settings.processing.volume, self.state.settings.processing.sampling
        src = {**vars(vol), **vars(samp)}
        return {k: src.get(k) for k in _TREND_PROBE_KEYS}

    def _advanced_live_breath_count(self, probe, v):
        """How many breaths the volume-based detector finds in the previewed file's
        (drift-corrected) volume with the STAGED breath-detection thresholds — the live
        read-out in Mechanics — advanced… while end-expiratory trend correction is off (see
        _trend_hint above). Deliberately always volume-based regardless of the staged
        segmentation method: only the volume probe is staged in this dialog, never flow.

        Uses the EXISTING ``compute.separateintobreathsbyvolume`` — the same function
        ``stage_mechanics_preview`` calls for volume-based segmentation — rather than a new
        peak-finding implementation in the UI layer, against a lightweight stand-in for the
        legacy settings namespace it expects and zero-filled arrays for the phase channels
        this dialog does not stage (flow/poes/pgas/pdi): only their LENGTH matters to the
        breath count, since only ``volume`` drives the peak search.

        Returns ``None`` if the thresholds cannot be evaluated at all, e.g. a minimum
        distance/width of 0 s, which ``scipy.signal.find_peaks`` rejects outright — a
        confidently wrong count would be worse than an explanatory line saying there is
        none."""
        import types

        from respmech.core import compute
        n = len(probe)
        zeros = np.zeros(n)
        stand_in = types.SimpleNamespace(
            processing=types.SimpleNamespace(
                mechanics=types.SimpleNamespace(
                    peakheight=v["height"], peakdistance=v["distance_s"],
                    peakwidth=v["width_s"], excludebreaths={})),
            input=types.SimpleNamespace(
                format=types.SimpleNamespace(
                    samplingfrequency=self.state.settings.input.format.sampling_frequency)))
        try:
            breaths = compute.separateintobreathsbyvolume(
                self._trend_probe_file or "", np.arange(n), zeros, probe,
                zeros, zeros, zeros, [], [], stand_in)
        except Exception:                   # noqa: BLE001 — a live hint is never fatal
            return None
        return len(breaths)

    def _on_batch_result(self, result):
        """Render the automatic test run. For a flow-bearing set: the per-breath table +
        Campbell (mechanics-only, so no EMG/fidelity here). For an EMG-only set (M-26):
        the segments tab's own per-segment table instead — there is no flow/volume/
        pressure to draw a Campbell diagram against, and the Mechanics tab holding
        ``self.table``/``self.campbell`` is not even shown for that shape (see
        ``subtab_plan``). The noise_report branch is retained for the direct-call path
        that still carries one, and applies to both shapes alike.

        Reads CURRENT caps (not the dispatching job's own frozen ``job.panels``, unlike
        ``_on_job_done``'s generic spinner/error bookkeeping) because this method has no
        ``job`` parameter at all — it is called generically as ``_RENDER[job.kind]
        (result)``. Safe only because ``_on_job_done`` already returned early on a stale
        job (``job.token != self._tokens[job.kind]``) before ever reaching here, and any
        settings edit that changes ``caps.mode`` synchronously bumps ``_tokens['batch']``
        (``sync_from_settings`` -> ``_cancel_inflight``) before the GUI event loop can
        deliver a stale job's queued ``finished`` signal — so by the time this runs,
        "not stale" already implies "caps unchanged since dispatch", i.e. this always
        agrees with what ``self._panels_for('batch')`` would also say. Re-check this
        invariant if `_RENDER` calls are ever made asynchronous or a caps-affecting edit
        is ever allowed to bypass `sync_from_settings`'s synchronous token bump."""
        cur = self._selected_filename()
        caps = Capabilities.from_settings_or_none(self.state.settings)
        emg_only = bool(caps is not None and caps.mode == "emg_only")
        fr = None
        if getattr(result, "files", None):
            fr = result.files.get(cur) or next(iter(result.files.values()), None)
        if fr is None or getattr(fr, "error", None):
            err = getattr(fr, "error", None) or "The run produced no result for this file."
            kind = getattr(fr, "error_kind", None) or str(err).split(":", 1)[0]
            if cur:
                self.file_rail.mark_result(cur, ok=False, error=str(err))
            self._qc_overview_not_assessed(
                err, chip=self.segments_qc_overview if emg_only else None)
            if kind in _SOFT_FILE_ERRORS:
                # A precondition failure of THIS recording, not a fault: the mech/segments
                # preview keeps drawing the channels, so don't raise a 'Test run failed'
                # card over them. The reason still has to live on the panel(s) — the
                # status line is a single shared label that the EMG/ECG jobs overwrite
                # moments later, which would leave a blank table explaining nothing.
                if emg_only:
                    self._segtable_model.set_dataframe(None)
                else:
                    self._table_model.set_dataframe(None)
                    self._set_wob_table_note(None)
                    self._fill_manoeuvres_table(None, None)
                    self.campbell.figure.clear(); self.campbell.draw()
                    self._forget_campbell()   # the export must not resurrect a cleared diagram
                for p in self._panels_for("batch"):
                    self._overlays[p].show_error(
                        f"Not processed — {short_error(str(err))}", str(err))
                return
            # raise so _on_job_done paints a copyable "Test run failed" error card
            raise _FileRunError(err)
        if cur:
            n_breaths = 0 if fr.breaths_table is None else len(fr.breaths_table)
            # M-32: role rides along with the same successful result breaths/verdict
            # already come from — see FileRailEntry.role's own docstring.
            self.file_rail.mark_result(cur, ok=True, breaths=n_breaths,
                                       role=getattr(fr, "role", None))
        is_reference_only = getattr(fr, "role", "tidal") == "reference"
        if is_reference_only:
            # M-30: no tidal breaths at all in this file — nothing for the Campbell
            # diagram/breath table to draw, but the Manoeuvres table (this file's whole
            # reason for being typed) has real content, so show that instead of an
            # empty or error-looking mechanics view. Falls through to the noise-report
            # handling below like every other shape (self-review finding: an EARLIER
            # version of this branch returned immediately here, which for a
            # reference-only file that ALSO carries EMG channels in a noise-reduction
            # test silently dropped the batch's auto-tuned prop_decrease and skipped
            # re-conditioning the EMG views — result.noise_report is built once per
            # whole test, independent of any one file's role).
            self._fill_table(fr.manoeuvres_table)
            self._table_panel._title_label.setFullText("Manoeuvres (reference-only file)")
            # the primary table ALREADY shows this file's manoeuvres (above) — the
            # separate stacked section below it would just repeat the same rows.
            self._fill_manoeuvres_table(None, None)
            self.campbell.figure.clear(); self.campbell.draw()
            self._forget_campbell()
            self._update_qc_overview(fr)
        elif emg_only:
            self._fill_segtable(fr.breaths_table)
            self._fill_manoeuvres_table(None, None)   # M-31 scope is the Mechanics tab only
            self._update_qc_overview(fr, chip=self.segments_qc_overview)
        else:
            self._fill_table(fr.breaths_table)
            self._campbell_manoeuvres = fr.manoeuvres     # read by the flow-only branch
            self._draw_campbell_or_loop(fr.breaths)
            self._update_qc_overview(fr)                  # P16 QC line, for THIS file only
            self._fill_manoeuvres_table(fr.manoeuvres_table, fr.manoeuvres)
        nr = getattr(result, "noise_report", None)
        if nr:
            self.render_noise_report(result)         # sets its own status incl. prop_decrease
            # the auto-selected suppression lives only in the report — write it back so
            # the spinbox AND the re-conditioned EMG views use the value the batch chose
            # (the core never mutates settings), then re-dispatch those EMG views. Only
            # the direct-call path carries a report now; the auto mechanics-only batch
            # does not, so it never triggers this redundant re-condition (the separate
            # 'noise' job owns that).
            chosen = nr.get("prop_decrease")
            if chosen is not None and self.state.settings.processing.emg.noise.auto_prop:
                self.state.settings.processing.emg.noise.prop_decrease = float(chosen)
            self._load_noise_params()                # reflect the chosen prop_decrease
            noise = self.state.settings.processing.emg.noise
            if noise.enabled and noise.auto_prop and self.state.settings.input.channels.emg:
                self._schedule("emg_all")
                self._schedule("emg_detail")
        elif is_reference_only:
            n_typed = len(fr.manoeuvres or {})
            self._set_status(
                f"Reference manoeuvres only — {n_typed} typed breath"
                f"{'s' if n_typed != 1 else ''}, no tidal table")
        else:
            n = len(fr.breaths_table)
            self._set_status(f"Test run OK: {n} breaths (nothing written)")

    def _fill_table(self, df):
        self._table_model.set_dataframe(df)
        resize_result_table(self.table)
        self._set_wob_table_note(df)

    def _fill_manoeuvres_table(self, df, manoeuvres):
        """Show the Manoeuvres sheet below the per-breath table (M-31), visible only
        once the previewed file's most recent test run actually typed at least one
        breath — ``manoeuvres`` (``FileResult.manoeuvres``, a ``{breath_no: ...}`` dict)
        decides visibility rather than ``df`` alone, matching ``build_manoeuvre_table``'s
        own "empty dict -> None" contract (``core/results.py``) so the two can never
        disagree. Passing ``(None, None)`` hides the section — used for a soft file
        error, an EMG-only file (out of this ticket's scope) and a reference-only file
        (whose own manoeuvres already fill the PRIMARY table, see ``_on_batch_result``),
        so an untyped or not-applicable file's panel looks exactly as it did before this
        ticket, with no empty second table left visible."""
        has_manoeuvres = bool(manoeuvres)
        self._manoeuvres_model.set_dataframe(df if has_manoeuvres else None)
        if has_manoeuvres:
            resize_result_table(self.manoeuvres_table)
        self._manoeuvres_section.setVisible(has_manoeuvres)

    def _set_wob_table_note(self, df):
        """Name the work-of-breathing source in the table's own header (D22,
        UI-overhaul), so a reader can see — without opening Advanced… — that the five
        wob* columns (``wobtotal``, ``wob_in_total``, ``wob_ex_total``, ``wob_in_ela``,
        ``wob_in_res``) are one whole-file value repeated on every row whenever
        "Work of breathing from" is Average (``processing.wob.calc_from``, default). The
        DataFrame's own column names are never touched — only this panel's title text —
        so the golden-pinned result tables are unaffected.
        """
        calc_from = self.state.settings.processing.wob.calc_from
        if calc_from == "average" and df is not None and "wobtotal" in getattr(df, "columns", ()):
            try:
                val = float(df["wobtotal"].iloc[0])
                text = (f"Per-breath results — work of breathing: {val:.2f} J·min⁻¹, "
                        "computed from the averaged breath, same on every row "
                        "(Advanced… → Work of breathing)")
            except (TypeError, ValueError, IndexError):
                text = ("Per-breath results — work of breathing computed from the "
                        "averaged breath, same on every row "
                        "(Advanced… → Work of breathing)")
        else:
            text = "Per-breath results"
        self._table_panel._title_label.setFullText(text)

    def _forget_campbell(self):
        """Drop the cached breaths and disable the export whenever the figure is cleared.

        The export re-renders from this cache (to force a light figure in dark mode), so a
        cache that outlives its diagram is not merely stale — it would write a figure for a
        file the user is no longer looking at, under that file's name. Clearing the figure
        and clearing what it was drawn from have to be the same act."""
        self._campbell_breaths = None
        self._campbell_manoeuvres = None
        try:
            self.btn_export_fig.setEnabled(False)
        except Exception:                       # pragma: no cover - button may not exist yet
            pass

    def _update_campbell_panel_title(self):
        """Name the Campbell panel — and its export button — for the current signal set
        (M-17, R7): "Campbell diagram" with Poes declared, "Flow-volume loop" without it.
        Cheap and idempotent, so it is safe to call on every settings sync as well as every
        draw — see ``_draw_campbell_or_loop`` and ``sync_from_settings``.

        ``titled_panel``'s own docstring warns that ``title_floor_chars`` (the Campbell
        panel's only caller of it) sizes the header's never-squeeze-below floor ONCE, from
        the title given at construction, and does not follow a later ``setFullText()`` — a
        floor sized for a short opt-in title would go stale against a longer one set here.
        Checked, not merely assumed: both titles this function ever sets are at least as
        long as the floor's own ``title_floor_chars`` cap (10), so ``min(len(title), 10)``
        is 10 either way and the floor this call inherits is identical regardless of which
        title built the panel. A THIRD title introduced later must satisfy the same check
        (>= 10 characters) or recompute the floor explicitly. Uses
        ``from_settings_or_none``: this runs on every settings sync, including
        ``PreviewScreen`` construction on ``MainWindow``'s no-try/except open path, so a
        malformed ``analysis.signals`` must degrade to the poes-less title rather than
        crash — see ``Capabilities.from_settings_or_none``."""
        caps = Capabilities.from_settings_or_none(self.state.settings)
        poes = caps is not None and caps.poes
        title = "Campbell diagram" if poes else "Flow-volume loop"
        panel = getattr(self, "_campbell_panel", None)
        if panel is not None:
            panel._title_label.setFullText(title)
        btn = getattr(self, "btn_export_fig", None)
        if btn is not None:
            btn.setText("Export Campbell…" if poes else "Export flow-volume…")
            # Not title.lower(): "Campbell" is a proper noun (E.J.M. Campbell), so a bare
            # lower() would misspell it mid-sentence.
            tooltip_noun = "Campbell diagram" if poes else "flow-volume loop"
            btn.setToolTip(f"Save the {tooltip_noun} as a PNG or PDF.")

    def _draw_campbell_or_loop(self, breaths, pal=None):
        """Dispatch the Campbell panel to whichever diagram this signal set can actually
        show (M-17, R7): the Campbell (volume-vs-Poes) diagram when Poes is declared, or a
        tidal flow-volume loop when it is not — a Poes-less analysis has no pressure trace
        to plot work of breathing against. Same cached-breaths/export plumbing either way
        (``_campbell_breaths``, ``btn_export_fig``), because both draw functions set it.
        Same ``from_settings_or_none`` tolerance as ``_update_campbell_panel_title`` (a
        malformed signal set degrades to the flow-volume loop rather than crashing)."""
        self._update_campbell_panel_title()
        caps = Capabilities.from_settings_or_none(self.state.settings)
        if caps is not None and caps.poes:
            self._draw_campbell(breaths, pal=pal)
            return
        # No Poes but a forced vital capacity typed in this file -> the tidal loops
        # inside that file's own MFVL; anything else keeps the plain flow-volume loop.
        placed = self._placed_mfvl(breaths)
        # the tidal-loops-in-MFVL picture has no per-breath loop to mark: skip the redraw
        self._loop_no_mark = placed is not None
        if placed is not None:
            self._draw_flow_volume_in_mfvl(breaths, placed, pal=pal)
        else:
            self._draw_flow_volume_loop(breaths, pal=pal)

    def _placed_mfvl(self, breaths):
        """``core.analysis.mfvl.placed_tidal_loops`` for the file just previewed, or ``None``
        (no typed FVC, no tidal breath, or the placement itself failing: a preview must
        degrade to the plain loop, never raise). Imported here so the compute core stays off
        the startup path."""
        manoeuvres = getattr(self, "_campbell_manoeuvres", None)
        if not manoeuvres:
            return None
        from respmech.core.analysis import mfvl as _mfvl        # noqa: PLC0415
        s = self.state.settings
        try:
            return _mfvl.placed_tidal_loops(breaths, manoeuvres, s.processing.mfvl,
                                            s.processing.lung_volume.ic)
        except Exception:                                       # noqa: BLE001
            return None

    def _draw_flow_volume_in_mfvl(self, breaths, placed, pal=None):
        """The Campbell panel for a Poes-less analysis whose file carries a forced vital
        capacity: the tidal loops drawn inside that file's MFVL (the same picture the
        ``flow-volume (tidal in MFVL).pdf`` figure writes), in the panel's own theme."""
        from respmech.core.plots import draw_flow_volume_mfvl   # noqa: PLC0415
        self._campbell_breaths = breaths        # kept so the export can re-render it light
        pal = _plot_pal() if pal is None else pal
        fig = self.campbell.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])
        draw_flow_volume_mfvl(ax, placed, loop=pal["mpl_loop"], mean=pal["mpl_accent"],
                              envelope=pal["fg"], marker=pal["mpl_zeroline"], label=pal["fg"])
        ax.set_xlabel(_MFVL_XLABEL_VARIANTS[0])
        ax.set_ylabel(_FV_YLABEL_VARIANTS[0])
        _fit_compact_figure(
            self.campbell, ax,
            legend_kw={"loc": "upper right", "frameon": False, "fontsize": 7},
            xlabel_variants=_MFVL_XLABEL_VARIANTS,
            ylabel_variants=_FV_YLABEL_VARIANTS)
        self.campbell.draw()
        self.btn_export_fig.setEnabled(True)         # a diagram now exists to export

    def _draw_campbell(self, breaths, pal=None):
        self._campbell_breaths = breaths        # kept so the export can re-render it light
        pal = _plot_pal() if pal is None else pal
        fig = self.campbell.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])
        kept = [b for b in breaths.values() if not b["ignored"]]
        for b in kept:
            ax.plot(b["volume"], b["poes"], color=pal["mpl_loop"], lw=0.7, alpha=0.5, zorder=1)
        self._plot_selected_loop(ax, kept, "volume", "poes")
        # P12: overlay the average breath bold, draw the elastic recoil (relaxation)
        # line EELV→EILV, and shade the inspiratory resistive work between the Poes
        # trace and it (the elastic triangle itself is not shaded here).
        self._overlay_campbell_work(ax, kept, pal)
        ax.axhline(0, color=pal["mpl_zeroline"], lw=0.8, zorder=0)
        ax.set_xlabel(_CAMPBELL_XLABEL_VARIANTS[0])
        ax.set_ylabel(_CAMPBELL_YLABEL_VARIANTS[0])
        # D12: volume on x (inverted) and Poes on y — the SAME direction the written
        # Campbell PDF uses (core/plots._pv_average). Before this ticket the two figures
        # were mirror images of each other for the same breaths. Emil's decision
        # 04-08-2026 was to fix the preview, not the writer: the writer's orientation is
        # the 1.x-inherited convention every past export already carries.
        ax.invert_xaxis()
        # No figure title on screen — the panel header carries it. tight_layout() is replaced
        # by _fit_compact_figure, which installs the live tight ENGINE (this figure carried a
        # one-shot PlaceHolderLayoutEngine, so it never re-solved when the splitter moved) and
        # shrinks the labels when the panel is short.
        _fit_compact_figure(
            self.campbell, ax,
            legend_kw={"loc": "lower right", "frameon": False, "fontsize": 7},
            xlabel_variants=_CAMPBELL_XLABEL_VARIANTS,
            ylabel_variants=_CAMPBELL_YLABEL_VARIANTS)
        self.campbell.draw()
        self.btn_export_fig.setEnabled(True)         # a diagram now exists to export

    def _draw_flow_volume_loop(self, breaths, pal=None):
        """The Campbell panel's stand-in for a Poes-less (Flow only) signal set (M-17, R7):
        a tidal flow-volume loop per breath, plotted from the SAME breath dicts
        ``_draw_campbell`` reads (``b["volume"]``/``b["flow"]`` are computed whenever flow
        is declared, unaffected by whether Poes is — see ``core.compute._make_breath``).
        Deliberately without the WOB/elastic-recoil overlay ``_overlay_campbell_work``
        draws for the Campbell diagram: every one of that overlay's own inputs
        (``wobtotal``, ``volumeavg``/``poesavg``, ``eelvavg``/``eilvavg``) comes from the
        pressures family and is never computed for a Poes-less analysis."""
        self._campbell_breaths = breaths        # kept so the export can re-render it light
        pal = _plot_pal() if pal is None else pal
        fig = self.campbell.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])
        kept = [b for b in breaths.values() if not b["ignored"]]
        for b in kept:
            ax.plot(b["volume"], b["flow"], color=pal["mpl_loop"], lw=0.7, alpha=0.5, zorder=1)
        marked = self._plot_selected_loop(ax, kept, "volume", "flow")
        ax.axhline(0, color=pal["mpl_zeroline"], lw=0.8, zorder=0)
        ax.set_xlabel(_FV_XLABEL_VARIANTS[0])
        ax.set_ylabel(_FV_YLABEL_VARIANTS[0])
        # No figure title on screen — the panel header carries it (_update_campbell_panel_title).
        _fit_compact_figure(
            self.campbell, ax,
            # no average/recoil overlay here to label (see docstring); a legend appears only
            # to name the marked breath's loop
            legend_kw={"loc": "upper right", "frameon": False, "fontsize": 7} if marked else None,
            xlabel_variants=_FV_XLABEL_VARIANTS,
            ylabel_variants=_FV_YLABEL_VARIANTS)
        self.campbell.draw()
        self.btn_export_fig.setEnabled(True)         # a diagram now exists to export

    def _plot_selected_loop(self, ax, kept, xkey, ykey):
        """Draw the marked breath's loop (see ``_select_breath``) over the grey ones, in the
        selection green with a legend entry ('breath #n'). Returns whether one was drawn.
        Skipped while an export re-renders the figure (``_loop_export``): a figure made for a
        report must not carry a screen-only mark."""
        n = self._selected_breath
        if n is None or getattr(self, "_loop_export", False):
            return False
        b = next((b for b in kept if b.get("number") == n), None)
        if b is None:
            return False
        ax.plot(b[xkey], b[ykey], color=SELECTED_BREATH_HEX, lw=1.9, alpha=1.0, zorder=5,
                label=f"breath #{n}")
        return True

    def _overlay_campbell_work(self, ax, kept, pal):
        """Draw the average PV loop + elastic recoil line + shaded inspiratory
        resistive work, so the Campbell diagram *shows* work of breathing, not just the loops.
        Defensive: any missing average field falls back to just the loops (no raise)."""
        if not kept:
            return
        b = kept[0]
        insp = b.get("inspiration", {})
        vavg, pavg = b.get("volumeavg"), b.get("poesavg")
        iv, ip = insp.get("volumeavg"), insp.get("poesavg")
        eelv, eilv = b.get("eelvavg"), b.get("eilvavg")   # each [volume, poes]
        try:
            import numpy as np
            if vavg is not None and pavg is not None and len(vavg):
                ax.plot(vavg, pavg, color=pal["mpl_accent"], lw=2.0, zorder=3,
                        label="average breath")
            if eelv is not None and eilv is not None:
                # modified Campbell (opt-in PEEPi): the hatched rectangle goes under the
                # recoil line, exactly as the written figure draws it -- nothing at all
                # (no patch, no legend entry) unless the feature produced a height.
                from respmech.core.plots import draw_peepi_rectangle, mean_peepi_rectangle_height
                draw_peepi_rectangle(ax, eilv, eelv, mean_peepi_rectangle_height(kept),
                                     color=pal["mpl_target"], zorder=2)
                # elastic recoil line: straight line between the two volume endpoints
                ax.plot([eelv[0], eilv[0]], [eelv[1], eilv[1]], color=pal["mpl_target"],
                        ls="--", lw=1.3, zorder=4, label="elastic recoil")
                if iv is not None and ip is not None and len(iv) > 1:
                    iv = np.asarray(iv, float); ip = np.asarray(ip, float)
                    # Poes on the recoil line at each inspiratory volume
                    line_p = np.interp(iv, sorted([eelv[0], eilv[0]]),
                                       [eelv[1], eilv[1]] if eelv[0] <= eilv[0] else [eilv[1], eelv[1]])
                    ax.fill_between(iv, ip, line_p, color=pal["mpl_accent"], alpha=0.18,
                                    zorder=2, label="inspiratory resistive work")
            wob = dict(b.get("wob") or {})
            if "wobtotal" in wob:
                # fg on mpl_bg (not mpl_loop, the colour of the loops the text sits over)
                # clears WCAG's 4.5:1 body-text floor in both themes, and the boxed
                # background keeps it legible over whichever loop happens to pass beneath.
                ax.text(0.02, 0.98, f"WOB {wob['wobtotal']:.2f} J·min⁻¹",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9,
                        fontweight="bold", color=pal["fg"],
                        bbox=dict(facecolor=pal["mpl_bg"], edgecolor="none", alpha=0.85, pad=2))
            # Top-right / bottom-left: with volume on x (inverted) and Poes on y, the loop
            # runs from EELV (low volume, near-baseline Poes → top-right) to EILV (high
            # volume, more negative Poes → bottom-left) — the SAME direction the written
            # Campbell PDF uses (core/plots._pv_average), so the two figures agree and this
            # legend no longer needs to counter a transposition. The WOB read-out above is
            # fixed at the top-left corner (axes fraction, independent of data), which
            # leaves bottom-right as the one corner neither the loop nor the read-out
            # claims — _draw_campbell's legend_kw places the legend there.
            # The legend is deliberately NOT placed here. At the height this panel gets, a
            # three-entry key covers the loops it labels, so whether to draw it at all is a
            # function of the height — which only the fit knows. _draw_campbell hands the
            # placement to _fit_compact_figure as legend_kw, and that sheds it when short.
            pass
        except Exception:                       # pragma: no cover - overlay is best-effort
            pass

"""The stacked, read-only preview of a recording's raw columns.

Two places draw the same thing: the channel-assignment dialog, where every column of the
file is shown so the user can say what each one is, and the Setup screen's channel summary,
where only the columns that ended up with a role are shown. Both want one trace per column,
coloured by the role it carries, stacked on a single shared time axis. They differ only in
which columns appear and in what sits in each row's header — a role dropdown in one, a
description in the other — so that is the one thing the caller supplies.

The role vocabulary lives here too, because the colour of a trace and the name of a role
have to agree between the two views or the summary stops being a readout of the dialog.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

import pyqtgraph as pg

from respmech.ui import plot_perf
from respmech.ui.plot_axis import MinPitchAxis

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

#: every role a channel mapping can carry
ASSIGNABLE = ("flow", "volume", "poes", "pgas", "pdi", "emg", "entropy")
#: role key -> dropdown label. "" is the (unused) sentinel and must stay first.
#: Entropy is deliberately absent: it is the one NON-EXCLUSIVE role — sample entropy may be
#: computed on a column that already carries flow or a pressure, and the shipped example
#: config does exactly that — so it gets its own per-column checkbox instead. A dropdown
#: cannot express "this column is both", and pretending otherwise is what used to delete
#: assignments on OK.
ROLES = [
    ("", "(unused)"),
    ("flow", "Flow"),
    ("volume", "Volume"),
    ("poes", "Poes — oesophageal pressure"),
    ("pgas", "Pgas — gastric pressure"),
    ("pdi", "Pdi — transdiaphragmatic"),
    ("emg", "EMG — diaphragm"),
]
#: role key -> how a summary row names it (a sentence, not a menu entry)
ROLE_NAMES = {
    "flow": "Flow signal",
    "volume": "Volume",
    "poes": "Oesophageal pressure (Poes)",
    "pgas": "Gastric pressure (Pgas)",
    "pdi": "Transdiaphragmatic pressure (Pdi)",
    "emg": "EMG",
    "entropy": "Entropy",
}
#: roles that may be assigned to at most one column (the multi-column roles are emg/entropy)
SINGLE = {"flow", "volume", "poes", "pgas", "pdi"}
#: roles the core analysis always requires (volume is optional — it can be integrated from
#: flow instead; emg/entropy are optional)
REQUIRED = ("flow", "poes", "pgas", "pdi")
REQUIRED_LABELS = {"flow": "Flow", "poes": "Poes", "pgas": "Pgas", "pdi": "Pdi"}

#: height of one column preview, in pixels
ROW_HEIGHT = 74
#: the last row additionally carries the tick values and the "Time (s)" label, which eat into
#: the plot area — without this its y labels are clipped at compact row heights
BOTTOM_AXIS_EXTRA = 28


def plot_palette():
    if _theme is not None:
        try:
            return _theme.plot_palette()
        except Exception:  # pragma: no cover - defensive
            pass
    return {"bg": "#FCFDFE", "fg": (51, 64, 77),
            "channels": {"flow": (44, 110, 155), "volume": (31, 122, 77),
                         "poes": (180, 50, 42), "pgas": (183, 121, 31), "pdi": (125, 91, 166)},
            "emg_cycle": [(44, 110, 155), (180, 50, 42), (31, 122, 77), (183, 121, 31),
                          (125, 91, 166), (14, 124, 123), (180, 80, 122), (92, 107, 122)],
            "separator": (150, 165, 180)}


def as_2d(matrix):
    """A raw matrix as a 2-D ``(samples, columns)`` float array (a lone column -> one col)."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def role_color(pal, role):
    """The trace colour for a role: its channel colour, a distinct hue for EMG/entropy,
    or a muted grey for unused."""
    if role in ("flow", "volume", "poes", "pgas", "pdi"):
        return pal["channels"][role]
    cyc = pal["emg_cycle"]
    if role == "emg":
        return cyc[5 % len(cyc)]
    if role == "entropy":
        return cyc[6 % len(cyc)]
    return pal["separator"]                     # unused -> de-emphasised


def _is_blank_or_placeholder(name):
    """True for an empty header or one of pandas' own artefacts ("__index", "Unnamed: 3")
    — shared by ``name_suffix`` (below) and the name-based role lookup (ticket D27), so the
    two can never disagree about what counts as "no real name"."""
    name = (name or "").strip()
    return not name or name.startswith("__") or name.lower().startswith("unnamed")


def name_suffix(names, i):
    """The source header for column i, shown after the generic index (e.g. ' · flow'),
    or '' when the file had no usable name for it."""
    name = names[i].strip() if i < len(names) else ""
    if _is_blank_or_placeholder(name):
        return ""
    return f"  ·  {name}"


def _looks_numeric(name):
    """True when ``name`` is nothing but a number — the column-index or first-data-row
    names pandas invents for a file with no real header row (ticket D27's own bug
    report: 'fragments of the first data row', not a channel name). Tries both '.' and
    ',' as the decimal point, since a header-less EU-formatted export (';'-separated,
    comma-decimal) produces comma-decimal fragments that plain ``float()`` would not
    recognise as numeric and could otherwise slip through as a "real" name."""
    name = (name or "").strip()
    if not name:
        return False
    for candidate in (name, name.replace(",", ".")):
        try:
            float(candidate)
            return True
        except ValueError:
            continue
    return False


#: role key -> case-insensitive substrings in a column's own header name that suggest it
#: (ticket D27). Every alternative is a recognised physiological abbreviation, not a
#: guess — e.g. "edi" (electrical activity of the diaphragm) for emg, "di" for pdi.
NAME_ROLE_KEYWORDS = {
    "flow": ("flow",),
    "volume": ("volume", "vol"),
    "poes": ("poes", "pes", "oes"),
    "pgas": ("pgas", "pga", "gastric"),
    "pdi": ("pdi", "di"),
    "emg": ("emg", "edi"),
}


def infer_role_from_name(name):
    """The single role a column's own header name suggests, or "" when nothing matches or
    more than one role matches equally well.

    Case-insensitive substring containment against ``NAME_ROLE_KEYWORDS``. A name can
    contain more than one role's keyword — e.g. "edi" contains pdi's short alias "di" as a
    literal substring — so ties are broken by preferring the LONGER keyword match (emg's
    "edi", 3 characters, over pdi's "di", 2): the more specific alias wins outright. Only a
    genuine tie at the longest length (two DIFFERENT roles matched by keywords of the same
    length) is reported as ambiguous — never guessed, per ticket D27."""
    if _is_blank_or_placeholder(name) or _looks_numeric(name):
        return ""
    low = name.strip().lower()
    best_role, best_len = "", 0
    for role, keywords in NAME_ROLE_KEYWORDS.items():
        role_len = max((len(kw) for kw in keywords if kw in low), default=0)
        if role_len == 0:
            continue
        if role_len > best_len:
            best_role, best_len = role, role_len
        elif role_len == best_len and role != best_role:
            best_role = ""                       # a tie between two DIFFERENT roles
    return best_role


def infer_roles_from_names(names):
    """{column index: role} for every column (column 0, the time axis, is never included)
    whose own header name suggests exactly one role — see ``infer_role_from_name``. Used to
    seed the channel-assignment dialog's dropdowns for a brand-new analysis with no saved
    mapping to seed from instead."""
    out = {}
    for i, name in enumerate(names):
        if i == 0:
            continue
        role = infer_role_from_name(name)
        if role:
            out[i] = role
    return out


class ColumnStack(QWidget):
    """One read-only preview per shown column, stacked on a shared time axis.

    ``columns`` are 0-based indices into the matrix, in display order; ``None`` means every
    column. ``header_factory(i, hbox)`` is called once per row with the column index and the
    header's layout, so the caller can add a dropdown, a swatch, whatever it needs — the
    generic "Column N · name" label is added first and exposed as ``headers[row]``.

    The previews are deliberately inert: the y-scale is the information, so it must not be
    pannable, zoomable or resettable by accident.
    """

    def __init__(self, fs, columns=None, header_factory=None, row_height=ROW_HEIGHT,
                 parent=None, sparkline=False):
        super().__init__(parent)
        self._fs = fs or 1.0
        self._columns = columns
        self._header_factory = header_factory
        self._row_height = row_height
        # B05: a caller-facing readout (Setup's channel summary) wants the signal visible
        # but without any axis apparatus — the header text already says the role, the
        # column and its name, so tick labels and a time axis would only repeat what the
        # row already says while eating the vertical space that made the card tall in the
        # first place. The channel-assignment DIALOG keeps the full, editable ColumnStack
        # (the one place a user needs the axis to actually read a value off the trace).
        self._sparkline = bool(sparkline)
        self._prefixes = {}
        self._names = []
        self.pal = plot_palette()
        self.plots, self.curves, self.headers = [], [], []
        self._rows = QVBoxLayout(self)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(10)

    def build(self, matrix, names, roles=None, prefixes=None):
        """Create the rows. Call once; use ``set_data`` afterwards to change file.

        ``prefixes`` maps a column index to text placed BEFORE the column number, so a graph
        says what it is without the reader having to look it up in a legend above."""
        matrix = as_2d(matrix)
        self._prefixes = dict(prefixes or {})
        self._names = list(names)
        cols = list(range(matrix.shape[1])) if self._columns is None else list(self._columns)
        pal = self.pal = plot_palette()
        t = np.arange(matrix.shape[0], dtype=float) / self._fs
        prev = None
        for row, i in enumerate(cols):
            box = QWidget(); cv = QVBoxLayout(box)
            cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(2)
            head = QHBoxLayout(); head.setContentsMargins(2, 0, 2, 0); head.setSpacing(8)
            lab = QLabel(self._header_text(i))
            head.addWidget(lab)
            if self._header_factory is not None:
                self._header_factory(i, head)
            head.addStretch(1)
            cv.addLayout(head)

            last = row == len(cols) - 1
            # MinPitchAxis (ticket B05): these rows are short even in the full (dialog)
            # mode — 74 px — and pyqtgraph draws its top tick level unconditionally, so a
            # left axis without this thins its own labels apart instead of overlapping
            # them. Harmless when the axis is hidden below (sparkline mode never draws it).
            plot = pg.PlotWidget(axisItems={"left": MinPitchAxis(orientation="left")})
            plot.setBackground(pal["bg"])
            if self._sparkline:
                # No axis apparatus at all: the header text already names the role, the
                # column and its source name, so ticks/time-axis would only repeat that
                # while costing the vertical space a compact summary exists to save.
                plot.hideAxis("left")
                plot.hideAxis("bottom")
                plot.setFixedHeight(self._row_height)
            else:
                plot.setFixedHeight(self._row_height + (BOTTOM_AXIS_EXTRA if last else 0))
                if _theme is not None:
                    _theme.align_left_axis(plot)   # stacked column previews share one left margin
                plot.getAxis("bottom").setStyle(showValues=last)
            plot.setMenuEnabled(False)
            plot.getViewBox().setMouseEnabled(x=False, y=False)
            plot.hideButtons()                          # no auto-range 'A' in the corner
            role = "" if roles is None else roles.get(i, "")
            # A saved mapping can name a column this file does not have — a re-export with
            # fewer channels, say. Draw the row blank rather than raising: the row still
            # tells the user which column the setting points at, which is the useful part.
            y = matrix[:, i] if i < matrix.shape[1] else np.full(matrix.shape[0], np.nan)
            curve = plot.plot(t[:len(y)], y, pen=pg.mkPen(role_color(pal, role), width=1),
                              connect="finite")
            if last and not self._sparkline:
                plot.setLabel("bottom", "Time (s)")
            if prev is not None:
                plot.setXLink(prev)
            prev = plot
            cv.addWidget(plot)
            self._rows.addWidget(box)
            self.plots.append(plot); self.curves.append(curve); self.headers.append(lab)
        self._shown = cols
        return self

    def _header_text(self, i):
        prefix = self._prefixes.get(i)
        stem = f"Column {i + 1}" + name_suffix(self._names, i)
        return f"{prefix}  ·  {stem}" if prefix else stem

    def set_data(self, matrix, names):
        """Re-plot from a different file, keeping the rows, roles and colours."""
        matrix = as_2d(matrix)
        self._names = list(names)
        t = np.arange(matrix.shape[0], dtype=float) / self._fs
        for row, i in enumerate(self._shown):
            y = (matrix[:, i] if i < matrix.shape[1]
                 else np.full(matrix.shape[0], np.nan))
            self.curves[row].setData(t[:len(y)], y)
            self.plots[row].enableAutoRange()
            self.headers[row].setText(self._header_text(i))

    def set_role(self, col_index, role):
        """Recolour one column's trace for the role it now carries."""
        row = self._shown.index(col_index)
        self.curves[row].setPen(pg.mkPen(role_color(self.pal, role), width=1))

    def viewports(self):
        return [p.viewport() for p in self.plots]

    def closeEvent(self, ev):
        """Self-cleanup for a ``ColumnStack`` closed WITHOUT a ``MainWindow`` around it
        (point 6 durability). Qt never delivers ``closeEvent`` to a child widget when its
        PARENT window closes — ``MainWindow.closeEvent`` (via ``ChannelSummary.close_plots``)
        and ``ChannelSetupDialog``'s ``finished`` signal both orchestrate ``close_plots()``
        explicitly for that reason, so this override only fires when a ``ColumnStack`` is
        itself the top-level widget being closed: standalone in a test (17 of them, per the
        point 6 investigation), or any future composition without a ``MainWindow`` or
        ``ChannelSetupDialog`` around it. ``close_plots()`` is safe to call twice
        (``self.plots`` is already ``[]`` on a second pass), so a stack that is BOTH
        orchestrated by an owner's cleanup AND later closed directly stays safe."""
        self.close_plots()
        super().closeEvent(ev)

    def close_plots(self) -> None:
        """Release every embedded ``PlotWidget``'s own context menus (``plot_perf``'s
        documented ``PlotItem``/``ViewBox`` cleanup), then drop them from ``self.plots``.

        Each ``PlotWidget()`` built in :meth:`build` constructs its own ``ctrlMenu`` (one
        ``QMenu`` plus six submenus) EAGERLY at construction, whether or not it is ever
        shown — ``setMenuEnabled(False)`` above only drops the ``ViewBoxMenu``, not this.
        Unlike Preview & QC's stacked plots (``plot_perf.close_plots``'s own docstring),
        NOTHING ever closed these: a ``ColumnStack`` instance is discarded either by
        ``ChannelSummary`` rebuilding its reading (a fresh mapping, a new file) or by the
        window that owns it going away — and in both cases the discarded instance stays
        fully REACHABLE (via ``ChannelSummary.stack`` until overwritten, or via the
        channel-assignment dialog's own ``self._stack`` for as long as the dialog exists),
        so it is never garbage a collector could reclaim on its own; only an explicit
        ``.close()`` releases the menus. Call this whenever a ``ColumnStack`` is about to
        be discarded, never mid-use — each ``PlotWidget`` is unusable afterwards.
        """
        for p in self.plots:
            plot_perf.close_plots(p)
        self.plots = []

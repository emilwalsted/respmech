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


def name_suffix(names, i):
    """The source header for column i, shown after the generic index (e.g. ' · flow'),
    or '' when the file had no usable name for it."""
    name = names[i].strip() if i < len(names) else ""
    if not name or name.startswith("__") or name.lower().startswith("unnamed"):
        return ""
    return f"  ·  {name}"


class ColumnStack(QWidget):
    """One read-only preview per shown column, stacked on a shared time axis.

    ``columns`` are 0-based indices into the matrix, in display order; ``None`` means every
    column. ``header_factory(i, hbox)`` is called once per row with the column index and the
    header's layout, so the caller can add a dropdown, a swatch, whatever it needs — the
    generic "Column N · name" label is added first and exposed as ``headers[row]``.

    The previews are deliberately inert: the y-scale is the information, so it must not be
    pannable, zoomable or resettable by accident.
    """

    def __init__(self, fs, columns=None, header_factory=None, parent=None):
        super().__init__(parent)
        self._fs = fs or 1.0
        self._columns = columns
        self._header_factory = header_factory
        self._names = []
        self.pal = plot_palette()
        self.plots, self.curves, self.headers = [], [], []
        self._rows = QVBoxLayout(self)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(10)

    def build(self, matrix, names, roles=None):
        """Create the rows. Call once; use ``set_data`` afterwards to change file."""
        matrix = as_2d(matrix)
        self._names = list(names)
        cols = list(range(matrix.shape[1])) if self._columns is None else list(self._columns)
        pal = self.pal = plot_palette()
        t = np.arange(matrix.shape[0], dtype=float) / self._fs
        prev = None
        for row, i in enumerate(cols):
            box = QWidget(); cv = QVBoxLayout(box)
            cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(2)
            head = QHBoxLayout(); head.setContentsMargins(2, 0, 2, 0); head.setSpacing(8)
            lab = QLabel(f"Column {i + 1}" + name_suffix(self._names, i))
            head.addWidget(lab)
            if self._header_factory is not None:
                self._header_factory(i, head)
            head.addStretch(1)
            cv.addLayout(head)

            last = row == len(cols) - 1
            plot = pg.PlotWidget()
            plot.setBackground(pal["bg"])
            plot.setFixedHeight(ROW_HEIGHT)
            plot.setMenuEnabled(False)
            plot.getViewBox().setMouseEnabled(x=False, y=False)
            plot.hideButtons()                          # no auto-range 'A' in the corner
            if _theme is not None:
                _theme.align_left_axis(plot)       # stacked column previews share one left margin
            plot.getAxis("bottom").setStyle(showValues=last)
            role = "" if roles is None else roles.get(i, "")
            y = matrix[:, i]
            curve = plot.plot(t[:len(y)], y, pen=pg.mkPen(role_color(pal, role), width=1),
                              connect="finite")
            if last:
                plot.setLabel("bottom", "Time (s)")
            if prev is not None:
                plot.setXLink(prev)
            prev = plot
            cv.addWidget(plot)
            self._rows.addWidget(box)
            self.plots.append(plot); self.curves.append(curve); self.headers.append(lab)
        self._shown = cols
        return self

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
            self.headers[row].setText(f"Column {i + 1}" + name_suffix(self._names, i))

    def set_role(self, col_index, role):
        """Recolour one column's trace for the role it now carries."""
        row = self._shown.index(col_index)
        self.curves[row].setPen(pg.mkPen(role_color(self.pal, role), width=1))

    def viewports(self):
        return [p.viewport() for p in self.plots]

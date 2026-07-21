"""Read-only readout of the assigned channels.

The Setup screen used to hold seven editable fields for the channel columns, which meant two
places could set them and the numbers had to be typed against a column count the user could
not see. Assignment now happens only in the channel dialog, where each column is plotted, so
Setup shows what was chosen rather than asking for it: one row per assigned role, and the
same stacked preview the dialog draws.

Only roles that HAVE a column appear — an empty summary is the honest rendering of "nothing
assigned yet", and is what the empty-state line says.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from respmech.ui.column_stack import ColumnStack, ROLE_NAMES, role_color
from respmech.ui.help_text import tooltip as _tip

#: role -> the settings path and description its row carries, so hovering the summary still
#: names the variable the way the deleted fields did
ROLE_HELP = {
    "flow": ("input.channels.flow",
             "Column holding the airflow signal, which reads negative during inspiration."),
    "volume": ("input.channels.volume",
               "Column holding the volume signal; or tick 'Calculate volume from flow' in "
               "Mechanics to derive it from flow instead."),
    "poes": ("input.channels.poes",
             "Column holding the oesophageal pressure signal, in cmH2O."),
    "pgas": ("input.channels.pgas",
             "Column holding the gastric pressure signal, in cmH2O."),
    "pdi": ("input.channels.pdi",
            "Column holding the transdiaphragmatic pressure signal, in cmH2O."),
    "emg": ("input.channels.emg",
            "Columns of the diaphragm EMG channels to analyse."),
    "entropy": ("input.channels.entropy",
                "Columns used for sample-entropy analysis. Entropy is not exclusive — a "
                "column may also carry another role."),
}
#: the order the rows read in, which is the order the analysis uses them
ORDER = ("flow", "volume", "poes", "pgas", "pdi", "emg", "entropy")

EMPTY_TEXT = "No channels assigned yet — click ‘Assign channels from data…’."

#: the previews here are a readout, not a working surface, so they are shorter than the
#: dialog's. Measured floor: below this the y tick labels of the wider-range channels clip
#: (48 loses Poes's end labels, 50 still clips Volume's top), so this is as compact as the
#: axis text allows rather than an aesthetic pick.
SUMMARY_ROW_HEIGHT = 56


def roles_of(channels, col):
    """Every role a 1-based column carries, in analysis order. Usually one, but entropy is
    non-exclusive, so a column can be both."""
    out = []
    for role in ORDER:
        value = getattr(channels, role, None)
        if role in ("emg", "entropy"):
            if col in {int(c) for c in (value or [])}:
                out.append(role)
        elif value and int(value) == col:
            out.append(role)
    return out


def _swatch(pal, role):
    dot = QLabel("●")
    r, g, b = role_color(pal, role)
    dot.setStyleSheet(f"color: rgb({r},{g},{b});")
    return dot


def describe(channels, role, integrate_from_flow=False):
    """The one-line readout for a role, or None when it has no column.

    Volume is the exception worth spelling out: with 'Calculate volume from flow' on, the
    column is ignored, and silently showing a column number that nothing reads is exactly
    the kind of quiet wrongness the summary exists to prevent."""
    if role == "volume" and integrate_from_flow:
        return "Volume: derived from flow"
    value = getattr(channels, role, None)
    if role in ("emg", "entropy"):
        cols = list(value or [])
        if not cols:
            return None
        return (f"{ROLE_NAMES[role]}: Column{'s' if len(cols) != 1 else ''} "
                + ", ".join(f"#{c}" for c in cols))
    if not value:
        return None
    return f"{ROLE_NAMES[role]}: Column #{value}"


def assigned_columns(channels):
    """Every column carrying any role, ascending and without repeats — a column can hold
    both a role and entropy, and must still be drawn once."""
    cols = set()
    for role in ORDER:
        value = getattr(channels, role, None)
        if role in ("emg", "entropy"):
            cols |= {int(c) for c in (value or [])}
        elif value:
            cols.add(int(value))
    return sorted(cols)


class ChannelSummary(QWidget):
    """One row per assigned role, over the preview of the columns they name."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(6)
        self.rows = []
        self.stack = None
        self._empty = QLabel(EMPTY_TEXT)
        self._empty.setProperty("status", "muted")
        self._empty.setWordWrap(True)
        self._box.addWidget(self._empty)

    def _clear(self):
        while self._box.count():
            item = self._box.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._empty:
                w.setParent(None)
        self.rows = []
        self.stack = None

    def show_mapping(self, channels, *, matrix=None, names=None, fs=1000,
                     integrate_from_flow=False):
        """Rebuild the readout. ``matrix``/``names`` are optional: with no readable file yet
        the rows still say which column each role points at, just without the traces."""
        self._clear()
        cols = assigned_columns(channels)
        if not cols:
            self._empty.setText(EMPTY_TEXT)
            self._box.addWidget(self._empty)
            self._empty.show()
            return self
        self._empty.hide()

        pal = ColumnStack(fs).pal
        for role in ORDER:
            text = describe(channels, role, integrate_from_flow)
            if text is None:
                continue
            row = QWidget(); rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
            rl.addWidget(_swatch(pal, role))
            lab = QLabel(text)
            lab.setToolTip(_tip(*ROLE_HELP[role]))
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            rl.addWidget(lab); rl.addStretch(1)
            self._box.addWidget(row)
            self.rows.append(lab)

        if matrix is not None:
            roles, prefixes = {}, {}
            for col in cols:
                carried = roles_of(channels, col)
                roles[col - 1] = carried[0] if carried else ""
                # name the graph by what it IS, ahead of where it sits, so a reader does not
                # have to carry the legend above in their head
                prefixes[col - 1] = " + ".join(ROLE_NAMES[r] for r in carried)
            shown = [c - 1 for c in cols]
            self.stack = ColumnStack(fs, columns=shown, row_height=SUMMARY_ROW_HEIGHT)
            self.stack.build(matrix, names or [], roles=roles, prefixes=prefixes)
            self._box.addWidget(self.stack)
        return self

    def texts(self):
        return [r.text() for r in self.rows]

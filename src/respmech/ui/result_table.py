"""The one DataFrame-backed result table, shared by Preview's per-breath table and
Run's averaged-metrics table (ticket A04, UI-overhaul).

WHY. Both screens used to build their own grid by hand: ``setHorizontalHeaderLabels``
+ a nested loop of ``QTableWidgetItem(str(value))``, with no formatting, no
alignment, no units and no column-width discipline. On the sample analysis that
produced 4800 px of columns in a 1222 px viewport, headers clipped into
indistinguishable siblings (``poes_tidal_swing`` -> ``oes_tidal_swin``, and the same
clip happens to ``pgas_``/``pdi_`` too), and cells showed 18-digit floats that elide
differently row to row, so two values in the same column cannot be compared. Nobody
could tell a value's unit without opening the written Excel file.

``ResultTableModel`` is a single ``QAbstractTableModel`` over the DataFrame instead:
it never builds 78 x N item objects, and it centralises formatting, alignment,
units and tooltips in one ``data()``/``headerData()`` pair so both screens see the
same behaviour. Column headers carry the engine's exact identifier on the first
line and the quantities-registry unit on the second (blank when the unit is
unknown) — the identifier never moves, because it is what is traceable to the
written spreadsheet and to a user's own scripts. Numeric cells render to 6
significant figures and right-align; the full-precision value survives in the
cell's tooltip and under ``Qt.ItemDataRole.UserRole`` so a copy never loses a
digit. ``configure_result_table()`` wires the common, non-editable, copyable
behaviour (edit triggers, selection mode, header height, column-width cap) that
both screens want identically.

A ``\\n`` in a header's DisplayRole does not by itself make Qt's QHeaderView any
taller (confirmed against this app's own offscreen Qt: the text DOES paint on two
lines once the header has the room, the header just never grows itself for it) —
so ``configure_result_table()`` sets the header height explicitly, from THIS
view's own font metrics rather than a pixel literal (macOS and Windows font
metrics differ enough that a literal is wrong on one platform or the other; see
CLAUDE.md's layout-test notes).
"""
from __future__ import annotations

import numbers

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QAbstractItemView, QTableView

from respmech.core import quantities as _quantities

# A single very wide column (a long file path, say) must not eat the whole viewport —
# measured 4800 px of 100-px-each columns on the sample analysis (48 columns) before
# this ticket; this caps any one column at a still-generous width and lets the reader
# widen it by hand if they want more.
_MAX_SECTION_PX = 220


def _is_plain_number(v) -> bool:
    """True for a real, non-boolean number. A literal Python ``bool`` IS a
    ``numbers.Real`` subclass, so it is excluded explicitly — but a pandas boolean
    COLUMN hands back ``numpy.bool_`` here, which registers as neither
    ``numbers.Real`` nor ``bool`` and is excluded anyway (falls through to the
    string branch below). The explicit check exists for the literal-``bool`` case
    (a column holding real Python booleans, e.g. via ``object`` dtype), which
    ``numpy.bool_`` would not otherwise be caught by."""
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _format_display(v) -> str:
    """6-significant-figure rendering for numbers (NOT .4g: that rounds 169.96689 to
    170 and 9.108336 to 9.108, dropping digits a researcher may be citing), plain
    ``str()`` for everything else (the shared code also renders the ``file`` column,
    which is strings). ``None`` renders blank rather than the literal word "None" —
    a missing value should look empty, not like leaked Python internals. A NaN
    stays visible as "nan": the core deliberately WRITES NaN into some EMG columns
    when a detector is unreliable (rather than an inflated number), so hiding it
    would hide a real, meaningful result."""
    if v is None:
        return ""
    if _is_plain_number(v):
        try:
            return f"{v:.6g}"
        except (TypeError, ValueError):          # pragma: no cover - defensive
            return str(v)
    return str(v)


class ResultTableModel(QAbstractTableModel):
    """Read-only table model over a pandas DataFrame."""

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = None
        self._columns: list = []
        self.set_dataframe(df)

    def set_dataframe(self, df) -> None:
        """Replace the underlying DataFrame. ``None`` (or an empty-columns frame)
        clears the table, matching the old ``setRowCount(0); setColumnCount(0)``."""
        self.beginResetModel()
        if df is None or len(df.columns) == 0:
            self._df = None
            self._columns = []
        else:
            self._df = df.reset_index(drop=True)
            self._columns = list(self._df.columns)
        self.endResetModel()

    # -- Qt.QAbstractTableModel -------------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        if parent.isValid() or self._df is None:
            return 0
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._df is None:
            return None
        raw = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return _format_display(raw)
        if role == Qt.ItemDataRole.ToolTipRole:
            # full precision — the truncated DisplayRole is a screen convenience,
            # never the only place the real value is readable. None matches
            # DisplayRole's blank rendering rather than the literal word "None".
            return "" if raw is None else str(raw)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if _is_plain_number(raw):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.UserRole:
            # the untouched cell value — what a copy is built from, so a copy never
            # loses precision the 6-sig-fig DisplayRole dropped
            return raw
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal or not (0 <= section < len(self._columns)):
            return super().headerData(section, orientation, role)
        col = str(self._columns[section])
        unit = _quantities.unit_for(col)
        if role == Qt.ItemDataRole.DisplayRole:
            # identifier on line 1, unit on line 2 — ALWAYS two lines (a blank
            # second line where the unit is unknown) so every section gets the
            # same header height.
            return f"{col}\n{unit}"
        if role == Qt.ItemDataRole.ToolTipRole:
            # both parts, in full, even where the header text itself would elide
            return f"{col} ({unit})" if unit else col
        return super().headerData(section, orientation, role)

    def column_id(self, index: int) -> str:
        """The engine's exact column identifier at ``index`` (no unit), or "" out
        of range. Used by callers that need the raw name, e.g. for export."""
        return str(self._columns[index]) if 0 <= index < len(self._columns) else ""


def configure_result_table(view: QTableView) -> None:
    """Wire the behaviour both result tables share: read-only, one wide column
    capped, header sized for two lines, and a table-scoped Ctrl/Cmd+C. Call once
    right after constructing the view; refilling the model afterwards
    (``set_dataframe()``) does not need to repeat this."""
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
    header = view.horizontalHeader()
    header.setMaximumSectionSize(_MAX_SECTION_PX)
    # two-line header height, computed from THIS view's own font metrics — never a
    # pixel literal (see the module docstring)
    fm = header.fontMetrics()
    header.setFixedHeight(2 * fm.height() + 12)
    install_copy_shortcut(view)


def resize_result_table(view: QTableView) -> None:
    """Re-fit column widths to the current contents. Call after every
    ``set_dataframe()`` — mirrors the old ``_fill_table``'s implicit resize, now
    explicit because the model itself doesn't know when a view is done consuming it."""
    view.resizeColumnsToContents()


def install_copy_shortcut(view: QTableView) -> QShortcut:
    """Ctrl/Cmd+C copies the selected rectangular range to the clipboard as TSV, so
    it pastes straight into a spreadsheet. Scoped to the table itself
    (``WidgetWithChildrenShortcut``) rather than hung off a future menu action: a
    menu ``QAction`` does not by itself give a table Ctrl+C, and the copy slot has
    to exist regardless of when/whether a menu bar calls it too."""
    sc = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), view,
                   context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
    sc.activated.connect(lambda: _copy_selection_as_tsv(view))
    return sc


def _copy_selection_as_tsv(view: QTableView) -> None:
    model = view.model()
    sel = view.selectionModel()
    if model is None or sel is None:
        return
    indexes = sel.selectedIndexes()
    if not indexes:
        return
    selected = {(i.row(), i.column()) for i in indexes}
    rows = sorted({r for r, _ in selected})
    cols = sorted({c for _, c in selected})
    # a header line is only meaningful when every row of the involved columns is
    # selected (selectedColumns() is exactly that test) — a partial rectangle in
    # the middle of the table should not paste a header row into the middle of a
    # spreadsheet
    include_header = bool(sel.selectedColumns())
    lines = []
    if include_header:
        header_cells = []
        for c in range(cols[0], cols[-1] + 1):
            name = model.headerData(c, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole)
            # the model's tooltip is already a single line ("id (unit)"); a TSV
            # header cell must stay on one line or it breaks the row structure
            header_cells.append(str(name).replace("\t", " ").strip())
        lines.append("\t".join(header_cells))
    for r in range(rows[0], rows[-1] + 1):
        cells = []
        for c in range(cols[0], cols[-1] + 1):
            # a Ctrl-click selection can be non-rectangular (e.g. two disjoint
            # cells, or two whole columns with an unselected one between them);
            # walking the bounding box of rows[0]..rows[-1] x cols[0]..cols[-1]
            # would then silently paste cells the user never selected. Leave any
            # cell that isn't ACTUALLY selected blank rather than fill it in.
            if (r, c) not in selected:
                cells.append("")
                continue
            v = model.index(r, c).data(Qt.ItemDataRole.UserRole)
            cells.append(str(v) if v is not None else "")
        lines.append("\t".join(cells))
    QGuiApplication.clipboard().setText("\n".join(lines))

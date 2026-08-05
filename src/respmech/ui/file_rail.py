"""The file rail: one row per file in the batch, with its own state — replacing
Preview & QC's plain ``file_combo`` and Run & results' separate ``files_table``
(ticket B02, UI-overhaul).

WHY. The old file selector was a non-editable, unsearchable ``QComboBox`` that knew
nothing about the batch: not which files had already been previewed, not which had
failed, not which carried a manual breath exclusion, and not which the manifest (B01)
had already flagged as a column-count or sampling-frequency outlier. Reaching the one
failed file in a 25-file batch meant scrolling a table past every success first. This
module gives every screen that shows "the files in this batch" ONE row-per-file widget,
built on top of :mod:`respmech.ui.manifest`'s ``Manifest``, that carries verdict,
breath count, exclusion count and manifest caveats together and can filter/re-order by
them.

``FileRailModel`` is Qt-necessary (``QAbstractListModel``) but holds no file-system or
worker state of its own — everything it knows is handed to it by ``set_manifest()`` and
the ``mark_*`` methods, exactly the split ``ResultTableModel`` (``ui/result_table.py``)
already uses for the per-breath/averages tables.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QLineEdit, QListView, QVBoxLayout, QWidget)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

#: extra Qt.UserRole roles FileRailModel exposes, beyond the built-in Display/ToolTip
NameRole = Qt.UserRole + 1
EntryRole = Qt.UserRole + 2
FailedRole = Qt.UserRole + 3   # sort key: 0 for a failed row, 1 for everything else


@dataclass
class FileRailEntry:
    """One row's full state: ``caveat`` comes from the manifest (an outlier's reason, or
    a detected-vs-configured sampling-frequency mismatch) and is refreshed on every
    :meth:`FileRailModel.set_manifest`; everything else is state the rail itself learns
    across previews/runs in THIS session and is preserved across a manifest rebuild for
    any filename that persists (a Setup edit that merely re-scans the same folder must
    not forget what has already been previewed)."""

    filename: str
    caveat: str | None = None
    seen: bool = False            # previewed/run at least once this session
    verdict: str = "unknown"      # "unknown" | "ok" | "failed"
    breaths: int | None = None
    error: str | None = None
    excluded_count: int = 0


def _row_text(e: FileRailEntry) -> str:
    glyph = {"ok": "✓", "failed": "✗", "unknown": "•"}[e.verdict]
    bits = [f"{glyph}  {e.filename}"]
    if e.excluded_count:
        n = e.excluded_count
        bits.append(f"[{n} excl]")
    if e.caveat:
        bits.append("⚠")
    return "   ".join(bits)


def _row_tooltip(e: FileRailEntry) -> str:
    lines = [e.filename]
    if e.verdict == "ok":
        n = e.breaths
        lines.append(f"{n} breath{'s' if n != 1 else ''}" if n is not None else "OK")
    elif e.verdict == "failed":
        lines.append(e.error or "Failed")
    else:
        lines.append("Not previewed or run yet this session.")
    if e.excluded_count:
        n = e.excluded_count
        lines.append(f"{n} breath{'s' if n != 1 else ''} manually excluded")
    if e.caveat:
        lines.append(f"⚠ {e.caveat}")
    if not e.seen:
        lines.append("(not seen this session)")
    return "\n".join(lines)


def _fg_colour(e: FileRailEntry):
    if _theme is None:
        return None
    t = _theme.active_theme()
    if not e.seen:
        return QColor(t["disabled_fg"])
    if e.verdict == "failed":
        return QColor(t["st_error_fg"])
    if e.caveat:
        return QColor(t["st_warn_fg"])
    if e.verdict == "ok":
        return QColor(t["st_ok_fg"])
    return None


class FileRailModel(QAbstractListModel):
    """One row per :class:`respmech.ui.manifest.Manifest` file — BOTH included files and
    outliers, since an outlier is still a file a user may want to open and inspect, just
    one the manifest has already flagged. Qt-necessary (unlike ``manifest.py`` itself),
    but every fact it renders is handed in, never computed here."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[FileRailEntry] = []
        self._by_name: dict[str, int] = {}

    # -- population -----------------------------------------------------
    def set_manifest(self, manifest) -> None:
        """Rebuild rows from ``manifest.files`` (``None`` clears the rail). Per-file
        result/seen/exclusion state is preserved for any filename that persists across
        the rebuild — only ``caveat`` is always refreshed from the new manifest."""
        old = {e.filename: e for e in self._entries}
        freq_mismatch_names = ({f.filename for f in manifest.freq_mismatches}
                               if manifest is not None else set())
        new_entries: list[FileRailEntry] = []
        for f in (manifest.files if manifest is not None else ()):
            if not f.included:
                caveat = f.exclude_reason
            elif f.filename in freq_mismatch_names:
                caveat = (f"detected {f.detected_fs} Hz sampling — settings say "
                         f"{manifest.settings_fs} Hz")
            else:
                caveat = None
            prev = old.get(f.filename)
            if prev is not None:
                new_entries.append(FileRailEntry(
                    filename=f.filename, caveat=caveat, seen=prev.seen, verdict=prev.verdict,
                    breaths=prev.breaths, error=prev.error, excluded_count=prev.excluded_count))
            else:
                new_entries.append(FileRailEntry(filename=f.filename, caveat=caveat))
        self.beginResetModel()
        self._entries = new_entries
        self._by_name = {e.filename: i for i, e in enumerate(self._entries)}
        self.endResetModel()

    def entry(self, filename: str) -> FileRailEntry | None:
        i = self._by_name.get(filename)
        return self._entries[i] if i is not None else None

    def filenames(self) -> list[str]:
        return [e.filename for e in self._entries]

    def any_failed(self) -> bool:
        return any(e.verdict == "failed" for e in self._entries)

    # -- per-file state, updated in place --------------------------------
    def mark_seen(self, filename: str) -> None:
        i = self._by_name.get(filename)
        if i is None or self._entries[i].seen:
            return
        self._entries[i].seen = True
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def mark_result(self, filename: str, *, ok: bool, breaths=None, error=None) -> None:
        """Record the verdict of the latest dry run / batch run FOR THIS FILE. A file
        that is not currently a row (filtered out of an old manifest, say) is silently
        ignored — the caller does not have to check membership first."""
        i = self._by_name.get(filename)
        if i is None:
            return
        e = self._entries[i]
        e.verdict = "ok" if ok else "failed"
        e.breaths = breaths if ok else None
        e.error = None if ok else error
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def set_excluded_count(self, filename: str, count: int) -> None:
        i = self._by_name.get(filename)
        if i is None or self._entries[i].excluded_count == count:
            return
        self._entries[i].excluded_count = count
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    # -- QAbstractListModel -----------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._entries)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None
        e = self._entries[index.row()]
        if role == Qt.DisplayRole:
            return _row_text(e)
        if role == NameRole:
            return e.filename
        if role == Qt.ToolTipRole:
            return _row_tooltip(e)
        if role == EntryRole:
            return e
        if role == FailedRole:
            return 0 if e.verdict == "failed" else 1
        if role == Qt.ForegroundRole:
            return _fg_colour(e)
        if role == Qt.FontRole and not e.seen:
            f = QFont()
            f.setItalic(True)
            return f
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable


class _FileRailProxy(QSortFilterProxyModel):
    """Contains-match filename filter (case-insensitive, plain substring — no regex
    surprises from a filename containing ``.`` or ``(``) plus an optional 'failed rows
    first' ordering that is independent of the filter and off by default (manifest
    order)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_text = ""
        self._failed_first = False

    def set_filter_text(self, text: str) -> None:
        self._filter_text = (text or "").strip().lower()
        # invalidateFilter()/invalidateRowsFilter() are both deprecated as of this Qt
        # version; plain invalidate() (filter + sort) is the warning-free replacement.
        self.invalidate()

    def set_failed_first(self, on: bool) -> None:
        if on == self._failed_first:
            return
        self._failed_first = on
        if on:
            self.sort(0, Qt.AscendingOrder)
        else:
            self.sort(-1)   # back to the source model's own (manifest) order

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._filter_text:
            return True
        name = self.sourceModel().index(source_row, 0, source_parent).data(NameRole) or ""
        return self._filter_text in name.lower()

    def lessThan(self, left, right):
        if self._failed_first:
            lf, rf = left.data(FailedRole), right.data(FailedRole)
            if lf != rf:
                return lf < rf
        return left.row() < right.row()


class FileRail(QWidget):
    """The filterable, stateful file list itself: a filter field over a
    :class:`FileRailModel`/``QListView`` pair.

    ``selectionChanged(str)`` fires whenever the CURRENT file identity changes, by
    whatever means — an explicit row click, :meth:`step` (◀/▶, PageUp/PageDown), or a
    programmatic :meth:`select_filename` — mirroring ``QComboBox.currentTextChanged``,
    which the file-switch logic this replaces already gates on ("did the file actually
    change") rather than caring how the change happened. It NEVER fires from typing in
    the filter field: filtering only changes what is VISIBLE, never the current
    selection — a partially typed name must not be able to reach the file switch."""

    selectionChanged = Signal(str)
    #: a row was explicitly double-activated (double-click / Enter) — distinct from a
    #: plain single-click selection, for a caller that wants to "open" rather than just
    #: preview-select (Run & results drilling back into Preview & QC).
    fileActivated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = FileRailModel(self)
        self._proxy = _FileRailProxy(self)
        self._proxy.setSourceModel(self._model)
        self._current: str | None = None
        self._suppress = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter files…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._proxy.set_filter_text)
        root.addWidget(self.filter_edit)

        self.view = QListView()
        self.view.setModel(self._proxy)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.view.setUniformItemSizes(True)
        self.view.setAlternatingRowColors(True)
        self.view.setToolTip("One row per file. Click to preview; ◀/▶ or "
                             "PageUp/PageDown also move the selection.")
        self.view.selectionModel().currentChanged.connect(self._on_view_current_changed)
        self.view.doubleClicked.connect(self._on_double_clicked)
        root.addWidget(self.view, 1)

    # -- manifest / rows --------------------------------------------------
    def set_manifest(self, manifest) -> None:
        """Rebuild the rows. When the rail currently has NO identity at all (never
        selected anything — a just-constructed screen's very first call), the first
        file is adopted QUIETLY: ``current_filename()`` reflects it immediately, but
        :attr:`selectionChanged` does NOT fire. This mirrors the old ``file_combo``,
        which auto-selected index 0 on first populate as a plain Qt/QComboBox side
        effect — never through ``currentTextChanged`` (its own populate ran under
        ``blockSignals``) — so nothing downstream ever reacted to that specific
        moment. A caller must keep matching that: a freshly built screen should show
        its first file as selected without ALSO silently kicking off background
        analysis before the user has done anything."""
        self._model.set_manifest(manifest)
        if self._current is None:
            names = self._model.filenames()
            if names:
                self._current = names[0]
        self._sync_view_selection()

    def filenames(self) -> list[str]:
        return self._model.filenames()

    def any_failed(self) -> bool:
        """Whether ANY row currently in the rail is marked failed — not just the rows a
        particular run just touched. A caller deciding whether to sort failed-first must
        judge from the rail's whole current state, or a subset re-run that fixes one
        file can silently un-sort another, untouched, still-failed file."""
        return self._model.any_failed()

    def visible_filenames(self) -> list[str]:
        """Filenames in the CURRENTLY DISPLAYED order — after the filter text and any
        failed-first sort are applied. Distinct from :meth:`filenames`, which is always
        the unfiltered manifest order."""
        return [self._proxy.index(r, 0).data(NameRole) for r in range(self._proxy.rowCount())]

    def count(self) -> int:
        return len(self._model.filenames())

    def entry(self, filename: str) -> FileRailEntry | None:
        return self._model.entry(filename)

    # -- identity -----------------------------------------------------------
    def current_filename(self) -> str | None:
        return self._current

    def select_filename(self, name: str | None) -> None:
        """Set the current file identity. Clears the filter first if the target isn't
        currently visible under it (a stale filter must never hide the very file a
        caller just asked to jump to — see MainWindow's Run-to-Preview drill-back and
        Preview's 'Process & write this file'). Emits :attr:`selectionChanged` exactly
        when the identity actually changes, matching ``QComboBox.setCurrentText``'s own
        currentTextChanged behaviour, which callers already gate their own no-op checks
        on (``name != self._previewed_file``)."""
        if name and self._find_proxy_row(name) is None and self.filter_edit.text():
            self.filter_edit.setText("")
        changed = name != self._current
        self._current = name
        self._sync_view_selection()
        if changed and name:
            self.mark_seen(name)
            self.selectionChanged.emit(name)

    def select_index(self, i: int) -> None:
        """Select the i-th row in manifest order (unfiltered) — a thin convenience over
        :meth:`select_filename` for a caller that already knows a fixed file order."""
        names = self.filenames()
        if 0 <= i < len(names):
            self.select_filename(names[i])

    def step(self, delta: int) -> None:
        """Move the current selection by ``delta`` among the VISIBLE (filtered) rows,
        clamped to the ends — the ◀/▶ buttons and PageUp/PageDown."""
        n = self._proxy.rowCount()
        if n <= 1:
            return
        cur = self._find_proxy_row(self._current)
        i = 0 if cur is None else cur
        i = min(max(i + delta, 0), n - 1)
        self.view.setCurrentIndex(self._proxy.index(i, 0))

    # -- per-file state, forwarded to the model ----------------------------
    def mark_seen(self, filename: str) -> None:
        self._model.mark_seen(filename)

    def mark_result(self, filename: str, *, ok: bool, breaths=None, error=None) -> None:
        self._model.mark_result(filename, ok=ok, breaths=breaths, error=error)

    def set_excluded_count(self, filename: str, count: int) -> None:
        self._model.set_excluded_count(filename, count)

    def sort_failed_first(self, on: bool) -> None:
        self._proxy.set_failed_first(on)
        self._sync_view_selection()

    # -- internals ------------------------------------------------------
    def _find_proxy_row(self, name):
        if not name:
            return None
        for r in range(self._proxy.rowCount()):
            if self._proxy.index(r, 0).data(NameRole) == name:
                return r
        return None

    def _sync_view_selection(self):
        """Re-assert the view's visual selection from ``self._current`` WITHOUT
        emitting — used after a manifest rebuild or a sort-order change, both of which
        can move or hide the current row without the current FILE changing."""
        r = self._find_proxy_row(self._current)
        self._suppress = True
        try:
            sel = self.view.selectionModel()
            if r is None:
                sel.clearCurrentIndex()
            else:
                self.view.setCurrentIndex(self._proxy.index(r, 0))
        finally:
            self._suppress = False

    def _on_view_current_changed(self, current, _previous):
        if self._suppress or not current.isValid():
            return
        name = current.data(NameRole)
        if name and name != self._current:
            self._current = name
            self.mark_seen(name)
            self.selectionChanged.emit(name)

    def _on_double_clicked(self, index):
        name = index.data(NameRole)
        if name:
            self.fileActivated.emit(name)

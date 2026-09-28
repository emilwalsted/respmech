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

from dataclasses import dataclass, field

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QLineEdit, QListView, QMenu,
                               QStyle, QStyledItemDelegate, QStyleOptionViewItem,
                               QVBoxLayout, QWidget)

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
    # this file's exclusion entry was recorded against a DIFFERENT (or unrecorded)
    # recordings folder than the one currently loaded — see
    # core.settings.carried_over_state. Meaningless (and always False) when
    # excluded_count is 0: there is nothing to have carried over.
    excluded_carried: bool = False
    # M-32: kind -> count of this file's typed manoeuvre breaths (core.settings.
    # BreathTypeEntry), e.g. {"ic": 2, "fvc": 1}. Empty dict (never None) when the file
    # has no typed breaths at all, so `if e.typed_counts:` is the one truthiness check
    # every caller needs.
    typed_counts: dict = field(default_factory=dict)
    # same carried-over-folder provenance as excluded_carried, but for breath_types:
    # true if ANY of this file's typed breaths carries a stale folder tag. Meaningless
    # (and always False) when typed_counts is empty.
    typed_carried: bool = False
    # 'self' | 'linked' | 'missing' | None — whether a typed manoeuvre's reference (its
    # baseline EELV, say) resolves within this file, is linked from another file, or is
    # unresolved. Resolution itself is M-34's scope; M-32 only owns the field and its
    # rendering, so this stays None until a later ticket starts calling set_reference().
    reference: str | None = None
    # number of EMG segments this file will produce under the CURRENT segmentation
    # settings — meaningful (an int) only for the EMG-only 'whole_file'/'separators'
    # methods, which are pure functions of settings alone (no signal data needed: N
    # separator times always make N+1 segments). None for 'flow'/'volume' segmentation
    # (breath-based, not segment-based) or when nothing has computed it yet.
    segments: int | None = None
    # 'tidal' | 'reference' | None — this file's core.pipeline.FileResult.role from the
    # most recent successful preview/run (M-30: a file with typed manoeuvres but no
    # tidal breaths at all runs as 'reference'-only). None until mark_result(ok=True)
    # has actually reported one, exactly like `breaths`.
    role: str | None = None
    # This file's processing.segmentation.overrides entry's cut_s + join_s
    # count combined (a manual repair of the automatic flow-/volume-based
    # segmentation) — 0 when the file has no entry, or an entry with both lists
    # empty, same "nothing worth reporting" guard as core.settings' own
    # _CARRIED_KINDS row for this field.
    overrides_count: int = 0


def _row_prefix(e: FileRailEntry) -> str:
    """The leading, never-elided glyph cluster: the plain ok/failed/unknown verdict,
    plus single-character badges for state a long filename must never be allowed to
    push off the visible rail — ``◆`` for a file carrying at least one typed manoeuvre
    breath, ``⇢``/``⇢?`` for a resolved/unresolved cross-file reference (M-34/M-37 own
    setting ``reference``; M-32 only owns rendering it). Kept a two-space-padded STRING
    (not a list the caller joins) so ``_row_text``/the eliding delegate agree byte-for-
    byte on what counts as "the fixed part" of a row."""
    glyph = {"ok": "✓", "failed": "✗", "unknown": "•"}[e.verdict]
    lead = [glyph]
    if e.typed_counts:
        lead.append("◆")
    if e.reference == "linked":
        lead.append("⇢")
    elif e.reference == "missing":
        lead.append("⇢?")
    return " ".join(lead) + "  "


def _row_suffix(e: FileRailEntry) -> str:
    """The trailing badges — exclusion count and manifest caveat — unchanged in shape
    from before M-32, just split out of ``_row_text`` so the eliding delegate can keep
    them intact while shortening only the filename between prefix and suffix."""
    bits = []
    if e.excluded_count:
        n = e.excluded_count
        mark = " ↺" if e.excluded_carried else ""
        bits.append(f"[{n} excl{mark}]")
    if e.overrides_count:
        bits.append(f"[{e.overrides_count} ovr]")
    if e.caveat:
        bits.append("⚠")
    return ("   " + "   ".join(bits)) if bits else ""


def _row_text(e: FileRailEntry) -> str:
    return f"{_row_prefix(e)}{e.filename}{_row_suffix(e)}"


def _elided_row_text(e: FileRailEntry, fm, avail_px: float) -> str:
    """``_row_text(e)`` with ONLY the filename segment elided (middle-elided, so both a
    long prefix and a long suffix survive) to fit ``avail_px`` pixels under font metrics
    ``fm``. The prefix/suffix glyphs are the whole reason a rail row exists at a glance
    — a badge is a handful of characters, cheap in pixels, and the one thing a maximally
    -shortened row still has to show — so they are never touched, even if that leaves
    less than ``avail_px`` worth of room for the filename (down to zero, "…" alone)."""
    prefix, suffix = _row_prefix(e), _row_suffix(e)
    fixed_w = fm.horizontalAdvance(prefix) + fm.horizontalAdvance(suffix)
    name_avail = max(avail_px - fixed_w, 0)
    elided = fm.elidedText(e.filename, Qt.TextElideMode.ElideMiddle, name_avail)
    return f"{prefix}{elided}{suffix}"


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
        note = " — carried over from a previous recordings folder" if e.excluded_carried else ""
        lines.append(f"{n} breath{'s' if n != 1 else ''} manually excluded{note}")
    if e.typed_counts:
        n = sum(e.typed_counts.values())
        kinds = ", ".join(f"{k} ×{c}" for k, c in sorted(e.typed_counts.items()))
        note = " — carried over from a previous recordings folder" if e.typed_carried else ""
        lines.append(f"{n} breath{'s' if n != 1 else ''} typed as a manoeuvre ({kinds}){note}")
    if e.reference == "linked":
        lines.append("⇢ reference resolved from another file")
    elif e.reference == "missing":
        lines.append("⇢? reference not resolved")
    if e.overrides_count:
        lines.append(
            f"{e.overrides_count} segmentation override{'s' if e.overrides_count != 1 else ''} "
            "(manual cut/join)")
    if e.segments is not None:
        lines.append(f"{e.segments} EMG segment{'s' if e.segments != 1 else ''}")
    if e.role == "reference":
        lines.append("Reference manoeuvres only — no tidal breaths")
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
                    breaths=prev.breaths, error=prev.error, excluded_count=prev.excluded_count,
                    excluded_carried=prev.excluded_carried, typed_counts=dict(prev.typed_counts),
                    typed_carried=prev.typed_carried, reference=prev.reference,
                    segments=prev.segments, role=prev.role))
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

    def mark_result(self, filename: str, *, ok: bool, breaths=None, error=None,
                    role=None) -> None:
        """Record the verdict of the latest dry run / batch run FOR THIS FILE. A file
        that is not currently a row (filtered out of an old manifest, say) is silently
        ignored — the caller does not have to check membership first.

        ``role`` (M-32) is ``core.pipeline.FileResult.role`` ('tidal'/'reference') from
        the SAME result the caller already has ``breaths`` from — like ``breaths``, only
        meaningful on success, so it is cleared to ``None`` on failure exactly the same
        way."""
        i = self._by_name.get(filename)
        if i is None:
            return
        e = self._entries[i]
        e.verdict = "ok" if ok else "failed"
        e.breaths = breaths if ok else None
        e.error = None if ok else error
        e.role = role if ok else None
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def set_caveat(self, filename: str, caveat: str | None) -> None:
        """Restore a manifest-derived caveat (outlier reason / frequency mismatch) a caller
        already knew, without a full :meth:`set_manifest` rebuild. Needed since B03: a
        caveat-free manifest rebuild (``manifest_from_filenames``, used when a batch run's
        OWN resolved file list repopulates the rail) always resets ``caveat`` to ``None`` for
        every row — by design for that caller, but wrong for a caller sharing the rail with
        a real manifest scan (Preview & QC's own ``build_manifest``) that already knew
        better. The caller is responsible for knowing what the caveat should be; this only
        writes it."""
        i = self._by_name.get(filename)
        if i is None or self._entries[i].caveat == caveat:
            return
        self._entries[i].caveat = caveat
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def set_excluded_count(self, filename: str, count: int, carried: bool = False) -> None:
        i = self._by_name.get(filename)
        if i is None:
            return
        e = self._entries[i]
        if e.excluded_count == count and e.excluded_carried == carried:
            return
        e.excluded_count = count
        e.excluded_carried = carried
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def set_overrides_count(self, filename: str, count: int) -> None:
        """Same shape as :meth:`set_excluded_count`, for
        ``processing.segmentation.overrides`` — no carried-folder flag (unlike
        exclude/typed): this count is compared by value so a caller recomputing the
        SAME count every sync never triggers a spurious repaint."""
        i = self._by_name.get(filename)
        if i is None:
            return
        e = self._entries[i]
        if e.overrides_count == count:
            return
        e.overrides_count = count
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def set_typed_state(self, filename: str, counts: dict, carried: bool = False) -> None:
        """Same shape as :meth:`set_excluded_count`, for ``breath_types`` — ``counts`` is
        a fresh ``kind -> count`` mapping (an empty dict, never ``None``, for "no typed
        breaths"). Compared by value, not identity, so a caller recomputing the SAME
        counts every sync (the common case) never triggers a spurious repaint."""
        i = self._by_name.get(filename)
        if i is None:
            return
        e = self._entries[i]
        if e.typed_counts == counts and e.typed_carried == carried:
            return
        e.typed_counts = dict(counts)
        e.typed_carried = carried
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def set_reference(self, filename: str, reference: str | None) -> None:
        """Plumbing for M-34/M-37: nothing in M-32 itself calls this with a non-``None``
        value, but the field, its rendering (``_row_prefix``/``_row_tooltip``) and this
        setter all exist now so a later ticket only has to start calling it."""
        i = self._by_name.get(filename)
        if i is None or self._entries[i].reference == reference:
            return
        self._entries[i].reference = reference
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    def set_segments(self, filename: str, segments: int | None) -> None:
        i = self._by_name.get(filename)
        if i is None or self._entries[i].segments == segments:
            return
        self._entries[i].segments = segments
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


#: rough allowance for the style's own item padding/margins around the text rect
#: (icon area, focus frame, left/right insets) — a fixed style-independent constant
#: rather than reading it from the delegate's own QStyle at paint time, since erring a
#: few pixels wide (eliding a touch earlier than strictly necessary) is harmless, while
#: erring narrow risks clipping a badge, the one thing this delegate exists to prevent.
_RAIL_ITEM_PADDING_PX = 10


class _RailItemDelegate(QStyledItemDelegate):
    """Elides ONLY the filename segment of a row's display text — never the leading
    verdict/typed/reference glyphs, nor the trailing exclusion/caveat badges — so a long
    recording name shortens where a reader least needs the full text, while the state
    glyphs a maximally-narrow 280 px rail exists to show stay fully visible regardless of
    filename length (M-32's own acceptance criterion). Reimplemented rather than relying
    on ``QAbstractItemView.setTextElideMode()``: that mode elides the row's single opaque
    display string as a whole, with no notion of where the filename starts/ends within
    it, so a plain ``Qt.ElideMiddle`` can (depending on how long the badges happen to be)
    still eat into a glyph instead of the filename."""

    def _prepare_option(self, option, index) -> QStyleOptionViewItem:
        """The actual per-row decision, split out of :meth:`paint` so a test can drive
        REAL delegate code (not just the standalone ``_elided_row_text`` pure function)
        without needing a live painter/``drawControl`` call. Returns an option whose
        ``text`` is already correctly elided and whose ``textElideMode`` is forced to
        ``ElideNone`` — see the inline comment below for why that second part matters."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        entry = index.data(EntryRole)
        if entry is not None:
            avail = opt.rect.width() - _RAIL_ITEM_PADDING_PX
            opt.text = _elided_row_text(entry, opt.fontMetrics, avail)
            # _RAIL_ITEM_PADDING_PX is a guessed, style-independent allowance for the
            # style's own content-rect margins (icon area, focus frame, selection
            # insets), which genuinely vary by style/theme/platform. If the real margin
            # ever exceeds the guess, CE_ItemViewItem's default Qt.ElideRight would
            # re-elide the string we already hand-shortened — cutting into the trailing
            # badges, the one thing this whole delegate exists to prevent. Disabling the
            # style's own elision makes that failure mode structurally impossible instead
            # of dependent on how good the padding guess is: worst case a row overflows
            # its cell by a few px, never a lost badge.
            opt.textElideMode = Qt.TextElideMode.ElideNone
        return opt

    def paint(self, painter, option, index):
        opt = self._prepare_option(option, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)


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
    #: M-32: a row's context menu asked to manage its cross-file reference. Unconnected
    #: by anything today — M-37 wires this up once reference resolution (M-34) exists —
    #: but the rail emits it now so that later ticket only has to add a slot.
    referencesRequested = Signal(str)

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
        self.filter_edit.setAccessibleName("Filter files")   # C02: named for QAccessible/screen readers
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._proxy.set_filter_text)
        root.addWidget(self.filter_edit)

        self.view = QListView()
        # C02: this is what used to be the Preview/Run file_combo — a value with no label
        # (QAccessible reported it announcing only its own current text). "Recording" names
        # what the value IS, matching the ticket's a11y review.
        self.view.setAccessibleName("Recording")
        self.view.setModel(self._proxy)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.view.setUniformItemSizes(True)
        self.view.setAlternatingRowColors(True)
        self.view.setToolTip("One row per file. Click to preview; ◀/▶ or "
                             "PageUp/PageDown also move the selection.")
        # M-32: an eliding delegate replaces the default one so a long filename shortens
        # in the MIDDLE of the filename specifically, never into the leading/trailing
        # state glyphs — see _RailItemDelegate's own docstring.
        self._delegate = _RailItemDelegate(self.view)
        self.view.setItemDelegate(self._delegate)
        self.view.selectionModel().currentChanged.connect(self._on_view_current_changed)
        self.view.doubleClicked.connect(self._on_double_clicked)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._on_context_menu)
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

    def mark_result(self, filename: str, *, ok: bool, breaths=None, error=None,
                    role=None) -> None:
        self._model.mark_result(filename, ok=ok, breaths=breaths, error=error, role=role)

    def set_caveat(self, filename: str, caveat: str | None) -> None:
        self._model.set_caveat(filename, caveat)

    def set_excluded_count(self, filename: str, count: int, carried: bool = False) -> None:
        self._model.set_excluded_count(filename, count, carried=carried)

    def set_overrides_count(self, filename: str, count: int) -> None:
        self._model.set_overrides_count(filename, count)

    def set_typed_state(self, filename: str, counts: dict, carried: bool = False) -> None:
        self._model.set_typed_state(filename, counts, carried=carried)

    def set_reference(self, filename: str, reference: str | None) -> None:
        self._model.set_reference(filename, reference)

    def set_segments(self, filename: str, segments: int | None) -> None:
        self._model.set_segments(filename, segments)

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

    def _build_row_context_menu(self, name: str) -> QMenu:
        """Construction only, no ``exec()`` — split out so a test can build the menu and
        trigger its action directly (the same pattern ``PreviewScreen._build_type_menu``
        already uses) without ever popping a real modal loop under offscreen Qt. Today's
        single action emits :attr:`referencesRequested`; left enabled unconditionally,
        since "no reference exists to manage yet" is a question M-34/M-37's own handler
        answers, not this menu."""
        menu = QMenu(self.view)
        # a transient popup QMenu is never destroyed on its own once closed — see
        # ui/CLAUDE.md's "A transient popup QMenu needs Qt.WA_DeleteOnClose" (the same
        # fix _MechanicsMixin._build_type_menu already applies for its own menu).
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        action = menu.addAction("Reference manoeuvres…")
        action.triggered.connect(lambda: self.referencesRequested.emit(name))
        return menu

    def _on_context_menu(self, pos):
        """A row's right-click menu (M-32). Acts on the row under the cursor — NOT
        necessarily the current selection, matching how a context menu is expected to
        act on whatever it was opened over."""
        index = self.view.indexAt(pos)
        if not index.isValid():
            return
        name = index.data(NameRole)
        if not name:
            return
        menu = self._build_row_context_menu(name)
        menu.exec(self.view.viewport().mapToGlobal(pos))

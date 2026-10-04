"""ReferencePickerDialog -- browse and link cross-file reference manoeuvres (M-37).

An analysed file's IC/FVC/baseline-IC/maximal-effort reference values do not have to
come from itself (``core.analysis.references``, M-34/M-35/M-36): another file's
already-typed manoeuvre breath can serve instead, either per file
(``processing.references``) or per participant group (``processing.reference_defaults``,
not exposed here -- see ``_MechanicsMixin._set_reference``'s "Use as IC reference for"
submenu for the group/all-files shortcuts this dialog does not replace). This dialog is
the one place all FOUR slots of a single target file's ``processing.references`` entry
can be reviewed and edited together, from any file's typed breaths.

Settings-only, no recording is read: the breath list for a chosen source file/slot comes
straight from ``processing.breath_types`` (whatever has already been typed via the
Mechanics type menu, M-31) -- exactly like ``core.analysis.references.resolve_reference``
itself only ever reasons about ``Settings``, never a file's actual samples.
"""
from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout)

from respmech.core.settings import BreathRef
from respmech.ui.help_text import tooltip as _tip

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

#: slot -> the label shown for its row. Order matches REFERENCE_SLOTS (ReferenceEntry's
#: own field order), so the dialog's row order never disagrees with check_links'.
_SLOT_LABELS = {
    "ic": "IC (inspiratory capacity)",
    "fvc": "FVC (forced vital capacity)",
    "baseline_ic": "Baseline IC",
    "max_insp": "Maximal inspiratory effort",
}
#: slot -> the BreathTypeEntry.kind values a source file's breath must carry to be
#: offered for that slot -- mirrors core.analysis.references._OWN_TYPED_KINDS exactly
#: for ic/fvc/max_insp. baseline_ic has no such kind of its own (a baseline is by nature
#: a DIFFERENT recording, typically quiet breathing, not a specific manoeuvre -- see
#: ReferenceEntry's own docstring), so None here means "any typed breath in the file".
_SLOT_SOURCE_KINDS = {
    "ic": frozenset({"ic", "ic_fvc"}),
    "fvc": frozenset({"fvc", "ic_fvc"}),
    "max_insp": frozenset({"max_insp", "sniff"}),
    "baseline_ic": None,
}
#: the "no source file" choice in a row's file combo.
_NONE_SENTINEL = "—"  # "—"


def _typed_breaths(settings, file, slot):
    kinds = _SLOT_SOURCE_KINDS[slot]
    return sorted(
        bt.breath for bt in settings.processing.breath_types
        if bt.file == file and (kinds is None or bt.kind in kinds))


class _SlotRow:
    """One reference slot's three widgets (source-file combo, source-breath combo,
    Clear button) plus the bookkeeping to read back a staged ``BreathRef | None``. A
    plain helper, not a QWidget itself -- ReferencePickerDialog owns the grid these are
    placed into."""

    def __init__(self, slot, settings, files, current):
        self.slot = slot
        self._settings = settings
        self._files = files
        # Whether the user has actually interacted with THIS row since the dialog
        # opened -- distinct from whatever value happens to be staged. The row's combos
        # can only ever represent a SINGLE breath (see staged() below), but a slot that
        # currently resolves via the file's own typed breaths (resolve_reference's
        # "own typed" fallback) can legitimately be backed by SEVERAL breaths at once
        # (IcSettings.aggregate combines them) -- pre-selecting current.breaths[0] here
        # already drops the rest from VIEW, and ReferencePickerDialog.staged() (below)
        # only ever writes a row the caller can prove the user touched, so an untouched
        # multi-breath reference is never silently collapsed to one breath by an OK
        # click that never looked at this slot at all.
        self.touched = False

        self.file_combo = QComboBox()
        self.file_combo.addItem(_NONE_SENTINEL, None)
        for f in files:
            self.file_combo.addItem(f, f)
        self.file_combo.setToolTip(_tip(
            f"processing.references[…].{slot}.file",
            "Which file's typed breath supplies this reference. — means none: "
            "the slot then falls back to a participant-group default, then (IC/FVC/"
            "maximal effort only) this file's own typed breath."))
        self.breath_combo = QComboBox()
        self.breath_combo.setToolTip(_tip(
            f"processing.references[…].{slot}.breaths",
            "The source file's already-typed breath to use — type a breath via the "
            "Mechanics right-click menu first if none are offered here."))
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setAutoDefault(False)
        self.clear_btn.setToolTip("Remove this row's source, falling back to whatever "
                                  "would otherwise resolve for this slot.")

        if current is not None and current.file in files:
            idx = self.file_combo.findData(current.file)
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
        self._refresh_breaths()
        if current is not None and current.breaths:
            idx = self.breath_combo.findData(current.breaths[0])
            if idx >= 0:
                self.breath_combo.setCurrentIndex(idx)

        # Connected AFTER the programmatic pre-selection above, so opening the dialog
        # on an already-resolved value never marks the row touched by itself.
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        self.breath_combo.currentIndexChanged.connect(self._on_touch)
        self.clear_btn.clicked.connect(self._on_clear)

    def _on_file_changed(self, *_args):
        self._on_touch()
        self._refresh_breaths()

    def _on_touch(self, *_args):
        self.touched = True

    def _refresh_breaths(self, *_args):
        self.breath_combo.clear()
        file = self.file_combo.currentData()
        if not file:
            self.breath_combo.setEnabled(False)
            return
        breaths = _typed_breaths(self._settings, file, self.slot)
        self.breath_combo.setEnabled(bool(breaths))
        if not breaths:
            self.breath_combo.addItem("(no typed breaths in this file)", None)
            return
        for b in breaths:
            self.breath_combo.addItem(f"Breath {b}", b)

    def _on_clear(self):
        idx = self.file_combo.findData(None)
        if idx >= 0:
            self.file_combo.setCurrentIndex(idx)

    def staged(self) -> BreathRef | None:
        file = self.file_combo.currentData()
        breath = self.breath_combo.currentData()
        if not file or breath is None:
            return None
        return BreathRef(file=file, breaths=[breath])


class ReferencePickerDialog(QDialog):
    """One target file's four reference slots, editable together.

    ``files`` is every source file offered in a row's file combo -- the caller's own
    batch glob (``ui.validation.matching_files``), NOT a manifest's majority-column-count
    subset (a reference source can be a differently-shaped, manoeuvre-only recording a
    column-count vote would otherwise exclude -- see ``check_links``'s own docstring).

    After ``exec()`` returns ``QDialog.Accepted``, :meth:`staged` holds all four slots'
    final ``BreathRef | None`` values -- an untouched slot simply repeats
    :func:`resolve_reference`'s opening value, so the caller can always overwrite the
    target entry's four fields wholesale rather than diffing which ones changed itself.
    ``QDialog.Rejected`` means "leave everything exactly as it was"; the caller must not
    write anything in that case.
    """

    def __init__(self, target_file, settings, files, parent=None):
        from respmech.core.analysis.references import REFERENCE_SLOTS, resolve_reference  # noqa: PLC0415

        super().__init__(parent)
        self.setWindowTitle(f"Reference manoeuvres — {target_file}")
        self.setModal(True)
        self._target_file = target_file

        lay = QVBoxLayout(self)
        intro = QLabel(
            f"Which breath, in which file, is {target_file}'s reference for each "
            "manoeuvre kind. Left as —, a file falls back to its participant "
            "group's default, then (IC/FVC/maximal effort only) its own typed breath.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        grid = QGridLayout()
        grid.addWidget(QLabel("Reference"), 0, 0)
        grid.addWidget(QLabel("Source file"), 0, 1)
        grid.addWidget(QLabel("Source breath"), 0, 2)
        self._rows = {}
        for row_i, slot in enumerate(REFERENCE_SLOTS, start=1):
            current = resolve_reference(target_file, slot, settings)
            row = _SlotRow(slot, settings, files, current)
            grid.addWidget(QLabel(_SLOT_LABELS[slot]), row_i, 0)
            grid.addWidget(row.file_combo, row_i, 1)
            grid.addWidget(row.breath_combo, row_i, 2)
            grid.addWidget(row.clear_btn, row_i, 3)
            self._rows[slot] = row
        lay.addLayout(grid)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        if _theme is not None:
            _theme.make_primary(ok_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        lay.addLayout(btn_row)

        self.setMinimumSize(520, 260)

    def staged(self) -> dict:
        """``{slot: BreathRef | None}`` for all four :data:`REFERENCE_SLOTS` -- what
        EVERY row currently shows, touched or not. A caller must consult
        :meth:`touched_slots` before writing any of these back: an untouched row simply
        echoes whatever :func:`resolve_reference` already resolved to when the dialog
        opened (truncated to one breath by this row's own single-breath combo, see
        ``_SlotRow.touched``'s docstring), and writing that value into an explicit
        ``ReferenceEntry`` would silently narrow a legitimate multi-breath own-typed
        reference the user never looked at."""
        return {slot: row.staged() for slot, row in self._rows.items()}

    def touched_slots(self) -> set:
        """Slot names the user actually edited (changed the source file/breath, or
        clicked Clear) since the dialog opened. The ONLY slots a caller may write back
        -- see :meth:`staged`'s docstring for why the other three must be left alone."""
        return {slot for slot, row in self._rows.items() if row.touched}

    def showEvent(self, ev):                # noqa: N802 - Qt API
        """Open at the requested size, never past the screen -- the same guard every
        other dialog in this codebase uses (see migration_report_dialog.py)."""
        super().showEvent(ev)
        if not getattr(self, "_clamped", False):
            self._clamped = True
            from respmech.ui import screen_fit  # noqa: PLC0415
            screen_fit.clamp_to_screen(self, prefer=QSize(640, 320))

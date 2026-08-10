"""Visual channel-assignment modal.

A dropdown at the top picks which (valid) data file to look at; below it every column of
that file is plotted stacked on a shared time axis, with a dropdown above each column that
designates its role (Flow, Volume, Poes, Pgas, Pdi, EMG, or unused). Single-signal roles
(flow/volume/poes/pgas/pdi) are mutually exclusive — picking one for a column clears it from
any other. Entropy is deliberately NOT in that dropdown: it is the one non-exclusive role,
so it has an independent checkbox per column and a column can be both. Each trace is drawn
in its role's colour so the mapping reads at a glance. Role assignments are per COLUMN, so they persist when you switch files (the files of
a batch share the same channel layout). On OK the caller reads ``selected_mapping()`` and
writes the channel columns into settings; the dialog is Qt-only and headless-testable.

Dark-mode aware: the plot backgrounds and trace colours come from ``theme.plot_palette()``
exactly like the other RespMech plots.
"""
from __future__ import annotations

import os
import re

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from respmech.ui import wheel as _wheel
from respmech.ui.column_stack import (REQUIRED, REQUIRED_LABELS, ROLES, SINGLE, ColumnStack,
                                      as_2d, infer_roles_from_names,
                                      name_suffix as _name_suffix,
                                      plot_palette, role_color)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

# kept as module-level aliases: the tests and _index_of_role read them by these names
_ROLES, _SINGLE, _REQUIRED, _REQUIRED_LABELS = ROLES, SINGLE, REQUIRED, REQUIRED_LABELS
_as_2d, _role_color, _plot_pal = as_2d, role_color, plot_palette   # legacy aliases

# ticket D01: pandas' own ParserError for a ragged row reads e.g. "Error tokenizing data.
# C error: Expected 3 fields in line 4, saw 9" — recognisable, but in pandas' vocabulary
# (fields, tokenizing, a C-level error prefix), not RespMech's (rows/values). Translated
# below into the phrasing a physiologist can act on.
_FIELD_COUNT_RE = re.compile(r"Expected (\d+) fields? in line (\d+), saw (\d+)")

_HEADER_BLOCK_HINT = (
    "Instrument exports often start with a header block (LabChart writes Interval= / "
    "ChannelTitle= / Range= lines). Delete the lines above the channel data, or "
    "re-export without the header.")

# The header-block guidance only makes sense for a mismatch NEAR THE TOP of the file — a
# preamble is a handful of lines, not thousands. A ragged row deep in an otherwise-good
# file (self-review finding: a truncated export, a quoted field pandas miscounts at line
# 5000) is a different problem, and telling a physiologist to "delete the lines above the
# channel data" when the fault is at the BOTTOM would send them looking in the wrong place.
_HEADER_BLOCK_HINT_MAX_LINE = 15


def _describe_read_failure(exc):
    """A short, non-technical phrase for why one file failed to load — the exception's own
    message when nothing more specific is recognised, or a translated, actionable phrasing
    when the shape of the failure (pandas' "Expected N fields… saw M" ParserError) is
    recognised. ``_HEADER_BLOCK_HINT`` is appended ONLY when the mismatch is near the top
    of the file (see ``_HEADER_BLOCK_HINT_MAX_LINE``) — the one real-world shape that
    actually IS a preamble/header block sitting above the actual channel data. Never
    raises."""
    msg = str(exc).strip()
    m = _FIELD_COUNT_RE.search(msg)
    if m:
        expected, line, saw = m.groups()
        plural = "" if saw == "1" else "s"
        phrase = f"row {line} has {saw} value{plural} but the first row has {expected}."
        if int(line) <= _HEADER_BLOCK_HINT_MAX_LINE:
            phrase += " " + _HEADER_BLOCK_HINT
        return phrase
    return msg or exc.__class__.__name__


class NoReadableFileError(ValueError):
    """Raised by ``ChannelSetupDialog.__init__`` when NONE of the candidate files could be
    read (ticket D01). A dedicated subclass, not a bare ``ValueError``: the caller
    (``SettingsScreen._open_channel_setup``) needs to tell THIS specific, already-diagnosed
    failure apart from an incidental ``ValueError`` an unrelated bug elsewhere in the
    constructor might raise (e.g. a numpy reshape error) — the latter must still show as a
    generic, full-traceback error, not be presented as if it were a plain-language
    diagnosis."""


def _no_files_readable_message(nfiles, failures):
    """The message for the :class:`NoReadableFileError` raised when every candidate file
    failed to load. ``failures`` is ``[(filename, exception), ...]`` in the order the files
    were tried — this names the FIRST one (the one a user would investigate first) and its
    cause, replacing the previous bare "None of the matching data files could be read."
    sentence, which named neither. Falls back to that same bare sentence if, somehow,
    nothing was recorded (defensive only — the caller always tries at least one file
    first)."""
    if not failures:
        return "None of the matching data files could be read."
    name, exc = failures[0]
    detail = _describe_read_failure(exc)
    if nfiles == 1:
        return f"{name} could not be read: {detail}"
    return f"None of the {nfiles} matching files could be read. {name}: {detail}"


def _mapping_names_no_role(m):
    """True when a channel-mapping dict asserts NOTHING — no column claims any role.

    Covers both an absent mapping (``None``/``{}``) AND the shape
    ``SettingsScreen._current_channel_mapping()`` always returns for a never-configured
    analysis: ``{'flow': None, 'volume': None, ..., 'emg': [], 'entropy': []}`` — every key
    present, every value falsy. That dict is truthy as a Python object (it has 7 keys), so
    a bare ``not m`` check would treat it as "an existing mapping" and defeat ticket D27's
    name-based seeding on the dialog's ordinary, everyday open path (the "Assign channels
    from data…" button always calls ``_open_channel_setup(initial=None)``, which fills
    ``initial`` from that very method) — seeding would then only ever fire from the one
    call site that happens to pass a literal ``{}`` (the guided auto-open), a real bug
    found in self-review of this ticket, verified against ``settings_screen.py``."""
    if not m:
        return True
    return (not any(m.get(r) for r in ("flow", "volume", "poes", "pgas", "pdi"))
            and not m.get("emg") and not m.get("entropy"))


class ChannelSetupDialog(QDialog):
    """Assign each raw data column to a physiological role, across a batch of files.

    ``files`` is a non-empty list of data-file paths (all sharing the same column count);
    ``loader`` is ``loader(path) -> (matrix, names)`` (matrix is samples x columns). A
    dropdown at the top switches which file's raw data is shown; the first file is the
    default. Role assignments are per column and persist across file switches. ``initial``
    pre-selects the dropdowns. After ``exec()``, ``selected_mapping()`` returns
    {'flow': col1based or None, ..., 'emg': [cols], 'entropy': [cols]}.

    ``integrate_from_flow`` (ticket D02) seeds the "No volume channel — derive volume by
    integrating flow" checkbox from the caller's current
    ``processing.volume.integrate_from_flow`` — read back after ``exec()`` via
    :meth:`integrate_from_flow`, next to :meth:`selected_mapping`, so the caller decides
    when and where to write it into settings (it is not part of the mapping dict: Volume
    is the one required-in-spirit role a batch can satisfy WITHOUT a column). Volume is
    deliberately absent from ``column_stack.REQUIRED`` — a flow-only rig with no separate
    volume channel is a supported setup (see the README), not missing data — so the OK
    gate below is satisfied by EITHER a Volume column or this checkbox, never both at
    once required. Auto-ticked at open when nothing has claimed the Volume role yet (no
    column, and the caller's own setting was already off): the exact state an unedited
    flow-only rig starts in, so the person with that rig does not have to go looking for
    the one control that unblocks OK. The checkbox and a real Volume column are kept
    mutually exclusive in BOTH directions — assigning Volume to a column un-ticks the box
    (``_on_role_changed``), and ticking the box clears any column still carrying Volume
    (``_on_volume_from_flow_toggled``) — because the core loader (``core/io/loaders.py``)
    gives ``integrate_from_flow`` priority over an assigned column when both are set, so
    leaving both on would silently ignore whichever the user set second, the exact
    silent-wrongness this ticket exists to close.

    ``suggest_from_names`` (ticket D27, default on): when ``initial`` names no role at all
    — either because it is falsy, or because every one of its values is (see
    ``_mapping_names_no_role``: a never-configured analysis' saved mapping is a dict with
    all 7 keys present but every value ``None``/``[]``, which is truthy as a Python object)
    — seed the dropdowns from a conservative, case-insensitive lookup in the file's own
    column names instead of leaving every row on "(unused)" — see
    ``column_stack.infer_roles_from_names``. Only that one path is touched: a caller WITH
    an existing mapping behaves exactly as before, and the entropy checkboxes are never
    seeded this way. A seeded row is marked "suggested" in its header
    (:meth:`_build_header`) until the user edits that row, so a guess can never look like a
    confirmed choice."""

    def __init__(self, files, fs, initial=None, loader=None, parent=None, excluded=None,
                integrate_from_flow=False, suggest_from_names=True):
        super().__init__(parent)
        self.setModal(True)
        # Opening size, clamped to the screen in showEvent. The HEIGHT is content-derived
        # there rather than fixed at 660: a real 21-channel rig produced 2521 px of stacked
        # previews behind a 444 px viewport, i.e. under four of twenty-one rows visible, and
        # the dialog never grew toward the screen it had room on.
        self._preferred = QSize(940, 660)
        self._files = list(files)
        self._loader = loader or (lambda p: (np.empty((0, 0)), []))
        self._fs = fs or 1.0
        self._cache = {}
        self._file_idx = 0
        # B01: files the caller's manifest scan already excluded (a different column count)
        # — named here so the banner below reconciles the "N files in the batch" it shows
        # with the true size of the matched folder, instead of silently only ever
        # describing the majority subset this dialog itself was handed.
        self._excluded = list(excluded or [])

        # The first file is the default, but a file can pass the cheap column probe and
        # still fail a full read (e.g. a ragged row) — fall forward to the first file that
        # actually loads, so one bad default never kills the whole modal (switching to a
        # later bad file is handled gracefully by _on_file_changed's revert). Every file's
        # own exception is kept (not just discarded), so that if NONE could be read, the
        # error below can name the file and the underlying cause instead of a bare
        # sentence (ticket D01).
        matrix0 = names0 = None
        start = 0
        failures = []                       # [(filename, exception), ...], in try order
        for start in range(len(self._files)):
            try:
                matrix0, names0 = self._get(self._files[start])
                break
            except Exception as exc:                       # remember, then try the next file
                failures.append((os.path.basename(self._files[start]), exc))
                continue
        if matrix0 is None:
            raise NoReadableFileError(_no_files_readable_message(len(self._files), failures))
        matrix0 = _as_2d(matrix0)
        self._file_idx = start
        self._ncols = matrix0.shape[1]
        self._names = list(names0)
        self.setWindowTitle("Assign channels — " + os.path.basename(self._files[start]))

        initial = initial or {}
        # ticket D27: only a brand-new analysis (no saved mapping at all) gets seeded from
        # the file's own column names. A single SINGLE-role (flow/volume/poes/pgas/pdi)
        # inferred for more than one column is itself a form of ambiguity -- at the batch
        # level, not just within one name -- so that role is left unseeded for every column
        # that claimed it, rather than silently picking one.
        self._suggested = set()                # 0-based column indices still marked "suggested"
        self._suggested_labels = []             # index-aligned with self._combos
        if suggest_from_names and _mapping_names_no_role(initial):
            inferred = infer_roles_from_names(self._names)
            single_hits, emg_cols = {}, []
            for col0, role in inferred.items():
                col1 = col0 + 1
                if role == "emg":
                    emg_cols.append(col1)
                else:
                    single_hits.setdefault(role, []).append(col1)
            seeded = {}
            seeded_cols = set()
            for role, cols in single_hits.items():
                if len(cols) == 1:
                    seeded[role] = cols[0]
                    seeded_cols.add(cols[0] - 1)
            if emg_cols:
                seeded["emg"] = emg_cols
                seeded_cols.update(c - 1 for c in emg_cols)
            if seeded_cols:
                initial = seeded
                self._suggested = seeded_cols
        self._suggested_count = len(self._suggested)     # frozen: the footnote's wording
        preselect = self._roles_from_mapping(initial)      # column index -> role key
        has_volume_col = "volume" in preselect.values()

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 12)
        v.setSpacing(6)

        # -- file selector (only valid data files; switching keeps the assignments) --
        frow = QHBoxLayout(); frow.setSpacing(8)
        flab = QLabel("Data file"); flab.setProperty("status", "muted")
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(240)
        self.file_combo.setMaximumWidth(440)             # a sensible size, not full-width
        self.file_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for f in self._files:
            self.file_combo.addItem(os.path.basename(f))
        self.file_combo.setCurrentIndex(start)          # default to the first loadable file
        self.file_combo.setToolTip(self.file_combo.currentText())   # recover a name elided at 440px
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        frow.addWidget(flab); frow.addWidget(self.file_combo); frow.addStretch(1)
        v.addLayout(frow)

        hint = QLabel("Pick what each column is. Flow, Volume, Poes, Pgas and Pdi take one "
                      "column each; EMG can take several. Tick Entropy on any column you also "
                      "want sample entropy for — including one that already has a role. "
                      "Assignments are kept when you switch files.")
        hint.setProperty("status", "muted"); hint.setWordWrap(True)
        v.addWidget(hint)

        # ticket D02: a rig with no separate volume channel (README-supported) still has
        # to reach OK somehow — see the class docstring for the priority/auto-uncheck
        # reasoning. Auto-ticked when nothing else has claimed Volume yet AND the caller's
        # own setting was already off, so an in-progress "derive from flow" choice from a
        # reopened analysis is never silently flipped back off just because this dialog
        # was opened again.
        self._volume_from_flow = QCheckBox(
            "No volume channel — derive volume by integrating flow")
        self._volume_from_flow.setToolTip(
            "Skip assigning a Volume column and compute it instead by integrating the "
            "Flow signal — the same 'Calculate volume from flow' setting also on Preview "
            "& QC ▸ Mechanics ▸ Advanced… ▸ Volume. Use this for a rig that records flow "
            "but not a separate, integrated volume channel.")
        self._volume_from_flow.setChecked(bool(integrate_from_flow) or not has_volume_col)
        self._volume_from_flow.toggled.connect(self._on_volume_from_flow_toggled)
        v.addWidget(self._volume_from_flow)

        nfiles = len(self._files)
        banner_text = (f"This mapping is applied to all {nfiles} file"
                      f"{'s' if nfiles != 1 else ''} in the batch.")
        if self._excluded:
            def _fmt_excluded(name, cols):
                detail = "unreadable" if cols is None else f"{cols} col{'s' if cols != 1 else ''}"
                return f"{name} ({detail})"
            names = ", ".join(_fmt_excluded(name, cols) for name, cols in self._excluded[:3])
            if len(self._excluded) > 3:
                names += f", +{len(self._excluded) - 3} more"
            n_ex = len(self._excluded)
            banner_text += (f" {n_ex} more file{'s' if n_ex != 1 else ''} matched but "
                            f"{'is' if n_ex == 1 else 'are'} not shown here: {names}.")
        banner = QLabel(banner_text)
        banner.setProperty("banner", True)
        banner.setProperty("status", "warn" if self._excluded else "info")
        banner.setWordWrap(True)
        v.addWidget(banner)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._scroll = scroll               # eventFilter redirects stray wheel events here
        # The stack itself is shared with the Setup screen's read-only channel summary
        # (ui/column_stack.py); the only dialog-specific part is what goes in each header.
        self._combos, self._entropy_boxes = [], []
        self._preselect = preselect
        self._stack = ColumnStack(self._fs, header_factory=self._build_header)
        self._stack.build(matrix0, self._names, roles=preselect)
        self._plots, self._curves, self._headers = (self._stack.plots, self._stack.curves,
                                                    self._stack.headers)
        scroll.setWidget(self._stack)
        # The wheel belongs to the column list: without this a role dropdown steps its own
        # selection (silently re-assigning a channel) and a preview graph zooms.
        self._wheel_guard = _wheel.guard_scroll_area(
            scroll, extra=[p.viewport() for p in self._plots])
        # file_combo sits ABOVE the scroll area, so scrolling it must do nothing rather than
        # move the list underneath — and certainly not switch files and reload every plot.
        self._wheel_swallow = _wheel.swallow_wheel(extra=[self.file_combo], parent=self)
        v.addWidget(scroll, 1)

        foot = QHBoxLayout()
        self.info = QLabel(""); self.info.setProperty("status", "muted")
        self.info.setWordWrap(True)          # the kept-role notes can run long
        foot.addWidget(self.info, 1); foot.addStretch(0)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        ok = QPushButton("OK")
        try:
            if _theme is not None:
                _theme.make_primary(ok)
        except Exception:                           # pragma: no cover
            pass
        ok.clicked.connect(self.accept)
        self._ok_btn = ok
        # Enter commits, Esc cancels. Without this Qt promotes Cancel (added first, so it
        # reads left of OK) as the default button and Enter discards the whole mapping.
        cancel.setAutoDefault(False)
        ok.setDefault(True)
        foot.addWidget(cancel); foot.addWidget(ok)
        v.addLayout(foot)
        self._refresh_info()

    def showEvent(self, ev):                # noqa: N802 - Qt API
        """Open big enough to show the channels there is room for, never past the screen."""
        super().showEvent(ev)
        if not getattr(self, "_clamped", False):
            self._clamped = True
            from respmech.ui import screen_fit  # noqa: PLC0415
            prefer = QSize(self._preferred)
            try:
                # ask for the whole stack plus the dialog's own chrome; clamp_to_screen cuts
                # it back to the work area, so a 21-channel rig opens as tall as the display
                # allows instead of at a fixed 660 px
                content = self._stack.sizeHint().height()
                chrome = self.sizeHint().height() - self._scroll.sizeHint().height()
                prefer.setHeight(max(prefer.height(), content + chrome))
            except Exception:               # pragma: no cover - sizing is best-effort
                pass
            screen_fit.clamp_to_screen(self, prefer=prefer)

    @property
    def _pal(self):
        """The plot palette actually in use — owned by the column stack."""
        return self._stack.pal

    def _build_header(self, i, head):
        """One header row: the time column is shown for reference but not assignable (the
        analysis derives time from the sampling frequency); every other column gets a role
        dropdown."""
        if i == 0:
            self._combos.append(None); self._entropy_boxes.append(None)
            self._suggested_labels.append(None)
            note = QLabel("time axis — not assignable"); note.setProperty("status", "muted")
            head.addWidget(note)
            return
        combo = QComboBox()
        for _key, label in _ROLES:
            combo.addItem(label)
        combo.setCurrentIndex(self._index_of_role(self._preselect.get(i, "")))
        combo.currentIndexChanged.connect(lambda _idx, ci=i: self._on_role_changed(ci))
        self._combos.append(combo)
        head.addWidget(combo)
        # ticket D27: a role seeded from the column's own name must never LOOK confirmed —
        # this tag disappears (see _on_role_changed) the moment the user touches the row.
        if i in self._suggested:
            tag = QLabel("suggested"); tag.setProperty("status", "info")
            tag.setToolTip("Guessed from the column name — check it.")
            self._suggested_labels.append(tag)
            head.addWidget(tag)
        else:
            self._suggested_labels.append(None)
        # Independent of the dropdown by construction — which is the whole point. Sample
        # entropy may be computed on any column, including one that already carries flow or
        # a pressure, so there is no conflict to resolve and no cross-column rule to get
        # wrong: ticking one column can never affect another.
        box = QCheckBox("Entropy")
        box.setToolTip("Also compute sample entropy on this column. Independent of the role "
                       "above — a column can be both.")
        box.setChecked((i + 1) in self._entropy_cols)
        box.toggled.connect(lambda _on, ci=i: self._on_entropy_toggled(ci))
        self._entropy_boxes.append(box)
        head.addWidget(box)

    # -- file loading / switching -------------------------------------------
    def _get(self, path):
        if path not in self._cache:
            self._cache[path] = self._loader(path)
        return self._cache[path]

    def _on_file_changed(self, idx):
        """Re-plot the newly-selected file's columns; the role dropdowns (assignments) are
        untouched, so they persist across files."""
        if not (0 <= idx < len(self._files)):
            return
        try:
            matrix, names = self._get(self._files[idx])
            matrix = _as_2d(matrix)
        except Exception:                           # probed OK but full read failed -> revert
            self.file_combo.blockSignals(True)
            self.file_combo.setCurrentIndex(self._file_idx)
            self.file_combo.blockSignals(False)
            self.info.setText("Could not read that file.")
            return
        self._file_idx = idx
        self.file_combo.setToolTip(self.file_combo.currentText())
        self._names = list(names)
        self._stack.set_data(matrix, names)

    def _name_suffix(self, i):
        return _name_suffix(self._names, i)

    # -- mapping <-> dropdown state -----------------------------------------
    def _roles_from_mapping(self, m):
        """Turn a {'flow': col, ..., 'emg': [cols]} mapping into a {col_index: role} dict
        (columns are 1-based in settings, 0-based here). A single role wins over emg if a
        column is reused, so a colliding pre-selection never silently drops the pressure
        channel.

        Entropy is NOT in this dict. It is the one non-exclusive role — sample entropy may be
        computed on a column that already carries flow or a pressure, and the shipped example
        config does exactly that — so it has its own per-column checkbox and needs no conflict
        resolution at all. ``_entropy_cols`` seeds those boxes; ``_entropy_kept`` holds the
        entropy columns this dialog cannot show a box for (the time axis, or past the end of
        the file), which ``selected_mapping`` re-emits rather than silently dropping.

        EMG can still collide with a single role, and one dropdown cannot show both, so a
        displaced EMG column is remembered and re-emitted — keyed on the column still showing
        the role it was seeded with (``_shadow_anchor``), so editing that column releases it
        and the memory can never migrate onto a channel the user has since re-assigned. Two
        single roles on one column are deliberately not preserved: that is an invalid mapping
        the QC strip blocks, not a state worth resurrecting."""
        out = {}
        self._shadowed = {}                  # 1-based column -> roles it carries but cannot show
        self._shadow_anchor = {}             # 1-based column -> the role its combo was seeded with
        self._offfile = {"emg": []}          # assigned past this file's last column
        self._entropy_cols = set()           # 1-based columns whose Entropy box starts ticked
        self._entropy_kept = []              # entropy this dialog cannot show, but must keep

        def _claim(col1, role):
            displaced = out.get(col1 - 1)
            if displaced == "emg" and displaced != role:
                self._shadowed.setdefault(col1, set()).add(displaced)
            out[col1 - 1] = role

        for c in (m.get("emg") or []):
            if not c:
                continue
            if 1 <= int(c) <= self._ncols:
                _claim(int(c), "emg")
            else:
                self._offfile["emg"].append(int(c))
        for c in (m.get("entropy") or []):
            if not c:
                continue
            # column 1 is the time axis and carries no checkbox, so it is kept, not shown
            if 2 <= int(c) <= self._ncols:
                self._entropy_cols.add(int(c))
            else:
                self._entropy_kept.append(int(c))
        for role in ("flow", "volume", "poes", "pgas", "pdi"):   # ...single roles win
            c = m.get(role)
            if c and 1 <= int(c) <= self._ncols:
                _claim(int(c), role)
        self._shadow_anchor = {c: out[c - 1] for c in self._shadowed}
        return out

    def _hidden_roles(self):
        """Roles carried by a column that is still showing the role it was seeded with, i.e.
        the ones ``selected_mapping`` will re-emit. Recomputed live, so changing a column
        releases its hidden roles with no signal bookkeeping to get wrong."""
        return {c: roles for c, roles in getattr(self, "_shadowed", {}).items()
                if self._role_of(c - 1) == self._shadow_anchor.get(c)}

    def _entropy_on(self, col_index):
        box = self._entropy_boxes[col_index]
        return box is not None and box.isChecked()

    def _display_role(self, col_index):
        """Which colour the trace carries. A column doing nothing BUT entropy is not
        'unused', so it takes the entropy colour rather than the muted grey."""
        role = self._role_of(col_index)
        return role or ("entropy" if self._entropy_on(col_index) else "")

    def _on_entropy_toggled(self, col_index):
        self._recolor(col_index)
        self._refresh_info()

    @staticmethod
    def _index_of_role(role):
        for i, (key, _label) in enumerate(_ROLES):
            if key == role:
                return i
        return 0

    def _role_of(self, col_index):
        combo = self._combos[col_index]
        return "" if combo is None else _ROLES[combo.currentIndex()][0]   # None == the time column

    def _dismiss_suggestion(self, col_index):
        """Ticket D27: a role seeded from the column's own name must stop looking like a
        guess the instant this row is touched — by the user directly, or (from
        ``_on_role_changed``'s collision handling below) displaced by an edit made
        elsewhere. Safe to call on a column that was never suggested."""
        self._suggested.discard(col_index)
        label = self._suggested_labels[col_index]
        if label is not None:
            label.setVisible(False)

    def _on_role_changed(self, col_index):
        role = self._role_of(col_index)
        self._dismiss_suggestion(col_index)
        if role in _SINGLE:                         # enforce one-column-per single role
            for j, combo in enumerate(self._combos):
                if combo is not None and j != col_index and self._role_of(j) == role:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)        # (unused)
                    combo.blockSignals(False)
                    self._recolor(j)
                    self._dismiss_suggestion(j)
        # ticket D02: a real Volume column now exists, so a still-ticked "derive from
        # flow" checkbox would silently win over it at run time (core/io/loaders.py gives
        # integrate_from_flow priority over an assigned column) — the exact kind of
        # silent-wrongness this ticket closes for the OPPOSITE case. Only fires on a
        # genuine user edit: the initial preselect sets the combos before this signal is
        # connected, so opening an already-"derive from flow" analysis never flips it.
        if role == "volume" and self._volume_from_flow.isChecked():
            self._volume_from_flow.setChecked(False)
        self._recolor(col_index)
        self._refresh_info()

    def _recolor(self, col_index):
        self._stack.set_role(col_index, self._display_role(col_index))

    def _on_volume_from_flow_toggled(self, checked):
        """The mutual-exclusion guard between the checkbox and a real Volume column, in
        the OTHER direction from ``_on_role_changed`` above (found in self-review: the
        original implementation only guarded column-then-checkbox, not
        checkbox-after-column). Without this, assigning a real Volume column (which
        un-ticks the box, as designed) and then manually RE-ticking it afterwards left
        both set with no warning — ``_refresh_info`` even suppresses the 'derived from
        flow' note in that state, since a column IS present, so the dialog would show
        a plain 'Ready' with no sign that the core loader (``core/io/loaders.py``) is
        about to silently ignore the column the user just picked (it gives
        ``integrate_from_flow`` priority whenever both are set). Ticking the box now
        clears any column still carrying the Volume role, symmetric to the reverse case."""
        if checked:
            for i in range(self._ncols):
                if self._role_of(i) == "volume":
                    combo = self._combos[i]
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)            # (unused)
                    combo.blockSignals(False)
                    self._recolor(i)
                    self._dismiss_suggestion(i)
        self._refresh_info()

    def _missing_required(self):
        """Required single roles (flow/poes/pgas/pdi) not yet assigned to any column."""
        present = {self._role_of(i) for i in range(self._ncols)}
        return [r for r in _REQUIRED if r not in present]

    def _volume_assigned(self):
        """True once some column carries the Volume role — as distinct from Volume being
        SATISFIED, which the checkbox can also do (see ``_refresh_info``)."""
        return any(self._role_of(i) == "volume" for i in range(self._ncols))

    def integrate_from_flow(self) -> bool:
        """The "derive volume from flow" checkbox's current state — read by the caller
        after ``exec()`` accepts, next to :meth:`selected_mapping`, and written into
        ``processing.volume.integrate_from_flow`` (not part of the mapping dict: this is
        a processing setting, not a channel column)."""
        return self._volume_from_flow.isChecked()

    def _refresh_info(self):
        """Show progress, and gate OK on the required roles being assigned, AND Volume
        being satisfied — a column OR the "derive from flow" checkbox, never both required
        at once (ticket D02: a flow-only rig with no separate volume channel is a
        supported setup, not missing data) — so a partial mapping can never silently leave
        a required channel at a stale/wrong column, and OK can never be reached with
        neither volume source chosen."""
        missing = self._missing_required()
        volume_ok = self._volume_assigned() or self._volume_from_flow.isChecked()
        if missing:
            names = ", ".join(_REQUIRED_LABELS[r] for r in missing)
            text = f"Assign {names} to continue"
        elif not volume_ok:
            text = "Assign Volume, or tick 'derive from flow', to continue"
        else:
            assigned = sum(1 for i in range(self._ncols) if self._display_role(i))
            text = f"Ready — {assigned} column{'s' if assigned != 1 else ''} assigned"
            if self._volume_from_flow.isChecked() and not self._volume_assigned():
                text += "  ·  volume derived from flow"
        # A role kept on a column that displays a different one would otherwise be invisible.
        # Say so in BOTH branches: while a required role is missing is exactly when the user
        # is still editing and might act on it.
        for note in self._kept_notes():
            text += f"  ·  {note}"
        # ticket D27: named while ANY seeded row is still unreviewed — in both branches, for
        # the same reason as the kept-role note above — and disappears on its own once every
        # seeded row has been touched (self._suggested shrinks in _dismiss_suggestion).
        if self._suggested:
            n = self._suggested_count
            text += (f"  ·  {n} column{'s' if n != 1 else ''} pre-filled from the column "
                    "names, check them.")
        self.info.setText(text)
        if getattr(self, "_ok_btn", None) is not None:
            self._ok_btn.setEnabled(not missing and volume_ok)

    def _kept_notes(self):
        """Human phrases for everything selected_mapping will re-emit but cannot show."""
        notes = []
        hidden = sorted(c for c, roles in self._hidden_roles().items() if "emg" in roles)
        if hidden:
            notes.append(f"EMG is also kept on column{'s' if len(hidden) != 1 else ''} "
                         + ", ".join(f"#{c}" for c in hidden))
        for label, cols in (("EMG", sorted(self._offfile["emg"])),
                            ("Entropy", sorted(self._entropy_kept))):
            if cols:
                notes.append(f"{label} column{'s' if len(cols) != 1 else ''} "
                             + ", ".join(f"#{c}" for c in cols)
                             + f" kept but not shown here (this file has "
                             + f"{self._ncols} columns, and column 1 is the time axis)")
        return notes

    def selected_mapping(self):
        """The chosen mapping: single roles -> 1-based column (or None), emg/entropy ->
        sorted list of 1-based columns."""
        m = {"flow": None, "volume": None, "poes": None, "pgas": None, "pdi": None,
             "emg": [], "entropy": []}
        for i in range(self._ncols):
            role = self._role_of(i)
            if not role:
                continue
            if role in _SINGLE:
                m[role] = i + 1                     # mutual exclusion keeps it unique
            else:
                m[role].append(i + 1)
        # EMG can be hidden behind a single role on a shared column, and either list can name
        # a column past this file's width. Pressing OK must never be a way to lose a channel
        # assignment the dialog never showed. set() also absorbs a column the user has since
        # picked explicitly, so nothing is listed twice.
        hidden = {c for c, roles in self._hidden_roles().items() if "emg" in roles}
        m["emg"] = sorted(set(m["emg"]) | hidden | set(self._offfile["emg"]))
        m["entropy"] = sorted({i + 1 for i in range(self._ncols) if self._entropy_on(i)}
                              | set(self._entropy_kept))
        return m

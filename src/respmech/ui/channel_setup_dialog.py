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

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from respmech.ui import wheel as _wheel
from respmech.ui.column_stack import (REQUIRED, REQUIRED_LABELS, ROLES, SINGLE, ColumnStack,
                                      as_2d, name_suffix as _name_suffix,
                                      plot_palette, role_color)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

# kept as module-level aliases: the tests and _index_of_role read them by these names
_ROLES, _SINGLE, _REQUIRED, _REQUIRED_LABELS = ROLES, SINGLE, REQUIRED, REQUIRED_LABELS
_as_2d, _role_color, _plot_pal = as_2d, role_color, plot_palette   # legacy aliases

class ChannelSetupDialog(QDialog):
    """Assign each raw data column to a physiological role, across a batch of files.

    ``files`` is a non-empty list of data-file paths (all sharing the same column count);
    ``loader`` is ``loader(path) -> (matrix, names)`` (matrix is samples x columns). A
    dropdown at the top switches which file's raw data is shown; the first file is the
    default. Role assignments are per column and persist across file switches. ``initial``
    pre-selects the dropdowns. After ``exec()``, ``selected_mapping()`` returns
    {'flow': col1based or None, ..., 'emg': [cols], 'entropy': [cols]}."""

    def __init__(self, files, fs, initial=None, loader=None, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.resize(940, 660)
        self._files = list(files)
        self._loader = loader or (lambda p: (np.empty((0, 0)), []))
        self._fs = fs or 1.0
        self._cache = {}
        self._file_idx = 0

        # The first file is the default, but a file can pass the cheap column probe and
        # still fail a full read (e.g. a ragged row) — fall forward to the first file that
        # actually loads, so one bad default never kills the whole modal (switching to a
        # later bad file is handled gracefully by _on_file_changed's revert).
        matrix0 = names0 = None
        start = 0
        for start in range(len(self._files)):
            try:
                matrix0, names0 = self._get(self._files[start])
                break
            except Exception:                              # try the next file
                continue
        if matrix0 is None:
            raise ValueError("None of the matching data files could be read.")
        matrix0 = _as_2d(matrix0)
        self._file_idx = start
        self._ncols = matrix0.shape[1]
        self._names = list(names0)
        self.setWindowTitle("Assign channels — " + os.path.basename(self._files[start]))

        initial = initial or {}
        preselect = self._roles_from_mapping(initial)      # column index -> role key

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
        nfiles = len(self._files)
        banner = QLabel(f"This mapping is applied to all {nfiles} file"
                        f"{'s' if nfiles != 1 else ''} in the batch.")
        banner.setProperty("banner", True)
        banner.setProperty("status", "info"); banner.setWordWrap(True)
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
        foot.addWidget(cancel); foot.addWidget(ok)
        v.addLayout(foot)
        self._refresh_info()

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

    def _on_role_changed(self, col_index):
        role = self._role_of(col_index)
        if role in _SINGLE:                         # enforce one-column-per single role
            for j, combo in enumerate(self._combos):
                if combo is not None and j != col_index and self._role_of(j) == role:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)        # (unused)
                    combo.blockSignals(False)
                    self._recolor(j)
        self._recolor(col_index)
        self._refresh_info()

    def _recolor(self, col_index):
        self._stack.set_role(col_index, self._display_role(col_index))

    def _missing_required(self):
        """Required single roles (flow/poes/pgas/pdi) not yet assigned to any column."""
        present = {self._role_of(i) for i in range(self._ncols)}
        return [r for r in _REQUIRED if r not in present]

    def _refresh_info(self):
        """Show progress, and gate OK on the required roles being assigned — so a partial
        mapping can never silently leave a required channel at a stale/wrong column."""
        missing = self._missing_required()
        if missing:
            names = ", ".join(_REQUIRED_LABELS[r] for r in missing)
            text = f"Assign {names} to continue"
        else:
            assigned = sum(1 for i in range(self._ncols) if self._display_role(i))
            text = f"Ready — {assigned} column{'s' if assigned != 1 else ''} assigned"
        # A role kept on a column that displays a different one would otherwise be invisible.
        # Say so in BOTH branches: while a required role is missing is exactly when the user
        # is still editing and might act on it.
        for note in self._kept_notes():
            text += f"  ·  {note}"
        self.info.setText(text)
        if getattr(self, "_ok_btn", None) is not None:
            self._ok_btn.setEnabled(not missing)

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

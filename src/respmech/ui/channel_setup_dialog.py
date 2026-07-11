"""Visual channel-assignment modal.

A dropdown at the top picks which (valid) data file to look at; below it every column of
that file is plotted stacked on a shared time axis, with a dropdown above each column that
designates its role (Flow, Volume, Poes, Pgas, Pdi, EMG, Entropy, or unused). Single-signal
roles (flow/volume/poes/pgas/pdi) are mutually exclusive — picking one for a column clears
it from any other. Each trace is drawn in its role's colour so the mapping reads at a
glance. Role assignments are per COLUMN, so they persist when you switch files (the files of
a batch share the same channel layout). On OK the caller reads ``selected_mapping()`` and
writes the channel columns into settings; the dialog is Qt-only and headless-testable.

Dark-mode aware: the plot backgrounds and trace colours come from ``theme.plot_palette()``
exactly like the other RespMech plots.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

import pyqtgraph as pg

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

# role key -> menu label. "" is the (unused) sentinel and must stay first.
_ROLES = [
    ("", "(unused)"),
    ("flow", "Flow"),
    ("volume", "Volume"),
    ("poes", "Poes — oesophageal pressure"),
    ("pgas", "Pgas — gastric pressure"),
    ("pdi", "Pdi — transdiaphragmatic"),
    ("emg", "EMG — diaphragm"),
    ("entropy", "Entropy"),
]
# roles that may be assigned to at most one column (the multi-column roles are emg/entropy)
_SINGLE = {"flow", "volume", "poes", "pgas", "pdi"}
# roles the core analysis always requires (volume is optional — it can be integrated from
# flow instead; emg/entropy are optional). OK stays disabled until each is assigned, so a
# partial mapping can never silently leave a required channel pointing at the wrong column.
_REQUIRED = ("flow", "poes", "pgas", "pdi")
_REQUIRED_LABELS = {"flow": "Flow", "poes": "Poes", "pgas": "Pgas", "pdi": "Pdi"}


def _plot_pal():
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


def _as_2d(matrix):
    """A raw matrix as a 2-D ``(samples, columns)`` float array (a lone column -> one col)."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _role_color(pal, role):
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
        pal = _plot_pal()
        self._pal = pal

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
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        frow.addWidget(flab); frow.addWidget(self.file_combo); frow.addStretch(1)
        v.addLayout(frow)

        hint = QLabel("Pick what each column is. Flow, Volume, Poes, Pgas and Pdi take one "
                      "column each; EMG and Entropy can take several. Assignments are kept "
                      "when you switch files.")
        hint.setProperty("status", "muted"); hint.setWordWrap(True)
        v.addWidget(hint)
        nfiles = len(self._files)
        banner = QLabel(f"This mapping is applied to all {nfiles} file"
                        f"{'s' if nfiles != 1 else ''} in the batch.")
        banner.setProperty("status", "info"); banner.setWordWrap(True)
        v.addWidget(banner)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget(); rows = QVBoxLayout(content)
        rows.setContentsMargins(0, 0, 0, 0); rows.setSpacing(10)

        t = np.arange(matrix0.shape[0], dtype=float) / self._fs
        self._combos, self._plots, self._curves, self._headers = [], [], [], []
        prev = None
        for i in range(self._ncols):
            col = QWidget(); cv = QVBoxLayout(col)
            cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(2)
            head = QHBoxLayout(); head.setContentsMargins(2, 0, 2, 0); head.setSpacing(8)
            lab = QLabel(f"Column {i + 1}" + self._name_suffix(i))
            head.addWidget(lab)
            if i == 0:
                # the first column is always the time axis — shown for reference but not
                # assignable (the analysis derives time from the sampling frequency)
                combo = None
                role0 = ""
                note = QLabel("time axis — not assignable"); note.setProperty("status", "muted")
                head.addWidget(note)
            else:
                combo = QComboBox()
                for _key, label in _ROLES:
                    combo.addItem(label)
                role0 = preselect.get(i, "")
                combo.setCurrentIndex(self._index_of_role(role0))
                combo.currentIndexChanged.connect(lambda _idx, ci=i: self._on_role_changed(ci))
                head.addWidget(combo)
            head.addStretch(1)
            cv.addLayout(head)

            plot = pg.PlotWidget()
            plot.setBackground(pal["bg"])
            plot.setFixedHeight(74)
            plot.setMenuEnabled(False)
            plot.getAxis("bottom").setStyle(showValues=(i == self._ncols - 1))
            y = matrix0[:, i]
            curve = plot.plot(t[:len(y)], y, pen=pg.mkPen(_role_color(pal, role0),
                                                          width=1), connect="finite")
            if i == self._ncols - 1:
                plot.setLabel("bottom", "Time (s)")
            if prev is not None:
                plot.setXLink(prev)
            prev = plot
            cv.addWidget(plot)
            rows.addWidget(col)
            self._combos.append(combo); self._plots.append(plot)
            self._curves.append(curve); self._headers.append(lab)

        scroll.setWidget(content)
        v.addWidget(scroll, 1)

        foot = QHBoxLayout()
        self.info = QLabel(""); self.info.setProperty("status", "muted")
        foot.addWidget(self.info); foot.addStretch(1)
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
        self._names = list(names)
        t = np.arange(matrix.shape[0], dtype=float) / self._fs
        for i in range(self._ncols):
            y = matrix[:, i] if i < matrix.shape[1] else np.full(matrix.shape[0], np.nan)
            self._curves[i].setData(t[:len(y)], y)
            self._plots[i].enableAutoRange()
            self._headers[i].setText(f"Column {i + 1}" + self._name_suffix(i))

    def _name_suffix(self, i):
        """The source header for column i, shown after the generic index (e.g. ' · flow'),
        or '' when the file had no usable name for it."""
        name = self._names[i].strip() if i < len(self._names) else ""
        if not name or name.startswith("__") or name.lower().startswith("unnamed"):
            return ""
        return f"  ·  {name}"

    # -- mapping <-> dropdown state -----------------------------------------
    def _roles_from_mapping(self, m):
        """Turn a {'flow': col, ..., 'emg': [cols]} mapping into a {col_index: role} dict
        (columns are 1-based in settings, 0-based here). A single role wins over emg/entropy
        if a column is reused, so a colliding pre-selection never silently drops the pressure
        channel."""
        out = {}
        for role in ("emg", "entropy"):             # write multi roles first...
            for c in (m.get(role) or []):
                if c and 1 <= int(c) <= self._ncols:
                    out[int(c) - 1] = role
        for role in ("flow", "volume", "poes", "pgas", "pdi"):   # ...single roles win
            c = m.get(role)
            if c and 1 <= int(c) <= self._ncols:
                out[int(c) - 1] = role
        return out

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
        role = self._role_of(col_index)
        self._curves[col_index].setPen(pg.mkPen(_role_color(self._pal, role), width=1))

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
            self.info.setText(f"Assign {names} to continue")
        else:
            assigned = sum(1 for i in range(self._ncols) if self._role_of(i))
            self.info.setText(f"Ready — {assigned} column{'s' if assigned != 1 else ''} assigned")
        if getattr(self, "_ok_btn", None) is not None:
            self._ok_btn.setEnabled(not missing)

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
        m["emg"] = sorted(m["emg"])
        m["entropy"] = sorted(m["entropy"])
        return m

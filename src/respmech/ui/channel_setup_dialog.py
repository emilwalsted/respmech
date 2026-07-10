"""Visual channel-assignment modal.

Shows every column of a raw data file stacked on a shared time axis; above each column
is a dropdown that designates its role (Flow, Volume, Poes, Pgas, Pdi, EMG, Entropy, or
unused). Single-signal roles (flow/volume/poes/pgas/pdi) are mutually exclusive — picking
one for a column clears it from any other. Each trace is drawn in its role's colour so the
mapping reads at a glance. On OK the caller reads ``selected_mapping()`` and writes the
channel columns into settings; the dialog itself is Qt-only and headless-testable.

Dark-mode aware: the plot backgrounds and trace colours come from ``theme.plot_palette()``
exactly like the other RespMech plots.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel,
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
    ("poes", "Poes"),
    ("pgas", "Pgas"),
    ("pdi", "Pdi"),
    ("emg", "EMG"),
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
    """Assign each raw data column to a physiological role.

    ``matrix`` is ``(samples, columns)``; ``initial`` is an optional current mapping
    ({'flow': col1based or None, ..., 'emg': [cols], 'entropy': [cols]}) used to
    pre-select the dropdowns. After ``exec()``, ``selected_mapping()`` returns the chosen
    mapping in the same shape."""

    def __init__(self, matrix, fs, initial=None, file_name="", parent=None, col_names=None):
        super().__init__(parent)
        self.setWindowTitle("Assign channels" + (f" — {file_name}" if file_name else ""))
        self.setModal(True)
        self.resize(920, 640)
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        self._matrix = matrix
        self._ncols = matrix.shape[1]
        self._names = list(col_names or [])
        self._fs = fs or 1.0
        pal = _plot_pal()
        self._pal = pal
        t = np.arange(matrix.shape[0], dtype=float) / self._fs

        initial = initial or {}
        preselect = self._roles_from_mapping(initial)      # column index -> role key

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 12)
        v.setSpacing(6)
        hint = QLabel("Pick what each column is. Flow, Volume, Poes, Pgas and Pdi take one "
                      "column each; EMG and Entropy can take several. Unassigned columns are ignored.")
        hint.setProperty("status", "muted"); hint.setWordWrap(True)
        v.addWidget(hint)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        from PySide6.QtWidgets import QFrame
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget(); rows = QVBoxLayout(content)
        rows.setContentsMargins(0, 0, 0, 0); rows.setSpacing(10)

        self._combos = []
        self._plots = []
        prev = None
        for i in range(self._ncols):
            col = QWidget(); cv = QVBoxLayout(col)
            cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(2)
            head = QHBoxLayout(); head.setContentsMargins(2, 0, 2, 0); head.setSpacing(8)
            lab = QLabel(f"Column {i + 1}" + self._name_suffix(i))
            combo = QComboBox()
            for _key, label in _ROLES:
                combo.addItem(label)
            combo.setCurrentIndex(self._index_of_role(preselect.get(i, "")))
            combo.currentIndexChanged.connect(lambda _idx, ci=i: self._on_role_changed(ci))
            head.addWidget(lab); head.addWidget(combo); head.addStretch(1)
            cv.addLayout(head)

            plot = pg.PlotWidget()
            plot.setBackground(pal["bg"])
            plot.setFixedHeight(74)
            plot.setMenuEnabled(False)
            plot.getAxis("bottom").setStyle(showValues=(i == self._ncols - 1))
            y = matrix[:, i]
            plot.plot(t[:len(y)], y, pen=pg.mkPen(_role_color(pal, preselect.get(i, "")),
                                                  width=1), connect="finite")
            if i == self._ncols - 1:
                plot.setLabel("bottom", "Time (s)")
            if prev is not None:
                plot.setXLink(prev)
            prev = plot
            cv.addWidget(plot)
            rows.addWidget(col)
            self._combos.append(combo)
            self._plots.append(plot)

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
        (columns are 1-based in settings, 0-based here)."""
        out = {}
        for role in ("flow", "volume", "poes", "pgas", "pdi"):
            c = m.get(role)
            if c and 1 <= int(c) <= self._ncols:
                out[int(c) - 1] = role
        for role in ("emg", "entropy"):
            for c in (m.get(role) or []):
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
        return _ROLES[self._combos[col_index].currentIndex()][0]

    def _on_role_changed(self, col_index):
        role = self._role_of(col_index)
        if role in _SINGLE:                         # enforce one-column-per single role
            for j, combo in enumerate(self._combos):
                if j != col_index and self._role_of(j) == role:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)        # (unused)
                    combo.blockSignals(False)
                    self._recolor(j)
        self._recolor(col_index)
        self._refresh_info()

    def _recolor(self, col_index):
        role = self._role_of(col_index)
        pen = pg.mkPen(_role_color(self._pal, role), width=1)
        for item in self._plots[col_index].listDataItems():
            item.setPen(pen)

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
            self.info.setText(f"Ready — {assigned} of {self._ncols} columns assigned")
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
                m[role] = i + 1                     # last one wins (mutual exclusion keeps it unique)
            else:
                m[role].append(i + 1)
        m["emg"] = sorted(m["emg"])
        m["entropy"] = sorted(m["entropy"])
        return m

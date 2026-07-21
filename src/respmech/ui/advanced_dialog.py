"""The "Advanced…" modal: real settings that are rarely the right thing to change.

A strip of controls above a plot has to earn its width. The handful of parameters anyone
actually reaches for stay there; the rest — shape guards, spectral-gate internals, diagnostic
exports — move behind a button, where they are still discoverable, still documented, and no
longer competing for attention with the two knobs that matter.

Staging, not mutation. Every edit lives in dialog-local widgets and reaches the settings only
when the caller commits on OK, which is how both existing modals in this app work
(ChannelSetupDialog, NoiseProfileDialog). Cancel therefore needs no undo, no snapshot and no
rollback: the settings were never touched. That property is worth more than it looks —
there is no transaction helper anywhere in this UI, so a modal that edited state directly
would have to invent one.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from respmech.ui.help_text import tooltip as _tip

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None


class Field:
    """One row: how to build it, how to read it, and what it is.

    ``kind`` is "int", "float" or "bool". ``path`` is the dotted settings path the tooltip
    names, so a control keeps saying which TOML key it writes even after it moves screens.
    """

    def __init__(self, key, label, kind, path, help, *, lo=0, hi=1_000_000, step=1,
                 decimals=0, suffix="", prefix="", note=None):
        self.key, self.label, self.kind = key, label, kind
        self.path, self.help, self.note = path, help, note
        self.lo, self.hi, self.step = lo, hi, step
        self.decimals, self.suffix, self.prefix = decimals, suffix, prefix

    def build(self, value):
        if self.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
        else:
            w = QSpinBox() if self.kind == "int" else QDoubleSpinBox()
            if self.kind == "float":
                w.setDecimals(self.decimals)
            w.setRange(self.lo, self.hi)
            w.setSingleStep(self.step)
            if self.suffix:
                w.setSuffix(self.suffix)
            if self.prefix:
                w.setPrefix(self.prefix)
            w.setValue((int if self.kind == "int" else float)(value))
        w.setToolTip(_tip(self.path, self.help))
        return w

    def read(self, w):
        return w.isChecked() if self.kind == "bool" else w.value()


class AdvancedDialog(QDialog):
    """``fields`` are Field specs; ``values`` maps key -> current value.

    ``derived`` is an optional callable taking the staged values and returning a line of
    text, recomputed on every edit — for a coupling the numbers alone do not show, such as
    an STFT window whose meaning in milliseconds depends on the sampling rate.
    """

    def __init__(self, title, fields, values, parent=None, intro=None, derived=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle(title)
        self._fields = list(fields)
        self._derived = derived
        self._widgets = {}

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 12)
        v.setSpacing(8)
        if intro:
            lab = QLabel(intro)
            lab.setWordWrap(True)
            lab.setProperty("status", "muted")
            v.addWidget(lab)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        for f in self._fields:
            w = f.build(values.get(f.key))
            self._widgets[f.key] = w
            row = QLabel(f.label)
            row.setToolTip(_tip(f.path, f.help))
            form.addRow(row, w)
            if f.note:
                hint = QLabel(f.note)
                hint.setWordWrap(True)
                hint.setProperty("status", "muted")
                form.addRow("", hint)
            sig = getattr(w, "toggled", None) or getattr(w, "valueChanged", None)
            if sig is not None:
                sig.connect(self._refresh_derived)
        v.addLayout(form)

        self.derived = QLabel("")
        self.derived.setWordWrap(True)
        self.derived.setProperty("status", "muted")
        self.derived.setVisible(derived is not None)
        v.addWidget(self.derived)

        foot = QHBoxLayout()
        foot.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        try:
            if _theme is not None:
                _theme.make_primary(self.btn_ok)
        except Exception:                       # pragma: no cover — styling is cosmetic
            pass
        foot.addWidget(self.btn_cancel)
        foot.addWidget(self.btn_ok)
        v.addLayout(foot)
        self._refresh_derived()

    def _refresh_derived(self, *_):
        if self._derived is None:
            return
        try:
            self.derived.setText(self._derived(self.values()) or "")
        except Exception:                       # pragma: no cover — a hint is never fatal
            self.derived.setText("")

    def values(self):
        """The staged values. A plain dict — the dialog never sees a Settings object."""
        return {f.key: f.read(self._widgets[f.key]) for f in self._fields}

    def widget(self, key):
        return self._widgets[key]


def apply_values(target, values):
    """Write staged values onto ``target``, returning True if anything actually changed.

    The caller uses the return value to decide whether to mark the analysis modified and
    schedule a recompute — pressing OK without touching anything should do neither.
    """
    changed = False
    for key, value in values.items():
        if getattr(target, key) != value:
            setattr(target, key, value)
            changed = True
    return changed

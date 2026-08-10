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

from PySide6.QtCore import QEventLoop, QSize, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame,
                               QHBoxLayout, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from respmech.ui import screen_fit as _screen_fit
from respmech.ui import wheel as _wheel
from respmech.ui.help_text import tooltip as _tip
from respmech.ui.section_flow import SectionCard, WrapLabel, install_sections

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None


class Field:
    """One row: how to build it, how to read it, and what it is.

    ``kind`` is "int", "float", "bool", "text" or "choice" (a combo — ``options`` is a list
    of ``(label, value)`` pairs and the stored value is the chosen data value). ``path`` is
    the dotted settings path the tooltip names, so a control keeps saying which TOML key it
    writes even after it moves screens.
    """

    def __init__(self, key, label, kind, path, help, *, lo=0, hi=1_000_000, step=1,
                 decimals=0, suffix="", prefix="", note=None, placeholder="", options=None,
                 depends_on=None, auto_text=""):
        self.key, self.label, self.kind = key, label, kind
        self.path, self.help, self.note = path, help, note
        self.lo, self.hi, self.step = lo, hi, step
        self.decimals, self.suffix, self.prefix = decimals, suffix, prefix
        self.placeholder = placeholder
        self.options = options or []
        # the key of a "bool" field in the same dialog this one is meaningless without —
        # e.g. "Resample to" only means something once "Resample before analysis" is on.
        # Must name a field built earlier in the same dialog's field list.
        self.depends_on = depends_on
        # for an optional numeric setting: the caption shown AT the minimum, which reads
        # back as None ("unset"). Lets a spin box express "let the code decide" without a
        # separate checkbox. Costs the ability to enter the minimum itself, so only use it
        # where that value is meaningless anyway.
        self.auto_text = auto_text

    def build(self, value):
        if self.kind == "choice":
            w = QComboBox()
            for label, val in self.options:
                w.addItem(label, val)
            idx = w.findData(value)
            w.setCurrentIndex(idx if idx >= 0 else 0)
            # A QComboBox derives its MINIMUM width from its widest ITEM, so one explanatory
            # option label sets the minimum width of the whole dialog: "Per-file maximum
            # (% of peak breath)" measured 538 px on Windows metrics and cost the EMG modal
            # a whole column. An explicit minimumWidth genuinely lowers the floor (Qt's
            # qSmartMinSize takes an explicit minimum over the hint) while sizeHint stays
            # natural, so the combo still opens wide enough to read and only elides once the
            # user squeezes the dialog. The full label is always in the popup.
            w.setMinimumWidth(w.fontMetrics().averageCharWidth() * 12 + 36)
        elif self.kind == "text":
            w = QPlainTextEdit()
            w.setPlainText(value or "")
            w.setFixedHeight(90)
            if self.placeholder:
                w.setPlaceholderText(self.placeholder)
        elif self.kind == "bool":
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
            if self.auto_text:
                # A caption, not a number: at its minimum the control shows this word
                # instead of a value. Keep it SHORT enough for the theme's 150px spin-box
                # cap — "Auto — use minimum breath depth" needed 473px on Windows metrics
                # and was clipped mid-word at every window size. Lifting the cap per widget
                # was tried and abandoned: a QSS max-width is re-applied at polish and beats
                # setMaximumWidth, so the field ended up 416px wide beside 168px siblings.
                # test_a_spin_box_never_clips_its_own_caption holds this line.
                w.setSpecialValueText(self.auto_text)   # shown when value == self.lo
            w.setValue((int if self.kind == "int" else float)(
                self.lo if (value is None and self.auto_text) else value))
        w.setToolTip(_tip(self.path, self.help))
        return w

    def read(self, w):
        if self.kind == "choice":
            return w.currentData()
        if self.kind == "text":
            return w.toPlainText()
        if self.kind == "bool":
            return w.isChecked()
        if self.auto_text and w.value() == self.lo:
            return None
        return w.value()


def _as_sections(fields):
    """Normalise ``fields`` to ``[(title, [Field])]``.

    A flat list becomes one untitled section, which is what keeps the two-field ECG modal —
    and every existing caller and test that passes a plain list — working unchanged.
    """
    items = list(fields)
    if items and isinstance(items[0], (tuple, list)) and len(items[0]) == 2 \
            and isinstance(items[0][1], (tuple, list)):
        return [(str(t), list(fs)) for t, fs in items]
    return [("", items)]


class AdvancedDialog(QDialog):
    """``fields`` are Field specs; ``values`` maps key -> current value.

    ``fields`` is either a flat list of :class:`Field` (one untitled section) or a list of
    ``(section title, [Field])`` pairs. Sections are laid out in width-responsive columns
    inside a scroll area, which is what keeps a twenty-setting modal on a laptop screen —
    see :mod:`respmech.ui.section_flow`.

    ``derived`` is an optional callable taking the staged values and returning a line of
    text, recomputed on every edit — for a coupling the numbers alone do not show, such as
    an STFT window whose meaning in milliseconds depends on the sampling rate. Pass
    ``derived_debounce_ms`` when that computation is not trivial arithmetic (e.g. a
    scipy-based search) so it does not run on every keystroke.

    The three things OUTSIDE the scroll area — the intro, the derived hint and the button
    row — are outside it on purpose. Whatever the content does, those stay reachable.

    ``modal`` (default ``True``) and ``on_apply`` turn this into a genuinely non-modal
    modal-with-an-Apply-button, for a caller with a live view behind the dialog that
    should redraw as values are committed rather than only once the dialog closes. Both
    default to the original behaviour, so every existing caller is unaffected.
    ``on_apply``, when given, adds a third "Apply" button that calls it with
    ``edited_values()`` WITHOUT closing the dialog, and moves the "edited since open"
    baseline forward so a later OK does not resend the same keys.
    """

    def __init__(self, title, fields, values, parent=None, intro=None, derived=None,
                 max_columns=3, modal=True, on_apply=None, derived_debounce_ms=0):
        super().__init__(parent)
        self._modal = modal
        self.setModal(modal)
        self._on_apply = on_apply
        self._derived_timer = None
        self.setWindowTitle(title)
        sections = _as_sections(fields)
        self._fields = [f for _t, fs in sections for f in fs]
        #: what the model held when this dialog opened — see edited_values()
        self._opened_with = {}
        self._derived = derived
        if derived is not None and derived_debounce_ms:
            # A debounce, not a rate limit: every edit restarts the timer, so only the
            # value the user stopped on is ever computed, never an intermediate one.
            self._derived_timer = QTimer(self)
            self._derived_timer.setSingleShot(True)
            self._derived_timer.setInterval(derived_debounce_ms)
            self._derived_timer.timeout.connect(self._refresh_derived_now)
        self._widgets = {}
        #: static note labels, keyed by field key (only present for fields whose Field.note
        #: was non-empty) — exposed via note() so a caller can turn a static hint into a
        #: live one by reaching in and updating its text on the widget signals it cares
        #: about, without this class needing to know what "live" means for every field.
        self._notes = {}
        self.cards = []
        # A grip is the affordance, not the mechanism: the dialog is resizable because its
        # layout minimum is small (see section_flow), and the grip is how the user is told.
        self.setSizeGripEnabled(True)

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 12)
        v.setSpacing(8)
        if intro:
            # capped: prose outside the scroll area contributes its full wrapped height to
            # the dialog's own minimum, so an uncapped paragraph can push the floor back
            # through the screen — the defect being fixed, arriving by a different door.
            lab = WrapLabel(intro, cap_lines=6)
            lab.setProperty("status", "muted")
            v.addWidget(lab)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        self.columns = install_sections(body, max_columns=max_columns)

        for sect_title, sect_fields in sections:
            card = SectionCard(sect_title)
            for f in sect_fields:
                w = f.build(values.get(f.key))
                self._widgets[f.key] = w
                self._opened_with[f.key] = f.read(w)   # as the widget rounded it, not raw
                row = WrapLabel(f.label)
                row.setToolTip(_tip(f.path, f.help))
                card.add_row(row, w)
                hint = None
                if f.note:
                    hint = WrapLabel(f.note)
                    hint.setProperty("status", "muted")
                    card.add_note(hint)
                    self._notes[f.key] = hint
                if f.depends_on is not None:
                    # the depended-on checkbox must have been built already (earlier in
                    # ``fields``); a field naming a later or unknown key is a caller bug.
                    # Say so: with sections the order is now set by a table in the caller,
                    # one step further from here, and the bare KeyError this used to raise
                    # surfaced as the Advanced button simply doing nothing.
                    dep = self._widgets.get(f.depends_on)
                    if dep is None:
                        raise ValueError(
                            f"field {f.key!r} depends on {f.depends_on!r}, which must be "
                            f"built before it — within the same section, or in an earlier "
                            f"one")
                    # the label and note grey out with the control — a bright caption above a
                    # disabled spin box reads as an active setting.
                    for part in (w, row, hint):
                        if part is not None:
                            part.setEnabled(dep.isChecked())
                            dep.toggled.connect(part.setEnabled)
                sig = (getattr(w, "toggled", None) or getattr(w, "valueChanged", None)
                       or getattr(w, "textChanged", None)
                       or getattr(w, "currentIndexChanged", None))
                if sig is not None:
                    sig.connect(self._refresh_derived)
            self.cards.append(card)
            self.columns.addWidget(card)

        self.scroll.setWidget(body)
        # A scroll area should never need a horizontal bar just to show one column, so give
        # it room for the widest card plus the vertical bar it will have.
        try:
            bar = self.scroll.verticalScrollBar().sizeHint().width()
            self.scroll.setMinimumWidth(self.columns.minimumSize().width() + bar + 4)
        except Exception:                   # pragma: no cover - sizing is best-effort
            pass
        # A QSpinBox or QComboBox inside a scroll area eats the wheel and steps ITSELF, so
        # scrolling past a control silently rewrites an analysis parameter. The guard must be
        # kept as an attribute: an event filter that is garbage-collected stops filtering and
        # restores the bug with every construction-time test still green (see wheel.py).
        self._wheel_guard = _wheel.guard_scroll_area(self.scroll)
        # A scrollable FIELD inside the scrollable form is a dead patch: the breath-count
        # QPlainTextEdit consumes the wheel even with nothing left to scroll, so the form
        # stops moving under the cursor and never resumes. guard_scroll_area does not cover
        # it — it only finds spin boxes and combos. wheel.py has shipped NestedWheelChain
        # for precisely this since the reactive work and this is its first caller.
        self._wheel_chains = []
        for w in self._widgets.values():
            if isinstance(w, QPlainTextEdit):
                chain = _wheel.NestedWheelChain(self.scroll, w, parent=self)
                w.viewport().installEventFilter(chain)
                self._wheel_chains.append(chain)
        v.addWidget(self.scroll, 1)

        self.derived = WrapLabel("", cap_lines=4)
        self.derived.setProperty("status", "muted")
        self.derived.setVisible(derived is not None)
        v.addWidget(self.derived)

        foot = QHBoxLayout()
        foot.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply = None
        if self._on_apply is not None:
            self.btn_apply = QPushButton("Apply")
            self.btn_apply.clicked.connect(self._apply_clicked)
            # Not the Enter default (see below) and not autoDefault: Apply is a deliberate
            # extra step, not the one Enter should reach for.
            self.btn_apply.setAutoDefault(False)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        # Enter must COMMIT. Both buttons default to autoDefault, and with no explicit default
        # Qt promotes the first one in the focus chain — which is Cancel, since it is added
        # first so it reads left of OK. Enter therefore threw away every staged edit, and on a
        # dialog too tall to show its own footer that was the only action still reachable.
        # A QPlainTextEdit field (the breath-count overrides) takes Enter for itself while it
        # has focus, so this costs nothing there.
        self.btn_cancel.setAutoDefault(False)
        self.btn_ok.setDefault(True)
        try:
            if _theme is not None:
                _theme.make_primary(self.btn_ok)
        except Exception:                       # pragma: no cover — styling is cosmetic
            pass
        foot.addWidget(self.btn_cancel)
        if self.btn_apply is not None:
            foot.addWidget(self.btn_apply)
        foot.addWidget(self.btn_ok)
        v.addLayout(foot)
        # Synchronous even when debounced: the dialog must open showing a real value, not
        # an empty line waiting out the first debounce window.
        self._refresh_derived_now()

    def sizeHint(self):                     # noqa: N802 - Qt API
        """The size that shows the columns, not the size QScrollArea would ask for.

        ``QScrollArea.sizeHint`` is clamped to 36x24 em REGARDLESS of the widget inside it,
        so a dialog that simply inherited it opened one column wide on a 27" display
        (measured: 576 px). Ask the column layout what it actually wants instead, and let
        :func:`screen_fit.clamp_to_screen` cut that down to the screen.
        """
        base = super().sizeHint()
        try:
            want_w = self.columns.sizeHint().width()
            bar = self.scroll.verticalScrollBar().sizeHint().width()
            m = self.layout().contentsMargins()
            width = want_w + bar + m.left() + m.right() + 4
            # the chrome (intro, derived line, footer) is whatever the dialog needs beyond
            # the scroll area's own hint, so measure it rather than assuming a number
            chrome = base.height() - self.scroll.sizeHint().height()
            height = chrome + self.columns.heightForWidth(want_w)
            return QSize(max(base.width(), width), max(base.height(), height))
        except Exception:                   # pragma: no cover - fall back to Qt's own hint
            return base

    def showEvent(self, ev):                # noqa: N802 - Qt API
        """Clamp to the screen the first time the dialog is shown.

        Here rather than in ``__init__``: the layout has to exist before it can be measured,
        and the dialog has to have a window handle before the RIGHT screen can be identified
        (a modal opens on its parent's monitor, which need not be the primary one).
        """
        super().showEvent(ev)
        if not getattr(self, "_clamped", False):
            self._clamped = True
            _screen_fit.clamp_to_screen(self)
            # Land the caret on the first setting, as it did before the scroll area existed.
            # A QScrollArea is a focusable widget in its own right and sits earlier in the
            # tab order than anything inside it, so without this the dialog opened with the
            # scroller focused: the first Tab went to a field instead of the second, and
            # typing did nothing.
            for f in self._fields:
                w = self._widgets.get(f.key)
                if w is not None and w.isEnabled() and w.focusPolicy() != Qt.NoFocus:
                    w.setFocus(Qt.OtherFocusReason)
                    break

    def exec(self):                         # noqa: N802 - Qt API
        """Block until closed, like ``QDialog.exec()`` — but for ``modal=False``, without
        making the dialog modal while it runs.

        ``QDialog.exec()`` sets ``Qt.WA_ShowModal`` UNCONDITIONALLY while it executes,
        regardless of a prior ``setModal(False)`` (verified empirically: ``isModal()`` was
        ``True`` mid-``exec()`` even though ``setModal(False)`` had been called first — Qt
        restores the attribute afterwards, but the whole point of a non-modal Advanced
        dialog is what happens WHILE it is open). So a genuinely non-modal blocking wait
        needs its own event loop instead of Qt's ``exec()``, which a caller can still treat
        exactly like a normal modal call: it returns only once the dialog is closed, with
        the same ``QDialog.Accepted``/``Rejected`` result.

        Also quits on ``destroyed`` — not reachable through this dialog's own Cancel/OK/✕
        (all three end in ``done()``, which emits ``finished``), but a defensive fallback
        for a parent being torn down with this dialog still open, which ordinary Qt child
        cleanup can do without ever calling ``close()``/``done()`` on the child. Without it
        the loop would spin forever, since ``finished`` is never emitted by C++-side
        destruction. Once ``destroyed`` has fired, ``self`` may no longer be a live
        object — read the RESULT captured in the closure, not ``self.result()``.
        """
        if not self._modal:
            self.setResult(QDialog.Rejected)
            self.show()
            loop = QEventLoop()
            closed = {"result": QDialog.Rejected}

            def _capture_and_quit():
                try:
                    closed["result"] = self.result()
                except RuntimeError:            # destroyed before finished() could fire
                    pass
                loop.quit()

            self.finished.connect(_capture_and_quit)
            self.destroyed.connect(loop.quit)
            loop.exec()
            return closed["result"]
        return super().exec()

    def done(self, r):                      # noqa: N802 - Qt API
        """Stop the debounce timer on close — an in-flight ``singleShot`` outliving the
        dialog it belongs to costs nothing observable today, but there is no reason to let
        a closed dialog's timer fire into torn-down widgets."""
        if self._derived_timer is not None:
            self._derived_timer.stop()
        super().done(r)

    def _apply_clicked(self):
        edited = self.edited_values()
        if not edited:
            return                          # nothing staged since open/last Apply: no-op
        self._on_apply(edited)
        # Move the "edited since open" baseline forward, so a later Apply or OK does not
        # recommit the same keys — harmless (apply_values is idempotent) but a wasted
        # recompute on every subsequent commit otherwise.
        self._opened_with.update(self.values())

    def _refresh_derived(self, *_):
        if self._derived is None:
            return
        if self._derived_timer is not None:
            self._derived_timer.start()     # restart the debounce window on every edit
            return
        self._refresh_derived_now()

    def _refresh_derived_now(self):
        if self._derived is None:
            return
        try:
            self.derived.setText(self._derived(self.values()) or "")
        except Exception:                       # pragma: no cover — a hint is never fatal
            self.derived.setText("")

    def edited_values(self):
        """Only the settings the USER actually changed in this dialog.

        ``values()`` returns everything the dialog holds, and committing all of it compares
        each against the model as it is NOW — which silently reverts anything the app wrote
        while the dialog was open. That is not hypothetical: with 'Auto' on, a finished noise
        sweep writes its chosen ``prop_decrease`` back to the model, so pressing OK without
        touching a thing put the stale opening value back, marked the analysis modified and
        queued a five-panel recompute. Comparing against the values this dialog OPENED with
        makes an untouched field a genuine no-op, whatever else happened meanwhile.
        """
        return {k: v for k, v in self.values().items()
                if k not in self._opened_with or self._opened_with[k] != v}

    def values(self):
        """The staged values. A plain dict — the dialog never sees a Settings object."""
        return {f.key: f.read(self._widgets[f.key]) for f in self._fields}

    def widget(self, key):
        return self._widgets[key]

    def note(self, key):
        """The static note label built for ``key`` (see Field.note), or None if that field
        was built without one. A caller wanting a note that tracks live edits gets its
        initial text for free from Field.note and updates this label's text itself on
        whichever widget signals it cares about — this dialog has no opinion on what
        "live" means for any given field."""
        return self._notes.get(key)


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

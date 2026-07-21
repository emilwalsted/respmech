"""Stop a form from editing itself when the user is only scrolling past it.

A QComboBox or QSpinBox accepts the wheel and steps its own value, so scrolling across one
silently changes whichever setting the cursor happened to cross. Measured before this
existed: a wheel over Setup's "EMG RMS window" moved it from 0.05 s to 0.02 s — changing
every reported EMG number — and marked the analysis modified and scheduled a recompute; a
wheel over a role dropdown in the channel dialog set that column to Flow, which, Flow being
mutually exclusive, moved the flow channel off the column it was on. A tab bar and a file
picker have the same shape: one notch and you are somewhere else.

Two behaviours, because there are two situations:

* Inside a scroll area the wheel should scroll it. The event is FORWARDED to the area's
  viewport rather than translated into a scrollbar movement here. That matters more than it
  looks: QAbstractSlider drives scrolling from angleDelta only, accumulates the fractional
  remainder across events for high-resolution wheels, clamps a single event to pageStep, and
  pages on Ctrl/Shift. Computing the offset by hand got all four subtly wrong and ignored
  horizontal deltas entirely, so scrolling ran at a different speed over a widget than over
  empty space beside it.
* Outside one there is nothing to scroll, so the event is simply swallowed.

Note this is NOT what stops a pyqtgraph plot zooming — ``ViewBox.wheelEvent`` already
ignores the wheel once ``setMouseEnabled(x=False, y=False)`` is set, and that is the control
that makes a read-only preview read-only. Guarding a plot viewport here only turns "nothing
happens" into "the list scrolls".
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox, QApplication,
                               QComboBox)

#: widget classes that take the wheel to edit themselves
STEALERS = (QAbstractSpinBox, QComboBox)


class WheelGuard(QObject):
    """Takes the wheel away from the widgets it is installed on.

    With ``area``, forwards to that scroll area's viewport; without one, swallows."""

    def __init__(self, area=None, parent=None):
        super().__init__(parent)
        self._area = area
        self._busy = False           # forwarding must not re-enter through its own target

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel:
            # Deliberately unconditional. Exempting a FOCUSED spin box would restore the
            # standard Qt "click in, then wheel to nudge" idiom, but it also restores the
            # bug in its most ordinary form: click a field to type a value, then scroll down
            # to see the rest of the form with the pointer still over it. Losing the nudge
            # costs a keystroke; the alternative silently rewrites a scientific parameter.
            if self._area is not None and not self._busy:
                self._busy = True
                try:
                    QApplication.sendEvent(self._area.viewport(), ev)
                finally:
                    self._busy = False
            return True              # consumed: never edits the widget under the cursor
        return super().eventFilter(obj, ev)


class NestedWheelChain(QObject):
    """Hands the wheel to the outer scroll area once the inner one cannot move further.

    A scrollable widget sitting inside a scrollable form is otherwise a dead patch: it
    consumes the wheel even when its own scrollbar is already at the end, so the form stops
    moving under the cursor and never resumes. Chaining is what every browser does."""

    def __init__(self, area, inner, parent=None):
        super().__init__(parent)
        self._area = area
        self._inner = inner

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel:
            bar = self._inner.verticalScrollBar()
            dy = ev.pixelDelta().y() or ev.angleDelta().y()
            spent = (bar.maximum() == bar.minimum()
                     or (dy < 0 and bar.value() >= bar.maximum())
                     or (dy > 0 and bar.value() <= bar.minimum()))
            if spent:
                QApplication.sendEvent(self._area.viewport(), ev)
                return True
        return super().eventFilter(obj, ev)


def chain_nested(area, inner):
    """Let ``inner`` scroll itself, then pass the wheel out to ``area``. Keep the result."""
    chain = NestedWheelChain(area, inner, parent=inner)
    inner.viewport().installEventFilter(chain)
    return chain


def _install(guard, targets):
    for w in targets:
        if w is not None:
            w.installEventFilter(guard)
    return guard


def _stealers_under(root):
    return root.findChildren(QAbstractSpinBox) + root.findChildren(QComboBox)


def guard_scroll_area(area, extra=()):
    """Redirect the wheel to ``area`` for every stealer inside it, plus ``extra``.

    Returns the guard, which the caller MUST keep a reference to — an event filter that is
    garbage-collected stops filtering, silently restoring the bug with every construction-time
    test still green. Discovery is by class at call time, so call again after adding widgets;
    installing the same filter twice on one widget is a no-op in Qt.

    ``extra`` names widgets not found by class — a pyqtgraph PlotWidget's viewport, say."""
    guard = WheelGuard(area, parent=area)
    return _install(guard, _stealers_under(area.widget() or area) + list(extra))


def swallow_wheel(root=None, extra=(), parent=None):
    """Swallow the wheel for every stealer under ``root``, plus ``extra``.

    For controls that sit outside any scroll area — the Preview strips, a tab bar, a file
    picker — where the correct outcome is that nothing happens at all."""
    guard = WheelGuard(None, parent=parent or root)
    targets = (_stealers_under(root) if root is not None else []) + list(extra)
    return _install(guard, targets)

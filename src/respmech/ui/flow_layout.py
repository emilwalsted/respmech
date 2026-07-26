"""A layout that lays widgets out in a row and WRAPS to the next line when short of width.

WHY. The Preview control strips are a row of chips (the ECG detector knobs, the noise-gate
knobs, the gated-peak knobs) plus a few buttons. In a plain ``QHBoxLayout`` that row cannot
wrap, so its minimum width is the SUM of every chip — and Qt propagates a layout's minimum up
to the window (``QLayout.SetDefaultConstraint``), which made the main window demand 1800 px.
A window cannot be resized below its minimum, so ``showMaximized()`` and the fit-to-screen
helper were both powerless and the window ran off the right edge of any smaller screen.

With this layout the minimum width is the width of the WIDEST SINGLE ITEM, not the sum, so the
window fits on a laptop screen and the chips simply wrap onto a second line when narrow —
every control stays visible and reachable, which matters on a strip where each knob is a real
analysis parameter.

It is the standard Qt "flow layout" contract: ``heightForWidth`` reports the height the items
need at a given width, so the enclosing box layout gives the strip a second line when needed.
Stretch/spacer items are ignored (a wrapping row has no meaningful trailing stretch).
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class FlowLayout(QLayout):
    """Left-to-right, wrapping. ``h``/``v`` are the gaps between items and between lines."""

    def __init__(self, parent=None, *, margin: int = 0, h: int = 10, v: int = 6):
        super().__init__(parent)
        self._items: list = []          # keep Python refs: Qt does not own these
        self._h, self._v = h, v
        self.setContentsMargins(margin, margin, margin, margin)

    # -- QLayout plumbing ----------------------------------------------------
    def addItem(self, item):            # noqa: N802 - Qt API
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):            # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):            # noqa: N802 - Qt API
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):      # noqa: N802 - Qt API
        return Qt.Orientations(Qt.Orientation(0))

    # -- the wrapping contract ----------------------------------------------
    def hasHeightForWidth(self):        # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width):    # noqa: N802 - Qt API
        return self._lay(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):        # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._lay(rect, apply=True)

    def sizeHint(self):                 # noqa: N802 - Qt API
        return self.minimumSize()

    def minimumSize(self):              # noqa: N802 - Qt API
        """The widest SINGLE item — the whole point: a wrapping row never demands the sum."""
        size = QSize(0, 0)
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    # -- the actual placement ------------------------------------------------
    def _lay(self, rect: QRect, *, apply: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = eff.x(), eff.y(), 0
        for it in self._items:
            hint = it.sizeHint()
            w, h = hint.width(), hint.height()
            if w <= 0 and h <= 0:                  # a stretch/spacer: nothing to place
                continue
            if x + w > eff.right() + 1 and line_h > 0:      # wrap to the next line
                x, y = eff.x(), y + line_h + self._v
                line_h = 0
            if apply:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += w + self._h
            line_h = max(line_h, h)
        return y + line_h - rect.y() + m.bottom()


def elide(label, text: str, max_px: int = 320) -> None:
    """Set ``text`` on ``label``, shortened to ``max_px`` with the full text in the tooltip.

    A QLabel's minimum width is its whole text, so an un-elided read-out (which carries a real
    recording's filename) can silently push the strip — and therefore the window — arbitrarily
    wide. Elide it and keep the full string one hover away.
    """
    label.setToolTip(text)
    fm = label.fontMetrics()
    # Shorten the TEXT itself (rather than squeezing the widget): a QLabel sized from a short
    # string has a correspondingly small minimum, which is what keeps the strip narrow. Do NOT
    # reach for QSizePolicy.Ignored here — it collapses the item to zero width in a layout that
    # sizes from sizeHint(), which makes the read-out vanish entirely.
    label.setText(fm.elidedText(text, Qt.ElideMiddle, max_px) if text else "")
    label.setMaximumWidth(max_px)

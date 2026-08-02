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

NESTING. A chip is itself a row of caption+field clusters, and a chip built on a plain
``QHBoxLayout`` is one indivisible item: the strip can wrap around it but never inside it, so
the chip's own minimum is still the SUM of its clusters. That is how the window minimum crept
back to 1516 px on Windows, whose wider font metrics made the ECG chip 1486 px on its own,
while the same chip measured 975 px on macOS and the guard never fired here. So chips use a
FlowLayout too, with each caption+field pair added via :meth:`FlowLayout.addLayout` — the pair
stays glued together and the chip's minimum becomes its widest single cluster (~270 px). The
window minimum goes 1005 -> 707 px here, and stays under the 1280 px ceiling even with the
font scaled to twice this machine's metrics (measured 1069 px), where before it broke through
at 1.75x — which is about where Windows sits.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLayout, QSizePolicy


class FlowLayout(QLayout):
    """Left-to-right, wrapping. ``h``/``v`` are the gaps between items and between lines."""

    def __init__(self, parent=None, *, margin: int = 0, h: int = 10, v: int = 6):
        super().__init__(parent)
        self._items: list = []          # keep Python refs: Qt does not own these
        self._h, self._v = h, v
        # Report the horizontal gap through QLayout.spacing() too: this layout keeps its own
        # gaps, so spacing() otherwise returned Qt's unrelated default and any caller (or
        # test) reading it back got a number this layout never uses.
        self.setSpacing(h)
        self.setContentsMargins(margin, margin, margin, margin)

    # -- QLayout plumbing ----------------------------------------------------
    def addItem(self, item):            # noqa: N802 - Qt API
        self._items.append(item)

    def addLayout(self, layout):        # noqa: N802 - Qt API
        """Add a nested layout as ONE wrappable item (QLayout has no addLayout of its own).

        Use it for a caption+field pair so the two never wrap apart. ``addChildLayout`` is what
        re-parents the pair's widgets onto THIS layout's widget, so they stay direct children
        of the chip and code that looks up ``widget.parent()`` still finds it.
        """
        self.addChildLayout(layout)
        self.addItem(layout)

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
        """The NATURAL size: everything on one line. Not the same as :meth:`minimumSize`.

        The two must differ once these nest. A chip's parent strip places it at its
        ``sizeHint()``, so a chip that reported its minimum as its hint would be handed the
        width of one cluster and wrap into a tall column even on a wide window. Reporting the
        single-line width keeps the chip looking exactly as before whenever the room is there,
        while ``minimumSize`` still promises it CAN be squeezed. The window's opening size is
        set explicitly by ``MainWindow._fit_to_screen``, so a larger hint does not widen it.
        """
        w = h = n = 0
        for it in self._items:
            s = it.sizeHint()
            if s.width() <= 0 and s.height() <= 0:      # a stretch/spacer: nothing to place
                continue
            w += s.width(); h = max(h, s.height()); n += 1
        w += self._h * max(0, n - 1)
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

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
        # Placement happens a ROW at a time, because an item's vertical position depends on
        # how tall the tallest item in its row turns out to be — which is not known until the
        # row is complete. Placing each item as it came put everything flush with the row's
        # top edge, so on the ECG strip the "Capture channel"/"Min height"/"Min gap" captions
        # and the two checkboxes sat visibly higher than the spin boxes and buttons beside
        # them (Emil, 30-07-2026). Buffer the row, then centre each item in it.
        row: list = []

        def flush(row_h: int) -> None:
            for item, ix, iw, ih in row:
                item.setGeometry(QRect(QPoint(ix, y + max(0, (row_h - ih) // 2)), QSize(iw, ih)))
            row.clear()

        for it in self._items:
            hint = it.sizeHint()
            w, h = hint.width(), hint.height()
            if w <= 0 and h <= 0:                  # a stretch/spacer: nothing to place
                continue
            # Never place an item wider than the strip. Without this an item simply keeps its
            # natural width and runs off the right edge once the strip is narrower than it —
            # which for a nested chip means the window can be squeezed (the minimum says so)
            # while its controls disappear past the edge, the original bug one level down.
            # An item that wraps internally (another FlowLayout) then reports the height it
            # needs at that width, so the strip grows a line for it instead of clipping it.
            if w > eff.width():
                w = eff.width()
                h = it.heightForWidth(w) if it.hasHeightForWidth() else h
            if x + w > eff.right() + 1 and line_h > 0:      # wrap to the next line
                if apply:
                    flush(line_h)
                x, y = eff.x(), y + line_h + self._v
                line_h = 0
            if apply:
                row.append((it, x, w, h))
            x += w + self._h
            line_h = max(line_h, h)
        if apply:
            flush(line_h)
        return y + line_h - rect.y() + m.bottom()


def install_flow(widget, *, h: int = 10, v: int = 4,
                 margins: tuple = (0, 0, 0, 0)) -> FlowLayout:
    """Give ``widget`` a FlowLayout and let its HEIGHT follow its width. Returns the layout.

    The second half is the part that is easy to miss. A widget only grows a second line if its
    size policy says its height depends on its width AND allows growth: with
    ``QSizePolicy.Maximum`` Qt caps it at the one-line ``sizeHint`` height, so the wrapped row
    is laid out correctly and then simply painted outside the widget. ``Preferred`` +
    ``setHeightForWidth`` is how "as tall as it needs to be at this width, and no taller" is
    actually spelled; the chips still hug their content because the plot area beside them
    carries the vertical stretch, so there is never spare height for them to take.
    """
    lay = FlowLayout(h=h, v=v)
    lay.setContentsMargins(*margins)
    widget.setLayout(lay)
    sp = widget.sizePolicy()
    sp.setVerticalPolicy(QSizePolicy.Preferred)
    sp.setHeightForWidth(True)
    widget.setSizePolicy(sp)
    return lay


def cluster(*widgets, gap: int = 4) -> QHBoxLayout:
    """A caption and its field, glued into one item for :meth:`FlowLayout.addLayout`.

    The tight ``gap`` inside a cluster against the strip's wider gap between them is what makes
    'Strength [ ]', 'Keep >= [ ]', 'Gate [ ]' read as pairs rather than an even bead line — and
    now also what guarantees a caption never wraps away from the field it names.
    """
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(gap)
    for w in widgets:
        lay.addWidget(w)
    return lay


class ElidingLabel(QLabel):
    """A label that shortens itself to whatever width it is given, instead of forcing its
    full text on the window.

    ``elide()`` below does the same thing ONCE, against a fixed cap — right for a read-out
    whose text changes and whose column width does not. This is for the other case: fixed
    text in a row that must survive being squeezed, such as the header subtitle. Measured on
    Windows font metrics that one label reported a 683 px minimum, and since a plain
    ``QHBoxLayout`` hands its minimum straight to the window, the header alone demanded
    1196 px — more than the Preview screen's 997 px and inside 84 px of a 1280 px screen.

    ``minimumSizeHint`` deliberately reports a small floor rather than the text width: that
    is the whole point, and it is what lets the window be narrower than the sentence. The
    full string always stays one hover away in the tooltip.
    """

    def __init__(self, text: str = "", parent=None, *, mode=Qt.ElideRight, floor: int = 24):
        super().__init__(parent)
        self._full = text or ""
        self._mode = mode
        self._floor = int(floor)
        super().setToolTip(self._full)
        self.setText(self._full)

    def setFullText(self, text: str) -> None:            # noqa: N802 - Qt-style API
        self._full = text or ""
        super().setToolTip(self._full)
        self._relayout()

    def fullText(self) -> str:                           # noqa: N802 - Qt-style API
        return self._full

    def minimumSizeHint(self):                           # noqa: N802 - Qt API
        s = super().minimumSizeHint()
        s.setWidth(min(s.width(), self._floor))
        return s

    def resizeEvent(self, ev):                           # noqa: N802 - Qt API
        super().resizeEvent(ev)
        self._relayout()

    def _relayout(self) -> None:
        fm = self.fontMetrics()
        width = max(0, self.width())
        # setText, not super().setText: a subclass that overrode setText would still be
        # honoured, and QLabel.setText on an unchanged string is a no-op anyway.
        self.setText(fm.elidedText(self._full, self._mode, width) if width else self._full)


class ElidingCheckBox(QCheckBox):
    """A check box whose caption shortens itself instead of forcing its width on the window.

    Same problem and same remedy as :class:`ElidingLabel`, but a check box cannot simply be
    swapped for a wrapping label without losing click-to-toggle on the caption — so it elides
    instead. The noise-profile dialog's "Use the whole expiration of this recording" measured
    598 px on Windows font metrics and, being one unbreakable item, WAS that dialog's entire
    minimum width (620 px); at 175% Windows accessibility text scaling it reached 921 px and
    at 225% it reached 1217 px, which is wider than a 1080p laptop at 175% display scaling —
    so the dialog's own commit button could not be brought on screen.

    The indicator plus a few characters is the floor; the full caption stays in the tooltip.
    """

    def __init__(self, text: str = "", parent=None, *, floor_chars: int = 6):
        super().__init__("", parent)
        self._full = text or ""
        self._floor_chars = int(floor_chars)
        # Without this the small minimumSizeHint below is simply ignored: a QCheckBox
        # defaults to QSizePolicy.Minimum, which carries no ShrinkFlag, and Qt's
        # qSmartMinSize then takes max(sizeHint, minimumSizeHint) — i.e. the FULL caption.
        # (ElidingLabel needs no equivalent: QLabel already defaults to Preferred, which
        # does shrink.) Measured: the noise dialog's minimum stayed at 620 px until this.
        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Preferred)
        self.setSizePolicy(sp)
        super().setToolTip(self._full)
        super().setText(self._full)

    def setText(self, text):                             # noqa: N802 - Qt API
        self._full = text or ""
        if not self.toolTip():
            super().setToolTip(self._full)
        self._relayout()

    def fullText(self) -> str:                           # noqa: N802 - Qt-style API
        return self._full

    def _chrome(self) -> int:
        """Indicator width plus its spacing — the part that is never text."""
        return self.iconSize().width() + 22

    def minimumSizeHint(self):                           # noqa: N802 - Qt API
        s = super().minimumSizeHint()
        fm = self.fontMetrics()
        s.setWidth(min(s.width(), fm.averageCharWidth() * self._floor_chars + self._chrome()))
        return s

    def resizeEvent(self, ev):                           # noqa: N802 - Qt API
        super().resizeEvent(ev)
        self._relayout()

    def _relayout(self) -> None:
        room = max(0, self.width() - self._chrome())
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight, room)
                        if room else self._full)


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

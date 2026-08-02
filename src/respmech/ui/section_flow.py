"""Titled section cards laid out in newspaper columns, re-flowing with the available width.

WHY. The "Advanced…" modals are long forms — twenty settings for the mechanics one — and a
long form in a single column is taller than a laptop screen. Qt propagates a layout's minimum
up to the window and a window can never be resized below its minimum, so the modal's OK and
Cancel ended up 254 px below the bottom edge of a 1080p Windows screen at 150% scaling with no
gesture available to recover them. That is the same failure ``flow_layout`` was written for,
one level up: there the offender was a row of chips whose minimum was the SUM of the chips,
here it is a column of rows whose minimum is the SUM of the rows.

The answer has the same shape too. ``SectionColumns.minimumSize()`` reports the widest single
CARD rather than the sum, so one column is always a legal answer and the dialog can always be
squeezed; and it reports NO minimum height at all, because the cards live in a scroll area and
demanding their full stack back from the window is precisely the bug. ``heightForWidth`` then
tells the scroll area how tall the content really is at the width it was given.

Three pieces, in dependency order:

* :class:`WrapLabel` — a wrapping label whose minimum is one WORD, so helper prose stops
  driving the width of the window it happens to sit in.
* :class:`SectionCard` — one titled group of form rows, able to measure itself honestly.
* :class:`SectionColumns` — the layout engine, plus :func:`install_sections` to attach it.

Measured on the real twenty-field mechanics content at Windows font metrics: the dialog's
minimum goes 636x904 -> 396x398, and the content height at a given width falls from 1830 px
(one column) to 688 px (three), i.e. from roughly three screens of scrolling to one.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QLayout, QSizePolicy, QWidget


class WrapLabel(QLabel):
    """A wrapping label whose MINIMUM width is its longest word, not its longest line.

    A word-wrapped ``QLabel`` still reports a ``minimumSizeHint`` wide enough for a
    good-looking wrap, and every layout above it inherits that. On the mechanics modal the two
    longest ``note`` paragraphs were between them the widest thing in the whole form, so prose
    — the single most compressible thing on the screen — was setting the modal's minimum
    width. It also gets worse as the font widens, which is why Windows suffered most.

    A one-word minimum makes prose the LAST thing to constrain a layout, while
    ``heightForWidth`` still reports the height it needs at whatever width it ends up with, so
    nothing is clipped: the card simply grows taller.
    """

    #: Nominal comfortable measure for ``sizeHint``, in characters. A QLabel's default hint
    #: for un-elided text is the whole paragraph on ONE line, which would make a card's
    #: "natural" width absurd and force the column count to one on any display.
    NOMINAL_CHARS = 34

    def __init__(self, text: str = "", parent=None, *, cap_lines: int = 0):
        super().__init__("", parent)
        self.setWordWrap(True)
        self._cap_lines = int(cap_lines)
        self._full = text or ""
        self._rendering = False
        sp = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self._render()

    def setText(self, text):                # noqa: N802 - Qt API
        if self._rendering:                 # the internal QLabel.setText below
            super().setText(text)
            return
        self._full = text or ""
        self._render()

    def fullText(self) -> str:              # noqa: N802 - Qt-style API
        return self._full

    def _render(self) -> None:
        """Bound prose that sits OUTSIDE a scroll area to ``cap_lines`` lines — by ELIDING
        it, never by clipping it.

        Prose is unbounded by nature and a label above the scroller contributes its full
        wrapped height to the dialog's minimum: measured on the mechanics modal, 305 px of a
        373 px minimum was the intro and the derived hint against 68 px of actual body. So it
        has to be bounded. The first attempt bounded it with ``setMaximumHeight``, which
        silently painted 3 of the 6 lines the ECG modal's intro needs at Windows metrics —
        half a sentence, with nothing on screen to say so. Eliding says it: the reader sees
        the ellipsis and the full text is one hover away.
        """
        self._rendering = True
        try:
            if self._cap_lines <= 0 or not self._full:
                super().setText(self._full)
                self.setToolTip(self._full if self._cap_lines > 0 else self.toolTip())
                return
            width = max(1, self.width())
            lines = self._wrap_lines(self._full, width)
            if len(lines) <= self._cap_lines:
                super().setText(self._full)
            else:
                keep = lines[:self._cap_lines]
                head = self._full[:keep[-1][0]]
                tail = self._full[keep[-1][0]:]
                super().setText(head + self.fontMetrics().elidedText(
                    tail, Qt.ElideRight, width))
            self.setToolTip(self._full)
            self.setMaximumHeight(self.fontMetrics().lineSpacing() * self._cap_lines + 6)
        finally:
            self._rendering = False

    def _wrap_lines(self, text: str, width: int):
        """[(start index, length)] for ``text`` wrapped at ``width``, via Qt's own layout —
        the only thing that agrees with what QLabel will actually paint."""
        from PySide6.QtGui import QTextLayout  # noqa: PLC0415
        layout = QTextLayout(text, self.font())
        layout.beginLayout()
        out = []
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            out.append((line.textStart(), line.textLength()))
        layout.endLayout()
        return out

    def resizeEvent(self, ev):              # noqa: N802 - Qt API
        super().resizeEvent(ev)
        if self._cap_lines > 0:
            self._render()                  # how much fits depends on the width it was given

    def minimumSizeHint(self):              # noqa: N802 - Qt API
        fm = self.fontMetrics()
        longest = max((fm.horizontalAdvance(w) for w in self._full.split()), default=0)
        return QSize(longest, fm.lineSpacing())

    def sizeHint(self):                     # noqa: N802 - Qt API
        nominal = self.fontMetrics().averageCharWidth() * self.NOMINAL_CHARS
        return QSize(nominal, self.heightForWidth(nominal))


class SectionCard(QGroupBox):
    """One titled group of form rows — the unit :class:`SectionColumns` moves between columns.

    A ``QGroupBox`` holding a ``QFormLayout``, and a class only so that it can report its own
    width honestly instead of inheriting whatever its longest note wants.
    """

    def __init__(self, title: str, parent=None):
        # "&&" on the DISPLAYED title, because a QGroupBox treats a single & as a MNEMONIC
        # marker: it is swallowed and the next character underlined instead, so
        # "RMS & normalisation" rendered as "RMS _normalisation" (Emil's 02-08-2026
        # screenshot). The raw text is kept so title() still round-trips what the caller
        # passed — tests and tooling read it back, and a doubled ampersand there would be a
        # different lie in place of the first.
        super().__init__((title or "").replace("&", "&&"), parent)
        self._raw_title = title or ""
        self.form = QFormLayout(self)
        # WrapLongRows is load-bearing, not cosmetic: minimumSizeHint below reports
        # max(caption, field) per row, and that is only true because this policy stacks the
        # field under its caption when the row will not fit side by side. Changing it makes
        # the reported minimum a lie and cards start painting past their column edge.
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.form.setHorizontalSpacing(12)
        self.form.setVerticalSpacing(6)
        sp = self.sizePolicy()
        sp.setVerticalPolicy(QSizePolicy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self._captions: list = []
        self._fields: list = []

    def title(self):                        # noqa: A003 - Qt API name
        """The title as GIVEN, not as escaped for display (see __init__)."""
        return self._raw_title

    def add_row(self, caption: QLabel, field: QWidget) -> None:
        self.form.addRow(caption, field)
        self._captions.append(caption)
        self._fields.append(field)

    def add_note(self, note: QLabel) -> None:
        """Add helper prose spanning the full card width.

        Spanning, not in the field column: a note placed in the field column wraps at field
        width, which both inflates the card's height and lets the note vote on the card's
        natural width — measured at 34 px of extra height per note and a wider card on top.
        """
        self.form.addRow(note)

    # -- honest self-measurement ---------------------------------------------
    @staticmethod
    def _effective_min_width(w) -> int:
        """The width a LAYOUT will actually treat as this widget's minimum.

        Not simply ``minimumSizeHint().width()``: Qt's ``qSmartMinSize`` lets an explicit
        ``minimumWidth`` REPLACE the hint, and bounds the result by ``maximumWidth``. Both
        matter here — a combo box is given an explicit minimum precisely so its widest item
        stops setting the dialog's floor, and the theme caps spin boxes with a QSS
        ``max-width``. Measuring the hint alone would report a floor the layout does not
        honour, and the card would then be laid out narrower than it claims to need.
        """
        explicit = w.minimumWidth()
        if explicit > 0:
            return explicit
        return min(w.minimumSizeHint().width(), w.maximumWidth())

    def _chrome(self) -> int:
        m, fm = self.contentsMargins(), self.form.contentsMargins()
        return m.left() + m.right() + fm.left() + fm.right()

    def _title_floor(self) -> int:
        """Qt draws a group-box title unclipped, so the title is a hard minimum of its own."""
        return self.fontMetrics().horizontalAdvance(self.title()) + 24

    def minimumSizeHint(self):              # noqa: N802 - Qt API
        """The WIDER of caption and control per row — never their sum.

        ``WrapLongRows`` puts the field underneath its caption when the row does not fit side
        by side, so the honest minimum of a row is ``max(caption, field)``. Summing them (the
        obvious first cut, and one that was measured before it was believed) hands the layout
        a minimum about 90% too large: the "Trend anchor — absolute threshold (legacy)" row is
        a 256 px caption beside a 227 px spin box, so its card reported 519 px summed against
        290 px wrapped.
        """
        rows = [max(c.minimumSizeHint().width(), self._effective_min_width(f))
                for c, f in zip(self._captions, self._fields)]
        want = max(rows, default=0) + self._chrome()
        # ...but never claim less than the form itself will take. The row calculation above
        # is a model of what WrapLongRows does, and a model can be optimistic: measured at
        # macOS metrics it reported 113 px for a card whose QFormLayout would not go below
        # 290, so the column layout handed the card 260 px and the form laid its captions out
        # at 272 — 38 px past the card's own edge. Qt does not CLIP that overflow, it paints
        # it (the trap flow_layout.py documents one level down), and with the body exactly as
        # wide as the viewport no scrollbar appeared to recover it: three captions genuinely
        # lost glyphs at the dialog's own advertised minimum width.
        want = max(want, self.form.minimumSize().width() + self._chrome())
        return QSize(max(want, self._title_floor()), super().minimumSizeHint().height())

    def sizeHint(self):                     # noqa: N802 - Qt API
        """Widest caption plus widest control side by side — the width at which no row wraps.

        Notes are excluded on purpose: prose wraps to whatever it is given, so letting one
        paragraph vote here would have it decide the column width of the entire dialog
        (measured: 614 px against 495 px for the trend card).
        """
        pairs = [c.sizeHint().width() + self.form.horizontalSpacing() + f.sizeHint().width()
                 for c, f in zip(self._captions, self._fields)]
        want = max(max(pairs, default=0) + self._chrome(), self._title_floor())
        return QSize(want, self.heightForWidth(want))


def partition_columns(heights, k: int, vgap: int):
    """Split ``heights`` into ``k`` CONTIGUOUS runs, minimising the tallest run.

    Contiguous — newspaper columns — rather than free masonry. Shortest-column-first packing
    is marginally tighter but it REORDERS the cards as the width changes, so the same setting
    sits in the left column at one window size and the right column at another. Someone
    hunting for a half-remembered parameter then has nothing stable to hunt with. Reading
    order here is the author's order at every width and on every platform.

    Exact, by dynamic programming, because ``n`` is tiny (7 sections on the largest dialog):
    ``best[c][i]`` is the best achievable tallest-run for cards ``i..`` in ``c`` columns.
    O(n^2 k) — fine at this size, and worth revisiting only if a dialog ever grows dozens of
    sections.
    """
    n = len(heights)
    if n == 0:
        return []
    k = max(1, min(k, n))
    pre = [0] * (n + 1)
    for i, h in enumerate(heights):
        pre[i + 1] = pre[i] + h

    def cost(i, j):
        return pre[j] - pre[i] + vgap * (j - i - 1)

    inf = float("inf")
    best = [[(inf, i) for i in range(n + 1)] for _ in range(k + 1)]
    for i in range(n + 1):
        best[0][i] = (0 if i == n else inf, i)
    for c in range(1, k + 1):
        for i in range(n - 1, -1, -1):
            bh, bj = inf, i + 1
            # every column takes at least one card, and must leave one for each column after
            for j in range(i + 1, n - (c - 1) + 1):
                h = max(cost(i, j), best[c - 1][j][0])
                if h < bh:
                    bh, bj = h, j
            best[c][i] = (bh, bj)
    runs, i = [], 0
    for c in range(k, 0, -1):
        j = best[c][i][1]
        runs.append((i, j))
        i = j
    return runs


class SectionColumns(QLayout):
    """Newspaper columns whose COUNT follows the available width.

    The contract, precisely — each half of it is load-bearing:

    * ``minimumSize().width()`` is the widest single card's minimum, never the sum. One
      column is always a legal answer, so the dialog can be squeezed to the width of its
      widest section. This is what makes the window resizable at all.
    * ``minimumSize().height()`` is ZERO. The cards live in a scroll area; demanding their
      full stacked height back from the window is exactly the defect being fixed.
    * ``heightForWidth(w)`` is the tallest column at the count ``w`` affords, which is what
      lets a ``QScrollArea`` with ``widgetResizable`` scroll exactly as far as it needs to.
    * ``sizeHint()`` is the natural size at ``max_columns`` comfortable columns — larger than
      the minimum on purpose, so the dialog OPENS showing its columns and is then clamped to
      the screen by :mod:`respmech.ui.screen_fit`.
    """

    #: How far below the "no row has to wrap" width a column may be squeezed before the
    #: layout would rather drop a column. 1.0 never squeezes, which on Windows metrics means
    #: one column on a 1280 px screen; 0.0 squeezes to the hard floor, giving three cramped
    #: columns on a laptop. 0.75 is the measured knee: two columns on a 1280 px screen with
    #: every card still above its own minimum, so only the two widest ROWS wrap.
    TIGHTEN = 0.75

    def __init__(self, parent=None, *, hgap: int = 16, vgap: int = 12, max_columns: int = 3,
                 margin: int = 0, tighten: float | None = None):
        super().__init__(parent)
        self._items: list = []              # keep Python refs: Qt does not own these
        self._hgap, self._vgap = hgap, vgap
        self._max_columns = max_columns
        self._tighten = self.TIGHTEN if tighten is None else tighten
        self._plan_cache: dict = {}
        self.setContentsMargins(margin, margin, margin, margin)

    # -- QLayout plumbing ----------------------------------------------------
    def addItem(self, item):                # noqa: N802 - Qt API
        self._items.append(item)
        self._plan_cache.clear()

    def count(self):
        return len(self._items)

    def itemAt(self, index):                # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):                # noqa: N802 - Qt API
        self._plan_cache.clear()
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def invalidate(self):
        self._plan_cache.clear()
        super().invalidate()

    def expandingDirections(self):          # noqa: N802 - Qt API
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):            # noqa: N802 - Qt API
        return True

    # -- the responsive contract ---------------------------------------------
    def _shown(self):
        """Only visible cards take part, so a hidden section leaves no hole in a column."""
        out = []
        for it in self._items:
            w = it.widget()
            if w is not None and w.isHidden():
                continue
            out.append(it)
        return out

    @staticmethod
    def _quantile(values, q: float) -> float:
        xs = sorted(values)
        if not xs:
            return 0
        pos = q * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

    def _comfort_width(self, items) -> int:
        """The width that suits MOST cards: the upper quartile of their natural widths.

        Deliberately not the maximum. One section of the mechanics modal is an outlier by
        content — a 42-character caption beside a spin box sized by its 31-character
        special-value text — and at Windows metrics it wants 1014 px against a 559 px upper
        quartile. Letting the maximum decide made every column 760 px wide, so two columns
        needed a 1536 px viewport and a 1080p laptop got one column and 1304 px of scrolling.
        Sizing to the quartile costs that one card a wrapped row (which is what
        ``WrapLongRows`` is for) and buys a second column on every screen we ship to.
        """
        return int(self._quantile([it.sizeHint().width() for it in items], 0.75))

    def _floor_width(self, items) -> int:
        return max((it.minimumSize().width() for it in items), default=0)

    def column_target(self, items=None) -> int:
        """The width a column wants — derived entirely from measured widget metrics, so it
        says the right thing on macOS and on Windows without a pixel literal anywhere."""
        items = self._shown() if items is None else items
        if not items:
            return 1
        return max(self._floor_width(items),
                   int(self._comfort_width(items) * self._tighten), 1)

    def columns_for(self, width: int) -> int:
        """How many columns ``width`` affords. Public because it is what the guards assert."""
        items = self._shown()
        if not items:
            return 0
        target = self.column_target(items)
        n = (width + self._hgap) // (target + self._hgap)
        return max(1, min(int(n), self._max_columns, len(items)))

    def _plan(self, width: int):
        """``(total height, column width, runs)`` at ``width``.

        Cached by integer width: ``heightForWidth`` is called several times per layout pass
        and every call measures every card.
        """
        key = int(width)
        hit = self._plan_cache.get(key)
        if hit is not None:
            return hit
        items = self._shown()
        m = self.contentsMargins()
        avail = max(0, width - m.left() - m.right())
        if not items:
            return (m.top() + m.bottom(), avail, [])
        k = self.columns_for(avail)
        colw = (avail - self._hgap * (k - 1)) // k
        # Below the widest card's own minimum, stop shrinking and let the scroll area grow a
        # horizontal bar instead. Laying a card out narrower than its minimum does not clip
        # it — Qt paints it past the edge, which is the original bug one level in.
        colw = max(colw, self._floor_width(items))
        heights = [it.heightForWidth(colw) if it.hasHeightForWidth()
                   else it.sizeHint().height() for it in items]
        heights = [max(h, it.minimumSize().height()) for h, it in zip(heights, items)]
        runs = partition_columns(heights, k, self._vgap)
        total = 0
        for i, j in runs:
            total = max(total, sum(heights[i:j]) + self._vgap * (j - i - 1))
        out = (total + m.top() + m.bottom(), colw, runs)
        self._plan_cache[key] = out
        return out

    def heightForWidth(self, width):        # noqa: N802 - Qt API
        return self._plan(width)[0]

    def sizeHint(self):                     # noqa: N802 - Qt API
        items = self._shown()
        if not items:
            return QSize(0, 0)
        k = max(1, min(self._max_columns, len(items)))
        target = self.column_target(items)
        m = self.contentsMargins()
        w = k * target + self._hgap * (k - 1) + m.left() + m.right()
        return QSize(w, self.heightForWidth(w))

    def minimumSize(self):                  # noqa: N802 - Qt API
        """Widest single card, and NO height — see the class docstring; both halves matter."""
        items = self._shown()
        m = self.contentsMargins()
        return QSize(self._floor_width(items) + m.left() + m.right(), 0)

    # -- placement -----------------------------------------------------------
    def setGeometry(self, rect):            # noqa: N802 - Qt API
        super().setGeometry(rect)
        items = self._shown()
        if not items:
            return
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        _total, colw, runs = self._plan(rect.width())
        for c, (i, j) in enumerate(runs):
            x = eff.x() + c * (colw + self._hgap)
            y = eff.y()
            for it in items[i:j]:
                h = (it.heightForWidth(colw) if it.hasHeightForWidth()
                     else it.sizeHint().height())
                h = max(h, it.minimumSize().height())
                it.setGeometry(QRect(QPoint(x, y), QSize(colw, h)))
                y += h + self._vgap


def install_sections(widget, *, hgap: int = 16, vgap: int = 12, max_columns: int = 3,
                     margins: tuple = (0, 0, 0, 0)) -> SectionColumns:
    """Give ``widget`` a :class:`SectionColumns` and let its HEIGHT follow its width.

    The second half is the part that is easy to miss, and it is the same trap
    ``flow_layout.install_flow`` documents: under ``QSizePolicy.Maximum`` Qt caps the widget
    at its one-column ``sizeHint`` height, lays the rest out correctly, and then paints it
    outside the widget.
    """
    lay = SectionColumns(hgap=hgap, vgap=vgap, max_columns=max_columns)
    lay.setContentsMargins(*margins)
    widget.setLayout(lay)
    sp = widget.sizePolicy()
    sp.setVerticalPolicy(QSizePolicy.Preferred)
    sp.setHeightForWidth(True)
    widget.setSizePolicy(sp)
    return lay

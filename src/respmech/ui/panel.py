"""Shared titled-panel chrome and page-fit scrolling (ticket B03, UI-overhaul).

Lifted out of ``PreviewScreen`` unchanged, so a future caller can reuse the exact same
"titled panel" header treatment and short-screen scrolling Preview & QC already uses for
its plots, instead of inventing (or, worse, forgetting to reuse) either one. Existing
Preview & QC call sites (``_titled``, ``_scrollable``/``_PageScroll``) now delegate here.

The Run drawer (``RunScreen``, embedded by ``PreviewScreen.install_run_drawer``) uses
``titled_panel`` for its own "Run log" panel, but deliberately does NOT wrap in
``scrollable()``/``FitScrollArea`` — see that class's own docstring for why a page whose
content collapses/expands at runtime is exactly the case ``FitScrollArea`` is wrong for.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from respmech.ui.flow_layout import ElidingLabel


def titled_panel(title, widget, corner=None):
    """A titled panel. ``corner`` is an optional widget pinned to the top-right of
    the title row (e.g. the detail-channel dropdown or the result-channel picker)."""
    # The panel's own margin is the single source of air: it keeps the title, the plot
    # and the right-pinned corner widget (channel selectors) all off the panel edge, so
    # nothing butts against the neighbouring panel or the nav.
    box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(8, 4, 8, 6); lay.setSpacing(2)
    header = QHBoxLayout(); header.setContentsMargins(0, 0, 0, 0); header.setSpacing(8)
    # ElidingLabel, not a plain QLabel: several panels keep this header current with a
    # live verdict via setFullText() — a fixed one-shot elide() cap can end up narrower
    # than even the panel's own STATIC title once the panel sits in a splitter column,
    # and never widens back out when the user gives it more room. This sizes to whatever
    # width it is actually given and always keeps the full text one hover away.
    lab = ElidingLabel(title); lab.setProperty("status", "muted")
    box._title_label = lab          # so a panel can keep its header current
    header.addWidget(lab); header.addStretch(1)
    if corner is not None:
        header.addWidget(corner, 0, Qt.AlignRight | Qt.AlignVCenter)
    lay.addLayout(header); lay.addWidget(widget, 1)
    return box


class FitScrollArea(QScrollArea):
    """A scroll area whose page is the VIEWPORT height when there is room for it, and its
    own minimum height when there is not.

    Qt's ``widgetResizable`` sizes the page from ``heightForWidth`` whenever the page's
    layout reports one — and Preview & QC's wrapping chip rows do. So the page came out at
    its NATURAL height on every screen (measured 1659 px for the noise page) instead of
    filling the viewport, and the bottom diagnostics row sat below the fold even on a
    2560x1400 display, where the whole page had fitted before. Scrolling is meant to be
    the answer to a SHORT screen, not a permanent tax on every screen.

    NOT used for the B03 Run drawer, despite that being the ticket that measured the
    problem this class solves (embedding all of Run & results permanently under Preview &
    QC's own content pushed the combined tab's minimum height to 611 px against a 547 px
    budget). Tried first, and reverted: this class's own fit-on-resize logic assumes the
    page's NATURAL height is fixed once built (true for Preview & QC's static subtab
    pages), and fights a page whose content the user can collapse/expand at runtime — an
    early fit pass against the page's larger pre-collapse size got stuck in this scroll
    area's own reported size and fed back into the outer layout, leaving the drawer at
    362 px even fully collapsed (worse than not using this at all). The drawer's own
    collapsed-by-default toggle (``RunScreen._results_section``) solves the same budget
    problem without needing a second, conflicting sizing mechanism on top.
    """

    def resizeEvent(self, ev):          # noqa: N802 - Qt API
        super().resizeEvent(ev)
        self._fit_page()

    def showEvent(self, ev):            # noqa: N802 - Qt API
        # A page that was not the current tab is laid out only when it is first shown,
        # and that pass happens after the filter below has had its chance — so without
        # this the tab the user switches TO keeps the natural height Qt gave it.
        super().showEvent(ev)
        self._fit_page()

    def setWidget(self, page):          # noqa: N802 - Qt API
        # widgetResizable is deliberately OFF (see the class docstring): with it on, Qt
        # sizes the page from its layout's heightForWidth and that decision is final —
        # overriding resizeEvent, filtering LayoutRequest and even calling resize()
        # directly afterwards all left the page at its natural height, measured.
        # So the page's size is this class's own job, recomputed on the three events that
        # can change it.
        super().setWidget(page)
        page.installEventFilter(self)
        self._fit_page()

    def eventFilter(self, obj, ev):     # noqa: N802 - Qt API
        if obj is self.widget() and ev.type() == QEvent.LayoutRequest:
            self._fit_page()
        return super().eventFilter(obj, ev)

    _fitting = False

    def _fit_page(self):
        # Re-entrancy guard AND a no-op check, both required: resizing the page emits the
        # very LayoutRequest this is called from, and pyqtgraph re-lays its axes out on
        # every resize, so without both this recursed until the stack blew.
        if self._fitting:
            return
        page = self.widget()
        if page is None:
            return
        vp = self.viewport().size()
        want = QSize(vp.width(), max(vp.height(), page.minimumSizeHint().height()))
        if page.size() == want:
            return
        self._fitting = True
        try:
            page.resize(want)
        finally:
            self._fitting = False


def scrollable(page, *, horizontal_policy=Qt.ScrollBarAsNeeded):
    """Wrap ``page`` in a :class:`FitScrollArea` (vertical scroll on a short screen; the
    page still expands to fill a tall window, so nothing changes on a large screen)."""
    area = FitScrollArea()
    area.setWidgetResizable(False)      # FitScrollArea sizes the page itself
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(page)
    area.setHorizontalScrollBarPolicy(horizontal_policy)
    return area

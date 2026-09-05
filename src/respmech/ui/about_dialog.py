"""The About box (C01, UI-overhaul) — self-bearing offline: version, author, licence
and citation, with both project URLs shown as plain, selectable text so they can be
copied without a browser (the app makes no network calls of its own; see Help menu
for the clickable equivalents that hand off to the system browser)."""
from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from respmech import __citation_year__, __version__

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

WEBSITE_URL = "https://www.respmech.dk"
GITHUB_URL = "https://github.com/emilwalsted/respmech"
AUTHOR = "Emil Ingerslev Walsted"


def _selectable_label(text: str) -> QLabel:
    """Plain, copyable text — no hyperlink, no browser round-trip required."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lab.setCursor(Qt.IBeamCursor)
    return lab


class AboutDialog(QDialog):
    """A small, self-contained "About RespMech" box. Every fact it shows (version,
    licence, citation, URLs) is baked into the app itself, so it stays fully usable
    with no network access."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About RespMech")
        self.setModal(True)
        # Parented to MainWindow but never reused — without this, every Help > About
        # RespMech… accept()/close() only HIDES the dialog, never destroys it, and one
        # more QDialog (with its labels/pixmap) accumulates under MainWindow per open.
        self.setAttribute(Qt.WA_DeleteOnClose)

        v = QVBoxLayout(self)
        v.setContentsMargins(28, 24, 28, 20)
        v.setSpacing(10)

        try:
            from respmech.ui.logo import logo_pixmap
            pm = logo_pixmap(72)
        except Exception:                        # pragma: no cover - logo is cosmetic
            pm = None
        if pm is not None and not pm.isNull():
            icon_lab = QLabel()
            icon_lab.setPixmap(pm)
            icon_lab.setAlignment(Qt.AlignHCenter)
            v.addWidget(icon_lab)

        title = QLabel(f"RespMech {__version__}")
        title.setProperty("role", "heading")
        title.setAlignment(Qt.AlignHCenter)
        v.addWidget(title)

        sub = QLabel("Respiratory mechanics · work of breathing · diaphragm EMG")
        sub.setProperty("status", "muted")
        sub.setAlignment(Qt.AlignHCenter)
        sub.setWordWrap(True)
        v.addWidget(sub)
        v.addSpacing(4)

        v.addWidget(_selectable_label(f"© 2019–2026 {AUTHOR}"))
        v.addWidget(_selectable_label(
            "This program is free software: you can redistribute it and/or modify it "
            "under the terms of the GNU General Public License as published by the "
            "Free Software Foundation, either version 3 of the License, or (at your "
            "option) any later version. It is distributed WITHOUT ANY WARRANTY; "
            "without even the implied warranty of MERCHANTABILITY or FITNESS FOR A "
            "PARTICULAR PURPOSE."))

        cite_head = QLabel("Citation")
        cite_head.setProperty("status", "muted")
        v.addSpacing(6)
        v.addWidget(cite_head)
        v.addWidget(_selectable_label(
            f"E Walsted, RespMech v{__version__}, {__citation_year__}, {GITHUB_URL}/, "
            "DOI: 10.5281/zenodo.3270825 (for a specific version's own DOI, see the "
            "Zenodo record linked from the README)"))

        v.addSpacing(6)
        v.addWidget(_selectable_label(WEBSITE_URL))
        v.addWidget(_selectable_label(GITHUB_URL))

        row = QHBoxLayout()
        row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        if _theme is not None:
            _theme.make_primary(close_btn)
        row.addWidget(close_btn)
        v.addLayout(row)

        self.setMinimumWidth(420)
        self.adjustSize()

    def showEvent(self, ev):                # noqa: N802 - Qt API
        """Open at the requested size, never past the screen — the same guard every
        other dialog in this codebase uses (see e.g. startup_dialog.py's own comment:
        adjustSize() alone reached 766 px on Windows font metrics against 650 px of
        usable height on a 1080p laptop at 150% scaling)."""
        super().showEvent(ev)
        if not getattr(self, "_clamped", False):
            self._clamped = True
            from respmech.ui import screen_fit  # noqa: PLC0415
            screen_fit.clamp_to_screen(self, prefer=QSize(460, self.height()))

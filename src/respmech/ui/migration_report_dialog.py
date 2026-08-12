"""The 1.x -> 2.x migration summary shown right after a legacy analysis is imported
(ticket D16). Unlike ``dialogs.TextViewerDialog`` (a monospace, unwrapped, NOT-wrapped
copyable TRACE viewer built for error detail), a migration report is not an error: word
wrap is on, the font is the ordinary proportional UI font, and the report's three
sections render as real headings instead of the literal ``#``/``##`` markdown
``MigrationReport.text()`` produces for a human reading it in a text editor.

The "Normalised" section (settings that changed BEHAVIOUR, not just name/location) sits
right under the intro paragraph, since that is the one thing a user actually needs to
read before trusting the converted analysis. The pure renames ("Mapped"/"Dropped") are
bookkeeping — correct, but not something anyone reads unless they are specifically
troubleshooting a field — so they sit behind a "Show the full field-by-field mapping"
disclosure, collapsed by default.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from respmech.settingsio.migrate import MigrationReport

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None


def _selectable(lab: QLabel) -> QLabel:
    """Plain, copyable text — matches about_dialog.py's ``_selectable_label``: a user
    who only wants one line (not the whole file via "Save report…") can select it."""
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lab.setCursor(Qt.IBeamCursor)
    return lab


def _heading(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.PlainText)
    # role="heading" (theme.py's QLabel[role="heading"] rule), not hand-rolled
    # bold/pointSize()+1 — the latter breaks on a platform where the app font has no
    # point size (theme.apply_theme falls back to setPixelSize there, so pointSize()
    # returns -1 and +1 would compute an invalid size 0), and drifts from any future
    # palette/typography change the rest of the app picks up for free.
    lab.setProperty("role", "heading")
    return _selectable(lab)


def _bullets(items: list[str]) -> QLabel:
    lab = QLabel("\n".join(f"•  {i}" for i in items) if items else "(none)")
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.PlainText)
    if not items:
        lab.setProperty("status", "muted")
    return _selectable(lab)


class MigrationReportDialog(QDialog):
    """Structured, wrapped presentation of a :class:`MigrationReport`. Held by the
    caller (see ``open_migration_report`` below) and replaced, never accumulated, on a
    repeat import — the same "prior=" convention ``dialogs.open_error_dialog`` uses."""

    def __init__(self, report: MigrationReport, source_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Migration report")
        self._report = report
        self._source_path = source_path

        lay = QVBoxLayout(self)

        intro = QLabel(self._intro_text())
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.PlainText)
        lay.addWidget(_selectable(intro))

        if report.normalised:
            lay.addWidget(_heading("Normalised (drift fixed)"))
            lay.addWidget(_bullets(report.normalised))

        self._toggle_btn = QPushButton("Show the full field-by-field mapping")
        self._toggle_btn.setAutoDefault(False)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.toggled.connect(self._on_toggle)
        lay.addWidget(self._toggle_btn)

        full_box = QWidget()
        full_lay = QVBoxLayout(full_box)
        full_lay.setContentsMargins(0, 0, 0, 0)
        full_lay.addWidget(_heading("Mapped (renamed/moved)"))
        full_lay.addWidget(_bullets(report.mapped))
        full_lay.addWidget(_heading("Dropped (not used by the code)"))
        full_lay.addWidget(_bullets(report.dropped))
        full_lay.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(full_box)
        self._scroll.setVisible(False)
        lay.addWidget(self._scroll, 1)

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save report…")
        self.save_btn.setAutoDefault(False)
        self.save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)              # Enter dismisses; single emphasised action
        close_btn.clicked.connect(self.accept)
        if _theme is not None:
            _theme.make_primary(close_btn)
        row.addStretch(1)
        row.addWidget(self.save_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)

        self.setMinimumSize(520, 360)
        self.resize(620, 480)

    def _intro_text(self) -> str:
        """A "Normalised" entry means the migrator changed behaviour to fix a drift
        between what a legacy example file said and what the legacy CODE actually read
        (see migrate.py's own module docstring) — by definition, not merely a rename.
        Re-running the SAME recording therefore will not exactly reproduce a v1.x
        number for any of the items listed below, and the docstring naming this ticket
        (D16) exists specifically because that consequence — not just the technical
        "what changed" — was previously easy to miss."""
        base = "Your settings were converted."
        if self._report.normalised:
            return (f"{base} Re-running this analysis will not exactly reproduce "
                    "RespMech 1.x's numbers: some settings needed more than a rename "
                    "to keep behaving the same way, listed below. Everything else "
                    "(not listed below) is a straightforward rename.")
        return (f"{base} Every change here is a straightforward rename — re-running "
                "this analysis reproduces RespMech 1.x's numbers exactly.")

    def _on_toggle(self, checked):
        self._scroll.setVisible(checked)
        self._toggle_btn.setText(
            "Hide the full field-by-field mapping" if checked
            else "Show the full field-by-field mapping")

    def _save(self):
        # ticket D16 ordered this not to touch the ANALYSIS Save-as… default (a
        # different dialog with its own "analysis" prefs bucket) — a distinct
        # "migration_report" bucket keeps this dialog's own P26 sticky-folder memory
        # without going anywhere near that one.
        from respmech.ui import prefs  # noqa: PLC0415
        start_dir = prefs.last_folder(
            "migration_report",
            os.path.dirname(self._source_path) if self._source_path else ".")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save migration report",
            os.path.join(start_dir, "migration-report.txt"), "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._report.text())
        except OSError as e:
            QMessageBox.warning(self, "Save migration report",
                                f"Could not save the report: {e}")
            return
        prefs.set_last_folder("migration_report", os.path.dirname(path))

    def showEvent(self, ev):                # noqa: N802 - Qt API
        """Open at the requested size, never past the screen — the same guard every
        other dialog in this codebase uses (see about_dialog.py's own comment)."""
        super().showEvent(ev)
        if not getattr(self, "_clamped", False):
            self._clamped = True
            from respmech.ui import screen_fit  # noqa: PLC0415
            screen_fit.clamp_to_screen(self, prefer=QSize(620, 480))


def open_migration_report(parent, report: MigrationReport, source_path: str | None = None,
                          prior=None) -> MigrationReportDialog:
    """Show the migration report, replacing ``prior`` if given so a repeat import does
    not accumulate windows — the same convention as ``dialogs.open_error_dialog``."""
    if prior is not None:
        try:
            prior.close()
            prior.deleteLater()
        except Exception:                          # pragma: no cover - prior already gone
            pass
    dlg = MigrationReportDialog(report, source_path=source_path, parent=parent)
    dlg.show()
    dlg.raise_()
    return dlg

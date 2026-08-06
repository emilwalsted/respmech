"""Shared UI state: the current settings object, plus load/save/migrate helpers.

Kept free of Qt so it is unit-testable without a display."""
from __future__ import annotations

from respmech.core.settings import Settings
from respmech.settingsio.migrate import migrate_file
from respmech.settingsio.toml_io import load_toml, save_toml


class AppState:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings_path: str | None = None
        # ticket C03 point 7/8: name an analysis that has no settings_path of its own yet
        # (an imported legacy .py, or the built-in sample) in the window title, instead of
        # the generic "new analysis" that made either indistinguishable from a blank one.
        self.display_name: str | None = None
        # the .py an imported analysis came from, so a re-import/Save-as flow can still
        # name its origin even after the migration report dialog has been dismissed.
        self.legacy_source_path: str | None = None
        # True only while the built-in sample (open_sample_analysis) is the loaded
        # analysis; cleared by every other load/import/new/save path below.
        self.is_sample: bool = False

    def load_toml(self, path: str):
        self.settings = load_toml(path)
        self.settings_path = path
        self.display_name = None
        self.legacy_source_path = None
        self.is_sample = False

    def save_toml(self, path: str):
        save_toml(self.settings, path)
        self.settings_path = path
        # The upgrade has now been persisted, so "upgraded since this analysis was saved"
        # is no longer true of the file on disk — stop prepending it to every run report.
        self.settings.notices = []
        # A save gives the analysis a real path (and, for an imported/sample analysis,
        # turns it into an ordinary saved one) — the title should name that file from now
        # on, not the origin it no longer needs.
        self.display_name = None
        self.legacy_source_path = None
        self.is_sample = False

    def import_legacy(self, path: str) -> str:
        self.settings, report = migrate_file(path)
        self.settings_path = None
        self.legacy_source_path = path
        import os  # noqa: PLC0415
        self.display_name = os.path.splitext(os.path.basename(path))[0]
        self.is_sample = False
        return report.text()

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

    def load_toml(self, path: str):
        self.settings = load_toml(path)
        self.settings_path = path

    def save_toml(self, path: str):
        save_toml(self.settings, path)
        self.settings_path = path

    def import_legacy(self, path: str) -> str:
        self.settings, report = migrate_file(path)
        self.settings_path = None
        return report.text()

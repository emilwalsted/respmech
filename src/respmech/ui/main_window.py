"""Main window — a tabbed shell holding the three screens, sharing one AppState."""
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from respmech import __version__
from respmech.ui.state import AppState
from respmech.ui.screens.settings_screen import SettingsScreen
from respmech.ui.screens.preview_screen import PreviewScreen
from respmech.ui.screens.run_screen import RunScreen


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None):
        super().__init__()
        self.state = state or AppState()
        self.setWindowTitle(f"RespMech {__version__}")
        self.resize(1100, 800)

        self.tabs = QTabWidget()
        self.settings_screen = SettingsScreen(self.state, on_settings_changed=self._on_settings_changed)
        self.preview_screen = PreviewScreen(self.state)
        self.run_screen = RunScreen(self.state)
        self.tabs.addTab(self.settings_screen, "1. Settings / batch setup")
        self.tabs.addTab(self.preview_screen, "2. Preview & tuning")
        self.tabs.addTab(self.run_screen, "3. Run")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    def _on_tab_changed(self, index):
        # Push the settings form into the shared state before preview/run use it.
        self.settings_screen.to_state()
        if self.tabs.widget(index) is self.preview_screen:
            self.preview_screen._refresh_files()

    def _on_settings_changed(self):
        pass

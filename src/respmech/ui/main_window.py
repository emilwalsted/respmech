"""Main window — a tabbed shell holding the three screens, sharing one AppState.

The Settings screen is the single source of change events; the main window is the
only place cross-screen signals are wired, so the Qt-free ``AppState`` never grows
Qt dependencies. A status bar mirrors each screen's status line.
"""
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
        self.resize(1180, 820)

        self.tabs = QTabWidget()
        self.settings_screen = SettingsScreen(self.state, on_settings_changed=self._on_settings_changed)
        self.preview_screen = PreviewScreen(self.state)
        self.run_screen = RunScreen(self.state)
        self.tabs.addTab(self.settings_screen, "1. Settings / batch setup")
        self.tabs.addTab(self.preview_screen, "2. Preview & tuning")
        self.tabs.addTab(self.run_screen, "3. Run")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        sc, pv, rn = self.settings_screen, self.preview_screen, self.run_screen
        sc.inputs_changed.connect(pv.refresh_files)
        sc.settings_changed.connect(pv.sync_from_settings)
        sc.settings_changed.connect(rn.refresh_actions)
        bar = self.statusBar()          # creates the status bar (offscreen-safe)
        for scr in (sc, pv, rn):
            scr.status_changed.connect(lambda msg, b=bar: b.showMessage(msg))
        bar.showMessage("Ready.")

    def _on_tab_changed(self, index):
        self.settings_screen.to_state()             # belt-and-suspenders sync
        w = self.tabs.widget(index)
        if w is self.preview_screen:
            self.preview_screen.refresh_files()
        elif w is self.run_screen:
            self.run_screen.refresh_actions()

    def _on_settings_changed(self):
        pass

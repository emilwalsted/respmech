"""Main window — a tabbed shell holding the three screens, sharing one AppState.

The Settings screen is the single source of change events; the main window is the
only place cross-screen signals are wired, so the Qt-free ``AppState`` never grows
Qt dependencies. A status bar mirrors each screen's status line.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QTabWidget, QVBoxLayout, QWidget)

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
        try:
            from respmech.ui.logo import app_icon
            icon = app_icon()
            if icon is not None:
                self.setWindowIcon(icon)
        except Exception:                       # pragma: no cover - icon is cosmetic
            pass
        self._fit_to_screen()

        self.tabs = QTabWidget()
        self.settings_screen = SettingsScreen(self.state, on_settings_changed=self._on_settings_changed)
        self.preview_screen = PreviewScreen(self.state)
        self.run_screen = RunScreen(self.state)
        self.tabs.addTab(self.settings_screen, "1. Settings / batch setup")
        self.tabs.addTab(self.preview_screen, "2. Preview & tuning")
        self.tabs.addTab(self.run_screen, "3. Run")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._make_header())
        col.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        sc, pv, rn = self.settings_screen, self.preview_screen, self.run_screen
        sc.inputs_changed.connect(pv.refresh_files)
        sc.settings_changed.connect(pv.sync_from_settings)
        sc.settings_changed.connect(rn.refresh_actions)
        # a noise reference chosen on the Preview graph (feature B) mirrors into Settings
        pv.noise_reference_changed.connect(sc.set_noise_reference)
        # while a batch runs, lock the Settings screen so its state can't be swapped
        # (e.g. Load TOML) out from under the running worker
        rn.run_started.connect(self._on_run_started)
        rn.run_finished.connect(self._on_run_finished)
        bar = self.statusBar()          # creates the status bar (offscreen-safe)
        for scr in (sc, pv, rn):
            scr.status_changed.connect(lambda msg, b=bar: b.showMessage(msg))
        bar.showMessage("Ready.")

    def _on_run_started(self):
        self.settings_screen.setEnabled(False)

    def _on_run_finished(self):
        self.settings_screen.setEnabled(True)

    def _make_header(self) -> QFrame:
        """A calm application header bar (title + subtitle) above the tabs."""
        header = QFrame()
        header.setObjectName("appHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 10, 18, 10)
        title = QLabel("RespMech")
        title.setObjectName("appTitle")
        sub = QLabel("Respiratory mechanics · work of breathing · diaphragm EMG")
        sub.setObjectName("appSubtitle")
        h.addWidget(title)
        h.addSpacing(12)
        h.addWidget(sub)
        h.addStretch(1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("appSubtitle")
        h.addWidget(ver)
        return header

    def _fit_to_screen(self, desired_w: int = 1180, desired_h: int = 820,
                       fraction: float = 0.92) -> None:
        """Size the window to a sensible fraction of the available screen and
        centre it. Falls back to the desired size when no usable screen geometry
        is reported (e.g. an unusual headless setup), so it never raises."""
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        if avail is None or avail.width() <= 0 or avail.height() <= 0:
            self.resize(desired_w, desired_h)
            return
        w = min(desired_w, int(avail.width() * fraction))
        h = min(desired_h, int(avail.height() * fraction))
        self.resize(w, h)
        frame = self.frameGeometry()          # centre within the work area
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

    def _on_tab_changed(self, index):
        self.settings_screen.to_state()             # belt-and-suspenders sync
        w = self.tabs.widget(index)
        if w is self.preview_screen:
            self.preview_screen.refresh_files()
        elif w is self.run_screen:
            self.run_screen.refresh_actions()

    def _on_settings_changed(self):
        pass

    def closeEvent(self, ev):
        # join every worker thread (Preview reactive jobs + a running batch) so none
        # outlives the window and gets destroyed while running
        for screen in (self.preview_screen, self.run_screen):
            try:
                screen.shutdown()
            except Exception:               # pragma: no cover - best-effort teardown
                pass
        super().closeEvent(ev)

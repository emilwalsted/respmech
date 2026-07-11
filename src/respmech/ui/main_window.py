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
        # No wizard "1./2./3." numbering — the real workflow loops back (tune -> re-run),
        # it is not one-way. ("&&" renders a literal "&"; a lone "&" is a tab mnemonic.)
        self._i_settings = self.tabs.addTab(self.settings_screen, "Setup")
        self._i_preview = self.tabs.addTab(self.preview_screen, "Preview && QC")
        self._i_run = self.tabs.addTab(self.run_screen, "Run && results")
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
        # the guided "new analysis" flow hides Preview/Run until every setting is valid
        sc.flow_ready_changed.connect(self._set_downstream_visible)
        # the window title names the active analysis and its unsaved-edits (dirty) state
        sc.analysis_state_changed.connect(self._update_window_title)
        self._update_window_title()
        # a noise reference chosen on the Preview graph (feature B) mirrors into Settings
        pv.noise_reference_changed.connect(sc.set_noise_reference)
        # P19: "Process & write this file" on Preview → run just that file on the Run screen
        pv.process_file_requested.connect(self._process_single_file)
        # P20: double-clicking a file in the Run results drills back into Preview
        rn.open_file_requested.connect(self._open_file_in_preview)
        # while a batch runs, lock the Settings screen so its state can't be swapped
        # (e.g. Load TOML) out from under the running worker
        rn.run_started.connect(self._on_run_started)
        rn.run_finished.connect(self._on_run_finished)
        bar = self.statusBar()          # creates the status bar (offscreen-safe)
        for scr in (sc, pv, rn):
            scr.status_changed.connect(lambda msg, b=bar: b.showMessage(msg))
        bar.showMessage("Ready.")

    def begin_session(self, cli_path: str | None = None):
        """Run the startup flow once the window is on screen: if an analysis file was
        passed on the command line it is already loaded, so open it (all cards + tabs);
        otherwise ask the user to start a New analysis (guided) or Open an existing one.
        Not called from ``__init__`` so headless/test construction keeps full access."""
        if cli_path:
            self.settings_screen.enter_open_mode()
            return
        from respmech.ui.startup_dialog import StartupDialog
        dlg = StartupDialog(self)
        dlg.exec()
        # a failed open (corrupt file) falls through to the guided New flow rather than
        # leaving the user in full mode over rolled-back defaults
        if dlg.mode == "open" and dlg.path and self.settings_screen.open_analysis(dlg.path):
            return
        self.settings_screen.enter_new_mode()

    def _set_downstream_visible(self, ready: bool):
        """Show/hide the Preview & Run tabs together (they both need valid settings)."""
        self.tabs.setTabVisible(self._i_preview, bool(ready))
        self.tabs.setTabVisible(self._i_run, bool(ready))

    def _update_window_title(self):
        """Name the active analysis in the title bar and flag unsaved edits with a bullet,
        so it is obvious which analysis is loaded and whether it has been saved."""
        import os  # noqa: PLC0415
        path = getattr(self.state, "settings_path", None)
        name = os.path.basename(path) if path else "new analysis (unsaved)"
        dirty = " •" if self.settings_screen.is_dirty() else ""
        self.setWindowTitle(f"RespMech {__version__} — {name}{dirty}")

    def _process_single_file(self, filename: str):
        """P19: switch to the Run screen and process+write just the previewed file."""
        self.tabs.setCurrentIndex(self._i_run)
        self.run_screen.run_single_file(filename)

    def _open_file_in_preview(self, filename: str):
        """P20: drill back from the Run results into Preview & QC for one file."""
        self.tabs.setCurrentIndex(self._i_preview)
        self.preview_screen.refresh_files()
        try:
            self.preview_screen.file_combo.setCurrentText(filename)
        except Exception:                       # pragma: no cover - best-effort selection
            pass

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

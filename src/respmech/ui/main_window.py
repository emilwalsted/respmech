"""Main window — a tabbed shell holding the three screens, sharing one AppState.

The Settings screen is the single source of change events; the main window is the
only place cross-screen signals are wired, so the Qt-free ``AppState`` never grows
Qt dependencies. A status bar mirrors each screen's status line.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMenu, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from respmech import __version__
from respmech.ui.state import AppState
from respmech.ui.screens.settings_screen import SettingsScreen
from respmech.ui.screens.preview_screen import PreviewScreen
from respmech.ui.screens.run_screen import RunScreen


# The menu shows fewer recents than prefs stores: a header menu has to stay glanceable.
_MENU_RECENTS = 5


def _recent_label(path: str, max_dir: int = 34) -> str:
    """Label one recent analysis: the file name plus just enough of its folder to tell two
    same-named analyses apart. Bare basenames collide across studies (every study has an
    "analysis.toml"); full paths are unreadable at menu width. The folder is elided at its
    HEAD — the leaf folder is the one that names the study."""
    import os  # noqa: PLC0415
    folder, name = os.path.split(path)
    home = os.path.expanduser("~")
    if folder.startswith(home):
        folder = "~" + folder[len(home):]
    if len(folder) > max_dir:
        folder = "…" + folder[-max_dir:]
    return f"{name}  —  {folder}" if folder else name


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
        # the guided "new analysis" flow LOCKS (does not hide) Preview/Run until every
        # setting is valid, so the whole workflow is visible as a stepper (P24)
        sc.flow_ready_changed.connect(self._set_downstream_enabled)
        # the window title names the active analysis and its unsaved-edits (dirty) state
        sc.analysis_state_changed.connect(self._update_window_title)
        self._update_window_title()
        # a noise reference chosen on the Preview graph (feature B) mirrors into Settings
        pv.noise_reference_changed.connect(sc.set_noise_reference)
        # Preview-owned settings (noise/ECG params, breath exclusions) land in the saved
        # .toml too, so a user edit there must dirty the analysis like any Setup edit —
        # the title, the close guard and Save-gating all read the same flag.
        pv.settings_edited.connect(sc._mark_dirty)
        # A Setup caution can depend on a field the Preview owns (the gated peak needs
        # remove_ecg, written by the ECG tab), so the strip has to be recomputed when Preview
        # edits the model — not only when Setup's own widgets change.
        pv.settings_edited.connect(sc.refresh_qc)
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
        if dlg.mode == "sample" and self.settings_screen.open_sample_analysis():   # P23
            return
        # P25: "New from last rig" inherits the last channel mapping into the guided flow
        self.settings_screen.enter_new_mode(use_last_rig=(dlg.mode == "new_rig"))

    def _set_downstream_enabled(self, ready: bool):
        """Lock/unlock the Preview & Run tabs together (they both need valid settings).
        They stay *visible* so the user always sees the three steps of the workflow;
        a tooltip on a locked step says why it can't be entered yet (P24)."""
        ready = bool(ready)
        hint = "" if ready else "Complete the Setup to unlock this step."
        for i in (self._i_preview, self._i_run):
            self.tabs.setTabEnabled(i, ready)
            self.tabs.setTabToolTip(i, hint)

    def _update_window_title(self):
        """Name the active analysis in the title bar and flag unsaved edits, so it is
        obvious which analysis is loaded and whether it has been saved. The unsaved
        analysis is named "new analysis", not "new analysis (unsaved)": the modified
        marker already carries that, and both would say it twice."""
        import os  # noqa: PLC0415
        path = getattr(self.state, "settings_path", None)
        name = os.path.basename(path) if path else "new analysis"
        dirty = " * (modified)" if self.settings_screen.is_dirty() else ""
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
        # the Analysis menu lives in the header, OUTSIDE the screen being locked, so it
        # needs locking too — otherwise Open/New could swap the settings out from under
        # the running worker, which is exactly what disabling the screen prevents.
        self.analysis_btn.setEnabled(False)

    def _on_run_finished(self):
        self.settings_screen.setEnabled(True)
        self.analysis_btn.setEnabled(True)

    def _make_header(self) -> QFrame:
        """A calm application header bar (brand · Analysis menu · subtitle) above the tabs."""
        header = QFrame()
        header.setObjectName("appHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 10, 18, 10)
        title = QLabel("RespMech")
        title.setObjectName("appTitle")
        sub = QLabel("Respiratory mechanics · work of breathing · diaphragm EMG")
        sub.setObjectName("appSubtitle")
        h.addWidget(title)
        h.addSpacing(16)
        h.addWidget(self._make_analysis_menu())     # right of the brand, on every tab
        h.addSpacing(16)
        h.addWidget(sub)
        h.addStretch(1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("appSubtitle")
        h.addWidget(ver)
        return header

    def _make_analysis_menu(self) -> QToolButton:
        """The analysis-file actions as one real menu in the header.

        These belong to the window, not to the Setup step: which analysis is open is not a
        setting, and the user must be able to save from any tab. One menu (rather than the
        row of buttons Setup used to carry) also keeps the header calm as actions are added.
        """
        sc = self.settings_screen
        menu = QMenu(self)
        menu.addAction("New analysis", sc.new_analysis)
        menu.addAction("Open analysis…", sc.open_analysis_dialog)
        menu.addSeparator()
        self._act_save = menu.addAction("Save", sc.save_analysis)
        self._act_save_as = menu.addAction("Save as…", sc.save_analysis_as)
        # The recents and the Save enable-state both depend on live state, so they are
        # rebuilt each time the menu drops down. Reading prefs at show time cannot go stale;
        # listening to a screen signal would (analysis_state_changed is emitted BEFORE
        # prefs.add_recent_analysis in _load, so it would read the list one open behind).
        self._recent_sep = menu.addSeparator()
        self._recent_actions = []
        menu.setToolTipsVisible(True)       # the full path lives in each recent's tooltip
        menu.aboutToShow.connect(self._refresh_analysis_menu)
        self.analysis_menu = menu
        self.analysis_btn = QToolButton()
        self.analysis_btn.setObjectName("appMenuButton")
        self.analysis_btn.setText("Analysis")
        self.analysis_btn.setMenu(menu)
        self.analysis_btn.setPopupMode(QToolButton.InstantPopup)   # a menu, not a button
        return self.analysis_btn

    def _refresh_analysis_menu(self):
        """Recompute the parts of the Analysis menu that depend on live state."""
        sc = self.settings_screen
        savable = sc.can_save()
        # Save is offered when the analysis is NEW (no file yet — the command falls through
        # to Save as…) or DIRTY (edits to write back); a clean opened file has nothing to
        # save. Save as… always names a new file, so it only needs savable settings.
        new = not getattr(self.state, "settings_path", None)
        self._act_save.setEnabled(savable and (sc.is_dirty() or new))
        self._act_save_as.setEnabled(savable)
        self._rebuild_recent_analyses()

    def _rebuild_recent_analyses(self):
        """Replace the menu's recents section with the current list (P26). Only the tracked
        recent actions are removed, so New/Open/Save keep their identity — and their
        shortcuts and enable-state — across rebuilds."""
        from respmech.ui import prefs  # noqa: PLC0415
        for act in self._recent_actions:
            self.analysis_menu.removeAction(act)
            act.deleteLater()      # addAction parented it to the menu; removeAction only
                                   # detaches it, so without this every menu-open leaks 5
        self._recent_actions = []
        # prefs.recent_analyses() already drops files that no longer exist: a dead menu
        # entry is noise, not information.
        recents = prefs.recent_analyses()[:_MENU_RECENTS]
        self._recent_sep.setVisible(bool(recents))
        for i, path in enumerate(recents, 1):
            act = self.analysis_menu.addAction(f"&{i}  {_recent_label(path)}")
            act.setToolTip(path)                       # the exact path, unelided
            act.triggered.connect(lambda _=False, p=path: self._open_recent(p))
            self._recent_actions.append(act)

    def _open_recent(self, path: str):
        """Open a recent analysis, honouring the same unsaved-changes guard as any other
        action that would discard edits."""
        if not self.settings_screen.confirm_discard_changes(
                "Open analysis", question="Save them before opening another analysis?"):
            return
        self.settings_screen.open_analysis(path)

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
        self.settings_screen.refresh_qc()           # cautions can depend on Preview-owned fields
        w = self.tabs.widget(index)
        if w is self.preview_screen:
            self.preview_screen.refresh_files()
        elif w is self.run_screen:
            self.run_screen.refresh_actions()

    def _on_settings_changed(self):
        pass

    def closeEvent(self, ev):
        # Ask about unsaved edits BEFORE tearing anything down: a cancelled close must
        # leave a fully live window, not one whose workers have already been joined.
        # Only a VISIBLE window asks — its close is a user action with someone there to
        # answer. A never-shown window closes programmatically (scripting, headless
        # tests), where a modal prompt would block forever with nobody to dismiss it.
        # A close arriving while ANY discard prompt is already up (close-over-close,
        # or Cmd+Q on top of an open-flow prompt) is aborted by the guard's own
        # re-entrancy latch — stacked window-modal boxes are cocoa's
        # "modalSession exited prematurely" recipe.
        if self.isVisible() and not self.settings_screen.confirm_discard_changes("Close RespMech"):
            ev.ignore()
            return
        # join every worker thread (Preview reactive jobs + a running batch) so none
        # outlives the window and gets destroyed while running
        for screen in (self.preview_screen, self.run_screen):
            try:
                screen.shutdown()
            except Exception:               # pragma: no cover - best-effort teardown
                pass
        super().closeEvent(ev)

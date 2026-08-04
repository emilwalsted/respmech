"""Main window — a tabbed shell holding the three screens, sharing one AppState.

The Settings screen is the single source of change events; the main window is the
only place cross-screen signals are wired, so the Qt-free ``AppState`` never grows
Qt dependencies. A status bar mirrors each screen's status line.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from respmech import __version__
from respmech.ui.flow_layout import ElidingLabel
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
        # A wheel that starts on the Setup form and drifts up onto the tab bar would
        # otherwise change tab — landing on Preview, which schedules work.
        from respmech.ui import wheel as _wheel
        self._wheel_guard = _wheel.swallow_wheel(extra=[self.tabs.tabBar()], parent=self)

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
        # a noise reference chosen on the Preview graph (feature B) mirrors into Settings.
        # Wrapped (not connected to sc.set_noise_reference directly): the user just acted on
        # PREVIEW, but the confirmation message it produces is emitted by Setup, so under
        # the per-tab bar-ownership rule it would be silently swallowed (Setup isn't the
        # active tab) and only resurface, stale, on a later visit to Setup. Force it onto
        # the bar immediately instead — this is an "X just happened" notification, not
        # ongoing per-screen state, so it does not need to wait for a tab switch.
        pv.noise_reference_changed.connect(self._on_noise_reference_changed)
        # Preview-owned settings (noise/ECG params, breath exclusions) land in the saved
        # .toml too, so a user edit there must dirty the analysis like any Setup edit —
        # the title, the close guard and Save-gating all read the same flag.
        pv.settings_edited.connect(sc._mark_dirty)
        # A Setup caution can depend on a field the Preview owns (the gated peak needs
        # remove_ecg, written by the ECG tab), so the strip has to be recomputed when Preview
        # edits the model — not only when Setup's own widgets change.
        pv.settings_edited.connect(sc.refresh_qc)
        # Mechanics + EMG conditioning are Preview-owned now: integrate_from_flow drives the
        # Setup channel summary's 'derived from flow' row, and normalisation drives the
        # 'You will get' normalised-EMG-sheet line — so both must refresh on a Preview edit.
        pv.settings_edited.connect(sc.sync_from_preview)
        # P19: "Process & write this file" on Preview → run just that file on the Run screen
        pv.process_file_requested.connect(self._process_single_file)
        # P20: double-clicking a file in the Run results drills back into Preview
        rn.open_file_requested.connect(self._open_file_in_preview)
        # while a batch runs, lock the Settings screen so its state can't be swapped
        # (e.g. Load TOML) out from under the running worker
        rn.run_started.connect(self._on_run_started)
        rn.run_finished.connect(self._on_run_finished)
        bar = self.statusBar()          # creates the status bar (offscreen-safe)
        # Each screen already keeps its own last message live in its own (hidden, for
        # Setup/Preview/Run alike) status QLabel — see each screen's _set_status. So the
        # bar needs no separate {screen: message} store of its own: on every status_changed
        # it just re-shows THAT screen's message, and only when the sender is the tab the
        # user is actually looking at. A run's progress is the one legitimate exception —
        # it is shown globally, prefixed "Run: ", for as long as a batch is in flight (see
        # _on_run_started/_on_run_finished), so a run started on this tab stays visible
        # while the user checks another one.
        self._run_active = False
        for scr in (sc, pv, rn):
            scr.status_changed.connect(lambda msg, s=scr: self._on_screen_status(s, msg))
        bar.showMessage("Ready.")

        # LAST, not first. This used to run before the tabs and screens existed, when
        # minimumSizeHint() still read 0x0 — so it sized against nothing and its clamp was a
        # no-op that looked like it worked. It has to see the built layout to clamp it.
        self._fit_to_screen()

    def begin_session(self, cli_path: str | None = None):
        """Run the startup flow once the window is on screen: if an analysis file was
        passed on the command line it is already loaded, so open it (all cards + tabs);
        otherwise ask the user to start a New analysis (guided) or Open an existing one.
        Not called from ``__init__`` so headless/test construction keeps full access."""
        if cli_path:
            self.settings_screen.enter_open_mode()
            # app.py loaded the file itself, so settings_screen._load never ran — surface
            # any schema upgrade here instead, or a command-line/drag-drop open would
            # reinterpret a retired setting with nothing on screen to say so.
            self.settings_screen.surface_notices()
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

    def _on_noise_reference_changed(self, file, intervals, use_expiration):
        """Forward a Preview-graph reference pick into Setup's model, then force its
        confirmation onto the bar regardless of the active tab — see the wiring comment
        in __init__ for why this can't just rely on the per-tab ownership rule."""
        self.settings_screen.set_noise_reference(file, intervals, use_expiration)
        self._show_settings_status()

    def _show_settings_status(self):
        """Force Setup's own current status onto the bar regardless of the active tab — used
        right after a window-level Analysis-menu action or a Preview-triggered Setup update
        (see the callers), both "X just happened" notifications a user expects to see from
        wherever they are, not only on a later visit to Setup. Falls back to "Ready." rather
        than a blank bar: some of these actions (New analysis) land in the guided flow, whose
        own feedback lives on Setup's guidance label rather than the status line."""
        self.statusBar().showMessage(self.settings_screen.status.text() or "Ready.")

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
        self._run_active = True

    def _on_run_finished(self):
        self.settings_screen.setEnabled(True)
        self.analysis_btn.setEnabled(True)
        self._run_active = False
        # Hand the bar straight back to the active tab's own message — otherwise the last
        # "Run: …" line would sit there indefinitely if the user never switches tabs again.
        self._show_active_tab_status()

    def _on_screen_status(self, screen, msg):
        """Route one screen's status_changed to the shared bar, per the ownership rule
        described where this is wired up in __init__: while a batch is in flight, the bar
        belongs to Run's progress ALONE (every other screen's status is suppressed, not just
        de-prioritised — otherwise _on_tab_changed's own refresh_files()/refresh_actions()
        calls, which fire on the newly-current tab, would cover the "Run: " line up again);
        otherwise a screen's message shows only while it is the active tab."""
        bar = self.statusBar()
        if self._run_active:
            if screen is self.run_screen:
                # Run's own outcome lines already start with "Run" ("Run failed — …", "Run
                # cancelled — …") — don't stutter "Run: Run failed — …" on top of them.
                prefix = "" if msg[:4].lower() == "run " else "Run: "
                bar.showMessage(f"{prefix}{msg}")
            return
        if screen is self.tabs.currentWidget():
            bar.showMessage(msg)

    def _show_active_tab_status(self):
        """Show the current tab's own last message on the bar (falling back to "Ready." for
        a genuinely empty one, e.g. Setup's guided-flow mode — whose stage guidance lives on
        its own label now, not the status line — so the bar never reads as simply blank)."""
        status = getattr(self.tabs.currentWidget(), "status", None)
        text = status.text() if status is not None else ""
        self.statusBar().showMessage(text or "Ready.")

    def _make_header(self) -> QFrame:
        """A calm application header bar (brand · Analysis menu · subtitle) above the tabs."""
        header = QFrame()
        header.setObjectName("appHeader")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 10, 18, 10)
        title = QLabel("RespMech")
        title.setObjectName("appTitle")
        # Eliding, not a plain QLabel: a QLabel's minimum width is its whole sentence, and a
        # QHBoxLayout hands its minimum to the window — this one line measured 683 px on
        # Windows font metrics and pushed the window's minimum to 1196 px, past a 1080p
        # @175% screen entirely. It is branding, so shortening it costs nothing; the full
        # text stays in the tooltip.
        sub = ElidingLabel("Respiratory mechanics · work of breathing · diaphragm EMG")
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
        menu = QMenu(self)
        # Wrapped, not connected to the SettingsScreen methods directly: these are window-
        # level actions reachable from every tab (that is the whole point of living in the
        # header, not on Setup), but the confirmation each one ends with ("Saved …", "Opened
        # …") is emitted BY Setup. Under the per-tab bar-ownership rule that message would be
        # silently dropped when triggered from Preview or Run and only resurface, stale, on a
        # later visit to Setup — wrong for an "X just happened" notification the user is
        # looking straight at the header for. Force it onto the bar immediately instead.
        menu.addAction("New analysis", self._new_analysis)
        menu.addAction("Open analysis…", self._open_analysis_dialog)
        menu.addSeparator()
        self._act_save = menu.addAction("Save", self._save_analysis)
        self._act_save_as = menu.addAction("Save as…", self._save_analysis_as)
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

    def _new_analysis(self):
        self.settings_screen.new_analysis()
        self._show_settings_status()

    def _open_analysis_dialog(self):
        self.settings_screen.open_analysis_dialog()
        self._show_settings_status()

    def _save_analysis(self):
        self.settings_screen.save_analysis()
        self._show_settings_status()

    def _save_analysis_as(self):
        self.settings_screen.save_analysis_as()
        self._show_settings_status()

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
        self._show_settings_status()

    def _fit_to_screen(self, desired_w: int = 1180, desired_h: int = 820,
                       fraction: float = 0.92) -> None:
        """Size the window to a sensible fraction of the available screen and centre it.

        Deliberately capped at ``desired_w``/``desired_h`` rather than sized from
        ``sizeHint()``: this window's natural size is the whole content of three screens and
        opening at it would fill any display. ``screen_fit.clamp_to_screen`` is the right
        tool for a DIALOG, whose natural size is the size it should open at; here only the
        upper bound and the centring are wanted.
        """
        from respmech.ui import screen_fit  # noqa: PLC0415

        avail = screen_fit.available_for(self)
        if avail is None:
            self.resize(desired_w, desired_h)
            return
        w = min(desired_w, int(avail.width() * fraction))
        h = min(desired_h, int(avail.height() * fraction))
        # Never below the window's own minimum — resize() would ignore it anyway, and the
        # centring below has to work from the size the window will actually have.
        floor = self.minimumSizeHint()
        w = max(w, min(floor.width(), avail.width()))
        h = max(h, min(floor.height(), avail.height()))
        self.resize(w, h)
        frame = self.frameGeometry()          # centre within the work area
        frame.moveCenter(avail.center())
        top_left = frame.topLeft()
        # Clamp into the work area: a window larger than the screen would otherwise be
        # centred off BOTH edges, losing the title bar and the tab strip together.
        top_left.setX(max(avail.left(), min(top_left.x(), avail.right() - frame.width() + 1)))
        top_left.setY(max(avail.top(), min(top_left.y(), avail.bottom() - frame.height() + 1)))
        self.move(top_left)

    def _on_tab_changed(self, index):
        self.settings_screen.to_state()             # belt-and-suspenders sync
        self.settings_screen.refresh_qc()           # cautions can depend on Preview-owned fields
        w = self.tabs.widget(index)
        if w is self.preview_screen:
            self.preview_screen.refresh_files()
        elif w is self.run_screen:
            self.run_screen.refresh_actions()
        # Re-show the INCOMING screen's own last message — refresh_files()/refresh_actions()
        # above may already have done this via _on_screen_status (index has already moved, so
        # currentWidget() is w), but this also covers a tab with nothing to refresh (Setup).
        # Skipped while a batch is running: the bar belongs to the global "Run: " progress
        # line then (see _on_screen_status), which a plain tab switch must not paint over —
        # switching TO Run needs no help either, since that line is already the live one.
        if not self._run_active:
            self._show_active_tab_status()

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

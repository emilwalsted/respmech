"""Main window — a tabbed shell holding Setup and the Preview & QC workspace, sharing
one AppState. Run & results (B03, UI-overhaul) is no longer a third tab: it lives inside
the workspace as a drawer under Preview & QC's file rail, so running a batch never means
leaving the file you are looking at.

The Settings screen is the single source of change events; the main window is the
only place cross-screen signals are wired, so the Qt-free ``AppState`` never grows
Qt dependencies. A status bar mirrors each screen's status line.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from respmech import __version__
from respmech.ui.about_dialog import AboutDialog, GITHUB_URL, WEBSITE_URL
from respmech.ui.flow_layout import ElidingLabel
from respmech.ui.state import AppState
from respmech.ui.screens.settings_screen import SettingsScreen
from respmech.ui.screens.preview_screen import PreviewScreen
from respmech.ui.screens.run_screen import RunScreen


# The menu shows fewer recents than prefs stores: a header menu has to stay glanceable.
_MENU_RECENTS = 5


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None):
        super().__init__()
        self.state = state or AppState()
        self.setWindowTitle(f"RespMech {__version__}")
        self.setAcceptDrops(True)   # C04: dropping a .toml/.py analysis onto the window opens it
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
        # B03 (UI-overhaul): Run & results is no longer its own tab. It is built the same
        # way as before (same class, same worker/thread lifecycle) and shares Preview & QC's
        # file rail — one set of per-file rows, not two — then PreviewScreen embeds it as a
        # titled "Run & results" section under its own file rail + subtabs, so the user can
        # dry-run, run and read the run report without leaving the file they are looking at.
        self.run_screen = RunScreen(self.state, file_rail=self.preview_screen.file_rail)
        self.preview_screen.install_run_drawer(self.run_screen)
        # No wizard "1./2./3." numbering — the real workflow loops back (tune -> re-run),
        # it is not one-way. ("&&" renders a literal "&"; a lone "&" is a tab mnemonic.)
        self._i_settings = self.tabs.addTab(self.settings_screen, "Setup")
        self._i_preview = self.tabs.addTab(self.preview_screen, "Preview && QC")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        # A wheel that starts on the Setup form and drifts up onto the tab bar would
        # otherwise change tab — landing on Preview, which schedules work.
        from respmech.ui import wheel as _wheel
        self._wheel_guard = _wheel.swallow_wheel(extra=[self.tabs.tabBar()], parent=self)

        self._file_recent_menu = None       # set by _build_menu_bar(), read by _rebuild_recent_analyses
        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._make_header())
        col.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self._build_menu_bar()

        sc, pv, rn = self.settings_screen, self.preview_screen, self.run_screen
        sc.inputs_changed.connect(pv.refresh_files)
        sc.settings_changed.connect(pv.sync_from_settings)
        sc.settings_changed.connect(rn.refresh_actions)
        # B03: the drawer's log-empty-state plan summary resolves the actual matched file
        # list (a real glob), so — like Preview's own rail rebuild above — it is refreshed
        # only on an actual input folder/mask edit, never on every settings_changed tick.
        sc.inputs_changed.connect(rn._update_plan_summary)
        # B04: the guided "new analysis" flow used to LOCK (not hide) the Preview & QC tab
        # until every setting was valid — a stepper that turned out to be invisible (a
        # QTabBar stylesheet colour overrides Qt's disabled palette, so a locked tab looks
        # identical to an unlocked one) and closed the wrong thing anyway (a first-time
        # user could not even look ahead). Every surface is reachable now; the one action
        # that stays gated is the Run drawer's primary button, via its own commitment
        # sheet (run_screen.py). sc.flow_ready_changed is unused for now — kept defined in
        # case a future screen wants a cheap "is the analysis fully valid" edge signal —
        # rather than wired to a receiver invented only to have one.
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
        # D24: a Preview write action (the breath-toggle click) rejected because a run is
        # active — forced onto the bar the same way noise_reference_changed's confirmation
        # is, immediately above, and for the same reason: the run-active bar-ownership rule
        # a few lines below in _on_screen_status would otherwise swallow it outright, since
        # "a run is active" is this message's only trigger condition.
        pv.write_action_blocked.connect(lambda msg: self.statusBar().showMessage(msg))
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
        # A breath toggle or a breath-count-overrides commit on Preview can resolve (or
        # create) carried-over state the Setup banner is showing/should show — without this,
        # confirming a carried exclusion from Preview leaves the banner on Setup naming a
        # file that no longer needs it (see _on_noise_reference_changed below for the
        # noise-reference write path, which goes through a different signal).
        pv.settings_edited.connect(sc._update_carried_banner)
        # P19: "Process & write this file" on Preview → run just that file on the Run drawer
        pv.process_file_requested.connect(self._process_single_file)
        # P20's "drill back into Preview" no longer needs its own signal/handler (B03):
        # the file rail is now the SAME widget instance for both, so selecting/activating a
        # row already re-renders Preview & QC's subtabs for it — there is no separate
        # destination left to drill back to.
        # while a batch runs, lock the Settings screen so its state can't be swapped
        # (e.g. Load TOML) out from under the running worker
        rn.run_started.connect(self._on_run_started)
        rn.run_finished.connect(self._on_run_finished)
        # C03 point 8: "Choose another folder…" in the temp-output confirmation changes
        # Setup's own widget (see both docstrings for why this can't be a direct model write).
        rn.output_folder_change_requested.connect(sc.set_output_folder)
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
        self._apply_startup_choice(dlg)

    def _apply_startup_choice(self, dlg):
        """Act on a finished ``StartupDialog`` — shared between the very first session
        window (``begin_session``) and 'Get started…' (ticket C03 point 2, reachable again
        later from the Analysis/File menu) so the two doors can never disagree about what
        each mode does."""
        # a failed open (corrupt file) falls through to the guided New flow rather than
        # leaving the user in full mode over rolled-back defaults
        if dlg.mode == "open" and dlg.path and self.settings_screen.open_analysis(dlg.path):
            return
        if dlg.mode == "sample" and self.settings_screen.open_sample_analysis():   # P23
            return
        # P25: "New from last rig" inherits the last channel mapping into the guided flow
        self.settings_screen.enter_new_mode(use_last_rig=(dlg.mode == "new_rig"))

    def _update_window_title(self):
        """Name the active analysis in the title bar and flag unsaved edits, so it is
        obvious which analysis is loaded and whether it has been saved. The unsaved
        analysis is named "new analysis", not "new analysis (unsaved)": the modified
        marker already carries that, and both would say it twice.

        Ticket C03 points 7/8: an analysis with no ``settings_path`` of its own is not
        always a blank one — a just-imported legacy .py or the built-in sample both land
        here too, and both used to be indistinguishable from "new analysis" in the title.
        ``is_sample`` wins over ``display_name`` (both are only ever set on the SAME
        path-less state, never together, but the sample is the more actionable thing to
        say if that ever changed)."""
        import os  # noqa: PLC0415
        path = getattr(self.state, "settings_path", None)
        if getattr(self.state, "is_sample", False):
            name = "Sample analysis (built-in)"
        elif path:
            name = os.path.basename(path)
        else:
            name = getattr(self.state, "display_name", None) or "new analysis"
        dirty = " * (modified)" if self.settings_screen.is_dirty() else ""
        self.setWindowTitle(f"RespMech {__version__} — {name}{dirty}")

    def _process_single_file(self, filename: str):
        """P19: process+write just the previewed file. The request originates on Preview &
        QC itself, and the Run drawer (B03) lives on that same tab, so there is no longer a
        separate screen to switch to — the tab is already the right one."""
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
        than a blank bar."""
        self.statusBar().showMessage(self.settings_screen.status.text() or "Ready.")

    def _on_run_started(self):
        self.settings_screen.setEnabled(False)
        # the Analysis menu lives in the header, OUTSIDE the screen being locked, so it
        # needs locking too — otherwise Open/New could swap the settings out from under
        # the running worker, which is exactly what disabling the screen prevents.
        self.analysis_btn.setEnabled(False)
        # D24 (UI-overhaul): Preview & QC used to enforce none of this — a breath-exclusion
        # click or "Process & write this file" during a batch looked like it worked while
        # silently missing the run entirely. Unlike Settings above, this is NOT a whole-
        # screen disable (B04 already reversed that pattern once elsewhere): graphs, zoom
        # and file navigation stay live, only the write actions lock. See
        # PreviewScreen.set_run_active's own docstring.
        self.preview_screen.set_run_active(True)
        self._run_active = True

    def _on_run_finished(self):
        self.settings_screen.setEnabled(True)
        self.analysis_btn.setEnabled(True)
        self.preview_screen.set_run_active(False)
        self._run_active = False
        # Hand the bar straight to Run's OWN last message (its outcome — "Run failed — …",
        # "Finished writing…" — is exactly what the user is looking at right after a run,
        # since the drawer lives on the tab they are already on) rather than the generic
        # "whichever tab is current" lookup _show_active_tab_status uses for a plain tab
        # switch: Run has had no tab of its own to be "the current one" since B03.
        self.statusBar().showMessage(self.run_screen.status.text() or "Ready.")

    def _on_screen_status(self, screen, msg):
        """Route one screen's status_changed to the shared bar, per the ownership rule
        described where this is wired up in __init__: while a batch is in flight, the bar
        belongs to Run's progress ALONE (every other screen's status is suppressed, not just
        de-prioritised — otherwise _on_tab_changed's own refresh_files()/refresh_actions()
        calls, which fire on the newly-current tab, would cover the "Run: " line up again);
        otherwise a screen's message shows only while it is the active tab. Run & results
        has no tab of its own any more (B03: it lives inside Preview & QC as a drawer), so
        its own idle-state messages ("Ready to run.", "Cannot run: …") are shown exactly
        when Preview & QC — the tab that actually hosts it — is the active one."""
        bar = self.statusBar()
        if self._run_active:
            if screen is self.run_screen:
                # Run's own outcome lines already start with "Run" ("Run failed — …", "Run
                # cancelled — …") — don't stutter "Run: Run failed — …" on top of them.
                prefix = "" if msg[:4].lower() == "run " else "Run: "
                bar.showMessage(f"{prefix}{msg}")
            return
        owner = self.preview_screen if screen is self.run_screen else screen
        if owner is self.tabs.currentWidget():
            bar.showMessage(msg)

    def _show_active_tab_status(self):
        """Show the current tab's own last message on the bar, falling back to "Ready." for
        a genuinely empty one so the bar never reads as simply blank."""
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
        # C01 (UI-overhaul): these are real QAction objects, with real shortcuts, so the
        # SAME objects can be added to the File menu built in _build_menu_bar() below — one
        # implementation and one enable-state behind two doors, never two that could disagree.
        self._act_new = QAction("New analysis", self)
        self._act_new.setShortcut(QKeySequence("Ctrl+N"))
        self._act_new.triggered.connect(self._new_analysis)
        menu.addAction(self._act_new)
        self._act_open = QAction("Open analysis…", self)
        self._act_open.setShortcut(QKeySequence("Ctrl+O"))
        self._act_open.triggered.connect(self._open_analysis_dialog)
        menu.addAction(self._act_open)
        menu.addSeparator()
        self._act_save = QAction("Save", self)
        self._act_save.setShortcut(QKeySequence("Ctrl+S"))
        self._act_save.triggered.connect(self._save_analysis)
        menu.addAction(self._act_save)
        self._act_save_as = QAction("Save as…", self)
        self._act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._act_save_as.triggered.connect(self._save_analysis_as)
        menu.addAction(self._act_save_as)
        menu.addSeparator()
        # ticket C03 point 2: both were previously ONE-SHOT — reachable only from the very
        # first window of a session (StartupDialog via begin_session) and, for the sample,
        # not reachable again at all once dismissed (open_sample_analysis has no OTHER
        # caller). Same QAction pattern as the four above: added to both this menu and the
        # File menu built in _build_menu_bar(), never a second implementation.
        self._act_get_started = QAction("Get started…", self)
        self._act_get_started.triggered.connect(self._get_started)
        menu.addAction(self._act_get_started)
        self._act_sample = QAction("Explore with sample data", self)
        self._act_sample.triggered.connect(self._explore_sample)
        menu.addAction(self._act_sample)
        # ticket C03 point 5: same-settings-new-folder, the realistic multi-subject case
        # ("Rammer": subject 1 was set up weeks ago, subjects 2-22 need the identical
        # protocol against a new recordings folder each).
        self._act_duplicate = QAction("Duplicate for another recordings folder…", self)
        self._act_duplicate.triggered.connect(self._duplicate_analysis)
        menu.addAction(self._act_duplicate)
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

    def _get_started(self):
        """Analysis/File > 'Get started…' (ticket C03 point 2): re-raise the startup
        chooser after the first window — guarded like every other action that would drop
        unsaved edits, since 'New analysis'/'Open…' inside the chooser both do exactly
        that."""
        if not self.settings_screen.confirm_discard_changes(
                "Get started", question="Save them before starting over?"):
            return
        from respmech.ui.startup_dialog import StartupDialog
        dlg = StartupDialog(self)
        dlg.exec()
        self._apply_startup_choice(dlg)
        self._show_settings_status()

    def _explore_sample(self):
        """Analysis/File > 'Explore with sample data' (ticket C03 point 2): the sample was
        previously reachable only from the FIRST window of a session — this makes it a
        real, repeatable door, guarded the same way 'New analysis' already is."""
        if not self.settings_screen.confirm_discard_changes(
                "Explore with sample data",
                question="Save them before exploring the built-in sample data?"):
            return
        self.settings_screen.open_sample_analysis()
        self._show_settings_status()

    def _duplicate_analysis(self):
        self.settings_screen.duplicate_for_another_folder()
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
        shortcuts and enable-state — across rebuilds.

        C01 (UI-overhaul): each recent is ONE QAction, added to BOTH the header's flat
        list and the File menu's "Open Recent" submenu (built in _build_menu_bar()) — so
        the two doors can never name a different set of recents."""
        from respmech.ui import prefs  # noqa: PLC0415
        for act in self._recent_actions:
            self.analysis_menu.removeAction(act)
            if self._file_recent_menu is not None:
                self._file_recent_menu.removeAction(act)
            act.deleteLater()      # addAction parented it to the menu; removeAction only
                                   # detaches it, so without this every menu-open leaks 5
        self._recent_actions = []
        # prefs.recent_analyses() already drops files that no longer exist: a dead menu
        # entry is noise, not information.
        recents = prefs.recent_analyses()[:_MENU_RECENTS]
        self._recent_sep.setVisible(bool(recents))
        if self._file_recent_menu is not None:
            self._file_recent_menu.setEnabled(bool(recents))
        for i, path in enumerate(recents, 1):
            act = QAction(f"&{i}  {prefs.recent_label(path)}", self)
            act.setToolTip(path)                       # the exact path, unelided
            act.triggered.connect(lambda _=False, p=path: self._open_recent(p))
            self.analysis_menu.addAction(act)
            if self._file_recent_menu is not None:
                self._file_recent_menu.addAction(act)
            self._recent_actions.append(act)

    def _open_recent(self, path: str):
        """Open a recent analysis, honouring the same unsaved-changes guard as any other
        action that would discard edits."""
        if not self.settings_screen.confirm_discard_changes(
                "Open analysis", question="Save them before opening another analysis?"):
            return
        self.settings_screen.open_analysis(path)
        self._show_settings_status()

    # -- drag-and-drop + double-click open (C04, UI-overhaul) ----------------
    @staticmethod
    def _dropped_analysis_path(mime):
        """The path of a single, local, ``.toml``/``.py`` file carried by ``mime`` — else
        ``None``. Several files, a folder, a remote URL, or an unrecognised extension are
        all rejected outright rather than picking one or guessing; ``settings_screen.
        open_analysis`` itself already routes by extension, so only those two need pass."""
        if mime is None or not mime.hasUrls():
            return None
        urls = mime.urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = urls[0].toLocalFile()
        if not path.casefold().endswith((".toml", ".py")):
            return None
        # Native separators, same reasoning as path_drop._single_local_path: toLocalFile()
        # says 'C:/Users/…' on Windows, and this path flows into open_analysis, the recents
        # menu and the window title — all places a Windows user reads it. No-op elsewhere.
        return os.path.normpath(path)

    def dragEnterEvent(self, event):
        if self._dropped_analysis_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._dropped_analysis_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        path = self._dropped_analysis_path(event.mimeData())
        if path is None:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self._open_dropped_path(path)

    def _open_dropped_path(self, path: str):
        """Open a ``.toml``/``.py`` analysis dropped onto the window — same unsaved-changes
        guard as every other way of opening (recents, the Open dialog, closing): the drop
        itself does not carry that guard, so a caller other than ``dropEvent`` must apply it
        (``settings_screen.open_analysis`` on its own discards unsaved edits silently)."""
        if not self.settings_screen.confirm_discard_changes(
                "Open analysis", question="Save them before opening another analysis?"):
            return
        self.settings_screen.open_analysis(path)
        self._show_settings_status()

    # -- menu bar (C01, UI-overhaul) -----------------------------------------
    def _build_menu_bar(self):
        """A real ``QMenuBar`` (File/View/Help) alongside the header's Analysis button —
        the same commands, reachable the way a desktop user actually looks for them: Qt
        merges this into the system menu bar on macOS and draws Alt-mnemonics on Windows,
        neither of which the header's QToolButton popup gave. File's New/Open/Save/Save
        as… are the EXACT SAME QAction objects the Analysis menu already built (added to a
        second container each), so there is one implementation and one enable-state
        behind two doors, never two that could quietly disagree. The header's Analysis
        button is left as it was: a second door, not the only one."""
        bar = self.menuBar()

        # Kept as attributes, not locals: a QMenu returned by addMenu() is parented to the
        # bar C++-side, but PySide6 needs a live Python reference to keep re-wrapping it
        # reliably (an unreferenced wrapper going out of scope has been observed to leave
        # the C++ side unusable — "Internal C++ object already deleted" the next time it
        # is touched, e.g. from a test or _refresh_view_menu below).
        file_menu = self._file_menu = bar.addMenu("&File")
        file_menu.addAction(self._act_new)
        file_menu.addAction(self._act_open)
        self._file_recent_menu = file_menu.addMenu("Open Recent")
        file_menu.addSeparator()
        file_menu.addAction(self._act_save)
        file_menu.addAction(self._act_save_as)
        file_menu.addSeparator()
        # ticket C03 point 2: same QAction objects as the header's Analysis menu (see
        # their own construction in _make_analysis_menu) — one enable-state, two doors.
        file_menu.addAction(self._act_get_started)
        file_menu.addAction(self._act_sample)
        file_menu.addAction(self._act_duplicate)
        file_menu.addSeparator()
        self._act_open_output = QAction("Open output folder", self)
        # Same slot the Run drawer's own button uses (run_screen.py) — one place decides
        # which folder that is (a subset "write elsewhere" target, else the analysis's
        # own output folder) and whether it currently exists.
        self._act_open_output.triggered.connect(self.run_screen._open_output_folder)
        file_menu.addAction(self._act_open_output)
        file_menu.addSeparator()
        self._act_close_window = QAction("Close Window", self)
        self._act_close_window.setShortcut(QKeySequence("Ctrl+W"))
        self._act_close_window.triggered.connect(self.close)
        file_menu.addAction(self._act_close_window)
        # Recomputes Save's enable-state and the recents on EITHER menu's open, not just
        # the Analysis button's — see _refresh_analysis_menu's own docstring for why this
        # can't just listen for a state-change signal instead.
        file_menu.aboutToShow.connect(self._refresh_analysis_menu)

        view_menu = self._view_menu = bar.addMenu("&View")
        self._view_actions = []
        # One action per ACTUAL tab, not a fixed Setup/Preview/Run trio: Run & results
        # stopped being its own tab in B03 (it lives inside Preview & QC as a drawer), so
        # there is no third tab to name here any more — see
        # memory/respmech-skill-udestaaende.md's B03 entry.
        for i in range(self.tabs.count()):
            act = QAction(self.tabs.tabText(i), self)
            if i < 9:
                act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.triggered.connect(lambda _checked=False, idx=i: self.tabs.setCurrentIndex(idx))
            view_menu.addAction(act)
            self._view_actions.append(act)
        # B04 retired all tab locking, so every action is enabled today — but a future
        # ticket could reintroduce gating, and this keeps the menu honest without needing
        # its own new plumbing if it does: it just reads isTabEnabled() again on open.
        view_menu.aboutToShow.connect(self._refresh_view_menu)

        help_menu = self._help_menu = bar.addMenu("&Help")
        act_docs = QAction("RespMech documentation", self)
        act_docs.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(f"{WEBSITE_URL}/documentation.html")))
        help_menu.addAction(act_docs)
        act_site = QAction("RespMech website", self)
        act_site.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(WEBSITE_URL)))
        help_menu.addAction(act_site)
        act_issue = QAction("Report an issue", self)
        act_issue.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(f"{GITHUB_URL}/issues")))
        help_menu.addAction(act_issue)
        help_menu.addSeparator()
        act_about = QAction("About RespMech…", self)
        # AboutRole: on macOS Qt moves this into the application menu (RespMech ▸ About
        # RespMech) instead of leaving it under Help, matching platform convention.
        act_about.setMenuRole(QAction.MenuRole.AboutRole)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _refresh_view_menu(self):
        """Mirror each tab's enabled state onto its View menu entry — see the "future
        ticket" note in _build_menu_bar for why this stays wired even though B04 leaves
        every tab enabled today."""
        for i, act in enumerate(self._view_actions):
            act.setEnabled(self.tabs.isTabEnabled(i))

    def _show_about(self):
        """Help > About RespMech… — a small, fully offline dialog (see about_dialog.py):
        version, licence, citation and both project URLs as plain selectable text."""
        AboutDialog(self).exec()

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
            # Self-review finding: the Run drawer's io-info (file count) is read from the
            # shared file rail, but nothing else re-reads it on a plain tab switch — a file
            # added or removed on disk between visits (no Settings edit involved) used to be
            # caught by the old "switching TO the Run tab" refresh_actions() call, which has
            # no tab-switch equivalent since Run stopped being its own tab.
            self.run_screen.refresh_actions()
        # Re-show the INCOMING screen's own last message — refresh_files() above may already
        # have done this via _on_screen_status (index has already moved, so currentWidget()
        # is w), but this also covers a tab with nothing to refresh (Setup).
        # Skipped while a batch is running: the bar belongs to the global "Run: " progress
        # line then (see _on_screen_status), which a plain tab switch must not paint over —
        # switching TO Preview & QC (which hosts the Run drawer) needs no help either, since
        # that line is already the live one.
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
        # Setup's channel summary builds its own PlotWidgets (ColumnStack) outside
        # Preview & QC's shutdown() — see ChannelSummary.close_plots()'s docstring for why
        # this is the dominant point 6 leak source once the above was fixed.
        try:
            self.settings_screen.channel_summary.close_plots()
        except Exception:               # pragma: no cover - best-effort teardown
            pass
        super().closeEvent(ev)

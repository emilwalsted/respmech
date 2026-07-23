"""GUI entry point: ``respmech-gui`` (or ``python -m respmech.ui.app``).

On start it shows a brand splash while the main window is built, then reveals the
window **maximized** (not fullscreen) once a short minimum splash time has passed.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback

# minimum time (ms) the splash stays visible before the window is revealed
_MIN_SPLASH_MS = 3000


def _fatal_startup(tb: str) -> None:
    """Report a startup failure with a copyable trace, falling back to stderr."""
    try:
        from PySide6.QtWidgets import QMessageBox
        from respmech.ui.dialogs import short_error
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("RespMech could not start")
        box.setText(short_error(tb))
        box.setDetailedText(tb)
        box.exec()
    except Exception:                    # pragma: no cover - last resort
        print(tb, file=sys.stderr)


def main(argv=None) -> int:
    # Diagnostic figures are written in a spawned child (core/io/_figure_process). In a
    # PACKAGED app sys.executable is this very binary, so without this call a spawned child
    # would re-run main() and launch a second copy of the GUI instead of doing its job. It is
    # a no-op everywhere else, and must run before anything else can spawn.
    import multiprocessing
    multiprocessing.freeze_support()

    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QTimer, QLoggingCategory
    from respmech.ui import theme
    from respmech.ui.dialogs import short_error
    from respmech.ui.splash import make_splash

    # Qt's font-database chatter ("Populating font family aliases … missing font
    # family …") is cosmetic and non-actionable for an end-user app; keep it out of
    # the console. (The splash's font stacks are also resolved to installed families.)
    QLoggingCategory.setFilterRules("qt.qpa.fonts.warning=false")

    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    # stable identity so QSettings (recent analyses, sticky folders, last rig) persists
    app.setOrganizationName("RespMech")
    app.setApplicationName("RespMech")

    # Show the splash as the VERY FIRST thing on screen (it is a self-contained
    # pixmap, so it needs neither the theme nor the icon), then keep it up for at
    # least _MIN_SPLASH_MS before the window is revealed.
    t0 = time.monotonic()
    splash = make_splash(app)       # None if Qt SVG support is unavailable
    if splash is not None:
        splash.show()
        splash.raise_()
        splash.activateWindow()
        app.processEvents()         # paint the splash immediately

    try:
        theme.apply_theme(app)      # Fusion + palette + QSS + plot styling (cosmetic)
    except Exception:               # noqa: BLE001 — never fail to start over theming
        traceback.print_exc()
    try:
        from respmech.ui.logo import app_icon
        icon = app_icon()
        if icon is not None:
            app.setWindowIcon(icon)
    except Exception:               # noqa: BLE001 — icon is cosmetic
        pass

    from respmech.ui.state import AppState
    from respmech.ui.main_window import MainWindow
    startup_error = None
    # casefold() so a Windows-style ANALYSIS.TOML / .Toml drag-drop is recognised too.
    toml_arg = argv[1] if (len(argv) > 1 and argv[1].casefold().endswith(".toml")) else None
    try:
        state = AppState()
        # Optional: a settings file passed on the command line is loaded on start.
        if toml_arg is not None:
            try:
                state.load_toml(toml_arg)
            except Exception:       # noqa: BLE001 — fall back to defaults, report later
                startup_error = (toml_arg, traceback.format_exc())
        win = MainWindow(state)
    except Exception:               # construction failed
        build_error = traceback.format_exc()
        # if a loaded settings file broke construction, retry once with defaults
        if toml_arg is not None and startup_error is None:
            try:
                win = MainWindow(AppState())
                startup_error = (toml_arg, build_error)
            except Exception:       # even defaults fail -> report and exit
                if splash is not None:
                    splash.close()
                _fatal_startup(traceback.format_exc())
                return 1
        else:
            if splash is not None:
                splash.close()
            _fatal_startup(build_error)
            return 1

    if splash is not None:          # keep the (modal) splash on top after the build
        splash.raise_()
        app.processEvents()

    def _reveal():
        # show the window first, THEN close the modal splash (which releases the
        # modality) so focus/activation of the window actually takes effect
        try:
            win.showMaximized()     # maximised to the screen, not fullscreen
        finally:
            if splash is not None:
                splash.finish(win)
        win.raise_()
        win.activateWindow()
        # Start the warm-up HERE, before begin_session() below: that call opens the modal
        # New/Open chooser and blocks until the user picks, so anything after it would not
        # run until the choice is already made. Starting it here spends exactly the idle
        # time we want — the seconds the user spends in the chooser.
        _warm_compute_core()
        if startup_error is not None:
            fn, tb = startup_error
            box = QMessageBox(win)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Could not load analysis")
            box.setText(f"Could not open '{fn}'. Starting fresh.\n\n{short_error(tb)}")
            box.setDetailedText(tb)
            box.show()
        # start the New/Open chooser flow; a successfully-loaded CLI file skips it and
        # opens directly, while a failed one falls through to the chooser.
        try:
            win.begin_session(cli_path=toml_arg if startup_error is None else None)
        except Exception:               # noqa: BLE001 — never let the chooser break startup
            traceback.print_exc()

    def _warm_compute_core():
        """Import the compute core in the background, once the window is up.

        The GUI no longer imports ``respmech.core.pipeline`` at startup (that cost 1.4 s
        of a 2.0 s launch), but deferring an import only *moves* the stall — it would
        otherwise land inside the user's first Run or file load, where it is more
        annoying. Doing it here spends the idle time instead.

        A plain daemon thread, deliberately: it touches no Qt object, so there is no
        thread affinity to get wrong (a QThreadPool/QRunnable would invite exactly that).
        Python's import system is thread-safe per module and these are leaf-ward imports
        with no cycle back into ``respmech.ui``, so there is no deadlock path; if the user
        is faster than the thread, the import lock simply makes the foreground import wait
        for the same work.
        """
        def _warm():
            try:
                import respmech.core.pipeline        # noqa: F401
                import respmech.core.io.writers      # noqa: F401
            except Exception:                        # noqa: BLE001 — best-effort only
                pass
        threading.Thread(target=_warm, name="respmech-warm-core", daemon=True).start()

    delay = (max(0, _MIN_SPLASH_MS - int((time.monotonic() - t0) * 1000))
             if splash is not None else 0)
    QTimer.singleShot(delay, _reveal)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

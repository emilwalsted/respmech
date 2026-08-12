"""GUI entry point: ``respmech-gui`` (or ``python -m respmech.ui.app``).

On start it shows a brand splash while the main window is built, then reveals the
window **maximized** (not fullscreen) once a short minimum splash time has passed.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback

# floor (ms) the splash stays visible before the window is revealed, once the window
# is actually built. Not the whole splash lifetime: building itself (splash paint +
# theme + MainWindow construction) already takes ~1.3-1.6 s on its own, so this only
# tops that up to a minimum, rather than adding a further fixed wait on top of it.
_MIN_SPLASH_MS = 1200


def _reveal_delay_ms(elapsed_ms: int, floor_ms: int = _MIN_SPLASH_MS) -> int:
    """How much longer (ms) the splash should stay up, given that building the window
    already took ``elapsed_ms``. Pure and Qt-free so it is trivially testable: 0 once
    building alone has already spent the floor, otherwise the remainder of it."""
    return max(0, floor_ms - elapsed_ms)


def _warm_compute_core():
    """Import the compute core in the background, as early as there is idle time to
    spend on it.

    The GUI no longer imports ``respmech.core.pipeline`` at startup (that cost 1.4 s
    of a 2.0 s launch), but deferring an import only *moves* the stall — it would
    otherwise land inside the user's first Run or file load, where it is more
    annoying. Starting it right after the splash is shown spends the window-build
    idle time (theming, ``MainWindow`` construction, ``respmech.ui`` imports)
    instead of the fixed splash-floor wait that used to follow it.

    A plain daemon thread, deliberately: it touches no Qt object, so there is no
    thread affinity to get wrong (a QThreadPool/QRunnable would invite exactly that).
    Python's import system is thread-safe per module, and ``respmech.core.pipeline``
    / ``io.writers`` (pandas, scipy) and ``respmech.ui.main_window`` (matplotlib,
    pyqtgraph, numpy) are leaf-ward imports with no cycle back into ``respmech.ui`` —
    disjoint apart from numpy, which is itself safe to import concurrently. If the
    user is faster than the thread, the import lock simply makes the foreground
    import wait for the same work.
    """
    def _warm():
        try:
            import respmech.core.pipeline        # noqa: F401
            import respmech.core.io.writers      # noqa: F401
        except Exception:                        # noqa: BLE001 — best-effort only
            pass
    try:
        # Called as a plain top-level statement in main(), before the window exists and
        # before any of the try/except blocks that follow it — unlike theme.apply_theme
        # and icon-loading (each explicitly guarded as "never fail to start over X"), an
        # unguarded RuntimeError here (e.g. "can't start new thread", real under a low
        # thread-count ulimit) would propagate out of main() with no window, no splash
        # cleanup and no _fatal_startup dialog. This is best-effort only: if the thread
        # cannot start, the import cost is simply paid later, on the first Run — exactly
        # where it landed before this warm-up existed.
        threading.Thread(target=_warm, name="respmech-warm-core", daemon=True).start()
    except Exception:                            # noqa: BLE001 — best-effort only
        pass


def resolve_startup_path(argv, opened_path=None):
    """Which ``.toml`` analysis (if any) should be opened at startup, and from where.

    Two origins can name a startup file, and on any one platform only ever one of them
    fires: ``opened_path``, from a native "open this file" launch (macOS's
    ``QEvent.FileOpen`` for a double-clicked/Dock-opened document — never present in
    ``argv`` on that platform, see ``qapp.RespMechApplication``), and ``argv[1]``, from a
    plain command-line/double-click launch on Windows/Linux, where no such event is ever
    delivered. ``opened_path`` wins if both are somehow present, being the more specific of
    the two signals. Qt-free so it is trivially testable without a running application.

    Only ``.toml`` is recognised here (a saved analysis) — a dropped ``.py`` legacy setup is
    handled separately, on an already-open window, by ``settings_screen.open_analysis``
    (drag-and-drop onto the main window, ticket C04), never through this startup path.
    """
    # casefold() so a Windows-style ANALYSIS.TOML / .Toml is recognised too, either way.
    if opened_path and opened_path.casefold().endswith(".toml"):
        return opened_path
    if len(argv) > 1 and argv[1].casefold().endswith(".toml"):
        return argv[1]
    return None


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

    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtCore import QTimer, QLoggingCategory
    from respmech.ui import theme
    from respmech.ui.dialogs import short_error
    from respmech.ui.qapp import RespMechApplication
    from respmech.ui.splash import make_splash

    # Qt's font-database chatter ("Populating font family aliases … missing font
    # family …") is cosmetic and non-actionable for an end-user app; keep it out of
    # the console. (The splash's font stacks are also resolved to installed families.)
    QLoggingCategory.setFilterRules("qt.qpa.fonts.warning=false")

    argv = list(sys.argv if argv is None else argv)
    app = RespMechApplication(argv)
    # stable identity so QSettings (recent analyses, sticky folders, last rig) persists
    app.setOrganizationName("RespMech")
    app.setApplicationName("RespMech")
    # Flush a FileOpen event macOS may have already queued for a cold, double-click launch
    # (Cocoa can deliver it as early as during QApplication construction) before anything
    # below reads app.opened_path — otherwise a fresh launch would see it one tick too late.
    app.processEvents()

    # Show the splash as the VERY FIRST thing on screen (it is a self-contained
    # pixmap, so it needs neither the theme nor the icon), then keep it up for at
    # least _MIN_SPLASH_MS after the window is built before revealing it.
    t0 = time.monotonic()
    splash = make_splash(app)       # None if Qt SVG support is unavailable
    if splash is not None:
        splash.show()
        splash.raise_()
        splash.activateWindow()
        app.processEvents()         # paint the splash immediately

    # Start the compute-core warm-up NOW, right after the splash is up: everything
    # between here and the window being shown (theming, MainWindow construction,
    # respmech.ui imports) is idle time the thread can spend instead of the user
    # waiting for it later, on the first Run or file load.
    _warm_compute_core()

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
    toml_arg = resolve_startup_path(argv, app.opened_path)
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

    # From here on the window exists: a LATER FileOpen event (app already running, a second
    # document opened from the Dock) is routed live through the same guarded open a
    # drag-and-drop uses (C04) — never through begin_session, which assumes a fresh window.
    # Relies on no processEvents() call happening between win's construction above and this
    # line (confirmed true today: none does) — one landing in between could let a FileOpen
    # event arrive while on_file_open is still unset, and it would be captured into
    # opened_path instead of forwarded live, silently missing this "already running" open.
    app.on_file_open = win._open_dropped_path

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

    delay = _reveal_delay_ms(int((time.monotonic() - t0) * 1000)) if splash is not None else 0
    QTimer.singleShot(delay, _reveal)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

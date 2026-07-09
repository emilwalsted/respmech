"""GUI entry point: ``respmech-gui`` (or ``python -m respmech.ui.app``).

On start it shows a brand splash while the main window is built, then reveals the
window **maximized** (not fullscreen) once a short minimum splash time has passed.
"""
from __future__ import annotations

import sys
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
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QTimer
    from respmech.ui import theme
    from respmech.ui.dialogs import short_error
    from respmech.ui.splash import make_splash

    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)

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
    toml_arg = argv[1] if (len(argv) > 1 and argv[1].endswith(".toml")) else None
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
        if startup_error is not None:
            fn, tb = startup_error
            box = QMessageBox(win)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Could not load settings")
            box.setText(f"Could not load '{fn}'. Starting with default settings.\n\n{short_error(tb)}")
            box.setDetailedText(tb)
            box.show()

    delay = (max(0, _MIN_SPLASH_MS - int((time.monotonic() - t0) * 1000))
             if splash is not None else 0)
    QTimer.singleShot(delay, _reveal)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

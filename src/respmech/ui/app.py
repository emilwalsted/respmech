"""GUI entry point: ``respmech-gui`` (or ``python -m respmech.ui.app``).

On start it shows a brand splash while the main window is built, then reveals the
window **maximized** (not fullscreen) once a short minimum splash time has passed.
"""
from __future__ import annotations

import sys
import time

# minimum time (ms) the splash stays up so it is actually seen even if the main
# window builds quickly
_MIN_SPLASH_MS = 1500


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from respmech.ui.state import AppState
    from respmech.ui.main_window import MainWindow
    from respmech.ui import theme
    from respmech.ui.splash import make_splash

    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    theme.apply_theme(app)          # Fusion + palette + QSS + plot styling; import-safe

    t0 = time.monotonic()
    splash = make_splash(app)       # None if Qt SVG support is unavailable
    if splash is not None:
        splash.show()
        app.processEvents()         # paint the splash before the (slower) build

    state = AppState()
    # Optional: a settings file passed on the command line is loaded on start.
    if len(argv) > 1 and argv[1].endswith(".toml"):
        try:
            state.load_toml(argv[1])
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not load {argv[1]}: {e}", file=sys.stderr)
    win = MainWindow(state)

    def _reveal():
        win.showMaximized()         # maximised to the screen, not fullscreen
        win.raise_(); win.activateWindow()
        if splash is not None:
            splash.finish(win)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    QTimer.singleShot(max(0, _MIN_SPLASH_MS - elapsed_ms), _reveal)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

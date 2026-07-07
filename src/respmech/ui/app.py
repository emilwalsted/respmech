"""GUI entry point: ``respmech-gui`` (or ``python -m respmech.ui.app``)."""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication
    from respmech.ui.state import AppState
    from respmech.ui.main_window import MainWindow

    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    state = AppState()
    # Optional: a settings file passed on the command line is loaded on start.
    if len(argv) > 1 and argv[1].endswith(".toml"):
        try:
            state.load_toml(argv[1])
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not load {argv[1]}: {e}", file=sys.stderr)
    win = MainWindow(state)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

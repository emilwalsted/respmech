"""``python -m respmech`` launches the GUI.

This is also the entry point the briefcase bundle runs (``python -m respmech``). The
command-line batch interface is the separate ``respmech`` console script
(``respmech.cli.__main__:main``) / ``python -m respmech.cli``.
"""
import sys

from respmech.ui.app import main

if __name__ == "__main__":
    sys.exit(main())

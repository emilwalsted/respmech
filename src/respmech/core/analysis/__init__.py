"""Qt-free, numeric-stack-free analysis building blocks.

Everything under this package (``signals.py``, ``registry.py``, and whatever
later tickets add alongside them) must stay importable without pulling in
numpy, scipy, pandas, Qt or ``respmech.core.compute`` at module level — see
``tests/unit/test_startup_imports.py``. Deliberately no re-exports here: each
submodule is imported directly (``from respmech.core.analysis.signals import
...``), so this file itself never becomes a place a future addition could
accidentally widen the import budget from.
"""

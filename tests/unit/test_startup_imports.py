"""The GUI must not import the compute core just to open a window.

``ui/validation.py`` used to import ``match_input_files`` — a pure ``os``/``fnmatch``
helper — at module level, and ``ui/workers.py`` imported ``run_batch``/``write_batch`` the
same way. Between them they pulled ``scipy.interpolate``, ``pandas`` and ``scipy.signal``
into GUI startup: 1.4 s of the 2.0 s it took to show a window, none of it Qt.

Both are now lazy (an in-function import, and two module-level shims respectively), and
``ui/app.py`` warms them on a background thread once the window is up. This test pins that
down, because a single careless ``from respmech.core.pipeline import …`` re-breaks it
silently and nothing else in the suite would notice.

It has to run in a **subprocess**: by the time the rest of the unit suite has executed,
the compute core is long since imported in this process, so an in-process assertion would
pass no matter what the GUI modules do.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")

# Importing the GUI shell must not drag any of these in.
FORBIDDEN = (
    "respmech.core.pipeline",
    "respmech.core.compute",
    "respmech.core.io.loaders",
    "pandas",
    "scipy.interpolate",
)

# The marker keeps parsing unambiguous: a clean run prints "LEAKED:" with nothing after
# it, which is easy to get wrong if the probe prints a bare (empty) line instead.
PROBE = """
import sys
import respmech.ui.main_window  # noqa: F401
forbidden = {forbidden!r}
print("LEAKED:" + ",".join(m for m in forbidden if m in sys.modules))
"""


def test_gui_startup_does_not_import_the_compute_core():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", PROBE.format(forbidden=FORBIDDEN)],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=300,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("LEAKED:")]
    assert marker, f"probe produced no result line:\n{proc.stdout}\n{proc.stderr}"
    leaked = [m for m in marker[-1][len("LEAKED:"):].split(",") if m]
    assert not leaked, (
        "importing respmech.ui.main_window pulled in the compute core: "
        + ", ".join(leaked)
        + "\nKeep these imports lazy — see ui/validation.py and ui/workers.py."
    )


def test_workers_exposes_patchable_batch_names():
    """The lazy shims must stay *module attributes*.

    ``tests/unit/test_gui_hardening.py`` monkeypatches ``workers.run_batch`` and
    ``workers.write_batch``; moving the imports inside ``BatchWorker.run`` instead of
    using shims removes those names and breaks it. That is not hypothetical — it is what
    the first attempt at this optimisation did.
    """
    from respmech.ui import workers
    assert callable(workers.run_batch)
    assert callable(workers.write_batch)

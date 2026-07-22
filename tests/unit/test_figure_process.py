"""Writing diagnostic figures in a separate process (core/io/_figure_process).

The isolation exists because write_batch runs on the GUI's worker thread while the Preview
screen uses matplotlib on the GUI thread, and matplotlib is not thread-safe. The contract
that makes it safe to ship is that it can only ever be a no-op:

  1. the child produces byte-identical figures to the in-process path;
  2. EVERY failure mode falls back to in-process rather than losing figures;
  3. the fallback is reported, not silent;
  4. the child never imports Qt — that is the entire point of the exercise.
"""
import os
import sys

import pytest

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

from respmech.core.io import _figure_process as fp  # noqa: E402


def _result(tmp_path):
    from respmech.core import pipeline
    s = synth_settings(str(tmp_path), remove_ecg=True,
                       data_out={"saveaveragedata": True, "savebreathbybreathdata": True})
    s.output.diagnostics.save_raw = True
    s.output.diagnostics.save_emg = True
    return pipeline.run_batch(s, only_files=["synth_case_A.csv"]), s


# -- 4. the child must be Qt-free ---------------------------------------------

def test_child_entry_point_never_imports_qt(tmp_path):
    """Run the child's own entry point in a bare interpreter and assert PySide6 never gets
    imported. If it did, the isolation would be theatre."""
    import subprocess
    code = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "import respmech.core.io._figure_process as fp;"
        "import respmech.core.plots;"          # what the child imports to do its work
        "print('PySide6' in sys.modules)"
    ) % os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(fp.__file__)))),)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", f"the child imported Qt: {out.stdout!r}"


# -- 1. identical output -------------------------------------------------------

def test_isolated_and_in_process_write_the_same_figures(tmp_path):
    res, s = _result(tmp_path)
    a, b = tmp_path / "iso", tmp_path / "here"
    a.mkdir(); b.mkdir()
    iso_written, iso_failures = fp.write_figures(res, s, str(a))
    in_written, in_failures = fp._in_process(res, s, str(b))

    assert [os.path.basename(p) for p in iso_written] == [os.path.basename(p) for p in in_written]
    assert iso_failures == in_failures
    assert iso_written, "no figures written — the comparison would be vacuous"
    for p, q in zip(sorted(iso_written), sorted(in_written)):
        # PDFs embed a creation date, so compare those by size rather than bytes
        if p.endswith(".pdf"):
            assert abs(os.path.getsize(p) - os.path.getsize(q)) < 2048, os.path.basename(p)
        else:
            assert open(p, "rb").read() == open(q, "rb").read(), os.path.basename(p)


# -- 2 & 3. every failure mode falls back, and says so -------------------------

def test_falls_back_when_spawning_is_disabled(tmp_path, monkeypatch):
    res, s = _result(tmp_path)
    monkeypatch.setattr(fp, "_CAN_SPAWN", None)
    monkeypatch.setenv(fp._DISABLE_ENV, "1")
    notes = []
    written, failures = fp.write_figures(res, s, str(tmp_path), on_fallback=notes.append)
    assert written, "the fallback must still write the figures"
    assert notes and "in-process" in notes[0]


def test_falls_back_when_the_child_raises(tmp_path, monkeypatch):
    res, s = _result(tmp_path)
    monkeypatch.setattr(fp, "_CAN_SPAWN", True)

    class Boom:
        def __enter__(self): raise OSError("cannot spawn here")
        def __exit__(self, *a): return False

    monkeypatch.setattr(fp, "_executor", lambda: Boom())
    notes = []
    written, _ = fp.write_figures(res, s, str(tmp_path), on_fallback=notes.append)
    assert written
    assert notes and "OSError" in notes[0] and "in-process" in notes[0]


def test_falls_back_when_the_child_times_out(tmp_path, monkeypatch):
    res, s = _result(tmp_path)
    monkeypatch.setattr(fp, "_CAN_SPAWN", True)
    monkeypatch.setattr(fp, "_WORK_TIMEOUT_S", 0.001)      # anything real will exceed this
    notes = []
    written, _ = fp.write_figures(res, s, str(tmp_path), on_fallback=notes.append)
    assert written, "a timeout must not lose the figures"
    assert notes and "in-process" in notes[0]


def test_a_failed_probe_is_remembered_and_does_not_respawn(tmp_path, monkeypatch):
    """A packaged app that cannot spawn must pay for the discovery once, not per batch."""
    monkeypatch.setattr(fp, "_CAN_SPAWN", None)
    calls = []

    class Boom:
        def __enter__(self): calls.append(1); raise OSError("no")
        def __exit__(self, *a): return False

    monkeypatch.setattr(fp, "_executor", lambda: Boom())
    assert fp._can_spawn() is False
    assert fp._can_spawn() is False
    assert len(calls) == 1, "the probe re-ran instead of caching its verdict"


def test_packaged_app_never_probes_and_re_launches(monkeypatch):
    """The shipped v2.3.0 bug: in a briefcase bundle sys.executable is the app binary, so the
    probe's ProcessPoolExecutor re-launched the whole GUI TWICE (resource tracker + worker) and
    then hung for the probe timeout before figures were written in-process. When sys.executable
    is not a Python interpreter, _can_spawn must decide False WITHOUT ever spawning."""
    monkeypatch.setattr(fp, "_CAN_SPAWN", None)
    monkeypatch.setattr(fp.sys, "executable",
                        "/Applications/RespMech.app/Contents/MacOS/RespMech")
    calls = []
    monkeypatch.setattr(fp, "_executor", lambda: calls.append(1))
    assert fp._can_spawn() is False
    assert calls == [], "a packaged app spawned a probe child — the re-launch-twice bug"


def test_packaged_in_process_is_silent_but_real_failures_still_report(tmp_path, monkeypatch):
    """A packaged build writes figures in-process BY DESIGN, so it must not log a per-run
    'subprocess unavailable' note; a genuine spawn failure elsewhere still reports."""
    res, s = _result(tmp_path)
    real_exe = sys.executable          # capture BEFORE monkeypatching (fp.sys is the real sys)
    # packaged build -> writes in-process, silently
    monkeypatch.setattr(fp, "_CAN_SPAWN", None)
    monkeypatch.setattr(fp.sys, "frozen", False, raising=False)
    monkeypatch.setattr(fp.sys, "executable",
                        "/Applications/RespMech.app/Contents/MacOS/RespMech")
    (tmp_path / "pkg").mkdir()
    notes = []
    written, _ = fp.write_figures(res, s, str(tmp_path / "pkg"), on_fallback=notes.append)
    assert written, "figures must still be written in-process"
    assert notes == [], f"packaged in-process should be silent, got {notes}"

    # a genuine spawn failure on a real interpreter is still reported
    monkeypatch.setattr(fp, "_CAN_SPAWN", None)
    monkeypatch.setattr(fp.sys, "executable", real_exe)

    class Boom:
        def __enter__(self): raise OSError("no spawn here")
        def __exit__(self, *a): return False
    monkeypatch.setattr(fp, "_executor", lambda: Boom())
    (tmp_path / "fail").mkdir()
    notes2 = []
    fp.write_figures(res, s, str(tmp_path / "fail"), on_fallback=notes2.append)
    assert notes2 and "in-process" in notes2[0], "a genuine spawn failure must still be reported"


def test_spawn_relaunch_detection(monkeypatch):
    """Dev/venv installs (python) keep the child isolation; every packaged shape is detected."""
    monkeypatch.setattr(fp.sys, "frozen", False, raising=False)
    for exe in ("/opt/anaconda3/bin/python3.13", "/x/.venv/bin/python", r"C:\Py\pythonw.exe"):
        monkeypatch.setattr(fp.sys, "executable", exe)
        assert fp._spawn_relaunches_the_app() is False, exe
    for exe in ("/Applications/RespMech.app/Contents/MacOS/RespMech",       # briefcase binary
                "/Applications/RespMech.app/Contents/Resources/python",     # python INSIDE a .app
                r"C:\Prog\RespMech.exe", ""):                               # windows binary / unknown
        monkeypatch.setattr(fp.sys, "executable", exe)
        assert fp._spawn_relaunches_the_app() is True, exe
    monkeypatch.setattr(fp.sys, "executable", "/x/.venv/bin/python")        # sys.frozen also counts
    monkeypatch.setattr(fp.sys, "frozen", True, raising=False)
    assert fp._spawn_relaunches_the_app() is True


# -- the writers.py seam -------------------------------------------------------

def test_writers_reports_the_fallback_in_its_failure_list(tmp_path, monkeypatch):
    from respmech.core.io import writers
    res, s = _result(tmp_path)
    monkeypatch.setattr(fp, "_CAN_SPAWN", None)
    monkeypatch.setenv(fp._DISABLE_ENV, "1")
    written, failures = writers._write_figures(res, s, str(tmp_path))
    assert written
    assert any("in-process" in msg for _name, msg in failures), failures


def test_entry_points_call_freeze_support():
    """Without it, a spawned child of a PACKAGED app re-runs main() and launches a second GUI."""
    import inspect
    from respmech.cli import __main__ as cli
    assert "freeze_support" in inspect.getsource(cli.main)
    from respmech.ui import app
    assert "freeze_support" in inspect.getsource(app.main)


# -- a blank column is a normal outcome, not a fault ---------------------------

def test_an_all_nan_rms_column_normalises_without_warning():
    """The cardiac-gated columns are NaN for a whole file when the quality guards refuse it,
    which made per_file_max normalisation emit "All-NaN slice encountered" on every run. The
    result was already right; only the noise was wrong."""
    import warnings
    import numpy as np
    import pandas as pd
    from respmech.core import summary
    from respmech.core.settings import Settings

    s = Settings.from_dict({"processing": {"emg": {"normalization": "per_file_max"}}})
    tbl = pd.DataFrame({
        "breath_no": [1, 2, 3],
        "rms_col_2": [0.1, 0.2, 0.4],           # a normal column
        "rms_gated_col_2": [np.nan] * 3,        # refused by the guards
        "rms_gated_max": [np.nan] * 3,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)     # any All-NaN warning now fails
        out = summary.normalize_emg_table(tbl, s)

    assert out is not None
    assert out["rms_col_2_pct"].tolist() == [25.0, 50.0, 100.0]   # unchanged behaviour
    assert out["rms_gated_col_2_pct"].isna().all()                # still blank, as it should be
    assert out["rms_gated_max_pct"].isna().all()

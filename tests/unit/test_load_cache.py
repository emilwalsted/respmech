"""Wave 2.4 — the per-run load + ECG-removal cache must be invisible to results.

When shared-profile noise reduction is on, the noise-building phase (reference clip +
auto_prop) and the main loop used to load and ECG-remove the same files repeatedly — the
reference file up to three times. ``pipeline._load_and_ecg`` memoises that work per run.
These tests pin the two things that make it safe to keep on: it changes no result, and it
actually removes the duplicate calls.
"""
import os

import numpy as np
import pytest

from _helpers import requires_synth, synth_settings

pytestmark = requires_synth()


def _run(settings):
    from respmech.core.pipeline import run_batch
    return run_batch(settings)


def _tables(result):
    return {f: {str(c): np.asarray(fr.breaths_table[c].to_numpy())
                for c in fr.breaths_table.columns}
            for f, fr in result.ok_files.items()}


def test_cache_eliminates_duplicate_load_and_ecg(monkeypatch, tmp_path):
    """With noise on, the noise phase + main loop should now touch each file's load and ECG
    removal exactly once — no file loaded or ECG-removed twice."""
    from respmech.core import pipeline as P

    loads, ecgs = [], []
    _orig_load, _orig_ecg = P._load, P._ecg_remove
    monkeypatch.setattr(P, "_load", lambda path, s: (loads.append(os.path.abspath(path)),
                                                     _orig_load(path, s))[1])
    monkeypatch.setattr(P, "_ecg_remove",
                        lambda s, emg, cancel_check=None: (ecgs.append(1),
                                                           _orig_ecg(s, emg, cancel_check=cancel_check))[1])

    settings = synth_settings(str(tmp_path), noise=True)
    result = _run(settings)
    nfiles = len(result.ok_files)
    assert nfiles > 0
    # each ok file loaded and ECG-removed exactly once
    assert len(loads) == len(set(loads)), f"a file was loaded more than once: {loads}"
    assert len(ecgs) == nfiles, f"expected {nfiles} ECG removals, got {len(ecgs)}"


def test_cache_does_not_change_results(tmp_path, monkeypatch):
    """Same batch, cache disabled (bound = 0) vs enabled, must be bit-identical."""
    from respmech.core import pipeline as P

    settings = synth_settings(str(tmp_path / "on"), noise=True)
    with_cache = _tables(_run(settings))

    monkeypatch.setattr(P, "_LOAD_CACHE_MAX", 0)          # never stores → every call recomputes
    settings2 = synth_settings(str(tmp_path / "off"), noise=True)
    without_cache = _tables(_run(settings2))

    assert set(with_cache) == set(without_cache)
    for f in with_cache:
        assert set(with_cache[f]) == set(without_cache[f])
        for c in with_cache[f]:
            x, y = with_cache[f][c], without_cache[f][c]
            ok = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y)
            assert ok, f"{f}/{c} differs with the cache on"


def test_no_cache_when_noise_off(monkeypatch, tmp_path):
    """No shared-profile noise → no noise-building phase → the cache is never populated, and
    every file is loaded exactly once (the plain path, unchanged)."""
    from respmech.core import pipeline as P

    loads = []
    _orig_load = P._load
    monkeypatch.setattr(P, "_load", lambda path, s: (loads.append(os.path.abspath(path)),
                                                     _orig_load(path, s))[1])
    settings = synth_settings(str(tmp_path))               # noise=False by default
    result = _run(settings)
    assert len(loads) == len(result.ok_files) == len(set(loads))

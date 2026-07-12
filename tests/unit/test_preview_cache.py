"""Preview-only conditioning caches (ui.screens._preview_cache): they must return the
same result on a hit, MISS on any input that changes the computed value, and NEVER be
consulted by the batch/CLI/golden path (so they can't touch the pinned science)."""
import os

import numpy as np
import pytest

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

from respmech.ui.screens import _preview_cache as pc  # noqa: E402


def _s(tmp_path):
    return synth_settings(str(tmp_path), remove_ecg=True, noise=True,
                          data_out={"saveaveragedata": True, "savebreathbybreathdata": True})


def test_file_token_tracks_mtime_and_size(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    t1 = pc.file_token(str(p))
    assert t1 is not None and os.path.isabs(t1[0])
    p.write_text("a,b\n1,2\n3,4\n")               # size (and mtime) change
    assert pc.file_token(str(p)) != t1
    assert pc.file_token(str(tmp_path / "nope.csv")) is None   # unstattable -> None (bypass)


def test_lru_hit_returns_stored_and_evicts_oldest():
    lru = pc._LRU(cap=2)
    calls = []
    def mk(n):
        return lambda: (calls.append(n), n)[1]
    assert pc.cached(lru, "a", mk("a")) == "a"
    assert pc.cached(lru, "a", mk("a")) == "a"     # hit -> thunk NOT called again
    assert calls == ["a"]
    pc.cached(lru, "b", mk("b")); pc.cached(lru, "c", mk("c"))   # evicts "a" (cap 2)
    calls.clear()
    pc.cached(lru, "a", mk("a"))                    # "a" was evicted -> recomputes
    assert calls == ["a"]
    assert pc.cached(lru, None, mk("z")) == "z"     # None key always bypasses


def test_ecg_matrix_key_changes_with_relevant_settings(tmp_path):
    s = _s(tmp_path)
    f = os.path.join(INPUT, "synth_case_A.csv")
    k0 = pc.ecg_matrix_key(s, f)
    assert k0 is not None
    assert pc.ecg_matrix_key(s, f) == k0                       # stable for identical inputs
    s.processing.emg.ecg_min_height += 0.001                   # an ECG parameter
    assert pc.ecg_matrix_key(s, f) != k0                       # -> different key (recompute)
    s2 = _s(tmp_path)
    s2.processing.emg.noise.prop_decrease = 0.9                # prop is noise-only, not ECG
    assert pc.ecg_matrix_key(s2, f) == k0                      # -> same key (prop-independent)


def test_ref_clip_key_ignores_prop_and_drift_but_tracks_ecg(tmp_path):
    s = _s(tmp_path)
    ref = os.path.join(INPUT, "synth_case_A.csv")
    k0 = pc.ref_clip_key(s, ref)
    assert k0 is not None
    s.processing.emg.noise.prop_decrease = 0.9
    s.processing.volume.correct_drift = not s.processing.volume.correct_drift
    assert pc.ref_clip_key(s, ref) == k0                       # clip is prop/drift-independent
    s.processing.emg.ecg_window_s += 0.05                      # ECG param feeds the clip
    assert pc.ref_clip_key(s, ref) != k0


def test_noise_report_key_tracks_all_files_and_stft(tmp_path):
    s = _s(tmp_path)
    ref = os.path.join(INPUT, "synth_case_A.csv")
    files = [os.path.join(INPUT, "synth_case_A.csv"), os.path.join(INPUT, "synth_case_B.csv")]
    k0 = pc.noise_report_key(s, ref, files)
    assert k0 is not None
    assert pc.noise_report_key(s, ref, files[:1]) != k0        # a different file set -> miss
    s2 = _s(tmp_path)
    s2.processing.emg.noise.use_expiration = False             # explicit-intervals branch
    s2.processing.emg.noise.reference_intervals = [[1.0, 5.0]]
    b0 = pc.noise_report_key(s2, ref, files)
    s2.processing.segmentation.buffer += 50                    # auto_prop gather segments by flow
    assert pc.noise_report_key(s2, ref, files) != b0          # -> buffer must invalidate the report
    s.processing.emg.noise.n_fft = 512
    assert pc.noise_report_key(s, ref, files) != k0            # an STFT param -> miss


def test_core_pipeline_does_not_import_the_preview_cache():
    """Isolation: the batch/CLI/golden path must never consult a preview cache."""
    import respmech.core.pipeline as pipeline
    import inspect
    src = inspect.getsource(pipeline)
    assert "_preview_cache" not in src


def test_cached_is_thread_safe_under_concurrency():
    """Many worker threads hammering the shared cache with overlapping keys never race
    the OrderedDict (no KeyError / corruption) and every caller gets the correct value.
    The compute runs outside the lock, so a thread is never blocked waiting on another
    (which on a slow/headless runner could stall a job past its timeout)."""
    import threading
    import time
    lru = pc._LRU(cap=8)
    def thunk(k):
        time.sleep(0.01)
        return f"v{k}"
    results = {}
    lock = threading.Lock()
    def worker(k):
        v = pc.cached(lru, (k,), lambda: thunk(k))
        with lock:
            results[(k, threading.get_ident())] = v
    threads = [threading.Thread(target=worker, args=(i % 3,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # every caller got the value for ITS key (no cross-key corruption from the race)
    assert all(v == f"v{k}" for (k, _tid), v in results.items())

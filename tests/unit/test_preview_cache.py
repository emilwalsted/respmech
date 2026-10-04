"""Preview-only conditioning caches (ui.screens._preview_cache): they must return the
same result on a hit, MISS on any input that changes the computed value, and NEVER be
consulted by the batch/CLI/golden path (so they can't touch the pinned science)."""
import os

import numpy as np
import pytest

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

from respmech.ui.screens import _preview_cache as pc  # noqa: E402
from respmech.core.settings import resolve_noise_reference_mode  # noqa: E402


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


def test_ref_clip_key_tracks_an_exclude_entrys_folder_in_the_expiration_branch(tmp_path):
    """Ticket B06: ExcludeEntry gained a `folder` provenance tag. It never changes which
    breaths compute excludes (core.compute keys purely on filename), but the entry it
    lives on can now change independently of (file, breaths) — e.g. a folder restamp with
    an unchanged breath set (see preview/_mechanics.py._toggle_breath) — so _exclude_key
    must carry it too, in the expiration branch where the clip depends on which breaths
    are masked out."""
    from respmech.core.settings import ExcludeEntry
    s = _s(tmp_path)
    s.processing.emg.noise.use_expiration = True      # force the branch _exclude_key feeds
    ref = os.path.join(INPUT, "synth_case_A.csv")
    s.processing.exclude_breaths.append(ExcludeEntry(file="synth_case_A.csv", breaths=[1],
                                                      folder="/data/S01"))
    k0 = pc.ref_clip_key(s, ref)
    assert k0 is not None
    s.processing.exclude_breaths[0].folder = "/data/S02"     # breaths unchanged, folder restamped
    assert pc.ref_clip_key(s, ref) != k0


def _emg_only_settings(tmp_path):
    return synth_settings(str(tmp_path), channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})


def test_ref_clip_key_expiration_branch_tracks_segmentation_method(tmp_path):
    """The expiration branch carries segmentation.method: since M-23, _emg_segmented's
    mask actually follows the configured method (it hardcoded 'flow' before), so a
    settings change here must miss."""
    s = _s(tmp_path)
    s.processing.emg.noise.use_expiration = True
    ref = os.path.join(INPUT, "synth_case_A.csv")
    k0 = pc.ref_clip_key(s, ref)
    assert k0 is not None
    s.processing.segmentation.method = "volume"
    assert pc.ref_clip_key(s, ref) != k0


def test_ref_clip_key_expiration_branch_tracks_volume_trend(tmp_path):
    """M-23: _emg_segmented's mask now also follows processing.volume.correct_trend (it
    never trend-corrected before), so toggling it -- or, once enabled, changing HOW it
    corrects -- must miss instead of serving a preview cached under the old trend."""
    s = _s(tmp_path)
    s.processing.emg.noise.use_expiration = True
    ref = os.path.join(INPUT, "synth_case_A.csv")
    k0 = pc.ref_clip_key(s, ref)
    assert k0 is not None
    s.processing.volume.correct_trend = True
    k1 = pc.ref_clip_key(s, ref)
    assert k1 != k0
    s.processing.volume.trend_method = "nearest"
    assert pc.ref_clip_key(s, ref) != k1


def test_ref_clip_key_expiration_branch_tracks_volume_drift_under_volume_method(tmp_path):
    """M-23: with segmentation.method == 'volume', _emg_segmented's mask is built by
    separateintobreathsbyvolume's peak search against the DRIFT-corrected volume (an
    absolute-height threshold), so drift correction can change which samples are
    detected as inspiratory/expiratory peaks even with trend correction untouched --
    unlike the flow-method case, where drift only changes a breath's reported VOLUME
    values, never its time boundaries. correct_drift and the peak thresholds must both
    invalidate the cache here."""
    s = _s(tmp_path)
    s.processing.emg.noise.use_expiration = True
    s.processing.segmentation.method = "volume"
    ref = os.path.join(INPUT, "synth_case_A.csv")
    k0 = pc.ref_clip_key(s, ref)
    assert k0 is not None
    s.processing.volume.correct_drift = False
    k1 = pc.ref_clip_key(s, ref)
    assert k1 != k0
    s.processing.segmentation.peak.height = 0.5
    assert pc.ref_clip_key(s, ref) != k1


def test_noise_report_key_auto_prop_tracks_volume_trend_and_drift_in_intervals_mode(tmp_path):
    """M-23: _build_noise_set's auto_prop gather segments EVERY batch file through
    segment_file with the globally configured method/drift/trend/peak settings,
    regardless of which mode the single reference CLIP resolves to. A test whose
    reference resolves to 'intervals' (the schema default noise=True gives) must still
    miss on a drift/trend edit once auto_prop is on, or the noise-fidelity report can be
    served stale after an edit that has nothing to do with the interval span itself."""
    s = _s(tmp_path)
    assert resolve_noise_reference_mode(s) == "intervals"
    s.processing.emg.noise.auto_prop = True
    files = [os.path.join(INPUT, "synth_case_A.csv"), os.path.join(INPUT, "synth_case_B.csv")]
    ref = os.path.join(INPUT, "synth_case_A.csv")
    k0 = pc.noise_report_key(s, ref, files)
    assert k0 is not None
    s.processing.volume.correct_trend = True
    k1 = pc.noise_report_key(s, ref, files)
    assert k1 != k0
    s.processing.volume.correct_drift = False
    assert pc.noise_report_key(s, ref, files) != k1


def test_ref_clip_key_rest_segments_branch_tracks_separators_and_a_rest_kind(tmp_path):
    """M-24's own acceptance criterion: two settings differing only in a rest-kind or one
    separator time give different ref_clip_keys in the rest_segments state."""
    from respmech.core.settings import BreathTypeEntry, SeparatorEntry
    s = _emg_only_settings(tmp_path)
    ref = os.path.join(INPUT, "synth_case_A.csv")
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=1, kind="rest"))
    k0 = pc.ref_clip_key(s, ref)
    assert k0 is not None

    s.processing.segmentation.separators.append(
        SeparatorEntry(file="synth_case_A.csv", times_s=[1.0]))
    k1 = pc.ref_clip_key(s, ref)
    assert k1 != k0                                    # a separator appeared -> miss

    s.processing.segmentation.separators[0].times_s = [2.0]
    k2 = pc.ref_clip_key(s, ref)
    assert k2 != k1                                    # the SAME separator moved -> miss

    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=2, kind="rest"))
    k3 = pc.ref_clip_key(s, ref)
    assert k3 not in (k0, k1, k2)                       # a second rest-typed segment -> miss


def test_ref_clip_key_keys_on_the_resolved_mode_not_the_unused_flag(tmp_path):
    """For an EMG-only set, use_expiration is never touched (stays at its True default) and
    is meaningless to the resolver — keying on the RESOLVED mode instead means switching
    from a rest-typed reference to an explicit interval is a real branch change (a miss),
    even though the raw flag never moved."""
    from respmech.core.settings import BreathTypeEntry
    s = _emg_only_settings(tmp_path)
    ref = os.path.join(INPUT, "synth_case_A.csv")
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    assert s.processing.emg.noise.use_expiration is True
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=1, kind="rest"))
    k_rest = pc.ref_clip_key(s, ref)
    s.processing.breath_types.clear()
    s.processing.emg.noise.reference_intervals = [[1.0, 2.0]]
    assert s.processing.emg.noise.use_expiration is True   # untouched by either branch
    k_intervals = pc.ref_clip_key(s, ref)
    assert k_rest != k_intervals


def test_separators_for_and_kinds_for_are_file_scoped(tmp_path):
    from respmech.core.settings import BreathTypeEntry, SeparatorEntry
    s = _emg_only_settings(tmp_path)
    s.processing.segmentation.separators.append(
        SeparatorEntry(file="synth_case_A.csv", times_s=[1.0, 2.0]))
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=1, kind="rest"))
    assert pc._separators_for(s, "synth_case_A.csv") == (1.0, 2.0)
    assert pc._separators_for(s, "synth_case_B.csv") == ()
    assert pc._kinds_for(s, "synth_case_A.csv") == ((1, "rest"),)
    assert pc._kinds_for(s, "synth_case_B.csv") == ()


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

"""Preview-only memoisation of the heavy EMG/noise staging.

The Preview screen recomputes several panels together on a scoped settings change
(emg_all + emg_detail + noise), and each independently loads + ECG-removes the same
file and rebuilds the same reference noise clip. These LRU caches deduplicate that
shared work WITHIN a recompute cycle (and across revisits of the same settings).

ISOLATION GUARANTEE — these caches are consulted ONLY from ``respmech.ui.workers``
(the preview staging functions). ``respmech.core`` (``run_batch`` / the CLI / the golden
harness) imports nothing from here and always recomputes from the real Settings, so a
cache can never influence the pinned scientific output.

KEY COMPLETENESS — a key must carry EVERY input that feeds the stored value, or a stale
entry would be shown to the user. Each key includes a per-file freshness token
(abspath + mtime_ns + size) because an in-place data edit fires no settings/refresh hook,
plus the exact settings fields the computation reads (verified against the stage_* code).
Erring toward MORE key fields only costs a cache miss; omitting one risks a stale panel.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict

_CAP = 4   # entries per cache — a handful of recently-tuned files is plenty

# The preview launches emg_all + emg_detail + noise on SEPARATE QThreads that hit these
# module-level caches concurrently, so every cache access is serialised under one lock and
# a per-key in-flight registry gives single-flight (concurrent same-key misses compute the
# shared reference clip ONCE, not three times — the whole point of the cache).
_lock = threading.Lock()
_inflight = {}   # cache key -> threading.Event, set when the owning thread finishes computing


class _LRU:
    def __init__(self, cap=_CAP):
        self._d = OrderedDict()
        self._cap = cap

    def get(self, key, default=None):
        with _lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return default

    def put(self, key, value):
        with _lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self._cap:
                self._d.popitem(last=False)

    def clear(self):
        with _lock:
            self._d.clear()


# one instance per distinct cached quantity
_REF_CLIP = _LRU()          # reference noise clip (multichannel), prop-independent
_ECG_MATRIX = _LRU()        # (raw_matrix, ecg_removed_matrix, applied, error) for a file
_NOISE_REPORT = _LRU()      # stage_noise_fidelity's report (test-wide)

_SENTINEL = object()


def cached(cache, key, thunk):
    """Return ``cache[key]`` or compute it with ``thunk()`` and store it — thread-safe and
    single-flight, so concurrent worker threads that miss the SAME key compute it once and
    the rest reuse the result. A ``None`` key (e.g. an unstattable file) bypasses the cache
    entirely (always recompute)."""
    if key is None:
        return thunk()
    hit = cache.get(key, _SENTINEL)
    if hit is not _SENTINEL:
        return hit
    # register (or find) the in-flight compute for this key
    with _lock:
        hit = cache._d.get(key, _SENTINEL)   # re-check under the lock (a put may have raced in)
        if hit is not _SENTINEL:
            cache._d.move_to_end(key)
            return hit
        ev = _inflight.get(key)
        owner = ev is None
        if owner:
            ev = _inflight[key] = threading.Event()
    if not owner:                            # another thread is computing this key -> wait for it
        ev.wait()
        hit = cache.get(key, _SENTINEL)
        return hit if hit is not _SENTINEL else thunk()   # recompute only if it was evicted/failed
    try:
        val = thunk()
        cache.put(key, val)
        return val
    finally:
        with _lock:
            _inflight.pop(key, None)
        ev.set()                             # release any waiters (they read the cache, or recompute)


def clear_all():
    """Drop every preview cache (called when the input folder/mask changes so a rebuilt
    file list can never collide with a stale entry — belt-and-braces on top of the
    freshness tokens in the keys)."""
    _REF_CLIP.clear()
    _ECG_MATRIX.clear()
    _NOISE_REPORT.clear()


# --- freshness token + key builders ----------------------------------------
def file_token(path):
    """(abspath, mtime_ns, size) — or None if the file cannot be stat'd (then the caller
    bypasses the cache and recomputes, never serving a stale entry)."""
    try:
        ap = os.path.abspath(path)
        st = os.stat(ap)
        return (ap, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _load_key(settings):
    """The inputs load() applies (channel mapping + format + flow/volume transforms)."""
    ch = settings.input.channels
    fmt = settings.input.format
    vol = settings.processing.volume
    return (tuple(ch.emg), tuple(ch.entropy), ch.flow, ch.volume, ch.poes, ch.pgas, ch.pdi,
            fmt.sampling_frequency, fmt.decimal, fmt.matlab_variant,
            vol.inverse_flow, vol.integrate_from_flow, vol.inverse_volume)


def _ecg_key(settings):
    e = settings.processing.emg
    return (e.remove_ecg, e.detect_channel, e.ecg_min_height, e.ecg_min_distance_s,
            e.ecg_min_width_s, e.ecg_window_s)


def _exclude_key(settings):
    return tuple((x.file, tuple(x.breaths)) for x in settings.processing.exclude_breaths)


def ecg_matrix_key(settings, file_path):
    """Key for a file's ECG-removed EMG matrix (prop-independent): file freshness +
    load inputs + ECG-removal parameters. Independent of noise/segmentation."""
    tok = file_token(file_path)
    if tok is None:
        return None
    return ("ecg", tok, _load_key(settings), _ecg_key(settings))


def ref_clip_key(settings, ref_path):
    """Key for the reference noise clip. Branch-split on use_expiration: the expiration
    branch depends on breath segmentation; the explicit-intervals branch does not. Excludes
    the noise STFT params + prop_decrease (the clip is prop/profile-independent) and volume
    drift/trend (the EMG clip comes from flow-based masks, unaffected by drift correction)."""
    tok = file_token(ref_path)
    if tok is None:
        return None
    n = settings.processing.emg.noise
    branch_a = bool(n.use_expiration or not n.reference_intervals)
    base = ("refclip", tok, _load_key(settings), _ecg_key(settings), branch_a)
    if branch_a:
        return base + (settings.processing.segmentation.buffer, _exclude_key(settings))
    return base + (tuple(tuple(iv) for iv in n.reference_intervals),
                   settings.input.format.sampling_frequency)


def noise_report_key(settings, ref_path, files):
    """Key for the test-wide noise report: the reference-clip key (load/ECG/segmentation +
    ref file), a freshness token for EVERY file in the auto_prop gather set, the STFT
    params, and auto/target (+ the fixed prop only when auto_prop is off)."""
    rc = ref_clip_key(settings, ref_path)
    if rc is None:
        return None
    toks = tuple(file_token(os.path.abspath(f)) for f in files)
    if any(t is None for t in toks):
        return None
    n = settings.processing.emg.noise
    key = ("noise", rc, toks, n.n_fft, n.hop_length, n.win_length, n.n_std_thresh,
           n.n_grad_freq, n.n_grad_time, bool(n.auto_prop), n.fidelity_target,
           (None if n.auto_prop else n.prop_decrease))
    if n.auto_prop:
        # the auto_prop gather (pipeline._build_noise_set) segments EVERY file by flow to
        # pick prop_decrease, which reads segmentation.buffer — and ref_clip_key only carries
        # it in the expiration branch. Add it here so a buffer edit invalidates the report in
        # the explicit-intervals branch too (else a stale fidelity frontier is shown).
        key += (settings.processing.segmentation.buffer,)
    return key


# expose the cache instances for the workers + tests
REF_CLIP = _REF_CLIP
ECG_MATRIX = _ECG_MATRIX
NOISE_REPORT = _NOISE_REPORT

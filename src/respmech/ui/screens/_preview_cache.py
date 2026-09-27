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

from respmech.core.settings import resolve_noise_reference_mode

_CAP = 4   # entries per cache — a handful of recently-tuned files is plenty

# The preview launches emg_all + emg_detail + noise on SEPARATE QThreads that hit these
# module-level caches concurrently, so every read/write of the shared OrderedDict is
# serialised under one short-held lock (the slow compute runs OUTSIDE the lock).
_lock = threading.Lock()


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
_ECG_REDUCTION = _LRU()     # stage_ecg_reduction's result (raw capture + processed + R-peaks)
                            # DEDICATED — never reuse _ECG_MATRIX (it gates on remove_ecg, stores no peaks)

_SENTINEL = object()


def cached(cache, key, thunk):
    """Return ``cache[key]`` or compute it with ``thunk()`` and store it. Thread-safe: the
    get/put touch the shared OrderedDict under a lock, but the (slow) ``thunk`` runs WITHOUT
    the lock so the emg_all/emg_detail/noise worker threads stay parallel and a cancelled
    worker is never blocked waiting on another. Concurrent misses of the same key may each
    compute (last write wins — same value); the cache still dedupes across recompute cycles.
    A ``None`` key (e.g. an unstattable file) bypasses the cache entirely (always recompute)."""
    if key is None:
        return thunk()
    hit = cache.get(key, _SENTINEL)
    if hit is not _SENTINEL:
        return hit
    val = thunk()
    cache.put(key, val)
    return val


def clear_all():
    """Drop every preview cache (called when the input folder/mask changes so a rebuilt
    file list can never collide with a stale entry — belt-and-braces on top of the
    freshness tokens in the keys)."""
    _REF_CLIP.clear()
    _ECG_MATRIX.clear()
    _NOISE_REPORT.clear()
    _ECG_REDUCTION.clear()


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
    # x.folder does not itself change which breaths compute ignores (core.compute keys
    # purely on filename — see core/_legacy_ns.py), but it is now part of an ExcludeEntry's
    # state and the file it feeds into can change independently of (file, breaths) — e.g.
    # a "Clear" on the carried-over banner removing the entry entirely, which (file,
    # breaths) alone already invalidates, but a folder restamp with an unchanged breath set
    # (a click that only confirms, never edits, the current selection) would not. Include
    # it so a cache hit can never silently outlive either kind of change.
    #
    # M-20: processing.breath_types is unioned into core's own ignored-breath set exactly
    # like exclude_breaths is (core._legacy_ns.to_legacy_ns), so a typed breath changes the
    # SAME cached mechanics/EMG output a manual exclusion would — a cache key that only
    # watched exclude_breaths would silently serve a stale preview after typing a breath.
    return (tuple((x.file, tuple(x.breaths), x.folder) for x in settings.processing.exclude_breaths),
           tuple((t.file, t.breath, t.kind, t.folder) for t in settings.processing.breath_types))


def ecg_matrix_key(settings, file_path):
    """Key for a file's ECG-removed EMG matrix (prop-independent): file freshness +
    load inputs + ECG-removal parameters. Independent of noise/segmentation."""
    tok = file_token(file_path)
    if tok is None:
        return None
    return ("ecg", tok, _load_key(settings), _ecg_key(settings))


def _separators_for(settings, filename):
    """``filename``'s manual segment-boundary times (M-21's ``SeparatorEntry``,
    ``processing.segmentation.separators``) -- part of :func:`ref_clip_key`'s
    ``rest_segments`` branch: the file's rest-typed segments are numbered relative to
    THESE boundaries (``core.pipeline.segment_file``), so a boundary edit changes which
    samples 'segment 2' even is, without touching a single kind."""
    for se in settings.processing.segmentation.separators:
        if se.file == filename:
            return tuple(se.times_s)
    return ()


def _kinds_for(settings, filename):
    """``filename``'s typed-breath (here: typed-SEGMENT) kinds (M-19's ``BreathTypeEntry``,
    ``processing.breath_types``) -- part of :func:`ref_clip_key`'s ``rest_segments`` branch:
    which segment numbers are 'rest' (and so feed the clip) can change independently of the
    separator times above."""
    return tuple(sorted((t.breath, t.kind) for t in settings.processing.breath_types
                        if t.file == filename))


def ref_clip_key(settings, ref_path):
    """Key for the reference noise clip. Branch-split on the RESOLVED mode (M-22's
    ``resolve_noise_reference_mode`` -- M-24), not the raw ``use_expiration``/
    ``reference_intervals`` pair directly: an EMG-only set ignores ``use_expiration``
    entirely (see the resolver's own docstring), so keying on it directly could hand out a
    stale hit across two settings the resolver treats identically, or (worse) two settings
    it resolves DIFFERENTLY as though they were the same.

    * ``expiration``: depends on breath segmentation + which breaths are excluded/typed
      (the clip is built from a flow-bearing file's own quiet-expiration mask).
    * ``rest_segments``: depends on the SAME segmentation method/buffer (``segment_file``
      builds the segments the same way the analysis itself will), plus which of THIS file's
      segments are separated where (:func:`_separators_for`) and typed 'rest'
      (:func:`_kinds_for`) -- two settings differing in only one separator time or one
      segment's kind must miss.
    * ``intervals``: an explicit ``[t0, t1]`` span plus the sampling rate that turns it into
      sample indices -- independent of segmentation entirely.
    * ``interburst``/``unresolved``: no clip-building implementation exists for either yet
      (:func:`respmech.core.pipeline._reference_noise_clip` raises for both, and
      ``Settings.validate()`` rejects them before a batch can even start) -- nothing beyond
      the common base below is meaningful to key on.

    Excludes the noise STFT params + prop_decrease (the clip is prop/profile-independent)
    and volume drift/trend (the EMG clip comes from flow-based masks, unaffected by drift
    correction) in every branch, as before."""
    tok = file_token(ref_path)
    if tok is None:
        return None
    mode = resolve_noise_reference_mode(settings)
    base = ("refclip", tok, _load_key(settings), _ecg_key(settings), mode)
    seg = settings.processing.segmentation
    if mode == "expiration":
        # `seg.method` costs nothing extra today (`_emg_segmented` hardcodes 'flow'
        # regardless of the configured method until M-23 lands) but is included now so a
        # cache entry keyed BEFORE that fix can never be served stale once the mask-building
        # actually starts reading it.
        return base + (seg.buffer, seg.method, _exclude_key(settings))
    if mode == "rest_segments":
        name = os.path.basename(ref_path)
        return base + (seg.method, seg.buffer, _separators_for(settings, name),
                       _kinds_for(settings, name), _exclude_key(settings))
    if mode == "intervals":
        n = settings.processing.emg.noise
        return base + (tuple(tuple(iv) for iv in n.reference_intervals),
                       settings.input.format.sampling_frequency)
    return base                          # 'interburst' / 'unresolved' -- see docstring


def noise_report_key(settings, ref_path, files):
    """Key for the test-wide noise report: the reference-clip key (load/ECG/segmentation +
    ref file), a freshness token for EVERY file in the auto_prop gather set, the STFT
    params, the segmentation method, and auto/target (+ the fixed prop only when auto_prop
    is off)."""
    rc = ref_clip_key(settings, ref_path)
    if rc is None:
        return None
    toks = tuple(file_token(os.path.abspath(f)) for f in files)
    if any(t is None for t in toks):
        return None
    n = settings.processing.emg.noise
    seg = settings.processing.segmentation
    key = ("noise", rc, toks, n.n_fft, n.hop_length, n.win_length, n.n_std_thresh,
           n.n_grad_freq, n.n_grad_time, bool(n.auto_prop), n.fidelity_target,
           (None if n.auto_prop else n.prop_decrease), seg.method)
    if n.auto_prop:
        # the auto_prop gather (pipeline._build_noise_set) segments EVERY file by flow to
        # pick prop_decrease, which reads segmentation.buffer — and ref_clip_key only carries
        # it in the expiration branch. Add it here so a buffer edit invalidates the report in
        # the explicit-intervals branch too (else a stale fidelity frontier is shown).
        #
        # M-24: also carry each gathered file's OWN separators/typed-kinds. auto_prop is
        # rejected by Settings.validate() for an EMG-only signal set today, so this is inert
        # in practice (no set that can carry separators/breath_types can also reach here with
        # auto_prop=True) -- included for completeness (KEY COMPLETENESS above) rather than
        # assume that pairing can never change, since ref_clip_key itself only ever sees ONE
        # file (the reference), never the whole gather set this report is keyed on.
        names = tuple(os.path.basename(f) for f in files)
        key += (seg.buffer,
               tuple(_separators_for(settings, name) for name in names),
               tuple(_kinds_for(settings, name) for name in names))
    return key


# expose the cache instances for the workers + tests
REF_CLIP = _REF_CLIP
ECG_MATRIX = _ECG_MATRIX
NOISE_REPORT = _NOISE_REPORT
ECG_REDUCTION = _ECG_REDUCTION

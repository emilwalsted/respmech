"""Batch pipeline: orchestrates load -> condition -> segment -> compute for each
file, emitting progress events and returning typed results.

This is the seam that makes the CLI and the Qt GUI share one engine: the core emits
:class:`ProgressEvent` objects (via a callback); the CLI prints them, the GUI turns
them into Qt signals. The core performs **no file writing and no plotting** — those
are consumers of the returned results (see ``respmech.core.io.writers`` and
``respmech.core.plots``).

Bug fixes applied here vs legacy ``analyse`` (documented):
* #1 the always-on EMG diagnostic plot (which crashed on excluded breaths) is not in
  the compute path at all — plotting is a separate, optional consumer.
* #2 ``entropycolumns`` is trimmed together with the other channels, so entropy is
  computed on aligned data (legacy left it untrimmed but indexed it with trimmed
  coordinates).
Trim precondition failures raise :class:`~respmech.core.compute.TrimError` with a
clear message rather than a generic trace.
"""
from __future__ import annotations

import fnmatch
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from respmech.core import compute
from respmech.core import emg as emglib
from respmech.core.analysis import lungvol as lungvollib
from respmech.core.analysis import normalisation as normalisationlib
from respmech.core.analysis import manoeuvres as manoeuvreslib
from respmech.core.analysis import mfvl as mfvllib
from respmech.core.analysis import pressure as pressurelib
from respmech.core.analysis import references as referenceslib
from respmech.core.analysis.signals import Capabilities
from respmech.core.io.loaders import load
from respmech.core.results import build_breath_table, build_manoeuvre_table, build_processed_data
from respmech.core.settings import Settings, resolve_noise_reference_mode
from respmech.core._legacy_ns import to_legacy_ns


@dataclass
class ProgressEvent:
    kind: str                      # file_start|stage|breath|file_done|file_error|writing|finished
    file: Optional[str] = None
    message: str = ""
    breath: Optional[int] = None
    total_breaths: Optional[int] = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class FileResult:
    file: str
    breaths_table: object = None       # per-breath DataFrame
    average_row: object = None         # 1-row DataFrame
    processed: object = None           # processed-data DataFrame or None
    breaths: object = None             # raw computed breaths (for plotting)
    ecg: object = None                 # ECG-removal diagnostics (n_peaks, suppression)
    signals: object = None             # diagnostic signal arrays for the plotting/audio consumer
    # M-29: {breath_no: core.analysis.manoeuvres.extract(...) result} for every TYPED
    # breath in this file (rest excluded -- see run_batch), and its Manoeuvres-sheet
    # DataFrame (None when manoeuvres is empty -- "the sheet exists only when a typed
    # breath is present", the ticket's own acceptance criterion).
    manoeuvres: dict = field(default_factory=dict)
    manoeuvres_table: object = None
    # M-30: 'tidal' (the ordinary shape -- a breaths_table/average_row from tidal
    # breathing) or 'reference' (every breath in this file is typed, none tidal --
    # a dedicated IC/FVC recording, say). breaths_table/average_row are both None
    # for a 'reference' file; manoeuvres/manoeuvres_table carry its actual content.
    role: str = "tidal"
    error: Optional[str] = None
    # exception class name, so consumers can tell a precondition failure of THIS recording
    # (TrimError, VolumeTrendError) from a real fault without parsing ``error``.
    error_kind: Optional[str] = None
    # Per-file quality notices (ecg_auto_detect mismatch, cardiac-gated peak refused):
    # the SAME text also raised via warnings.warn below, which reaches a stderr
    # nobody sees in a packaged app -- this is what lets
    # core.io.writers._write_run_report put the ecg_auto_detect quality check and the
    # cardiac-gated peak's NaN reason where an app user can actually read them.
    notices: list = field(default_factory=list)
    # Cross-file references actually resolved FOR this file (core.analysis.references.
    # attach, the batch's post-loop pass): {slot: {'source', 'breaths', 'n', 'value'}},
    # one entry per reference slot that resolved -- 'ic' only for now (fvc/baseline_ic/
    # max_insp have no consuming column yet, see references.attach's own docstring). A
    # slot with no resolution at all (nothing named it, or the family itself is absent
    # from this analysis) is simply not a key here -- there is nothing to report.
    references_used: dict = field(default_factory=dict)
    # The "Pressure normalised" table (core.analysis.normalisation.attach, opt-in via
    # processing.pressure.normalization.enabled): one row per tidal breath, the inspiratory
    # Poes/Pdi/EMG against the file's maximal-effort reference plus the tension-time indices.
    # None when the analysis is off or nothing could be normalised.
    pressure_normalised: object = None


@dataclass
class BatchResult:
    files: dict = field(default_factory=dict)   # filename -> FileResult
    average_table: object = None                # concatenated average rows
    noise_report: object = None                 # shared-profile prop + per-channel fidelity/ΔSNR
    ecg_auto_report: object = None              # auto-detected ECG settings + diagnostics (or None)
    # Manoeuvre extractions for reference SOURCE files that are not themselves part of
    # this run's own file list (the batch's forepass, run BEFORE the main loop --
    # core.analysis.references.external_reference_sources/attach):
    # {source_filename: {breath_no: core.analysis.manoeuvres.extract(...) result}}.
    # A source that failed entirely (could not load/segment) has NO entry here at all
    # -- see reference_errors below -- so `references.get(name)` alone can never be
    # mistaken for "this source resolved with zero typed breaths" (an empty dict IS a
    # valid, if unusual, outcome: a matched source with no typed breaths in it).
    references: dict = field(default_factory=dict)
    # Every forepass failure, keyed (source_filename, breath_no_or_None): a whole
    # source file that failed to load/segment is keyed with breath_no=None; one typed
    # breath's own extraction failing (the rest of that source still usable) is keyed
    # with its breath number, mirroring the main loop's own per-breath manoeuvre-
    # extraction try/except. Values are "ExceptionType: message" strings, the same
    # shape FileResult.error already uses.
    reference_errors: dict = field(default_factory=dict)
    # Batch-wide summary of cross-file reference resolution, built once by
    # core.analysis.references.attach() -- see that function's own docstring for the
    # shape. Empty ({}) until attach() runs (a fresh BatchResult, or a run that never
    # reaches the post-loop pass, e.g. cancelled early).
    analysis_plan: dict = field(default_factory=dict)

    @property
    def ok_files(self):
        return {k: v for k, v in self.files.items() if v.error is None}

    @property
    def failed_files(self):
        return {k: v for k, v in self.files.items() if v.error is not None}


def _emit(cb: Optional[ProgressCallback], event: ProgressEvent):
    if cb is not None:
        cb(event)


# --- pre-analysis resampling (all channels -> one common rate) ---------------
# Gated on ``processing.sampling.resample``; default OFF ⇒ byte-identical (golden-safe).
# One common rate keeps a single time base, so breath segmentation (on flow) and the
# EMG sliced by the same sample indices stay consistent. See the resampling-revival note.

def _resample_channels(data, fs_in, fs_out):
    """Anti-aliased polyphase resample of every loaded channel from fs_in to fs_out.
    ``data`` is the 7-tuple returned by ``load``; 1-D signals and the 2-D entropy/EMG
    matrices are all resampled along the time axis (0)."""
    from scipy import signal as _sig
    g = np.gcd(int(fs_out), int(fs_in))
    up, down = int(fs_out) // g, int(fs_in) // g

    def rs(x):
        if x is None:
            return x
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            return arr
        return _sig.resample_poly(arr, up, down, axis=0)

    flow, volume, poes, pgas, pdi, ent, emg = data
    return (rs(flow), rs(volume), rs(poes), rs(pgas), rs(pdi),
            rs(ent) if len(ent) else ent, rs(emg) if len(emg) else emg)


def _first_present_length(*arrays) -> int:
    """The sample count of the first non-empty array among ``arrays`` — used to build a
    raw time axis without assuming any one particular channel (conventionally flow) is
    present. Every channel in a valid recording shares one sample count, so whichever
    non-empty one is checked first gives the same answer; the generality exists for the
    day flow itself can be absent (M-21's EMG-only segmentation, which needs a raw time
    axis before any flow-based trim/segment step runs), not because it changes anything
    for today's E2 scope: flow is always present there, so it is always the first
    non-empty argument, and this always resolves to the same answer as before."""
    for arr in arrays:
        n = len(arr)
        if n > 0:
            return n
    return 0


def _load(path, s):
    """Load a file, then (only when a pre-analysis resample is active) resample every
    channel from the file's true rate to the analysis rate. ``load`` integrates volume
    from flow using ``samplingfrequency`` (loaders.py), so the load itself must run at
    the true file rate; the value is temporarily restored for the load call, then left
    at the analysis rate for all downstream compute."""
    fs_out = int(s.input.format.samplingfrequency)
    fs_in = int(getattr(s.input.format, "samplingfrequency_in", fs_out))
    if fs_in == fs_out:
        return load(path, s)
    saved = s.input.format.samplingfrequency
    s.input.format.samplingfrequency = fs_in
    try:
        data = load(path, s)
    finally:
        s.input.format.samplingfrequency = saved
    return _resample_channels(data, fs_in, fs_out)


def _coupled_stft(cfg, fs_in, fs_out):
    """Scale the noise STFT window to preserve its TIME extent at the new rate: n_fft
    means a different duration at a different sample rate, so a 128 ms window authored at
    2000 Hz would collapse to a handful of samples after downsampling. n_fft/win are kept
    powers of two; hop scales with them. Returns (n_fft, hop_length, win_length)."""
    ratio = fs_out / fs_in

    def _pow2(n):
        n = max(8, int(round(n)))
        return 1 << (n - 1).bit_length()

    n_fft = _pow2(cfg.n_fft * ratio)
    win_length = _pow2(cfg.win_length * ratio)
    hop_length = max(1, int(round(cfg.hop_length * ratio)))
    return n_fft, hop_length, win_length


def apply_resample_override(settings: Settings, s) -> "tuple | None":
    """Mutate the legacy namespace ``s`` in place so every downstream ``_load()``/
    :func:`segment_file` call resamples to ``processing.sampling.resample_to_frequency``
    when ``processing.sampling.resample`` is on -- a no-op (``s`` untouched) when
    resampling is off, or configured to the file's own native rate.

    ``run_batch`` and the ``respmech breaths`` CLI (``cli.__main__.cmd_breaths``) both
    call this, rather than each re-deriving the same two-line override: a caller that
    built its own ``s`` (via ``core._legacy_ns.to_legacy_ns``) but skipped this call
    would segment at the file's NATIVE rate while ``run_batch`` segments at the
    RESAMPLED one -- a genuinely different array (and, since
    ``processing.mechanics.breathseparationbuffer`` is a raw sample-count window, a
    different real time window for the same segmentation decision). Returns the
    ``(n_fft, hop_length, win_length)`` STFT override :func:`_build_noise_set` needs at
    the new rate (``None`` when this call was a no-op) -- callers that never build a
    noise profile (``cmd_breaths``) can simply discard it."""
    fs_in = int(s.input.format.samplingfrequency)
    if not settings.processing.sampling.resample:
        return None
    fs_out = int(settings.processing.sampling.resample_to_frequency)
    if fs_out <= 0 or fs_out == fs_in:
        return None
    s.input.format.samplingfrequency = fs_out          # analysis rate for all downstream
    s.input.format.samplingfrequency_in = fs_in         # true file rate, read by _load
    return _coupled_stft(settings.processing.emg.noise, fs_in, fs_out)


def _diag_wanted(settings) -> bool:
    """True when any diagnostic figure or the WAV export is enabled — i.e. when the
    per-file diagnostic signal arrays need to be retained on the FileResult."""
    dg = settings.output.diagnostics
    return bool(dg.save_pv_average or dg.save_pv_individual or dg.save_raw or dg.save_trimmed
                or dg.save_drift or getattr(dg, "save_emg", False)
                or settings.processing.emg.save_sound)


def _ecg_remove(s, emgcolumnsraw, cancel_check=None):
    """Run ECG removal on the (full, untrimmed) raw EMG. Returns (emgcols, diag) where
    diag reports R-peak count and peak-window RMS suppression on the detect channel,
    or None if ECG removal is off. Detection/template parameters come from the
    test-level settings and are applied identically to every file."""
    emgcols = np.array(emgcolumnsraw)
    if not s.processing.emg.remove_ecg:
        return emgcols, None
    detect = s.processing.emg.column_detect
    raw_detect = np.array(emgcols[:, detect], dtype=float)
    emgcols_ecg, _ecgw, peaks_s = emglib.remove_ecg(
        emgcols, emgcols[:, detect],
        samplingfrequency=s.input.format.samplingfrequency,
        ecgminheight=s.processing.emg.minheight,
        ecgmindistance=s.processing.emg.mindistance,
        ecgminwidth=s.processing.emg.minwidth,
        windowsize=s.processing.emg.windowsize,
        cancel_check=cancel_check)
    emgcols = np.array(emgcols_ecg)
    fs = s.input.format.samplingfrequency
    peaks_samp = (np.asarray(peaks_s) * fs).astype(int)
    before = emglib.peak_window_rms(raw_detect, peaks_samp, fs)
    after = emglib.peak_window_rms(np.array(emgcols[:, detect], dtype=float), peaks_samp, fs)
    supp = (1.0 - after / before) if (before and before == before) else float("nan")
    diag = {"n_peaks": int(len(peaks_samp)), "detect_channel": int(detect),
            "peak_rms_before": before, "peak_rms_after": after, "suppression": supp,
            "peaks_s": np.asarray(peaks_s, dtype=float)}   # R-peak times (untrimmed), for the EMG figure
    return emgcols, diag


# --- Per-run load + ECG-removal cache (Wave 2.4) -----------------------------------
#
# When shared-profile noise reduction is on, the noise-building phase loads and ECG-removes
# the reference file and the first auto_prop files, and then the main loop loads and
# ECG-removes them AGAIN — the reference file is processed up to three times (reference clip,
# auto_prop, main loop). ``_load_and_ecg`` memoises ``(_load, _ecg_remove)`` by absolute path
# for one ``run_batch``, so each file is loaded and ECG-removed at most once.
#
# CACHE-AND-COPY, so this is provably identical to computing fresh every time: the cache
# holds a private pristine snapshot that is never handed out, and every consumer gets either
# its own fresh computation (miss) or a copy (hit). No aliasing, so no consumer can perturb
# another's arrays — exactly the guarantee the old code got from `_ecg_remove` returning a
# fresh `np.array` on every call. Both `_load` and `_ecg_remove` are pure functions of
# (path/array, settings), which are fixed for the run, so a cached result equals a fresh one.
_LOAD_CACHE_MAX = 8          # bound: a many-tiny-files run degrades to today's recompute, not OOM


def _copy_snapshot(snap):
    """Deep-ish copy of a ``(_load result, emgcols_ecg, ecg_diag)`` snapshot — every ndarray
    is copied so the returned snapshot shares no storage with the cached one."""
    load_result, emgcols_ecg, ecg_diag = snap
    lr = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in load_result)
    ecg = emgcols_ecg.copy() if isinstance(emgcols_ecg, np.ndarray) else emgcols_ecg
    if ecg_diag is None:
        diag = None
    else:
        diag = dict(ecg_diag)
        ps = diag.get("peaks_s")
        if isinstance(ps, np.ndarray):
            diag["peaks_s"] = ps.copy()
    return (lr, ecg, diag)


def _load_and_ecg(path, s, cache=None, cancel_check=None):
    """``(_load(path), *_ecg_remove(emg))`` for one file, memoised per run by abspath.

    Returns ``(load_result, emgcols_ecg, ecg_diag)``. On a hit, a fresh copy is returned and
    the cached snapshot is left pristine; on a miss the fresh computation is returned and a
    copy is stored (so the cache never shares storage with any caller)."""
    key = os.path.abspath(path)
    if cache is not None and key in cache:
        return _copy_snapshot(cache[key])
    load_result = _load(path, s)
    emg = load_result[6]
    emgcols_ecg, ecg_diag = _ecg_remove(s, emg, cancel_check=cancel_check)
    snap = (load_result, emgcols_ecg, ecg_diag)
    if cache is not None and len(cache) < _LOAD_CACHE_MAX:
        cache[key] = _copy_snapshot(snap)
    return snap


def _auto_detect_ecg_settings(settings, s, allfiles, progress=None, cache=None):
    """Auto-detect the ECG-removal detection/template parameters ONCE per test and apply
    them to every file, mirroring ``noise.auto_prop`` (auto-selected once, applied
    identically, never re-tuned per file). This is ``core.emg.suggest_ecg_settings`` — the
    same analysis the GUI's ECG tab "Auto-suggest" button runs on the previewed file — run
    here on ``ecg_reference_file`` (or the batch's first matched file when unset) so a
    settings.toml can drive it from the CLI without ever opening the GUI.

    Mutates ``s.processing.emg`` (the legacy namespace ``_ecg_remove`` reads) in place and
    returns the suggestion dict (plus ``reference_file``) for ``BatchResult.ecg_auto_report``.
    Primes ``cache`` with this file's load + ECG removal (Wave 2.4 style) so the main loop /
    noise-profile phase does not reload it."""
    emg_cfg = settings.processing.emg
    ref = emg_cfg.ecg_reference_file
    if ref:
        path = os.path.abspath(os.path.join(s.input.inputfolder, ref))
        if not os.path.isfile(path):
            raise ValueError(
                f"processing.emg.ecg_reference_file '{ref}' does not exist "
                f"(resolved to '{path}')")
    else:
        path = os.path.abspath(allfiles[0])
    _emit(progress, ProgressEvent(
        "stage", message=f"auto-detecting ECG settings from {os.path.basename(path)}"))
    try:
        load_result = _load(path, s)
        raw_emg = load_result[6]                              # keep native dtype for _ecg_remove
        fs = s.input.format.samplingfrequency
        sug = emglib.suggest_ecg_settings(np.asarray(raw_emg, dtype=float), fs)
    except Exception as e:
        # D18 point 4: this reads exactly ONE file, so there is no "skip and continue"
        # available the way the auto_prop collection loop below has. A file that EXISTS
        # (the isfile check above already handles a missing one) but cannot be read as
        # data must not abort the whole batch over the single reference file. Fall back
        # to the already-configured manual ECG settings -- left untouched below, since
        # this path returns before mutating them -- with a warning, so an unsupervised
        # run still produces output.
        warnings.warn(
            f"{os.path.basename(path)}: ecg_auto_detect could not read this reference "
            f"file ({e}) -- falling back to the configured manual ECG settings.")
        return None

    s.processing.emg.column_detect = int(sug["detect_channel"])
    s.processing.emg.minheight = float(sug["ecg_min_height"])
    s.processing.emg.mindistance = float(sug["ecg_min_distance_s"])
    s.processing.emg.minwidth = float(sug["ecg_min_width_s"])
    s.processing.emg.windowsize = float(sug["ecg_window_s"])

    if cache is not None and len(cache) < _LOAD_CACHE_MAX:
        # Prime with _ecg_remove(s, raw_emg) -- the SAME (native-dtype) array _load_and_ecg
        # would pass on a cache miss. Priming with the float-cast copy used for the
        # suggestion above would make this one file's cached ECG-removal result diverge
        # from every other file's (e.g. silently higher precision on integer-dtype raw EMG,
        # since core.emg.subtractecg mutates its input in place), breaking the "applied
        # identically to every file" guarantee for the reference file specifically.
        emgcols_ecg, ecg_diag = _ecg_remove(s, raw_emg)
        cache[path] = _copy_snapshot((load_result, emgcols_ecg, ecg_diag))

    report = dict(sug)
    report["reference_file"] = os.path.basename(path)
    return report


def _process_emg(s, emgcolumnsraw, startix, endix, noise_set=None, ecg_precomputed=None):
    """ECG removal -> trim -> shared-profile noise reduction (applied identically to
    every file when ``noise_set`` is provided). Returns (emgcols, ecg_diag, stages) where
    ``stages`` holds the trimmed EMG at each conditioning step (raw / ECG-removed /
    noise-reduced) for the diagnostic figures — ``None`` for a step that did not run.
    The returned ``emgcols`` is byte-identical to the previous behaviour.

    ``ecg_precomputed`` is an optional ``(emgcols_ecg, ecg_diag)`` from the per-run cache
    (Wave 2.4); when given, the ECG-removal step is reused rather than recomputed. It is the
    output of ``_ecg_remove(s, emgcolumnsraw)`` on the same file, so the result is unchanged."""
    raw_trim = np.asarray(emgcolumnsraw, dtype=float)[startix:endix]
    if ecg_precomputed is not None:
        emgcols_ecg, ecg_diag = ecg_precomputed
    else:
        emgcols_ecg, ecg_diag = _ecg_remove(s, emgcolumnsraw)
    ecg_trim = np.asarray(emgcols_ecg, dtype=float)[startix:endix]
    emgcols = ecg_trim
    if noise_set is not None:
        emgcols = noise_set.apply_columns(ecg_trim)
    stages = {"raw": raw_trim,
              "ecg_removed": ecg_trim if s.processing.emg.remove_ecg else None,
              "noise_reduced": emgcols if noise_set is not None else None}
    return emgcols, ecg_diag, stages


@dataclass
class Trimmed:
    """Everything ``segment_file`` computes for one file besides ``breaths`` itself:
    the trimmed/conditioned signals, the raw (untrimmed) arrays and the EMG-conditioning
    diagnostics. ``run_batch``'s main loop needs all of this afterwards (the ECG
    auto-detect quality check, cardiac-gated peak EMG, the diagnostic ``signals`` dict)
    without recomputing any of it."""
    timecol: object
    flow: object
    volume: object
    poes: object
    pgas: object
    pdi: object
    entropycolumns: object
    emgcolumns: object
    startix: int
    endix: int
    vol_uncorrected: object
    zerovol: object
    driftvol: object
    raw_timecol: object
    raw_flow: object
    raw_volume: object
    raw_poes: object
    raw_pgas: object
    raw_pdi: object
    raw_emgcolumns: object
    ecg_diag: object = None
    emg_stages: object = None
    # Soft per-file notices from applying processing.segmentation.overrides
    # (an out-of-range/colliding cut, a join with no nearby automatic boundary) --
    # always [] when the file has no override entry, or an entry with both cut_s
    # and join_s empty (segment_file never even calls compute.apply_segmentation_
    # overrides in that case -- see that function's own docstring on why this is
    # what makes the empty-overrides byte-identity guarantee hold by construction).
    segmentation_notices: list = field(default_factory=list)


def segment_file(settings: Settings, s, path, *, cache=None, cancel_check=None,
                  filename=None, noise_set=None, ecg_precomputed=None, progress=None):
    """Load, trim, EMG-condition, volume-correct and segment ONE file into breaths.

    This is ``run_batch``'s per-file trim/zero/drift/trend/segment sequence, extracted
    verbatim (behaviour-neutral: every golden scenario is byte-identical before and
    after) so a future reference/manoeuvre pre-pass and a ``respmech breaths`` CLI can
    segment a file through exactly the code the main loop uses, instead of
    re-implementing trim/zero/drift/trend/segment themselves.

    ``cache`` is the same per-run load+ECG-removal cache ``run_batch`` already threads
    through (Wave 2.4, see ``_load_and_ecg`` above): a hit is popped (consumed once, same
    as the main loop did before this extraction), a miss loads fresh. ``noise_set`` and
    ``ecg_precomputed`` are the batch-level shared EMG-conditioning inputs; both default
    to ``None`` (no shared noise reduction, no cached ECG removal) for a caller outside
    the main batch loop, such as a reference-only file that never goes through the noise
    profile. ``progress``/``cancel_check`` behave exactly as they do in ``run_batch``.

    Returns ``(breaths, trimmed)``. The caller still owns everything this function does
    NOT do: the ECG auto-detect quality-check warning (it needs ``BatchResult.
    ecg_auto_report``, batch-level state this function has no reason to know about) and
    everything after segmentation (``check_breaths``, boundary notices, mechanics)."""
    if filename is None:
        filename = os.path.basename(path)
    key = os.path.abspath(path)
    _cached = cache.pop(key, None) if cache is not None else None
    if _cached is not None:
        (flowraw, volumeraw, poesraw, pgasraw, pdiraw, entropycolumnsraw,
         emgcolumnsraw), _ecg_full, _ecg_diag_full = _cached
        if ecg_precomputed is None:
            ecg_precomputed = (_ecg_full, _ecg_diag_full)
    else:
        flowraw, volumeraw, poesraw, pgasraw, pdiraw, entropycolumnsraw, emgcolumnsraw = _load(path, s)
    n_raw = _first_present_length(flowraw, volumeraw, poesraw, pgasraw, pdiraw, emgcolumnsraw)
    timecolraw = np.arange(0, n_raw, dtype=int) / s.input.format.samplingfrequency

    # EMG-only: there is no flow channel to find zero-crossing breath boundaries
    # on, so there is nothing to `trim()` to -- the whole raw recording IS the analysis
    # window, and there is no volume to zero/drift-correct/trend-correct either (an
    # EMG-only signal set never has one). `separateintobreaths`'s own whole_file/
    # separators methods do the actual splitting below, exactly like the flow/volume
    # methods do on the trimmed window on the other branch.
    emg_only = getattr(s, "capabilities", Capabilities.FULL).mode == "emg_only"
    if emg_only:
        timecol, flow, volume = timecolraw, flowraw, volumeraw
        poes, pgas, pdi = poesraw, pgasraw, pdiraw
        startix, endix = 0, n_raw
        entropycolumns = entropycolumnsraw
    else:
        _emit(progress, ProgressEvent("stage", file=filename, message="trimming"))
        (timecol, flow, volume, poes, pgas, pdi, _emgtrim, startix, endix) = compute.trim(
            timecolraw, flowraw, volumeraw, poesraw, pgasraw, pdiraw,
            np.array(emgcolumnsraw) if len(emgcolumnsraw) else np.array([]), s)

        # Bug #2 fix: trim entropy columns with the same window as everything else.
        entropycolumns = entropycolumnsraw[startix:endix] if len(entropycolumnsraw) else entropycolumnsraw

    emgcolumns = []
    ecg_diag = None
    emg_stages = None
    if len(emgcolumnsraw) > 0:
        _emit(progress, ProgressEvent("stage", file=filename, message="processing EMG"))
        emgcolumns, ecg_diag, emg_stages = _process_emg(
            s, emgcolumnsraw, startix, endix, noise_set, ecg_precomputed=ecg_precomputed)

    if emg_only:
        vol_uncorrected = zerovol = driftvol = volume
    else:
        _emit(progress, ProgressEvent("stage", file=filename, message="volume correction"))
        vol_uncorrected = volume                                  # trimmed, pre-zero
        zerovol = compute.zero(volume)
        driftvol = compute.correctdrift(zerovol, s) if s.processing.mechanics.correctvolumedrift else zerovol
        volume = compute.correcttrend(driftvol, s) if s.processing.mechanics.correctvolumetrend else driftvol

    _emit(progress, ProgressEvent("stage", file=filename, message="segmenting breaths"))
    breaths = compute.separateintobreaths(
        s.processing.mechanics.separateby, filename, timecol, flow, volume,
        poes, pgas, pdi, entropycolumns, emgcolumns, s)

    # Repair the automatic segmentation from processing.segmentation.overrides,
    # if this file has an entry AND actually names a cut/join (an entry with both
    # lists empty -- a no-op the UI could still leave behind -- is treated exactly
    # like no entry at all). Restricted to a genuinely flow-/volume-bearing set:
    # whole_file/separators already have their own manual-boundary mechanism
    # (SeparatorEntry), and Settings.validate() never lets the two
    # coexist for the same analysis. Guarding the CALL itself (rather than relying
    # on apply_segmentation_overrides to be a no-op on empty input) is what makes
    # "empty overrides is byte-identical" hold by construction, not by happenstance.
    segmentation_notices: list = []
    if not emg_only and s.processing.mechanics.separateby in ("flow", "volume"):
        override = next(
            (e for e in settings.processing.segmentation.overrides if e.file == filename), None)
        if override is not None and (override.cut_s or override.join_s):
            _emit(progress, ProgressEvent(
                "stage", file=filename, message="applying segmentation overrides"))
            breaths, _override_notices = compute.apply_segmentation_overrides(
                filename, breaths, override.cut_s, override.join_s,
                timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns,
                s.input.format.samplingfrequency, s.processing.mechanics.breathseparationbuffer,
                ignored_breaths=compute.ignorebreaths(filename, s),
                kinds=compute.breathkinds(filename, s))
            segmentation_notices = [notice.message for notice in _override_notices]

    trimmed = Trimmed(
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=entropycolumns, emgcolumns=emgcolumns,
        startix=startix, endix=endix,
        vol_uncorrected=vol_uncorrected, zerovol=zerovol, driftvol=driftvol,
        raw_timecol=timecolraw, raw_flow=flowraw, raw_volume=volumeraw,
        raw_poes=poesraw, raw_pgas=pgasraw, raw_pdi=pdiraw,
        raw_emgcolumns=emgcolumnsraw,
        ecg_diag=ecg_diag, emg_stages=emg_stages,
        segmentation_notices=segmentation_notices,
    )
    return breaths, trimmed


def _emg_segmented(settings, s, path, cache=None, cancel_check=None, *, exclude_typed_from_expiration=False):
    """Segment a file through ``segment_file`` (M-23) and return (emg, insp_mask,
    exp_mask) over that segmentation's own EMG -- ECG-removed but never noise-reduced
    (``segment_file``'s ``noise_set=None`` default), exactly as before. Used to build
    the noise reference (expiration) and to gather active/quiet EMG for prop selection.

    Before M-23 this hardcoded flow-method segmentation and never trend-corrected the
    volume (``separateintobreaths('flow', ...)``, ``compute.zero(vT)`` with no
    ``correcttrend``, regardless of ``processing.mechanics.separateby``/
    ``processing.volume.correct_trend``), so a volume-segmented or trend-corrected
    analysis built its noise masks from breaths it never actually used for the
    analysis itself. Routing through ``segment_file`` (M-06) makes the
    mask follow the SAME configured method/trend every other consumer of that
    trim/zero/drift/trend/segment sequence already uses -- a deliberate, documented
    numerical change for exactly those settings combinations (this ticket); every
    existing (flow-method, no-trend) scenario is unaffected, since ``segment_file``
    then reduces to byte-identical behaviour.

    ``cache`` (Wave 2.4): ``segment_file``'s own cache lookup is CONSUMING
    (``cache.pop``), one-shot by design for the main loop's single pass over each file
    -- but this function must not drain the SHARED cache, or the main loop's own later
    ``segment_file`` call for the same file (the reference file always; every batch
    file too, under ``auto_prop``) would miss and reload/re-remove-ECG from scratch,
    breaking the very "each file loaded/ECG-removed at most once per run" invariant
    Wave 2.4 exists for (``test_load_cache.py`` pins it). So: read (or, on a miss,
    compute and prime) the snapshot from the REAL shared ``cache`` via the same
    non-consuming ``_load_and_ecg`` the old code called directly, then hand
    ``segment_file`` a throwaway one-entry cache of its own to pop from -- it gets its
    guaranteed hit, the real ``cache`` is never touched by the pop, and the main loop
    finds the entry still there later.

    ``exclude_typed_from_expiration`` (M-22, decision 11): when true, a breath typed
    via ``processing.breath_types`` (any kind -- an IC/FVC/sniff/max_insp/other
    manoeuvre, never tidal breathing) contributes NOTHING to the returned expiration
    mask, because a manoeuvre's expiration is not diaphragm-quiet the way an ordinary
    tidal breath's is. A no-op when the file carries no typed breaths at all (the
    overwhelming common case, and every existing golden/synthetic scenario), so the
    default ``False`` is only a documentation nicety, not a behavioural difference --
    ``_reference_noise_clip``'s own expiration branch is the one caller that needs
    ``True``: it is building the reference the noise profile is trusted for, not
    merely sampling activity across the batch (``_build_noise_set``'s ``auto_prop``
    gather, the OTHER caller, intentionally keeps the pre-existing, unfiltered
    behaviour -- see that function's own docstring)."""
    snap = _load_and_ecg(path, s, cache=cache, cancel_check=cancel_check)
    seg_cache = {os.path.abspath(path): snap}
    breaths, trimmed = segment_file(settings, s, path, cache=seg_cache, cancel_check=cancel_check)
    emg_full = trimmed.emgcolumns
    filename = os.path.basename(path)
    kinds = compute.breathkinds(filename, s) if exclude_typed_from_expiration else {}
    ins = np.zeros(len(trimmed.flow), bool); ex = np.zeros(len(trimmed.flow), bool); p = 0
    for breathno, b in breaths.items():
        ni = len(b["inspiration"]["time"]); ne = len(b["expiration"]["time"])
        ins[p:p + ni] = True
        if kinds.get(breathno) is None:
            ex[p + ni:p + ni + ne] = True
        p += ni + ne
    n = min(len(emg_full), len(ins))
    return emg_full[:n], ins[:n], ex[:n]


def _rest_segments_clip(settings, s, path, *, cache=None, cancel_check=None):
    """The EMG-only counterpart of the 'expiration' branch below: concatenate the
    reference file's own segments typed ``'rest'`` (M-22). Reuses ``segment_file``
    (M-06/M-21) so the clip is built from the SAME whole_file/separators segmentation
    -- and the same kind lookup -- the analysis itself will use; boundaries never
    drift between what a preview shows and what the reference clip is actually built
    from. ``noise_set=None`` (segment_file's default): there is no profile to apply
    yet, this call exists to BUILD one, mirroring ``_emg_segmented``'s own
    ECG-removed-but-not-noise-reduced clip for the flow-bearing branch."""
    breaths, _trimmed = segment_file(settings, s, path, cache=cache, cancel_check=cancel_check)
    parts = [np.asarray(b["emgcols"]) for b in breaths.values() if b.get("kind") == "rest"]
    if not parts:
        # Settings.validate() already rejects an 'unresolved' mode before a batch starts,
        # but resolve_noise_reference_mode only checks the SETTINGS shape (does a
        # BreathTypeEntry name this file with kind 'rest'?) -- it cannot know the file
        # actually still has that many segments once loaded (a shorter re-recording, an
        # edited separators list). Named the same way every other "could not build the
        # reference" failure in this function is, one line down.
        raise ValueError(f"no 'rest'-typed segment found in {os.path.basename(path)}")
    return np.concatenate(parts, axis=0)


def _reference_noise_clip(settings, s, cache=None, cancel_check=None):
    """Build the EMG-free noise reference clip (multichannel) once per test.

    Unlike ``_build_noise_set``'s ``auto_prop`` collection loop, a file that fails here
    is never merely skipped: it IS the shared noise reference, so if it cannot be read
    the profile cannot be built at all (D18 point 3). The failure is re-raised naming
    the reference file in plain English, since the underlying loader error may not
    mention it at all (e.g. a bare ``EmptyDataError`` for a 0-byte file) -- but the
    ORIGINAL exception type is preserved where possible (falling back to ``ValueError``
    only when the type itself refuses a single-message constructor): callers such as
    ``ui/workers.py``'s ``stage_noise_fidelity`` and ``ui/screens/run_screen.py``'s
    fix-hint lookup key their handling off ``DataValidationError``/``TrimError``/
    ``FileNotFoundError`` specifically, and a bare ``ValueError`` here would silently
    fall through both (self-review finding, 10-08-2026).

    M-22: which of the four buildable sources (``resolve_noise_reference_mode``'s
    ``'expiration'``/``'intervals'``/``'rest_segments'`` -- ``'interburst'`` has no
    implementation yet, and ``'unresolved'`` never reaches here, both rejected by
    ``Settings.validate()`` first) is now decided by that ONE resolver, not by
    re-deriving the same predicate here. For a flow-bearing analysis this reproduces
    the exact rule this codebase has always used (``use_expiration or not
    reference_intervals``) byte-for-byte -- see the resolver's own docstring."""
    ns_cfg = settings.processing.emg.noise
    ref = ns_cfg.reference_file
    if not ref:
        raise ValueError("processing.emg.noise.reference_file is required when noise reduction is enabled")
    path = os.path.join(s.input.inputfolder, ref)
    fs = s.input.format.samplingfrequency
    mode = resolve_noise_reference_mode(settings)
    try:
        if mode == "expiration":
            emg_full, ins, ex = _emg_segmented(settings, s, path, cache=cache, cancel_check=cancel_check,
                                               exclude_typed_from_expiration=True)
            clip = emg_full[ex]   # diaphragm-quiet expiration of the rest reference
        elif mode == "intervals":
            _load_result, emg_ecg, _diag = _load_and_ecg(path, s, cache=cache, cancel_check=cancel_check)
            parts = [emg_ecg[int(t0 * fs):int(t1 * fs)] for t0, t1 in ns_cfg.reference_intervals]
            clip = np.concatenate(parts, axis=0)
        elif mode == "rest_segments":
            clip = _rest_segments_clip(settings, s, path, cache=cache, cancel_check=cancel_check)
        else:
            # 'interburst' and 'unresolved' are both rejected by Settings.validate()
            # while noise reduction is enabled -- reachable here only via a settings
            # object that skipped validate() (a hand-built test double, a future caller
            # that forgot to validate first). Fail loudly rather than guess.
            raise ValueError(
                f"processing.emg.noise.reference_mode={mode!r} cannot be built into a "
                "reference clip yet")
    except Exception as e:
        msg = f"Could not read the noise reference file '{ref}': {e}"
        try:
            wrapped = type(e)(msg)
        except TypeError:
            wrapped = ValueError(msg)
        raise wrapped from e
    return clip


def _build_noise_set(settings, s, files, progress=None, clip=None, cancel_check=None, stft=None,
                     cache=None):
    from respmech.core import noise as noiselib
    cfg = settings.processing.emg.noise
    # ``stft`` (n_fft, hop, win) overrides the configured window when a pre-analysis
    # resample is active, so the spectral gate keeps a fixed TIME window at the new rate.
    n_fft, hop_length, win_length = stft if stft is not None else (cfg.n_fft, cfg.hop_length, cfg.win_length)
    # ``clip`` lets the PREVIEW pass a shared/cached reference clip; batch/CLI pass None
    # (default) and build it here exactly as before — byte-identical for the golden path.
    if clip is None:
        clip = _reference_noise_clip(settings, s, cache=cache, cancel_check=cancel_check)
    profiles = noiselib.build_profiles(
        clip, s.input.format.samplingfrequency, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_std_thresh=cfg.n_std_thresh,
        n_grad_freq=cfg.n_grad_freq, n_grad_time=cfg.n_grad_time, cancel_check=cancel_check)

    if cfg.auto_prop:
        # Gather active/quiet EMG across the test (capped) to choose prop ONCE. Built on
        # inspiration/expiration PHASES (_emg_segmented, now via segment_file -- follows
        # the configured separateby/trend since M-23, no longer hardcoded to flow) -- an
        # EMG-only signal set has no phases to gather by, and Settings.validate() (M-22)
        # already refuses auto_prop=True for one before a batch can reach here, so this
        # loop never needs to handle that case; it stays exactly as it was pre-M-22.
        act, qui, cap, unreadable, last_exc = [], [], 40000, [], None
        for fi in files:
            try:
                emg_full, ins, ex = _emg_segmented(settings, s, os.path.abspath(fi), cache=cache,
                                                   cancel_check=cancel_check)
            except Exception as e:
                # D18 point 2: a file that cannot be read cannot contribute to a profile
                # it cannot be read for either. Skip it here so it fails normally, as an
                # ordinary per-file error, in run_batch's own main loop below (which has
                # its own guard) -- instead of aborting the whole batch before a single
                # file has actually been processed. This also makes the outcome
                # independent of where the bad file sorts: the old code could reach the
                # 40000-sample cap (and so never reach a bad file sorted late) or hit it
                # first (and abort), making a batch's success depend on file order.
                unreadable.append(os.path.basename(fi))
                last_exc = e
                continue
            act.append(emg_full[ins]); qui.append(emg_full[ex])
            if sum(len(a) for a in act) >= cap:
                break
        if not act:
            # Self-review finding (10-08-2026): if EVERY file failed above, falling
            # through would hand an empty list to np.concatenate below, which raises a
            # bare "need at least one array to concatenate" -- naming no file and no
            # cause, i.e. strictly worse than the abort this ticket exists to fix. Name
            # what actually happened instead, and -- second self-review finding, found by
            # a regression this surfaced in ui/workers.py's stage_noise_fidelity (which
            # has no per-file loop of its own to fall back into; it calls this function
            # directly and depends on catching a TrimError/DataValidationError BY TYPE
            # to build its own friendly message) -- preserve `last_exc`'s TYPE the same
            # way ``_reference_noise_clip`` does, rather than always raising a bare
            # ValueError that such a type-keyed caller cannot recognise.
            msg = ("Could not auto-select the noise reduction strength (auto_prop): none of "
                  f"the {len(files)} batch file{'s' if len(files) != 1 else ''} could be "
                  f"read ({', '.join(unreadable)}): {last_exc}")
            try:
                wrapped = type(last_exc)(msg)
            except TypeError:
                wrapped = ValueError(msg)
            raise wrapped from last_exc
        if unreadable:
            # Self-review finding (10-08-2026): the skip above was otherwise completely
            # silent -- no warning, no progress event -- so a batch could complete having
            # quietly chosen its shared noise-reduction strength from fewer files than
            # the user thinks it did.
            warnings.warn(
                f"auto_prop: {len(unreadable)} of {len(files)} file"
                f"{'s' if len(unreadable) != 1 else ''} could not be read and were "
                f"excluded from noise-reduction-strength selection: "
                f"{', '.join(unreadable)}")
        active = np.concatenate(act, axis=0)[:cap]
        quiet = np.concatenate(qui, axis=0)[:cap]
        prop, report = noiselib.select_prop_decrease(
            profiles, active, quiet, s.input.format.samplingfrequency,
            target=cfg.fidelity_target, cancel_check=cancel_check)
    else:
        prop, report = cfg.prop_decrease, {"prop_decrease": cfg.prop_decrease, "auto": False}
    _emit(progress, ProgressEvent("stage", message=f"noise profile built (prop_decrease={prop})"))
    return noiselib.NoiseProfileSet(profiles, prop), report


def match_input_files(folder: str, pattern: str) -> list:
    """Files in ``folder`` matching ``pattern``, sorted. Case-INSENSITIVE and safe
    against glob metacharacters in the folder name — unlike ``glob.glob``, which is
    case-sensitive on macOS but not Windows (so a mixed-case batch would process
    different files, and the shared EMG noise profile would diverge numerically between
    the two platforms), and which treats ``[`` / ``*`` in a folder name like
    ``Study [2024]`` as a pattern that matches nothing. ``os.listdir`` sidesteps the
    folder-name problem; ``fnmatchcase`` on lowered names is deterministic on both OSes.
    Leading-dot (Unix-hidden) files are excluded unless the pattern is explicitly
    dot-leading, mirroring ``glob``. A pattern may carry a subdirectory (``raw/*.csv``,
    or ``raw\\*.csv`` authored on Windows) — as ``glob.glob(join(folder, pattern))`` did —
    in which case the directory part is resolved against ``folder`` and only the filename
    part is matched, so a hand-edited/migrated config with a path-bearing mask keeps
    working instead of silently matching nothing."""
    pattern = pattern or "*.*"
    norm = pattern.replace("\\", "/")
    if "/" in norm:                            # split a path-bearing pattern into subdir + filemask
        subdir, pattern = norm.rsplit("/", 1)
        if subdir:
            folder = os.path.join(folder, *subdir.split("/"))
        pattern = pattern or "*.*"
    if not os.path.isdir(folder):
        return []
    pat = pattern.lower()
    hidden_ok = pat.startswith(".")
    out = []
    for name in os.listdir(folder):
        if not hidden_ok and name.startswith("."):
            continue
        full = os.path.join(folder, name)
        if fnmatch.fnmatchcase(name.lower(), pat) and os.path.isfile(full):
            out.append(full)
    return sorted(out)


def _load_external_references(settings: Settings, s, files: list, *, cache: dict,
                               cancel_check: Optional[Callable[[], bool]] = None,
                               progress: Optional[ProgressCallback] = None):
    """The batch's forepass (runs BEFORE the main per-file loop): load, segment and
    extract manoeuvre values from every reference-source file
    (``core.analysis.references.external_reference_sources``) that is named by
    ``processing.references``/``reference_defaults`` but is NOT already part of
    ``files`` -- the batch's own list of files the main loop is about to process.

    Reuses exactly the same primitives the main loop's own manoeuvre extraction does
    (``segment_file``, ``manoeuvres.extract``, ``manoeuvres.apply_repeatability``) so a
    source loaded here and one loaded as an ordinary in-batch reference-only file
    (M-30) produce byte-identical results for the same breath. A source is dropped
    from consideration ENTIRELY if it cannot be loaded/segmented at all (its whole-file
    failure is recorded, never raised -- the batch keeps going); one breath's own
    extraction failing does not drop the rest of that same source's typed breaths.

    ``cancel_check``, checked before each source: stops loading further sources
    (``break``, keeping whatever was already loaded) rather than raising or returning
    early itself -- the caller checks again right after this returns and does the
    actual early return, exactly like it already does between two ordinary files in
    the main loop, so a cancel flagged mid-forepass has the same visible effect as one
    flagged mid-batch instead of being silently swallowed.

    Returns ``(references, errors)`` -- see ``BatchResult.references``/
    ``reference_errors`` for the exact shapes. Never raises."""
    sources = referenceslib.external_reference_sources(settings, files)
    references: dict = {}
    errors: dict = {}
    for src in sources:
        if cancel_check is not None and cancel_check():
            break
        path = os.path.join(s.input.inputfolder, src)
        _emit(progress, ProgressEvent(
            "stage", file=src, message="loading external reference source"))
        try:
            breaths, _trimmed = segment_file(
                settings, s, path, cache=cache, cancel_check=cancel_check, filename=src)
        except Exception as e:
            errors[(src, None)] = f"{type(e).__name__}: {e}"
            continue
        tidal_breaths = [b for b in breaths.values() if not b["ignored"]]
        rows: dict = {}
        for breathno, breath in breaths.items():
            kind = breath.get("kind")
            if not kind or kind == "rest":
                continue
            try:
                rows[breathno] = manoeuvreslib.extract(
                    breath, kind, tidal_breaths, s.capabilities, s)
                # M-42: fvc_metrics is deliberately NOT part of manoeuvres.extract
                # itself (that module's own docstring keeps it out of scope) --
                # merged in here, and identically in the main loop below, so an
                # external reference source's own FVC row matches an in-batch one.
                mfvllib.apply_to_row(rows[breathno], breath, tidal_breaths,
                                     float(s.input.format.samplingfrequency))
            except Exception as e:
                errors[(src, breathno)] = f"{type(e).__name__}: {e}"
        if rows:
            manoeuvreslib.apply_repeatability(rows, s.processing.lung_volume.ic)
            mfvllib.apply_tlc_consistency(rows, s.processing.lung_volume.ic)
        references[src] = rows
    return references, errors


def is_subset_run(settings: Settings, only_files: Optional[list]) -> bool:
    """True when ``only_files`` restricts a run to fewer than everything ``settings.input``
    would otherwise match — never true for ``None``, and never true when ``only_files``
    happens to name every matching file (a "subset" that is not actually one).

    This is the single yes/no test the write layer (``core.io.writers.write_batch``) and the
    GUI (``ui.workers.BatchWorker``, ``ui.screens.run_screen.RunScreen``) all use to decide
    whether cohort-level outputs — the Average/Cohort-summary workbooks and the cohort Campbell
    figure — may be written at all, since those are built across the WHOLE study and a partial
    run silently rebuilding them from a handful of files was the data-loss bug this exists to
    prevent (ticket A05). Matched against the exact same file list ``run_batch`` itself uses
    (``match_input_files`` on ``settings.input.folder``/``settings.input.files``), so this
    agrees with what a run actually processed rather than a UI-level approximation.

    Compares the INTERSECTION of ``only_files`` with what currently matches, not
    ``only_files`` directly: ``run_batch`` itself only ever processes
    ``{f for f in allfiles if basename(f) in only_files}`` (see below), so a stale name in
    ``only_files`` that no longer matches anything (e.g. a file deleted from the input
    folder between a run and a later "Re-run failed") must not, by itself, make an
    otherwise-complete run look like a subset — every file that would actually be
    processed is still all of them."""
    if not only_files:
        return False
    allfiles = match_input_files(settings.input.folder, settings.input.files)
    allnames = {os.path.basename(f) for f in allfiles}
    given = {os.path.basename(f) for f in only_files}
    return bool(given) and (given & allnames) != allnames


def run_batch(settings: Settings, progress: Optional[ProgressCallback] = None,
              cancel_check: Optional[Callable[[], bool]] = None,
              only_files: Optional[list] = None) -> BatchResult:
    """Process the batch. ``progress`` receives :class:`ProgressEvent`s. If
    ``cancel_check`` returns True (checked before each file), processing stops
    cooperatively. ``only_files`` (basenames) restricts the run — used for GUI test
    runs on a single file."""
    settings.validate()
    s = to_legacy_ns(settings)

    # Pre-analysis resample: fix the analysis rate up front so the shared noise profile
    # AND every per-file load/compute use the same (possibly resampled) rate. Default OFF
    # (fs_out == fs_in) ⇒ _load is a pass-through and the run is byte-identical.
    stft_override = apply_resample_override(settings, s)

    allfiles = match_input_files(s.input.inputfolder, s.input.files)   # the full test (defines the shared profile)
    files = [f for f in allfiles if only_files is None or os.path.basename(f) in set(only_files)]
    if len(files) == 0:
        raise FileNotFoundError(
            f"No input files found for '{s.input.files}' in '{s.input.inputfolder}'")

    result = BatchResult()
    average_rows = []

    # Per-run load + ECG-removal cache (Wave 2.4). Only useful when the noise-building phase
    # or ECG auto-detect runs (they are the sole sources of duplicate load/ECG work); the main
    # loop drains it by popping each file it reaches, so it never holds more than was primed.
    load_cache: dict = {}

    # Auto-detect the ECG-removal settings ONCE per test (opt-in; off -> unchanged, golden-safe),
    # BEFORE the shared noise profile is built below (its EMG segmentation itself ECG-removes
    # every file) and before the main loop, so every consumer sees the same, final settings.
    if settings.processing.emg.ecg_auto_detect:
        if not settings.processing.emg.remove_ecg:
            raise ValueError(
                "processing.emg.ecg_auto_detect requires processing.emg.remove_ecg to be enabled")
        if len(s.input.data.columns_emg) == 0:
            raise ValueError(
                "processing.emg.ecg_auto_detect requires input.channels.emg to be configured")
        result.ecg_auto_report = _auto_detect_ecg_settings(
            settings, s, allfiles, progress, cache=load_cache)

    # ONE shared noise profile + parameter set for the whole test, built once and
    # applied identically to every file (never re-tuned per file). Built from the
    # full test even when only_files restricts processing (e.g. a GUI test run), so a
    # single file is denoised exactly as it would be in the full batch.
    noise_set = None
    if settings.processing.emg.noise.enabled and len(s.input.data.columns_emg) > 0:
        _emit(progress, ProgressEvent("stage", message="building shared noise profile"))
        noise_set, result.noise_report = _build_noise_set(settings, s, allfiles, progress,
                                                          stft=stft_override, cache=load_cache)

    # Cross-file reference forepass: load every reference SOURCE file this run's own
    # `files` does not already contain (a dedicated IC/FVC recording, say, or a source
    # this particular (only_files-restricted) run excludes) and extract its typed
    # breaths' manoeuvre values, BEFORE the main loop -- so a subset run gives the same
    # reference rows a full run would (this ticket's own acceptance test). A source
    # already in `files` needs no separate load here: the main loop extracts its
    # manoeuvres itself, in `result.files[...].manoeuvres`, exactly like an in-batch
    # reference-only file (M-30) already does.
    result.references, result.reference_errors = _load_external_references(
        settings, s, files, cache=load_cache, cancel_check=cancel_check, progress=progress)
    # The forepass's own loop only BREAKS on a cancellation flagged mid-load (never
    # raises/returns itself -- see its own docstring); check again here, exactly like
    # the main loop's per-file check just below, so a cancel flagged during the
    # forepass has the SAME effect (an immediate, silent "cancelled" return) a cancel
    # flagged between two ordinary files already has, instead of a run that ignores it
    # and quietly falls through into the main loop (self-review finding).
    if cancel_check is not None and cancel_check():
        _emit(progress, ProgressEvent("finished", message="cancelled"))
        return result

    for fi in files:
        if cancel_check is not None and cancel_check():
            _emit(progress, ProgressEvent("finished", message="cancelled"))
            return result
        file = os.path.abspath(fi)
        filename = os.path.basename(file)
        _emit(progress, ProgressEvent("file_start", file=filename, message="loading"))
        file_notices: list[str] = []          # surfaced in FileResult.notices below
        try:
            # Trim/EMG-condition/volume-correct/segment this file: the cache pop that
            # used to happen right here (Wave 2.4) now happens inside segment_file itself, so
            # a future caller outside this loop (a reference pre-pass, the `respmech breaths`
            # CLI) gets exactly the same conditioning without re-implementing it.
            breaths, trimmed = segment_file(
                settings, s, file, cache=load_cache, cancel_check=cancel_check,
                filename=filename, noise_set=noise_set, progress=progress)
            timecol, flow, volume = trimmed.timecol, trimmed.flow, trimmed.volume
            poes, pgas, pdi = trimmed.poes, trimmed.pgas, trimmed.pdi
            entropycolumns, emgcolumns = trimmed.entropycolumns, trimmed.emgcolumns
            startix, endix = trimmed.startix, trimmed.endix
            vol_uncorrected, zerovol, driftvol = trimmed.vol_uncorrected, trimmed.zerovol, trimmed.driftvol
            timecolraw, flowraw, volumeraw = trimmed.raw_timecol, trimmed.raw_flow, trimmed.raw_volume
            poesraw, pgasraw, pdiraw = trimmed.raw_poes, trimmed.raw_pgas, trimmed.raw_pdi
            emgcolumnsraw = trimmed.raw_emgcolumns
            ecg_diag, emg_stages = trimmed.ecg_diag, trimmed.emg_stages
            for _msg in trimmed.segmentation_notices:
                warnings.warn(f"{filename}: {_msg}")
                file_notices.append(_msg)
                _emit(progress, ProgressEvent("warning", file=filename, message=f"{filename}: {_msg}"))

            if len(emgcolumnsraw) > 0:
                # ecg_auto_detect derives the shared detection parameters from ONE reference
                # file; a heart rate or R-amplitude that differs enough on THIS file can mean
                # those parameters miss real beats here even though they fit the reference
                # file fine. Warn per file (never fail the run) so an unsupervised batch
                # leaves a visible trail of files worth revisiting -- the same detection_quality
                # check robust_peak already uses, just evaluated unconditionally here.
                # ``result.ecg_auto_report is not None`` (not just the settings flag, D18
                # point 4): when the auto-detect reference file itself could not be read,
                # ``_auto_detect_ecg_settings`` falls back to the manual settings and
                # returns ``None`` -- the flag is still on, but there is no reference
                # file's name left to quote below, and the fallback already warned once.
                if (settings.processing.emg.ecg_auto_detect and ecg_diag is not None
                        and result.ecg_auto_report is not None):
                    _peaks = np.asarray(ecg_diag["peaks_s"], float)
                    if _peaks.size == 0:
                        _msg = (
                            f"ecg_auto_detect (reference "
                            f"{result.ecg_auto_report['reference_file']}) found no R-peaks on "
                            "this file -- ECG removal likely did nothing here.")
                        warnings.warn(f"{filename}: {_msg}")
                        file_notices.append(_msg)
                    else:
                        _dq = emglib.detection_quality(_peaks, s.processing.emg.mindistance)
                        if not _dq["ok"]:
                            _msg = (
                                f"ecg_auto_detect quality check failed "
                                f"({_dq['reason']}) -- the shared parameters (from "
                                f"{result.ecg_auto_report['reference_file']}) may not fit this "
                                "file; consider processing.emg.ecg_reference_file or manual "
                                "settings for it.")
                            warnings.warn(f"{filename}: {_msg}")
                            file_notices.append(_msg)

            # EMG-only: none of trim_boundary_notices (a flow-trim-truncation
            # check -- there was no trim), vefactor (60 / len(flow)/fs would ZeroDivisionError
            # on an empty flow array) or calculateaveragebreaths (indexes
            # breath["inspiration"]/["expiration"], which a phase-less segment does not
            # have) apply to a signal set with no flow channel at all.
            emg_only = getattr(s, "capabilities", Capabilities.FULL).mode == "emg_only"

            # M-30: a file where EVERY breath is typed (no tidal breathing at all) is not
            # an error -- it is a reference-only file (a dedicated IC/FVC recording, say).
            # Detected up front, before check_breaths would otherwise refuse it by name
            # (`check_breaths` cannot itself tell "nothing to analyse" apart from "this IS
            # the analysis, just not a tidal one" -- both look identical as "used == 0").
            # EMG-only is excluded here: on that signal set only 'rest'-typed segments are
            # ever ignored (`_merged_exclude_breaths`), so an EMG-only file with zero
            # non-ignored segments already means a genuinely empty file, not a
            # manoeuvre-only one -- check_breaths' existing error is still the right one.
            tidal_breaths = [b for b in breaths.values() if not b["ignored"]]
            has_typed = any(
                b.get("kind") and b["kind"] != "rest" for b in breaths.values())
            reference_only = (not emg_only) and (not tidal_breaths) and has_typed

            if not reference_only:
                # Stop here rather than compute mechanics for an empty set: everything
                # below is per-breath work, and the results layer can only report the
                # empty table as an opaque internal error.
                compute.check_breaths(breaths, filename, s)
            else:
                _msg = "no tidal breaths — reference manoeuvres only"
                file_notices.append(_msg)
                _emit(progress, ProgressEvent(
                    "stage", file=filename, message=f"{filename}: {_msg}"))

            if not emg_only and not reference_only:
                # K-035: the boundary breath trim KEEPS is never verified as complete — warn
                # when it is much shorter than this file's own typical breath, instead of
                # analysing it silently as whole. Live (ProgressEvent) as well as recorded
                # (file_notices -> run-report.txt), same as the other per-file quality
                # notices below. The live event's own message is what the CLI/Run log
                # actually print (unlike file_start/file_error, "warning" events have no
                # separate filename slot in either consumer) -- prefix it here, or the
                # printed line silently loses which file it is about.
                for _msg in compute.trim_boundary_notices(breaths, s):
                    warnings.warn(f"{filename}: {_msg}")
                    file_notices.append(_msg)
                    _emit(progress, ProgressEvent("warning", file=filename, message=f"{filename}: {_msg}"))

                vefactor = 60 / (len(flow) / s.input.format.samplingfrequency)
                bcnt = len(breaths)
                for bc in s.processing.mechanics.breathcounts:
                    if bc[0] == filename:
                        bcnt = bc[1]
                        break

                avgvolumein, avgvolumeex, avgpoesin, avgpoesex = compute.calculateaveragebreaths(breaths, s)

            # Opt-in cardiac-gated peak EMG needs the R-peaks the ECG-removal stage already
            # found, plus one per-file judgement of whether that peak set is complete enough
            # to gate on. An undetected beat is neither subtracted nor blanked, so gating a
            # file with missed beats reports a heartbeat with extra confidence — hence the
            # guard, evaluated once here rather than per breath.
            # M-30 self-review finding: skipped for a reference-only file too -- the ONLY
            # consumers of gate_peaks/gate_ok/gate_reason are the per-breath mechanics loop
            # (already skipped below, `if not reference_only:`) and compute_segment_emg,
            # neither of which a reference-only file ever reaches. Without this guard, a
            # reference-only file with poor R-peak detection would still get a spurious
            # "cardiac-gated peak EMG reported as NaN" notice for a computation that was
            # never attempted and has no corresponding column in any output of this file.
            gate_peaks, gate_ok, gate_reason = None, True, ""
            if not reference_only and s.processing.emg.robust_peak.enabled:
                gate_peaks = np.asarray(ecg_diag["peaks_s"], float) if ecg_diag else None
                if gate_peaks is None or gate_peaks.size == 0:
                    gate_ok = False
                    gate_reason = "no R-peaks available — is processing.emg.remove_ecg on?"
                else:
                    dq = emglib.detection_quality(
                        gate_peaks, s.processing.emg.mindistance,
                        long_rr_factor=s.processing.emg.robust_peak.long_rr_factor,
                        max_long_rr_frac=s.processing.emg.robust_peak.max_long_rr_frac,
                        hr_ceiling_margin=s.processing.emg.robust_peak.hr_ceiling_margin)
                    gate_ok, gate_reason = dq["ok"], dq["reason"]
                    if not gate_ok:
                        warnings.warn(f"{filename}: cardiac-gated peak EMG reported as NaN — "
                                      f"{gate_reason}")
                if not gate_ok:
                    # Recorded for BOTH refusal paths above (no R-peaks at all, and a
                    # detection_quality failure) -- the first path never warned at all before,
                    # so the "no R-peaks -- is remove_ecg on?" reason was invisible everywhere.
                    file_notices.append(f"cardiac-gated peak EMG reported as NaN — {gate_reason}")

            # M-30: a reference-only file has nothing for this loop to do (every breath
            # is typed, so `tidal_breaths` is empty and the loop's own
            # `if breath["ignored"]: continue` would skip every single one anyway) --
            # skipped explicitly rather than relying on that, since `vefactor`/
            # `avgvolumein` etc. are never computed for such a file either.
            if not reference_only:
                total = sum(1 for b in breaths.values() if not b["ignored"])
                done = 0
                # Opt-in PEEPi (core.analysis.pressure): reads the breath BEFORE each one
                # in the recording, ignored or not (its expiration tail holds the start of
                # the pre-flow deflection), so the neighbour is looked up over ALL breaths.
                peepi_on = (not emg_only and s.processing.pressure.peepi.enabled
                            and s.capabilities.flow and s.capabilities.poes)
                if (not emg_only and s.processing.pressure.peepi.enabled and not peepi_on):
                    file_notices.append(
                        "PEEPi analysis is enabled but this signal set lacks Flow or Poes -- skipped")
                _order = list(breaths)
                _prev_of = {k: (breaths[_order[i - 1]] if i else None) for i, k in enumerate(_order)}
                peepi_notes: list = []
                for breathno in breaths:
                    breath = breaths[breathno]
                    if breath["ignored"]:
                        continue
                    if emg_only:
                        # No inspiration/expiration split to compute mechanics from --
                        # compute_segment_emg (RMS/integral-EMG/gated-peak/entropy, whole
                        # segment only) is the entire per-segment computation, and the
                        # segment's own span replaces the flow-derived timing group.
                        compute.compute_segment_emg(breath, breath, s, cancel_check, gate_peaks,
                                                    gate_ok, gate_reason, phases=breath["has_phases"])
                        fs = s.input.format.samplingfrequency
                        time = np.atleast_1d(breath["time"])
                        seg_start_s = float(time[0]) if time.size else 0.0
                        # len(time)/fs (not time[-1] - time[0]) matches how every other
                        # duration in this codebase is derived (e.g. calculatemechanics's
                        # ti/te/ttot = len(...)/samplingfrequency) -- a sample COUNT, not
                        # the gap between the first and last sample's own timestamps.
                        seg_duration_s = time.size / fs
                        breath["mechanics"] = OrderedDict([
                            ("seg_start_s", seg_start_s),
                            ("seg_end_s", seg_start_s + seg_duration_s),
                            ("seg_duration_s", seg_duration_s),
                        ])
                    else:
                        compute.calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, s,
                                                   cancel_check=cancel_check, peaks_s=gate_peaks,
                                                   detection_ok=gate_ok, detection_reason=gate_reason)
                        if peepi_on:
                            try:
                                _note = pressurelib.attach(breath, _prev_of[breathno], bcnt, vefactor, s)
                            except Exception as e:
                                # isolated like mfvl.attach: the new columns stay NaN for this
                                # breath instead of one unexpected fault failing the file
                                breath.pop("pressure_ext", None)
                                breath.pop("peepi_added", None)
                                _note = f"PEEPi failed: {type(e).__name__}: {e}"
                            if _prev_of[breathno] is None:
                                # the first breath of a recording never has a predecessor: blank
                                # by design, so no notice (it would repeat on every file)
                                _note = None
                            if _note:
                                peepi_notes.append((breath["number"], _note))
                    done += 1
                    _emit(progress, ProgressEvent("breath", file=filename, breath=done, total_breaths=total))

            if not reference_only and peepi_notes:
                # One notice per file, not per breath: the first breath of every recording
                # has no predecessor, so a per-breath notice would repeat on every file.
                _detail = "; ".join(f"#{n}: {m}" for n, m in peepi_notes[:5])
                if len(peepi_notes) > 5:
                    _detail += f"; and {len(peepi_notes) - 5} more"
                file_notices.append(
                    f"PEEPi reported as NaN for {len(peepi_notes)} breath(s) -- {_detail}")

            # M-29/M-30: typed (non-`rest`) breaths never reach calculatemechanics above
            # (M-19 unions every typed kind into `excludebreaths` on a flow-bearing set,
            # so the loop's own `if breath["ignored"]: continue` skips them) -- extract
            # their manoeuvre values here instead, straight from the raw breath dict.
            # EMG-only breaths have no inspiration/expiration split (`has_phases=False`,
            # empty flow/volume/pressure arrays) for `extract` to read, so this is
            # flow-bearing only, same guard as the boundary-notice/vefactor block above.
            # Runs UNCHANGED for a reference-only file too: `extract()` (and the
            # `_low_effort`/`_ic_eelv_pre`/`_boundary` helpers it calls) is already a pure
            # function of one breath plus `tidal_breaths` and degrades gracefully when
            # that list is empty (falls back to the breath's own pre-inspiratory sample,
            # flags every manoeuvre BOUNDARY -- there is no tidal context to judge low
            # effort or eelv stability against, which is the honest answer, not a bug).
            fr_manoeuvres: dict = {}
            mfvl_fev1_source: str | None = None
            if not emg_only:
                for breathno in breaths:
                    breath = breaths[breathno]
                    kind = breath.get("kind")
                    if not kind or kind == "rest":
                        continue
                    try:
                        fr_manoeuvres[breathno] = manoeuvreslib.extract(
                            breath, kind, tidal_breaths, s.capabilities, s)
                        # M-42: fvc_metrics is deliberately NOT part of manoeuvres.
                        # extract itself (that module's own docstring keeps it out of
                        # scope) -- merged in here, identically to the external-
                        # reference forepass above, so an in-batch FVC row and one
                        # loaded as an external reference source agree.
                        mfvllib.apply_to_row(fr_manoeuvres[breathno], breath, tidal_breaths,
                                             float(s.input.format.samplingfrequency))
                    except Exception as e:
                        _msg = (f"breath #{breathno} ({kind}) manoeuvre extraction failed: "
                               f"{type(e).__name__}: {e}")
                        warnings.warn(f"{filename}: {_msg}")
                        file_notices.append(_msg)
                if fr_manoeuvres:
                    manoeuvreslib.apply_repeatability(fr_manoeuvres, s.processing.lung_volume.ic)
                    mfvllib.apply_tlc_consistency(fr_manoeuvres, s.processing.lung_volume.ic)
                    # M-42: stamps breath['mfvl_ext'] on every TIDAL breath of THIS
                    # file (same-file fvc/ic reference only -- see mfvl.attach's own
                    # docstring for why), BEFORE build_breath_table below joins it in
                    # exactly like breath['wob'] already is. Never for a reference-
                    # only file (M-30) or an EMG-only signal set (no tidal breaths
                    # with a real flow-volume trace to compare against an MEFV curve).
                    # Self-review finding: one breath's own manoeuvre extraction failing
                    # (the try/except a few lines above) leaves a half-built row in
                    # fr_manoeuvres that attach() was not written to tolerate -- caught
                    # here, per-file, the same isolation every other failure mode in
                    # this loop already gets, rather than letting it take the whole
                    # file down.
                    if not reference_only and tidal_breaths:
                        try:
                            mfvl_notice, mfvl_fev1_source = mfvllib.attach(
                                fr_manoeuvres=fr_manoeuvres, breaths=breaths,
                                tidal_breaths=tidal_breaths, filename=filename,
                                settings=settings, s=s)
                        except Exception as e:
                            _msg = f"MFVL placement failed: {type(e).__name__}: {e}"
                            warnings.warn(f"{filename}: {_msg}")
                            file_notices.append(_msg)
                        else:
                            if mfvl_notice:
                                file_notices.append(f"{filename}: {mfvl_notice}")
            manoeuvres_table = build_manoeuvre_table(fr_manoeuvres)

            # M-30: a reference-only file has no tidal breath table or average row to
            # build at all -- skip build_breath_table (which would otherwise raise
            # NoBreathsError, correctly, for a set with nothing tidal in it) rather than
            # feed it a set it will always refuse.
            if reference_only:
                breaths_table, average_row = None, None
            else:
                breaths_table, average_row = build_breath_table(filename, breaths, s)
                # M-42: fev1_source is a TEXT column (unlike everything else
                # mfvl.attach computed) -- self-review finding: joining a string
                # column in via breath['mfvl_ext'] (build_breath_table's own
                # mechanics.mean() reduction, same join point as breath['wob'])
                # breaks that reduction for the WHOLE file, not just this column.
                # Set directly on the already-built tables instead, the same
                # post-hoc pattern core.analysis.references.attach's own
                # ic_ref_source (also text) already uses.
                if mfvl_fev1_source is not None:
                    breaths_table["fev1_source"] = mfvl_fev1_source
                    average_row["fev1_source"] = mfvl_fev1_source
            processed = None
            if s.output.data.saveprocesseddata:
                processed = build_processed_data(breaths, s)

            # Retain the diagnostic signal arrays for the plotting/audio consumer (only when
            # a figure or WAV export could need them). Never touches the result DataFrames.
            signals = None
            if _diag_wanted(settings):
                fs = s.input.format.samplingfrequency
                # timecol (and breath["time"]) run in ABSOLUTE seconds starting at startix/fs,
                # so the R-peak capture times stay absolute too — just clip to the trimmed window.
                emg_peaks = np.asarray(ecg_diag["peaks_s"], float) if ecg_diag else np.array([])
                if emg_peaks.size:
                    emg_peaks = emg_peaks[(emg_peaks >= startix / fs) & (emg_peaks <= endix / fs)]
                signals = {
                    "fs": float(fs),
                    "time": np.asarray(timecol, float), "flow": np.asarray(flow, float),
                    "vol_uncorrected": np.asarray(vol_uncorrected, float),
                    "vol_zeroed": np.asarray(zerovol, float),
                    "vol_drift": np.asarray(driftvol, float), "vol_final": np.asarray(volume, float),
                    "drift_on": bool(s.processing.mechanics.correctvolumedrift),
                    "trend_on": bool(s.processing.mechanics.correctvolumetrend),
                    "raw_time": np.asarray(timecolraw, float), "raw_flow": np.asarray(flowraw, float),
                    "raw_volume": np.asarray(volumeraw, float),
                    # An absent pressure channel (caps.poes/pgas/pdi False) is None here, not an
                    # empty array — R1's "documented absence, not a hidden empty/NaN" principle.
                    # The plots/plan consumers guard against None themselves (core/plots.py); every
                    # existing (full-channel) consumer sees an array unchanged, since
                    # caps.poes/pgas/pdi are always True on that path.
                    "raw_poes": np.asarray(poesraw, float) if s.capabilities.poes else None,
                    "raw_pgas": np.asarray(pgasraw, float) if s.capabilities.pgas else None,
                    "raw_pdi": np.asarray(pdiraw, float) if s.capabilities.pdi else None,
                    "emg_stages": emg_stages, "emg_peaks": emg_peaks,
                    "emg_cols": list(s.input.data.columns_emg),
                }

            result.files[filename] = FileResult(
                file=filename, breaths_table=breaths_table, average_row=average_row,
                processed=processed, breaths=breaths, ecg=ecg_diag, signals=signals,
                manoeuvres=fr_manoeuvres, manoeuvres_table=manoeuvres_table,
                role="reference" if reference_only else "tidal",
                notices=file_notices)
            # M-30: average_rows feeds result.average_table (the cross-file "Average
            # breathdata" concat below) -- a reference-only file's average_row is always
            # None (there is no tidal average to contribute), so it is left out rather
            # than appended as a row of NaNs.
            if average_row is not None:
                average_rows.append(average_row)
            done_message = (f"{len(fr_manoeuvres)} reference manoeuvre"
                            f"{'s' if len(fr_manoeuvres) != 1 else ''}" if reference_only
                            else f"{total} breaths")
            _emit(progress, ProgressEvent("file_done", file=filename, message=done_message))

        except Exception as e:
            result.files[filename] = FileResult(file=filename, error=f"{type(e).__name__}: {e}",
                                                error_kind=type(e).__name__)
            _emit(progress, ProgressEvent("file_error", file=filename, message=str(e)))

    # Cross-file reference efterpass: resolve every OK tidal file's IC reference and
    # attach vol_ic_ref/ic_ref_n/ic_ref_source (references.attach's own docstring has
    # the full contract) BEFORE average_table is concatenated, so the new columns are
    # already on each average_row when pd.concat's default outer join runs below.
    # `require_references` can DEMOTE a file (fr.error/fr.error_kind set on the same
    # FileResult already in result.files) -- average_rows is therefore rebuilt from
    # result.ok_files AFTER attach() runs, never reused from the plain list the main
    # loop above appended to, so a demoted file's row is excluded from the cohort the
    # same way an ordinary main-loop failure already is. When nothing was demoted
    # (the common case: require_references off, or every file resolved) this is the
    # exact same set of average_row objects, in the same order, as the list above.
    referenceslib.attach(result, settings, allfiles)
    # M-36: operating lung volumes (EELV/EILV/IRV per tidal breath, TLC/VC/delta_ic
    # per file) CONSUME references.attach's vol_ic_ref, so it must run strictly after
    # -- and, like it, before average_rows is rebuilt below, so the new columns are
    # already on each average_row when pd.concat's outer join runs.
    lungvollib.attach(result, settings, allfiles)
    # Normalisation to a maximal manoeuvre (opt-in): reads the per-file max_insp/sniff
    # reference the main loop's extraction (or the forepass) already produced, and adds a
    # separate table, so it touches neither breaths_table nor average_row.
    normalisationlib.attach(result, settings)
    average_rows = [fr.average_row for fr in result.ok_files.values()
                    if fr.average_row is not None]

    if average_rows:
        import pandas as pd
        result.average_table = average_rows[0] if len(average_rows) == 1 else pd.concat(average_rows)
    _emit(progress, ProgressEvent("finished", message=f"{len(result.ok_files)}/{len(files)} files ok"))
    return result

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
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from respmech.core import compute
from respmech.core import emg as emglib
from respmech.core.io.loaders import load
from respmech.core.results import build_breath_table, build_processed_data
from respmech.core.settings import Settings
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
    error: Optional[str] = None


@dataclass
class BatchResult:
    files: dict = field(default_factory=dict)   # filename -> FileResult
    average_table: object = None                # concatenated average rows
    noise_report: object = None                 # shared-profile prop + per-channel fidelity/ΔSNR

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


def _emg_segmented(path, s, cache=None, cancel_check=None):
    """Load a file, ECG-remove + trim its EMG, and segment breaths (by flow).
    Returns (emg_ecg_trimmed, insp_mask, exp_mask). Used to build the noise
    reference (expiration) and to gather active/quiet EMG for prop selection.
    ``cache`` (Wave 2.4) memoises the load + ECG removal across the run."""
    (flow, vol, poes, pgas, pdi, ent, emg), emgcols_ecg, _diag = _load_and_ecg(
        path, s, cache=cache, cancel_check=cancel_check)
    tc = np.arange(len(flow)) / s.input.format.samplingfrequency
    tcT, fT, vT, pT, gT, dT, _e, si, ei = compute.trim(
        tc, flow, vol, poes, pgas, pdi, np.array(emg) if len(emg) else np.array([]), s)
    emg_full = emgcols_ecg[si:ei]                 # ECG-removed (cached) then trimmed, as before
    volc = compute.correctdrift(compute.zero(vT), s) if s.processing.mechanics.correctvolumedrift else compute.zero(vT)
    br = compute.separateintobreaths("flow", os.path.basename(path), tcT, fT, volc,
                                     pT, gT, dT, [], emg_full, s)
    ins = np.zeros(len(fT), bool); ex = np.zeros(len(fT), bool); p = 0
    for b in br.values():
        ni = len(b["inspiration"]["poes"]); ne = len(b["expiration"]["poes"])
        ins[p:p + ni] = True; ex[p + ni:p + ni + ne] = True; p += ni + ne
    n = min(len(emg_full), len(ins))
    return emg_full[:n], ins[:n], ex[:n]


def _reference_noise_clip(settings, s, cache=None, cancel_check=None):
    """Build the EMG-free noise reference clip (multichannel) once per test."""
    ns_cfg = settings.processing.emg.noise
    ref = ns_cfg.reference_file
    if not ref:
        raise ValueError("processing.emg.noise.reference_file is required when noise reduction is enabled")
    path = os.path.join(s.input.inputfolder, ref)
    fs = s.input.format.samplingfrequency
    # Prefer expiration-based reference (many STFT frames -> stable estimate). Explicit
    # intervals are used only when use_expiration is False (a deliberate override).
    if ns_cfg.use_expiration or not ns_cfg.reference_intervals:
        emg_full, ins, ex = _emg_segmented(path, s, cache=cache, cancel_check=cancel_check)
        clip = emg_full[ex]   # diaphragm-quiet expiration of the rest reference
    else:
        _load_result, emg_ecg, _diag = _load_and_ecg(path, s, cache=cache, cancel_check=cancel_check)
        parts = [emg_ecg[int(t0 * fs):int(t1 * fs)] for t0, t1 in ns_cfg.reference_intervals]
        clip = np.concatenate(parts, axis=0)
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
        # Gather active/quiet EMG across the test (capped) to choose prop ONCE.
        act, qui, cap = [], [], 40000
        for fi in files:
            emg_full, ins, ex = _emg_segmented(os.path.abspath(fi), s, cache=cache,
                                               cancel_check=cancel_check)
            act.append(emg_full[ins]); qui.append(emg_full[ex])
            if sum(len(a) for a in act) >= cap:
                break
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
    fs_in = int(s.input.format.samplingfrequency)
    stft_override = None
    if settings.processing.sampling.resample:
        fs_out = int(settings.processing.sampling.resample_to_frequency)
        if fs_out > 0 and fs_out != fs_in:
            s.input.format.samplingfrequency = fs_out          # analysis rate for all downstream
            s.input.format.samplingfrequency_in = fs_in        # true file rate, read by _load
            stft_override = _coupled_stft(settings.processing.emg.noise, fs_in, fs_out)

    allfiles = match_input_files(s.input.inputfolder, s.input.files)   # the full test (defines the shared profile)
    files = [f for f in allfiles if only_files is None or os.path.basename(f) in set(only_files)]
    if len(files) == 0:
        raise FileNotFoundError(
            f"No input files found for '{s.input.files}' in '{s.input.inputfolder}'")

    result = BatchResult()
    average_rows = []

    # ONE shared noise profile + parameter set for the whole test, built once and
    # applied identically to every file (never re-tuned per file). Built from the
    # full test even when only_files restricts processing (e.g. a GUI test run), so a
    # single file is denoised exactly as it would be in the full batch.
    noise_set = None
    # Per-run load + ECG-removal cache (Wave 2.4). Only useful when the noise-building phase
    # runs (it is the sole source of duplicate load/ECG work); the main loop drains it by
    # popping each file it reaches, so it never holds more than the noise phase loaded.
    load_cache: dict = {}
    if settings.processing.emg.noise.enabled and len(s.input.data.columns_emg) > 0:
        _emit(progress, ProgressEvent("stage", message="building shared noise profile"))
        noise_set, result.noise_report = _build_noise_set(settings, s, allfiles, progress,
                                                          stft=stft_override, cache=load_cache)

    for fi in files:
        if cancel_check is not None and cancel_check():
            _emit(progress, ProgressEvent("finished", message="cancelled"))
            return result
        file = os.path.abspath(fi)
        filename = os.path.basename(file)
        _emit(progress, ProgressEvent("file_start", file=filename, message="loading"))
        try:
            # Reuse the load + ECG removal if the noise-building phase already did it for this
            # file (Wave 2.4). pop() → the entry is consumed once and freed, so the cache never
            # outgrows the noise phase. Miss ⇒ load fresh and let _process_emg do ECG removal.
            _cached = load_cache.pop(file, None)
            if _cached is not None:
                (flowraw, volumeraw, poesraw, pgasraw, pdiraw, entropycolumnsraw,
                 emgcolumnsraw), _ecg_full, _ecg_diag_full = _cached
                ecg_precomputed = (_ecg_full, _ecg_diag_full)
            else:
                flowraw, volumeraw, poesraw, pgasraw, pdiraw, entropycolumnsraw, emgcolumnsraw = _load(file, s)
                ecg_precomputed = None
            timecolraw = np.arange(0, len(flowraw), dtype=int) / s.input.format.samplingfrequency

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

            _emit(progress, ProgressEvent("stage", file=filename, message="volume correction"))
            vol_uncorrected = volume                                  # trimmed, pre-zero
            zerovol = compute.zero(volume)
            driftvol = compute.correctdrift(zerovol, s) if s.processing.mechanics.correctvolumedrift else zerovol
            volume = compute.correcttrend(driftvol, s) if s.processing.mechanics.correctvolumetrend else driftvol

            _emit(progress, ProgressEvent("stage", file=filename, message="segmenting breaths"))
            breaths = compute.separateintobreaths(
                s.processing.mechanics.separateby, filename, timecol, flow, volume,
                poes, pgas, pdi, entropycolumns, emgcolumns, s)

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
            gate_peaks, gate_ok, gate_reason = None, True, ""
            if s.processing.emg.robust_peak.enabled:
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

            total = sum(1 for b in breaths.values() if not b["ignored"])
            done = 0
            for breathno in breaths:
                breath = breaths[breathno]
                if breath["ignored"]:
                    continue
                compute.calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, s,
                                           cancel_check=cancel_check, peaks_s=gate_peaks,
                                           detection_ok=gate_ok, detection_reason=gate_reason)
                done += 1
                _emit(progress, ProgressEvent("breath", file=filename, breath=done, total_breaths=total))

            breaths_table, average_row = build_breath_table(filename, breaths, s)
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
                    "raw_volume": np.asarray(volumeraw, float), "raw_poes": np.asarray(poesraw, float),
                    "raw_pgas": np.asarray(pgasraw, float), "raw_pdi": np.asarray(pdiraw, float),
                    "emg_stages": emg_stages, "emg_peaks": emg_peaks,
                    "emg_cols": list(s.input.data.columns_emg),
                }

            result.files[filename] = FileResult(
                file=filename, breaths_table=breaths_table, average_row=average_row,
                processed=processed, breaths=breaths, ecg=ecg_diag, signals=signals)
            average_rows.append(average_row)
            _emit(progress, ProgressEvent("file_done", file=filename, message=f"{total} breaths"))

        except Exception as e:
            result.files[filename] = FileResult(file=filename, error=f"{type(e).__name__}: {e}")
            _emit(progress, ProgressEvent("file_error", file=filename, message=str(e)))

    if average_rows:
        import pandas as pd
        result.average_table = average_rows[0] if len(average_rows) == 1 else pd.concat(average_rows)
    _emit(progress, ProgressEvent("finished", message=f"{len(result.ok_files)}/{len(files)} files ok"))
    return result

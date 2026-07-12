"""Batch pipeline: orchestrates load -> condition -> segment -> compute for each
file, emitting progress events and returning typed results.

This is the seam that makes the CLI and the Qt GUI share one engine: the core emits
:class:`ProgressEvent` objects (via a callback); the CLI prints them, the GUI turns
them into Qt signals. The core performs **no file writing and no plotting** — those
are consumers of the returned results (see ``respmech.core.io.writers`` and
``respmech.plots``).

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
    kind: str                      # file_start|stage|breath|file_done|file_error|finished
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
            "peak_rms_before": before, "peak_rms_after": after, "suppression": supp}
    return emgcols, diag


def _ecg_remove_trim(s, emgcolumnsraw, startix, endix, cancel_check=None):
    """ECG removal (no diagnostics) then trim — used for internal reference building."""
    emgcols, _ = _ecg_remove(s, emgcolumnsraw, cancel_check=cancel_check)
    return emgcols[startix:endix]


def _process_emg(s, emgcolumnsraw, startix, endix, noise_set=None):
    """ECG removal -> trim -> shared-profile noise reduction (applied identically to
    every file when ``noise_set`` is provided). Returns (emgcols, ecg_diag)."""
    emgcols, ecg_diag = _ecg_remove(s, emgcolumnsraw)
    emgcols = emgcols[startix:endix]
    if noise_set is not None:
        emgcols = noise_set.apply_columns(emgcols)
    return emgcols, ecg_diag


def _emg_segmented(path, s, cancel_check=None):
    """Load a file, ECG-remove + trim its EMG, and segment breaths (by flow).
    Returns (emg_ecg_trimmed, insp_mask, exp_mask). Used to build the noise
    reference (expiration) and to gather active/quiet EMG for prop selection."""
    flow, vol, poes, pgas, pdi, ent, emg = load(path, s)
    tc = np.arange(len(flow)) / s.input.format.samplingfrequency
    tcT, fT, vT, pT, gT, dT, _e, si, ei = compute.trim(
        tc, flow, vol, poes, pgas, pdi, np.array(emg) if len(emg) else np.array([]), s)
    emg_full = _ecg_remove_trim(s, emg, si, ei, cancel_check=cancel_check)
    volc = compute.correctdrift(compute.zero(vT), s) if s.processing.mechanics.correctvolumedrift else compute.zero(vT)
    br = compute.separateintobreaths("flow", os.path.basename(path), tcT, fT, volc,
                                     pT, gT, dT, [], emg_full, s)
    ins = np.zeros(len(fT), bool); ex = np.zeros(len(fT), bool); p = 0
    for b in br.values():
        ni = len(b["inspiration"]["poes"]); ne = len(b["expiration"]["poes"])
        ins[p:p + ni] = True; ex[p + ni:p + ni + ne] = True; p += ni + ne
    n = min(len(emg_full), len(ins))
    return emg_full[:n], ins[:n], ex[:n]


def _reference_noise_clip(settings, s, cancel_check=None):
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
        emg_full, ins, ex = _emg_segmented(path, s, cancel_check=cancel_check)
        clip = emg_full[ex]   # diaphragm-quiet expiration of the rest reference
    else:
        flow, vol, poes, pgas, pdi, ent, emg = load(path, s)
        emg_ecg = _ecg_remove_trim(s, emg, 0, len(emg), cancel_check=cancel_check)  # full length (no trim)
        parts = [emg_ecg[int(t0 * fs):int(t1 * fs)] for t0, t1 in ns_cfg.reference_intervals]
        clip = np.concatenate(parts, axis=0)
    return clip


def _build_noise_set(settings, s, files, progress=None, clip=None, cancel_check=None):
    from respmech.core import noise as noiselib
    cfg = settings.processing.emg.noise
    # ``clip`` lets the PREVIEW pass a shared/cached reference clip; batch/CLI pass None
    # (default) and build it here exactly as before — byte-identical for the golden path.
    if clip is None:
        clip = _reference_noise_clip(settings, s, cancel_check=cancel_check)
    profiles = noiselib.build_profiles(
        clip, s.input.format.samplingfrequency, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.win_length, n_std_thresh=cfg.n_std_thresh,
        n_grad_freq=cfg.n_grad_freq, n_grad_time=cfg.n_grad_time, cancel_check=cancel_check)

    if cfg.auto_prop:
        # Gather active/quiet EMG across the test (capped) to choose prop ONCE.
        act, qui, cap = [], [], 40000
        for fi in files:
            emg_full, ins, ex = _emg_segmented(os.path.abspath(fi), s, cancel_check=cancel_check)
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
    if settings.processing.emg.noise.enabled and len(s.input.data.columns_emg) > 0:
        _emit(progress, ProgressEvent("stage", message="building shared noise profile"))
        noise_set, result.noise_report = _build_noise_set(settings, s, allfiles, progress)

    for fi in files:
        if cancel_check is not None and cancel_check():
            _emit(progress, ProgressEvent("finished", message="cancelled"))
            return result
        file = os.path.abspath(fi)
        filename = os.path.basename(file)
        _emit(progress, ProgressEvent("file_start", file=filename, message="loading"))
        try:
            flowraw, volumeraw, poesraw, pgasraw, pdiraw, entropycolumnsraw, emgcolumnsraw = load(file, s)
            timecolraw = np.arange(0, len(flowraw), dtype=int) / s.input.format.samplingfrequency

            _emit(progress, ProgressEvent("stage", file=filename, message="trimming"))
            (timecol, flow, volume, poes, pgas, pdi, _emgtrim, startix, endix) = compute.trim(
                timecolraw, flowraw, volumeraw, poesraw, pgasraw, pdiraw,
                np.array(emgcolumnsraw) if len(emgcolumnsraw) else np.array([]), s)

            # Bug #2 fix: trim entropy columns with the same window as everything else.
            entropycolumns = entropycolumnsraw[startix:endix] if len(entropycolumnsraw) else entropycolumnsraw

            emgcolumns = []
            ecg_diag = None
            if len(emgcolumnsraw) > 0:
                _emit(progress, ProgressEvent("stage", file=filename, message="processing EMG"))
                emgcolumns, ecg_diag = _process_emg(s, emgcolumnsraw, startix, endix, noise_set)

            _emit(progress, ProgressEvent("stage", file=filename, message="volume correction"))
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

            total = sum(1 for b in breaths.values() if not b["ignored"])
            done = 0
            for breathno in breaths:
                breath = breaths[breathno]
                if breath["ignored"]:
                    continue
                compute.calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, s,
                                           cancel_check=cancel_check)
                done += 1
                _emit(progress, ProgressEvent("breath", file=filename, breath=done, total_breaths=total))

            breaths_table, average_row = build_breath_table(filename, breaths, s)
            processed = None
            if s.output.data.saveprocesseddata:
                processed = build_processed_data(breaths, s)

            result.files[filename] = FileResult(
                file=filename, breaths_table=breaths_table, average_row=average_row,
                processed=processed, breaths=breaths, ecg=ecg_diag)
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

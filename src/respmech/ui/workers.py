"""Background worker that runs the batch off the GUI thread.

The core's progress-event callback is the seam (see ``core.pipeline``): the worker
simply forwards core :class:`ProgressEvent`s as Qt signals, so the GUI never blocks
and can stream output live. Cancellation is cooperative (checked between files).

Usage (moveToThread pattern):

    thread = QThread()
    worker = BatchWorker(settings, write=True)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    ... connect worker.progress / worker.failed ...
    thread.start()
"""
from __future__ import annotations

import os
import traceback

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from respmech.core.settings import Settings
from respmech.core._cancel import Cancelled


# ``run_batch`` and ``write_batch`` are the two imports that pulled scipy and pandas into
# GUI startup. They are deliberately module-level *shims* rather than imports moved inside
# BatchWorker.run: the unit suite monkeypatches these names on this module
# (tests/unit/test_gui_hardening.py), which needs them to exist as module attributes.
# Because BatchWorker.run resolves the module global at call time, the monkeypatch still
# takes effect through the shim. Settings/Cancelled stay eager -- they are cheap, and
# Settings is used in annotations.

def run_batch(*args, **kwargs):
    """Lazy shim for :func:`respmech.core.pipeline.run_batch` — see the note above."""
    from respmech.core.pipeline import run_batch as _fn
    return _fn(*args, **kwargs)


def write_batch(*args, **kwargs):
    """Lazy shim for :func:`respmech.core.io.writers.write_batch` — see the note above."""
    from respmech.core.io.writers import write_batch as _fn
    return _fn(*args, **kwargs)


def write_planned(*args, **kwargs):
    """Lazy shim for :func:`respmech.core.io.writers.write_planned` — see the note above."""
    from respmech.core.io.writers import write_planned as _fn
    return _fn(*args, **kwargs)


class BatchWorker(QObject):
    progress = Signal(object)          # ProgressEvent
    file_done = Signal(str, object)    # filename, average-row DataFrame
    file_failed = Signal(str, str)     # filename, error
    finished = Signal(object)          # BatchResult (or None if cancelled/failed)
    failed = Signal(str)               # fatal error message

    def __init__(self, settings: Settings, write: bool = True, only_files=None):
        super().__init__()
        self._settings = settings
        self._write = write
        self._only = only_files
        self._cancel = False
        # Set True the instant this thread commits to the uninterruptible write phase (A07)
        # — a plain attribute, read directly by RunScreen._cancel() on the GUI thread. A
        # Cancel click cannot wait for the "writing" ProgressEvent to make it through Qt's
        # QueuedConnection (necessarily asynchronous, since this thread must never touch
        # widgets): that gap is exactly wide enough for a click landing in it to still log
        # the old, plain "cancelling…" line for a click that in fact arrived too late to
        # cancel anything. CPython's GIL makes a bool assignment/read atomic across threads,
        # so this closes the race to the width of one bytecode op instead of one event-loop
        # tick.
        self._writing = False
        # D20: the paths write_batch actually wrote, read by RunScreen._on_finished before
        # self._worker is discarded, so the finished-status line can name what was written
        # instead of re-deriving a partial guess from the pre-flight plan (which — unlike
        # this — never covers diagnostics figures or the cohort summary). Stays empty for a
        # dry run (write_batch is never called) or a run whose write raised before returning.
        self.written = []

    @Slot()
    def cancel(self):
        self._cancel = True

    def _on_event(self, ev):
        # Surface per-file completion/failure as dedicated signals for live updates.
        if ev.kind == "file_done":
            self.file_done.emit(ev.file, None)
        elif ev.kind == "file_error":
            self.file_failed.emit(ev.file, ev.message)
        self.progress.emit(ev)

    @Slot()
    def run(self):
        # Analysis and writing fail differently: a failed run has no result, but a
        # failed WRITE still has a fully computed result worth showing.
        try:
            result = run_batch(self._settings, progress=self._on_event,
                               cancel_check=lambda: self._cancel,
                               only_files=self._only)
        except Cancelled:  # cooperatively aborted mid-file (superseded/settings change/shutdown)
            # Cancelled subclasses BaseException so the `except Exception` below can't catch
            # it; deliver None (the preview drops a superseded payload silently) and let the
            # thread exit cleanly instead of crashing the QThread with an unhandled exception.
            self.finished.emit(None)
            return
        except Exception:  # fatal (e.g. no files, invalid settings)
            self.failed.emit(traceback.format_exc())
            self.finished.emit(None)
            return
        # Latch the cancel decision ONCE, when the analysis phase ends, so a Cancel
        # click that lands during writing can't be reported as "no output written".
        cancelled = self._cancel
        if self._write and not cancelled:
            self._writing = True     # see __init__ — set before anything else in this phase
            # Signal the Run screen that the (previously silent) write phase has begun, so it
            # can switch the progress bar to its animated busy state and stop looking frozen.
            # Lazy import keeps pipeline out of GUI startup (Wave 1.4); it is already loaded
            # here because run_batch above imported it.
            from respmech.core.pipeline import ProgressEvent
            self._on_event(ProgressEvent("writing", message="writing output"))
            try:
                # is_subset_run inside the try: it re-lists the input folder, and a folder
                # that becomes transiently unreadable between analysis and writing must be
                # reported as a write failure like any other, not crash the worker thread
                # with an unhandled exception (no failed/finished signal at all).
                from respmech.core.pipeline import is_subset_run
                cohort_outputs = not is_subset_run(self._settings, self._only)
                self.written = write_batch(result, self._settings, self._settings.output.folder,
                                           progress=self._on_event, cohort_outputs=cohort_outputs)
            except Exception:
                self.failed.emit(
                    "Analysis completed, but writing the output failed. Some files may be "
                    f"partially written in:\n  {self._settings.output.folder}\n\n"
                    + traceback.format_exc())
                self.finished.emit(result)      # still deliver the computed result
                return
        # Final result carries every file's table for the results view.
        self.finished.emit(None if cancelled else result)


class WriteWorker(QObject):
    """Write an ALREADY-COMPUTED ``BatchResult`` to a folder, off the GUI thread (ticket
    A06's "Write results to another folder…" — offered when the analysis succeeded but the
    original write failed). Never re-runs the analysis: ``write_planned`` only replays
    ``write_batch``'s own write logic against the frozen ``result``/``settings`` this worker
    was constructed with. On a worker thread for the same reason ``BatchWorker``'s write
    phase is: the figure step alone can take many seconds, and this app's one rule for
    anything that slow is that it must never block the window."""
    finished = Signal(object)      # list of written paths
    failed = Signal(str)           # error message (traceback)

    def __init__(self, result, settings, plan, outputfolder: str):
        super().__init__()
        self._result = result
        self._settings = settings
        self._plan = plan
        self._outputfolder = outputfolder

    @Slot()
    def run(self):
        try:
            written = write_planned(self._result, self._settings, self._plan,
                                    outputfolder=self._outputfolder)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(written)


# --------------------------------------------------------------------------- #
# EMG-conditioning preview (Preview & tuning screen). Stages ONE EMG channel
# through the exact core conditioning the batch would apply, so the tuning view
# matches the real run. Pure compute here (Qt-free, no disk writes) so it is both
# unit-testable headless and safe to run on a QThread.
# --------------------------------------------------------------------------- #
def _cached_reference_clip(settings: Settings, s, cancel_check=None):
    """The reference noise clip, memoised per (reference-file freshness + load/ECG/
    segmentation inputs) so emg_all + emg_detail + noise share ONE build of it within a
    scoped recompute instead of three. Read-only (callers only build profiles from it),
    so no defensive copy is needed. Preview-only wrapper around the core clip builder."""
    from respmech.core.pipeline import _reference_noise_clip
    from respmech.ui.screens import _preview_cache as _pc
    ref = settings.processing.emg.noise.reference_file
    ref_path = os.path.join(s.input.inputfolder, ref) if ref else None
    key = _pc.ref_clip_key(settings, ref_path) if ref_path else None
    return _pc.cached(_pc.REF_CLIP, key,
                      lambda: _reference_noise_clip(settings, s, cancel_check=cancel_check))


def _detect_channel(s, nch):
    """The ECG capture channel index, validated.

    The preview used to CLAMP an out-of-range index to the last channel while the core
    indexes it straight (``emgcols[:, detect]``, pipeline._ecg_remove) — so a misconfigured
    detect_channel silently captured off the wrong channel here and only blew up in the run.
    Fail here too, with a message that says which setting is wrong."""
    detect = int(s.processing.emg.column_detect)
    if not (0 <= detect < nch):
        raise ValueError(
            f"ECG detection channel #{detect + 1} is outside the {nch} configured EMG "
            f"channel{'s' if nch != 1 else ''}. Pick a capture channel that exists.")
    return detect


def _effective_prop(settings: Settings, s):
    """The suppression strength the BATCH would apply, for the EMG preview panels.

    Under ``auto_prop`` the run does not use ``cfg.prop_decrease`` at all — ``_build_noise_set``
    picks the value with ``select_prop_decrease`` and applies that. The preview panels used to
    apply their own snapshot value, so they showed a differently-gated signal than the run
    (permanently so when the fidelity job failed, since only its success writes the chosen
    value back into settings). Resolve it from the shared, already-computed noise report
    instead — a pure cache LOOKUP, never a compute: if the report is not there yet the panels
    fall back to the snapshot exactly as before, and the fidelity job's completion
    re-dispatches them with the chosen value."""
    cfg = settings.processing.emg.noise
    if not cfg.auto_prop:
        return float(cfg.prop_decrease)
    try:
        from respmech.core.pipeline import match_input_files          # noqa: PLC0415
        from respmech.ui.screens import _preview_cache as _pc         # noqa: PLC0415
        ref = cfg.reference_file
        if not ref:
            return float(cfg.prop_decrease)
        ref_path = os.path.join(s.input.inputfolder, ref)
        key = _pc.noise_report_key(settings, ref_path,
                                   match_input_files(s.input.inputfolder, s.input.files))
        report = _pc.NOISE_REPORT.get(key) if key is not None else None
        chosen = (report or {}).get("prop_decrease")
        return float(chosen) if chosen is not None else float(cfg.prop_decrease)
    except Exception:                                # noqa: BLE001 — never fail a render over this
        return float(cfg.prop_decrease)


def _load_and_condition(settings: Settings, s, file_path: str, cancel_check=None):
    """Load a file's EMG matrix and ECG-remove ALL channels, memoised per (file freshness
    + load + ECG params) so emg_all and emg_detail share one ECG pass (the dominant EMG
    cost) instead of each doing the full-matrix subtraction. Returns COPIES of
    ``(raw[n,nch], conditioned[n,nch], ecg_applied, ecg_error)`` so a caller that mutates
    the conditioned matrix in place (the per-channel noise apply) cannot corrupt the
    cached entry."""
    from respmech.ui.screens import _preview_cache as _pc

    def _compute():
        from respmech.core.io.loaders import load  # noqa: PLC0415
        from respmech.core import emg as emglib     # noqa: PLC0415
        fs = int(s.input.format.samplingfrequency)
        flow_col, _v, _p, _g, _d, _e, emg = load(file_path, s)
        flow_col = np.asarray(flow_col, dtype=float).ravel()   # full-length, untrimmed (for the EMG-vs-flow overlay)
        emg = np.asarray(emg, dtype=float)
        if emg.ndim == 1:
            emg = emg[:, None]
        raw = np.array(emg, dtype=float)
        conditioned = np.array(emg, dtype=float)
        applied, error = False, None
        nch = emg.shape[1]
        if nch and settings.processing.emg.remove_ecg:
            try:
                detect = _detect_channel(s, nch)
                proc, _w, _pk = emglib.remove_ecg(
                    np.array(emg), emg[:, detect], samplingfrequency=fs,
                    ecgminheight=s.processing.emg.minheight,
                    ecgmindistance=s.processing.emg.mindistance,
                    ecgminwidth=s.processing.emg.minwidth,
                    windowsize=s.processing.emg.windowsize,
                    cancel_check=cancel_check)
                conditioned = np.asarray(proc, dtype=float)
                if conditioned.ndim == 1:
                    conditioned = conditioned[:, None]
                applied = True
            except Exception:                          # noqa: BLE001 — preview stays useful
                conditioned = np.array(emg, dtype=float)
                error = traceback.format_exc()
        return raw, conditioned, applied, error, flow_col

    raw, cond, applied, error, flow_col = _pc.cached(
        _pc.ECG_MATRIX, _pc.ecg_matrix_key(settings, file_path), _compute)
    return (np.array(raw, dtype=float), np.array(cond, dtype=float), applied, error,
            np.array(flow_col, dtype=float))


def stage_emg_channel(settings: Settings, file_path: str, channel_index: int,
                      cancel_check=None) -> dict:
    """Stage one EMG channel of ``file_path`` through the conditioning pipeline and
    return time-domain + PSD arrays for plotting:

        raw -> ECG-removed (averaged-template subtraction)
            -> shared-profile spectral noise reduction (this channel only)

    Reuses the core exactly as ``run_batch`` does. The noise stage builds this
    channel's profile from the same reference clip the batch would use
    (``pipeline._reference_noise_clip``), so the preview is faithful. Stages that
    are not configured fall through unchanged (no reference -> ``noise`` == ``ecg``).
    Never writes to disk and never touches Qt."""
    from scipy.signal import welch

    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core import noise as noiselib

    s = to_legacy_ns(settings)
    fs = int(s.input.format.samplingfrequency)
    # stages 1+2 (load + all-channel ECG removal) are shared with emg_all via the cache
    raw_mat, cond_mat, ecg_applied, ecg_error, flow_full = _load_and_condition(
        settings, s, file_path, cancel_check=cancel_check)
    nch = raw_mat.shape[1]
    if nch == 0:
        raise ValueError("No EMG channels are configured for this file.")
    ch = max(0, min(int(channel_index), nch - 1))
    raw = raw_mat[:, ch].astype(float)
    ecg = cond_mat[:, ch].astype(float)                # == raw when remove_ecg off / failed
    cols = list(settings.input.channels.emg)
    col = cols[ch] if ch < len(cols) else ch + 1

    # -- stage 3: shared-profile spectral noise reduction on this channel --
    cfg = settings.processing.emg.noise
    noise_out = ecg
    noise_applied = False
    noise_error = None
    prop = _effective_prop(settings, s)
    # Gate exactly as run_batch does (pipeline: `noise.enabled and columns_emg`). Gating on
    # reference_file alone meant unticking noise reduction still showed a spectrally gated
    # trace — a signal the run would not produce, with the fidelity panel (which DOES honour
    # `enabled`) gone from beside it.
    if cfg.enabled and cfg.reference_file:
        try:
            clip = np.asarray(_cached_reference_clip(settings, s, cancel_check=cancel_check),
                              dtype=float)
            if clip.ndim == 1:
                clip = clip[:, None]
            prof = noiselib.NoiseProfile.from_clip(
                clip[:, min(ch, clip.shape[1] - 1)], fs,
                n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
                n_std_thresh=cfg.n_std_thresh, n_grad_freq=cfg.n_grad_freq,
                n_grad_time=cfg.n_grad_time)
            noise_out = np.asarray(prof.apply(ecg, prop), dtype=float)
            noise_applied = True
        except Exception:                              # noqa: BLE001 — e.g. clip too short
            noise_out = ecg
            noise_error = traceback.format_exc()

    t = np.arange(len(raw), dtype=float) / fs

    def _psd(x):
        x = np.asarray(x, dtype=float)
        f, p = welch(x, fs, nperseg=int(min(1024, len(x))) or 1)
        return f, p

    freqs, psd_raw = _psd(raw)
    _, psd_ecg = _psd(ecg)
    _, psd_noise = _psd(noise_out)
    return {
        "channel": ch, "col": col, "fs": fs, "t": t,
        "raw": raw, "ecg": ecg, "noise": noise_out, "flow": flow_full,
        "freqs": freqs, "psd_raw": psd_raw, "psd_ecg": psd_ecg, "psd_noise": psd_noise,
        "band": noiselib.EMG_BAND, "noise_applied": noise_applied, "prop_decrease": prop,
        "ecg_applied": ecg_applied, "ecg_error": ecg_error, "noise_error": noise_error,
    }


class EmgConditioningWorker(QObject):
    """Off-thread EMG-conditioning staging for the Preview screen. Same
    ``moveToThread`` seam as :class:`BatchWorker`: it never touches Qt widgets and
    only emits results. ``finished`` carries the arrays dict (or ``None`` on error)."""
    finished = Signal(object)   # dict of arrays, or None
    failed = Signal(str)        # error message

    def __init__(self, settings: Settings, file_path: str, channel_index: int):
        super().__init__()
        self._settings = settings
        self._path = file_path
        self._ch = channel_index
        self._cancelled = False

    @Slot()
    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        from respmech.core._cancel import Cancelled
        if self._cancelled:                 # superseded before we got the thread
            self.finished.emit(None)
            return
        try:
            data = stage_emg_channel(self._settings, self._path, self._ch,
                                     cancel_check=lambda: self._cancelled)
            self.finished.emit(data)
        except Cancelled:                   # superseded mid-compute -> aborted cleanly, no result
            self.finished.emit(None)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
            self.finished.emit(None)


def stage_emg_all_channels(settings: Settings, file_path: str, cancel_check=None) -> dict:
    """Stage EVERY configured EMG channel of ``file_path`` (raw -> conditioned = ECG
    removal + shared-profile noise reduction) for the "result" view where the user
    picks which channels to see. ECG removal runs once for all channels; the noise
    profile is built per channel from the shared reference clip. Pure compute, Qt-free,
    no disk writes."""
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core import noise as noiselib

    s = to_legacy_ns(settings)
    fs = int(s.input.format.samplingfrequency)
    cols = list(settings.input.channels.emg)
    # load + all-channel ECG removal, shared with emg_detail via the cache; the cache
    # returns COPIES, so the in-place per-channel noise apply below cannot corrupt it
    emg, conditioned, ecg_applied, ecg_error, flow_full = _load_and_condition(
        settings, s, file_path, cancel_check=cancel_check)
    nch = emg.shape[1]
    if nch == 0:
        raise ValueError("No EMG channels are configured for this file.")

    noise_applied = False
    noise_error = None
    cfg = settings.processing.emg.noise
    prop = _effective_prop(settings, s)
    if cfg.enabled and cfg.reference_file:             # same gate as run_batch (see stage_emg_channel)
        try:
            clip = np.asarray(_cached_reference_clip(settings, s, cancel_check=cancel_check),
                              dtype=float)
            if clip.ndim == 1:
                clip = clip[:, None]
            # Denoise into a SCRATCH copy and adopt it only once EVERY channel has succeeded.
            # Applying in place meant a failure partway (or a cancel) left channels 0..i-1
            # denoised and the rest not, while the payload still said noise_applied=False —
            # a mixed stack presented as ECG-only.
            out = np.array(conditioned, dtype=float)
            for i in range(nch):
                if cancel_check is not None and cancel_check():
                    from respmech.core._cancel import Cancelled
                    raise Cancelled()
                prof = noiselib.NoiseProfile.from_clip(
                    clip[:, min(i, clip.shape[1] - 1)], fs,
                    n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
                    n_std_thresh=cfg.n_std_thresh, n_grad_freq=cfg.n_grad_freq,
                    n_grad_time=cfg.n_grad_time)
                out[:, i] = prof.apply(conditioned[:, i], prop)
            conditioned = out
            noise_applied = True
        except Exception:                              # noqa: BLE001
            noise_error = traceback.format_exc()

    t = np.arange(emg.shape[0], dtype=float) / fs
    return {
        "t": t, "fs": fs, "cols": cols, "flow": flow_full,
        "raw": [emg[:, i].astype(float) for i in range(nch)],
        "conditioned": [conditioned[:, i].astype(float) for i in range(nch)],
        "noise_applied": noise_applied, "noise_error": noise_error, "prop_decrease": prop,
        "ecg_applied": ecg_applied, "ecg_error": ecg_error,
    }


def stage_raw_emg(settings: Settings, file_path: str) -> dict:
    """Load just the RAW EMG channels of ``file_path`` (no ECG/noise processing) for
    the noise-profile picker. Pure compute, Qt-free, no disk writes."""
    from respmech.core.io.loaders import load
    from respmech.core._legacy_ns import to_legacy_ns

    s = to_legacy_ns(settings)
    fs = int(s.input.format.samplingfrequency)
    cols = list(settings.input.channels.emg)
    flow, _vol, _poes, _pgas, _pdi, _ent, emg = load(file_path, s)
    emg = np.asarray(emg, dtype=float)
    if emg.ndim == 1:
        emg = emg[:, None]
    t = np.arange(emg.shape[0], dtype=float) / fs
    return {"t": t, "fs": fs, "cols": cols,
            "flow": np.asarray(flow, dtype=float).ravel(),   # full-length, untrimmed (EMG-vs-flow overlay)
            "raw": [emg[:, i].astype(float) for i in range(emg.shape[1])]}


def stage_ecg_reduction(settings: Settings, file_path: str, cancel_check=None) -> dict:
    """Stage the "EMG – ECG reduction" tab: detect the R-peaks on the chosen capture channel and
    (when ECG removal is ON) subtract the averaged ECG template from every EMG channel. The tab
    is a live TUNING surface, so the R-peak "capture" is ALWAYS computed and shown; when removal
    is OFF the processed stack shows the RAW channels (matching what the run will do), so the user
    can dial in a capture that catches only R-peaks before enabling removal. Qt-free; memoised in
    a DEDICATED preview cache (never the shared ECG_MATRIX). Returns raw capture + processed
    channels + R-peak TIMES (s) + full-length flow (for the shared flow overlay)."""
    from respmech.core.io.loaders import load
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core import emg as emglib
    from respmech.ui.screens import _preview_cache as _pc

    s = to_legacy_ns(settings)
    fs = int(s.input.format.samplingfrequency)
    cols = list(settings.input.channels.emg)

    def _compute():
        flow, _v, _p, _g, _d, _e, emg = load(file_path, s)
        emg = np.asarray(emg, dtype=float)
        if emg.ndim == 1:
            emg = emg[:, None]
        nch = emg.shape[1]
        if nch == 0:
            raise ValueError("No EMG channels are configured for this file.")
        flow_full = np.asarray(flow, dtype=float).ravel()
        detect = _detect_channel(s, nch)
        t = np.arange(emg.shape[0], dtype=float) / fs
        remove = bool(settings.processing.emg.remove_ecg)
        ecg_error = None
        peaks = np.array([], dtype=float)
        processed = [emg[:, i].astype(float) for i in range(nch)]
        # None (not NaN) while removal is off: there is nothing to suppress, so the tab
        # must show no number at all, distinct from "computed but degenerate" (NaN).
        suppression = None
        try:
            if remove:                                     # detect + subtract (the real pipeline)
                proc, _w, peak_times = emglib.remove_ecg(
                    np.array(emg), emg[:, detect], samplingfrequency=fs,
                    ecgminheight=s.processing.emg.minheight,
                    ecgmindistance=s.processing.emg.mindistance,
                    ecgminwidth=s.processing.emg.minwidth,
                    windowsize=s.processing.emg.windowsize, cancel_check=cancel_check)
                proc = np.asarray(proc, dtype=float)
                if proc.ndim == 1:
                    proc = proc[:, None]
                processed = [proc[:, i].astype(float) for i in range(proc.shape[1])]
                peaks = np.asarray(peak_times, dtype=float)
                # Same peak-window-RMS suppression core.pipeline._ecg_remove reports (and
                # writers.py persists to run-report.txt) — computed here too, on the SAME
                # helper, so the tuning surface has a number to judge "Min height"/"Min
                # gap" by before a real run ever happens.
                peaks_samp = (peaks * fs).astype(int)
                before = emglib.peak_window_rms(emg[:, detect].astype(float), peaks_samp, fs)
                after = emglib.peak_window_rms(proc[:, detect].astype(float), peaks_samp, fs)
                suppression = (1.0 - after / before) if (before and before == before) else float("nan")
            else:                                          # detect ONLY (cheap); processed == raw
                from scipy import signal as _sig
                pk, _props = _sig.find_peaks(
                    emg[:, detect], height=s.processing.emg.minheight,
                    distance=s.processing.emg.mindistance * fs,
                    width=s.processing.emg.minwidth * fs)
                peaks = pk / fs
        except Cancelled:
            raise
        except Exception:                                  # noqa: BLE001 — bad params -> raw + no capture
            ecg_error = traceback.format_exc()
            suppression = None
        return {"t": t, "fs": fs, "cols": cols, "detect": detect,
                "detect_col": cols[detect] if detect < len(cols) else detect + 1,
                "raw_capture": emg[:, detect].astype(float), "processed": processed,
                "peaks": peaks, "flow": flow_full, "ecg_applied": remove, "ecg_error": ecg_error,
                "suppression": suppression}

    got = _pc.cached(_pc.ECG_REDUCTION, _pc.ecg_matrix_key(settings, file_path), _compute)
    # Copy on return, the same contract _load_and_condition documents: this dict is a SHARED
    # cache entry, and a renderer that ever normalised/decimated an array in place would
    # silently poison every later hit.
    out = dict(got)
    for k in ("raw_capture", "flow", "t", "peaks"):
        if isinstance(out.get(k), np.ndarray):
            out[k] = np.array(out[k], dtype=float)
    if isinstance(out.get("processed"), list):
        out["processed"] = [np.array(a, dtype=float) for a in out["processed"]]
    return out


def load_raw_matrix(settings: Settings, file_path: str):
    """Read EVERY column of a raw data file for the visual channel-assignment dialog.

    Returns ``(matrix, names)``: ``matrix`` is a ``(samples, columns)`` float array with NO
    channel mapping applied (non-numeric cells -> NaN); ``names`` are the source column
    headers (or "" when a format has none). Mirrors the core loaders' file reading on the
    UI side so ``core`` stays untouched — crucially, the columns are 1-based-aligned with
    what ``core.load`` will index at run time. Raises ``ValueError`` for an unsupported
    file type."""
    import os

    import pandas as pd  # noqa: PLC0415

    from respmech.core.io.loaders import _read_table  # noqa: PLC0415 — same encoding-tolerant reader as core.load

    ext = os.path.splitext(file_path)[1].lower()
    fmt = settings.input.format

    def _from_df(df):
        names = [str(c) for c in df.columns]
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return arr, names

    if ext in (".csv",):
        # match core.loadcsv: honour the decimal separator (';'-CSV for a comma decimal)
        dec = getattr(fmt, "decimal", ".") or "."
        arr, names = _from_df(_read_table(file_path, sep=";" if dec == "," else ",", decimal=dec))
    elif ext in (".txt",):
        dec = getattr(fmt, "decimal", ".") or "."
        arr, names = _from_df(_read_table(file_path, sep="\t", decimal=dec))
    elif ext in (".xlsx",):
        arr, names = _from_df(pd.read_excel(file_path))
    elif ext in (".mat",):
        import scipy.io as sio  # noqa: PLC0415
        from collections import OrderedDict  # noqa: PLC0415
        try:
            data = OrderedDict(sio.loadmat(file_path))
            if getattr(fmt, "matlab_variant", "mac") == "mac":
                # legacy format 2: core indexes list(data.items()) 1-based, metadata
                # (__header__/__version__/__globals__) INCLUDED — keep them as placeholder
                # columns so the dialog's column N lines up with core.load's column N.
                items = list(data.items())
                names = [str(k) for k, _v in items]
                raw = [v for _k, v in items]
            else:                              # windows/format 1: a single data_block1
                raw = list(data["data_block1"])
                names = ["" for _ in raw]
        except Exception as e:                 # match core's actionable guidance
            raise ValueError("Cannot read MATLAB file — verify the MATLAB file format "
                             "setting (windows/mac); otherwise export to CSV.") from e
        nrows = 0
        parsed = []
        for v in raw:
            try:
                col = np.asarray(v, dtype=float).ravel()
            except Exception:                  # metadata / non-numeric -> placeholder
                col = None
            parsed.append(col)
            if col is not None:
                nrows = max(nrows, col.size)
        cols = []
        for col in parsed:
            if col is None or col.size != nrows:
                cols.append(np.full(nrows, np.nan))
            else:
                cols.append(col)
        arr = np.column_stack(cols) if cols else np.empty((0, 0))
    else:
        raise ValueError(f"Unsupported input file type: {ext}")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if len(names) != arr.shape[1]:             # keep names aligned even if a format lied
        names = (list(names) + [""] * arr.shape[1])[:arr.shape[1]]
    return arr, names


def probe_data_columns(settings: Settings, file_path: str):
    """Cheaply determine a file's column count for listing valid data files — reads only
    the header/one row (a full load happens only when a file is actually selected). Returns
    the column count, or ``None`` if the file is not a loadable data file."""
    import os

    import pandas as pd  # noqa: PLC0415

    from respmech.core.io.loaders import _read_table  # noqa: PLC0415 — encoding-tolerant reader

    ext = os.path.splitext(file_path)[1].lower()
    fmt = settings.input.format
    try:
        if ext == ".csv":
            dec = getattr(fmt, "decimal", ".") or "."
            return len(_read_table(file_path, sep=";" if dec == "," else ",", decimal=dec, nrows=1).columns)
        if ext == ".txt":
            dec = getattr(fmt, "decimal", ".") or "."
            return len(_read_table(file_path, sep="\t", decimal=dec, nrows=1).columns)
        if ext in (".xlsx",):
            return len(pd.read_excel(file_path, nrows=1).columns)
        if ext == ".mat":                       # no cheap header probe for MATLAB
            matrix, _names = load_raw_matrix(settings, file_path)
            return int(matrix.shape[1])
    except Exception:
        return None
    return None


def peek_columns(settings: Settings, file_path: str):
    """Cheap column-count probe for the manifest scanner (``ui/manifest.py``): an 8 KB
    byte-peek of the first line for .csv/.txt — measured at 0.06 ms/file against
    ``probe_data_columns``'s 46 ms/file for the same files, with the same result — so the
    format read-out and the QC strip can re-probe a whole folder on every input edit
    without the multi-second cost a full pandas read would add for a hundred recordings.

    Delegates straight to ``probe_data_columns`` for .xlsx/.mat, where a byte-peek is
    meaningless (an .xlsx is a zip archive of XML, not delimited text — sniffing its raw
    bytes previously misreported it as "1 columns, whitespace-separated"). Honours the same
    delimiter selection as ``probe_data_columns``/``load_raw_matrix`` (';' for a
    comma-decimal CSV; tab for .txt), so the two can never disagree on the same file.
    Returns the column count, or ``None`` if the file could not be read.

    KNOWN LIMITATION: only the FIRST line is inspected. A file whose header parses fine but
    whose data rows are ragged (a short row further down, or a quoted field containing the
    delimiter that a naive ``str.split`` miscounts) is not distinguishable from a well-formed
    file here — it still surfaces as a run-time ``DataValidationError`` when the batch
    actually runs, exactly as before this scanner existed. Accepted for the same reason the
    scan is a byte-peek at all: a row-by-row validation would cost close to what
    ``probe_data_columns`` already costs, defeating the point."""
    import os

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in (".csv", ".txt"):
        return probe_data_columns(settings, file_path)
    try:
        with open(file_path, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return None
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = head.decode("utf-16", errors="replace")
    else:
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = head.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    line = text.splitlines()[0].strip() if text else ""
    if not line:
        return None
    fmt = settings.input.format
    dec = getattr(fmt, "decimal", ".") or "."
    sep = "\t" if ext == ".txt" else (";" if dec == "," else ",")
    return len(line.split(sep))


def peek_header_warning(settings: Settings, file_path: str):
    """Cheap two-line consistency probe for the manifest scanner (ticket D01): flags a file
    whose FIRST non-blank line looks like it belongs to an instrument export's preamble
    rather than real channel data — either because that line's own field count is under 3
    (a real multi-channel recording needs at least a time column plus two signals; a header
    line like LabChart's 'Interval=<TAB>0.001 s' has exactly two), or because the first two
    non-blank lines disagree on field count. Delimited-text files only (.csv/.txt); returns
    ``None`` for any other extension, an unreadable head, a head this function's fixed
    encoding chain cannot decode (see below — narrower than it sounds), or a first line
    that is itself empty — the same "nothing to say" cases ``peek_columns`` already
    returns ``None`` for. Never raises.

    Deliberately its OWN small decode block rather than sharing ``peek_columns``'s: the two
    probes answer different questions (this one wants up to two non-blank lines,
    ``peek_columns`` wants exactly line zero, blank or not) and folding them into one
    shared helper risked silently changing ``peek_columns``'s existing, already-tested
    blank-first-line behaviour for the sake of a probe that postdates it — a handful of
    duplicated lines is the safer trade. In practice the "undecodable head" case above
    never actually happens for a non-UTF-16 file: ``latin-1`` maps every byte, so the
    decode loop always succeeds by the time it gets there — the guard exists for the same
    defensive reason ``peek_columns``'s does, not because it is expected to fire.

    ONLY sniffs the first 8 KB, same as ``peek_columns``: a data-row mismatch further into
    the file (a short row, a quoted delimiter) is not caught here, only the specific,
    common case of a header/preamble block ahead of the real data. That surfaces later as a
    run-time ``DataValidationError``, exactly as before this probe existed. When the file is
    at least 8 KB and the buffer's last line has no trailing newline, that line may be
    mid-row rather than genuinely short — dropped before comparison (self-review finding: a
    500-column recording's own row can exceed 8 KB, and a truncated SECOND row read as
    "fewer fields" would otherwise manufacture a false disagreement on a perfectly uniform,
    clean file)."""
    import os

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in (".csv", ".txt"):
        return None
    try:
        with open(file_path, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return None
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = head.decode("utf-16", errors="replace")
    else:
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = head.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    if not text:
        return None
    all_lines = text.splitlines()
    if len(head) >= 8192 and not text.endswith(("\n", "\r")):
        all_lines = all_lines[:-1]              # drop the possibly mid-row trailing line
    first_line = all_lines[0].strip() if all_lines else ""
    if not first_line:
        return None
    lines = [ln.strip() for ln in all_lines if ln.strip()][:2]
    fmt = settings.input.format
    dec = getattr(fmt, "decimal", ".") or "."
    sep = "\t" if ext == ".txt" else (";" if dec == "," else ",")
    counts = [len(ln.split(sep)) for ln in lines]
    if counts[0] < 3:
        plural = "" if counts[0] == 1 else "s"
        return (f"the first row has only {counts[0]} field{plural} — too few to be "
               "channel data")
    if len(counts) > 1 and counts[1] != counts[0]:
        return (f"the first two rows disagree on the number of fields "
               f"({counts[0]} vs {counts[1]})")
    return None


def probe_sampling_frequency(settings: Settings, file_path: str):
    """Cheap per-file sampling-frequency probe for the manifest scanner: read only column 0
    (the time axis), capped at 5000 rows — measured at 4.3 ms and a correct 2000 Hz on a
    60,000-row file. NEVER uses ``load_raw_matrix``, which reads every column of the whole
    file (21.7 ms/file for a single column alone).

    Returns ``None`` for .mat (no comparable cheap column-0 read) or .xlsx, or when the
    column does not look like regular time-in-seconds — see :func:`detect_sampling_frequency`,
    which this delegates the actual inference to. The 5000-row cap only bounds the COST for a
    delimited-text engine (pandas' C CSV parser stops reading once ``nrows`` is satisfied);
    openpyxl parses an Excel sheet before pandas applies that cap, so ``nrows=5000`` does
    nothing to bound an .xlsx read's cost — measured at 37-306 ms/file depending on the
    sheet's true row count (NOT capped at 5000, up to 30s+ for a hundred-file batch), against
    this function's ONE synchronous call per input edit. Ruled out rather than shipped with
    that cost: this only skips the (secondary) frequency-mismatch caution for Excel input,
    never the (primary) column-count/outlier detection, which stays cheap via
    ``probe_data_columns``'s single-row read."""
    import os

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".mat", ".xls", ".xlsx"):
        return None
    import pandas as pd  # noqa: PLC0415
    from respmech.core.io.loaders import _read_table  # noqa: PLC0415
    fmt = settings.input.format
    dec = getattr(fmt, "decimal", ".") or "."
    try:
        if ext == ".csv":
            df = _read_table(file_path, sep=";" if dec == "," else ",", decimal=dec,
                             usecols=[0], nrows=5000)
        elif ext == ".txt":
            df = _read_table(file_path, sep="\t", decimal=dec, usecols=[0], nrows=5000)
        else:
            return None
    except Exception:                       # noqa: BLE001 — best-effort, never blocks the scan
        return None
    if df.shape[1] == 0:
        return None
    col = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    return detect_sampling_frequency(col)


def detect_decimal(file_path: str, fallback: str = ".") -> str:
    """Best-effort decimal-separator detection for a .csv or tab-delimited .txt file.
    Defaults to '.' and only switches to ',' when '.' genuinely fails to parse the numbers
    (a comma-decimal European export reads mostly as NaN under '.') AND ',' is clearly
    better — so a US file with occasional comma-THOUSANDS integers ('1,024'), which '.'
    still parses fine on most cells, is not mis-detected as comma-decimal. Returns
    ``fallback`` on doubt (see the caller's own fallback contract, e.g.
    ``ui.screens.settings_screen.SettingsScreen._detect_decimal``, which passes the
    CURRENT setting so a doubtful file never overwrites an explicit or loaded-analysis
    choice).

    Ticket D03: for a .csv, each decimal candidate is read with the DELIMITER it
    implies — ';' for a comma decimal, ',' for a point decimal, the same pairing the
    loader itself uses (``core.io.loaders._read_table`` / this module's
    ``load_raw_matrix``/``probe_data_columns``) — so the two interpretations are compared
    on equal footing instead of both parsed with the SAME separator. A semicolon-delimited,
    comma-decimal export used to fail both candidates identically that way (every field
    read as one mostly-NaN column regardless of which decimal was tried), so the function
    always fell back and could never actually detect a European CSV. .txt is unchanged:
    always tab-delimited for both candidates."""
    import os

    import pandas as pd  # noqa: PLC0415

    ext = os.path.splitext(file_path)[1].lower()
    frac = {}
    for dec in (".", ","):
        sep = "\t" if ext == ".txt" else (";" if dec == "," else ",")
        try:
            df = pd.read_csv(file_path, sep=sep, decimal=dec, nrows=300)
            num = df.apply(pd.to_numeric, errors="coerce")
            total = int(num.size) or 1
            frac[dec] = int(num.notna().to_numpy().sum()) / total
        except Exception:
            frac[dec] = 0.0
    if frac.get(".", 0.0) >= 0.9:                # '.' parses almost everything -> trust it
        return "."
    if frac.get(",", 0.0) > frac.get(".", 0.0) + 0.2 and not _commas_look_like_thousands(file_path):
        return ","                              # ',' clearly better and not a thousands separator
    return fallback


def _commas_look_like_thousands(file_path: str) -> bool:
    """True when every comma in the sampled numbers is followed by exactly three digits
    ('1,024') — the thousands-separator pattern, which must NOT be read as a decimal comma.
    A real comma-decimal export has varying trailing-digit counts ('1,5', '12,34')."""
    import re  # noqa: PLC0415

    try:
        with open(file_path, "r", encoding="latin-1") as fh:  # latin-1 maps every byte, never raises
            head = fh.read(20000)
    except Exception:
        return False
    groups = re.findall(r"\d,(\d+)", head)
    return len(groups) >= 3 and all(len(g) == 3 for g in groups)


def detect_sampling_frequency(time_column):
    """Infer the sampling frequency (Hz) from a time column given in SECONDS: 1 / median(dt),
    but only when the column looks like real, regularly-sampled time — monotonic, positive,
    with a uniform, non-integer step and a physiologically plausible rate. Returns an int or
    ``None`` when it does not look like seconds (e.g. a sample-index column, or irregular
    timestamps), so the caller can fall back to the user-entered value."""
    t = np.asarray(time_column, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 20:
        return None
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size < 10:
        return None
    med = float(np.median(dt))
    if med <= 0:
        return None
    if med >= 1.0 and abs(med - round(med)) < 1e-6:     # integer steps -> a sample index, not seconds
        return None
    if float(np.std(dt)) > 0.25 * med:                  # irregular spacing -> not a clean sample clock
        return None
    fs = 1.0 / med
    if not (10.0 <= fs <= 200000.0):                    # implausible -> don't trust it
        return None
    return int(round(fs))


def stage_noise_fidelity(settings: Settings, cancel_check=None) -> dict:
    """Build the shared noise profile for the WHOLE test and measure the EMG fidelity
    frontier + the auto-selected suppression strength — the per-test noise summary the
    Preview screen shows, decoupled from the full mechanics test run. Pure compute,
    Qt-free, no disk writes. Returns the noise ``report`` dict (``frontier`` /
    ``prop_decrease`` / ``fidelity_target`` / ``channels``)."""
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core.pipeline import _build_noise_set, match_input_files
    from respmech.core import compute
    from respmech.core.io.loaders import DataValidationError

    s = to_legacy_ns(settings)
    # SAME matcher as run_batch: the shared noise profile must be built from exactly the
    # file set the batch will process, or the frontier the user tunes against diverges
    # from the real run — and between macOS and Windows. See match_input_files.
    allfiles = match_input_files(s.input.inputfolder, s.input.files)
    if not allfiles:
        raise FileNotFoundError(f"No input files found for '{s.input.files}'.")
    from respmech.ui.screens import _preview_cache as _pc
    ref = settings.processing.emg.noise.reference_file
    ref_path = os.path.join(s.input.inputfolder, ref) if ref else None

    def _compute():
        clip = _cached_reference_clip(settings, s, cancel_check=cancel_check)  # shared clip build
        # NB no `stft=` override here, unlike run_batch (pipeline: `_coupled_stft(...)` when a
        # pre-analysis resample is on, which keeps the gate's TIME window fixed at the new
        # rate). Correct today ONLY because the preview never resamples — `samplingfrequency_in`
        # is set inside run_batch alone, so to_legacy_ns() here never carries it. Reviving
        # resampling must thread the same override through this call, or the frontier the user
        # tunes against silently stops matching the run.
        return _build_noise_set(settings, s, allfiles, clip=clip, cancel_check=cancel_check)[1]

    key = _pc.noise_report_key(settings, ref_path, allfiles) if ref_path else None
    try:
        # Copy on return: a shared cache entry, and _apply_noise_report reads it while the
        # renderers hold on to the frontier lists (same contract as _load_and_condition).
        import copy as _copy                                # noqa: PLC0415
        report = _copy.deepcopy(
            _pc.cached(_pc.NOISE_REPORT, key, _compute))    # test-wide report; keyed on all-file tokens
        # D09: the reference clip's frame count travels WITH the report, so the
        # fidelity panel can carry the core's stability warning visibly (from_clip
        # says it through warnings.warn, which reaches a stderr nobody sees). The
        # clip build is the same shared, cached one _compute used, so this costs a
        # cache hit — and it is recomputed here, ON the copy, rather than stored in
        # the cache entry, so older cached reports simply lack the key and the
        # caption hides itself instead of showing a stale count.
        try:
            from respmech.ui.stft_frames import stft_frame_count   # noqa: PLC0415
            n = settings.processing.emg.noise
            clip = _cached_reference_clip(settings, s, cancel_check=cancel_check)
            report["noise_clip_frames"] = stft_frame_count(
                len(clip), n.win_length, n.hop_length)
        except Exception:                                   # noqa: BLE001
            pass          # the report itself is intact; the caption just stays hidden
        return report
    except (compute.TrimError, DataValidationError, FileNotFoundError, ImportError) as e:
        # A user-fixable precondition on the reference/input files — most often a misassigned or
        # inverted flow channel, so the reference has no segmentable breaths for the quiet-
        # expiration clip. Surface it as a clean, single-line, actionable message (rendered by
        # _on_noise_result -> _FileRunError) instead of a raw traceback. Genuine bugs
        # (IndexError / KeyError / numeric) are NOT caught here and still propagate.
        name = os.path.basename(ref_path) if ref_path else "the rest reference"
        return {"error": (f"Could not build the noise profile from '{name}' or the input files: "
                          f"{e} Check the flow channel and inverse-flow assignment, or turn off "
                          f"'use expiration' and set explicit reference intervals.")}


class EmgAllChannelsWorker(QObject):
    """Off-thread staging of all EMG channels for the result view."""
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: Settings, file_path: str):
        super().__init__()
        self._settings = settings
        self._path = file_path
        self._cancelled = False

    @Slot()
    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        from respmech.core._cancel import Cancelled
        if self._cancelled:
            self.finished.emit(None)
            return
        try:
            self.finished.emit(stage_emg_all_channels(
                self._settings, self._path, cancel_check=lambda: self._cancelled))
        except Cancelled:                   # superseded mid-compute -> aborted cleanly
            self.finished.emit(None)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
            self.finished.emit(None)


# --------------------------------------------------------------------------- #
# Mechanics channel preview staging (Preview & tuning screen). Pure compute
# (Qt-free, no disk writes): load -> trim -> zero/drift-correct -> breath split,
# returning plot-ready arrays. The synchronous _preview() and the off-thread
# reactive job both call this, so they render identically.
# --------------------------------------------------------------------------- #
def stage_mechanics_preview(settings: Settings, file_path: str) -> dict:
    """Stage a file's mechanics channels for the preview: load, trim to the
    analysis window, zero + optional drift-correct the volume, and split into
    breaths. Returns plot-ready arrays plus breath spans (with 1-based numbers
    and the ignored flag). Reuses the core exactly as ``run_batch`` does; never
    writes to disk and never touches Qt."""
    import os

    from respmech.core import compute
    from respmech.core.io.loaders import load
    from respmech.core._legacy_ns import to_legacy_ns

    s = to_legacy_ns(settings)
    flow, volume, poes, pgas, pdi, ent, emg = load(file_path, s)
    fs = s.input.format.samplingfrequency
    tc = np.arange(len(flow)) / fs
    name = os.path.basename(file_path)

    def _emg2d(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return a[:, None] if a.size else a.reshape(0, 0)
        return a

    try:
        (tcT, flowT, volT, poesT, pgasT, pdiT, _e, startix, endix) = compute.trim(
            tc, flow, volume, poes, pgas, pdi,
            np.array(emg) if len(emg) else np.array([]), s)
    except compute.TrimError as e:
        # Breath detection needs a trimmable flow signal; when it fails (e.g. wrong flow
        # channel / inverseflow), still return the RAW channels so the preview can show
        # the signal and the user can fix it — instead of a blank 'failed' error card.
        series = {"flow": np.asarray(flow, float), "volume": np.asarray(volume, float),
                  "poes": np.asarray(poes, float), "pgas": np.asarray(pgas, float),
                  "pdi": np.asarray(pdi, float)}
        return {
            "name": name, "fs": fs, "t": tc, "series": series, "spans": [],
            "label_y": float(np.nanmax(series["flow"])) if series["flow"].size else 0.0,
            "startix": 0, "endix": len(flow), "nbreaths": 0, "emg": _emg2d(emg),
            "emg_flow": np.asarray(flow, dtype=float),   # UNTRIMMED (aligns with the untrimmed 'emg')
            "trim_error": str(e),
            "trend_error": None,
            "vol_drift": None,
        }

    voldrift = (compute.correctdrift(compute.zero(volT), s)
                if s.processing.mechanics.correctvolumedrift else compute.zero(volT))
    volc, trend_error = voldrift, None
    if s.processing.mechanics.correctvolumetrend:      # match the batch pipeline (audit #11)
        try:
            volc = compute.correcttrend(voldrift, s)
        except compute.VolumeTrendError as e:
            # A precondition failure, not a bug: keep the DRIFT-corrected volume so the
            # channels stay readable and the thresholds can be tuned right here. Same soft
            # landing TrimError gets above — but the batch still refuses the file, so the
            # status line has to say so.
            trend_error = str(e)
    breaths = compute.separateintobreaths(
        s.processing.mechanics.separateby, name, tcT, flowT,
        volc, poesT, pgasT, pdiT, [], [], s)

    t = np.arange(len(flowT)) / fs
    series = {"flow": flowT, "volume": volc, "poes": poesT, "pgas": pgasT, "pdi": pdiT}
    spans, cum = [], 0
    for bno, b in breaths.items():
        length = len(np.atleast_1d(b["poes"]))
        spans.append((bno, cum / fs, (cum + length) / fs, bool(b["ignored"])))
        cum += length
    label_y = float(np.nanmax(series["flow"])) if len(series["flow"]) else 0.0
    return {
        "name": name, "fs": fs, "t": t, "series": series,
        "spans": spans, "label_y": label_y, "startix": startix, "endix": endix,
        "nbreaths": len(breaths), "emg": _emg2d(emg),
        "emg_flow": np.asarray(flow, dtype=float),   # UNTRIMMED (aligns with the untrimmed 'emg')
        "trim_error": None,
        "trend_error": trend_error,
        # drift-corrected volume: what the trend detector actually sees, so the advanced
        # dialog can count anchors for THIS file live while the thresholds are edited.
        "vol_drift": np.asarray(voldrift, float),
    }


class FnWorker(QObject):
    """Generic off-thread wrapper: run ``fn(*args, **kwargs)`` and emit the
    result. Same ``moveToThread`` seam as the other workers (never touches Qt);
    used for the mechanics-preview staging. ``finished`` carries the return value
    (or ``None`` on error)."""
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._cancelled = False

    @Slot()
    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        import inspect
        from respmech.core._cancel import Cancelled
        if self._cancelled:
            self.finished.emit(None)
            return
        kwargs = dict(self._kwargs)
        # inject a live cancel checker only for targets that accept one (e.g. the noise
        # job); the mechanics staging has no cancel_check param and is left untouched
        try:
            if "cancel_check" in inspect.signature(self._fn).parameters:
                kwargs.setdefault("cancel_check", lambda: self._cancelled)
        except (TypeError, ValueError):     # builtins / lambdas without a signature
            pass
        try:
            self.finished.emit(self._fn(*self._args, **kwargs))
        except Cancelled:                   # superseded mid-compute -> aborted cleanly
            self.finished.emit(None)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
            self.finished.emit(None)

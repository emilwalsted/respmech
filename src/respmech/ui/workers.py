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

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from respmech.core.pipeline import run_batch
from respmech.core.io.writers import write_batch
from respmech.core.settings import Settings


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
        try:
            result = run_batch(self._settings, progress=self._on_event,
                               cancel_check=lambda: self._cancel,
                               only_files=self._only)
            if self._write and not self._cancel:
                write_batch(result, self._settings, self._settings.output.folder)
            # Final result carries every file's table for the results view.
            self.finished.emit(None if self._cancel else result)
        except Exception as e:  # fatal (e.g. no files, invalid settings)
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit(None)


# --------------------------------------------------------------------------- #
# EMG-conditioning preview (Preview & tuning screen). Stages ONE EMG channel
# through the exact core conditioning the batch would apply, so the tuning view
# matches the real run. Pure compute here (Qt-free, no disk writes) so it is both
# unit-testable headless and safe to run on a QThread.
# --------------------------------------------------------------------------- #
def stage_emg_channel(settings: Settings, file_path: str, channel_index: int) -> dict:
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

    from respmech.core.io.loaders import load
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core import emg as emglib
    from respmech.core import noise as noiselib
    from respmech.core.pipeline import _reference_noise_clip

    s = to_legacy_ns(settings)
    fs = int(s.input.format.samplingfrequency)
    flow, vol, poes, pgas, pdi, ent, emg = load(file_path, s)
    emg = np.asarray(emg, dtype=float)
    if emg.ndim == 1:
        emg = emg[:, None]
    nch = emg.shape[1]
    if nch == 0:
        raise ValueError("No EMG channels are configured for this file.")
    ch = max(0, min(int(channel_index), nch - 1))
    raw = emg[:, ch].astype(float)

    # -- stage 2: ECG removal (test-level detect params, applied to all channels) --
    ecg = raw
    try:
        detect = max(0, min(int(s.processing.emg.column_detect), nch - 1))
        proc, _win, _peaks = emglib.remove_ecg(
            emg, emg[:, detect], samplingfrequency=fs,
            ecgminheight=s.processing.emg.minheight,
            ecgmindistance=s.processing.emg.mindistance,
            ecgminwidth=s.processing.emg.minwidth,
            windowsize=s.processing.emg.windowsize)
        proc = np.asarray(proc, dtype=float)
        if proc.ndim == 1:
            proc = proc[:, None]
        ecg = proc[:, ch].astype(float)
    except Exception:                                  # noqa: BLE001 — preview stays useful
        ecg = raw

    # -- stage 3: shared-profile spectral noise reduction on this channel --
    cfg = settings.processing.emg.noise
    noise_out = ecg
    noise_applied = False
    if cfg.reference_file:
        try:
            clip = np.asarray(_reference_noise_clip(settings, s), dtype=float)
            if clip.ndim == 1:
                clip = clip[:, None]
            prof = noiselib.NoiseProfile.from_clip(
                clip[:, min(ch, clip.shape[1] - 1)], fs,
                n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
                n_std_thresh=cfg.n_std_thresh, n_grad_freq=cfg.n_grad_freq,
                n_grad_time=cfg.n_grad_time)
            noise_out = np.asarray(prof.apply(ecg, cfg.prop_decrease), dtype=float)
            noise_applied = True
        except Exception:                              # noqa: BLE001 — e.g. clip too short
            noise_out = ecg
            noise_applied = False

    t = np.arange(len(raw), dtype=float) / fs

    def _psd(x):
        x = np.asarray(x, dtype=float)
        f, p = welch(x, fs, nperseg=int(min(1024, len(x))) or 1)
        return f, p

    freqs, psd_raw = _psd(raw)
    _, psd_ecg = _psd(ecg)
    _, psd_noise = _psd(noise_out)
    return {
        "channel": ch, "fs": fs, "t": t,
        "raw": raw, "ecg": ecg, "noise": noise_out,
        "freqs": freqs, "psd_raw": psd_raw, "psd_ecg": psd_ecg, "psd_noise": psd_noise,
        "band": noiselib.EMG_BAND, "noise_applied": noise_applied,
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

    @Slot()
    def run(self):
        try:
            data = stage_emg_channel(self._settings, self._path, self._ch)
            self.finished.emit(data)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit(None)


def stage_emg_all_channels(settings: Settings, file_path: str) -> dict:
    """Stage EVERY configured EMG channel of ``file_path`` (raw -> conditioned = ECG
    removal + shared-profile noise reduction) for the "result" view where the user
    picks which channels to see. ECG removal runs once for all channels; the noise
    profile is built per channel from the shared reference clip. Pure compute, Qt-free,
    no disk writes."""
    from respmech.core.io.loaders import load
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core import emg as emglib
    from respmech.core import noise as noiselib
    from respmech.core.pipeline import _reference_noise_clip

    s = to_legacy_ns(settings)
    fs = int(s.input.format.samplingfrequency)
    cols = list(settings.input.channels.emg)
    flow, vol, poes, pgas, pdi, ent, emg = load(file_path, s)
    emg = np.asarray(emg, dtype=float)
    if emg.ndim == 1:
        emg = emg[:, None]
    nch = emg.shape[1]
    if nch == 0:
        raise ValueError("No EMG channels are configured for this file.")

    conditioned = np.array(emg, dtype=float)
    if settings.processing.emg.remove_ecg:
        try:
            detect = max(0, min(int(s.processing.emg.column_detect), nch - 1))
            proc, _w, _p = emglib.remove_ecg(
                np.array(emg), emg[:, detect], samplingfrequency=fs,
                ecgminheight=s.processing.emg.minheight,
                ecgmindistance=s.processing.emg.mindistance,
                ecgminwidth=s.processing.emg.minwidth,
                windowsize=s.processing.emg.windowsize)
            conditioned = np.asarray(proc, dtype=float)
            if conditioned.ndim == 1:
                conditioned = conditioned[:, None]
        except Exception:                              # noqa: BLE001
            conditioned = np.array(emg, dtype=float)

    noise_applied = False
    cfg = settings.processing.emg.noise
    if cfg.reference_file:
        try:
            clip = np.asarray(_reference_noise_clip(settings, s), dtype=float)
            if clip.ndim == 1:
                clip = clip[:, None]
            for i in range(nch):
                prof = noiselib.NoiseProfile.from_clip(
                    clip[:, min(i, clip.shape[1] - 1)], fs,
                    n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length,
                    n_std_thresh=cfg.n_std_thresh, n_grad_freq=cfg.n_grad_freq,
                    n_grad_time=cfg.n_grad_time)
                conditioned[:, i] = prof.apply(conditioned[:, i], cfg.prop_decrease)
            noise_applied = True
        except Exception:                              # noqa: BLE001
            noise_applied = False

    t = np.arange(emg.shape[0], dtype=float) / fs
    return {
        "t": t, "fs": fs, "cols": cols,
        "raw": [emg[:, i].astype(float) for i in range(nch)],
        "conditioned": [conditioned[:, i].astype(float) for i in range(nch)],
        "noise_applied": noise_applied,
    }


class EmgAllChannelsWorker(QObject):
    """Off-thread staging of all EMG channels for the result view."""
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: Settings, file_path: str):
        super().__init__()
        self._settings = settings
        self._path = file_path

    @Slot()
    def run(self):
        try:
            self.finished.emit(stage_emg_all_channels(self._settings, self._path))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit(None)

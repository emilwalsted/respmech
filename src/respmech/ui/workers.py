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

import traceback

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
        # Analysis and writing fail differently: a failed run has no result, but a
        # failed WRITE still has a fully computed result worth showing.
        try:
            result = run_batch(self._settings, progress=self._on_event,
                               cancel_check=lambda: self._cancel,
                               only_files=self._only)
        except Exception:  # fatal (e.g. no files, invalid settings)
            self.failed.emit(traceback.format_exc())
            self.finished.emit(None)
            return
        # Latch the cancel decision ONCE, when the analysis phase ends, so a Cancel
        # click that lands during writing can't be reported as "no output written".
        cancelled = self._cancel
        if self._write and not cancelled:
            try:
                write_batch(result, self._settings, self._settings.output.folder)
            except Exception:
                self.failed.emit(
                    "Analysis completed, but writing the output failed. Some files may be "
                    f"partially written in:\n  {self._settings.output.folder}\n\n"
                    + traceback.format_exc())
                self.finished.emit(result)      # still deliver the computed result
                return
        # Final result carries every file's table for the results view.
        self.finished.emit(None if cancelled else result)


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
    cols = list(settings.input.channels.emg)
    col = cols[ch] if ch < len(cols) else ch + 1

    # -- stage 2: ECG removal (honours remove_ecg; test-level detect params) --
    ecg = raw
    ecg_applied = False
    ecg_error = None
    if settings.processing.emg.remove_ecg:
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
            ecg_applied = True
        except Exception:                              # noqa: BLE001 — preview stays useful
            ecg = raw
            ecg_error = traceback.format_exc()

    # -- stage 3: shared-profile spectral noise reduction on this channel --
    cfg = settings.processing.emg.noise
    noise_out = ecg
    noise_applied = False
    noise_error = None
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
        "raw": raw, "ecg": ecg, "noise": noise_out,
        "freqs": freqs, "psd_raw": psd_raw, "psd_ecg": psd_ecg, "psd_noise": psd_noise,
        "band": noiselib.EMG_BAND, "noise_applied": noise_applied,
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
        if self._cancelled:                 # superseded before we got the thread
            self.finished.emit(None)
            return
        try:
            data = stage_emg_channel(self._settings, self._path, self._ch)
            self.finished.emit(data)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
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
    ecg_applied = False
    ecg_error = None
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
            ecg_applied = True
        except Exception:                              # noqa: BLE001
            conditioned = np.array(emg, dtype=float)
            ecg_error = traceback.format_exc()

    noise_applied = False
    noise_error = None
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
            noise_error = traceback.format_exc()

    t = np.arange(emg.shape[0], dtype=float) / fs
    return {
        "t": t, "fs": fs, "cols": cols,
        "raw": [emg[:, i].astype(float) for i in range(nch)],
        "conditioned": [conditioned[:, i].astype(float) for i in range(nch)],
        "noise_applied": noise_applied, "noise_error": noise_error,
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
    _flow, _vol, _poes, _pgas, _pdi, _ent, emg = load(file_path, s)
    emg = np.asarray(emg, dtype=float)
    if emg.ndim == 1:
        emg = emg[:, None]
    t = np.arange(emg.shape[0], dtype=float) / fs
    return {"t": t, "fs": fs, "cols": cols,
            "raw": [emg[:, i].astype(float) for i in range(emg.shape[1])]}


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

    ext = os.path.splitext(file_path)[1].lower()
    fmt = settings.input.format

    def _from_df(df):
        names = [str(c) for c in df.columns]
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return arr, names

    if ext in (".csv",):
        arr, names = _from_df(pd.read_csv(file_path))
    elif ext in (".txt",):
        dec = getattr(fmt, "decimal", ".") or "."
        arr, names = _from_df(pd.read_csv(file_path, sep="\t", decimal=dec))
    elif ext in (".xls", ".xlsx"):
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

    ext = os.path.splitext(file_path)[1].lower()
    fmt = settings.input.format
    try:
        if ext == ".csv":
            return len(pd.read_csv(file_path, nrows=1).columns)
        if ext == ".txt":
            dec = getattr(fmt, "decimal", ".") or "."
            return len(pd.read_csv(file_path, sep="\t", decimal=dec, nrows=1).columns)
        if ext in (".xls", ".xlsx"):
            return len(pd.read_excel(file_path, nrows=1).columns)
        if ext == ".mat":                       # no cheap header probe for MATLAB
            matrix, _names = load_raw_matrix(settings, file_path)
            return int(matrix.shape[1])
    except Exception:
        return None
    return None


def detect_decimal(file_path: str, fallback: str = ".") -> str:
    """Best-effort decimal-separator detection for a tab-delimited .txt file. Defaults to
    '.' and only switches to ',' when '.' genuinely fails to parse the numbers (a comma-
    decimal European export reads mostly as NaN under '.') AND ',' is clearly better — so a
    US file with occasional comma-THOUSANDS integers ('1,024'), which '.' still parses fine
    on most cells, is not mis-detected as comma-decimal. Returns ``fallback`` on doubt."""
    import pandas as pd  # noqa: PLC0415

    frac = {}
    for dec in (".", ","):
        try:
            df = pd.read_csv(file_path, sep="\t", decimal=dec, nrows=300)
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
        with open(file_path, "r", errors="ignore") as fh:
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


def stage_noise_fidelity(settings: Settings) -> dict:
    """Build the shared noise profile for the WHOLE test and measure the EMG fidelity
    frontier + the auto-selected suppression strength — the per-test noise summary the
    Preview screen shows, decoupled from the full mechanics test run. Pure compute,
    Qt-free, no disk writes. Returns the noise ``report`` dict (``frontier`` /
    ``prop_decrease`` / ``fidelity_target`` / ``channels``)."""
    import glob
    import os
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core.pipeline import _build_noise_set

    s = to_legacy_ns(settings)
    allfiles = sorted(glob.glob(os.path.join(s.input.inputfolder, s.input.files)))
    if not allfiles:
        raise FileNotFoundError(f"No input files found for '{s.input.files}'.")
    _noise_set, report = _build_noise_set(settings, s, allfiles)   # full test defines the profile
    return report


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
        if self._cancelled:
            self.finished.emit(None)
            return
        try:
            self.finished.emit(stage_emg_all_channels(self._settings, self._path))
        except Exception as e:  # noqa: BLE001
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
            "trim_error": str(e),
        }

    volc = (compute.correctdrift(compute.zero(volT), s)
            if s.processing.mechanics.correctvolumedrift else compute.zero(volT))
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
        "nbreaths": len(breaths), "emg": _emg2d(emg), "trim_error": None,
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
        if self._cancelled:
            self.finished.emit(None)
            return
        try:
            self.finished.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())
            self.finished.emit(None)

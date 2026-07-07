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

import glob
import ntpath
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
    error: Optional[str] = None


@dataclass
class BatchResult:
    files: dict = field(default_factory=dict)   # filename -> FileResult
    average_table: object = None                # concatenated average rows

    @property
    def ok_files(self):
        return {k: v for k, v in self.files.items() if v.error is None}

    @property
    def failed_files(self):
        return {k: v for k, v in self.files.items() if v.error is not None}


def _emit(cb: Optional[ProgressCallback], event: ProgressEvent):
    if cb is not None:
        cb(event)


def _process_emg(s, flowraw, emgcolumnsraw, startix, endix, filename):
    """ECG removal (on raw) -> trim -> noise reduction. Mirrors legacy analyse."""
    emgcols = np.array(emgcolumnsraw)
    ecgw = []
    if s.processing.emg.remove_ecg:
        emgcols_ecg, ecgw, _peaks = emglib.remove_ecg(
            emgcols, emgcols[:, s.processing.emg.column_detect],
            samplingfrequency=s.input.format.samplingfrequency,
            ecgminheight=s.processing.emg.minheight,
            ecgmindistance=s.processing.emg.mindistance,
            ecgminwidth=s.processing.emg.minwidth,
            windowsize=s.processing.emg.windowsize)
        emgcols = np.array(emgcols_ecg)
    emgcols = emgcols[startix:endix]

    if s.processing.emg.remove_noise:
        noiseprofile, noiseprofilepath = [], ""
        for nop in s.processing.emg.noise_profile:
            if nop[0] == filename:
                noiseprofilepath = nop[1]
                noiseprofile = nop[2]
                break
        if len(noiseprofile) == 0:
            raise ValueError(f"Noise profile interval not specified for file '{filename}'")
        npcolumns = emgcols
        if len(noiseprofilepath) > 0:
            _, _, _, _, _, _, npcolumns = load(noiseprofilepath, _NSWrap(s))
        nrcols = [emglib.reducenoise(np.array(emgcols[:, i]), noiseprofile,
                                     np.asarray(npcolumns)[:, i] if len(npcolumns) else [],
                                     s.input.format.samplingfrequency)
                  for i in range(emgcols.shape[1])]
        nr = np.array(nrcols).T
        nr = np.pad(nr, ((0, len(emgcols) - len(nr)), (0, 0)), 'constant')
        emgcols = nr
    return emgcols


class _NSWrap:
    """load() expects an object with .input/.processing attributes (legacy shape).
    The pipeline already holds the legacy-ns; wrap it so load() can be reused for a
    noise-profile source file."""
    def __init__(self, ns):
        self.input = ns.input
        self.processing = ns.processing
        self.output = ns.output


def run_batch(settings: Settings, progress: Optional[ProgressCallback] = None,
              cancel_check: Optional[Callable[[], bool]] = None,
              only_files: Optional[list] = None) -> BatchResult:
    """Process the batch. ``progress`` receives :class:`ProgressEvent`s. If
    ``cancel_check`` returns True (checked before each file), processing stops
    cooperatively. ``only_files`` (basenames) restricts the run — used for GUI test
    runs on a single file."""
    settings.validate()
    s = to_legacy_ns(settings)

    filepath = os.path.join(s.input.inputfolder, s.input.files)
    files = sorted(glob.glob(filepath))
    if only_files is not None:
        files = [f for f in files if ntpath.basename(f) in set(only_files)]
    if len(files) == 0:
        raise FileNotFoundError(f"No input files found for '{filepath}'")

    result = BatchResult()
    average_rows = []

    for fi in files:
        if cancel_check is not None and cancel_check():
            _emit(progress, ProgressEvent("finished", message="cancelled"))
            return result
        file = os.path.abspath(fi)
        filename = ntpath.basename(file)
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
            if len(emgcolumnsraw) > 0:
                _emit(progress, ProgressEvent("stage", file=filename, message="processing EMG"))
                emgcolumns = _process_emg(s, flowraw, emgcolumnsraw, startix, endix, filename)

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
                compute.calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, s)
                done += 1
                _emit(progress, ProgressEvent("breath", file=filename, breath=done, total_breaths=total))

            breaths_table, average_row = build_breath_table(filename, breaths, s)
            processed = None
            if s.output.data.saveprocesseddata:
                processed = build_processed_data(breaths, s)

            result.files[filename] = FileResult(
                file=filename, breaths_table=breaths_table, average_row=average_row,
                processed=processed, breaths=breaths)
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

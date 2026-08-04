"""Reactive-job bookkeeping shared by PreviewScreen's scheduler: the per-kind
panel/label tables, the settings-path -> job-kind mapping, and the _Job/
_FileRunError types. Split out of preview_screen.py (ticket A02); moved verbatim."""

from __future__ import annotations

import copy
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QObject, QSize, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui import plot_perf
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers
from respmech.ui import wheel as _wheel
from respmech.ui.flow_layout import (FlowLayout, cluster as _cluster,
                                     elide as _elide, install_flow as _install_flow)
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None


# reactive job kinds -> which panel overlays they own, and the spinner caption.
# 'batch' (the test run: table + Campbell) is NOT auto-run — only the button triggers
# it. 'noise' auto-computes the fidelity frontier + suppression, decoupled from batch.
# sub-tab titles — a leading chevron marks the intended flow: Mechanics › ECG reduction ›
# noise reduction. Kept as constants so the insert code and the tests share one source.
_TAB_MECH = "Mechanics"
_TAB_ECG = "› EMG – ECG reduction"
_TAB_NOISE = "› EMG – noise reduction"

_PANELS = {"mech": ["channels", "raw"], "batch": ["table", "campbell"],
           "ecg": ["ecg_capture", "ecg_stack"],
           "emg_all": ["result"], "emg_detail": ["detail", "detail_psd"], "noise": ["fidelity"]}
_SPIN_TEXT = {"mech": "Loading channels…", "batch": "Running test…",
              "ecg": "Removing ECG…",
              "emg_all": "Conditioning channels…", "emg_detail": "Staging detail…",
              "noise": "Measuring fidelity…"}
# human labels for status lines + panel error cards
_KIND_LABEL = {"mech": "Channel preview", "batch": "Test run", "ecg": "ECG reduction",
               "emg_all": "EMG result", "emg_detail": "EMG detail",
               "noise": "Noise fidelity"}
# the kinds that run automatically (on file select / settings change / Refresh). The
# test run ('batch') is now automatic too, but MECHANICS-ONLY (no ECG/EMG work).
_AUTO_KINDS = ("mech", "batch", "ecg", "emg_all", "emg_detail", "noise")
# the kinds whose result depends on the SELECTED file. 'noise' is deliberately absent: the
# fidelity/noise profile is test-wide (built from the reference file + the whole input set),
# so switching the previewed file must NOT blank or rebuild it. See _begin_file_switch.
_FILE_KINDS = ("mech", "batch", "ecg", "emg_all", "emg_detail")


def _kinds_for_settings_path(path):
    """Which auto kinds a changed Settings field (dotted path, as produced by
    ``dataclasses.asdict``) can affect, so a settings edit recomputes only the impacted
    panels. GOLDEN-SAFE: this only scopes the PREVIEW; the batch/CLI always use the real
    Settings, never this map. Erring WIDE is safe (over-recompute); erring narrow risks a
    stale panel — so any field NOT classified below falls through to ALL kinds. The
    exhaustive per-kind field lists were derived + adversarially verified against each
    stage_* function; the coarse buckets here stay on the safe (wide) side of them."""
    # output / diagnostics and the optional pre-resample never surface in any preview panel
    if path == "output" or path.startswith("output.") or path.startswith("processing.sampling"):
        return frozenset()
    # the EMG channel SET: mechanics shows the raw EMG traces and all EMG/noise panels use
    # them; the mechanics test run strips EMG, so it is unaffected
    if path == "input.channels.emg":
        return frozenset(("mech", "ecg", "emg_all", "emg_detail", "noise"))
    if path.startswith("processing.emg"):
        if path.startswith("processing.emg.robust_peak"):
            # Writes-only, draws-nothing — like output.* above. The cardiac-gated peak adds
            # columns to the workbook; no preview panel plots a per-breath maximum, so there
            # is nothing to redraw. Without this rule it would fall through to the EMG
            # conditioning case below and re-run ECG + both EMG panels + the noise frontier
            # on every tick of a checkbox that changes none of them.
            return frozenset()
        if path == "processing.emg.outlier_rms_sd_limit":
            return frozenset(("batch",))          # a batch-table outlier flag only
        if path in ("processing.emg.normalization", "processing.emg.save_sound",
                    "processing.emg.plot_yscale"):
            return frozenset(("emg_all", "emg_detail"))   # display/output-ish -> redraw EMG panels
        # EMG conditioning (remove_ecg / detect_channel / ecg_* / rms_window_s / remove_noise) +
        # noise.* params: the ECG-reduction tab + the EMG/noise panels (the mechanics test run
        # forces EMG + ECG + noise off, so it is unaffected)
        return frozenset(("ecg", "emg_all", "emg_detail", "noise"))
    # mechanics-only compute that feeds the test-run table/Campbell exclusively
    if (path.startswith("processing.wob") or path.startswith("processing.ptp")
            or path.startswith("processing.entropy") or path.startswith("processing.breath_counts")):
        return frozenset(("batch",))
    # breath segmentation feeds the mechanics panels AND — via the noise reference clip's
    # expiration segmentation and the auto_prop gather — the EMG conditioning + fidelity
    # panels (which build NoiseProfile.from_clip on a buffer-dependent clip). So a
    # segmentation edit must recompute all five, or the EMG traces go stale when auto_prop
    # is off (nothing else re-dispatches them). The ECG cache keeps the extra work cheap.
    if path.startswith("processing.segmentation"):
        return frozenset(_AUTO_KINDS)
    # volume drift/trend + explicit breath exclusion: the mechanics panels (+ noise, a
    # cache hit); these do NOT change the flow-based EMG reference masks, so the EMG panels
    # are left alone.
    if (path.startswith("processing.volume.correct") or path.startswith("processing.volume.trend")
            or path.startswith("processing.exclude_breaths")):
        return frozenset(("mech", "batch", "noise"))
    # channels core / format / volume inverse+integrate (all applied inside load()),
    # channels.entropy (validated in every load path), input.folder/files, and anything not
    # matched above -> recompute everything (safe default; never leaves a panel stale)
    return frozenset(_AUTO_KINDS)


def _changed_settings_paths(old, new):
    """Dotted paths of the leaves that differ between two Settings snapshots."""
    from dataclasses import asdict
    changed = set()

    def _walk(a, b, prefix):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                _walk(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        elif a != b:
            changed.add(prefix)

    _walk(asdict(old), asdict(new), "")
    return changed


@dataclass(eq=False)
class _Job:
    """A single in-flight reactive computation (one kind at a time is current).

    ``eq=False`` keeps identity-based equality/hashing so a job can live in the
    ``_draining`` set. ``error`` is per-job (never kind-keyed) so a superseded
    job's late failure cannot poison its fresh successor's result."""
    kind: str
    token: int
    thread: object
    worker: object
    error: object = None


# threads that would not stop within the shutdown budget are parked here (kept
# referenced forever) so CPython never GCs a still-running QThread and aborts.
_ORPHANED_THREADS = []

# Cap the number of worker QThreads running AT ONCE. A file select fans out up to 5 heavy,
# largely GIL-bound staging jobs; starting them all together on a 2-core machine thrashes the
# GIL + oversubscribes OpenBLAS, starving the GUI thread so queued `finished` signals (and the
# UI) stall for seconds — badly on Windows (coarse GIL hand-off + no offscreen message pump),
# where it tips reactive tests past their timeout. Running at most 2 keeps the GUI responsive;
# the rest queue and start as slots free. See _pump_pool / _launch.
_MAX_ACTIVE = 2


class _FileRunError(RuntimeError):
    """A per-file analysis error (the core returned FileResult.error). Raised so
    _on_job_done labels it 'failed' — not a 'display error' — but still routes it
    to the copyable error card."""

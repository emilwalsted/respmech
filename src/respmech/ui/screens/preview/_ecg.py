"""PreviewScreen's ECG-reduction sub-tab. Split out of preview_screen.py
(ticket A02); moved verbatim."""

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
                                     install_flow as _install_flow)
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

from ._plot_helpers import _pen, _plot_pal

#: the ECG capture panel's base title; _on_ecg_result appends the live R-peak count/removal
#: state to this so the panel keeps naming what it draws AND carries a persistent verdict,
#: rather than one that only ever lived a few seconds in the shared status bar.
_ECG_CAPTURE_TITLE = "Raw capture channel — detected R-peaks (▼)"

#: the processed-stack panel's two base titles. Which one is current depends on
#: settings.processing.emg.remove_ecg (mirrored into data["ecg_applied"]), never a fixed
#: string: with removal OFF this panel shows the RAW channels (see stage_ecg_reduction's
#: docstring), and a title that still says "ECG-processed" over R-takker in full swing reads
#: as "the removal is broken" rather than "the removal is off".
_ECG_PROCESSED_TITLE_ON = "ECG-processed EMG channels"
_ECG_PROCESSED_TITLE_OFF = "EMG channels — removal is OFF (capture preview)"


class _EcgMixin:

    # -- EMG – ECG reduction tab -------------------------------------------
    def _build_ecg_tab(self):
        """Tune the ECG artefact removal: a top settings row (capture channel + params +
        Auto-suggest), the RAW capture channel with the detected R-peaks marked, and the
        ECG-processed channels below — all with the flow background + capture markers."""
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)

        def _cap(text, tip=""):
            la = QLabel(text); la.setProperty("status", "muted")
            if tip:
                la.setToolTip(tip)
            return la

        self.ecg_capture_channel = QComboBox()
        self.ecg_capture_channel.setToolTip("EMG channel used to detect the R-waves (heartbeats). "
                                            "Pick the channel with the clearest ECG and weakest EMG "
                                            "(often the middle channel).")
        self.remove_ecg = QCheckBox("Remove ECG")
        self.remove_ecg.setToolTip(_help_tip("processing.emg.remove_ecg",
                                   "Subtract an averaged ECG template from every EMG channel; off by default."))
        # Mirrors noise.auto_prop's "Auto" checkbox: analyse ONE reference file once and apply
        # the result to every file in the batch, instead of tuning the fields below by hand.
        # This is the same core.emg.suggest_ecg_settings the button below runs on the previewed
        # file — Auto-suggest stays a manual, per-file preview aid; this is its unsupervised,
        # whole-batch counterpart (core.pipeline._auto_detect_ecg_settings), now reachable
        # without leaving the GUI to hand-edit a settings.toml.
        self.ecg_auto_batch = QCheckBox("Auto (whole batch)")
        self.ecg_auto_batch.setToolTip(_help_tip(
            "processing.emg.ecg_auto_detect",
            "Detect the settings below ONCE from a reference file (the first matched file, "
            "unless processing.emg.ecg_reference_file names another) and apply them "
            "identically to every file in the batch. Requires Remove ECG. A file whose "
            "heart rate diverges from the reference can have real beats missed without "
            "the run failing — check run-report.txt after a run for a per-file warning."))
        self.ecg_auto_batch.toggled.connect(self._on_ecg_param_changed)
        self.ecg_min_height = QDoubleSpinBox()
        self.ecg_min_height.setRange(0.0, 1_000_000.0); self.ecg_min_height.setDecimals(6); self.ecg_min_height.setSingleStep(0.0001)
        self.ecg_min_height.setToolTip(_help_tip("processing.emg.ecg_min_height",
                                       "Minimum height of an R-wave peak (in signal units) on the capture channel."))
        self.ecg_min_distance = QDoubleSpinBox()
        self.ecg_min_distance.setRange(0.05, 2.0); self.ecg_min_distance.setSingleStep(0.05); self.ecg_min_distance.setDecimals(3); self.ecg_min_distance.setSuffix(" s")
        self.ecg_min_distance.setToolTip(_help_tip("processing.emg.ecg_min_distance_s",
                                         "Minimum time between heartbeats (the refractory gap)."))
        self.ecg_min_width = QDoubleSpinBox()
        self.ecg_min_width.setRange(0.0, 0.1); self.ecg_min_width.setSingleStep(0.001); self.ecg_min_width.setDecimals(4); self.ecg_min_width.setSuffix(" s")
        self.ecg_min_width.setToolTip(_help_tip("processing.emg.ecg_min_width_s",
                                      "Minimum width of an R-wave peak."))
        self.ecg_window = QDoubleSpinBox()
        self.ecg_window.setRange(0.05, 1.0); self.ecg_window.setSingleStep(0.05); self.ecg_window.setDecimals(3); self.ecg_window.setSuffix(" s")
        self.ecg_window.setToolTip(_help_tip("processing.emg.ecg_window_s",
                                   "Width of the ECG template averaged and subtracted around each beat (QRS-T)."))
        for wdg in (self.remove_ecg,):
            wdg.toggled.connect(self._on_ecg_param_changed)
        for wdg in (self.ecg_min_height, self.ecg_min_distance, self.ecg_min_width, self.ecg_window):
            wdg.valueChanged.connect(self._on_ecg_param_changed)
        self.ecg_capture_channel.currentIndexChanged.connect(self._on_ecg_param_changed)
        self.btn_ecg_autosuggest = QPushButton("Auto-suggest settings")
        self.btn_ecg_autosuggest.setToolTip("Analyse the raw EMG channels: pick the channel with the "
                                            "clearest ECG and propose settings that capture the R-peaks "
                                            "and only those.")
        self.btn_ecg_autosuggest.clicked.connect(self._on_ecg_autosuggest)

        self.ecg_opts = QFrame(); self.ecg_opts.setObjectName("ecgChip")
        self.ecg_opts.setStyleSheet("#ecgChip { border: 1px solid rgba(128, 128, 128, 0.30); border-radius: 8px; }")
        # A FlowLayout of caption+field clusters, not one QHBoxLayout: this chip is the widest
        # thing on Preview & QC, and as a single unbreakable row its minimum was the sum of all
        # five clusters — 975 px here, 1486 px on Windows' wider metrics, which put the whole
        # window past a 1280 px laptop screen (see flow_layout's module docstring).
        row = _install_flow(self.ecg_opts, h=10, v=4, margins=(11, 3, 11, 3))
        row.addLayout(_cluster(_cap("Capture channel"), self.ecg_capture_channel))
        row.addWidget(self.remove_ecg)
        row.addWidget(self.ecg_auto_batch)
        row.addLayout(_cluster(_cap("Min height", self.ecg_min_height.toolTip()),
                               self.ecg_min_height))
        row.addLayout(_cluster(_cap("Min gap", self.ecg_min_distance.toolTip()),
                               self.ecg_min_distance))
        # Min width and Window are NOT on the strip: a shape guard at 0.001 s and a
        # physiologically fixed template width are not what anyone reaches for while watching
        # the detected beats, and they crowded out the two that are. They keep their widgets
        # (Auto-suggest writes them, and from_state/to_state still round-trip them) — the
        # widgets simply live in the Advanced dialog's layout instead of here.
        self.btn_ecg_advanced = QPushButton("Advanced…")
        self.btn_ecg_advanced.setProperty("compact", True)
        self.btn_ecg_advanced.setToolTip("Detector shape guard and template width — rarely "
                                         "the right thing to change.")
        self.btn_ecg_advanced.clicked.connect(self._open_ecg_advanced)

        # Captured once, before AUTO_BATCH_HINT ever overwrites them: _update_actions restores
        # each widget's own help tooltip when "Auto (whole batch)" is off, rather than leaving
        # it blank (unlike NEEDS_ECG_HINT's widgets above, whose "off" tooltip is empty).
        self._ecg_auto_gated_base_tooltips = {
            w: w.toolTip() for w in (self.ecg_capture_channel, self.ecg_min_height,
                                      self.ecg_min_distance, self.btn_ecg_advanced,
                                      self.btn_ecg_autosuggest)}

        strip = FlowLayout(h=10, v=6)          # wraps rather than forcing the window wide
        strip.addWidget(self.ecg_opts)
        strip.addWidget(self.btn_ecg_advanced); strip.addWidget(self.btn_ecg_autosuggest)
        v.addLayout(strip)

        _bg = _plot_pal()["bg"]
        self.ecg_capture_plot = pg.PlotWidget(); self.ecg_capture_plot.setBackground(_bg)
        _theme.set_plot_floor(self.ecg_capture_plot)
        plot_perf.tune_widget(self.ecg_capture_plot)
        self.ecg_capture_plot.setLabel("bottom", "Time (s)")
        # Just the unit: this panel is the 1/4-height slot of the splitter below (stretch 1:3,
        # ~71px of viewbox at the default window), and a pyqtgraph left label is rotated, so
        # its LENGTH runs along that height. The panel title names the channel.
        self.ecg_capture_plot.setLabel("left", "a.u.")
        self.ecg_processed_plots = pg.GraphicsLayoutWidget(); self.ecg_processed_plots.setBackground(_bg)
        _theme.set_plot_floor(self.ecg_processed_plots)
        split = QSplitter(Qt.Vertical)
        # kept as an attribute (not just added to the splitter) so _on_ecg_result can keep
        # its title current with the detection result — see _ECG_CAPTURE_TITLE below.
        self._ecg_capture_box = self._titled(_ECG_CAPTURE_TITLE, self.ecg_capture_plot)
        split.addWidget(self._ecg_capture_box)
        # kept as an attribute too (like _ecg_capture_box above) so _on_ecg_result can keep
        # this title honest about whether removal is actually on — see _ECG_PROCESSED_TITLE_*.
        self._ecg_processed_box = self._titled(_ECG_PROCESSED_TITLE_ON, self.ecg_processed_plots)
        split.addWidget(self._ecg_processed_box)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 3)
        v.addWidget(split, 1)
        return w

    def _refresh_ecg_channels(self):
        """Populate the capture-channel combo; default it to the MIDDLE EMG channel (typically
        the weakest-EMG / clearest-ECG electrode) when the analysis has not configured ECG yet.

        detect_channel is an INDEX into input.channels.emg, not a column number, so
        re-assigning the EMG channels silently re-points it — and if the new list is shorter,
        points it past the end. The combo clamped for display only, so the screen showed one
        electrode while the model held an index the core would crash on
        (``emgcols[:, detect]``). The clamp is written back here, and said out loud: which
        electrode the heartbeats are read from is a scientific choice, not a detail."""
        cols = list(self.state.settings.input.channels.emg)
        e = self.state.settings.processing.emg
        if len(cols) > 1 and int(e.detect_channel) == 0 and not e.remove_ecg:
            e.detect_channel = len(cols) // 2         # preview-only seed (golden reads the model directly)
        if cols and not (0 <= int(e.detect_channel) < len(cols)):
            was = int(e.detect_channel)
            e.detect_channel = max(0, min(was, len(cols) - 1))
            self._set_status(
                f"ECG capture channel {was + 1} is beyond the {len(cols)} assigned EMG "
                f"channel{'s' if len(cols) != 1 else ''} — reset to EMG col "
                f"{cols[e.detect_channel]}. Check it is the electrode you want.")
        self.ecg_capture_channel.blockSignals(True)
        self.ecg_capture_channel.clear()
        for c in cols:
            self.ecg_capture_channel.addItem(f"EMG col {c}")
        if cols:
            self.ecg_capture_channel.setCurrentIndex(max(0, min(int(e.detect_channel), len(cols) - 1)))
        self.ecg_capture_channel.blockSignals(False)

    def _load_ecg_params(self):
        """Reflect the ECG settings into the tab's widgets without firing a recompute."""
        self._loading_ecg = True
        try:
            e = self.state.settings.processing.emg
            # An analysis file can carry ecg_auto_detect=True with remove_ecg=False.
            # Settings.validate rejects that pair, and the interactive path clears it (see
            # _on_ecg_param_changed) — but LOADING one did not. Measured on Emil's S07.toml,
            # 30-07-2026: the checkbox came up ticked AND disabled, every preview was gated
            # out with "Settings incomplete", and the mechanics test run died on
            # `SettingsError: processing.emg.ecg_auto_detect requires
            # processing.emg.remove_ecg to be enabled` behind a raw traceback. There was no
            # way to untick the box without re-enabling Remove ECG first.
            #
            # So repair it here, in the one funnel every ECG load goes through. Repairing
            # rather than refusing is the right way round for the GUI: remove_ecg is the
            # switch the user sees, and auto-detect is a sub-option of it, exactly like the
            # gated detail fields elsewhere on this strip. The CLI keeps raising, because a
            # hand-written config with that pair IS a mistake worth reporting rather than
            # silently altering.
            if e.ecg_auto_detect and not e.remove_ecg:
                e.ecg_auto_detect = False
                self._ecg_auto_repaired = True
            self.remove_ecg.setChecked(bool(e.remove_ecg))
            self.ecg_auto_batch.setChecked(bool(e.ecg_auto_detect))
            self.ecg_min_height.setValue(float(e.ecg_min_height))
            self.ecg_min_distance.setValue(float(e.ecg_min_distance_s))
            self.ecg_min_width.setValue(float(e.ecg_min_width_s))
            self.ecg_window.setValue(float(e.ecg_window_s))
            cols = list(self.state.settings.input.channels.emg)
            if cols:
                self.ecg_capture_channel.setCurrentIndex(max(0, min(int(e.detect_channel), len(cols) - 1)))
        finally:
            self._loading_ecg = False

    def _announce_ecg_auto_repair(self):
        """Tell the user that an impossible ECG combination was corrected on load.

        Not silent: the file said "auto-detect the whole batch", and the analysis now says
        it does not. Marking it dirty is deliberate too, so saving keeps the corrected
        pair instead of writing the invalid one back."""
        if not self._ecg_auto_repaired:
            return
        self._ecg_auto_repaired = False
        self.settings_edited.emit()
        self._set_status(
            "This analysis had ECG auto-detect switched on with Remove ECG switched off, "
            "which cannot run. Auto-detect has been turned off. Tick Remove ECG first if "
            "you want it back."
        )

    def _on_ecg_param_changed(self, *_):
        if self._loading_ecg:
            return
        e = self.state.settings.processing.emg
        e.remove_ecg = self.remove_ecg.isChecked()
        if not e.remove_ecg and self.ecg_auto_batch.isChecked():
            # Auto-detect requires Remove ECG (Settings.validate). Clear it here rather than
            # leave remove_ecg=False with ecg_auto_detect=True stuck: _update_actions would
            # then disable ecg_auto_batch while it is still checked, and the user could not
            # untick it again without re-enabling Remove ECG first.
            self.ecg_auto_batch.blockSignals(True)
            self.ecg_auto_batch.setChecked(False)
            self.ecg_auto_batch.blockSignals(False)
        e.ecg_auto_detect = self.ecg_auto_batch.isChecked()
        e.ecg_min_height = float(self.ecg_min_height.value())
        e.ecg_min_distance_s = float(self.ecg_min_distance.value())
        e.ecg_min_width_s = float(self.ecg_min_width.value())
        e.ecg_window_s = float(self.ecg_window.value())
        e.detect_channel = max(0, self.ecg_capture_channel.currentIndex())
        self.settings_edited.emit()      # ECG params land in the .toml -> mark dirty
        # The noise controls are gated on ECG removal, and that gate lives one strip away —
        # without this it stays stale until some other event happens to refresh it, leaving
        # them greyed out immediately after the user has turned Remove ECG on.
        self._update_actions(status=False)
        # ECG removal feeds the EMG/noise panels too, so recompute those alongside this tab.
        self._request_autorun({"ecg", "emg_all", "emg_detail", "noise"})

    def _open_ecg_advanced(self):
        """The two ECG parameters that are not worth strip space.

        Staged, not live: the dialog holds its own widgets and nothing reaches the settings
        unless OK is pressed, so Cancel needs no undo. On OK the commit goes through the same
        funnel a strip edit uses, which is what keeps the dirty flag and the recompute scope
        correct without repeating either here."""
        from respmech.ui.advanced_dialog import AdvancedDialog, Field
        e = self.state.settings.processing.emg
        fields = [
            Field("ecg_min_width_s", "Minimum peak width", "float",
                  "processing.emg.ecg_min_width_s",
                  "Minimum width of an R-wave peak. A shape guard against counting a narrow "
                  "spike as a heartbeat; the default rarely needs moving.",
                  lo=0.0, hi=0.1, step=0.001, decimals=4, suffix=" s"),
            Field("ecg_window_s", "Template width", "float", "processing.emg.ecg_window_s",
                  "Width of the ECG template averaged and subtracted around each beat "
                  "(QRS-T). Physiologically fixed — 0.4 s is right for adults.",
                  lo=0.05, hi=1.0, step=0.05, decimals=3, suffix=" s"),
        ]
        dlg = AdvancedDialog(
            "ECG removal — advanced", fields,
            {f.key: getattr(e, f.key) for f in fields}, parent=self,
            intro="Detection is driven by the capture channel, Min height and Min gap on the "
                  "strip. These two shape the template rather than finding the beats.")
        if dlg.exec() != QDialog.Accepted:
            return
        from respmech.ui.advanced_dialog import apply_values
        if not apply_values(e, dlg.values()):
            return                       # OK without an edit: no dirty flag, no recompute
        self._load_ecg_params()          # keep the (hidden) widgets in step with the model
        self.settings_edited.emit()
        self._request_autorun({"ecg", "emg_all", "emg_detail", "noise"})

    def _on_ecg_autosuggest(self):
        """Analyse the raw EMG and fill in the ECG settings (channel + params), then recompute."""
        path = self._current_file()
        if not path:
            self._set_status("Pick a file first, then Auto-suggest can analyse its EMG.")
            return
        try:
            from respmech.ui.workers import stage_raw_emg
            from respmech.core.emg import suggest_ecg_settings
            data = stage_raw_emg(self.state.settings, path)
            matrix = np.column_stack(data["raw"]) if data.get("raw") else np.empty((0, 0))
            fs = int(self.state.settings.input.format.sampling_frequency or data.get("fs") or 1)
            sug = suggest_ecg_settings(matrix, fs)
        except Exception:                              # noqa: BLE001
            self._set_status(f"Auto-suggest failed — {short_error(traceback.format_exc())}")
            return
        e = self.state.settings.processing.emg
        e.detect_channel = int(sug["detect_channel"])
        e.ecg_min_height = float(sug["ecg_min_height"])
        e.ecg_min_distance_s = float(sug["ecg_min_distance_s"])
        e.ecg_min_width_s = float(sug["ecg_min_width_s"])
        e.ecg_window_s = float(sug["ecg_window_s"])
        self._load_ecg_params()                        # reflect into the widgets (guarded)
        self.settings_edited.emit()      # Auto-suggest rewrote the ECG settings -> mark dirty
        diag = sug.get("_diagnostics", {})
        cols = list(self.state.settings.input.channels.emg)
        col = cols[e.detect_channel] if e.detect_channel < len(cols) else e.detect_channel
        if diag.get("confidence") == "low":
            self._set_status("Auto-suggest: no clear ECG found — set conservative defaults on the "
                             f"middle channel (col {col}). Tune the settings and watch the capture.")
        else:
            bpm = diag.get("est_bpm")
            self._set_status(f"Auto-suggest: capture on col {col}"
                             + (f" (~{bpm:.0f} bpm)" if bpm else "")
                             + ". Review the ▼ markers and tune if needed.")
        self._request_autorun({"ecg", "emg_all", "emg_detail", "noise"})

    def _on_ecg_result(self, data):
        """Render the ECG-reduction tab: the raw capture channel + detected R-peaks, and the
        ECG-processed channels below — each with the flow background + capture markers."""
        pal = _plot_pal()
        t = np.asarray(data["t"])
        peaks = data.get("peaks", [])
        self.ecg_capture_plot.clear()
        self.ecg_capture_plot.plot(t, np.asarray(data["raw_capture"]), pen=_pen(pal["raw_trace"]))
        add_flow_background(self.ecg_capture_plot.getPlotItem(), t, data.get("flow"), pal)
        add_ecg_capture_markers(self.ecg_capture_plot.getPlotItem(), peaks, pal)
        self._limit_x(self.ecg_capture_plot.getPlotItem(), t)

        self.ecg_processed_plots.clear()
        self._ecg_capture_subplots = []
        cols = data.get("cols", [])
        proc = data.get("processed", [])
        cycle = pal["emg_cycle"]
        for i, ch in enumerate(proc):
            p = self.ecg_processed_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            if _theme is not None:
                _theme.align_left_axis(p)
            p.plot(t, np.asarray(ch), pen=_pen(cycle[i % len(cycle)]))
            add_flow_background(p, t, data.get("flow"), pal)
            add_ecg_capture_markers(p, peaks, pal)
            self._limit_x(p, t)
            self._ecg_capture_subplots.append(p)
        # x-values on the bottom channel only; y-zoom shared (all ECG-processed EMG)
        self._style_channel_stack(self.ecg_processed_plots, self._ecg_capture_subplots,
                                  link_y=True)

        npk = int(np.asarray(peaks).size)
        col = data.get("detect_col", "?")
        applied = bool(data.get("ecg_applied"))
        state = "ECG removed" if applied else "capture preview (removal is OFF for the run)"
        # Only meaningful when removal actually ran: stage_ecg_reduction returns None (not
        # NaN) while removal is off, and NaN when it ran but could not be computed (e.g. no
        # peaks) — both are "nothing to show", so guard on == self first (NaN != NaN).
        supp = data.get("suppression")
        supp_txt = ""
        if applied and supp is not None and supp == supp:
            supp_txt = f" · suppression {supp:.0%}"
        base_title = _ECG_PROCESSED_TITLE_ON if applied else _ECG_PROCESSED_TITLE_OFF
        if data.get("ecg_error"):
            self._set_status(f"ECG reduction — capture on col {col}: {short_error(data['ecg_error'])}")
            self._set_ecg_capture_title(f"{_ECG_CAPTURE_TITLE}  ·  detection failed — "
                                        f"{short_error(data['ecg_error'])}")
            # stage_ecg_reduction falls back to the RAW channels whenever detection/removal
            # raises (see its docstring), regardless of the remove_ecg SETTING — so the
            # error title must say OFF too, or it repeats the exact bug this ticket exists
            # to fix: "ECG-processed" over a stack that is, right now, unprocessed.
            self._set_ecg_processed_title(f"{_ECG_PROCESSED_TITLE_OFF}  ·  detection failed")
        else:
            self._set_status(f"ECG reduction — capture on col {col}: {npk} R-peaks · {state}{supp_txt}.")
            self._set_ecg_capture_title(
                f"{_ECG_CAPTURE_TITLE}  ·  {npk} R-peak{'s' if npk != 1 else ''} "
                f"on col {col} · {state}")
            self._set_ecg_processed_title(f"{base_title}{supp_txt}")

    def _set_ecg_capture_title(self, text=None):
        """Keep the ECG capture panel's own header naming the live detection result — a
        persistent verdict, unlike the status-bar message above, which is gone within a
        second or so. ``text=None`` (or omitted) resets it to the bare base title, used when
        the panel is blanked for a file switch (see screen.py's _clear_*_panels)."""
        label = getattr(self._ecg_capture_box, "_title_label", None)
        if label is not None:
            label.setFullText(text if text is not None else _ECG_CAPTURE_TITLE)

    def _set_ecg_processed_title(self, text=None):
        """Keep the processed-stack panel's header naming whether removal is actually on —
        the same persistent-verdict treatment as :meth:`_set_ecg_capture_title`. ``text=None``
        (or omitted) resets it to the CURRENT ``remove_ecg`` setting's base title, used when
        the panel is blanked for a file switch (see screen.py's _clear_*_panels); the next
        ``_on_ecg_result`` corrects it (with a suppression figure, if any) once a real result
        is in. Unlike the capture panel's neutral base text, the processed panel's ON title is
        itself a claim ("ECG-processed"), so — unlike :meth:`_set_ecg_capture_title` — the
        reset must not hard-code ON: with removal off, a blanked panel that still says
        "ECG-processed" is the same misleading verdict this ticket exists to fix, just shown
        a beat early, and the settings-incomplete gate in screen.py's ``_schedule`` can leave
        a cleared panel sitting on this default for a while (no ``_on_ecg_result`` fires while
        gated)."""
        label = getattr(self._ecg_processed_box, "_title_label", None)
        if label is None:
            return
        if text is not None:
            label.setFullText(text)
            return
        applied = bool(self.state.settings.processing.emg.remove_ecg)
        label.setFullText(_ECG_PROCESSED_TITLE_ON if applied else _ECG_PROCESSED_TITLE_OFF)

"""PreviewScreen's EMG + noise-reduction sub-tab: raw/result/detail EMG views,
the rest-region noise reference picker and the fidelity-frontier preview.
Split out of preview_screen.py (ticket A02); moved verbatim."""

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
from respmech.ui.noise_profile_dialog import NOISE_ACCENT
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

from ._figure_fit import _CompactFigureFitter, _fit_compact_figure, refit_compact_figure
from ._jobs import _FileRunError, _TAB_ECG, _TAB_NOISE
from ._plot_helpers import _FitAxis, _check_icon_url, _pen, _plot_pal, _rms_envelope, _tick_colour


#: why the noise controls are inert until ECG removal is on — one wording, used by both the
#: tooltip and the status line so they cannot drift apart
NEEDS_ECG_HINT = ("Turn on 'Remove ECG' first: the noise profile is measured on the "
                  "ECG-cleaned signal, so with the heartbeat still in the EMG it would be "
                  "modelled as steady background noise — which it is not.")

#: why the gated peak is inert until ECG removal is on
NEEDS_ECG_GATE_HINT = ("Turn on 'Remove ECG' first: the gate is built from the heartbeats "
                       "the ECG stage detects, so with it off there are none to blank and "
                       "the gated columns come back empty.")

#: the fidelity panel's base title; _set_fidelity_panel_title appends the chosen suppression
#: and the worst-channel fidelity so the panel carries a persistent verdict instead of one
#: that only ever lived a few seconds in the shared status bar.
_FIDELITY_TITLE = "Noise fidelity frontier (1 = untouched)"

#: appended to the fidelity panel's tooltip (never to the visible, elided title — see
#: ElidingLabel) so the definition is one hover away wherever the panel's header is used.
#: The band and the "values above 1 are routine" note answer the question nothing on this
#: tab otherwise answers (D04, UI-overhaul); the parenthetical names the Advanced field this
#: mirrors, in its own wording, so the two never have to be kept in sync by hand.
_FIDELITY_TOOLTIP_SUFFIX = (
    "\n\nFidelity is the fraction of a channel's inspiratory EMG power, in the 20–250 Hz "
    "band shaded on the Detail PSD panel, that survives noise removal (1.0 = untouched). "
    "Values a little above 1 are routine, not a defect: the spectral gate masks magnitude and "
    "phase separately, so the reconstruction can hold slightly more in-band power than the "
    "original did. The dotted line marks the fidelity target from Advanced → "
    "“Fidelity target (Keep ≥)”.")


class _EmgNoiseMixin:

    def _build_emg_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)

        # detail-channel dropdown (pinned into the Detail panel title, top-right) and the
        # result-channel picker (pinned into the Conditioned-result panel title, top-right)
        self.emg_channel = QComboBox()
        self.emg_channel.setToolTip("EMG channel shown in the single-channel Detail panel.")
        self.emg_channel.currentIndexChanged.connect(self._on_emg_channel_changed)
        self.result_checks_holder = QWidget()
        # A WRAPPING row: this floats in the plot's top band, and a plain QHBoxLayout keeps
        # its full natural width there — measured at 12 EMG channels it wanted 1400 px in a
        # 1242 px plot, so the last channel's checkbox was clipped away entirely and could
        # not be reached at all. Wrapping lets the band grow a line instead.
        self.result_checks_layout = _install_flow(self.result_checks_holder, h=24, v=2)

        # -- one compact controls strip above the plots ----------------------
        # the 'Set noise profile' button + the noise-reduction params (in a subtle
        # rounded chip) on a single row, instead of two full-width rows.
        self.btn_set_noise = QPushButton("Set noise profile")
        self.btn_set_noise.setToolTip("Open a picker to mark a rest span across the raw EMG "
                                      "channels as the shared noise reference for this test.")
        self.btn_set_noise.clicked.connect(self._open_noise_profile_dialog)

        # The on/off toggle for EMG-noise processing — the one control the user asked to keep
        # on the tab (it left the Setup screen with the rest of the noise settings).
        self.noise_enabled = QCheckBox("Reduce EMG noise")
        self.noise_enabled.setToolTip(_help_tip(
            "processing.emg.noise.enabled",
            "Subtract a shared spectral noise profile (built from a rest reference) from every "
            "EMG channel. Pick the reference with 'Set noise profile'; off by default."))
        self.noise_enabled.toggled.connect(self._on_noise_enabled_changed)
        self.noise_ref_readout = QLabel("")
        self.noise_ref_readout.setProperty("status", "muted")
        self.noise_auto = QCheckBox("Auto")
        self.noise_auto.setToolTip(_help_tip(
            "processing.emg.noise.auto_prop",
            "Automatically picks the strongest suppression that still keeps every channel at or "
            "above the fidelity target; on by default."))
        self.noise_auto.toggled.connect(self._on_noise_param_changed)

        def _cap(text, tip=""):        # a compact muted caption; full text lives in the tooltip
            la = QLabel(text); la.setProperty("status", "muted")
            if tip:
                la.setToolTip(tip)
            return la

        # noise-reduction params grouped in a subtle rounded chip (active once a
        # reference file is set — _update_actions enables/disables the whole chip)
        self.noise_opts = QFrame()
        self.noise_opts.setObjectName("noiseChip")
        self.noise_opts.setStyleSheet(
            "#noiseChip { border: 1px solid rgba(128, 128, 128, 0.30); border-radius: 8px; }")
        # tight caption→field pairing (4 px) with a wider gap (10 px) BETWEEN groups so
        # 'Strength [ ]', 'Keep ≥ [ ]', 'Gate [ ]' read as clusters, not an even bead line.
        # A FlowLayout, not a QHBoxLayout: a chip built on a plain row is one indivisible item
        # whose minimum is the SUM of its clusters, and the enclosing strip can only wrap around
        # it, never inside it (see flow_layout's module docstring).
        nrow = _install_flow(self.noise_opts, h=10, v=4, margins=(11, 3, 11, 3))
        nrow.addLayout(_cluster(_cap("Noise", self.noise_auto.toolTip()), self.noise_auto, gap=8))
        # No "Gate" cluster and no gated-peak chip: the spectral-gate threshold was a straight
        # duplicate of the Advanced modal's "Spectral gate threshold", and the cardiac-gated
        # peak is an output-only add-on nothing on this tab draws — both live in Advanced now.
        # The strip carries only what the module docstring calls the knobs that matter, which
        # is what stopped it wrapping to two lines on a laptop (Emil, 02-08-2026).

        self.btn_emg_advanced = QPushButton("Advanced…")
        self.btn_emg_advanced.setProperty("compact", True)
        self.btn_emg_advanced.setToolTip("The cardiac-gated peak, spectral-gate internals, "
                                         "quality guards and the diagnostic exports.")
        self.btn_emg_advanced.clicked.connect(self._open_emg_advanced)

        # A WRAPPING row, not a QHBoxLayout: these chips together are ~1700 px wide, and a
        # non-wrapping row's minimum is their SUM, which Qt propagates up as the main window's
        # minimum width — the window then could not be resized to fit a laptop screen and ran
        # off the right edge. Wrapping makes the minimum the widest single chip instead.
        strip = FlowLayout(h=10, v=6)
        # The on/off checkbox lives on the STRIP, not inside the noise_opts chip: that chip is
        # disabled whenever noise is off, and a Qt child of a disabled parent cannot be
        # re-enabled — so a checkbox to turn noise ON, placed inside it, would grey itself out
        # exactly when it is needed. On the strip it stays reachable (gated only on ECG being on).
        strip.addWidget(self.noise_enabled)
        strip.addWidget(self.btn_set_noise)
        strip.addWidget(self.noise_ref_readout)
        strip.addWidget(self.noise_opts)
        strip.addWidget(self.btn_emg_advanced)
        v.addLayout(strip)

        # plots — three rows: raw stack | detail channel on top, the full-width
        # conditioned result in the middle, the two small diagnostics below (see the
        # splitter assembly further down for why each panel sits where it does).
        _bg = _plot_pal()["bg"]
        self.emg_raw_plots = pg.GraphicsLayoutWidget()
        self.emg_raw_plots.setAccessibleName("Raw EMG channels")   # C02: named for QAccessible/screen readers
        _theme.set_plot_floor(self.emg_raw_plots)
        self.emg_raw_plots.setBackground(_bg)
        # Both working views carry a height-aware left axis (see _FitAxis): at the compact
        # heights this tab now defaults to, stock pyqtgraph printed the y labels through
        # each other. Neither has an in-plot legend any more — at ~43 px of data area a
        # legend row covered nearly half the trace it was labelling. The conditioned view
        # needs no replacement, its corner picker IS the key (same emg_cycle colours); the
        # detail view names its processing stages in the title band instead.
        self.emg_result_plots = pg.PlotWidget(axisItems={"left": _FitAxis(orientation="left")})
        _theme.set_plot_floor(self.emg_result_plots)
        plot_perf.tune_widget(self.emg_result_plots)
        self.emg_result_plots.setBackground(_bg)
        self.emg_result_plots.setLabel("bottom", "Time (s)")
        # "Conditioned" is in the panel title; the label shortens itself when the panel is
        # too short to draw it whole (see _FitAxis.set_label_variants)
        self.emg_result_plots.getPlotItem().getAxis("left").set_label_variants("EMG (a.u.)", "EMG", None)
        self.emg_plots = pg.PlotWidget(axisItems={"left": _FitAxis(orientation="left")})
        _theme.set_plot_floor(self.emg_plots)
        plot_perf.tune_widget(self.emg_plots)
        self.emg_plots.setBackground(_bg)
        self.emg_plots.getPlotItem().getAxis("left").set_label_variants("EMG (a.u.)", "EMG", None)
        # Pin both left axes to one width. They are stacked and read against a SHARED time
        # axis (the detail view has no x caption of its own precisely because the conditioned
        # view below carries it), but each picks its label wording from its OWN height — so
        # after a splitter drag one said "EMG (a.u.)" and the other "EMG", and their data
        # areas started 6.8 px apart, stepping the two time axes out of line.
        for _pw in (self.emg_plots, self.emg_result_plots):
            _theme.align_left_axis(_pw.getPlotItem())
        # No "Time (s)" caption here: the conditioned view directly below carries it on an
        # identical axis, and the caption cost 18 px out of a 43 px data area — a 40% gain
        # in trace height for a word the panel underneath already says. The tick numbers
        # stay, so this view is still readable on its own.
        self.emg_trace_key = QLabel()
        self.emg_trace_key.setTextFormat(Qt.RichText)
        # the DIAGNOSTIC floor, not the working-view one: these two share the compact
        # reference row (a canvas reports a 10px minimum of its own, so a floor is needed)
        self.emg_psd_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        _theme.set_plot_floor(self.emg_psd_canvas, _theme.PLOT_DIAG_MIN_HEIGHT)

        self.fidelity_canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
        _theme.set_plot_floor(self.fidelity_canvas, _theme.PLOT_DIAG_MIN_HEIGHT)
        # both diagnostics re-fit their furniture to the height a splitter drag leaves them
        self._compact_fitters = [_CompactFigureFitter(self.emg_psd_canvas),
                                 _CompactFigureFitter(self.fidelity_canvas)]
        split = QSplitter(Qt.Vertical)
        # Detail channel and Conditioned result are the two working views — the conditioning
        # stages and the finished EMG, each against time — so both span the full width, where
        # width is what makes them readable. Neither carries an in-plot legend: at this
        # panel height a legend row covered the trace it was labelling (see _set_trace_key).
        # Title + picker float INSIDE each plot's top band rather than on a row above it —
        # two full-width rows of vertical space given back to the graphs (see _PlotTitleOverlay).
        split.addWidget(self._titled_overlay("Detail channel", self.emg_plots,
                                             corner=self.emg_channel,
                                             extra=self.emg_trace_key))  # stages named here
        split.addWidget(self._titled_overlay("Conditioned result",   # picker names the channels
                                             self.emg_result_plots,
                                             corner=self.result_checks_holder))
        # The bottom row holds three reference panels: the raw channels alongside the two
        # compact diagnostics (the fidelity frontier and the detail PSD). None needs the width,
        # and the raw stack is a reference the working views are read against.
        bottom = QSplitter(Qt.Horizontal)
        # The raw stack scrolls INSIDE its panel. Its per-channel floor is what keeps each
        # trace readable, but as the panel's own minimum that floor became the bottom row's
        # minimum too — 3 channels demanded ~300 px, 12 would demand over 1100, so the row
        # could neither shrink to a sensible default nor be dragged (every panel already sat
        # at its minimum, and a splitter can only redistribute slack). Scrolling here keeps
        # the traces readable AND gives the outer splitter something to move.
        self._emg_raw_scroll = QScrollArea()
        self._emg_raw_scroll.setWidgetResizable(True)
        self._emg_raw_scroll.setFrameShape(QFrame.NoFrame)
        self._emg_raw_scroll.setWidget(self.emg_raw_plots)
        self._emg_raw_wheel = _wheel.guard_scroll_area(
            self._emg_raw_scroll, extra=[self.emg_raw_plots.viewport()])
        bottom.addWidget(self._titled("Raw EMG channels", self._emg_raw_scroll))
        self._fidelity_panel = self._titled(_FIDELITY_TITLE, self.fidelity_canvas)
        # set once here too (not only in _set_fidelity_panel_title) so the explanation is
        # available before any file has been run and produced a report to reset the title
        self._fidelity_panel._title_label.setToolTip(_FIDELITY_TITLE + _FIDELITY_TOOLTIP_SUFFIX)
        bottom.addWidget(self._fidelity_panel)
        self._psd_panel = self._titled("Detail PSD", self.emg_psd_canvas)
        bottom.addWidget(self._psd_panel)
        self._emg_diag_row = bottom
        split.addWidget(bottom)
        # 1 : 1 : 2 — the two working views TOGETHER default to the same share as the
        # reference row, which with the fixed chrome above lands the tab on roughly equal
        # thirds: chrome+settings / the two working views / the three reference panels
        # (Emil, 02-08-2026). Stretch factors alone only divide SPARE space, so the sizes
        # are seeded explicitly too; the user can still drag from here.
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 1); split.setStretchFactor(2, 2)
        split.setSizes([200, 200, 400])
        v.addWidget(split, 1)
        # clicking a numbered breath in any EMG plot toggles its exclusion too
        self.emg_raw_plots.scene().sigMouseClicked.connect(self._on_emg_raw_clicked)
        self.emg_plots.scene().sigMouseClicked.connect(self._on_emg_detail_clicked)
        self.emg_result_plots.scene().sigMouseClicked.connect(self._on_emg_result_clicked)
        return w

    def _open_emg_advanced(self):
        """Spectral-gate internals, the gated-peak quality guards, and the two diagnostic
        exports that never had a control at all.

        n_fft / win_length / hop_length are exposed SEPARATELY rather than collapsed into one
        "window" control. Collapsing them would change computed output for any analysis where
        they disagree — and the goldens use the defaults, so the goldens could not catch it.
        The coupling is communicated instead, in milliseconds at the current rate, because
        256 samples means something different at 1000 Hz and at 2000 Hz."""
        from respmech.ui.advanced_dialog import AdvancedDialog, Field, apply_values
        e = self.state.settings.processing.emg
        n, rp = e.noise, e.robust_peak
        fs = self.state.settings.input.format.sampling_frequency or 1000

        supp_fields = [
            # Suppression strength and the fidelity target moved here off the strip
            # (Emil, 02-08-2026): with 'Auto' on — the default — the strength is chosen for
            # you and the field is inert, so two permanently-visible spin boxes were paying
            # strip width for a case most runs never enter. 'Auto' itself stays on the strip.
            Field("prop_decrease", "Suppression strength (when Auto is off)", "float",
                  "processing.emg.noise.prop_decrease",
                  "How aggressively to remove noise when Auto is off, from 0 (none) to 1 "
                  "(maximum); default 0.6. With Auto on, this is overwritten by the strength "
                  "the run chooses.",
                  lo=0.0, hi=1.0, step=0.05, decimals=2),
            Field("fidelity_target", "Fidelity target (Keep ≥)", "float",
                  "processing.emg.noise.fidelity_target",
                  "Smallest fraction of inspiratory EMG power that must survive noise "
                  "removal, from 0 to 1; default 0.8. This is what Auto optimises against.",
                  lo=0.50, hi=0.99, step=0.05, decimals=2),
        ]
        stft_fields = [
            Field("n_std_thresh", "Spectral gate threshold", "float",
                  "processing.emg.noise.n_std_thresh",
                  "How many standard deviations above the noise profile a frequency bin must "
                  "be to survive. The default of 1.0 is essentially never changed.",
                  lo=0.0, hi=10.0, step=0.1, decimals=1),
            Field("n_fft", "STFT length (n_fft)", "int", "processing.emg.noise.n_fft",
                  "FFT length in samples for the spectral gate; a power of two.",
                  lo=16, hi=8192, step=16),
            Field("win_length", "STFT window", "int", "processing.emg.noise.win_length",
                  "Analysis window in samples. Kept separate from n_fft on purpose: writing "
                  "both from one control would change the output of any analysis where they "
                  "differ.", lo=16, hi=8192, step=16),
            Field("hop_length", "STFT hop", "int", "processing.emg.noise.hop_length",
                  "Advance between successive windows, in samples.", lo=1, hi=8192, step=8),
            Field("n_grad_freq", "Mask smoothing — frequency", "int",
                  "processing.emg.noise.n_grad_freq",
                  "Frequency bins the suppression mask is smoothed over.", lo=0, hi=64),
            Field("n_grad_time", "Mask smoothing — time", "int",
                  "processing.emg.noise.n_grad_time",
                  "Time frames the suppression mask is smoothed over.", lo=0, hi=64),
        ]
        gate_fields = [
            Field("min_survival", "Least of each phase that must survive", "float",
                  "processing.emg.robust_peak.min_survival",
                  "Below this fraction the gated value is left blank rather than reported "
                  "from a sliver of breath. Measured default 0.40.",
                  lo=0.0, hi=1.0, step=0.05, decimals=2),
            Field("min_island_s", "Shortest usable stretch between beats", "float",
                  "processing.emg.robust_peak.min_island_s",
                  "Gaps shorter than this cannot hold a full RMS window, so they are ignored.",
                  lo=0.0, hi=2.0, step=0.05, decimals=3, suffix=" s"),
            Field("long_rr_factor", "Missed-heartbeat factor", "float",
                  "processing.emg.robust_peak.long_rr_factor",
                  "A gap this many times the median R-R is treated as a missed heartbeat.",
                  lo=1.1, hi=5.0, step=0.1, decimals=2),
            Field("max_long_rr_frac", "Missed heartbeats tolerated", "float",
                  "processing.emg.robust_peak.max_long_rr_frac",
                  "Above this fraction of long gaps the detection is not trusted and the "
                  "gated columns are left blank.", lo=0.0, hi=1.0, step=0.01, decimals=3),
            Field("hr_ceiling_margin", "Heart-rate ceiling margin", "float",
                  "processing.emg.robust_peak.hr_ceiling_margin",
                  "How close the detected rate may come to the ceiling implied by Min gap "
                  "before the result is distrusted. If this is firing, the fix is usually to "
                  "lower Min gap on the › EMG – ECG reduction tab, not to raise this.",
                  lo=0.0, hi=0.5, step=0.05, decimals=2),
        ]
        emg_fields = [
            Field("rms_window_s", "RMS window length", "float", "processing.emg.rms_window_s",
                  "Sliding-window length for the EMG RMS envelope; each breath takes its "
                  "largest windowed value (default 0.05). Defines the reported EMG number.",
                  lo=0.01, hi=0.5, step=0.01, decimals=3, suffix=" s"),
            Field("outlier_rms_sd_limit", "RMS outlier limit", "float",
                  "processing.emg.outlier_rms_sd_limit",
                  "Replace any breath's EMG RMS lying more than this many SD from the "
                  "across-breath mean; 0 = off (default).",
                  lo=0.0, hi=10.0, step=0.5, decimals=1, suffix=" SD"),
            Field("normalization", "Amplitude normalisation", "choice",
                  "processing.emg.normalization",
                  "Also report each file's EMG RMS as a percentage of a per-file reference "
                  "(adds a normalised sheet to the output; never changes the raw RMS).",
                  options=[("None — raw amplitude", "none"),
                           ("Per-file maximum (% of peak breath)", "per_file_max"),
                           ("Per-file mean (% of mean breath)", "per_file_mean")]),
        ]
        out_fields = [
            Field("save_sound", "Export each EMG stage as WAV", "bool",
                  "processing.emg.save_sound",
                  "Writes an audio file per channel per conditioning stage. Diagnostic; off "
                  "by default.",
                  note="Adds several files per recording."),
        ]

        def _ms(v):
            return (f"At {fs} Hz: window {1000 * v['win_length'] / fs:.0f} ms, "
                    f"hop {1000 * v['hop_length'] / fs:.0f} ms, "
                    f"n_fft {1000 * v['n_fft'] / fs:.0f} ms.")

        # The cardiac-gated peak, moved here off the strip (Emil, 02-08-2026): an
        # output-only add-on nothing on the tab draws did not earn strip space. Its ECG
        # prerequisite is enforced below at open time.
        ecg_on = bool(e.remove_ecg)
        gp_fields = [
            Field("enabled", "Add gated-peak columns", "bool",
                  "processing.emg.robust_peak.enabled",
                  "The peak EMG of a breath is its single loudest moment — but on a "
                  "recording where the heartbeat bleeds into the EMG, that moment is often "
                  "a leftover heartbeat, not the diaphragm. Tick this to ALSO measure the "
                  "peak from just the heartbeat-free stretches of each breath. It changes "
                  "nothing drawn here: it adds extra 'gated' columns to the saved data, and "
                  "the existing numbers never change."),
            Field("gate_half_width_s", "Blanked around each heartbeat", "float",
                  "processing.emg.robust_peak.gate_half_width_s",
                  "Blanked either side of each detected heartbeat, so the default 0.120 "
                  "discards a 0.240 s window in total. It must cover the heartbeat's "
                  "footprint in the RMS envelope, which is roughly the RMS window length "
                  "plus the QRS duration.",
                  # "±" in the field: the stored value is a HALF-width, and a bare
                  # "0.120 s" would understate the discarded signal by a factor of two.
                  lo=0.02, hi=0.5, step=0.01, decimals=3, prefix="± ", suffix=" s",
                  depends_on="enabled"),
        ]

        prefixed = ([("emg", f) for f in emg_fields]
                    + [("noise", f) for f in supp_fields]
                    + [("noise", f) for f in stft_fields]
                    + [("robust_peak", f) for f in gp_fields]
                    + [("robust_peak", f) for f in gate_fields]
                    + [("emg", f) for f in out_fields])
        owner = {"noise": n, "robust_peak": rp, "emg": e}
        values = {f.key: getattr(owner[grp], f.key) for grp, f in prefixed}
        # The groups already existed here as separate lists and were flattened away on
        # the way into the dialog; the cards just stop throwing that structure out.
        dlg = AdvancedDialog(
            "EMG — advanced",
            # Column balancing is CONTIGUOUS (section_flow keeps the author's reading order
            # at every width), so this order is a layout decision, not just a taxonomy.
            # Emil asked for Diagnostics under RMS to shorten the modal, and at THREE columns
            # it does (730 vs 840 px). At TWO — which is what a 1280 px laptop gets — it made
            # the body 156 px TALLER (808 vs 652), because it pushed the 443 px spectral-gate
            # card into the first column. The fix is finer cards rather than a reorder: the
            # one oversized card is split into the two things it actually was, which balances
            # at both counts, and Diagnostics goes back to the end where it costs nothing.
            # "and", not "&": a QGroupBox title treats & as a mnemonic marker, so
            # "RMS & normalisation" rendered as "RMS _normalisation".
            [("RMS and normalisation", emg_fields),
             ("Noise suppression", supp_fields),
             ("Spectral gate (STFT)", stft_fields),
             ("Gated peak (saved output)", gp_fields),
             ("Heartbeat and island guards", gate_fields),
             ("Diagnostics", out_fields)],
            values, parent=self,
            intro="The strip keeps the noise on/off toggle, the reference picker and "
                  "Auto. Everything that tunes HOW the gate is computed lives here.",
            derived=_ms)
        # 'Auto' owns the suppression strength: with it on the run overwrites whatever is
        # here, so an edit would silently do nothing. The strip control used to grey out for
        # exactly this reason; the field kept the behaviour when it moved.
        if bool(n.auto_prop):
            w = dlg.widget("prop_decrease")
            w.setEnabled(False)
            w.setToolTip("Auto is on, so the run picks the suppression strength — turn Auto "
                         "off on the strip to set it by hand.")
        if not ecg_on:
            # Same prerequisite the strip control enforced, same words: the gate is built
            # from the heartbeats the ECG stage detects, so without it the columns come
            # back blank. depends_on greys the width field with its checkbox; this greys
            # the checkbox itself.
            for f in gp_fields:
                w = dlg.widget(f.key)
                w.setEnabled(False)
                w.setToolTip(NEEDS_ECG_GATE_HINT)
        if dlg.exec() != QDialog.Accepted:
            return
        # edited_values(), not values(): an untouched OK must not write back a value the
        # app changed while the modal was open (a finished noise sweep does exactly that).
        staged = dlg.edited_values()
        changed_owners = set()
        for grp, f in prefixed:
            if f.key in staged and apply_values(owner[grp], {f.key: staged[f.key]}):
                changed_owners.add(grp)
        if not changed_owners:
            return                       # OK without an edit: no dirty flag, no recompute
        self._load_noise_params()
        self.settings_edited.emit()
        if changed_owners == {"robust_peak"}:
            # robust_peak.* is written by the batch and drawn by no preview panel
            # (_kinds_for_settings_path maps it to no kinds at all) — the contract the old
            # strip control kept, kept here: recomputing four panels for columns none of
            # them show would be pure noise.
            self._update_actions(status=False)
            return
        # RMS window / normalisation feed the EMG panels; the outlier limit feeds the batch
        # (test run) table, so 'batch' is in the set now that the modal edits it.
        self._request_autorun({"ecg", "emg_all", "emg_detail", "noise", "batch"})

    # -- EMG tab visibility / channels -------------------------------------
    def _update_emg_tab_visibility(self):
        """Show the two EMG sub-tabs only when EMG channels exist, in the fixed pipeline order
        Mechanics(0) › EMG–ECG reduction(1) › EMG–noise reduction(2)."""
        has_emg = bool(self.state.settings.input.channels.emg)
        if has_emg:
            if self.subtabs.indexOf(self._ecg_tab) < 0:
                self.subtabs.insertTab(1, self._ecg_tab, _TAB_ECG)     # right after Mechanics(0)
            if self.subtabs.indexOf(self._emg_tab) < 0:
                self.subtabs.insertTab(2, self._emg_tab, _TAB_NOISE)   # after the ECG tab(1)
        else:
            for tab in (self._emg_tab, self._ecg_tab):
                idx = self.subtabs.indexOf(tab)
                if idx >= 0:
                    self.subtabs.removeTab(idx)

    def _refresh_emg_channels(self):
        cols = list(self.state.settings.input.channels.emg)
        cur = self.emg_channel.currentIndex()
        self.emg_channel.blockSignals(True)
        self.emg_channel.clear()
        for c in cols:
            self.emg_channel.addItem(f"EMG col {c}")
        if 0 <= cur < self.emg_channel.count():
            self.emg_channel.setCurrentIndex(cur)
        self.emg_channel.blockSignals(False)
        # rebuild the result-channel picker: each checkbox's indicator IS the channel's
        # coloured box with the tick inside it, and 'col N' sits right beside its colour;
        # generous spacing keeps the channels visually separate.
        while self.result_checks_layout.count():
            item = self.result_checks_layout.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.setParent(None)
        self.result_checks = []
        cycle = _plot_pal()["emg_cycle"]
        for i, c in enumerate(cols):
            r, g, b = cycle[i % len(cycle)][:3]
            check = _check_icon_url(_tick_colour((r, g, b)))
            cb = QCheckBox(f"col {c}")
            cb.setChecked(i < 3)
            # C02: a widget-level setStyleSheet() shadows the app-wide QSS for every
            # selector it touches — including pseudo-states the LOCAL sheet never
            # mentions. These per-channel checkboxes redefine QCheckBox::indicator for
            # their colour coding, which silently ate the app-wide ::indicator:focus
            # rule added for C02 (found in review: focusing one produced zero visible
            # difference, the exact bug this ticket exists to close). So the focus ring
            # has to be repeated HERE too. Same footprint arithmetic as theme.py's global
            # rule: 15px content + 1px border == 13px content + 2px border == 17px outer,
            # unchanged by focus. Accent colour (not the channel colour) marks focus, same
            # semantic as every other focusable control in the app; read live so it tracks
            # the active theme rather than freezing at light mode.
            _accent = _theme.active_theme()["accent"] if _theme else "#2C6E9B"
            cb.setStyleSheet(
                "QCheckBox { spacing: 6px; }"
                "QCheckBox::indicator { width: 15px; height: 15px; border-radius: 3px; "
                f"border: 1px solid rgba({r}, {g}, {b}, 0.9); background: rgba({r}, {g}, {b}, 0.25); }}"
                f"QCheckBox::indicator:checked {{ background: rgb({r}, {g}, {b}); "
                f"image: url('{check}'); }}"
                "QCheckBox:focus { outline: none; }"
                f"QCheckBox::indicator:focus {{ width: 13px; height: 13px; "
                f"border: 2px solid {_accent}; border-radius: 3px; }}")
            cb.toggled.connect(self._render_emg_result)
            self.result_checks_layout.addWidget(cb)
            self.result_checks.append((c, cb))

    # -- noise-window options (owned here, gated on reference file) ---------
    def _load_noise_params(self):
        n = self.state.settings.processing.emg.noise
        self._loading_noise = True
        try:
            self.noise_enabled.setChecked(n.enabled)
            self.noise_auto.setChecked(n.auto_prop)
        finally:
            self._loading_noise = False
        self._refresh_noise_readout()
        # Called from both the constructor and every sync_from_settings (D07, UI-overhaul):
        # one fix here covers "opened a fresh AppState" and "opened/loaded a saved
        # analysis" alike. Before this the Detail plot's reference band was seeded ONCE at
        # construction from whatever settings existed then — an empty AppState's default
        # 0.0-1.0 — and never moved again except through an interactive "Set noise profile"
        # approval, so every session that OPENS a saved reference (the app's own declared
        # workflow: set up once, revisit for many subjects) saw a band pointing at the
        # first second of the recording regardless of where the reference actually was.
        self._refresh_noise_reference_band()

    def _refresh_noise_readout(self):
        """What reference the picker chose, beside the enable toggle. 'Not set' is a real
        state and says so — a ticked 'Reduce EMG noise' with no reference runs nothing."""
        n = self.state.settings.processing.emg.noise
        # Elided: a real recording's filename is far longer than the sample's, and a QLabel's
        # minimum width is its whole text — un-elided it would push the strip (and with it the
        # window's minimum width) arbitrarily wide. The full text stays in the tooltip.
        # 220 px, not the 320 px default: this read-out shares one line with the toggle, the
        # picker button, the Auto chip and Advanced, and at 320 px on Windows font metrics the
        # five together came to 1292 px against a 1258 px page — the strip wrapped to two lines
        # for a caption whose full text is one hover away.
        cap = 220
        if not n.reference_file:
            _elide(self.noise_ref_readout, "· no reference set", cap)
        elif n.use_expiration or not n.reference_intervals:
            _elide(self.noise_ref_readout, f"· {n.reference_file}, every expiration", cap)
        else:
            spans = ", ".join(f"{a:.2f}–{b:.2f} s" for a, b in n.reference_intervals)
            _elide(self.noise_ref_readout, f"· {n.reference_file}, {spans}", cap)

    def _on_noise_enabled_changed(self, *_):
        if self._loading_noise:
            return
        self.state.settings.processing.emg.noise.enabled = self.noise_enabled.isChecked()
        self.settings_edited.emit()
        self._update_actions(status=False)          # the reference/opts gate follows the toggle
        self._request_autorun({"emg_all", "emg_detail", "noise", "batch"})

    def _on_noise_param_changed(self, *_):
        if self._loading_noise:
            return
        n = self.state.settings.processing.emg.noise
        n.auto_prop = self.noise_auto.isChecked()
        self.settings_edited.emit()      # noise params land in the .toml -> mark dirty
        self._update_actions()
        self._request_autorun()          # re-condition with the new noise params

    def _align_noise_strip(self):
        """Match the 'Set noise profile' button to the noise chip's height so the strip
        reads as one aligned band. Keyed off the chip's *sizeHint* (its natural, laid-out
        height), never its live allocated height: the button's minimum height feeds back
        into the strip's row height, so reading the live height would let the chip balloon
        a little more on every tab switch (the chip is clamped to Maximum vertically, but
        sizeHint keeps this idempotent regardless)."""
        h = self.noise_opts.sizeHint().height()
        if h > 1 and self.btn_set_noise.minimumHeight() != h:
            self.btn_set_noise.setMinimumHeight(h)

    def _toggle_from_emg_click(self, ev, plot_items, offset):
        if not self._breath_spans:
            return
        try:
            if ev.isAccepted():
                return
            pos = ev.scenePos()
        except Exception:                              # noqa: BLE001
            return
        for p in plot_items:
            # a click on the plot's legend (text/frame) is unaccepted by pyqtgraph
            # and would otherwise fall through to the breath underneath it — the legend
            # sits bottom-left on the detail/result plots (sceneBoundingRect tracks it
            # wherever it is anchored, so this guard follows the move)
            leg = getattr(p, "legend", None)
            if leg is not None and leg.isVisible() and leg.sceneBoundingRect().contains(pos):
                return
            vb = p.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(pos):
                bno = self._breath_at(vb.mapSceneToView(pos).x() - offset)
                if bno is not None:
                    self._toggle_breath(bno)
                return

    def _on_emg_raw_clicked(self, ev):
        self._toggle_from_emg_click(ev, self._emg_raw_subplots, self._trim_offset_s)

    def _on_emg_detail_clicked(self, ev):
        if getattr(self._noise_region, "mouseHovering", False):
            return                                     # let the movable noise region own the click
        self._toggle_from_emg_click(ev, [self.emg_plots.getPlotItem()], self._trim_offset_s)

    def _on_emg_result_clicked(self, ev):
        self._toggle_from_emg_click(ev, [self.emg_result_plots.getPlotItem()], self._trim_offset_s)

    # -- feature B: noise-profile region -----------------------------------
    def _ensure_noise_region(self):
        reg = self._noise_region
        if reg is None:
            ivals = self.state.settings.processing.emg.noise.reference_intervals
            try:                                     # a malformed settings file must not abort construction
                t0, t1 = (float(ivals[0][0]), float(ivals[0][1])) if ivals else (0.0, 1.0)
            except Exception:                        # noqa: BLE001 — any bad shape -> cosmetic default
                t0, t1 = 0.0, 1.0
            # a read-only indicator of the current noise reference span; selection now
            # happens in the 'Set noise profile' modal, so this no longer needs dragging.
            # Brand orange (NOISE_ACCENT, shared with the picker dialog) + a dashed outline
            # and a low fill, not theme.py's old noise_region blue: that token was the SAME
            # hue as breath_incl_brush (only the alpha differed), so a marked rest span
            # painted no differently from the inspiratory shading it sits inside (D07).
            reg = pg.LinearRegionItem(
                values=(t0, t1), movable=False,
                brush=pg.mkBrush(*NOISE_ACCENT, 40),
                pen=pg.mkPen(NOISE_ACCENT, width=1.5, style=Qt.DashLine))
            reg.setZValue(10)
            self._noise_region = reg
        if reg.scene() is None:
            self.emg_plots.addItem(reg)

    def _ensure_noise_label(self):
        """The small 'noise reference' caption at the band's left edge (D07) — without a
        label the dashed band alone doesn't say what it is, and it sits over a channel
        whose own y-axis title is 'EMG (a.u.)', not 'noise'."""
        lbl = self._noise_label
        if lbl is None:
            lbl = pg.TextItem("noise reference", color=pg.mkColor(*NOISE_ACCENT), anchor=(0, 1))
            f = QFont(); f.setPointSizeF(9.0)
            lbl.setFont(f)
            lbl.textItem.document().setDocumentMargin(0)
            self._noise_label = lbl
        if lbl.scene() is None:
            # ignoreBounds: a view-pinned item that still fed childrenBounds would
            # re-inflate every autorange pass, same reasoning as the breath labels.
            self.emg_plots.addItem(lbl, ignoreBounds=True)
        return lbl

    def _refresh_noise_reference_band(self):
        """Make the Detail plot's shaded band say what it actually is: this test's saved
        noise reference, on the file it was taken from — or nothing, when that would be
        misleading. Read from settings + the currently displayed file, so calling this
        after ANY change to either (a new file, a loaded analysis, an accepted picker
        selection, a switch to 'every expiration') is always correct; nothing here needs
        the caller to know which of those happened (D07, UI-overhaul).

        Hidden, not shown at a stale span, whenever there is no interval to point at in
        THIS file: 'every expiration' has no single span, and a file other than
        n.reference_file has no reference of its own — the test's one shared profile was
        built from a different recording entirely."""
        # Drop the previous pin FIRST, same discipline _draw_breath_overlays uses via
        # _mech_unpin: the label is a REUSED widget (not recreated per render), so without
        # this every render would leave its own sigYRangeChanged connection live on top of
        # the last one — each individually harmless (idempotent setPos), but accumulating
        # forever on a screen the user leaves open for a whole session.
        self._noise_label_unpin()
        self._noise_label_unpin = lambda: None
        # The item is ensured UNCONDITIONALLY, before the visibility decision below, so it
        # always exists (a malformed reference_intervals must not abort construction —
        # _ensure_noise_region's own try/except falls back to a cosmetic default) even
        # while hidden. Found in review: this used to be created only on the show=True
        # path, so a fresh PreviewScreen (no file selected yet) or a re-attach after
        # emg_plots.clear() left it None rather than merely invisible.
        self._ensure_noise_region()
        n = self.state.settings.processing.emg.noise
        shown = self._selected_filename()
        ivals = n.reference_intervals
        show = bool(shown and not n.use_expiration and ivals and shown == (n.reference_file or ""))
        t0 = t1 = None
        if show:
            try:
                t0, t1 = float(ivals[0][0]), float(ivals[0][1])
            except Exception:                        # noqa: BLE001 — malformed shape -> nothing to show
                show = False
        if not show:
            self._noise_region.setVisible(False)
            if self._noise_label is not None:
                self._noise_label.setVisible(False)
            return
        self._noise_region.setRegion((t0, t1))
        self._noise_region.setVisible(True)
        lbl = self._ensure_noise_label()
        lbl.setPos(t0, lbl.pos().y())
        lbl.setVisible(True)
        # pinned to the top of the CURRENT view, same mechanism the breath-number labels
        # use, so it survives a y-zoom/pan on the Detail plot instead of drifting off
        self._noise_label_unpin = self._pin_breath_labels(self.emg_plots.getPlotItem(), {"noise_ref": lbl})

    def _apply_noise_expiration(self):
        """Define the shared noise reference as every expiration of the current file, the
        alternative to a hand-marked span. Both halves are written here so the pair can never
        disagree: the core resolves them as `use_expiration or not reference_intervals`, so
        leaving stale intervals behind would make the choice ambiguous on re-open."""
        name = self._selected_filename()
        if not name:
            return None
        n = self.state.settings.processing.emg.noise
        n.reference_file = name
        n.reference_intervals = []
        n.reference_folder = self.state.settings.input.folder
        n.use_expiration = True
        self.noise_reference_changed.emit(name, [], True)
        self._refresh_noise_readout()
        self._refresh_noise_reference_band()          # no single span any more -> hide it
        self._set_status(f"Noise profile ← {name}, built from every expiration. "
                         "Enable 'Reduce EMG noise' to apply it.")
        self._update_actions()
        self._request_autorun()
        return True

    def _apply_noise_reference(self, t0, t1):
        """Apply [t0, t1] s of the current file as the shared noise reference: write it
        into settings, mirror it (signal + the inline detail-plot indicator), and
        re-condition. Shared by the noise-profile modal and the legacy inline region."""
        name = self._selected_filename()
        if not name:
            return None
        t0, t1 = float(min(t0, t1)), float(max(t0, t1))
        if t1 - t0 <= 0:
            self._set_status("The selected noise region has zero width.")
            return None
        n = self.state.settings.processing.emg.noise
        n.reference_file = name
        n.reference_intervals = [[t0, t1]]
        n.reference_folder = self.state.settings.input.folder
        n.use_expiration = False
        fs = self.state.settings.input.format.sampling_frequency or 0
        span = int(round((t1 - t0) * fs))
        nframes = 1 + max(0, span - n.win_length) // max(1, n.hop_length)
        self.noise_reference_changed.emit(name, [[t0, t1]], False)
        self._refresh_noise_readout()
        self._refresh_noise_reference_band()          # reflect the choice on the detail plot
        self._set_status(
            f"Noise profile ← {name} [{t0:.2f}–{t1:.2f} s]: {span} samples ≈ {nframes} STFT frames "
            + ("(good)" if nframes >= 8 else "(short — pick a longer rest span)")
            + ". Enable 'Reduce EMG noise' to apply it.")
        self._update_actions()
        self._request_autorun()          # re-condition against the new reference
        return (t0, t1)

    def _use_region_as_noise(self):
        reg = self._noise_region
        if reg is None or not self._selected_filename():
            self._set_status("Draw a rest region on the detail plot, then apply.")
            return None
        t0, t1 = reg.getRegion()
        return self._apply_noise_reference(t0, t1)

    def _open_noise_profile_dialog(self):
        """Open the modal noise-profile picker over the current file's raw EMG channels;
        on accept, apply the marked span as the shared noise reference."""
        path = self._current_file()
        if not path:
            return
        from respmech.ui.noise_profile_dialog import NoiseProfileDialog
        from respmech.ui.workers import stage_raw_emg
        try:
            data = stage_raw_emg(self.state.settings, path)
        except Exception:                            # noqa: BLE001 — copyable status
            self._set_status(f"Could not load EMG for the noise picker — {short_error(traceback.format_exc())}")
            return
        if not data["raw"]:
            self._set_status("This file has no EMG channels to pick a noise profile from.")
            return
        from respmech.ui.noise_profile_dialog import EXPIRATION
        n = self.state.settings.processing.emg.noise
        dlg = NoiseProfileDialog(data["raw"], data["t"], data["fs"], data["cols"],
                                 parent=self, file_name=self._selected_filename(),
                                 flow=data.get("flow"), reference_file=n.reference_file or "")
        dlg.use_expiration.setChecked(bool(n.use_expiration or not n.reference_intervals))
        # Seed the picker with the reference already saved for this test (D07), so a user
        # who opens the dialog to CHECK what is set — the app's own declared workflow of
        # setting up once and revisiting for many subjects — sees it shaded and can accept
        # as a no-op, instead of an empty picker that only shows something once you drag a
        # new one over it. Skipped for 'every expiration': that mode has no single span to
        # shade, and setChecked(True) above already gives it its own unaffected behaviour.
        if not n.use_expiration and n.reference_intervals:
            try:
                t0, t1 = float(n.reference_intervals[0][0]), float(n.reference_intervals[0][1])
            except Exception:                        # noqa: BLE001 — malformed shape -> nothing to seed
                pass
            else:
                dlg._seed_reference(t0, t1)
        if dlg.exec() == QDialog.Accepted:
            sel = dlg.selected_region()
            if sel is EXPIRATION:
                self._apply_noise_expiration()
            elif sel is not None:
                self._apply_noise_reference(sel[0], sel[1])

    # -- EMG detail / result / test triggers (manual re-runs) --------------
    def _on_emg_channel_changed(self, *_):
        if self._current_file():
            QTimer.singleShot(0, lambda: self._schedule("emg_detail"))

    # -- EMG detail render (single channel) --------------------------------
    def _on_emg_detail_result(self, data):
        self.render_emg_time(data)
        self.render_emg_psd(data)
        col = data.get("col", int(data.get("channel", 0)) + 1)
        if data.get("noise_applied"):
            applied = "noise reduction applied"
        elif data.get("noise_error"):
            applied = "noise conditioning failed — showing ECG-removed"
        else:
            applied = "no noise reference — raw vs ECG-removed"
        note = " · ECG removal failed (showing raw)" if data.get("ecg_error") else ""
        self._set_status(f"EMG detail (col {col}): {applied}{note} (band 20–250 Hz marked).")

    def render_emg_time(self, data):
        t = np.asarray(data["t"])
        pal = _plot_pal()
        ch = pal["channels"]
        self.emg_plots.clear()
        self.emg_plots.plot(t, np.asarray(data["raw"]), pen=_pen(pal["raw_trace"]), name="raw")
        self.emg_plots.plot(t, np.asarray(data["ecg"]), pen=_pen(ch["flow"]), name="ECG-removed")
        final = np.asarray(data["ecg"])
        if data.get("noise_applied"):
            final = np.asarray(data["noise"])
            self.emg_plots.plot(t, final, pen=_pen(ch["volume"]), name="noise-reduced")
        # P13: overlay the RMS envelope the analysis quantifies (± the conditioned signal),
        # so the picked amplitude is visible over the raw trace it is taken from.
        fs = self.state.settings.input.format.sampling_frequency or (1.0 / (t[1] - t[0]) if len(t) > 1 else 1.0)
        env = _rms_envelope(final, (self.state.settings.processing.emg.rms_window_s or 0.05) * fs)
        # the one pen in this plot that was a literal; every other trace here is themed, so
        # in dark mode the envelope alone stayed a light-palette red. The light table's
        # 'poes' IS (180, 50, 42), so light mode is unchanged.
        env_pen = pg.mkPen(pal["channels"]["poes"], width=2)
        self.emg_plots.plot(t, env, pen=env_pen, name="RMS envelope")
        self.emg_plots.plot(t, -env, pen=env_pen)
        # discrete full-length flow silhouette behind the EMG, to read bursts against respiration
        add_flow_background(self.emg_plots.getPlotItem(), t, data.get("flow"), pal)
        # y label / x caption are set once in _build_noise_tab; the axis re-picks the label
        # wording itself on every resize, so re-setting it here would undo that choice.
        # the key rides in the title band, not over the trace; rebuilt per render because
        # 'noise-reduced' is only there when noise conditioning actually ran
        stages = [("raw", pal["raw_trace"]), ("ECG-removed", ch["flow"])]
        if data.get("noise_applied"):
            stages.append(("noise-reduced", ch["volume"]))
        stages.append(("RMS envelope", pal["channels"]["poes"]))
        self._set_trace_key(stages)
        self._limit_x(self.emg_plots.getPlotItem(), t)
        self._refresh_noise_reference_band()          # emg_plots.clear() above dropped it
        self._detail_label_y = self._safe_top(
            np.asarray(data["raw"]), np.asarray(data["ecg"]),
            np.asarray(data["noise"]) if data.get("noise_applied") else None)
        self._repaint_view_breaths("detail")

    def render_emg_psd(self, data):
        freqs = np.asarray(data["freqs"], dtype=float)
        band = data.get("band", (20.0, 250.0))
        pal = _plot_pal()
        fig = self.emg_psd_canvas.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])

        def _db(p):
            return 10.0 * np.log10(np.maximum(np.asarray(p, dtype=float), 1e-30))

        ax.plot(freqs, _db(data["psd_raw"]), color=pal["mpl_muted"], lw=1.0, label="raw")
        ax.plot(freqs, _db(data["psd_ecg"]), color=pal["mpl_accent"], lw=1.2, label="ECG-removed")
        if data.get("noise_applied"):
            ax.plot(freqs, _db(data["psd_noise"]), color=pal["mpl_ok"], lw=1.2, label="noise-reduced")
        ax.axvspan(band[0], band[1], color=pal["mpl_warn"], alpha=0.12, label="EMG band 20–250 Hz")
        ax.set_xlim(0, min(500.0, float(freqs[-1]) if len(freqs) else 500.0))
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB)")
        # The channel goes in the PANEL HEADER, not in the figure: a figure title has to fit
        # the canvas, and whether it fits changes with every splitter drag — the header is a
        # real label with room, and it never has to be re-decided on resize.
        col = data.get("col", int(data.get("channel", 0)) + 1)
        try:
            # setFullText, not setText: the title label is an ElidingLabel now (A03) — a raw
            # setText would be overwritten by the OLD full text on the next resizeEvent.
            self._psd_panel._title_label.setFullText(f"Detail PSD — col {col}")
        except Exception:                   # pragma: no cover - header is cosmetic
            pass
        _fit_compact_figure(self.emg_psd_canvas, ax, legend_kw={"fontsize": 7})
        self.emg_psd_canvas.draw()

    # -- result view (all channels, pick which to show) --------------------
    def _on_emg_all_result(self, data):
        self._emg_all = data
        self._render_emg_result()
        # ecg_applied and noise_applied are independent switches (either can be on with the
        # other off) — composed here rather than hard-coded, so this label cannot claim
        # "(ECG + noise)" while ECG removal is actually off (it did, before this fix: the
        # noise_applied branch never checked ecg_applied at all).
        ecg, noise = bool(data.get("ecg_applied")), bool(data.get("noise_applied"))
        if ecg and noise:
            applied = "conditioned (ECG + noise)"
        elif ecg:
            applied = "ECG-removed"
        elif noise:
            applied = "noise-reduced"
        else:
            applied = "raw"
        notes = []
        if data.get("ecg_error"):
            notes.append("ECG removal failed")
        if data.get("noise_error"):
            notes.append("noise conditioning failed")
        note = (" · " + ", ".join(notes)) if notes else ""
        self._set_status(f"EMG result: {applied}{note}. Tick channels above to show/hide them.")

    def _render_emg_result(self, *_):
        """Overlay the conditioned result for the ticked channels."""
        data = self._emg_all
        self.emg_result_plots.clear()
        if not data:
            self._result_label_y = None
            self._bov.get("result", {}).get("unpin", lambda: None)()
            self._bov["result"] = {"items": [], "regions": {}, "texts": {}, "unpin": lambda: None}
            return
        t = np.asarray(data["t"])
        cols = data.get("cols", [])
        cycle = _plot_pal()["emg_cycle"]
        checked = {c for c, cb in self.result_checks if cb.isChecked()}
        for i, c in enumerate(cols):
            if c not in checked or i >= len(data["conditioned"]):
                continue
            self.emg_result_plots.plot(t, np.asarray(data["conditioned"][i]),
                                       pen=_pen(cycle[i % len(cycle)]), name=f"col {c}")
        # discrete full-length flow silhouette behind the ticked EMG channels
        add_flow_background(self.emg_result_plots.getPlotItem(), t, data.get("flow"), _plot_pal())
        # no legend call: this plot has none. Its corner picker names every channel against
        # the very colour it is drawn in (same emg_cycle), so an in-plot legend duplicated
        # the key AND covered the trace.
        self._limit_x(self.emg_result_plots.getPlotItem(), t)
        self._result_label_y = self._safe_top(*[np.asarray(c) for c in data.get("conditioned", [])])
        self._repaint_view_breaths("result")

    # -- fidelity render ----------------------------------------------------
    def _draw_fidelity(self, nr):
        """Draw the EMG noise fidelity frontier from a noise ``report`` dict onto the
        fidelity canvas. Shared by the auto noise job and the manual test run."""
        frontier = nr.get("frontier", {})
        chosen = nr.get("prop_decrease")
        target = nr.get("fidelity_target", 0.8)
        emg_cols = list(self.state.settings.input.channels.emg)
        pal = _plot_pal()
        fig = self.fidelity_canvas.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])
        if frontier:
            props = sorted(frontier)
            nch = len(next(iter(frontier.values())))
            for ch in range(nch):
                lbl = f"EMG col {emg_cols[ch]}" if ch < len(emg_cols) else f"EMG ch {ch + 1}"
                ax.plot(props, [frontier[p][ch] for p in props], marker="o", ms=3, label=lbl)
            ax.axhline(target, color=pal["mpl_target"], ls=":", lw=1)
            # Named IN THE PLOT, not the legend: the legend already carries one entry per EMG
            # channel (unbounded — this same file's _build_emg_tab has measured a real
            # 12-channel case) plus "chosen = …", and _fit_compact_figure sheds the WHOLE
            # legend first when the panel is short — one more entry would be the first thing
            # to disappear exactly when the panel is tightest for room.
            # Anchored to whichever edge "chosen" is NOT near: select_prop_decrease sweeps
            # upward and keeps the highest prop_decrease that still clears the target, so a
            # right-edge chosen (an easy target, aggressively suppressed) is a common outcome,
            # not a rare one — anchoring the target label at a fixed right edge would then
            # collide with the "chosen" line and its own axvline most of the time.
            target_x, target_ha = 0.99, "right"
            if chosen is not None and len(props) > 1 and props[-1] != props[0]:
                chosen_frac = (chosen - props[0]) / (props[-1] - props[0])
                if chosen_frac > 0.75:
                    target_x, target_ha = 0.02, "left"
            ax.text(target_x, target, f"target {target:.2f}", transform=ax.get_yaxis_transform(),
                    ha=target_ha, va="bottom", fontsize=7, color=pal["mpl_target"], clip_on=False)
            if chosen is not None:
                ax.axvline(chosen, color=pal["mpl_error"], ls="--", lw=1.2, label=f"chosen = {chosen}")
            # Longest wording that fits; the panel sits in a splitter, so the room changes.
            # 'prop_decrease' survives every rung — it is what the Advanced modal calls this
            # setting — while "Noise-" (the panel header already says it) and the 0–1 range
            # (the ticks show it) are what go first.
            xlabels = ("Noise-suppression strength (prop_decrease, 0–1)",
                       "Suppression strength (prop_decrease)",
                       "Suppression (prop_decrease)",
                       "prop_decrease")
            # Rotated, the y-label's length runs along the axes HEIGHT (~206px in a 300px
            # panel), so a wordy one overflows. The panel header carries the title and the
            # "1 = untouched" scale hint horizontally, where there is room for them.
            ax.set_ylabel("Fidelity retained")
            # core.noise.fidelity() is an UNBOUNDED ratio (in-band inspiratory power after
            # noise removal, divided by before): NoiseProfile.apply masks magnitude and the
            # imaginary part separately, so the reconstruction routinely holds MORE in-band
            # power than the original — measured 1.281-1.340 across all 30 points of the
            # bundled sample data's frontier. A fixed (0, 1.02) clamp drew every one of those
            # points above the top of the axes: no visible curves at all, which reads as total
            # signal loss rather than the routine over-1 case it actually is. Derive the
            # ceiling from what is actually being drawn (and the target line) instead of
            # assuming the data always sits under 1.
            all_vals = [v for fids in frontier.values() for v in fids]
            # Guarded, not assumed non-empty: nothing today calls this with a channel-less
            # frontier (screen.py only requests the noise job with EMG channels selected,
            # and stage_noise_fidelity itself requires them), but max() on an empty sequence
            # is a hard crash, and this function has no other reason to depend on that.
            # target's own margin is wider than the data's (1.08 vs 1.05): it also has to
            # leave room for the "target 0.XX" text sitting just above its line, at the
            # smallest panel height this figure ever gets (PLOT_DIAG_MIN_HEIGHT).
            hi = max(1.02, target * 1.08, 1.05 * max(all_vals) if all_vals else 0.0)
            ax.set_ylim(0, hi)
            if hi > 1.05:
                # Only worth a reference line once the axis genuinely extends past
                # "untouched" — on the normal (0, 1.02) axis it would sit indistinguishably
                # close to the top tick and just add clutter.
                ax.axhline(1.0, color=pal["mpl_zeroline"], lw=0.8, zorder=0)
        _fit_compact_figure(self.fidelity_canvas, ax,
                            legend_kw={"fontsize": 7} if frontier else None,
                            xlabel_variants=xlabels if frontier else None)
        # settle the wording against the layout this render produces, not the previous one
        refit_compact_figure(self.fidelity_canvas)

    def _set_fidelity_panel_title(self, chosen, worst=None, target=None):
        """Keep the fidelity panel's own header naming the chosen suppression and the
        worst-channel fidelity against target — a persistent verdict, unlike the status-bar
        message the two callers below also set, which is gone within a second or so.
        ``chosen=None`` resets it to the bare base title, used when the panel is blanked for
        a file switch (see screen.py's _clear_*_panels)."""
        label = getattr(self._fidelity_panel, "_title_label", None)
        if label is None:
            return
        if chosen is None:
            label.setFullText(_FIDELITY_TITLE)
        else:
            label.setFullText(f"{_FIDELITY_TITLE}  ·  prop {chosen} · worst-channel fidelity "
                              f"{worst:.2f} (target {target:.2f})")
        # setFullText() just reset the tooltip to the (possibly elided) visible title alone —
        # append the definition every time, so it survives every verdict update rather than
        # only surviving until the first one.
        label.setToolTip(label.toolTip() + _FIDELITY_TOOLTIP_SUFFIX)

    def render_noise_report(self, result):
        """Draw the fidelity frontier + status from a full batch result (with ECG
        suppression from its ok_files). Used by the manual test run."""
        nr = result.noise_report
        self._draw_fidelity(nr)
        chosen = nr.get("prop_decrease")
        target = nr.get("fidelity_target", 0.8)
        worst = min((r["fidelity"] for r in nr.get("channels", [])), default=float("nan"))
        self._set_fidelity_panel_title(chosen, worst, target)
        msg = (f"Noise: prop_decrease {chosen} chosen "
               f"(worst-channel fidelity {worst:.2f} vs target {target:.2f})")
        for fr in result.ok_files.values():
            if getattr(fr, "ecg", None):
                msg += f" · ECG suppression {fr.ecg.get('suppression', float('nan')):.0%}"
                break
        self._set_status(msg)

    # -- auto noise/fidelity job (decoupled from the test run) --------------
    def _on_noise_result(self, report):
        # A user-fixable precondition surfaced by stage_noise_fidelity as {"error": msg} (e.g. a
        # misassigned flow channel, so the reference has no segmentable breaths). Raise FIRST so
        # _on_job_done paints a clean 'Noise fidelity failed' card with the message — otherwise the
        # error dict would render a silently blank frontier, mark the panel as done, and even
        # re-dispatch the EMG views, hiding the failure and suppressing the auto-retry.
        if isinstance(report, dict) and report.get("error"):
            raise _FileRunError(report["error"])
        self._apply_noise_report(report)

    def _apply_noise_report(self, nr):
        """Render the fidelity frontier, adopt the auto-selected suppression (so the
        spinbox + re-conditioned EMG use the value the sweep chose — the core never
        mutates settings), and re-dispatch the EMG conditioning views with it."""
        self._draw_fidelity(nr)
        chosen = nr.get("prop_decrease")
        noise = self.state.settings.processing.emg.noise
        if chosen is not None and noise.auto_prop:
            noise.prop_decrease = float(chosen)
            # This is an APP-driven change, not a user edit — fold it into the diff baseline
            # so the next user edit's dependency-scope diff does not mistake it for a change
            # and needlessly recompute the EMG/noise panels.
            if self._last_synced_settings is not None:
                self._last_synced_settings.processing.emg.noise.prop_decrease = float(chosen)
        self._load_noise_params()                    # reflect the chosen prop in the spinbox
        if nr.get("frontier"):
            target = nr.get("fidelity_target", 0.8)
            worst = min((r["fidelity"] for r in nr.get("channels", [])), default=float("nan"))
            self._set_fidelity_panel_title(chosen, worst, target)
            self._set_status(f"Noise: prop_decrease {chosen} chosen "
                             f"(worst-channel fidelity {worst:.2f} vs target {target:.2f}).")
        if noise.enabled and noise.auto_prop and self.state.settings.input.channels.emg:
            self._schedule("emg_all")                # re-condition with the chosen suppression
            self._schedule("emg_detail")

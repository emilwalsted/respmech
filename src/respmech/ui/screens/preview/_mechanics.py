"""PreviewScreen's Mechanics sub-tab: the channel/flow-volume-pressure stack,
the Campbell diagram and per-breath table, and the click-to-exclude breath
overlays. Split out of preview_screen.py (ticket A02); moved verbatim."""

from __future__ import annotations

import copy
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTableView,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QObject, QSize, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QFont, QFontMetrics

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui import plot_perf
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers
from respmech.ui import wheel as _wheel
from respmech.ui.flow_layout import (ElidingLabel, FlowLayout, cluster as _cluster,
                                     elide as _elide, install_flow as _install_flow)
from respmech.ui.result_table import (ResultTableModel, configure_result_table,
                                      resize_result_table)
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

from ._figure_fit import _CompactFigureFitter, _fit_compact_figure
from ._jobs import _FileRunError, _PANELS
from ._plot_helpers import SciAxis, _CHANNELS, _pen, _plot_pal, _restrict_body_wheel_to_x



def _parse_breath_counts(text, entry_cls, folder=None):
    """Parse the 'filename = count' lines of the breath-count override box into entries.
    Splits on the LAST '=' so a filename may itself contain one; skips blank/malformed lines.

    ``folder`` (the CURRENT ``input.folder``) is stamped onto every entry the box
    produces. This dialog holds one flat list — the file the box currently names, current
    input folder included — so every line it commits belongs to that folder by
    construction; an untouched OK never gets here at all (see the caller, which only
    re-parses when the user actually edited this field), so this cannot relabel an entry
    the user never looked at."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        fpart, _, cpart = line.rpartition("=")
        try:
            cnt = int(cpart.strip())
        except ValueError:
            continue
        if fpart.strip():
            out.append(entry_cls(fpart.strip(), cnt, folder))
    return out

# Per-file errors that mean "this recording cannot support this step", not "something
# broke". The mech preview lands them softly and explains them in the status line, so the
# batch panel must not also paint a hard 'Test run failed' card over the same thing.
_SOFT_FILE_ERRORS = ("TrimError", "VolumeTrendError", "NoBreathsError")

# The Mechanics-advanced fields that change the volume the trend detector sees. The live
# trough count is only valid while these still match the rendered preview it was taken
# from (see _trend_hint).
_TREND_PROBE_KEYS = ("integrate_from_flow", "correct_drift", "inverse_flow",
                     "inverse_volume", "resample", "resample_to_frequency")

# Campbell diagram axis-label ladders (D12/UI-overhaul). The volume axis must ALWAYS
# name its datum — EELV or "end-expiration" — never bottom out at the ambiguous
# "Volume (L)", which on a Campbell diagram reads as absolute lung volume, a different
# quantity. Poes gets its own ladder now that the axis swap below moves it onto the
# height-constrained axis the volume label used to occupy.
_CAMPBELL_XLABEL_VARIANTS = ("Lung volume above end-expiration (L)",
                            "Volume above EELV (L)", "V−EELV (L)")
_CAMPBELL_YLABEL_VARIANTS = ("Oesophageal pressure  Poes (cmH₂O)", "Poes (cmH₂O)", "Poes")


class _MechanicsMixin:

    def _build_mech_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)
        # Cursor read-out (time, value at the pointer) — above the graphs, right-aligned, and
        # never wrapping: it updates on every mouse move, so a wrapping label made the whole
        # stack jump a row up and down. A fixed single line on top stays put.
        self.crosshair_label = QLabel("")
        self.crosshair_label.setProperty("status", "muted")
        self.crosshair_label.setWordWrap(False)
        self.crosshair_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # All the mechanics settings live behind here now (they left the Setup screen); this
        # is the tab they shape, so it is where they belong.
        self.btn_mech_advanced = QPushButton("Advanced… (breath detection, volume, WOB)")
        self.btn_mech_advanced.setProperty("compact", True)
        self.btn_mech_advanced.setToolTip(
            "Breath segmentation, work-of-breathing source, volume/drift corrections, "
            "resampling and per-file breath-count overrides.")
        self.btn_mech_advanced.clicked.connect(self._open_mech_advanced)
        # A persistent one-line caption — breath count, exclusion count, the click-to-
        # exclude instruction — kept current by _render_preview and _toggle_breath (see
        # _set_mech_caption). It used to live only in the transient status bar, where an
        # EMG job finishing a second later would silently erase it.
        self.mech_caption = ElidingLabel("")
        self.mech_caption.setProperty("status", "muted")
        _xh = QHBoxLayout(); _xh.setContentsMargins(6, 0, 6, 0)
        _xh.addWidget(self.btn_mech_advanced)
        _xh.addSpacing(10)
        _xh.addWidget(self.mech_caption, 1)
        _xh.addWidget(self.crosshair_label, 0, Qt.AlignRight)
        v.addLayout(_xh)
        split = QSplitter(Qt.Vertical)
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setAccessibleName("Mechanics signals")   # C02: named for QAccessible/screen readers
        _theme.set_plot_floor(self.plots)
        self.plots.setBackground(_plot_pal()["bg"])
        self.plots.scene().sigMouseClicked.connect(self._on_plot_clicked)
        # P17: a crosshair that tracks the cursor across the X-linked channel stack and
        # reads out (time, value); built once, the per-channel lines are (re)made per render.
        self._crosshair_lines = []
        self._crosshair_proxy = pg.SignalProxy(
            self.plots.scene().sigMouseMoved, rateLimit=60, slot=self._on_mech_mouse_moved)
        split.addWidget(self.plots)
        lower = QSplitter(Qt.Horizontal)
        self.table = QTableView()
        self.table.setAccessibleName("Per-breath results")   # C02: named for QAccessible/screen readers
        self._table_model = ResultTableModel()
        self.table.setModel(self._table_model)
        # breath_no is already a table COLUMN; the row header numbered rows 1..N
        # regardless, which after an exclusion disagreed with breath_no's own
        # numbering (1, 3, 4, 5 …) — hide the row header rather than carry two
        # numbering schemes that can say different things about the same row.
        self.table.verticalHeader().setVisible(False)
        configure_result_table(self.table)
        # a table squeezed to one visible row is as useless as a flattened graph
        _theme.set_plot_floor(self.table)
        lower.addWidget(self.table)
        self.campbell = FigureCanvasQTAgg(Figure(figsize=(4, 4)))
        self.campbell.setAccessibleName("Campbell diagram")   # C02: named for QAccessible/screen readers
        # A matplotlib canvas reports a 10 px minimum, so it was the one Preview graph with
        # no floor at all — measured 90 px on the target screen and 70 px at the window's own
        # minimum, which for the panel a reader takes the recoil line and work fill off is a
        # box, not a diagram.
        _theme.set_plot_floor(self.campbell)
        # A titled panel, not a bare canvas. It used to butt flush against the table and the
        # panel edge (Emil, 02-08-2026), which a hand-rolled box with _titled's margins fixed
        # — but the figure's own title was still the only thing naming the panel, and at the
        # height this panel actually gets (130 px measured) matplotlib cannot fit a title:
        # it was drawn with its top 10 px outside the figure, so it read as a half-cut
        # "Campbell diagram". The header names it at any height and costs the same room the
        # clipped title did; the figure keeps its title only for the export (_export_campbell).
        self._campbell_fitter = _CompactFigureFitter(self.campbell)
        lower.addWidget(self._titled("Campbell diagram", self.campbell))
        # D12: the diagram is what the user came for, not the number dump beside it — at
        # 3:1 (720/240) a maximised window gave the table 1265 px and the diagram 406x249,
        # a 3x weight difference in favour of the numbers. ~58/42 (matches the stretch
        # factor below) gives the diagram room to be legible without starving the table.
        lower.setStretchFactor(0, 7); lower.setStretchFactor(1, 5)
        lower.setSizes([560, 400])
        split.addWidget(lower)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        v.addWidget(split)
        # P16/P17: a thin action bar — batch QC overview + crosshair read-out + export
        bar = QHBoxLayout()
        self.qc_overview = QLabel("")
        self.qc_overview.setProperty("banner", True)   # box baked at first polish (theme.py)
        self.qc_overview.setProperty("status", "muted")
        self.qc_overview.setToolTip(
            "Quality overview of the CURRENTLY PREVIEWED file's most recent test run — "
            "not a batch summary. See the file rail for every file's exclusion count.")
        self.btn_export_fig = QPushButton("Export Campbell…")
        self.btn_export_fig.setEnabled(False)          # enabled once a diagram is drawn
        self.btn_export_fig.setToolTip("Save the Campbell diagram as a PNG or PDF.")
        self.btn_export_fig.clicked.connect(self._export_campbell)
        # P19: process AND write just this file, so a tuned file can be produced without
        # re-running the whole batch (reuses the Run screen's write machinery).
        self.btn_process_file = QPushButton("Process && write this file")
        self.btn_process_file.setEnabled(False)
        self.btn_process_file.setToolTip("Run and write output for the previewed file only.")
        self.btn_process_file.clicked.connect(self._process_this_file)
        bar.addWidget(self.qc_overview); bar.addStretch(1)
        bar.addWidget(self.btn_process_file); bar.addWidget(self.btn_export_fig)
        v.addLayout(bar)
        return w

    def _process_this_file(self):
        name = self._previewed_file or self._selected_filename()
        if name:
            self.process_file_requested.emit(os.path.basename(name))

    def _reset_qc_overview(self):
        """Neutral QC chip + disabled 'Process & write' — the same nulstillingsdiscipline
        ``_forget_campbell`` already applies to the export button, extended to this chip
        and this button: clearing what a panel shows and clearing the widget that judges
        it must be the same act. Called from ``_clear_file_panels`` (a file switch/blank);
        the mechanics render (``_render_preview``) is what re-enables the button for a
        file that actually loaded."""
        self.qc_overview.setText("QC:  —")
        self.qc_overview.setProperty("status", "muted")
        self.qc_overview.style().unpolish(self.qc_overview)
        self.qc_overview.style().polish(self.qc_overview)
        self.btn_process_file.setEnabled(False)

    def _qc_overview_not_assessed(self, detail):
        """The chip's honest state while the test run itself failed or was skipped —
        called from ``_on_batch_result``'s error branches and from ``_on_job_done``'s
        'batch' failure branch. Purely presentational: no computed value changes."""
        self.qc_overview.setText(f"QC:  not assessed — {short_error(str(detail))}")
        self.qc_overview.setProperty("status", "warn")
        self.qc_overview.style().unpolish(self.qc_overview)
        self.qc_overview.style().polish(self.qc_overview)

    def _update_qc_overview(self, fr):
        """P16: a persistent, at-a-glance quality summary of the current test run —
        breaths used/excluded plus a conservative flag for any non-physiological
        per-breath value (so a suspect run is obvious without reading the table)."""
        bt = getattr(fr, "breaths_table", None)
        total = len(fr.breaths) if getattr(fr, "breaths", None) else (len(bt) if bt is not None else 0)
        used = len(bt) if bt is not None else 0
        excl = max(0, total - used)
        flags = []
        if bt is not None and len(bt):
            for c in ("vt", "ti", "te", "ttot"):
                if c in bt.columns and (bt[c] <= 0).any():
                    flags.append(f"{c}≤0")
            for c in ("wobtotal", "vt", "ve"):          # core metrics must be finite
                if c in bt.columns and not np.isfinite(bt[c].to_numpy(dtype=float)).all():
                    flags.append(f"{c} NaN")
        excl_note = ""
        # fr.file, NOT self._selected_filename(): _on_batch_result's own fallback
        # (`result.files.get(cur) or next(iter(result.files.values()), None)`) can hand this
        # a DIFFERENT file's FileResult than the one currently selected (e.g. a stale batch
        # result racing a file switch, or cur missing from the result set) — checking the
        # selected name against fr's own exclusion count would then blame/clear the wrong
        # file. fr.file is a real FileResult field; getattr defends the same "fr may be a
        # plain SimpleNamespace" case _qc_overview's other fields already guard against.
        if excl and self._exclusion_carried_for(getattr(fr, "file", None)):
            excl_note = " (carried over from a previous recordings folder)"
        msg = f"QC:  {used} breaths used" + (f",  {excl} excluded{excl_note}" if excl else "")
        status = "ok"
        if flags:
            msg += "   ⚠ " + "; ".join(dict.fromkeys(flags))
            status = "warn"
        else:
            msg += "   ·  no flags"
        self.qc_overview.setText(msg)
        self.qc_overview.setProperty("status", status)
        self.qc_overview.style().unpolish(self.qc_overview)
        self.qc_overview.style().polish(self.qc_overview)

    # -- P17 crosshair + figure export -------------------------------------
    def _on_mech_mouse_moved(self, evt):
        if not self._channel_plots or not self._crosshair_lines:
            return
        pos = evt[0]
        for p, ln in zip(self._channel_plots, self._crosshair_lines):
            if p.sceneBoundingRect().contains(pos):
                mp = p.getViewBox().mapSceneToView(pos)
                for l in self._crosshair_lines:
                    l.setPos(mp.x()); l.show()
                axis = p.getAxis("left")
                lab = axis.labelText or "value"
                # only a named axis has a real unit to report; "value" is the no-name fallback
                unit = getattr(axis, "_unit", "") if lab != "value" else ""
                suffix = f" {unit}" if unit else ""
                self.crosshair_label.setText(f"t = {mp.x():.3f} s     {lab} = {mp.y():.3g}{suffix}")
                return

    def _export_campbell(self):
        from PySide6.QtWidgets import QFileDialog       # noqa: PLC0415
        base = (self._previewed_file or "campbell").rsplit(".", 1)[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Campbell diagram", f"{base} – Campbell.png",
            "PNG image (*.png);;PDF document (*.pdf)")
        if not path:
            return
        # Export LIGHT, whatever the app is wearing. The on-screen figure follows the theme,
        # so in dark mode saving the live facecolor produced a near-black PNG/PDF — which is
        # a figure destined for a report or a paper. The batch writers already pin their
        # output light (core/plot_style) for the same reason; this is the interactive export
        # catching up. Re-rendered rather than recoloured: the dark palette's brightened
        # traces score barely 2:1 on white, so only a genuine light redraw is legible.
        breaths = getattr(self, "_campbell_breaths", None)
        redrawn = False
        try:
            if breaths is not None and _theme is not None and _theme.is_dark():
                # set BEFORE the call it guards: _draw_campbell begins by clearing the
                # figure, so a failure part-way through still leaves the on-screen diagram
                # destroyed and the finally-branch below is the only thing that puts it back
                redrawn = True
                self._draw_campbell(breaths, pal=_theme._PLOT_LIGHT)
            # The export is a stand-alone figure with no panel header around it, so it gets
            # the title back that the on-screen panel leaves to its header — and then loses
            # it again straight away. Leaving it set showed the title twice on screen, once
            # in the header and once in the figure, on every light-mode export (dark mode
            # hid the leak, because it redraws the figure in the finally-branch below).
            fig = self.campbell.figure
            try:
                for _ax in fig.axes:
                    _ax.set_title("Campbell diagram")
                fig.savefig(path, dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            finally:
                for _ax in fig.axes:
                    _ax.set_title("")
                self.campbell.draw_idle()
            self._set_status(f"Saved Campbell diagram → {path}")
        except Exception as e:                          # noqa: BLE001
            self._set_status(f"Could not save figure: {short_error(str(e))}")
        finally:
            if redrawn:                                 # put the on-screen figure back
                try:
                    self._draw_campbell(breaths)
                except Exception:                       # pragma: no cover - defensive
                    pass

    def _open_mech_advanced(self):
        """Every mechanics setting: breath segmentation, work-of-breathing source, the volume
        and drift corrections, resampling, and the per-file breath counts. They left the Setup
        screen and live here, beside the Mechanics preview they shape, editing the model
        directly (like the ECG params) so Setup never reverts them on a tab change."""
        from respmech.core.settings import BreathCountEntry
        from respmech.ui.advanced_dialog import AdvancedDialog, Field, apply_values
        s = self.state.settings
        seg, peak = s.processing.segmentation, s.processing.segmentation.peak
        vol, samp, wob, ptp = (s.processing.volume, s.processing.sampling,
                               s.processing.wob, s.processing.ptp)
        fields = [
            ("seg", Field("method", "Signal used to split breaths", "choice",
                          "processing.segmentation.method",
                          "Which signal marks where each breath begins and ends.",
                          options=[("Flow", "flow"), ("Volume", "volume")])),
            ("wob", Field("calc_from", "Work of breathing from", "choice",
                          "processing.wob.calc_from",
                          "One averaged breath (default), or each breath then averaged.",
                          options=[("Average", "average"), ("Individual", "individual")])),
            ("vol", Field("integrate_from_flow", "Calculate volume from flow", "bool",
                          "processing.volume.integrate_from_flow",
                          "Derive volume by integrating flow instead of a separate channel.")),
            ("vol", Field("correct_drift", "Correct volume drift", "bool",
                          "processing.volume.correct_drift",
                          "Remove slow baseline drift from the volume trace; ON by default.")),
            ("vol", Field("correct_trend", "Correct end-expiratory trend", "bool",
                          "processing.volume.correct_trend",
                          "Remove a between-breath trend in end-expiratory lung volume.")),
            ("vol", Field("trend_method", "Trend interpolation", "choice",
                          "processing.volume.trend_method",
                          "How the end-expiratory trend is interpolated between breaths.",
                          options=[("Linear", "linear"), ("Nearest", "nearest"),
                                   ("Cubic", "cubic"), ("Quadratic", "quadratic"),
                                   ("Previous", "previous"), ("Next", "next")],
                          depends_on="correct_trend")),
            ("vol", Field("trend_peak_min_prominence_frac",
                          "Trend anchor — minimum breath depth", "float",
                          "processing.volume.trend_peak_min_prominence_frac",
                          "How deep a trough must be, as a fraction of the recording's own "
                          "volume range, to count as end-expiratory.",
                          lo=0.001, hi=0.999, step=0.01, decimals=3,
                          depends_on="correct_trend",
                          note="Scales with each recording, so it works at any tidal volume "
                               "and in any volume unit. Lower it only if breaths are missed "
                               "on a recording that also contains a large manoeuvre.")),
            ("vol", Field("trend_peak_min_distance_s", "Trend anchor — minimum spacing",
                          "float", "processing.volume.trend_peak_min_distance_s",
                          "Minimum time between two detected end-expiratory troughs.",
                          lo=0.001, hi=60.0, step=0.05, decimals=4, suffix=" s",
                          depends_on="correct_trend")),
            ("vol", Field("trend_peak_min_height",
                          "Trend anchor — absolute threshold (legacy)", "float",
                          "processing.volume.trend_peak_min_height",
                          "Fixed depth below the recording's HIGHEST volume that a trough "
                          "must reach. Overrides the breath-depth rule above. Set to Auto "
                          "(the lowest value) to use 'Trend anchor — minimum breath depth' "
                          "instead, which is the default.",
                          # the sentinel sits just BELOW zero because 0 is itself a legal
                          # legacy value ("no absolute gate"); sharing the minimum would
                          # silently rewrite such an analysis to Auto on the next OK. With
                          # decimals=3 the only reachable negative IS the sentinel.
                          lo=-0.001, hi=1_000_000.0, step=0.05, decimals=3,
                          # Just "Auto": the old "Auto — use minimum breath depth" is a
                          # sentence inside a numeric field and needed 473 px on Windows
                          # font metrics against the ~340 px a column can give it, so it
                          # was permanently clipped mid-word. What it said now lives in the
                          # tooltip above, which has room for it.
                          auto_text="Auto",
                          depends_on="correct_trend",
                          note="Only for reproducing an analysis made before v2.3.3. It is "
                               "measured against the whole recording, so a value larger than "
                               "the recording's volume range finds no troughs at all.")),
            ("vol", Field("inverse_flow", "Invert the flow signal", "bool",
                          "processing.volume.inverse_flow",
                          "Flip the flow sign if inspiration reads positive.")),
            ("vol", Field("inverse_volume", "Invert the volume signal", "bool",
                          "processing.volume.inverse_volume", "Flip the volume sign.")),
            ("samp", Field("resample", "Resample before analysis", "bool",
                           "processing.sampling.resample",
                           "Resample every recording to a common rate first; off by default.")),
            ("samp", Field("resample_to_frequency", "Resample to", "int",
                           "processing.sampling.resample_to_frequency",
                           "Target rate (Hz). Keep ≥ ~1000 Hz with EMG channels present.",
                           lo=1, hi=1_000_000, step=100, suffix=" Hz",
                           depends_on="resample")),
            ("seg", Field("buffer", "Breath-separation buffer", "int",
                          "processing.segmentation.buffer",
                          "Guard samples added around each detected breath boundary.",
                          lo=0, hi=100_000, step=10)),
            ("peak", Field("height", "Breath peak — minimum height", "float",
                           "processing.segmentation.peak.height",
                           "Minimum peak height for breath detection (volume-based).",
                           lo=0.0, hi=1_000_000.0, step=0.01, decimals=4)),
            ("peak", Field("distance_s", "Breath peak — minimum distance", "float",
                           "processing.segmentation.peak.distance_s",
                           "Minimum time between detected breath peaks.",
                           lo=0.0, hi=60.0, step=0.05, decimals=4, suffix=" s")),
            ("peak", Field("width_s", "Breath peak — minimum width", "float",
                           "processing.segmentation.peak.width_s",
                           "Minimum width of a detected breath peak.",
                           lo=0.0, hi=60.0, step=0.05, decimals=4, suffix=" s")),
            ("wob", Field("avg_resampling_obs", "Average-breath resampling points", "int",
                          "processing.wob.avg_resampling_obs",
                          "Points each breath is resampled to for the average breath / WOB.",
                          lo=10, hi=100_000, step=10)),
            ("ptp", Field("baseline_window_s", "PTP baseline window", "float",
                          "processing.ptp.baseline_window_s",
                          "End-expiratory window whose mean is the PTP baseline.",
                          lo=0.0, hi=1.0, step=0.01, decimals=4, suffix=" s")),
        ]
        owner = {"seg": seg, "peak": peak, "vol": vol, "samp": samp, "wob": wob, "ptp": ptp}
        values = {f.key: getattr(owner[grp], f.key) for grp, f in fields}
        # breath counts round-trip as one 'file = count' line each, edited as text
        bc_field = Field("breath_counts", "Breath-count overrides", "text",
                         "processing.breath_counts",
                         "Per-file override of the breath count used for per-minute scaling — "
                         "one 'filename = count' per line. Blank = each file's detected count.",
                         placeholder="one 'filename = count' per line",
                         note="Excluded breaths are NOT set here — click a breath in the "
                              "channel plots above to include/exclude it; the file rail "
                              "shows each file's exclusion count.")
        bc_text = "\n".join(f"{e.file} = {e.count}" for e in s.processing.breath_counts)
        # Which card each setting is shown on, in display order. Grouping only: the flat
        # ``fields`` list above still drives building, reading and committing, so a setting
        # cannot change meaning by being moved between cards. A key that gates others
        # (``correct_trend``, ``resample``) must come first WITHIN its own card, because
        # ``depends_on`` resolves against widgets already built.
        sections = [
            ("Breath detection", ["method", "buffer", "height", "distance_s", "width_s"]),
            ("Work of breathing", ["calc_from", "avg_resampling_obs"]),
            ("Volume", ["integrate_from_flow", "correct_drift",
                        "inverse_flow", "inverse_volume"]),
            ("End-expiratory trend", ["correct_trend", "trend_method",
                                      "trend_peak_min_prominence_frac",
                                      "trend_peak_min_distance_s", "trend_peak_min_height"]),
            ("Sampling", ["resample", "resample_to_frequency"]),
            ("Pressure–time product", ["baseline_window_s"]),
            # not "Breath-count overrides" — the one field on this card is already called
            # that, and a card whose title repeats its only row reads like a mistake
            ("Per-file overrides", ["breath_counts"]),
        ]
        by_key = {f.key: f for _g, f in fields}
        by_key[bc_field.key] = bc_field
        # Cards are filled by CONSUMING from a working copy, which makes the table's three
        # failure modes harmless instead of silent:
        #   * a key listed twice picks the field once — listing it again finds nothing, so
        #     there is no second widget to shadow the first in ``_widgets`` and swallow its
        #     edits (measured before this: 21 rows built against 20 widgets);
        #   * a mistyped key matches nothing and its real field simply stays unplaced;
        #   * anything left unplaced lands on "Other" rather than vanishing from the dialog
        #     while still being committed on OK — the worst possible outcome for a
        #     scientific parameter.
        # An empty card is dropped, so a typo shows up as a setting in the wrong place
        # rather than as a mysterious empty box.
        remaining = dict(by_key)
        all_fields = []
        for title, keys in sections:
            picked = [remaining.pop(k) for k in keys if k in remaining]
            if picked:
                all_fields.append((title, picked))
        if remaining:
            all_fields.append(("Other", list(remaining.values())))
        vals = dict(values); vals["breath_counts"] = bc_text

        def _commit(staged):
            """Write ``staged`` (an ``edited_values()`` dict) onto the model and, if
            anything actually changed, mark the analysis modified and recompute. Shared by
            Apply (mid-dialog) and OK (on close) so the two commit through exactly the same
            path — Apply is not a second, parallel way to write these settings."""
            changed = False
            for grp, f in fields:
                if f.key in staged and apply_values(owner[grp], {f.key: staged[f.key]}):
                    changed = True
            # only when the user actually edited the box — ``staged`` carries edited keys
            # only, so an untouched commit must not re-parse (and re-write) the entries
            if "breath_counts" in staged:
                parsed = _parse_breath_counts(staged["breath_counts"], BreathCountEntry,
                                              folder=s.input.folder)
                if parsed != s.processing.breath_counts:
                    s.processing.breath_counts = parsed
                    changed = True
            if not changed:
                return False              # commit without an edit: no dirty flag, no recompute
            self.settings_edited.emit()
            self._request_autorun()      # mechanics feed mech + batch (+ the noise clip)
            return True

        def _trend_hint(v):
            """The live line under the thresholds: end-expiratory trough count while trend
            correction is on (the number that decides whether the run succeeds and cannot
            be read off the settings alone), or a live BREATH count while it is off (see
            _advanced_live_breath_count) — the number this dialog otherwise gives no
            feedback on at all until the dialog is closed and the preview redraws."""
            probe = self._trend_probe
            # The probe is the drift-corrected volume from the LAST rendered preview. This
            # same dialog also stages the settings that BUILD that volume, so once any of
            # them is touched the probe describes a signal the run will not use — and a
            # confident count about the wrong signal is worse than no count. Shared by both
            # branches below.
            stale = ({k: v.get(k) for k in _TREND_PROBE_KEYS} != self._trend_probe_shape)
            if not v.get("correct_trend"):
                if probe is None or not len(probe):
                    return "No previewed file to count breaths in yet."
                if stale:
                    return ("Volume conditioning changed — press OK, then reopen this "
                            "dialog for a breath count.")
                n = self._advanced_live_breath_count(probe, v)
                if n is None:
                    return "Could not count breaths with these breath-detection thresholds."
                return (f"In {self._trend_probe_file}: {n} breath{'s' if n != 1 else ''} "
                        "found with these thresholds.")
            from respmech.core import compute
            need = compute._TREND_MIN_ANCHORS.get(v["trend_method"], 2)
            if probe is None or not len(probe):
                return f"Needs at least {need} end-expiratory troughs in every file."
            if stale:
                return ("Volume conditioning changed — press OK, then reopen this dialog "
                        f"for a trough count. Needs at least {need}.")
            n = compute.trend_anchors(
                probe, s.input.format.sampling_frequency,
                min_height=v["trend_peak_min_height"],
                min_prominence_frac=v["trend_peak_min_prominence_frac"],
                min_distance_s=v["trend_peak_min_distance_s"]).size
            return (f"In {self._trend_probe_file}: volume range {float(np.ptp(probe)):.2f}, "
                    f"{n} trough(s) found — "
                    + ("OK." if n >= need
                       else f"NOT ENOUGH (needs {need}); the run will fail this file."))

        # Not modal (see AdvancedDialog.exec): the channel stack behind this dialog is
        # exactly what the fields on it shape, so Apply has to be visible while it redraws.
        # derived_debounce_ms=250: the live breath count runs a real peak search, not
        # trivial arithmetic like the ECG/EMG modals' derived lines, so it must not fire on
        # every keystroke.
        dlg = AdvancedDialog("Mechanics — advanced", all_fields, vals, parent=self,
                             intro="How breaths are detected, how work of breathing is "
                                   "computed, and the volume/drift corrections. The defaults "
                                   "suit ordinary recordings.",
                             derived=_trend_hint, modal=False, on_apply=_commit,
                             derived_debounce_ms=250)
        if dlg.exec() != QDialog.Accepted:
            return
        _commit(dlg.edited_values())          # see AdvancedDialog.edited_values

    # -- channel preview (Mechanics), synchronous render -------------------
    def _preview(self):
        path = self._current_file()
        if not path:
            return
        try:
            data = stage_mechanics_preview(self.state.settings, path)
            self._render_preview(data)               # render also inside the guard
        except Exception:                            # noqa: BLE001 — copyable error card
            detail = traceback.format_exc()
            self._set_status(f"Channel preview failed — {short_error(detail)}")
            for p in ("channels", "raw"):
                self._overlays[p].show_error("Channel preview failed", detail)

    def _set_mech_caption(self, total, excluded):
        """Update the Mechanics tab's persistent caption (breath count, exclusion count,
        the click-to-exclude instruction) — see self.mech_caption in _build_mech_tab."""
        n = f"{total} breath{'s' if total != 1 else ''}"
        excl = f", {excluded} excluded" if excluded else ""
        self.mech_caption.setFullText(
            f"{n}{excl} — click a shaded breath to include/exclude (red = excluded).")

    def _render_preview(self, data):
        """Draw the mechanics channel stack + breath overlays and the raw EMG
        stack from staged arrays. GUI-thread only; shared by the synchronous
        _preview() and the async 'mech' job."""
        fs = data["fs"]
        series = data["series"]
        t = data["t"]
        spans = data["spans"]
        self._trim_offset_s = data["startix"] / fs      # trimmed span -> absolute EMG time
        self._breaths = list(spans)
        self._clear_panel_overlays("channels", "raw")   # dismiss any stale error card
        self.plots.clear()
        self._channel_plots = []
        self._crosshair_lines = []                       # cleared with the plots; rebuilt below
        pal = _plot_pal()
        # Per ROW, not for the stack as a whole — see theme.set_stack_floor. Five channels
        # sharing a 130 px container is 10 px of trace each.
        _theme.set_stack_floor(self.plots, len(_CHANNELS))
        for i, (key, label, colour) in enumerate(_CHANNELS):
            p = self.plots.addPlot(row=i, col=0, axisItems={"left": SciAxis(orientation="left")})
            p.showGrid(x=True, y=True, alpha=pal["grid_alpha"])
            # name over unit on two centred lines — "Poes (cmH₂O)" on one line is taller than
            # the short stacked plot and the rotated label overran its neighbour
            name, _, unit = label.partition(" (")
            p.getAxis("left").set_channel_label(name, unit.rstrip(")"))
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            # scroll over the graph zooms time only; y-zoom is reserved for the y-axis itself
            _restrict_body_wheel_to_x(p.getViewBox())
            y = series[key]
            p.plot(t[:len(y)], y, pen=_pen(pal["channels"].get(key, colour)))
            # P22: a subtle zero-reference baseline on every channel, so the sign and
            # excursion of each signal read at a glance.
            p.addLine(y=0, pen=pg.mkPen(pal["separator"], width=1, style=Qt.DashLine))
            self._limit_x(p, t[:len(y)])          # can't zoom/pan past the data
            self._channel_plots.append(p)
        # x-values on the bottom channel only; y-zoom is NOT shared — flow, volume and the
        # pressures are different units, so a common y-range would flatten most of them
        self._style_channel_stack(self.plots, self._channel_plots, link_y=False)
        # P17: one crosshair line per channel, hidden until the cursor enters the stack
        self._crosshair_lines = []
        for p in self._channel_plots:
            ln = p.addLine(x=0, pen=pg.mkPen(pal["separator"], width=1, style=Qt.DashLine))
            ln.hide()
            self._crosshair_lines.append(ln)
        self._draw_breath_overlays(spans, data["label_y"],
                                   carried=self._exclusion_carried_for(data["name"]))
        self._render_raw_stack(data["emg"], fs, data.get("emg_flow"))
        # breaths are now known -> (re)number any EMG detail/result already rendered
        self._repaint_view_breaths("detail")
        self._repaint_view_breaths("result")

        self._previewed_file = data["name"]
        # what the trend detector sees for THIS file — feeds the live anchor count in
        # Mechanics — advanced… (see _trend_hint), together with the volume-conditioning
        # settings it was produced under, so the count can be withdrawn once they change
        self._trend_probe = data.get("vol_drift")
        self._trend_probe_file = data["name"]
        self._trend_probe_shape = self._trend_probe_settings()
        self.btn_process_file.setEnabled(True)       # a file is loaded → can process just it
        if data.get("trim_error"):
            self._set_status(f"{data['name']}: showing raw channels — could not detect "
                             f"breaths. {data['trim_error']}")
            # no spans in this branch (see stage_mechanics_preview's TrimError fallback) ->
            # nothing to click, so the persistent caption must not keep asserting the
            # previous file's breath count over a panel with none.
            self.mech_caption.setFullText("")
        elif data.get("trend_error"):
            # Deliberately showing something the batch will NOT produce, so say so.
            self._set_status(f"{data['name']}: {data['nbreaths']} breaths, showing the "
                             f"DRIFT-corrected volume — the end-expiratory trend correction "
                             f"was skipped here and will fail this file in a run. "
                             f"{data['trend_error']}")
            # breaths ARE detected and clickable here (only the trend correction failed),
            # so the caption still applies — same as the plain success branch below.
            nign = sum(1 for _n, _a, _b, ig in spans if ig)
            self._set_mech_caption(data['nbreaths'], nign)
        else:
            nign = sum(1 for _n, _a, _b, ig in spans if ig)
            self._set_status(
                f"{data['name']}: {data['nbreaths']} breaths"
                + (f" ({nign} excluded)" if nign else "")
                + f", trimmed to {data['startix'] / fs:.2f}–{data['endix'] / fs:.2f} s. "
                "Click a shaded breath to include/exclude (red = excluded).")
            self._set_mech_caption(data['nbreaths'], nign)

    def _render_raw_stack(self, emg, fs, flow=None):
        """Draw the stacked raw EMG channels and keep the noise region alive. A discrete
        full-length flow silhouette is superimposed behind each channel so EMG activity can
        be read against the respiration cycle (``flow`` is untrimmed, aligned to the emg)."""
        emg = np.asarray(emg, dtype=float)
        if emg.ndim == 1:
            emg = emg[:, None]
        self._emg_raw_subplots = []
        self._raw_label_y = None
        self.emg_raw_plots.clear()
        if emg.size == 0 or emg.ndim != 2 or emg.shape[1] == 0:
            self._paint_breaths("raw", [], self._trim_offset_s, 0.0)   # drop stale overlays
            self._ensure_noise_region()
            return
        cols = list(self.state.settings.input.channels.emg)
        t = np.arange(emg.shape[0]) / fs
        cycle = _plot_pal()["emg_cycle"]
        _theme.set_stack_floor(self.emg_raw_plots, emg.shape[1])   # per channel, not per stack
        for i in range(emg.shape[1]):
            p = self.emg_raw_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            # This is now a compact reference panel (bottom-row third), so the channels are
            # short. Drop pyqtgraph's "(×0.001)" SI suffix: rotated, it made the label longer
            # than the channel is tall, and the three labels overran each other.
            p.getAxis("left").enableAutoSIPrefix(False)
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            p.plot(t, emg[:, i], pen=_pen(cycle[i % len(cycle)]))
            add_flow_background(p, t, flow, _plot_pal())   # discrete respiration reference, behind
            self._limit_x(p, t)
            self._emg_raw_subplots.append(p)
        # x-values on the bottom channel only; y-zoom shared across the raw EMG channels
        self._style_channel_stack(self.emg_raw_plots, self._emg_raw_subplots, link_y=True)
        self._ensure_noise_region()
        self._raw_label_y = self._safe_top(emg[:, 0])
        self._repaint_view_breaths("raw")

    # -- feature A: breath overlays + include/exclude ----------------------
    @staticmethod
    def _breath_brush(ignored, carried=False):
        """``carried``: this exclusion was recorded against a different (or unrecorded)
        recordings folder than the one now loaded — see ``_exclusion_carried_for``. Drawn
        HATCHED instead of solid, so an inherited exclusion reads differently from one made
        in the folder actually on screen without needing a second colour (which would
        collide with the included/excluded palette already in use elsewhere)."""
        pal = _plot_pal()
        if not ignored:
            return pg.mkBrush(*pal["breath_incl_brush"])
        colour = pg.mkColor(*pal["breath_excl_brush"])
        if carried:
            return QBrush(colour, Qt.BDiagPattern)
        return QBrush(colour, Qt.SolidPattern)

    @staticmethod
    def _breath_label_color(ignored):
        pal = _plot_pal()
        return pg.mkColor(*(pal["breath_excl_label"] if ignored else pal["breath_incl_label"]))

    @staticmethod
    def _limit_x(plot, t):
        """Bound a plot's x view to the data's time extent so it cannot zoom/pan past
        the recording (req: no zooming out beyond the data)."""
        try:
            t = np.asarray(t, dtype=float)
            t = t[np.isfinite(t)]
            if t.size < 2:
                return
            x0, x1 = float(t.min()), float(t.max())
            if x1 > x0:
                vb = plot.getViewBox()
                vb.setLimits(xMin=x0, xMax=x1)
                vb.setXRange(x0, x1, padding=0)
        except Exception:                        # noqa: BLE001 — cosmetic
            pass

    def _breath_text(self, num, ignored):
        """A breath-number TextItem, sized for the SHORT stacked mechanics channel plots
        (~46 px of data area each): at the 13 pt app font the box is 23 px, so the headroom
        needed to show it swallows half the plot.

        Both steps matter, and the second is the one that counts: the default box is 23 px
        almost entirely because QTextDocument adds a 4 px margin on every side — setting a
        smaller font ALONE leaves it at 23 px. Font 9 pt + documentMargin 0 gives ~15 px."""
        txt = pg.TextItem(self._breath_label(num),
                          color=self._breath_label_color(ignored), anchor=(0.5, 0.0))
        f = QFont(); f.setPointSizeF(9.0)
        txt.setFont(f)
        txt.textItem.document().setDocumentMargin(0)
        txt.updateTextPos()
        return txt

    @staticmethod
    def _label_px(texts, default=25.0, margin=4.0):
        """The tallest breath label's rendered height in pixels (+ a little breathing room),
        measured rather than assumed — a TextItem is taller than its font size."""
        try:
            h = max(t.boundingRect().height() for t in (texts.values() if hasattr(texts, "values") else texts))
            return float(h) + margin if h > 0 else default + margin
        except Exception:                        # noqa: BLE001 — cosmetic
            return default + margin

    def _label_headroom(self, plot, label_px=25.0, frac=0.22):
        """Expand the label-carrying plot's y range upward so the breath labels — pinned
        at the view top, hanging down into the view — start over an empty band instead of
        sitting on the signal. Re-fits y to the DATA first so repeated repaints of a
        persistent plot (detail / result) don't compound the headroom.

        The headroom is sized in PIXELS, not as a fraction of the data span: a TextItem is
        a fixed pixel height whatever the scale, so on a short plot a fixed fraction leaves
        less band than the label needs (five stacked mechanics channels leave each ~70 px,
        where 22% ≈ 15 px — less than the label). Solving
        ``new_span = span + label_px * new_span / height_px`` for the exact expansion makes
        the label occupy ``label_px`` at the top at any height; ``frac`` is only the
        fallback when the plot has no laid-out height yet."""
        try:
            vb = plot.getViewBox()
            vb.enableAutoRange(y=True)           # snap y back to the data extent…
            vb.updateAutoRange()                 # …(not the previous, already-expanded view)
            (x0, x1), (y0, y1) = vb.viewRange()
            span = y1 - y0
            if span <= 0:
                return
            h_px = float(vb.height() or 0.0)
            extra = (span * (label_px / (h_px - label_px))
                     if h_px > label_px + 6 else frac * span)
            vb.setYRange(y0, y1 + extra, padding=0)   # (this disables y auto again)
        except Exception:                        # noqa: BLE001 — cosmetic
            pass

    def _pin_breath_labels(self, plot, txt_map):
        """Keep the breath-number labels pinned to the TOP of the visible view under
        y-zoom/pan — the same behaviour as the red capture marks: a sigYRangeChanged
        slot that only setPos()es the existing TextItems (wheel-zoom fires this
        continuously, so nothing is re-created) and self-disconnects once its labels
        leave the view. The labels MUST be added with ignoreBounds=True: a view-pinned
        item that still fed childrenBounds would re-inflate every autorange pass.
        Returns an unpin callable for eager teardown on repaint."""
        vb = plot.getViewBox()
        texts = list(txt_map.values())
        if vb is None or not texts:
            return lambda: None

        def _unpin():
            try:
                vb.sigYRangeChanged.disconnect(_reposition)
            except Exception:                          # noqa: BLE001
                pass

        def _reposition(*_):
            try:
                if texts[0].getViewBox() is None:      # repainted/cleared -> self-remove
                    _unpin()
                    return
                top = vb.viewRange()[1][1]
                for t in texts:
                    t.setPos(t.pos().x(), top)
            except Exception:                          # noqa: BLE001 — cosmetic
                _unpin()

        vb.sigYRangeChanged.connect(_reposition)
        _reposition()                                  # place at the CURRENT view top now
        return _unpin

    @staticmethod
    def _breath_label(num):
        """The per-breath number label, e.g. '#3'. The word 'breath' is intentionally
        omitted from per-breath numbering — it is implicit from context on every graph."""
        return f"#{num}"

    def _draw_breath_overlays(self, spans, label_y=0.0, carried=False):
        self._mech_unpin()               # the old labels are torn down with their pin slot
        self._breath_spans = {n: (t0, t1) for (n, t0, t1, _ig) in spans}
        self._breath_regions = {n: [] for (n, _0, _1, _ig) in spans}
        self._breath_texts = {}
        sep = _plot_pal()["separator"]
        tip = ("Excluded — carried over from a previous recordings folder. Confirm on the "
               "Setup screen or click the breath to make this folder's own decision."
               if carried else None)
        for n, t0, t1, ignored in spans:
            for plot in self._channel_plots:
                reg = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                          brush=self._breath_brush(ignored, carried=carried),
                                          pen=pg.mkPen(None))
                reg.setZValue(-10)
                if ignored and tip:
                    reg.setToolTip(tip)
                plot.addItem(reg)
                self._breath_regions[n].append(reg)
                plot.addLine(x=t0, pen=pg.mkPen(sep, width=1, style=Qt.DashLine))
            if self._channel_plots:
                txt = self._breath_text(n, ignored)
                txt.setPos((t0 + t1) / 2.0, label_y)
                self._channel_plots[0].addItem(txt, ignoreBounds=True)
                self._breath_texts[n] = txt
        if self._channel_plots and self._breath_texts:
            # size the headroom from the label's REAL rendered height, not a guess
            self._label_headroom(self._channel_plots[0], label_px=self._label_px(self._breath_texts))
            self._mech_unpin = self._pin_breath_labels(self._channel_plots[0], self._breath_texts)

    def _breath_at(self, t):
        for n, (t0, t1) in self._breath_spans.items():
            if t0 <= t <= t1:
                return n
        return None

    def _on_plot_clicked(self, ev):
        if not self._breath_spans:
            return
        try:
            pos = ev.scenePos()
        except Exception:                              # noqa: BLE001
            return
        for plot in self._channel_plots:
            vb = plot.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(pos):
                bno = self._breath_at(vb.mapSceneToView(pos).x())
                if bno is not None:
                    self._toggle_breath(bno)
                return

    def _toggle_breath(self, breath_no):
        name = self._selected_filename()
        if not name or breath_no not in self._breath_spans:
            return None
        excl = self.state.settings.processing.exclude_breaths
        entry = next((e for e in excl if e.file == name), None)
        if entry is None:
            # ONLY a brand-new entry gets stamped with the current folder here. An EXISTING
            # entry's folder is deliberately left untouched by a plain toggle, even when the
            # user is un-excluding one of ITS OWN breaths: entry.folder is one tag for the
            # WHOLE file, but excl.breaths can hold a MIX of breaths the user just decided on
            # and others still carried from a different folder that this click never looked
            # at. Restamping on every touch (an earlier version of this fix did) would
            # silently "confirm" those untouched breaths too — exactly the invisible
            # application this ticket exists to stop, just moved one click later. Per the
            # ticket, an entry only stops reading as carried via a fresh creation here or via
            # the Setup banner's "Clear" (core.settings.clear_carried_over); "Keep" is a
            # pure dismiss and — like this — never restamps either, matching "as long as the
            # user has chosen to keep them, a carried exclusion is still drawn hatched", not
            # "keeping it once makes it look native from then on". Known accepted
            # imprecision: a genuinely NEW breath added to an EXISTING carried entry still
            # reads as carried until the whole entry is cleared — ExcludeEntry.folder is one
            # tag per FILE, not per breath, by the ticket's own design.
            entry = ExcludeEntry(file=name, breaths=[], folder=self.state.settings.input.folder)
            excl.append(entry)
        now_excluded = breath_no not in entry.breaths
        if now_excluded:
            entry.breaths = sorted(set(entry.breaths) | {breath_no})
        else:
            entry.breaths = [b for b in entry.breaths if b != breath_no]
            if not entry.breaths:
                excl.remove(entry)
        self.settings_edited.emit()      # exclude_breaths lands in the .toml -> mark dirty
        for reg in self._breath_regions.get(breath_no, []):
            reg.setBrush(self._breath_brush(now_excluded)); reg.update()
        txt = self._breath_texts.get(breath_no)
        if txt is not None:
            txt.setColor(self._breath_label_color(now_excluded))
        # recolour the same breath in every EMG view that has it painted
        for view in ("raw", "detail", "result"):
            rec = self._bov.get(view)
            if not rec:
                continue
            for reg in rec["regions"].get(breath_no, []):
                try:
                    reg.setBrush(self._breath_brush(now_excluded)); reg.update()
                except Exception:                      # noqa: BLE001
                    pass
            t = rec["texts"].get(breath_no)
            if t is not None:
                try:
                    t.setColor(self._breath_label_color(now_excluded))
                except Exception:                      # noqa: BLE001
                    pass
        nexcl = len({b for e in excl if e.file == name for b in e.breaths} & set(self._breath_spans))
        # the rail's badge is the raw exclusion count (matches _sync_rail_exclusions, which
        # reads it the same way for every file the settings name — not the nexcl above,
        # which is scoped to spans of the file CURRENTLY previewed). entry.breaths is []
        # once excl.remove(entry) above has run, so this is correct even then. carried is
        # always False here: the folder restamp just above means this entry, by
        # definition, now matches the current folder.
        self.file_rail.set_excluded_count(name, len(entry.breaths), carried=False)
        # excluding/including a breath must update the AVERAGED result in lockstep with the
        # overlay — otherwise the Campbell loop + per-breath table stay stale and the user
        # tunes blind. Recompute the (mechanics-only) test run, debounced.
        self._request_batch_recompute()
        self._set_status(
            f"{name}: breath {breath_no} {'excluded' if now_excluded else 'included'} "
            f"({nexcl}/{len(self._breath_spans)} excluded). Recomputing the average…")
        self._set_mech_caption(len(self._breath_spans), nexcl)
        return now_excluded

    def _request_batch_recompute(self):
        """Debounced recompute of the mechanics test run (Campbell + per-breath table) after
        a breath is toggled, so the averaged result tracks the overlay live. Inert headless
        (never spins the loop), like _request_autorun."""
        if getattr(self, "_batch_recompute_pending", False):
            return
        self._batch_recompute_pending = True
        QTimer.singleShot(0, self._run_batch_recompute)

    def _run_batch_recompute(self):
        self._batch_recompute_pending = False
        self._schedule("batch")

    # -- breath overlays on the EMG views ----------------------------------
    def _excluded_now(self):
        """Live set of excluded 1-based breath numbers for the current file — the
        single source of truth for overlay colour (matches what the core ignores)."""
        name = self._selected_filename()
        entry = next((e for e in self.state.settings.processing.exclude_breaths if e.file == name), None)
        return set(entry.breaths) if entry else set()

    def _exclusion_carried_for(self, name):
        """True iff ``name``'s exclusion entry was recorded against a DIFFERENT (or
        unrecorded) recordings folder than the one currently loaded — the file-scoped
        counterpart of ``core.settings.carried_over_state``, used to hatch the overlay
        (``_breath_brush``) and word the QC line (``_update_qc_overview``) for exactly the
        file on screen, without recomputing the whole-analysis state on every repaint."""
        from respmech.core.settings import is_carried_folder
        entry = next((e for e in self.state.settings.processing.exclude_breaths if e.file == name), None)
        if entry is None or not entry.breaths:
            return False
        return is_carried_folder(entry.folder, self.state.settings.input.folder)

    def _paint_breaths(self, view, plot_items, offset, label_y):
        """Idempotent per-view overlay: remove the view's previous items (guarded),
        then shade every breath + a boundary line on each plot and a number on the
        first plot. Colour is taken from the LIVE exclusion set."""
        rec = self._bov.get(view)
        if rec:
            rec.get("unpin", lambda: None)()           # detach the old pin slot eagerly
            for plot, item in rec.get("items", []):
                try:
                    plot.removeItem(item)
                except Exception:                      # noqa: BLE001
                    pass
        self._bov[view] = {"items": [], "regions": {}, "texts": {}, "unpin": lambda: None}
        if not plot_items or not self._breaths:
            return
        excl = self._excluded_now()
        carried = self._exclusion_carried_for(self._selected_filename())
        reg_map = self._bov[view]["regions"]; txt_map = self._bov[view]["texts"]
        items = self._bov[view]["items"]
        sep = _plot_pal()["separator"]
        for (num, t0, t1, _ig) in self._breaths:
            ignored = num in excl
            a, b = t0 + offset, t1 + offset
            for p in plot_items:
                reg = pg.LinearRegionItem(values=(a, b), movable=False,
                                          brush=self._breath_brush(ignored, carried=carried),
                                          pen=pg.mkPen(None))
                reg.setZValue(-10)
                p.addItem(reg)
                items.append((p, reg)); reg_map.setdefault(num, []).append(reg)
                line = p.addLine(x=a, pen=pg.mkPen(sep, width=1, style=Qt.DashLine))
                items.append((p, line))
            txt = self._breath_text(num, ignored)
            txt.setPos((a + b) / 2.0, label_y)
            plot_items[0].addItem(txt, ignoreBounds=True)
            items.append((plot_items[0], txt)); txt_map[num] = txt
        self._label_headroom(plot_items[0], label_px=self._label_px(txt_map))   # labels above the signal
        self._bov[view]["unpin"] = self._pin_breath_labels(plot_items[0], txt_map)

    def _repaint_view_breaths(self, view):
        """Repaint one EMG view IF it has rendered real data (label_y sentinel set)."""
        if view == "raw":
            plot_items, label_y = self._emg_raw_subplots, self._raw_label_y
        elif view == "detail":
            plot_items, label_y = [self.emg_plots.getPlotItem()], self._detail_label_y
        elif view == "result":
            plot_items, label_y = [self.emg_result_plots.getPlotItem()], self._result_label_y
        else:
            return
        if label_y is None or not plot_items:
            return
        try:
            self._paint_breaths(view, plot_items, self._trim_offset_s, label_y)
        except Exception:                              # noqa: BLE001 — overlay is cosmetic
            pass

    def _reset_breath_state(self):
        """On a file change, drop stale overlays + spans so a new-file EMG job that
        finishes before the new mech never shows the previous file's numbers."""
        self._mech_unpin()
        self._mech_unpin = lambda: None
        for rec in self._bov.values():
            rec.get("unpin", lambda: None)()
            for plot, item in rec.get("items", []):
                try:
                    plot.removeItem(item)
                except Exception:                      # noqa: BLE001
                    pass
        self._bov = {}
        self._breaths = []
        self._breath_spans = {}
        self._breath_regions = {}
        self._breath_texts = {}
        self._emg_raw_subplots = []
        self._raw_label_y = self._detail_label_y = self._result_label_y = None
        self._trim_offset_s = 0.0
        # a result-checkbox toggle re-renders self._emg_all synchronously (no token
        # gate); drop the previous file's staged result so it can't repaint the old
        # curves and re-arm the result sentinel with them
        self._emg_all = None
        self.emg_result_plots.clear()
        # _previewed_file only advances on a COMPLETED mech render, so leaving it set
        # here would make a flip back to the last-rendered file look like "no change"
        # and skip this very reset while the new file's jobs are still in flight
        self._previewed_file = None
        # the persistent caption is a claim about breaths on screen — drop it here too, or
        # it would keep asserting the PREVIOUS file's breath/exclusion count over an emptied
        # Mechanics tab until (or unless) the new file's render lands.
        self.mech_caption.setFullText("")

    def _trend_probe_settings(self):
        """The volume-conditioning settings behind the current trend probe, in the same
        shape the advanced dialog stages them (see _TREND_PROBE_KEYS / _trend_hint)."""
        vol, samp = self.state.settings.processing.volume, self.state.settings.processing.sampling
        src = {**vars(vol), **vars(samp)}
        return {k: src.get(k) for k in _TREND_PROBE_KEYS}

    def _advanced_live_breath_count(self, probe, v):
        """How many breaths the volume-based detector finds in the previewed file's
        (drift-corrected) volume with the STAGED breath-detection thresholds — the live
        read-out in Mechanics — advanced… while end-expiratory trend correction is off (see
        _trend_hint above). Deliberately always volume-based regardless of the staged
        segmentation method: only the volume probe is staged in this dialog, never flow.

        Uses the EXISTING ``compute.separateintobreathsbyvolume`` — the same function
        ``stage_mechanics_preview`` calls for volume-based segmentation — rather than a new
        peak-finding implementation in the UI layer, against a lightweight stand-in for the
        legacy settings namespace it expects and zero-filled arrays for the phase channels
        this dialog does not stage (flow/poes/pgas/pdi): only their LENGTH matters to the
        breath count, since only ``volume`` drives the peak search.

        Returns ``None`` if the thresholds cannot be evaluated at all, e.g. a minimum
        distance/width of 0 s, which ``scipy.signal.find_peaks`` rejects outright — a
        confidently wrong count would be worse than an explanatory line saying there is
        none."""
        import types

        from respmech.core import compute
        n = len(probe)
        zeros = np.zeros(n)
        stand_in = types.SimpleNamespace(
            processing=types.SimpleNamespace(
                mechanics=types.SimpleNamespace(
                    peakheight=v["height"], peakdistance=v["distance_s"],
                    peakwidth=v["width_s"], excludebreaths={})),
            input=types.SimpleNamespace(
                format=types.SimpleNamespace(
                    samplingfrequency=self.state.settings.input.format.sampling_frequency)))
        try:
            breaths = compute.separateintobreathsbyvolume(
                self._trend_probe_file or "", np.arange(n), zeros, probe,
                zeros, zeros, zeros, [], [], stand_in)
        except Exception:                   # noqa: BLE001 — a live hint is never fatal
            return None
        return len(breaths)

    def _on_batch_result(self, result):
        """Render the automatic mechanics test run: the per-breath table + Campbell (the
        batch is mechanics-only, so no EMG/fidelity here). The noise_report branch is
        retained for the direct-call path that still carries one."""
        cur = self._selected_filename()
        fr = None
        if getattr(result, "files", None):
            fr = result.files.get(cur) or next(iter(result.files.values()), None)
        if fr is None or getattr(fr, "error", None):
            err = getattr(fr, "error", None) or "The run produced no result for this file."
            kind = getattr(fr, "error_kind", None) or str(err).split(":", 1)[0]
            if cur:
                self.file_rail.mark_result(cur, ok=False, error=str(err))
            self._qc_overview_not_assessed(err)
            if kind in _SOFT_FILE_ERRORS:
                # A precondition failure of THIS recording, not a fault: the mech preview
                # keeps drawing the channels, so don't raise a 'Test run failed' card over
                # them. The reason still has to live on these two panels — the status line
                # is a single shared label that the EMG/ECG jobs overwrite moments later,
                # which would leave a blank table and Campbell explaining nothing.
                self._table_model.set_dataframe(None)
                self.campbell.figure.clear(); self.campbell.draw()
                self._forget_campbell()   # the export must not resurrect a cleared diagram
                for p in _PANELS["batch"]:
                    self._overlays[p].show_error(
                        f"Not processed — {short_error(str(err))}", str(err))
                return
            # raise so _on_job_done paints a copyable "Test run failed" error card
            raise _FileRunError(err)
        if cur:
            self.file_rail.mark_result(cur, ok=True, breaths=len(fr.breaths_table))
        self._fill_table(fr.breaths_table)
        self._draw_campbell(fr.breaths)
        self._update_qc_overview(fr)                  # P16 QC line, for THIS file only
        nr = getattr(result, "noise_report", None)
        if nr:
            self.render_noise_report(result)         # sets its own status incl. prop_decrease
            # the auto-selected suppression lives only in the report — write it back so
            # the spinbox AND the re-conditioned EMG views use the value the batch chose
            # (the core never mutates settings), then re-dispatch those EMG views. Only
            # the direct-call path carries a report now; the auto mechanics-only batch
            # does not, so it never triggers this redundant re-condition (the separate
            # 'noise' job owns that).
            chosen = nr.get("prop_decrease")
            if chosen is not None and self.state.settings.processing.emg.noise.auto_prop:
                self.state.settings.processing.emg.noise.prop_decrease = float(chosen)
            self._load_noise_params()                # reflect the chosen prop_decrease
            noise = self.state.settings.processing.emg.noise
            if noise.enabled and noise.auto_prop and self.state.settings.input.channels.emg:
                self._schedule("emg_all")
                self._schedule("emg_detail")
        else:
            n = len(fr.breaths_table)
            self._set_status(f"Test run OK: {n} breaths (nothing written)")

    def _fill_table(self, df):
        self._table_model.set_dataframe(df)
        resize_result_table(self.table)

    def _forget_campbell(self):
        """Drop the cached breaths and disable the export whenever the figure is cleared.

        The export re-renders from this cache (to force a light figure in dark mode), so a
        cache that outlives its diagram is not merely stale — it would write a figure for a
        file the user is no longer looking at, under that file's name. Clearing the figure
        and clearing what it was drawn from have to be the same act."""
        self._campbell_breaths = None
        try:
            self.btn_export_fig.setEnabled(False)
        except Exception:                       # pragma: no cover - button may not exist yet
            pass

    def _draw_campbell(self, breaths, pal=None):
        self._campbell_breaths = breaths        # kept so the export can re-render it light
        pal = _plot_pal() if pal is None else pal
        fig = self.campbell.figure
        fig.clear()
        fig.set_facecolor(pal["mpl_bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(pal["mpl_bg"])
        kept = [b for b in breaths.values() if not b["ignored"]]
        for b in kept:
            ax.plot(b["volume"], b["poes"], color=pal["mpl_loop"], lw=0.7, alpha=0.5, zorder=1)
        # P12: overlay the average breath bold, draw the elastic recoil (relaxation)
        # line EELV→EILV, and shade the inspiratory work between the Poes trace and it.
        self._overlay_campbell_work(ax, kept, pal)
        ax.axhline(0, color=pal["mpl_zeroline"], lw=0.8, zorder=0)
        ax.set_xlabel(_CAMPBELL_XLABEL_VARIANTS[0])
        ax.set_ylabel(_CAMPBELL_YLABEL_VARIANTS[0])
        # D12: volume on x (inverted) and Poes on y — the SAME direction the written
        # Campbell PDF uses (core/plots._pv_average). Before this ticket the two figures
        # were mirror images of each other for the same breaths. Emil's decision
        # 04-08-2026 was to fix the preview, not the writer: the writer's orientation is
        # the 1.x-inherited convention every past export already carries.
        ax.invert_xaxis()
        # No figure title on screen — the panel header carries it. tight_layout() is replaced
        # by _fit_compact_figure, which installs the live tight ENGINE (this figure carried a
        # one-shot PlaceHolderLayoutEngine, so it never re-solved when the splitter moved) and
        # shrinks the labels when the panel is short.
        _fit_compact_figure(
            self.campbell, ax,
            legend_kw={"loc": "lower right", "frameon": False, "fontsize": 7},
            xlabel_variants=_CAMPBELL_XLABEL_VARIANTS,
            ylabel_variants=_CAMPBELL_YLABEL_VARIANTS)
        self.campbell.draw()
        self.btn_export_fig.setEnabled(True)         # a diagram now exists to export

    def _overlay_campbell_work(self, ax, kept, pal):
        """Draw the average PV loop + elastic recoil line + shaded inspiratory work,
        so the Campbell diagram *shows* the work of breathing, not just the loops.
        Defensive: any missing average field falls back to just the loops (no raise)."""
        if not kept:
            return
        b = kept[0]
        insp = b.get("inspiration", {})
        vavg, pavg = b.get("volumeavg"), b.get("poesavg")
        iv, ip = insp.get("volumeavg"), insp.get("poesavg")
        eelv, eilv = b.get("eelvavg"), b.get("eilvavg")   # each [volume, poes]
        try:
            import numpy as np
            if vavg is not None and pavg is not None and len(vavg):
                ax.plot(vavg, pavg, color=pal["mpl_accent"], lw=2.0, zorder=3,
                        label="average breath")
            if eelv is not None and eilv is not None:
                # elastic recoil line: straight line between the two volume endpoints
                ax.plot([eelv[0], eilv[0]], [eelv[1], eilv[1]], color=pal["mpl_target"],
                        ls="--", lw=1.3, zorder=4, label="elastic recoil")
                if iv is not None and ip is not None and len(iv) > 1:
                    iv = np.asarray(iv, float); ip = np.asarray(ip, float)
                    # Poes on the recoil line at each inspiratory volume
                    line_p = np.interp(iv, sorted([eelv[0], eilv[0]]),
                                       [eelv[1], eilv[1]] if eelv[0] <= eilv[0] else [eilv[1], eelv[1]])
                    ax.fill_between(iv, ip, line_p, color=pal["mpl_accent"], alpha=0.18,
                                    zorder=2, label="inspiratory work")
            wob = dict(b.get("wob") or {})
            if "wobtotal" in wob:
                # fg on mpl_bg (not mpl_loop, the colour of the loops the text sits over)
                # clears WCAG's 4.5:1 body-text floor in both themes, and the boxed
                # background keeps it legible over whichever loop happens to pass beneath.
                ax.text(0.02, 0.98, f"WOB {wob['wobtotal']:.2f} J·min⁻¹",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9,
                        fontweight="bold", color=pal["fg"],
                        bbox=dict(facecolor=pal["mpl_bg"], edgecolor="none", alpha=0.85, pad=2))
            # Top-right / bottom-left: with volume on x (inverted) and Poes on y, the loop
            # runs from EELV (low volume, near-baseline Poes → top-right) to EILV (high
            # volume, more negative Poes → bottom-left) — the SAME direction the written
            # Campbell PDF uses (core/plots._pv_average), so the two figures agree and this
            # legend no longer needs to counter a transposition. The WOB read-out above is
            # fixed at the top-left corner (axes fraction, independent of data), which
            # leaves bottom-right as the one corner neither the loop nor the read-out
            # claims — _draw_campbell's legend_kw places the legend there.
            # The legend is deliberately NOT placed here. At the height this panel gets, a
            # three-entry key covers the loops it labels, so whether to draw it at all is a
            # function of the height — which only the fit knows. _draw_campbell hands the
            # placement to _fit_compact_figure as legend_kw, and that sheds it when short.
            pass
        except Exception:                       # pragma: no cover - overlay is best-effort
            pass

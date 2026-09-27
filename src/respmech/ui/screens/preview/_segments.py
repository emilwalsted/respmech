"""PreviewScreen's 'EMG – segments' sub-tab (M-26): the dedicated stack + action band +
per-segment result table for an EMG-only signal set, replacing Mechanics as the first
sub-tab for that shape (see ``_emg_noise.py``'s ``subtab_plan``). The renderer here is
the real, interactive follow-up to M-25's provisional one (``_render_emg_segments_
preview``, which drew into the ECG-reduction/noise-reduction tabs' shared raw EMG stack
with no click surface at all — see that method's own docstring, since replaced by
:meth:`_SegmentsMixin._render_segments_preview`)."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QPushButton, QScrollArea,
                               QSplitter, QTableView, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

import pyqtgraph as pg

from respmech.ui.flow_layout import ElidingLabel
from respmech.ui.plot_overlays import add_flow_background
from respmech.ui import wheel as _wheel
from respmech.ui.result_table import (ResultTableModel, configure_result_table,
                                      resize_result_table)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

from ._plot_helpers import _pen, _plot_pal


class _SegmentsMixin:

    def _build_segments_tab(self):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 6, 0, 0)
        # A persistent one-line caption, same pattern as Mechanics' own mech_caption:
        # segment count / exclusion count / the click-to-exclude instruction, kept
        # current by _render_segments_preview and _toggle_breath (_set_mech_caption is
        # reused as-is — it is a plain "N/M excluded" formatter, not Mechanics-specific).
        self.segments_caption = ElidingLabel("")
        self.segments_caption.setProperty("status", "muted")
        cap_row = QHBoxLayout(); cap_row.setContentsMargins(6, 0, 6, 0)
        cap_row.addWidget(self.segments_caption, 1)
        v.addLayout(cap_row)

        split = QSplitter(Qt.Vertical)
        self.segments_plots = pg.GraphicsLayoutWidget()
        self.segments_plots.setAccessibleName("EMG segments")   # C02: QAccessible/screen readers
        _theme.set_plot_floor(self.segments_plots)
        self.segments_plots.setBackground(_plot_pal()["bg"])
        # Scrolls INSIDE its panel, same reason _emg_raw_scroll does (_emg_noise.py): the
        # per-channel floor is what keeps each trace readable, and as a splitter pane's
        # own minimum it would otherwise demand the whole stack's height at once.
        self._segments_scroll = QScrollArea()
        self._segments_scroll.setWidgetResizable(True)
        self._segments_scroll.setFrameShape(QFrame.NoFrame)
        self._segments_scroll.setWidget(self.segments_plots)
        self._segments_wheel = _wheel.guard_scroll_area(
            self._segments_scroll, extra=[self.segments_plots.viewport()])
        split.addWidget(self._titled("EMG channels", self._segments_scroll))

        self.segtable = QTableView()
        self.segtable.setAccessibleName("Per-segment results")   # C02
        self._segtable_model = ResultTableModel()
        self.segtable.setModel(self._segtable_model)
        self.segtable.verticalHeader().setVisible(False)
        configure_result_table(self.segtable)
        _theme.set_plot_floor(self.segtable)
        split.addWidget(self._titled("Per-segment results", self.segtable))
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        v.addWidget(split)

        self.segments_plots.scene().sigMouseClicked.connect(self._on_segments_clicked)
        self._segments_subplots = []
        self._segments_label_y = None
        return w

    def _build_segments_action_band(self):
        """The segments tab's own QC verdict + per-file action, mirroring
        ``_MechanicsMixin._build_mech_action_band`` exactly (same pinned-outside-the-
        scrolling-page placement, same reason: on a short screen the verdict must not be
        the one thing scrolled out of view). 'Place separators' (M-27) slots in beside
        the button — out of THIS ticket's scope (see the plan's own "Uden for omfang").

        ``segments_qc_overview`` (not a spacer) carries the row's stretch factor, same
        role ``_build_mech_action_band``'s ``mech_window_label`` plays: an ElidingLabel
        with the slack gives the WHOLE row somewhere to squeeze on Windows metrics
        (``test_the_segments_action_band_fits_on_windows_metrics``) — a bare
        ``addStretch()`` contributes nothing to either the row's natural or its minimum
        width, so it cannot be what makes a row squeezable."""
        band = QWidget()
        band.setMinimumHeight(36)
        band.setMaximumHeight(40)
        bar = QHBoxLayout(band)
        bar.setContentsMargins(6, 0, 6, 0)
        self.segments_qc_overview = ElidingLabel("QC:  —")
        self.segments_qc_overview.setProperty("banner", True)
        self.segments_qc_overview.setProperty("status", "muted")
        self.segments_qc_overview.setToolTip(
            "Quality overview of the CURRENTLY PREVIEWED file's most recent test run — "
            "not a batch summary. See the file rail for every file's exclusion count.")
        bar.addWidget(self.segments_qc_overview, 1)   # M-27's button lands beside this
        self.btn_process_segments_file = QPushButton("Process && write this file")
        self.btn_process_segments_file.setEnabled(False)
        self.btn_process_segments_file.setToolTip(
            "Run and write output for the previewed file only.")
        self.btn_process_segments_file.clicked.connect(self._process_this_file)
        bar.addWidget(self.btn_process_segments_file)
        return band

    def _update_segments_stack_floor(self):
        """The segments tab's own eftergivende-floor update (M-26), bound via its own
        ``_MechStackFloorFitter`` instance — see that class's docstring on why it now
        takes an ``update_fn`` instead of always calling ``_update_mech_stack_floor``."""
        if _theme is None:
            return
        try:
            area = getattr(self, "_segments_tab", None)
            vp_h = area.viewport().height() if area is not None else 0
            n = len(self.state.settings.input.channels.emg) or 1
            _theme.set_stack_floor(self.segments_plots, n,
                                   viewport_height=vp_h if vp_h > 0 else None)
        except RuntimeError:                  # pragma: no cover - deleted C++ widget
            pass

    def _on_segments_clicked(self, ev):
        self._toggle_from_emg_click(ev, self._segments_subplots, self._trim_offset_s)

    def _render_segments_stack(self, emg, fs, flow=None):
        """Draw the segments tab's own stacked EMG channels, with REAL click-to-exclude/
        right-click-to-type overlays (``_draw_breath_overlays``) — unlike
        ``_render_raw_stack``'s passive per-view overlay (``_paint_breaths``/``_bov``),
        this stack IS the tab a user interacts with for an EMG-only set, so it drives
        ``_breath_spans``/``_breath_regions``/``_breath_texts`` directly, exactly as the
        Mechanics channel stack does for a flow-bearing set — safe because the two are
        mutually exclusive (see ``_draw_breath_overlays``'s own docstring)."""
        emg = np.asarray(emg, dtype=float)
        if emg.ndim == 1:
            emg = emg[:, None]
        self._segments_subplots = []
        self._segments_label_y = None
        self.segments_plots.clear()
        if emg.size == 0 or emg.ndim != 2 or emg.shape[1] == 0:
            self._draw_breath_overlays([], plots=[])   # drop stale overlays
            return
        cols = list(self.state.settings.input.channels.emg)
        t = np.arange(emg.shape[0]) / fs
        cycle = _plot_pal()["emg_cycle"]
        _theme.set_stack_floor(self.segments_plots, emg.shape[1])   # per channel, not per stack
        for i in range(emg.shape[1]):
            p = self.segments_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", f"col {cols[i]}" if i < len(cols) else f"EMG {i + 1}")
            p.getAxis("left").enableAutoSIPrefix(False)
            if _theme is not None:
                _theme.align_left_axis(p)          # keep the stacked channels x-aligned
            p.plot(t, emg[:, i], pen=_pen(cycle[i % len(cycle)]))
            add_flow_background(p, t, flow, _plot_pal())   # discrete respiration reference, behind
            self._limit_x(p, t)
            self._segments_subplots.append(p)
        self._style_channel_stack(self.segments_plots, self._segments_subplots, link_y=True)
        self._segments_label_y = self._safe_top(emg[:, 0])
        self._draw_breath_overlays(self._breaths, label_y=self._segments_label_y or 0.0,
                                   plots=self._segments_subplots)

    def _render_segments_preview(self, data):
        """The 'segments' job's render entry point (EMG-only signal sets, M-26): draws
        the segments tab's own dedicated, interactive stack (``_render_segments_stack``)
        AND keeps the ECG-reduction/noise-reduction tabs' raw EMG stack current
        (``_render_raw_stack``, as M-25's provisional renderer already did) — those tabs
        remain visible for an EMG-only set (``subtab_plan``, ``caps.emg`` is always true
        for that mode) and show the same physical channels; ``_toggle_from_emg_click``'s
        hit-testing there reads the SAME ``_breath_spans`` this method now populates, so
        clicking on either the segments tab or the raw EMG stack both work.

        Synchronous, unlike 'mech' (``_render_preview_async``): D15's deferred-stage
        split exists for the mechanics stack's own cost profile (up to 1210 items across
        five channel plots on a long recording), which ``BreathSpansItem`` already
        bounds to O(1) per plot regardless of span count — an EMG-only set's segment
        count is small by construction (whole_file = 1, separators = a user-placed
        handful), so there is no equivalent cost here to split around."""
        self._mech_render_gen += 1
        self._trim_offset_s = 0.0                # already the file's own absolute clock
        # Fold (ignored, kind) into ONE pseudo-kind, exactly as stage_mechanics_preview
        # already does for a breath's own kind-or-'excluded'-or-None -- the shared overlay
        # machinery (_paint_breaths/_draw_breath_overlays/_breath_brush) only ever reads a
        # single kind, and stage_emg_segments_preview's own spans (a Qt-free, directly
        # testable shape) keep 'ignored' and 'kind' apart instead of pre-folding them.
        self._breaths = [
            (num, t0, t1, kind if kind else ("excluded" if ignored else None))
            for (num, t0, t1, ignored, kind) in data["spans"]
        ]
        self._render_segments_stack(data["emg"], data["fs"], data.get("emg_flow"))
        self._render_raw_stack(data["emg"], data["fs"], data.get("emg_flow"))
        # segments are now known -> (re)number any EMG detail/result already rendered
        self._repaint_view_breaths("detail")
        self._repaint_view_breaths("result")
        self._previewed_file = data["name"]
        # D24 (mirrors _render_preview_stage1's own gating of btn_process_file): gated
        # jointly with _run_active so a file switch mid-run does not silently re-enable
        # the button a running batch has locked.
        self._process_ready = True
        self.btn_process_segments_file.setEnabled(not self._run_active)
        nseg = len(data["spans"])
        nign = sum(1 for (_n, _a, _b, ignored, _k) in data["spans"] if ignored)
        if data.get("segment_error"):
            # A precondition failure of THIS recording's separator configuration, not a
            # bug (EmgSegmentationError) -- status line only, never a copyable
            # 'failed' card: mirrors stage_mechanics_preview's own TrimError branch,
            # which shows the raw channels with an explanatory status too.
            self._set_status(f"{data['name']}: Not processed — {data['segment_error']}")
            self.segments_caption.setFullText("")
        else:
            self._set_status(
                f"{data['name']}: {nseg} segment{'s' if nseg != 1 else ''}"
                + (f" ({nign} excluded)" if nign else "") + ".")
            self.segments_caption.setFullText(
                f"{nseg} segment{'s' if nseg != 1 else ''}"
                + (f", {nign} excluded" if nign else "")
                + ". Click a shaded segment to include/exclude (red = excluded).")

    def _fill_segtable(self, df):
        self._segtable_model.set_dataframe(df)
        resize_result_table(self.segtable)

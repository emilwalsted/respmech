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

from respmech.core.analysis.segments import remap_segment_number
from respmech.core.settings import SeparatorEntry
from respmech.ui.flow_layout import ElidingLabel
from respmech.ui.plot_overlays import add_flow_background
from respmech.ui import wheel as _wheel
from respmech.ui.result_table import (ResultTableModel, configure_result_table,
                                      resize_result_table)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

from ._plot_helpers import SeparatorLinesItem, _pen, _plot_pal

#: M-27's own tooltip while placement is unavailable, worded exactly as the ticket
#: requires — whole-file mode has no boundaries to place (one segment, always).
_WHOLE_FILE_TOOLTIP = (
    "Whole-file mode has one segment — choose Manual separators in Setup ▸ "
    "Signals ▸ Change… to place separators.")


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
        # M-27: whether 'Place separators' is currently armed (read by the shared
        # _toggle_from_emg_click funnel in _emg_noise.py, checked before it does
        # anything else) + the SeparatorLinesItem drawn on each segments subplot.
        self._separators_armed = False
        self._separator_items = []
        return w

    def _build_segments_action_band(self):
        """The segments tab's own QC verdict + per-file action, mirroring
        ``_MechanicsMixin._build_mech_action_band`` exactly (same pinned-outside-the-
        scrolling-page placement, same reason: on a short screen the verdict must not be
        the one thing scrolled out of view). Three elements now sit in this row: the QC
        label, M-27's own 'Place separators' toggle, and the process button —
        ``test_the_segments_action_band_fits_on_windows_metrics`` covers all three.

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
        # M-27: checkable, so its own pressed state IS the "armed" flag the shared click
        # funnel reads (_toggle_from_emg_click) — toggled() keeps _separators_armed and
        # the checked visual in lockstep with no separate bookkeeping to drift.
        self.btn_place_separators = QPushButton("Place separators")
        self.btn_place_separators.setCheckable(True)
        self.btn_place_separators.toggled.connect(self._on_place_separators_toggled)
        bar.addWidget(self.btn_place_separators)
        self.btn_process_segments_file = QPushButton("Process && write this file")
        self.btn_process_segments_file.setEnabled(False)
        self.btn_process_segments_file.setToolTip(
            "Run and write output for the previewed file only.")
        self.btn_process_segments_file.clicked.connect(self._process_this_file)
        bar.addWidget(self.btn_process_segments_file)
        self._update_separators_button()
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

    def _render_segments_stack(self, emg, fs, flow=None, filename=None):
        """Draw the segments tab's own stacked EMG channels, with REAL click-to-exclude/
        right-click-to-type overlays (``_draw_breath_overlays``) — unlike
        ``_render_raw_stack``'s passive per-view overlay (``_paint_breaths``/``_bov``),
        this stack IS the tab a user interacts with for an EMG-only set, so it drives
        ``_breath_spans``/``_breath_regions``/``_breath_texts`` directly, exactly as the
        Mechanics channel stack does for a flow-bearing set — safe because the two are
        mutually exclusive (see ``_draw_breath_overlays``'s own docstring).

        ``filename`` (M-27) is the file THIS render is for — never read from
        ``self._previewed_file``, which the caller (``_render_segments_preview``) only
        updates AFTER this call returns, so it would still name the PREVIOUS file on a
        file switch. Drives which file's ``processing.segmentation.separators`` entry
        the new ``SeparatorLinesItem`` per subplot draws (``_update_separator_lines``)."""
        emg = np.asarray(emg, dtype=float)
        if emg.ndim == 1:
            emg = emg[:, None]
        self._segments_subplots = []
        self._segments_label_y = None
        self._separator_items = []
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
        self._update_separator_lines(filename)
        self._draw_breath_overlays(self._breaths, label_y=self._segments_label_y or 0.0,
                                   plots=self._segments_subplots)

    def _update_separator_lines(self, filename):
        """(Re)draw M-27's separator markers on every current segments subplot from
        ``processing.segmentation.separators`` — called from ``_render_segments_stack``
        after every (re)render, the same reactive path any other settings edit already
        follows, so a placement/removal shows the instant its recompute comes back."""
        entry = next((e for e in self.state.settings.processing.segmentation.separators
                     if e.file == filename), None)
        times = list(entry.times_s) if entry is not None else []
        self._separator_items = []
        for p in self._segments_subplots:
            item = SeparatorLinesItem()
            item.set_times(times)
            item.setZValue(-5)          # above BreathSpansItem's fill (-10), below the trace
            p.addItem(item)
            self._separator_items.append(item)

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
        # M-31: a segment has no inspiration/expiration split for manoeuvres.suggest_fvc
        # to read (has_phases=False, see core.analysis.segments) — never suggested here,
        # and explicitly cleared so a suggestion from a PREVIOUSLY viewed flow-bearing
        # file cannot survive a switch to an EMG-only one (this render path never goes
        # through _render_preview_stage1, which is the only other place this is set).
        self._suggested_fvc = None
        self._update_separators_button()
        self._render_segments_stack(data["emg"], data["fs"], data.get("emg_flow"), data["name"])
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

    # -- M-27: manual separators (placement, removal, renumbering) ----------

    def _update_separators_button(self):
        """Enable/disable + tooltip 'Place separators' for the CURRENT segmentation
        method and run state — called once at construction and again at the top of
        every ``_render_segments_preview`` (the one place a settings load/edit is
        guaranteed to have already landed). ``whole_file`` has exactly one segment by
        construction (there is nothing to cut), so the button is disabled with the
        ticket's own wording rather than left clickable-but-inert; a run in progress
        disables it the same way the process button already is (``set_run_active``
        mirrors this same rule for a run that STARTS after the tab is already showing
        'separators' mode)."""
        method = self.state.settings.processing.segmentation.method
        if method == "whole_file":
            if self.btn_place_separators.isChecked():
                self.btn_place_separators.setChecked(False)   # also clears _separators_armed
            self.btn_place_separators.setEnabled(False)
            self.btn_place_separators.setToolTip(_WHOLE_FILE_TOOLTIP)
        else:
            self.btn_place_separators.setEnabled(not self._run_active)
            self.btn_place_separators.setToolTip(
                "Click a channel trace to add a separator there, or click an existing "
                "separator (within a few pixels) to remove it.")

    def _on_place_separators_toggled(self, checked):
        self._separators_armed = checked

    def _set_separators(self, file, times):
        """Rewrite ``file``'s manual separator times and renumber every existing
        ``ExcludeEntry``/``BreathTypeEntry`` for it in lockstep, so a segment's own
        exclusion/typing follows it through an insertion or removal instead of
        silently re-attaching to whatever segment happens to carry its OLD number
        afterwards — M-27's whole reason for existing: without this, excluding
        segment 3 and then inserting a separator earlier in the file would silently
        turn into excluding a DIFFERENT segment 3.

        ``times``: the FULL new sorted list of separator times (seconds, the
        recording's own absolute clock). The renumbering compares each existing
        entry's OLD segment START time (derived from the OLD bounds this file had
        before the call) against the NEW bounds ``times`` implies, mapping it to
        whichever new segment now contains that same instant — see
        ``core.analysis.segments.remap_segment_number`` for the single rule that
        covers insertion, removal/merge and no-op renumbering alike. Folder is
        stamped ONLY when a brand-new ``SeparatorEntry`` is created here, never on an
        edit of an existing one — the same carried-over-state rule ``ExcludeEntry``/
        ``BreathTypeEntry`` already follow (see ``_set_breath_type`` in
        ``_mechanics.py``)."""
        proc = self.state.settings.processing
        seg = proc.segmentation
        entry = next((e for e in seg.separators if e.file == file), None)
        old_bounds = [0.0] + (sorted(entry.times_s) if entry is not None else [])
        new_times = sorted(times)
        new_bounds = [0.0] + new_times

        def remap(old_number):
            return remap_segment_number(old_bounds, new_bounds, old_number)

        excl_entry = next((e for e in proc.exclude_breaths if e.file == file), None)
        if excl_entry is not None and excl_entry.breaths:
            excl_entry.breaths = sorted({remap(b) for b in excl_entry.breaths})

        for t in proc.breath_types:
            if t.file != file:
                continue
            t.breath = remap(t.breath)
            # t_onset_s krydstjekkes: re-anchor the advisory onset time to the segment's
            # ACTUAL new start rather than leave it pointing at the pre-edit instant —
            # keeps the anchor meaningful for the NEXT edit instead of drifting further
            # from the truth with every subsequent placement/removal.
            idx = min(max(t.breath - 1, 0), len(new_bounds) - 1)
            t.t_onset_s = new_bounds[idx]
        # A merge can fold two previously distinct typed breaths onto the SAME new
        # number -- Settings.validate()'s "typed more than once" invariant forbids two
        # BreathTypeEntry rows for one (file, breath), so keep the first one this file's
        # own list still holds (lowest original breath number, since the list is walked
        # in append order) and drop the rest, exactly as a user resolving the same
        # collision by hand would have to pick one.
        seen = set()
        deduped = []
        for t in proc.breath_types:
            key = (t.file, t.breath)
            if t.file == file:
                if key in seen:
                    continue
                seen.add(key)
            deduped.append(t)
        proc.breath_types[:] = deduped
        # A merge can ALSO fold a previously EXCLUDED breath and a previously TYPED
        # breath onto the same new number — the other half of Settings.validate()'s
        # invariant ("both typed and excluded"), and one this remap must guard just as
        # deliberately as the typed-vs-typed collision above (self-review finding: an
        # earlier version of this method only resolved collisions WITHIN breath_types,
        # leaving exactly this cross-kind pairing to silently write an invalid Settings
        # that blocked processing until a user noticed and re-clicked the merged
        # segment). A typed breath already carries the same "excluded from the tidal
        # average" effect a plain exclusion does (see BREATH_KINDS's own docstring) plus
        # a kind a plain exclusion does not, so typed wins: drop the collided number
        # from the plain exclusion rather than the richer typed entry.
        if excl_entry is not None and excl_entry.breaths:
            typed_here = {t.breath for t in proc.breath_types if t.file == file}
            excl_entry.breaths = sorted(set(excl_entry.breaths) - typed_here)
            if not excl_entry.breaths:
                proc.exclude_breaths.remove(excl_entry)

        if entry is None:
            seg.separators.append(SeparatorEntry(file=file, times_s=new_times,
                                                 folder=self.state.settings.input.folder))
        else:
            entry.times_s = new_times

    def _toggle_separator_at(self, name, t, vb):
        """Decide add vs. remove for a click at absolute-recording-time ``t`` on
        ``name``, using ``vb``'s OWN current pixel scale for the removal tolerance
        (``vb.viewPixelSize()[0] * 6`` — unit-testable with a fake ``vb`` exposing
        just that one method, no real Qt scene needed) — a fixed-seconds tolerance
        would feel wildly different zoomed in versus zoomed out; a fixed-pixel one
        tracks what the user actually sees. Never validates ``t`` against the
        recording's own duration itself: ``segments.separators()`` already does
        exactly that (``EmgSegmentationError``), and ``_render_segments_preview``
        already turns that into the SAME graceful 'Not processed — …' status line a
        bad separator from any other source gets — duplicating the bound here would
        be a second, divergent copy of a rule that already lives in one place."""
        proc = self.state.settings.processing
        entry = next((e for e in proc.segmentation.separators if e.file == name), None)
        existing = list(entry.times_s) if entry is not None else []
        tol = abs(vb.viewPixelSize()[0]) * 6
        nearest = min(existing, key=lambda s: abs(s - t)) if existing else None
        if nearest is not None and abs(nearest - t) <= tol:
            new_times = [s for s in existing if s != nearest]
            verb = "removed"
            at = nearest
        else:
            new_times = existing + [t]
            verb = "placed"
            at = t
        self._set_separators(name, new_times)
        self.settings_edited.emit()
        # The wide, all-files _sync_rail_breath_state() (M-32; formerly the
        # exclusion-only _sync_rail_exclusions()): unlike an ordinary exclude/type
        # toggle, _set_separators can change the EXCLUSION COUNT for this file by
        # renumbering/merging entries (see its own docstring) AND always changes this
        # file's own segment COUNT — and since M-32, this is the one funnel that already
        # recomputes every file's exclusion/typed/segment badges correctly; a narrower,
        # single-file version of that recompute does not exist today.
        self._sync_rail_breath_state()
        # Wide, not just {"segments", "batch"}: _kinds_for_settings_path treats every
        # processing.segmentation.* field (this one included) as needing the full
        # _AUTO_KINDS recompute, deliberately -- it feeds segment_file/the noise
        # reference clip's rest_segments branch exactly like buffer does. _set_separators
        # can ALSO renumber a rest-typed BreathTypeEntry, which that same function's own
        # wide rule for processing.breath_types exists to cover. Requesting only the two
        # kinds this ticket's own render touches would under-recompute the noise profile
        # whenever a placement/removal moves which segment is typed 'rest'.
        self._request_autorun()
        self._set_status(f"Separator {verb} at {at:.2f} s in {name}.")

    def _place_or_remove_separator(self, ev, plot_items, offset):
        """The armed counterpart of ``_toggle_from_emg_click``: while 'Place
        separators' is checked, every click this shared funnel would otherwise resolve
        as an include/exclude toggle instead places or removes a separator. Mirrors
        ``_toggle_from_emg_click``'s own event-resolution exactly (same best-effort
        button check, same scene-position lookup, same per-plot ViewBox hit test) so
        the two behave identically about WHICH click counts, differing only in what
        the click then does."""
        if self._run_active:
            # Same forced-onto-the-status-bar escape hatch _set_breath_type uses: a
            # non-run_screen status alone would be invisible while a run suppresses it.
            msg = "Separator placement is locked while a run is in progress."
            self._set_status(msg)
            self.write_action_blocked.emit(msg)
            return
        name = self._selected_filename()
        if not name:
            return
        try:
            if ev.isAccepted():
                return
            button = getattr(ev, "button", None)
            if button is not None and button() != Qt.LeftButton:
                return
            pos = ev.scenePos()
        except Exception:                              # noqa: BLE001
            return
        for p in plot_items:
            vb = p.getViewBox()
            if vb is not None and vb.sceneBoundingRect().contains(pos):
                t = vb.mapSceneToView(pos).x() - offset
                self._toggle_separator_at(name, t, vb)
                return

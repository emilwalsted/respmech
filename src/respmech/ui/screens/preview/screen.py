# PreviewScreen's own module: the shared setup, the reactive job scheduler/
# thread pool and file selection, assembling the three sub-tab mixins
# (_MechanicsMixin, _EcgMixin, _EmgNoiseMixin). Split out of preview_screen.py
# (ticket A02). The docstring below is the original screen-level one, moved
# verbatim.
"""Screen 2 — source-data preview & tuning.

Two sub-tabs (the EMG one appears only when EMG channels are configured):

* **Mechanics** — the flow/volume/pressure channels on a shared time axis with
  detected breath boundaries and the trim window; a Campbell diagram and the
  per-breath table. Breaths are shaded/numbered and can be included/excluded by
  clicking.
* **EMG processing** — a view of the raw EMG channels; a "result" view of the
  conditioned EMG (pick which channels to show); a single-channel detail (raw ▸
  ECG-removed ▸ noise-reduced, time + PSD); a draggable rest-region selector that
  sets the shared noise reference; and the noise-window options (active once a
  reference file is set) with a fidelity-frontier preview.

**Reactive auto-run.** Selecting a file automatically kicks off every runnable
computation. Independent jobs (mechanics preview; the full test run that yields
the Campbell/table/fidelity in one pass; the EMG staging) run concurrently, each
on its own worker thread; panels that depend on a value (the auto-selected noise
suppression) are re-run in order once that value is known. Every sub-panel shows
its own spinner while its job is in flight and clears it on completion.

The scheduling entry points are reached only through ``QTimer.singleShot`` (or a
button click), so under a headless test — which never spins the Qt event loop —
no worker threads start and the direct render/compute methods stay synchronous.
All computation reuses the core; nothing here writes to disk.
"""

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

from ._mechanics import _MechanicsMixin
from ._ecg import _EcgMixin
from ._emg_noise import _EmgNoiseMixin, NEEDS_ECG_HINT
from ._busy_overlay import BusyOverlay
from ._figure_fit import _PlotTitleOverlay
from ._jobs import (_TAB_MECH, _PANELS, _SPIN_TEXT, _KIND_LABEL, _AUTO_KINDS,
                   _FILE_KINDS, _kinds_for_settings_path, _changed_settings_paths,
                   _Job, _ORPHANED_THREADS, _MAX_ACTIVE, _FileRunError)


#: why the manual ECG fields (and Auto-suggest) are inert while "Auto (whole batch)" is on
AUTO_BATCH_HINT = ("'Auto (whole batch)' is on: these values will be overwritten at run "
                   "time from the reference file, so editing them (or clicking "
                   "Auto-suggest) here has no lasting effect. Untick it to edit by hand.")


class PreviewScreen(_MechanicsMixin, _EcgMixin, _EmgNoiseMixin, QWidget):
    status_changed = Signal(str)
    # emitted when a noise reference is chosen on the graph (feature B)
    noise_reference_changed = Signal(str, object, bool)
    # emitted when the USER edits a Preview-owned setting that lands in the saved .toml
    # (noise/ECG params, breath exclusions). main_window routes it to the Setup screen's
    # dirty funnel so the title, close guard and Save all reflect the edit. Programmatic
    # fills (_load_*_params, sync_from_settings, the detect_channel seed) and app-derived
    # writes (the auto-picked prop_decrease) deliberately do NOT emit it.
    settings_edited = Signal()
    # emitted to process AND write just the previewed file (P19)
    process_file_requested = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        # reactive-job bookkeeping (replaces the old single self._thread/_worker)
        self._jobs = {}                  # kind -> _Job (the current owner)
        self._draining = set()           # superseded jobs (thread started), kept alive until they self-clean
        self._launch_queue = []          # _Jobs registered but not yet thread.start()ed (concurrency cap)
        self._active = set()             # _Jobs whose thread is started, compute not yet delivered
        self._reaping = set()            # _Jobs delivered; thread quitting — ref held until thread.finished
        self._tokens = {"mech": 0, "batch": 0, "ecg": 0, "emg_all": 0, "emg_detail": 0, "noise": 0}
        self._overlays = {}              # panel key -> BusyOverlay
        # Debounced auto-recompute: rapid edits (a spinbox drag / multi-keystroke entry)
        # restart one single-shot timer, collapsing the burst into ONE re-dispatch after a
        # short quiet period. The timer only fires while an event loop spins, so it stays
        # inert in headless tests (no thread starts synchronously) — same property as the
        # previous QTimer.singleShot(0) coalescing. _pending_kinds accumulates which panels
        # the pending run should recompute.
        self._autorun_timer = QTimer(self)
        self._autorun_timer.setSingleShot(True)
        self._autorun_timer.setInterval(300)         # ms
        self._autorun_timer.timeout.connect(self._run_autorun)
        self._pending_kinds = set()
        self._last_synced_settings = None   # snapshot for dependency-scoped invalidation (diff)
        self._noise_has_result = False   # has the fidelity panel a current result? (first-compute guard)
        self._previewed_file = None
        # drift-corrected volume of the previewed file + its name: the live anchor count
        # in Mechanics — advanced… counts troughs in this (see _trend_hint).
        self._trend_probe = None
        self._trend_probe_file = ""
        self._trend_probe_shape = None
        self._blanked_for_invalid = False   # see _schedule's settings gate
        self._loading_noise = False
        self._loading_ecg = False
        self._ecg_auto_repaired = False   # see _load_ecg_params' repair of a stuck auto-detect
        self._ecg_capture_subplots = []   # per-channel PlotItems of the ECG-processed stack
        # breath-overlay interaction state (feature A)
        self._channel_plots = []
        self._breath_spans = {}
        self._breath_regions = {}
        self._breath_texts = {}
        self._mech_unpin = lambda: None   # detaches the mechanics label-pin slot
        # draggable noise-selection region (feature B)
        self._noise_region = None
        # staged all-channel EMG (result view)
        self._emg_all = None
        self.result_checks = []          # list of (col:int, QCheckBox)
        # breath overlays in the EMG views (Mechanics keeps its own _breath_* state)
        self._breaths = []               # last mech spans [(num, t0, t1, ignored), ...] (TRIMMED s)
        self._trim_offset_s = 0.0        # startix/fs — maps a trimmed span to absolute EMG time
        self._bov = {}                   # view -> {'items':[(plot,item)], 'regions':{num:[reg]}, 'texts':{num:txt}}
        self._emg_raw_subplots = []      # per-channel PlotItems of the raw stack (hit-testing)
        self._raw_label_y = None
        self._detail_label_y = None
        self._result_label_y = None
        self._build()
        # render dispatch by job kind (built after _build so the methods exist)
        self._RENDER = {
            "mech": self._render_preview,
            "batch": self._on_batch_result,
            "ecg": self._on_ecg_result,
            "emg_all": self._on_emg_all_result,
            "emg_detail": self._on_emg_detail_result,
            "noise": self._on_noise_result,
        }

    # -- construction -------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(11, 11, 11, 11)   # deterministic, matches Setup
        bar = QHBoxLayout()
        self.file_combo = QComboBox()
        # Size the selector to the recording names, not the window: names are short
        # (~100px at 13pt), but a stretch-factor row hands an uncapped combo ALL the
        # slack. "wide" caps the contents at 320px; a longer name elides in the closed
        # field but shows in full in the popup.
        self.file_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.file_combo.setProperty("formField", "wide")
        # P27: step through files without leaving the plots — buttons + PageUp/PageDown.
        self.btn_prev_file = QPushButton("◀")
        self.btn_next_file = QPushButton("▶")
        for b, tip in ((self.btn_prev_file, "Previous file (PgUp)"),
                       (self.btn_next_file, "Next file (PgDn)")):
            # The nav QSS gives the arrow 6px side air, and its min-width:0 overwrites
            # any code-side minimum at polish — the width simply tracks sizeHint, so the
            # glyph cannot clip. The max only stops layout slack ballooning the button.
            b.setProperty("nav", True)
            b.setMaximumWidth(34); b.setToolTip(tip)
        self.btn_prev_file.clicked.connect(lambda: self._step_file(-1))
        self.btn_next_file.clicked.connect(lambda: self._step_file(+1))
        # 'Refresh' recomputes every auto panel (not the test run); the file list itself
        # refreshes automatically when file-related settings change.
        self.btn_refresh_all = QPushButton("Refresh")
        self.btn_refresh_all.setToolTip("Recompute all preview panels for the current file.")
        self.btn_refresh_all.clicked.connect(self._refresh_all)
        bar.addWidget(QLabel("File:")); bar.addWidget(self.btn_prev_file)
        bar.addWidget(self.file_combo); bar.addWidget(self.btn_next_file)
        bar.addSpacing(12)   # detach Refresh a little from the ◀ file ▶ cluster
        bar.addWidget(self.btn_refresh_all)
        # park the slack AFTER Refresh: QPushButton's Minimum h-policy would otherwise
        # let the buttons balloon to fill the row once the combo stops stretching
        bar.addStretch(1)
        root.addLayout(bar)
        from PySide6.QtGui import QShortcut, QKeySequence  # noqa: PLC0415
        QShortcut(QKeySequence(Qt.Key_PageUp), self, activated=lambda: self._step_file(-1))
        QShortcut(QKeySequence(Qt.Key_PageDown), self, activated=lambda: self._step_file(+1))
        # Status is shown ONLY in the main-window bottom status bar (via status_changed).
        # Keep the label object (state holder + tests read pv.status) but never place it
        # under the file selector.
        self.status = QLabel("Pick a file — everything runs automatically.")
        self.status.setWordWrap(True)
        self.status.setParent(self)
        self.status.hide()

        self.subtabs = QTabWidget()
        # Each page is wrapped in a scroll area, and it is the WRAPPER that goes into the tab
        # widget — so _update_emg_tab_visibility's indexOf/insertTab/removeTab keep working on
        # the same objects. ``.widget()`` on any of these gives the page content back.
        self._mech_page = self._build_mech_tab()
        self._ecg_page = self._build_ecg_tab()     # inserted between Mechanics + noise when EMG cols exist
        self._emg_page = self._build_emg_tab()
        self._mech_tab = self._scrollable(self._mech_page)
        self._ecg_tab = self._scrollable(self._ecg_page)
        self._emg_tab = self._scrollable(self._emg_page)
        self.subtabs.addTab(self._mech_tab, _TAB_MECH)
        root.addWidget(self.subtabs, 1)

        # The control strips sit directly above wheel-zoomable plots, so an overshoot while
        # zooming used to land on a spin box and step it — and every ECG/noise parameter here
        # writes straight into the analysis, marks it modified and schedules a recompute.
        # file_combo was worse still: one notch silently switched the previewed recording.
        #
        # The pages scroll now, so a wheel over a spin box FORWARDS to that page's scroll area
        # rather than doing nothing — a control that swallows the wheel in a scrollable page is
        # a dead patch the form stops moving under. The plots are deliberately left out: a
        # wheel over a graph zooms its time axis (see _restrict_body_wheel_to_x), which is a
        # documented interaction, so vertical travel over a graph is the scroll bar's job.
        # file_combo and the tab bar sit OUTSIDE every scroll area and keep swallowing: one
        # notch there still switches the recording or the tab.
        # ``extra``: the matplotlib canvases and the results table are not spin boxes or
        # combos, so guard_scroll_area's by-class discovery misses them — and unlike a
        # pyqtgraph plot they do nothing with a wheel of their own. Without this they were
        # dead patches: the page simply stopped moving under the cursor over the Campbell
        # diagram, the fidelity figure and the breath table.
        _passive = []
        for _page in (self._mech_page, self._ecg_page, self._emg_page):
            _passive.append([w for w in _page.findChildren(FigureCanvasQTAgg)]
                            + [w for w in _page.findChildren(QTableWidget)])
        self._page_wheel_guards = [
            _wheel.guard_scroll_area(a, extra=ex)
            for a, ex in zip((self._mech_tab, self._ecg_tab, self._emg_tab), _passive)]
        # A panel scrolled half out of view must not report its spinner or its error card
        # off-screen — see BusyOverlay.centre_on_visible.
        for _a in (self._mech_tab, self._ecg_tab, self._emg_tab):
            _a.verticalScrollBar().valueChanged.connect(self._recentre_overlays)
        self._wheel_guard = _wheel.swallow_wheel(
            extra=[self.subtabs.tabBar(), self.file_combo], parent=self)

        # one spinner overlay per panel (each fed by exactly one job kind)
        self._overlays = {
            "channels": BusyOverlay(self.plots),
            # one overlay per REAL panel: parenting this to the lower splitter instead put a
            # single spinner across the table+Campbell pair, reading as a third panel.
            "table": BusyOverlay(self.table),
            "campbell": BusyOverlay(self.campbell),
            "ecg_capture": BusyOverlay(self.ecg_capture_plot),
            "ecg_stack": BusyOverlay(self.ecg_processed_plots),
            # the VIEWPORT, not the stack: the raw stack scrolls inside its panel and is
            # taller than what is on screen, so an overlay covering the stack centred its
            # spinner and its error card below the fold. The viewport is exactly the visible
            # band, which is where a message about this panel has to appear.
            "raw": BusyOverlay(self._emg_raw_scroll.viewport()),
            "result": BusyOverlay(self.emg_result_plots),
            "detail": BusyOverlay(self.emg_plots),
            "detail_psd": BusyOverlay(self.emg_psd_canvas),   # the detail job also renders the PSD
            "fidelity": BusyOverlay(self.fidelity_canvas),
        }

        self._refresh_emg_channels()
        self._refresh_ecg_channels()
        self._load_noise_params()
        self._load_ecg_params()
        self._ensure_noise_region()
        self._update_emg_tab_visibility()
        # If loading repaired a stuck auto-detect, say so rather than quietly changing what
        # the file asked for. Deferred to the event loop because the status label and the
        # dirty marker both belong to a window that is still being constructed here.
        if self._ecg_auto_repaired:
            QTimer.singleShot(0, self._announce_ecg_auto_repair)

        self.file_combo.currentTextChanged.connect(self._on_file_selected)
        # the chip's themed height isn't known until the EMG sub-tab is first laid out;
        # match the 'Set noise profile' button to it then, so the strip is one band.
        self.subtabs.currentChanged.connect(lambda *_: QTimer.singleShot(0, self._align_noise_strip))
        self.refresh_files()

    class _PageScroll(QScrollArea):
        """A scroll area whose page is the VIEWPORT height when there is room for it, and its
        own minimum height when there is not.

        Qt's ``widgetResizable`` sizes the page from ``heightForWidth`` whenever the page's
        layout reports one — and these pages' wrapping chip rows do. So the page came out at
        its NATURAL height on every screen (measured 1659 px for the noise page) instead of
        filling the viewport, and the bottom diagnostics row sat below the fold even on a
        2560x1400 display, where the whole page had fitted before. Scrolling is meant to be
        the answer to a SHORT screen, not a permanent tax on every screen.
        """

        def resizeEvent(self, ev):          # noqa: N802 - Qt API
            super().resizeEvent(ev)
            self._fit_page()

        def showEvent(self, ev):            # noqa: N802 - Qt API
            # A page that was not the current tab is laid out only when it is first shown,
            # and that pass happens after the filter below has had its chance — so without
            # this the tab the user switches TO keeps the natural height Qt gave it.
            super().showEvent(ev)
            self._fit_page()

        def setWidget(self, page):          # noqa: N802 - Qt API
            # widgetResizable is deliberately OFF (see the class docstring): with it on, Qt
            # sizes the page from its layout's heightForWidth and that decision is final —
            # overriding resizeEvent, filtering LayoutRequest and even calling resize()
            # directly afterwards all left the page at its 1659 px natural height, measured.
            # So the page's size is this class's own job, recomputed on the three events that
            # can change it.
            super().setWidget(page)
            page.installEventFilter(self)
            self._fit_page()

        def eventFilter(self, obj, ev):     # noqa: N802 - Qt API
            if obj is self.widget() and ev.type() == QEvent.LayoutRequest:
                self._fit_page()
            return super().eventFilter(obj, ev)

        _fitting = False

        def _fit_page(self):
            # Re-entrancy guard AND a no-op check, both required: resizing the page emits the
            # very LayoutRequest this is called from, and pyqtgraph re-lays its axes out on
            # every resize, so without both this recursed until the stack blew.
            if self._fitting:
                return
            page = self.widget()
            if page is None:
                return
            vp = self.viewport().size()
            want = QSize(vp.width(), max(vp.height(), page.minimumSizeHint().height()))
            if page.size() == want:
                return
            self._fitting = True
            try:
                page.resize(want)
            finally:
                self._fitting = False

    def _recentre_overlays(self, *_):
        """Re-centre every visible busy/error card on the part of its panel still in view."""
        for ov in getattr(self, "_overlays", {}).values():
            if ov is not None and ov.isVisible():
                ov.centre_on_visible()

    def _scrollable(self, page):
        """Put a subtab page in a vertical scroll area.

        This is what lets the graphs keep a readable minimum height (theme.PLOT_MIN_HEIGHT)
        without that minimum reaching the window: a page taller than the viewport scrolls
        instead of compressing its panels into stacked axes. ``setWidgetResizable(True)``
        means the page still EXPANDS to fill a tall window, so nothing changes on a large
        screen — the scroll bar only appears when the room genuinely is not there.
        """
        area = self._PageScroll()
        area.setWidgetResizable(False)      # _PageScroll sizes the page itself
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(page)
        # The control strips already wrap (flow_layout), so width is handled by wrapping
        # rather than by scrolling; AsNeeded rather than AlwaysOff so an extreme font still
        # gets a bar instead of clipped controls.
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return area

    @staticmethod
    def _style_channel_stack(glw, plots, *, link_y=False, time_label="Time (s)"):
        """Shared styling for a vertical stack of channel subplots in a GraphicsLayoutWidget.

        - x tick VALUES appear only on the bottom plot; the rest give that strip back to the
          trace, so a tall stack reads as one aligned block rather than a ladder of axes.
        - the rows sit tight, a hairline apart.
        - x is always linked: pan or zoom the time axis once and every channel follows.
        - y is linked only when the channels share units — the EMG and ECG-processed stacks
          do, so zooming one zooms all; the mechanics stack mixes L/s, L and cmH2O, so it
          must NOT, or the pressures would flatten the flow.
        """
        if not plots:
            return
        try:
            glw.ci.layout.setVerticalSpacing(2)
        except Exception:                        # pragma: no cover — spacing is cosmetic
            pass
        first = plots[0]
        for p in plots:
            plot_perf.tune(p)                    # decimation + view clipping (display only)
        for i, p in enumerate(plots):
            last = i == len(plots) - 1
            ax = p.getAxis("bottom")
            ax.setStyle(showValues=last)
            if last:
                if time_label:
                    p.setLabel("bottom", time_label)
            else:
                # keep the axis (its ticks still drive the x-grid) but collapse the strip the
                # tick numbers used to occupy
                ax.setLabel(None)
                ax.setHeight(8)
            if i > 0:
                p.setXLink(first)
                if link_y:
                    p.setYLink(first)
        if link_y and len(plots) > 1:
            # A shared y makes zooming one channel zoom all — but pyqtgraph collapses every
            # linked plot onto plots[0]'s range, which would clip a louder electrode down to
            # a quieter first one (and overflow its flow silhouette). Set the shared range to
            # the UNION of every channel's data instead, so the sync never hides signal.
            ys = [p.getViewBox().childrenBounds()[1] for p in plots]
            ys = [b for b in ys if b and b[0] is not None and b[1] is not None]
            if ys:
                lo, hi = min(b[0] for b in ys), max(b[1] for b in ys)
                pad = (hi - lo) * 0.05 or 1.0
                first.getViewBox().setYRange(lo - pad, hi + pad, padding=0)

    def _set_trace_key(self, stages):
        """Name the detail plot's traces in its title band, as colour swatch + label.

        Rich text in a QLabel rather than a pyqtgraph legend: a legend is an item inside the
        ViewBox, so wherever it is anchored it covers signal — at the compact height this tab
        defaults to, its single row hid close to half the data area. The band above the plot
        is already reserved (the channel picker sets its height) and had the room going spare.
        """
        muted = _theme.active_theme().get("text_muted", "#666") if _theme is not None else "#666"
        parts = []
        for name, colour in stages:
            hexcol = pg.mkColor(colour).name()
            parts.append(f'<span style="color:{hexcol};">&#9644;</span>'
                         f'<span style="color:{muted};">&nbsp;{name}</span>')
        # Entries are separated by a REAL space, not more &nbsp;. The gap between a swatch and
        # its own name must not break, but if the separators do not break either the label has
        # no wrap opportunity at all: it then broke mid-entry into four lines where two were
        # needed, and the band grew from 45 px to 78 px.
        self.emg_trace_key.setText("&nbsp;&nbsp; ".join(parts))
        self.emg_trace_key.adjustSize()
        for ov in getattr(self, "_plot_title_overlays", []):
            ov.fit()               # the key changed width; re-pin the band it sits in


    def _titled_overlay(self, title, plot, corner=None, extra=None):
        """Like :meth:`_titled`, but the title row floats INSIDE the plot's top band
        instead of sitting above it (see _PlotTitleOverlay for why). Same outer margins
        as _titled so the panels keep one shared inset."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.setSpacing(0)
        lay.addWidget(plot, 1)
        if not hasattr(self, "_plot_title_overlays"):
            self._plot_title_overlays = []
        self._plot_title_overlays.append(_PlotTitleOverlay(plot, title, corner, extra))
        return box

    def _fit_plot_overlays(self):
        """Re-pin the floating titles/pickers — the result-channel picker regrows on every
        channel-list change, and only the overlay knows where its pieces belong."""
        for ov in getattr(self, "_plot_title_overlays", []):
            ov.fit()

    @staticmethod
    def _titled(title, widget, corner=None):
        """A titled panel. ``corner`` is an optional widget pinned to the top-right of
        the title row (e.g. the detail-channel dropdown or the result-channel picker)."""
        # The panel's own margin is the single source of air: it keeps the title, the plot
        # and the right-pinned corner widget (channel selectors) all off the panel edge, so
        # nothing butts against the neighbouring panel or the nav.
        box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(8, 4, 8, 6); lay.setSpacing(2)
        header = QHBoxLayout(); header.setContentsMargins(0, 0, 0, 0); header.setSpacing(8)
        lab = QLabel(title); lab.setProperty("status", "muted")
        box._title_label = lab          # so a panel can keep its header current
        header.addWidget(lab); header.addStretch(1)
        if corner is not None:
            header.addWidget(corner, 0, Qt.AlignRight | Qt.AlignVCenter)
        lay.addLayout(header); lay.addWidget(widget, 1)
        return box

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- file list (reactive) ----------------------------------------------
    def refresh_files(self):
        s = self.state.settings
        # drop the preview caches only when the input folder/mask ACTUALLY changes (the file
        # set may differ) — not on every entry into the Preview tab, which would defeat the
        # cross-revisit reuse. Per-file freshness tokens in the keys guard in-place edits.
        fm = ((s.input.folder or ""), (s.input.files or ""))
        if fm != getattr(self, "_cache_fm", None):
            from respmech.ui.screens import _preview_cache
            _preview_cache.clear_all()
            self._cache_fm = fm
        prev = self.file_combo.currentText()
        folder = (s.input.folder or "").strip()
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        files = []
        if folder and os.path.isdir(folder):
            from respmech.ui.validation import matching_files
            files = [os.path.basename(f) for f in matching_files(folder, s.input.files)]
            self.file_combo.addItems(files)
        self.file_combo.blockSignals(False)
        if prev in files:
            self.file_combo.setCurrentText(prev)
        elif prev:
            # the current file changed silently while signals were blocked (prev is
            # gone from the list -> the combo auto-selected files[0], or emptied):
            # take the same cleanup path an interactive switch would, so stale breath
            # overlays never linger clickable against the wrong (or no) file
            self._begin_file_switch()
        if not folder:
            self._set_status("Set an input folder in Settings.")
        elif not os.path.isdir(folder):
            self._set_status(f"Input folder not found: {folder}")
        elif not files:
            self._set_status(f"No files match '{s.input.files}' in that folder.")
        else:
            self._set_status(f"{len(files)} file(s) — pick one; everything runs automatically.")
        self._update_actions()

    def _refresh_files(self):        # kept for existing wiring + tests
        self.refresh_files()

    def sync_from_settings(self):
        self._refresh_emg_channels()
        self._refresh_ecg_channels()
        self._load_noise_params()
        self._load_ecg_params()
        self._update_emg_tab_visibility()
        self._update_actions()
        # Dependency-scoped invalidation: diff the settings against the last-synced snapshot
        # and recompute ONLY the panels whose inputs actually changed. An EMG-only edit no
        # longer re-runs the mechanics preview + test run; a mechanics-only edit no longer
        # re-conditions EMG or rebuilds the noise profile. The first sync (no baseline) and
        # any unclassified field recompute everything, so a panel can never be left stale.
        cur = self.state.settings
        if self._last_synced_settings is None:
            kinds = set(_AUTO_KINDS)
        else:
            kinds = set()
            for p in _changed_settings_paths(self._last_synced_settings, cur):
                kinds |= _kinds_for_settings_path(p)
        self._last_synced_settings = copy.deepcopy(cur)
        if not kinds:
            return                   # nothing preview-relevant changed -> leave panels as they are
        self._cancel_inflight(kinds)  # abort only the now-stale panels' jobs
        self._request_autorun(kinds)

    def _current_file(self):
        name = self.file_combo.currentText()
        return os.path.join(self.state.settings.input.folder, name) if name else None

    def _step_file(self, delta: int):
        """P27: move the file selection by ``delta`` (keyboard/▲▼), clamped to the list."""
        n = self.file_combo.count()
        if n <= 1:
            return
        i = min(max(self.file_combo.currentIndex() + delta, 0), n - 1)
        if i != self.file_combo.currentIndex():
            self.file_combo.setCurrentIndex(i)

    # -- enablement / guidance ---------------------------------------------
    def _settings_ok(self):
        try:
            self.state.settings.validate()
            return True, ""
        except Exception as e:              # noqa: BLE001
            return False, str(e)

    def _job_running(self, kind):
        return kind in self._jobs

    def _update_actions(self, status=True):
        """Refresh button/widget enablement. ``status=False`` updates enablement
        only and never touches the status line — used after a job completes so a
        fresh result/error message is not clobbered by a guidance hint."""
        has_file = bool(self.file_combo.currentText())
        ok, why = self._settings_ok()
        emg = self.state.settings.processing.emg
        noise_on, ref = emg.noise.enabled, emg.noise.reference_file
        has_emg = bool(self.state.settings.input.channels.emg)
        # The floating titles/pickers re-pin here because this runs after every channel-list
        # change — the moment the result picker has regrown and needs repositioning.
        self._fit_plot_overlays()
        # Noise reduction runs on whatever the ECG stage produced (core/pipeline _process_emg),
        # and its reference clip is ECG-cleaned too. It is not a hard requirement — the core
        # will happily denoise a raw signal — but the profile would then model the cardiac
        # artefact as steady background noise, which it is not: it is large and periodic. So
        # the controls stay visible and explain themselves rather than silently misbehaving.
        ecg_on = emg.remove_ecg
        jr = self._job_running
        self.btn_refresh_all.setEnabled(has_file)
        self.emg_channel.setEnabled(has_emg)
        # The enable checkbox and the reference picker only need ECG removal on (so the user
        # can turn noise on and choose a reference); the parameter chip additionally needs
        # noise actually enabled with a reference set.
        self.noise_enabled.setEnabled(has_emg and ecg_on)
        self.btn_set_noise.setEnabled(has_file and has_emg and ecg_on)
        self.noise_opts.setEnabled(bool(has_emg and ecg_on and noise_on and ref))
        for w in (self.noise_enabled, self.btn_set_noise, self.noise_opts):
            w.setToolTip("" if ecg_on else NEEDS_ECG_HINT)
        # (The gated peak's ECG prerequisite is enforced in the Advanced modal now, where
        # the control moved — its fields open disabled with NEEDS_ECG_GATE_HINT until
        # 'Remove ECG' is on.)
        # "Auto (whole batch)" itself needs ECG removal on (same requirement Settings.validate
        # enforces); once checked, the fields it will overwrite at run time grey out — same
        # pattern the noise strength field uses in Advanced, so a manual edit there can't
        # look like it did nothing.
        # The fields themselves keep their existing, always-editable-when-EMG-present
        # enablement otherwise (pre-configuring before ticking Remove ECG is unaffected).
        self.ecg_auto_batch.setEnabled(has_emg and ecg_on)
        auto_batch_on = self.ecg_auto_batch.isChecked()
        # Auto-suggest writes the same 5 fields the batch auto-detect will overwrite anyway,
        # so it is gated alongside them — otherwise a click looks like it did something (a
        # success status, updated widgets, a re-rendered preview) that a real run discards.
        for w in (self.ecg_capture_channel, self.ecg_min_height, self.ecg_min_distance,
                  self.btn_ecg_advanced, self.btn_ecg_autosuggest):
            w.setEnabled(not auto_batch_on)
            w.setToolTip(AUTO_BATCH_HINT if auto_batch_on else self._ecg_auto_gated_base_tooltips[w])
        if not status:
            return
        if has_emg and noise_on and not ecg_on:
            self._set_status(NEEDS_ECG_HINT)
        elif noise_on and not ref:
            self._set_status("Noise reduction is on — pick a reference file (Settings) or "
                             "click 'Set noise profile' to mark a rest span in this file.")
        elif ecg_on and auto_batch_on:
            self._set_status("'Auto (whole batch)' is on — this preview still shows the last "
                             "manual/Auto-suggest settings; the real run auto-detects its own "
                             "parameters from the reference file for every file, and reports "
                             "the per-file result in run-report.txt.")
        elif has_file and not ok:
            self._set_status(f"Settings incomplete: {why}")

    def _invalidate_inflight(self, kinds=None):
        """Bump the token of each targeted kind so any in-flight job for it is dropped by
        the acceptance check in _on_job_done when it lands — even if it finished in the
        same event-loop iteration as the change, before the (debounced) autorun bumps
        tokens. A superseded job still clears its own spinner. ``kinds`` defaults to ALL:
        on a file switch pass only the file-dependent kinds so the still-valid, test-wide
        noise job is not needlessly invalidated."""
        for k in (self._tokens if kinds is None else kinds):
            if k in self._tokens:
                self._tokens[k] += 1

    def _cancel_inflight(self, kinds=None):
        """Abort the in-flight jobs of ``kinds`` (default all): cancel each worker
        (cooperative), invalidate its token so a late result is dropped, and move it to
        draining so its thread is still joined on completion. Used on a settings change /
        file switch / Refresh — a file switch scopes to the file-dependent kinds so the
        test-wide noise job keeps running."""
        target = set(_AUTO_KINDS if kinds is None else kinds)
        for kind, job in list(self._jobs.items()):
            if kind not in target:
                continue
            if job in self._launch_queue:
                self._launch_queue.remove(job)         # never started -> drop (no thread/finished)
                self._clear_panel_overlays(*_PANELS[kind])   # ...and its spinner (no _on_job_done to stop it)
            else:
                if hasattr(job.worker, "cancel"):
                    try:
                        job.worker.cancel()
                    except Exception:                  # noqa: BLE001
                        pass
                self._draining.add(job)                # still running -> joined on completion
            del self._jobs[kind]
        self._invalidate_inflight(target)

    def _noise_applicable(self):
        """True when the noise/fidelity job would produce a panel (EMG channels present,
        noise enabled, a reference file set) — same gate as _schedule('noise')."""
        ncfg = self.state.settings.processing.emg.noise
        return bool(self.state.settings.input.channels.emg) and ncfg.enabled and bool(ncfg.reference_file)

    def _begin_file_switch(self):
        """Shared cleanup when the current file changes: blank the FILE-dependent panels,
        abort their in-flight jobs, and re-dispatch them for the new file. The test-wide
        noise/fidelity panel is left intact — switching the previewed file cannot change
        its result — so it is neither blanked nor rebuilt, EXCEPT on its first compute
        (no result yet). Reached from an interactive combo change and from refresh_files
        silently switching the current file when the previous one vanished."""
        kinds = set(_FILE_KINDS)
        if self._noise_applicable() and not self._noise_has_result:
            kinds.add("noise")                 # first compute only; an existing result survives the switch
        self._clear_file_panels("noise" in kinds)
        self._cancel_inflight(kinds)           # abort only the file-dependent jobs (noise keeps running)
        self._request_autorun(kinds)

    def _clear_file_panels(self, include_noise=False):
        """Blank the FILE-dependent panels for a file switch, leaving the test-wide
        fidelity panel (and its last result) intact unless ``include_noise`` (a first
        compute is being scheduled)."""
        self.plots.clear(); self._channel_plots = []
        self.table.setRowCount(0); self.table.setColumnCount(0)
        self.campbell.figure.clear(); self.campbell.draw()
        self._forget_campbell()      # the export must not resurrect a cleared diagram
        self.ecg_capture_plot.clear(); self.ecg_processed_plots.clear(); self._ecg_capture_subplots = []
        self.emg_raw_plots.clear()
        self.emg_plots.clear()
        self._set_trace_key([])       # the legend this replaced was emptied by plots.clear()
        self.emg_psd_canvas.figure.clear(); self.emg_psd_canvas.draw()
        if include_noise:
            self.fidelity_canvas.figure.clear(); self.fidelity_canvas.draw()
            self._noise_has_result = False
        self._reset_breath_state()   # clears _bov/_breaths/_emg_all/result plot/_previewed_file
        self._ensure_noise_region()  # re-attach the rest-region selector to the cleared detail plot
        panels = [p for k in _FILE_KINDS for p in _PANELS[k]]
        if include_noise:
            panels += _PANELS["noise"]
        self._clear_panel_overlays(*panels)   # dismiss stale spinners/cards on the cleared panels

    def _refresh_all(self):
        """The 'Refresh' button: recompute every auto panel (NOT the test run) for the
        current file. Aborts any in-flight auto jobs first, then re-dispatches."""
        if not self._current_file():
            return
        self._cancel_inflight()
        self._request_autorun()

    def _clear_all_panels(self):
        """Blank every panel to its start/empty state (as when the screen first opens):
        mechanics channels, the test-run table + Campbell, the raw EMG stack, the
        conditioned result, the detail time + PSD, and the fidelity frontier.
        _reset_breath_state additionally drops breath overlays + staged EMG."""
        self.plots.clear(); self._channel_plots = []
        self.table.setRowCount(0); self.table.setColumnCount(0)
        self.campbell.figure.clear(); self.campbell.draw()
        self._forget_campbell()      # the export must not resurrect a cleared diagram
        self.ecg_capture_plot.clear(); self.ecg_processed_plots.clear(); self._ecg_capture_subplots = []
        self.emg_raw_plots.clear()
        self.emg_plots.clear()
        self._set_trace_key([])       # the legend this replaced was emptied by plots.clear()
        self.emg_psd_canvas.figure.clear(); self.emg_psd_canvas.draw()
        self.fidelity_canvas.figure.clear(); self.fidelity_canvas.draw()
        self._noise_has_result = False
        self._reset_breath_state()   # clears _bov/_breaths/_emg_all/result plot/_previewed_file
        self._ensure_noise_region()  # re-attach the rest-region selector to the cleared detail plot
        self._clear_panel_overlays(*self._overlays)   # dismiss stale spinners / error cards

    def _on_file_selected(self, name):
        self._update_actions()
        if name and name != self._previewed_file:
            self._begin_file_switch()

    # -- reactive scheduling ------------------------------------------------
    def _request_autorun(self, kinds=None):
        """Coalesce change events into a single re-dispatch after a short quiet period
        (~300 ms). Restarting the single-shot timer on each edit debounces a spinbox drag
        / multi-keystroke entry into ONE recompute. ``kinds`` scopes which panels the
        pending run recomputes (default: all auto kinds); pending kinds accumulate across
        edits until the timer fires. Never starts a thread synchronously — the timer only
        fires while an event loop spins, so this is inert in headless tests."""
        self._pending_kinds |= set(_AUTO_KINDS if kinds is None else kinds)
        self._autorun_timer.start()          # start() on a running single-shot timer restarts it

    def _run_autorun(self):
        kinds = self._pending_kinds or set(_AUTO_KINDS)
        self._pending_kinds = set()
        self._schedule_all(kinds)

    def _schedule_all(self, kinds=None):
        for k in (_AUTO_KINDS if kinds is None else kinds):
            if k in _AUTO_KINDS:             # ignore anything that is not an auto kind
                self._schedule(k)

    def _schedule(self, kind):
        """Evaluate the kind's gate, build its worker, and (only if it will
        actually launch) bump the kind's token and start it concurrently. The
        token is bumped LAST so a gated-out reschedule cannot supersede — and
        discard the result of — an in-flight job it never replaced."""
        path = self._current_file()
        if not path:
            return
        # EVERY kind loads the recording, and every load resolves the channel columns — so an
        # incomplete mapping has to gate them all, not just "batch". This used to be masked by
        # the Setup spin boxes, whose minimum of 1 meant a required channel could never be
        # unset; with the assignment dialog as the sole writer it genuinely can be, and each
        # ungated kind would render a panel full of raw loader traceback instead.
        ok, why = self._settings_ok()
        if not ok:
            # Blank the traces, not just the spinners. Unlike the EMG sub-tabs — which
            # _update_emg_tab_visibility removes outright — the Mechanics tab is always
            # present, and sync_from_settings does not clear panels. So clearing the channel
            # mapping after a preview left the previous mapping's plots on screen, looking
            # like a current result for settings that can no longer produce one.
            # _settings_ok is global, so every kind gates out together: blank once on the way
            # in, and re-arm when the settings become valid again.
            if not self._blanked_for_invalid:
                # Cancel first, and for EVERY kind. Blanking alone does not bump the tokens,
                # so a worker launched while the settings were still valid would complete
                # afterwards, pass the acceptance check in _on_job_done, and repaint the very
                # traces we just cleared — over a status line claiming success. Verified.
                self._cancel_inflight(set(_AUTO_KINDS))
                self._clear_file_panels(include_noise=True)
                self._blanked_for_invalid = True
            self._clear_panel_overlays(*_PANELS[kind])
            self._set_status(f"Settings incomplete: {why}")
            return
        if self._blanked_for_invalid:
            # The blank was global, so the recovery has to be too. sync_from_settings scopes
            # the rebuild to the kinds the repairing edit touched, which is right for an
            # ordinary edit and wrong here: repair a field that scopes to {"batch"} and only
            # the table returns, leaving five panels blank for good under a status reading
            # "Test run OK". Every GUI-reachable invalidation happens to scope wide, so this
            # only bites a hand-edited analysis — which is exactly the case the blank exists
            # for. The debounce coalesces this with the edit's own request.
            self._blanked_for_invalid = False
            self._request_autorun(set(_AUTO_KINDS))
        has_emg = bool(self.state.settings.input.channels.emg)
        # snapshot the settings on the GUI thread so a worker never reads them while
        # the GUI mutates them in place (Settings is a plain, deep-copyable model)
        snap = copy.deepcopy(self.state.settings)
        worker = None
        if kind == "mech":
            worker = FnWorker(stage_mechanics_preview, snap, path)
        elif kind == "ecg":
            if not has_emg:
                self._clear_panel_overlays(*_PANELS[kind])   # gated out -> drop a stale spinner/card
                return
            worker = FnWorker(stage_ecg_reduction, snap, path)
        elif kind == "batch":
            # MECHANICS-ONLY test run: drop EMG so run_batch skips ECG removal, EMG RMS
            # and noise reduction (that work is shown separately in the EMG tab).
            snap.input.channels.emg = []
            snap.processing.emg.remove_ecg = False
            snap.processing.emg.noise.enabled = False
            worker = BatchWorker(snap, write=False, only_files=[os.path.basename(path)])
        elif kind == "emg_all":
            if not has_emg:
                self._clear_panel_overlays(*_PANELS[kind])   # gated out -> drop a stale spinner/card
                return
            worker = EmgAllChannelsWorker(snap, path)
        elif kind == "emg_detail":
            if not has_emg:
                self._clear_panel_overlays(*_PANELS[kind])
                return
            ch = max(0, self.emg_channel.currentIndex())
            worker = EmgConditioningWorker(snap, path, ch)
        elif kind == "noise":
            ncfg = self.state.settings.processing.emg.noise
            if not has_emg or not ncfg.enabled or not ncfg.reference_file:
                self._clear_panel_overlays(*_PANELS[kind])   # no reference -> no fidelity panel
                self._noise_has_result = False
                return                                 # no fidelity without a noise reference
            worker = FnWorker(stage_noise_fidelity, snap)
        if worker is None:
            return
        self._tokens[kind] += 1
        self._launch(kind, self._tokens[kind], worker)

    def _launch(self, kind, token, worker):
        old = self._jobs.pop(kind, None)
        if old is not None:
            if old in self._launch_queue:
                # never started -> no running thread (no GC hazard) and its `finished` would
                # never fire, so drop it rather than draining (which would wedge the drain)
                self._launch_queue.remove(old)
            else:
                if hasattr(old.worker, "cancel"):
                    try:
                        old.worker.cancel()
                    except Exception:              # noqa: BLE001
                        pass
                self._draining.add(old)            # still running -> keep referenced until it self-cleans
        for p in _PANELS[kind]:
            self._overlays[p].start(_SPIN_TEXT[kind])
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        job = _Job(kind, token, thread, worker)
        self._jobs[kind] = job
        # Worker -> GUI delivery is EXPLICITLY queued. AutoConnection decides
        # queued-vs-direct at emit time, and under a fast-emit race it can resolve a
        # bound-method connection to DIRECT — running the render and the overlay
        # stop()/hide() on the worker thread, which starts the spinner QProgressBar's
        # style-animation timer cross-thread ("QBasicTimer::start: Timers cannot be
        # started from another thread" on cocoa). Qt.QueuedConnection pins delivery to
        # the GUI thread regardless of timing; sender() still resolves under it.
        # (Receivers must still be bound methods of self, never lambdas — a lambda has
        # no receiver QObject for the queued delivery to land on.)
        worker.finished.connect(self._job_finished, Qt.QueuedConnection)
        if hasattr(worker, "failed"):
            worker.failed.connect(self._job_failed_slot, Qt.QueuedConnection)
        thread.finished.connect(self._release_thread, Qt.QueuedConnection)   # non-blocking cleanup when the thread exits exec()
        self._launch_queue.append(job)                  # start it when a concurrency slot is free
        self._pump_pool()
        self._update_actions(status=False)

    def _pump_pool(self):
        """Start queued jobs up to the concurrency cap. Called after a launch and after each
        job is delivered (freeing a slot), so at most _MAX_ACTIVE worker threads run at once."""
        while self._launch_queue and len(self._active) < _MAX_ACTIVE:
            job = self._launch_queue.pop(0)
            self._active.add(job)
            job.thread.start()

    def _release_thread(self):
        """Queued to the GUI thread by thread.finished: the thread has exited its event loop,
        so its strong ref can be dropped + deletion scheduled. NEVER runs while the thread is
        alive (that is why the ref is held in _reaping until here) — preserving the crash-safe
        'never GC a running QThread' invariant without a blocking join on the GUI thread."""
        thread = self.sender()
        for job in list(self._reaping):
            if job.thread is thread:
                self._reaping.discard(job)
                job.thread.deleteLater()
                break

    def _find_job(self, worker):
        for j in list(self._jobs.values()) + list(self._draining):
            if j.worker is worker:
                return j
        return None

    def _job_failed_slot(self, msg):
        job = self._find_job(self.sender())
        if job is not None:
            job.error = msg                # per-job, so a drained job can't poison its successor

    def _job_finished(self, result):
        job = self._find_job(self.sender())
        if job is not None:
            self._on_job_done(job, result)

    def _on_job_done(self, job, result):
        # NON-BLOCKING teardown: post a quit and hold a strong ref in _reaping until the
        # thread's `finished` fires (_release_thread) — the GUI thread never blocks in a join,
        # so sibling jobs' queued `finished` signals (and the UI) keep flowing. Freeing this
        # slot lets the next queued job start.
        job.thread.quit()
        self._active.discard(job)
        self._reaping.add(job)
        if self._jobs.get(job.kind) is job:
            del self._jobs[job.kind]
        self._draining.discard(job)
        self._pump_pool()
        # superseded by a newer same-kind job: drop the payload silently, and
        # only clear the spinner if no newer owner took over the panels — but
        # never wipe an error card the current owner already painted (a stale
        # draining job finishing late must not erase the live job's error).
        if job.token != self._tokens[job.kind]:
            if job.kind not in self._jobs:
                for p in _PANELS[job.kind]:
                    if not self._overlays[p].error:
                        self._overlays[p].stop()
            self._update_actions(status=False)
            return
        label = _KIND_LABEL.get(job.kind, job.kind)
        err = job.error
        if err or result is None:
            detail = err or "The computation returned no data."
            self._set_status(f"{label} failed — {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(f"{label} failed", detail)
            self._update_actions(status=False)
            return
        try:
            self._RENDER[job.kind](result)
        except _FileRunError as e:                     # a per-file analysis error -> "failed"
            detail = str(e)
            self._set_status(f"{label} failed — {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(f"{label} failed", detail)
            self._update_actions(status=False)
            return
        except Exception:                              # noqa: BLE001 — a rendering bug
            detail = traceback.format_exc()
            self._set_status(f"{label} — display error: {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(f"{label} — display error", detail)
            self._update_actions(status=False)
            return
        for p in _PANELS[job.kind]:
            self._overlays[p].stop()
        if job.kind == "noise":
            self._noise_has_result = True   # the fidelity panel now holds a current result
        self._update_actions(status=False)

    def shutdown(self, wait_ms=5000):
        """Cancel and join every worker thread (current + draining). Called from
        MainWindow.closeEvent so nothing is destroyed while still running.

        A QThread whose worker is still mid-compute cannot be interrupted (the
        staging/`run_batch` calls are monolithic), so we join with a budget and,
        for any thread that will not stop in time, keep it referenced forever in
        the module-level parking list rather than let CPython GC a running
        QThread (which would call std::terminate and abort the process)."""
        # Disarm the debounced auto-run FIRST: a torn-down screen must never keep a loaded
        # 300ms timer that could later fire into another screen's event loop and launch real,
        # never-cancelled jobs on a dead screen (a cross-test/tab thread leak). Clearing the
        # pending set makes a stray timeout that already slipped through a no-op.
        self._autorun_timer.stop()
        self._pending_kinds.clear()
        self._launch_queue.clear()                     # unstarted jobs: no thread to join
        # a superseded-running job can be in both _draining and _active; a delivered one in
        # _reaping (its thread may still be quitting) — join every started thread exactly once
        jobs = list(set(self._jobs.values()) | self._draining | self._active | self._reaping)
        for j in jobs:
            if hasattr(j.worker, "cancel"):
                try:
                    j.worker.cancel()
                except Exception:                      # noqa: BLE001
                    pass
        for j in jobs:
            j.thread.quit()
        for j in jobs:
            if not j.thread.wait(wait_ms):
                _ORPHANED_THREADS.append((j.thread, j.worker))
        self._jobs.clear()
        self._draining.clear()
        self._active.clear()
        self._reaping.clear()

    def panel_busy(self, key):
        ov = self._overlays.get(key)
        return bool(ov and ov.busy)

    def busy_panels(self):
        return {k for k, o in self._overlays.items() if o.busy}

    def panel_error(self, key):
        """The full error detail shown in a panel's error card, or None."""
        ov = self._overlays.get(key)
        return ov.error if ov is not None else None

    def _clear_panel_overlays(self, *keys):
        """Dismiss any spinner/error card on these panels — called by the
        synchronous render paths so a successful redraw never leaves a stale
        error card raised over fresh content."""
        for k in keys:
            ov = self._overlays.get(k)
            if ov is not None:
                ov.stop()

    def _safe_top(self, *arrays, default=1.0):
        """A finite, headless-safe top-of-signal value for placing the number label."""
        best = None
        for a in arrays:
            if a is None:
                continue
            a = np.asarray(a, dtype=float)
            finite = a[np.isfinite(a)] if a.size else a
            if finite.size == 0:
                continue
            m = float(finite.max())
            best = m if best is None else max(best, m)
        return best if best is not None else default

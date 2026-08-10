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
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.file_rail import FileRail
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui.manifest import build_manifest
from respmech.ui import plot_perf
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers
from respmech.ui import wheel as _wheel
from respmech.ui.flow_layout import (ElidingLabel, FlowLayout, cluster as _cluster,
                                     elide as _elide, install_flow as _install_flow)
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None

from ._mechanics import _MechanicsMixin, _MechStackFloorFitter
from ._ecg import _EcgMixin
from ._emg_noise import _EmgNoiseMixin, NEEDS_ECG_HINT
from ._busy_overlay import BusyOverlay
from ._figure_fit import _PlotTitleOverlay
from ._jobs import (_TAB_MECH, _PANELS, _SPIN_TEXT, _KIND_LABEL, _AUTO_KINDS,
                   _FILE_KINDS, _kinds_for_settings_path, _changed_settings_paths,
                   _Job, _ORPHANED_THREADS, _MAX_ACTIVE, _FileRunError)


#: why the manual ECG fields (and Auto-suggest) are inert while "Auto-detect for the batch" is on
AUTO_BATCH_HINT = ("Auto-detect for the batch is on: these five settings are re-derived at "
                   "run time from the first matched recording and applied to every file, so "
                   "anything you set here (including Auto-suggest) is discarded. Untick it "
                   "to set them by hand.")


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
    # D24: a write action (currently just the breath-toggle click) was rejected because a
    # run is active. NOT routed through status_changed/_set_status: while a run is active,
    # MainWindow._on_screen_status suppresses every screen's status EXCEPT run_screen's own
    # (see its docstring), specifically so a run's progress line isn't stomped by an
    # unrelated screen's status on a tab switch — but that suppression also swallows THIS
    # message whole, since its only trigger condition is "a run is active", the exact
    # window in which it would otherwise be dropped. This is the same "X just happened,
    # show it now regardless of the active tab" case noise_reference_changed already
    # exists for (see _on_noise_reference_changed) — reused the pattern instead of
    # special-casing the shared suppression logic itself.
    write_action_blocked = Signal(str)

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
        self._manifest_cache = {}           # (path, mtime, size)-keyed probe memo for the file rail's manifest
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
        # bumped by every _render_preview_stage1/_reset_breath_state call; a deferred
        # _render_preview_async continuation checks it still matches before touching
        # the mechanics panels, so a stale render (superseded by a file switch) is a
        # silent no-op instead of drawing over the newer file's plots (ticket D15)
        self._mech_render_gen = 0
        # draggable noise-selection region (feature B)
        self._noise_region = None
        self._noise_label = None          # 'noise reference' caption at the band's left edge
        self._noise_label_unpin = lambda: None   # detaches the label's top-of-view pin (D07)
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
        # D24 (UI-overhaul): whether a batch is currently running (set by MainWindow via
        # set_run_active, wired to RunScreen's run_started/run_finished) and whether the
        # currently previewed file has a drawn diagram to write ("Process & write this
        # file" is only ever meaningful with both true). Kept as two separate flags rather
        # than folding the run-lock straight into btn_process_file.setEnabled(...) at each
        # call site, because a file switch mid-run (allowed — B04: the graphs/rail keep
        # working, only the WRITE actions are gated) re-renders the Mechanics stack and
        # would otherwise silently re-enable the button through the normal "file loaded"
        # path in _mechanics._render_preview_stage1.
        self._run_active = False
        self._process_ready = False
        self._build()
        # render dispatch by job kind (built after _build so the methods exist)
        self._RENDER = {
            # _render_preview_async, not _render_preview: the reactive job path defers
            # its two expensive stages via QTimer.singleShot so the GUI thread does not
            # freeze for the whole render (ticket D15) — see its docstring. It owns
            # stopping _PANELS['mech']'s busy overlays itself; _on_job_done below skips
            # its own generic stop for this one kind for exactly that reason.
            "mech": self._render_preview_async,
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
        # P27: step through files without leaving the plots — buttons + keyboard shortcuts,
        # targeted at the file rail's selection (see _build_workspace below). Tooltips are
        # filled in once the real shortcuts exist below (C02), so they show the platform's
        # own key glyphs instead of a hardcoded string.
        self.btn_prev_file = QPushButton("◀")
        self.btn_next_file = QPushButton("▶")
        # C02: an icon button's accessible name defaults to its visible glyph ("◀"/"▶"),
        # which QAccessible confirmed a screen reader would read literally. The glyph stays
        # as the visible label; the accessible name is the actual action.
        self.btn_prev_file.setAccessibleName("Previous file")
        self.btn_next_file.setAccessibleName("Next file")
        for b in (self.btn_prev_file, self.btn_next_file):
            # The nav QSS gives the arrow 6px side air, and its min-width:0 overwrites
            # any code-side minimum at polish — the width simply tracks sizeHint, so the
            # glyph cannot clip. The max only stops layout slack ballooning the button.
            b.setProperty("nav", True)
            b.setMaximumWidth(34)
        self.btn_prev_file.clicked.connect(lambda: self.file_rail.step(-1))
        self.btn_next_file.clicked.connect(lambda: self.file_rail.step(+1))
        # 'Refresh' recomputes every auto panel (not the test run); the file list itself
        # refreshes automatically when file-related settings change.
        self.btn_refresh_all = QPushButton("Refresh")
        self.btn_refresh_all.setToolTip("Recompute all preview panels for the current file.")
        self.btn_refresh_all.clicked.connect(self._refresh_all)
        bar.addWidget(QLabel("File:")); bar.addWidget(self.btn_prev_file)
        bar.addWidget(self.btn_next_file)
        bar.addSpacing(12)   # detach Refresh a little from the ◀ ▶ cluster
        bar.addWidget(self.btn_refresh_all)
        bar.addStretch(1)
        root.addLayout(bar)
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
        # D14: keep the Mechanics stack's floor eftergivende for whatever viewport this
        # scroll area actually has right now — see _MechStackFloorFitter.
        self._mech_stack_fitter = _MechStackFloorFitter(self, self._mech_tab.viewport())
        # D14: the QC verdict + its two per-file actions live OUTSIDE the scrolling page now
        # (see _build_mech_action_band) — built here so it can be pinned in THIS screen's
        # own root layout, below self.subtabs, instead of scrolling away with the page.
        self._mech_action_band = self._build_mech_action_band()

        # B02: the file rail — one row per file, replacing the old file_combo. A left-hand
        # panel rather than folded into the top bar, since a filterable, stateful row list
        # needs real vertical room a toolbar cannot give it. Narrow but resizable: wide
        # enough for a filename plus its state glyphs, never so wide it eats the plots.
        self.file_rail = FileRail()
        self.file_rail.setMinimumWidth(170)
        self.file_rail.setMaximumWidth(280)

        # C02: file-navigation shortcuts. Ctrl+[ / Ctrl+] is the primary pair — drawn as
        # ⌘[ / ⌘] on macOS, the Safari/Xcode "previous/next" convention — plus a
        # layout-independent alias Alt+Left/Alt+Right, because "[" itself needs Alt/Option
        # on a Danish or German keyboard. Both are window-scoped (the default QShortcut
        # context), meant to work anywhere on this screen. PageUp/PageDown are kept for
        # continuity but rescoped to the file rail with WidgetWithChildrenShortcut: parented
        # on ``self`` with the default context they used to fire before a focused widget
        # (e.g. the breath table) ever saw the key — PageDown in the table scrolled nothing
        # and switched the file out from under the user instead.
        from PySide6.QtGui import QShortcut, QKeySequence  # noqa: PLC0415
        sc_prev = QShortcut(QKeySequence("Ctrl+["), self, activated=lambda: self.file_rail.step(-1))
        sc_next = QShortcut(QKeySequence("Ctrl+]"), self, activated=lambda: self.file_rail.step(+1))
        QShortcut(QKeySequence("Alt+Left"), self, activated=lambda: self.file_rail.step(-1))
        QShortcut(QKeySequence("Alt+Right"), self, activated=lambda: self.file_rail.step(+1))
        sc_pgup = QShortcut(QKeySequence(Qt.Key_PageUp), self.file_rail,
                             activated=lambda: self.file_rail.step(-1))
        sc_pgup.setContext(Qt.WidgetWithChildrenShortcut)
        sc_pgdn = QShortcut(QKeySequence(Qt.Key_PageDown), self.file_rail,
                             activated=lambda: self.file_rail.step(+1))
        sc_pgdn.setContext(Qt.WidgetWithChildrenShortcut)
        self._sc_prev_file = sc_prev
        self._sc_next_file = sc_next
        self.btn_prev_file.setToolTip(
            f"Previous file ({sc_prev.key().toString(QKeySequence.NativeText)})")
        self.btn_next_file.setToolTip(
            f"Next file ({sc_next.key().toString(QKeySequence.NativeText)})")

        self._workspace_split = QSplitter(Qt.Horizontal)
        self._workspace_split.addWidget(self.file_rail)
        self._workspace_split.addWidget(self.subtabs)
        self._workspace_split.setStretchFactor(0, 0)
        self._workspace_split.setStretchFactor(1, 1)
        self._workspace_split.setSizes([210, 900])
        root.addWidget(self._workspace_split, 1)
        # D14: fixed under the workspace, same as settings_screen.py pins its own QC strip
        # (self.qc) below its scrolling form — see _build_mech_action_band's docstring.
        root.addWidget(self._mech_action_band)
        self._update_mech_action_band_visibility()

        # The control strips sit directly above wheel-zoomable plots, so an overshoot while
        # zooming used to land on a spin box and step it — and every ECG/noise parameter here
        # writes straight into the analysis, marks it modified and schedules a recompute.
        #
        # The pages scroll now, so a wheel over a spin box FORWARDS to that page's scroll area
        # rather than doing nothing — a control that swallows the wheel in a scrollable page is
        # a dead patch the form stops moving under. The plots are deliberately left out: a
        # wheel over a graph zooms its time axis (see _restrict_body_wheel_to_x), which is a
        # documented interaction, so vertical travel over a graph is the scroll bar's job.
        # The tab bar sits OUTSIDE every scroll area and keeps swallowing: one notch there
        # still switches the tab. The file rail's QListView is deliberately NOT in this list
        # (unlike the old file_combo): a QListView scrolling on a wheel is the ordinary,
        # wanted behaviour, not the "one notch silently switches the recording" footgun a
        # QComboBox had — there is no swallow to guard against.
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
            extra=[self.subtabs.tabBar()], parent=self)

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
        self._load_noise_params()            # this already refreshes the band (D07) — no
        self._load_ecg_params()               # file is selected yet here, so it stays hidden;
        self._update_emg_tab_visibility()     # the OLD unconditional _ensure_noise_region()
        # call that used to sit here bypassed that and always showed reference_intervals[0]
        # regardless of which file refresh_files() (below) was about to select — found in
        # review: a fresh construction could paint the wrong file's span for one frame.
        # If loading repaired a stuck auto-detect, say so rather than quietly changing what
        # the file asked for. Deferred to the event loop because the status label and the
        # dirty marker both belong to a window that is still being constructed here.
        if self._ecg_auto_repaired:
            QTimer.singleShot(0, self._announce_ecg_auto_repair)

        self.file_rail.selectionChanged.connect(self._on_file_selected)
        # the chip's themed height isn't known until the EMG sub-tab is first laid out;
        # match the 'Set noise profile' button to it then, so the strip is one band.
        self.subtabs.currentChanged.connect(lambda *_: QTimer.singleShot(0, self._align_noise_strip))
        # D14: the action band only makes sense for the Mechanics sub-tab; cheap (show/hide),
        # so unlike _align_noise_strip above this needs no debounce of its own.
        self.subtabs.currentChanged.connect(lambda *_: self._update_mech_action_band_visibility())
        self.refresh_files()

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

        Lifted out to :mod:`respmech.ui.panel` (ticket B03) so the Run drawer can reuse
        the exact same short-screen scrolling instead of a second copy of it; this stays
        as a thin wrapper so the existing ``self._scrollable(...)`` call sites here are
        unaffected. The control strips already wrap (flow_layout), so width is handled by
        wrapping rather than by scrolling — AsNeeded rather than AlwaysOff so an extreme
        font still gets a bar instead of clipped controls.
        """
        from respmech.ui.panel import scrollable
        return scrollable(page, horizontal_policy=Qt.ScrollBarAsNeeded)

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
    def _titled(title, widget, corner=None, *, title_floor_chars=None):
        """A titled panel. ``corner`` is an optional widget pinned to the top-right of
        the title row (e.g. the detail-channel dropdown or the result-channel picker).
        ``title_floor_chars``: see :func:`respmech.ui.panel.titled_panel`.

        Lifted out to :mod:`respmech.ui.panel` (ticket B03) so the Run drawer's "Run log"
        panel can reuse the exact same header treatment instead of a second hand-rolled
        one; this stays as a thin wrapper so every existing ``self._titled(...)`` call
        site here (mechanics/ECG/EMG mixins) is unaffected."""
        from respmech.ui.panel import titled_panel
        return titled_panel(title, widget, corner, title_floor_chars=title_floor_chars)

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- Run drawer (ticket B03) ---------------------------------------------
    def install_run_drawer(self, run_screen_widget):
        """Embed Run & results (ticket B03) below the file rail / subtabs workspace,
        replacing its old life as a separate third tab. ``MainWindow`` still builds the
        ``RunScreen`` (it owns the cross-screen wiring, run_started/run_finished included)
        and hands the finished widget in here once, right after construction — this screen
        just gives it a home, so the user can dry-run, run and read the run report without
        ever leaving the file they are looking at.

        Embedded BARE — no extra title chrome, and deliberately NOT through
        :func:`respmech.ui.panel.scrollable`. ``RunScreen`` already carries its own compact
        "Run & results ▸" toggle row as the ONLY thing visible while collapsed (see its
        ``_build``); a second titled-panel header on top of that was pure redundant chrome,
        and ``FitScrollArea`` fights a page whose OWN content collapses/expands (it forces
        a reflow to at least the viewport height on every fit pass, right for Preview &
        QC's static subtab pages, wrong here) — measured, that combination left the drawer
        at 362 px even fully collapsed, an early fit pass against a since-shrunk page
        having stuck in the scroll area's own reported size and fed back into this layout.
        ``RunScreen``'s own collapsed minimum is already small; nothing here needs a
        second, conflicting sizing mechanism on top of it."""
        self._run_drawer = run_screen_widget
        self.layout().addWidget(self._run_drawer)

    def set_run_active(self, active):
        """D24 (UI-overhaul): while a batch is running, lock the WRITE actions this screen
        offers, not the whole surface — MainWindow calls this from RunScreen's
        run_started/run_finished (a run started via the drawer's own Run/Dry-run buttons,
        via "Process & write this file", or via a write-elsewhere retry all funnel through
        those two signals, so one call site here covers all three).

        Deliberately NOT ``self.setEnabled(False)``: ticket B04 already reversed a
        whole-surface lock once (it silently disabled a tab whose stylesheet made
        "disabled" and "enabled" look identical, and it closed browsing/zooming the user
        had every reason to keep doing mid-run). Graphs, zoom and file navigation stay
        live; only ``btn_process_file`` (gated jointly with ``_process_ready``, the "is
        there actually a diagram to write" flag — see its own docstring on why the two
        can't be merged) and breath-toggle clicks (``_toggle_breath`` itself checks
        ``_run_active`` — it is the single funnel both the Mechanics-stack click and every
        EMG plot's click handler call through) are closed, each with a reason a user who
        clicks anyway can actually read."""
        self._run_active = active
        self.btn_process_file.setEnabled(self._process_ready and not active)
        self.btn_process_file.setToolTip(
            "Locked while a run is in progress." if active
            else "Run and write output for the previewed file only.")

    # -- file list (reactive) ----------------------------------------------
    def refresh_files(self):
        s = self.state.settings
        # drop the preview caches only when the input folder/mask ACTUALLY changes (the file
        # set may differ) — not on every entry into the Preview tab, which would defeat the
        # cross-revisit reuse. Per-file freshness tokens in the keys guard in-place edits.
        fm = ((s.input.folder or ""), (s.input.files or ""))
        fm_changed = fm != getattr(self, "_cache_fm", None)   # captured BEFORE the reassign
        if fm_changed:
            from respmech.ui.screens import _preview_cache
            _preview_cache.clear_all()
            self._cache_fm = fm
        prev = self.file_rail.current_filename()
        folder = (s.input.folder or "").strip()
        # B02: the rail is fed the same Manifest Setup's own read-out/QC strip build from
        # (ui/manifest.py, ticket B01) — so a column-count outlier or a sampling-frequency
        # mismatch shows the same caveat here it does on Setup. A per-screen cache (never
        # Setup's own) since this screen's manifest need not be built at the same moments.
        manifest = (build_manifest(folder, s.input.files, s, cache=self._manifest_cache)
                   if folder and os.path.isdir(folder) else None)
        # set_manifest() quietly adopts files[0] as the identity when the rail had none at
        # all yet (a freshly built screen's very first call) — see its docstring. That is
        # the ONLY case this method must not itself drive a switch for: it mirrors the old
        # file_combo, which auto-selected index 0 on first populate as a bare Qt side
        # effect (its own populate ran under blockSignals) and never fired
        # currentTextChanged for it, so nothing downstream ever reacted to that moment
        # either. Losing this made every test that merely constructs a PreviewScreen kick
        # off unsolicited background analysis work it never used to (found via a genuine,
        # reproducible layout-race regression in test_window_fits_screen.py).
        self.file_rail.set_manifest(manifest)
        files = self.file_rail.filenames()
        self._sync_rail_exclusions()
        if prev in files:
            self.file_rail.select_filename(prev)   # re-assert the highlight; identity unchanged
        elif files:
            if prev:
                # a REAL previous file vanished from the (rebuilt) list — switch to the
                # first file instead, same cleanup an interactive switch would do (this
                # DOES emit, since the identity actually changes: _on_file_selected ->
                # _begin_file_switch follows from that signal)
                self.file_rail.select_filename(files[0])
            # else: prev was None (first-ever call) -> set_manifest already quietly
            # adopted files[0] above; nothing more to do here.
        elif prev:
            # the list emptied entirely: nothing for select_filename(None) to emit (there is
            # no new identity), so the switch has to be driven explicitly
            self.file_rail.select_filename(None)
            self._begin_file_switch()
        if not folder:
            self._set_status("Set an input folder in Setup.")
        elif not os.path.isdir(folder):
            self._set_status(f"Input folder not found: {folder}")
        elif not files:
            self._set_status(f"No files match '{s.input.files}' in that folder.")
        elif not fm_changed and prev in files and self._previewed_file == prev:
            # the same file is already selected and drawn, AND the folder/mask that produced
            # `files` hasn't actually changed since last time — a plain tab revisit (every
            # entry into Preview calls refresh_files, see main_window._on_tab_changed) must
            # not reset the status line over what may be a persistent panel result already
            # on screen for it. The fm_changed guard matters even though `prev` still matches:
            # widening the mask from 1 to 3 files leaves the same filename selected, but "1
            # file" would otherwise keep reading on screen against a rail that now holds 3.
            pass
        elif len(files) == 1:
            self._set_status("1 file — everything runs automatically.")
        else:
            self._set_status(f"{len(files)} files — pick one; everything runs automatically.")
        self._update_actions()

    def _sync_rail_exclusions(self):
        """Reflect ``exclude_breaths`` as the rail's exclusion badge for EVERY file
        currently in the rail — for every file it names, not only the one currently
        previewed (ticket requirement: the badge must be visible without selecting the
        file first), AND explicitly zeroed for a file the CURRENT analysis names no
        exclusion for. The zeroing matters because ``set_manifest`` preserves a
        persisting file's rail state across a rebuild: opening a second analysis over
        the same folder/mask, with no exclusion for a file the FIRST analysis had
        excluded breaths in, must not leave that first analysis's stale badge on
        screen."""
        from respmech.core.settings import is_carried_folder
        current_folder = self.state.settings.input.folder
        counts = {e.file: len(e.breaths) for e in self.state.settings.processing.exclude_breaths}
        carried = {e.file: is_carried_folder(e.folder, current_folder)
                  for e in self.state.settings.processing.exclude_breaths if e.breaths}
        for name in self.file_rail.filenames():
            self.file_rail.set_excluded_count(name, counts.get(name, 0),
                                              carried=carried.get(name, False))

    def _refresh_files(self):        # kept for existing wiring + tests
        self.refresh_files()

    def sync_from_settings(self):
        self._refresh_emg_channels()
        self._refresh_ecg_channels()
        self._load_noise_params()
        self._load_ecg_params()
        self._update_emg_tab_visibility()
        self._sync_rail_exclusions()   # a loaded analysis file can bring its own exclusions
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

    def _selected_filename(self) -> str:
        """The single accessor for the previewed file's identity (basename) — every
        other place that needs it (breath toggling, exclusion keying, the noise
        reference picker, 'Process & write this file') reads it from here, never from
        the rail's widgets directly (B02: the identity used to be ``file_combo``'s own
        ``currentText()``, read from half a dozen call sites)."""
        return self.file_rail.current_filename() or ""

    def _current_file(self):
        name = self._selected_filename()
        return os.path.join(self.state.settings.input.folder, name) if name else None

    def _step_file(self, delta: int):
        """P27: move the file selection by ``delta`` (keyboard/▲▼), clamped to the
        visible list — kept as a thin wrapper (existing wiring + tests) over the rail's
        own :meth:`FileRail.step`."""
        self.file_rail.step(delta)

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
        has_file = bool(self._selected_filename())
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
        # "Auto-detect for the batch" itself needs ECG removal on (same requirement Settings.validate
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
        # Blank BEFORE enabling (going off) / BEFORE disabling has no such ordering need going
        # on, but doing it first either way means the two remaining strip fields (capture
        # channel, being a combo, is left showing its stale index) are never simultaneously
        # enabled and still showing the placeholder — no event-loop turn happens between
        # these two loops today, so it is not an observable state either way, but there is no
        # reason to leave that ordering to chance for whoever edits this next.
        for w in (self.ecg_min_height, self.ecg_min_distance):
            w.set_blanked(auto_batch_on)
        for w in (self.ecg_capture_channel, self.ecg_min_height, self.ecg_min_distance,
                  self.btn_ecg_advanced, self.btn_ecg_autosuggest):
            w.setEnabled(not auto_batch_on)
            w.setToolTip(AUTO_BATCH_HINT if auto_batch_on else self._ecg_auto_gated_base_tooltips[w])
        # A PERSISTENT caption, not the status line below: this used to be a _set_status
        # call in the branch that has since been removed here, and an EMG or noise job
        # finishing a moment later silently overwrote it — the same failure this ticket's
        # sibling fixed for the ECG panel titles (_set_ecg_capture_title/_processed_title).
        self._set_ecg_caption(auto_batch_on)
        if not status:
            return
        if has_emg and noise_on and not ecg_on:
            self._set_status(NEEDS_ECG_HINT)
        elif noise_on and not ref:
            self._set_status("Noise reduction is on — click 'Set noise profile' to mark a "
                             "rest span in this file.")
        elif has_file and not ok:
            self._set_status(f"Setup incomplete: {why}")

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
        self._table_model.set_dataframe(None)
        self.campbell.figure.clear(); self.campbell.draw()
        self._forget_campbell()      # the export must not resurrect a cleared diagram
        self.ecg_capture_plot.clear(); self.ecg_processed_plots.clear(); self._ecg_capture_subplots = []
        self._set_ecg_capture_title()   # drop the previous file's R-peak count/state
        self._set_ecg_processed_title()  # and the previous file's ON/OFF + suppression verdict
        self.emg_raw_plots.clear()
        self.emg_plots.clear()
        self._set_trace_key([])       # the legend this replaced was emptied by plots.clear()
        self.emg_psd_canvas.figure.clear(); self.emg_psd_canvas.draw()
        if include_noise:
            self.fidelity_canvas.figure.clear(); self.fidelity_canvas.draw()
            self._update_fidelity_caption(None)   # D09: a cleared panel carries no stability warning
            self._set_fidelity_panel_title(None)   # test-wide result is being recomputed too
            self._noise_has_result = False
        self._reset_breath_state()   # clears _bov/_breaths/_emg_all/result plot/_previewed_file/mech_caption
        # emg_plots.clear() above dropped the region AND the label (D07) — re-attach both
        # and recompute visibility against the file this switch just landed on, rather
        # than blindly re-showing whatever span was last set (found in review: the old
        # _ensure_noise_region() call here could repaint a stale/mismatched band in the
        # window before the async detail render corrects it).
        self._refresh_noise_reference_band()
        self._reset_qc_overview()    # neutral chip + disabled 'Process & write' (B02)
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
        self._table_model.set_dataframe(None)
        self.campbell.figure.clear(); self.campbell.draw()
        self._forget_campbell()      # the export must not resurrect a cleared diagram
        self.ecg_capture_plot.clear(); self.ecg_processed_plots.clear(); self._ecg_capture_subplots = []
        self._set_ecg_capture_title()   # drop the previous analysis' R-peak count/state
        self._set_ecg_processed_title()  # and the previous analysis' ON/OFF + suppression verdict
        self.emg_raw_plots.clear()
        self.emg_plots.clear()
        self._set_trace_key([])       # the legend this replaced was emptied by plots.clear()
        self.emg_psd_canvas.figure.clear(); self.emg_psd_canvas.draw()
        self.fidelity_canvas.figure.clear(); self.fidelity_canvas.draw()
        self._update_fidelity_caption(None)   # D09: a cleared panel carries no stability warning
        self._set_fidelity_panel_title(None)
        self._noise_has_result = False
        self._reset_breath_state()   # clears _bov/_breaths/_emg_all/result plot/_previewed_file
        self._refresh_noise_reference_band()   # re-attach + recompute (see _clear_file_panels, D07)
        self._reset_qc_overview()    # neutral chip + disabled 'Process & write' (B02)
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
            self._set_status(f"Setup incomplete: {why}")
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
                # The diagnosis rides along in the card itself now, not only behind the "i"
                # button — the status line carrying it is transient and, per the shared
                # status-bar ownership rules, may not even be visible from this tab.
                self._overlays[p].show_error(f"{label} failed — {short_error(detail)}", detail)
            if job.kind == "batch":
                # a worker-level failure (before any per-file FileResult existed) — the
                # per-file branch inside _on_batch_result itself covers the more common
                # case of a FileResult that carries an error
                cur = self._selected_filename()
                if cur:
                    self.file_rail.mark_result(cur, ok=False, error=detail)
                self._qc_overview_not_assessed(detail)
            self._update_actions(status=False)
            return
        try:
            self._RENDER[job.kind](result)
        except _FileRunError as e:                     # a per-file analysis error -> "failed"
            detail = str(e)
            self._set_status(f"{label} failed — {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(f"{label} failed — {short_error(detail)}", detail)
            self._update_actions(status=False)
            return
        except Exception:                              # noqa: BLE001 — a rendering bug
            detail = traceback.format_exc()
            self._set_status(f"{label} — display error: {short_error(detail)}")
            for p in _PANELS[job.kind]:
                self._overlays[p].show_error(
                    f"{label} — display error: {short_error(detail)}", detail)
            if job.kind == "batch":
                # a rendering bug in _on_batch_result itself (not a per-file analysis
                # error, which _FileRunError above already covers) — the chip must not
                # keep showing a stale prior verdict over a display-error card
                self._qc_overview_not_assessed(detail)
            self._update_actions(status=False)
            return
        # 'mech' is deliberately excluded: _render_preview_async only just returned from
        # its SYNCHRONOUS first stage here (see _RENDER['mech']) — the two deferred
        # stages that draw the breath overlays and the raw stack are still pending on
        # the event loop, and it stops _PANELS['mech']'s own overlays itself once the
        # LAST one completes (ticket D15). Stopping them here would hide the busy
        # spinner over a panel that has not actually finished drawing yet.
        if job.kind != "mech":
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
        """Dismiss any spinner/error card on these panels — called when a panel
        set is being reset wholesale (a file switch, a settings gate) and
        nothing should be left showing on it, busy or not."""
        for k in keys:
            ov = self._overlays.get(k)
            if ov is not None:
                ov.stop()

    def _clear_panel_errors(self, *keys):
        """Dismiss a stale ERROR card only — used at the start of a render
        (_render_preview_stage1) so a previous attempt's failure never sits over
        fresh content, WITHOUT hiding a spinner a still-in-flight job's overlay
        may legitimately be showing (see BusyOverlay.clear_error)."""
        for k in keys:
            ov = self._overlays.get(k)
            if ov is not None:
                ov.clear_error()

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

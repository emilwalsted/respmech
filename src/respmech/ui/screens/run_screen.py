"""Run & results — run the batch with live output and status.

Since ticket B03 this is no longer its own tab: ``PreviewScreen.install_run_drawer``
embeds this widget below Preview & QC's own file rail + subtabs, collapsed by default
behind a "Run & results ▸" toggle (see ``_build``'s header row and ``_results_section``).
The class and its worker-thread wiring are unchanged by that move.

Computation runs off the GUI thread (``BatchWorker`` on a ``QThread``); the worker
forwards the core's progress events as signals, so the log and progress bar update
live and the UI never freezes. Start/Cancel; a results table + summary fill on
completion. Run/Dry-run are disabled while settings are invalid or a run is busy.
"""
from __future__ import annotations

import copy
import os
import tempfile

from PySide6.QtCore import Qt, Signal, QThread, QTimer, QUrl, QElapsedTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QTableView, QPlainTextEdit,
                               QVBoxLayout, QWidget)

from respmech.ui.dialogs import TextViewerDialog, exception_type_name, short_error
from respmech.ui.file_rail import FileRail
from respmech.ui.flow_layout import ElidingLabel, install_flow as _install_flow
from respmech.ui.manifest import group_readout, manifest_from_filenames
from respmech.ui.panel import titled_panel
from respmech.ui.result_table import ResultTableModel, configure_result_table, resize_result_table
from respmech.ui.validation import blockers as _shared_blockers, matching_files, path_problem
from respmech.ui.workers import BatchWorker, WriteWorker

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover - theme is import-safe, but stay defensive
    _theme = None

# threads that will not stop within the shutdown budget are parked here (kept
# referenced) rather than GC'd while running, which would abort the process.
_ORPHANED_THREADS = []

# Per-file precondition failures the user can act on, and where the control lives. Keyed
# by FileResult.error_kind.
_FIX_HINTS = {
    "VolumeTrendError": "Preview & QC ▸ Mechanics ▸ Advanced… ▸ 'Trend anchor — minimum "
                        "breath depth' / 'Trend anchor — absolute threshold (legacy)' "
                        "(or untick 'Correct end-expiratory trend').",
    "TrimError": "Setup ▸ channel assignment, or 'Invert the flow signal' under "
                 "Preview & QC ▸ Mechanics ▸ Advanced….",
    "NoBreathsError": "Preview & QC ▸ Mechanics ▸ Advanced… ▸ 'Signal used to split "
                      "breaths' and the 'Breath peak' thresholds; or re-include excluded "
                      "breaths by clicking them in the Preview & QC channel view.",
    "DataValidationError": "Setup ▸ channel assignment — the recording's column count "
                           "or contents do not match the channels configured there.",
    "DegenerateBreathError": "Preview & QC ▸ Mechanics ▸ Advanced… ▸ Breath detection "
                             "(peak thresholds / breath-separation buffer); or exclude "
                             "the breath by clicking it in the Preview & QC channel view.",
}


def _fmt_duration(seconds) -> str:
    """Elapsed time, precise to the second: "45s" or "3m 12s"."""
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _fmt_eta(seconds) -> str:
    """A REMAINING-time estimate, deliberately coarser than ``_fmt_duration`` and always
    tilde-prefixed (A07: "an estimate must always read as an estimate") — rounded to the
    minute once it rounds up to a full minute or more, since a rate extrapolated from a
    handful of files does not deserve second-level precision. Rounds first, THEN buckets —
    rounding 59.6s to "~60s" instead of "~1m" would say the same thing two different ways
    depending on which side of a whole minute the raw estimate happened to land."""
    secs = round(max(0.0, seconds))
    if secs < 60:
        return f"~{max(1, secs)}s"
    return f"~{max(1, round(secs / 60))}m"


class RunScreen(QWidget):
    status_changed = Signal(str)
    run_started = Signal()          # a batch run began (main window locks Settings)
    run_finished = Signal()         # the batch run ended
    # ticket C03 point 8: "Choose another folder…" in the temp-output confirmation asks
    # Setup to change the folder rather than writing self.state.settings.output.folder
    # directly — Setup's out_folder QLineEdit is the live widget to_state() reads on the
    # next edit, and a model-only write here would be silently reverted by it (main_window
    # wires this to settings_screen.set_output_folder, mirroring pv.noise_reference_changed).
    output_folder_change_requested = Signal(str)

    def __init__(self, state, file_rail: FileRail | None = None):
        super().__init__()
        self.state = state
        # B03: the ONE shared file rail (Preview & QC's — see MainWindow), so per-file rows
        # are never shown twice. A caller that does not pass one (a standalone construction,
        # as several tests do) gets its own private rail — it simply is not placed anywhere
        # in this screen's own layout (see _build), which is fine for those call sites.
        self.file_rail = file_rail if file_rail is not None else FileRail()
        self._thread = None
        self._worker = None
        self._fatal_msg = None
        self._error_dialog = None
        self._incomplete_shown = False
        self._last_write = False        # did the most recent completed run write files?
        self._last_failed = []          # basenames that failed in the last run (P18 re-run)
        self._only_files = None         # subset the current run is restricted to (P18/P19)
        self._last_plan = None          # the plan the last _append_plan() built (A06)
        self._last_result = None        # the last computed BatchResult, for "write elsewhere"
        # The settings this run actually used, frozen at _start() time — Settings stays
        # editable once a run finishes, so a "write elsewhere" retry (possibly long after,
        # once the analysis has already succeeded but writing failed) must reuse exactly
        # what produced the result, never whatever the user has since typed into Setup.
        self._run_settings_snapshot = None
        self._write_thread = None       # "write elsewhere" thread (A06) — a run must treat
        self._write_worker = None       # this as busy exactly like self._thread
        self._write_target_folder = None   # last successful "write elsewhere" destination
        self._pending_write_folder = None  # folder a write-elsewhere in flight is targeting
        self._last_report_path = None      # B03: resolved once, at enable time — see _run_report_path
        self._run_files = []            # this run's file list, from _append_plan (A07: batch-wide bar)
        self._file_index = 0            # 1-based index of the file currently being computed
        self._ok_file_count = 0         # files that finished computing OK — the write phase's
                                        # figure step draws exactly these, before result exists
        self._cancel_requested = False  # a Cancel click happened this run, whatever phase it landed in
        self._figs_total = None         # None until the first per-file figure event arrives
        self._figs_done = 0
        self._figs_current_file = ""
        # B04 (commitment sheet): the one glob this screen performs (input folder x mask),
        # memoised on (folder, mask, folder mtime) so it costs nothing on every keystroke —
        # see _cached_matching_files. Shared by the commitment sheet, the primary button's
        # enablement/tooltip and the log's empty-state plan summary, so all three read the
        # exact same file list and can never quietly disagree.
        self._matched_files = []
        self._matched_files_key = None
        # D17: set by _confirm_overwrite's "Clear previous results first" checkbox, read
        # and acted on by _start (never inside the dialog itself, so a cancelled dialog
        # can never delete anything).
        self._clear_existing_first = False
        self._build()
        # Called inline, NOT deferred via QTimer.singleShot(0, ...): that was tried first
        # (self-review finding: this resolves the real matched file list and calls
        # core.io.plan.plan_outputs, which lazy-imports pandas/scipy the first time it
        # runs, so doing it synchronously here pulls those imports into every MainWindow
        # construction). It was reverted: a construction-time singleShot(0) is a QTimer
        # that outlives the widget whenever the event loop never spins before the widget
        # is destroyed (true of nearly every unit test that builds a RunScreen/MainWindow
        # and closes it without pumping events) — hundreds of them accumulate across a
        # full test run and ALL fire in a burst the first time something ELSE finally
        # calls processEvents(), each invoking a bound method on an already-destroyed
        # widget. Measured: a reproducible segfault in a completely unrelated test
        # (test_section_flow.py), present in 3 consecutive full-suite runs, gone the
        # instant this went back to a plain call. `_start()`'s own `_append_plan` already
        # pays this same import cost on every Run/Dry-run click; this just pays it once,
        # at construction, like the rest of this screen's setup.
        self._update_plan_summary()
        self.refresh_actions()

    def _build(self):
        # Collapsed, this screen is ONE summary row, and 11 px of card padding above and
        # below a 26 px row is 22 px of the workspace's vertical budget spent on air: the
        # noise tab's "thirds" guard measured chrome+settings at 43% of an 850 px window
        # on the Windows runner (test_the_noise_tab_divides_into_thirds, red 10-08-2026),
        # and the drawer's padding was the largest single compressible piece. So the
        # margins now FOLLOW the drawer's state — a slim (11, 3, 11, 3) while it is the
        # always-on row B03 promised would cost "as close to nothing as this screen can
        # make it", and the full (11, 11, 11, 11) card framing (matching Setup) only while
        # the section is actually open. _on_toggle_results owns the switch; a run's
        # auto-expand goes through the same toggle, so there is exactly one path.
        root = QVBoxLayout(self); root.setContentsMargins(11, 3, 11, 3)
        self._root_lay = root

        # B03: a single compact row is ALL that shows outside the collapsible section below
        # — everything else (action bar, io-info, progress, log, averages table) lives
        # inside ``_results_section``, collapsed by default. Embedding Run & results
        # permanently under Preview & QC's own file rail + subtabs shrinks the window's
        # OTHER carefully-tuned proportions for every pixel this screen keeps always-on
        # (measured: even just the action bar + two io-info lines, ~140 px collapsed, was
        # enough to push test_the_noise_tab_divides_into_thirds' "chrome+settings" share
        # from ~33% to ~45% of an 800 px window) — so the collapsed footprint has to be as
        # close to nothing as this screen can make it, not merely "smaller than before".
        # A run auto-expands this (see _set_running), so nobody has to remember the toggle
        # exists just to see their own results.
        header_row = QHBoxLayout()
        # ElideRight, not ElideMiddle: this line's own most useful prefix — the file COUNT —
        # sits at the very start ("N files from …"), unlike read_info/write_info (a single
        # path, where the informative part is the tail). Middle-eliding it risked cutting
        # through the arrow separator and reading as one fused, ambiguous path.
        self._drawer_summary = ElidingLabel(mode=Qt.ElideRight)
        self._drawer_summary.setProperty("status", "muted")
        # "&&", not "&": Qt reads a lone ampersand in button text as a mnemonic marker and
        # eats it, underlining whatever follows. Here that is a SPACE, so the button rendered
        # as "Run _results ▸" — visible in the shipped v2.4.0 and in the documentation
        # screenshots taken from it. Every other ampersand in this app's widget text is
        # already doubled for the same reason ("Preview && QC", "Process && write this file").
        self.btn_toggle_results = QPushButton("Run && results ▸")
        # The compact button style (6 px vertical padding vs the default): this button IS
        # the collapsed drawer's whole height, so its padding is workspace budget — same
        # thirds-guard arithmetic as the margins note above, worth ~6-8 px more.
        self.btn_toggle_results.setProperty("compact", True)
        self.btn_toggle_results.setCheckable(True)
        self.btn_toggle_results.setAutoDefault(False)
        self.btn_toggle_results.setToolTip(
            "Show the run controls, log and per-study averages (or hide them again).")
        self.btn_toggle_results.toggled.connect(self._on_toggle_results)
        header_row.addWidget(self._drawer_summary, 1)
        header_row.addWidget(self.btn_toggle_results, 0)
        root.addLayout(header_row)

        self._results_section = QWidget()
        section = QVBoxLayout(self._results_section)
        section.setContentsMargins(0, 8, 0, 0); section.setSpacing(6)
        self._results_section.setVisible(False)
        root.addWidget(self._results_section, 1)

        # The commitment sheet (ticket B04): an always-visible summary of what the primary
        # action below is about to commit to — file count/source, planned output count/
        # destination, and every reason it can't start yet (or a plain "Ready to run.").
        # This, plus the primary button's own tooltip (see refresh_actions), is the ONLY
        # gating left after B04 — no tab, no card and no disclosure stage hides anything
        # any more.
        self._commitment = QLabel("")
        self._commitment.setWordWrap(True)
        self._commitment.setProperty("banner", True)
        self._commitment.setProperty("status", "info")
        section.addWidget(self._commitment)

        # A WRAPPING row, not a QHBoxLayout. Measured on Windows font metrics these five
        # buttons sum to 1184 px, and a plain row's minimum IS that sum — which Qt propagates
        # to the window, so this bar alone set the whole application's 1234 px minimum width
        # (more than Preview's 997 px). Wrapping makes the minimum the widest SINGLE button
        # instead. Exactly the defect flow_layout.py was written for; see its docstring.
        # The old trailing stretch is gone: a wrapping row has no meaningful stretch, so the
        # buttons now read left-to-right in priority order rather than splitting left/right.
        self.action_bar = QWidget()
        bar = _install_flow(self.action_bar, h=8, v=6)
        self.btn_run = QPushButton("Run batch")
        self.btn_dry = QPushButton("Dry run (no files written)")
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.setEnabled(False)
        self.btn_rerun = QPushButton("Re-run failed")
        self.btn_rerun.setEnabled(False)    # enabled after a run that had failures (P18)
        self.btn_rerun.setToolTip("Re-run only the files that failed in the last run.")
        self.btn_open = QPushButton("Open output folder")
        self.btn_open.setEnabled(False)     # enabled once a run has written results
        self.btn_show_plan = QPushButton("Show full output plan…")
        self.btn_show_plan.setEnabled(False)     # enabled once a pre-flight plan exists
        self.btn_write_elsewhere = QPushButton("Write results to another folder…")
        self.btn_write_elsewhere.setEnabled(False)  # enabled only when writing just failed
        self.btn_write_elsewhere.setToolTip(
            "Available when the analysis finished but writing its results failed. Writes "
            "the SAME, already-computed results to a folder you choose — no re-analysis.")
        self.btn_show_report = QPushButton("Show run report")
        self.btn_show_report.setEnabled(False)   # enabled once a run has written run-report.txt
        self.btn_show_report.setToolTip(
            "Available once a run has written its own run-report.txt — a plain-English "
            "account of what was read, kept, skipped and written.")
        if _theme is not None:
            _theme.make_primary(self.btn_run)
        self.btn_run.clicked.connect(lambda: self._start(write=True))
        self.btn_dry.clicked.connect(lambda: self._start(write=False))
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_rerun.clicked.connect(self._rerun_failed)
        self.btn_open.clicked.connect(self._open_output_folder)
        self.btn_show_plan.clicked.connect(self._show_full_plan)
        self.btn_write_elsewhere.clicked.connect(self._write_elsewhere)
        self.btn_show_report.clicked.connect(self._show_run_report)
        for _b in (self.btn_run, self.btn_dry, self.btn_cancel, self.btn_rerun, self.btn_open,
                  self.btn_show_report, self.btn_show_plan, self.btn_write_elsewhere):
            bar.addWidget(_b)
        section.addWidget(self.action_bar)

        # B03: what a run would touch — the one thing this screen used to say nothing
        # about until a dry run or a real run had already been started. Middle elision (a
        # long shared network path elides worst at the end, since the leaf name is what
        # identifies it) with the unabridged path always one hover away. Lives inside the
        # collapsible section (see _build's header_row above); the always-visible summary
        # line keeps a condensed version of the same two facts.
        self.read_info = ElidingLabel(mode=Qt.ElideMiddle)
        self.read_info.setProperty("status", "muted")
        self.write_info = ElidingLabel(mode=Qt.ElideMiddle)
        self.write_info.setProperty("status", "muted")
        section.addWidget(self.read_info)
        section.addWidget(self.write_info)

        self.progress = QProgressBar()
        # The bar's own centred % text is drawn in one colour over a two-tone track (dark
        # accent fill + light trough), so it can never contrast with both — black on dark
        # blue in light mode. Hide it and put the % in the status label below, which the
        # theme keeps readable in either mode.
        self.progress.setTextVisible(False)
        # B03: slimmer, and hidden whenever nothing is running (see _set_running) — a bar
        # that spans the whole width at ~26 px, sitting at a static full accent colour once
        # a run finishes, used to be the single most prominent thing on an otherwise idle
        # screen while claiming the opposite.
        self.progress.setMaximumHeight(6)
        self.progress.hide()
        section.addWidget(self.progress)
        # Status is shown ONLY in the main-window bottom status bar now (like Setup and
        # Preview — see main_window._on_screen_status): this was the one screen that showed
        # its status twice at once, here AND in the shared bar. The label stays as the state
        # holder _set_status writes (and tests read pv/rn.status), just never placed on screen.
        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)
        section.addWidget(self.status)
        self.status.hide()

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        # B03: a muted empty state that explains the one distinction that actually matters
        # before anything has run — dry run previews, Run batch writes — instead of a blank
        # box whose only other explanation lived in a status label the user may never visit.
        self.log.setPlaceholderText(
            "Nothing has run yet.\n\n"
            "Dry run (no files written) previews exactly what a run would read and write, "
            "without changing anything on disk. Run batch does the same read, and actually "
            "writes the results.")
        log_panel = titled_panel("Run log", self.log)

        # P20/B02/B03: per-file results live in the ONE shared file rail (Preview & QC's —
        # see MainWindow/PreviewScreen.install_run_drawer), not a second copy here — a file's
        # verdict/breath count/exclusions must read the same wherever it is shown.
        average_caption = QLabel("Average per file — the rows written to data/Average breathdata.xlsx")
        average_caption.setProperty("role", "heading")
        self.table = QTableView()           # the averaged-metrics table
        self._table_model = ResultTableModel()
        self.table.setModel(self._table_model)
        configure_result_table(self.table)
        # a table with zero rows already reads as empty via its column headers alone; a
        # muted line under the caption additionally SAYS so instead of leaving it to a
        # first-time user to infer from a blank grid.
        self._table_empty_hint = QLabel("No batch run yet — this fills in after Run batch or Dry run.")
        self._table_empty_hint.setProperty("status", "muted")
        table_col = QWidget(); table_lay = QVBoxLayout(table_col)
        table_lay.setContentsMargins(0, 0, 0, 0); table_lay.setSpacing(4)
        table_lay.addWidget(average_caption)
        table_lay.addWidget(self._table_empty_hint)
        table_lay.addWidget(self.table, 1)
        self._table_model.modelReset.connect(self._update_table_empty_hint)
        self._update_table_empty_hint()

        section.addWidget(log_panel, 2)
        section.addWidget(table_col, 1)

        # Heartbeat for the write phase. The busy progress bar animates on its own, but the
        # figure step can be tens of seconds of silence, and we do not want the window's only
        # sign of life to depend on whether the styled bar animates on a given platform. This
        # ticking elapsed counter is a guaranteed-visible "still working" signal, and it also
        # tells the user how long the write is taking.
        self._write_clock = QElapsedTimer()
        self._write_stage = ""
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._tick_write)
        # D20: a clock over the WHOLE run (compute + write), restarted in _start once the
        # user's own guard-dialog time (overwrite confirmation etc.) is behind it — unlike
        # self._write_clock above, which only covers the uninterruptible write phase and
        # would report the wrong (too short) duration for the finished-status line.
        # Started here too (not just restarted in _start): QElapsedTimer.elapsed() is
        # documented as undefined on a timer that was never started, and a handful of
        # existing tests call _on_finished() directly without going through _start() first
        # (self-review finding — no crash observed, but an unstarted timer's elapsed() can
        # read as a large, meaningless value that _fmt_duration then silently clamps to
        # "0s", a wrong-but-quiet duration rather than an undefined one).
        self._run_clock = QElapsedTimer()
        self._run_clock.start()

    # -- helpers ------------------------------------------------------------
    def _settings_ok(self):
        try:
            self.state.settings.validate()
            return True, ""
        except Exception as e:              # noqa: BLE001
            return False, str(e)

    def _cached_matching_files(self):
        """``matching_files`` over the current input folder/mask, memoised on
        (folder, mask, folder mtime) — the ONE glob this screen performs. The commitment
        sheet, the primary button's enablement/tooltip and the log's empty-state plan
        summary all read this instead of scanning the folder again (ticket B04): a real
        directory scan is worst on exactly the kind of network drive this whole screen
        already has to be careful about (see ``_update_io_info``'s own file-rail-not-glob
        approach for the same reason). ``_start`` still runs the full, write-probing
        ``validation.path_problem`` a moment later regardless, so a cache that is one
        heartbeat stale can at worst delay this screen noticing an edit by one keystroke
        — never let an actually-broken run start."""
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        mask = s.input.files or "*.*"
        if not folder or not os.path.isdir(folder):
            self._matched_files, self._matched_files_key = [], None
            return []
        try:
            mtime = os.path.getmtime(folder)
        except OSError:
            mtime = None
        key = (folder, mask, mtime)
        if key != self._matched_files_key:
            self._matched_files_key = key
            try:
                self._matched_files = matching_files(folder, mask)
            except Exception:                      # noqa: BLE001 - display-only, never fatal
                self._matched_files = []
        return self._matched_files

    def _blockers(self):
        """Every reason the primary Run action cannot start yet, worst first, as full
        sentences naming the control to fix (ticket B04) — the ONE list the commitment
        sheet and the primary button's enablement/tooltip both read, so they can never
        name a different blocker than each other. Delegates to ``ui.validation.blockers``
        (ticket B07), the SAME ordered check (collision, then core validation, then path)
        Setup's QC strip now reads too, so the two screens can never disagree either — the
        already-memoised match list keeps this exactly as cheap per keystroke as before."""
        return _shared_blockers(self.state.settings, matches=self._cached_matching_files())

    def refresh_actions(self):
        blockers = self._blockers()
        ok = not blockers
        # A "write elsewhere" retry shares the log/status/result state with a fresh run —
        # letting a new run start while one is still writing would have both threads drive
        # the same widgets at once (found in self-review: not a crash, but a real race).
        busy = self._thread is not None or self._write_thread is not None
        self.btn_run.setEnabled(ok and not busy)
        self.btn_dry.setEnabled(ok and not busy)
        tip = blockers[0] if blockers else ""
        self.btn_run.setToolTip(tip)
        self.btn_dry.setToolTip(tip)
        if not ok:
            self._set_status(f"Setup incomplete: {blockers[0]}")
            self._incomplete_shown = True
        elif self._incomplete_shown and not busy:
            self._set_status("Ready to run.")     # clear the stale warning once valid
            self._incomplete_shown = False
        self._update_io_info()
        self._update_commitment(blockers)

    def _update_commitment(self, blockers=None):
        """The commitment sheet built in ``_build`` (ticket B04): what a run commits to —
        file count/source, planned output count/destination — from the SAME planner
        (``core.io.plan.plan_outputs``) ``write_batch`` itself is measured against, never a
        second hand-rolled count; then every current blocker as a full sentence, or a
        plain statement that the run can start."""
        if blockers is None:
            blockers = self._blockers()
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        out = (s.output.folder or "").strip()
        files = self._cached_matching_files()
        head = f"{len(files)} file{'s' if len(files) != 1 else ''} from {folder or '(input not set)'}"
        if files:
            try:
                from respmech.core.io.plan import plan_outputs
                plan = plan_outputs(s, files, cohort_outputs=True)
                cap = "up to " if plan.is_cap else ""
                head += (f" — will write {cap}{plan.total_count} "
                        f"file{'s' if plan.total_count != 1 else ''} into "
                        f"{out or '(output not set)'}")
            except Exception:                      # noqa: BLE001 - display-only, never fatal
                head += f" — into {out or '(output not set)'}"
        else:
            head += f" — into {out or '(output not set)'}"
        tail = ("⚠ " + "  ·  ".join(blockers)) if blockers else "Ready to run."
        self._commitment.setText(f"{head}\n{tail}")
        self._commitment.setProperty("status", "warn" if blockers else "info")
        st = self._commitment.style()
        if st is not None:
            st.unpolish(self._commitment); st.polish(self._commitment)

    def _update_io_info(self):
        """The two-line read/write summary (B03): folder and mask come straight out of
        Settings (free), and the file COUNT comes from the shared file rail's ALREADY-BUILT
        manifest (:meth:`FileRail.count`, an O(1) read of rows Preview's own scan already
        populated) — never a fresh :func:`matching_files` glob here. This runs on every
        ``settings_changed`` tick (see MainWindow's wiring), so a per-call folder scan would
        be a filesystem round trip per keystroke, worst on exactly the kind of network drive
        this whole screen already has to be careful about elsewhere (``path_problem``)."""
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        mask = s.input.files or "*.*"
        out = (s.output.folder or "").strip()
        n = self.file_rail.count()
        self.read_info.setFullText(
            f"Reading: {n} file{'s' if n != 1 else ''} matching '{mask}' in {folder or '(input folder not set)'}")
        self.write_info.setFullText(f"Writing: {out or '(output folder not set)'}")
        # the always-visible condensed line (collapsed-state summary): the same two facts,
        # one line, so the folder/mask/count/output are on screen even before the section
        # is ever expanded.
        self._drawer_summary.setFullText(
            f"{n} file{'s' if n != 1 else ''} from {folder or '(input not set)'} "
            f"→ {out or '(output not set)'}")

    def _update_plan_summary(self):
        """The log's empty-state placeholder (B03): unlike ``_update_io_info`` above, this
        DOES resolve the actual matched file list so it can name what a run would write —
        via the shared, memoised ``_cached_matching_files`` (B04) rather than a second,
        independent glob. Wired to ``inputs_changed`` in MainWindow (an actual folder/mask
        edit), which is also when the cache itself needs refreshing — a plain
        ``settings_changed`` tick reuses whatever the cache already holds. Deliberately NOT
        ``_append_plan``: that writes into the log itself and headlines "DRY RUN — nothing
        will be written", which would be a lie before any run has actually been asked for."""
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        mask = s.input.files or "*.*"
        out = (s.output.folder or "").strip()
        files = self._cached_matching_files()
        lines = [f"{len(files)} file{'s' if len(files) != 1 else ''} matching '{mask}' "
                f"in {folder or '(not set)'}."]
        total_line = None
        if files:
            try:
                from respmech.core.io.plan import plan_outputs
                plan = plan_outputs(s, files, cohort_outputs=True)
                cap = "up to " if plan.is_cap else ""
                total_line = (f"A run would write {cap}{plan.total_count} "
                              f"file{'s' if plan.total_count != 1 else ''} into "
                              f"{out or '(output folder not set)'}.")
            except Exception:                      # noqa: BLE001 - display-only, never fatal
                total_line = None
        lines.append(total_line or f"Results would be written into {out or '(output folder not set)'}.")
        lines.append("")
        lines.append("Dry run (no files written) previews this without writing anything; "
                     "Run batch writes it for real.")
        self.log.setPlaceholderText("\n".join(lines))

    def _update_table_empty_hint(self):
        self._table_empty_hint.setVisible(self._table_model.rowCount() == 0)

    def _on_toggle_results(self, checked):
        self._results_section.setVisible(checked)
        # Doubled for the same reason as the constructor's label above — this is the OTHER
        # half of the same bug: fixing only one leaves the ampersand reappearing the first
        # time the drawer is opened or closed.
        self.btn_toggle_results.setText("Run && results ▾" if checked else "Run && results ▸")
        # Card framing only while the section is open; collapsed, the drawer is a summary
        # row and pays a summary row's margins — see _build's note on the thirds guard.
        self._root_lay.setContentsMargins(11, 11 if checked else 3, 11, 11 if checked else 3)

    def _run_report_path(self) -> str | None:
        """The most recently written ``run-report*.txt`` in the active output folder, or
        ``None``. A subset write (P18/P19) names its report ``run-report (partial,
        <timestamp>).txt`` rather than the full run's ``run-report.txt`` (A05, so a partial
        write can never silently replace the full run's own record) — the exact name is not
        otherwise known to this screen (``BatchWorker.finished`` delivers only the computed
        ``BatchResult``, not ``write_batch``'s own returned file list), so this reads the
        folder rather than assuming a filename.

        Reads ``self._run_settings_snapshot`` (the settings the JUST-FINISHED run actually
        used), never the live, still-editable ``self.state.settings`` — self-review finding:
        a user is free to change the output folder in Setup the moment a run ends, and the
        live folder reading this against would then resolve to a DIFFERENT study's report,
        not the one this run just wrote. Falls back to live settings only when no run has
        completed yet (so a standalone call before any run still resolves something sane).

        Every filesystem call is inside the same try/except: ``getmtime`` on a candidate
        that a concurrent process deletes between the ``listdir`` and the sort is exactly as
        real a race as ``listdir`` itself failing, and letting it escape here would leave
        ``run_finished`` never emitted (this is called from ``_on_finished`` BEFORE that
        signal) — MainWindow's Settings-lock from ``_on_run_started`` would then never lift."""
        out = (self._write_target_folder or (self._run_settings_snapshot or self.state.settings)
              .output.folder or "").strip()
        if not out or not os.path.isdir(out):
            return None
        try:
            candidates = [f for f in os.listdir(out) if f.startswith("run-report") and f.endswith(".txt")]
            if not candidates:
                return None
            candidates.sort(key=lambda f: os.path.getmtime(os.path.join(out, f)), reverse=True)
            return os.path.join(out, candidates[0])
        except OSError:
            return None

    def _show_run_report(self):
        path = self._last_report_path
        if not path:
            self._set_status("No run report available yet.")
            return
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            text = f"Could not read {os.path.basename(path)}: {e}"
        dlg = TextViewerDialog("Run report", text, self,
                               intro=f"{os.path.basename(path)} — the app's own plain-English "
                                    "account of what this run read, kept, skipped and wrote.")
        dlg.show()
        dlg.raise_()

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- pre-flight & output (P5/P7, plan-driven since A06) ------------------
    def _append_plan(self, write: bool):
        """Log a pre-flight plan so a run is never a black box: what is read, and
        (for a real run) exactly what will be written — or, for a dry run, would be.

        Built from ``core.io.plan.plan_outputs`` — the SAME function ``write_batch``'s own
        figure job list is drawn from — so this can never again drift from what a real run
        actually writes the way the old, hand-maintained list did (measured: it named 4
        files where a real run wrote 14, and never mentioned ``diagnostics/`` at all).

        Reads ``self._run_settings_snapshot`` when ``_start`` has already frozen one for
        this run, falling back to the live ``self.state.settings`` otherwise — a caller
        that wants a preview of the CURRENT settings without going through ``_start`` (a
        test, or a future "preview the plan" affordance) still gets one.

        Returns the resolved file list (after the subset filter) — ``_start`` keeps it as
        ``self._run_files``, the denominator the compute-phase progress bar (A07) needs and
        cannot otherwise get without re-deriving it."""
        s = self._run_settings_snapshot or self.state.settings
        files = matching_files(s.input.folder, s.input.files)
        if self._only_files:                            # a subset / re-run (P18/P19)
            keep = set(self._only_files)
            files = [f for f in files if os.path.basename(f) in keep]
        head = "DRY RUN — nothing will be written." if not write else "RUN — writing output."
        self._append(f"── {head} ──")
        subset = f" (subset: {len(files)} of {len(matching_files(s.input.folder, s.input.files))})" \
            if self._only_files else ""
        self._append(f"Input:  {len(files)} file{'s' if len(files) != 1 else ''} matching "
                     f"'{s.input.files}' in {s.input.folder}{subset}")
        for f in files:
            self._append(f"    • {os.path.basename(f)}")

        from respmech.core.io.plan import plan_outputs
        cohort_outputs = not self._safe_is_subset_run(self._only_files)

        # D19: the SAME grouping read-out Setup shows live under "Group files by",
        # registered here before the run instead of only being visible afterwards by
        # opening the written "By group" sheet. Built from this resolved ``files`` list's
        # basenames -- the SAME set ``run_batch`` actually iterates, unlike Setup's own
        # manifest-derived list before its self-review fix (10-08-2026) mistakenly
        # narrowed it to a UI-side "majority column count" subset; see
        # ``group_readout``'s and ``SettingsScreen._update_group_readout``'s docstrings
        # for that history. Shown only when a cohort summary will actually be written --
        # the same ``save_average and cohort_outputs`` gate ``plan_outputs``/
        # ``write_batch`` use below (self-review fix: the line used to appear
        # unconditionally, describing a "By group" sheet that a subset re-run or
        # save_average=False would never write, directly under an output list that
        # correctly listed no such file -- contradicting this method's own "never a
        # black box" rationale).
        if s.output.data.save_average and cohort_outputs:
            _gr_status, gr_text = group_readout([os.path.basename(f) for f in files], s)
            if gr_text:
                self._append(f"Grouping: {gr_text}")

        plan = plan_outputs(s, files, cohort_outputs=cohort_outputs)
        self._last_plan = plan
        self.btn_show_plan.setEnabled(True)

        verb = "Would write" if not write else "Will write"
        cap = "up to " if plan.is_cap else ""
        total = plan.total_count
        self._append(f"Output: {verb} {cap}{total} file{'s' if total != 1 else ''} "
                     f"into {s.output.folder}:")
        # Over ~10 files the old per-file listing ran to dozens of near-identical lines
        # (measured: 57 lines at 25 files) — summarise by category instead, with the full
        # list one click away via "Show full output plan…".
        if len(files) > 10:
            for g in plan.groups:
                if g.count == 0:
                    continue
                prefix = "up to " if g.is_cap else ""
                self._append(f"    • {prefix}{g.count} {g.category}")
            self._append("    (see 'Show full output plan…' for the complete file list)")
        else:
            for p in plan.all_paths():
                self._append(f"    • {p}")
        self._append("")     # blank line before the live run log
        return files

    def _show_full_plan(self):
        """The complete, ungrouped file list for the last pre-flight plan (ticket A06):
        the compact summary above is enough to read at a glance, but every path must still
        be one click away."""
        plan = self._last_plan
        if plan is None:
            return
        lines = []
        for g in plan.groups:
            if g.count == 0:
                continue
            prefix = "up to " if g.is_cap else ""
            lines.append(f"{prefix}{g.count} × {g.category}" + (f"  ({g.target})" if g.target else ""))
            lines.extend(f"    {p}" for p in g.paths)
            lines.append("")
        intro = f"Everything the current settings would write into {plan.outputfolder}."
        # A "write elsewhere" redirect (A06 point 7) means the results this plan describes
        # were, in the end, actually written somewhere else — say so, or the dialog would
        # silently point at a folder that never received them (self-review finding).
        if self._write_target_folder and self._write_target_folder != plan.outputfolder:
            intro += (f" (This run's results were instead actually written to "
                     f"{self._write_target_folder}.)")
        dlg = TextViewerDialog("Full output plan", "\n".join(lines).strip(), self, intro=intro)
        dlg.show()
        dlg.raise_()

    def _existing_output(self):
        """(newest_mtime, count) of results already in the output folder, or None.

        Scans ``diagnostics/`` as well as ``data/`` and the root provenance files (A06
        point 4): a folder that holds ONLY previously-written figures — because the
        provenance files or ``data/`` were moved or deleted — used to return None here,
        so a run into it overwrote roughly 35 MB of figures with no warning at all."""
        out = (self.state.settings.output.folder or "").strip()
        if not out or not os.path.isdir(out):
            return None
        hits = []
        for p in (os.path.join(out, "analysis-used.toml"),
                  os.path.join(out, "run-report.txt")):
            if os.path.isfile(p):
                hits.append(p)
        for sub in ("data", "diagnostics"):
            d = os.path.join(out, sub)
            if os.path.isdir(d):
                hits += [os.path.join(d, n) for n in os.listdir(d)
                        if os.path.isfile(os.path.join(d, n))]
        if not hits:
            return None
        newest = max(os.path.getmtime(p) for p in hits)
        return newest, len(hits)

    def _existing_output_categories(self):
        """Per-folder counts of what is already in the output folder — the overwrite
        dialog names these instead of a bare number, so the user sees WHAT is at stake,
        matching the categories the plan itself uses (ticket A06 point 4)."""
        out = (self.state.settings.output.folder or "").strip()
        cats = []
        if not out or not os.path.isdir(out):
            return cats
        root_hits = [n for n in ("analysis-used.toml", "run-report.txt")
                    if os.path.isfile(os.path.join(out, n))]
        if root_hits:
            cats.append(f"{len(root_hits)} provenance file{'s' if len(root_hits) != 1 else ''}")
        for sub, label in (("data", "data workbook/CSV file"),
                           ("diagnostics", "diagnostic figure/audio file")):
            d = os.path.join(out, sub)
            if os.path.isdir(d):
                n = len([x for x in os.listdir(d) if os.path.isfile(os.path.join(d, x))])
                if n:
                    cats.append(f"{n} {label}{'s' if n != 1 else ''}")
        return cats

    def _existing_output_files(self):
        """Every existing result file in the output folder, as a path relative to it — the
        same shape ``core.io.plan.plan_outputs`` writes into ``Plan.all_paths()``. Ticket
        D17's overwrite guard intersects this against a run's own planned outputs so it can
        tell the user which existing files it would actually replace versus leave behind,
        instead of a blanket "will be overwritten" that used to include files a narrower
        run's mask or settings never even touch."""
        out = (self.state.settings.output.folder or "").strip()
        rels = []
        if not out or not os.path.isdir(out):
            return rels
        for name in ("analysis-used.toml", "run-report.txt"):
            if os.path.isfile(os.path.join(out, name)):
                rels.append(name)
        for sub in ("data", "diagnostics"):
            d = os.path.join(out, sub)
            if os.path.isdir(d):
                rels.extend(f"{sub}/{n}" for n in os.listdir(d)
                           if os.path.isfile(os.path.join(d, n)))
        return rels

    def _overwrite_split(self, files):
        """(replaced, remaining) — how many of the output folder's EXISTING result files a
        run over ``files`` would actually replace, versus how many belong to an earlier run
        with a different mask/settings and would be left exactly as they are (ticket D17).
        Compares against ``core.io.plan.plan_outputs`` for ``files`` — the SAME planner the
        commitment sheet and the pre-flight log already read — so this can never name a
        different set of paths than what the run itself is about to attempt."""
        from respmech.core.io.plan import plan_outputs
        existing = set(self._existing_output_files())
        try:
            planned = set(plan_outputs(self.state.settings, files, cohort_outputs=True).all_paths())
        except Exception:                      # noqa: BLE001 - display-only, never fatal
            # self-review finding: an empty `planned` here would report "0 replaced, all
            # will remain" — reassuring wording for a confirmation dialog that exists
            # specifically to warn the user before an overwrite. On the rare planning
            # failure, assume the WORST (everything existing is at risk) rather than the
            # most comfortable answer.
            planned = set(existing)
        return len(existing & planned), len(existing - planned)

    def _clear_existing_output(self) -> int:
        """Ticket D17's "Clear previous results first" checkbox, acted on: remove every
        FILE directly under ``data/`` and ``diagnostics/`` — never anything else, never a
        subfolder — before the worker thread starts. Lives here in the UI layer, not
        ``core.io.writers.write_batch``, which the CLI shares and must never start deleting
        files on its own initiative; only ever called from ``_start`` after
        ``_confirm_overwrite`` has returned with the checkbox ticked.

        Returns the number of files that could NOT be removed (a locked file, a permission
        wall) — self-review finding: a bare ``except OSError: pass`` left a partial clear
        completely silent, which is exactly the kind of dishonesty this ticket exists to
        stop. The caller (``_start``) surfaces a non-zero count in the log/status rather
        than letting the user believe "Clear previous results first" fully succeeded when
        it did not."""
        out = (self.state.settings.output.folder or "").strip()
        if not out:
            return 0
        failed = 0
        for sub in ("data", "diagnostics"):
            d = os.path.join(out, sub)
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        failed += 1
        return failed

    def _safe_is_subset_run(self, only_files) -> bool:
        """``core.pipeline.is_subset_run``, guarded: it re-lists the input folder
        (``os.listdir``), which — like the very same check inside ``BatchWorker.run``
        (see its comment) — can raise on a folder that has gone transiently unreadable
        (a dropped network drive, a permissions change). Unlike the worker's write path,
        both GUI call sites of this (before a run starts, and after one has already
        finished successfully) have no "fail the write" outcome to report through — so on
        any error this defaults to False (treat it as a plain, non-subset write) rather
        than let an unhandled exception escape a Qt slot. ``_start``'s caller already ran
        ``path_problem`` moments earlier, which narrows this to a genuine race; the
        default keeps that race harmless instead of crashing the screen."""
        from respmech.core.pipeline import is_subset_run
        try:
            return is_subset_run(self.state.settings, only_files)
        except OSError:
            return False

    def _overwrite_prompt_text(self, existing, files) -> str:
        """The full-run overwrite dialog's body text (ticket D17), split out from
        ``_confirm_overwrite`` so its content — which of the existing result files THIS
        run would actually replace versus leave behind — can be tested without driving the
        modal itself. Replaces the old blanket "will be overwritten", which named every
        existing file regardless of whether the run about to start would ever touch it."""
        from datetime import datetime
        newest, count = existing
        when = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
        out = self.state.settings.output.folder
        replaced, remaining = self._overwrite_split(files)
        cats = self._existing_output_categories()
        detail = "; ".join(cats) if cats else f"{count} result file{'s' if count != 1 else ''}"
        if remaining:
            split_txt = (f"{replaced} existing result file{'s' if replaced != 1 else ''} will "
                        f"be replaced by this run; {remaining} from an earlier run will "
                        "remain — not touched by the current settings.")
        else:
            split_txt = f"All {replaced} will be replaced by this run."
        return f"{out}\n\nalready contains {detail}, from a run on {when}.\n\n{split_txt}\n\nContinue?"

    def _confirm_overwrite(self, existing) -> bool:
        """The full-run overwrite guard (ticket D17). Offers "Clear previous results
        first" (``self._clear_existing_first``, read and acted on by ``_start`` — never
        here, so a cancelled dialog can never delete anything)."""
        # self-review finding: this used to re-glob the input folder directly instead of
        # reusing the screen's own memoised match list — the ONE glob this screen is
        # supposed to perform (see _cached_matching_files' own docstring).
        files = self._cached_matching_files()
        text = self._overwrite_prompt_text(existing, files)
        self._clear_existing_first = False
        box = QMessageBox(self)
        box.setWindowTitle("Output folder already has results")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(text)
        checkbox = QCheckBox("Clear previous results first (removes everything currently in "
                            "data/ and diagnostics/ before this run starts)")
        box.setCheckBox(checkbox)
        yes = box.addButton(QMessageBox.StandardButton.Yes)
        box.addButton(QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.exec()
        if box.clickedButton() is not yes:
            return False
        self._clear_existing_first = checkbox.isChecked()
        return True

    def _cohort_output_names(self) -> str:
        """The cohort-level output names to mention in subset-write UI text, joined into one
        readable phrase — shared by the subset dialog and the post-write log note so the two
        can never name a different set of artefacts."""
        bits = ["Average breathdata.xlsx", "Cohort summary.xlsx"]
        if self.state.settings.output.diagnostics.save_pv_individual:
            bits.append("the cohort Campbell figure")
        return bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]

    def _full_run_timestamp(self) -> str | None:
        """When the study's cohort-level outputs were last built by a FULL run, or ``None``
        if they never have been — anchored to ``data/Average breathdata.xlsx``'s own mtime, a
        file only a full run ever writes. Deliberately NOT ``_existing_output()`` (whose
        "newest" mtime is a max over every file under ``data/``, including this very output
        folder's own PER-FILE artefacts): after even one prior subset write, that generic
        mtime reflects the subset write, not the full run, and would make subset UI text
        claim "from a full run on <time>" for a time that was never a full run."""
        out = self.state.settings.output.folder
        avg_path = os.path.join(out, "data", "Average breathdata.xlsx")
        if not os.path.isfile(avg_path):
            return None
        from datetime import datetime
        return datetime.fromtimestamp(os.path.getmtime(avg_path)).strftime("%Y-%m-%d %H:%M")

    def _confirm_overwrite_subset(self, files) -> bool:
        """The subset-specific overwrite guard (A05): the generic ``_confirm_overwrite``
        reads as "yes, replace the results I just adjusted", but for a subset write it
        silently rebuilt the WHOLE study's cohort spreadsheets from the one or two files
        being processed — exactly the data-loss bug this ticket exists to fix. This dialog
        names what actually gets written or replaced, and states plainly what is left alone
        instead of a generic file count."""
        out = self.state.settings.output.folder
        n = len(files)
        names = ", ".join(files)
        subj = "this file's" if n == 1 else "these files'"
        cohort_txt = self._cohort_output_names()
        when = self._full_run_timestamp()
        cohort_state = (f"they still reflect the full run at {when}" if when
                        else "no full run has produced them yet")
        ans = QMessageBox.question(
            self, "Write a subset into existing results",
            f"{out}\n\n"
            f"This will write or replace {subj} enabled per-file outputs — breathdata "
            f"workbook, processed CSV, diagnostic figures — for {n} file{'s' if n != 1 else ''}: "
            f"{names}.\n\n"
            f"{cohort_txt}, built from the whole study, will NOT be touched — {cohort_state}. "
            f"This write's own run report and settings snapshot are saved under a separate, "
            f"timestamped name rather than replacing the full run's.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return ans == QMessageBox.Yes

    def _subset_cohort_note(self, files) -> str:
        """A persistent, plain-English line for the log after a subset write (A05):
        Read back from DISK (the just-written or left-alone ``Average breathdata.xlsx``),
        never tracked in memory, so it is correct even the first time this output folder is
        touched, or after the app has been restarted between runs."""
        n = len(files)
        plural = "s" if n != 1 else ""
        # "in this write", not "since <time>": this counts only the files just processed,
        # not a running total across every subset write since the last full run (which
        # would need state this app does not keep) — phrasing it as a total would overclaim.
        when = self._full_run_timestamp()
        note = (f"{n} file{plural} processed in this write.")
        if when:
            return (f"{self._cohort_output_names()} are still from the full run at {when} — "
                    f"{note} Run the full batch to update them.")
        return (f"No full-batch {self._cohort_output_names()} exist yet — "
                f"{note} Run the full batch to create them.")

    def _open_output_folder(self):
        # a successful "write elsewhere" (A06) points here at its NEW destination until the
        # next run starts fresh — see _start(), which clears this back to None.
        out = (self._write_target_folder or self.state.settings.output.folder or "").strip()
        if out and os.path.isdir(out):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(out)))
        else:
            self._set_status("Output folder does not exist yet.")

    def _confirm_temp_output(self):
        """Ticket C03 point 8: a REAL run (write=True) whose output folder resolves inside
        the built-in sample's own temp subfolder writes into a place the operating system
        may delete without warning — a saved-but-forgotten sample-derived analysis
        (``open_sample_analysis`` writes both folders under
        ``tempfile.gettempdir()/respmech_sample``), reopened as a plain saved .toml months
        later with nothing on screen left to say "sample" (see ``AppState.load_toml``).
        Deliberately PATH-based, not a flag check on ``AppState.is_sample``: that is
        exactly the case this guards against, since the flag has already been forgotten by
        the time a saved analysis is reopened, but the output folder's actual location has
        not changed.

        Scoped to the SAMPLE's own subfolder rather than "anywhere under the OS temp
        directory": ``tempfile.gettempdir()`` is also where pytest's own ``tmp_path``
        fixture lives on this platform, and a broader check fired on every real-run test
        in this suite that uses ``tmp_path`` as its output folder — popping a real,
        un-mocked confirmation dialog and hanging the test on its modal event loop
        (caught in self-review: a real, wholesale test-suite hang, not a style nit).
        The sample's own subfolder name is specific enough that this narrower check still
        catches every case the ticket names. Returns True to proceed."""
        out = (self.state.settings.output.folder or "").strip()
        if not out:
            return True
        out_abs = os.path.abspath(out)
        sample_root = os.path.abspath(os.path.join(tempfile.gettempdir(), "respmech_sample"))
        try:
            under_temp = os.path.commonpath([out_abs, sample_root]) == sample_root
        except ValueError:                      # different drives on Windows
            under_temp = False
        if not under_temp:
            return True
        choice = self._ask_temp_output_choice()
        if choice == "continue":
            return True
        if choice == "choose":
            d = QFileDialog.getExistingDirectory(self, "Choose output folder", out_abs)
            if d:
                # Ask Setup to make the change (see output_folder_change_requested's own
                # docstring for why this screen must not write the model directly) and let
                # the user press Run again against the commitment sheet's fresh plan,
                # rather than silently carrying on with a folder just picked in passing.
                self.output_folder_change_requested.emit(d)
        return False

    def _ask_temp_output_choice(self):
        """The actual modal — split out from ``_confirm_temp_output`` so a test can
        substitute the answer directly ('continue' | 'choose' | 'cancel') instead of
        fighting QMessageBox's own modal event loop, the same pattern already used for
        ``_confirm_overwrite``/``_confirm_overwrite_subset`` elsewhere in this screen."""
        box = QMessageBox(self)
        box.setWindowTitle("Temporary output folder")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            "This writes into a temporary folder that your operating system may delete.")
        btn_continue = box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        btn_choose = box.addButton("Choose another folder…", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(btn_continue)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_continue:
            return "continue"
        if clicked is btn_choose:
            return "choose"
        return "cancel"

    # -- run lifecycle ------------------------------------------------------
    def _start(self, write: bool, only_files=None):
        if self._thread is not None or self._write_thread is not None:
            # D24: this used to be a silent no-op. The buttons that normally reach here
            # (Run/Dry-run, "Process & write this file") are already disabled while busy
            # (refresh_actions()/PreviewScreen.set_run_active — see main_window's
            # run_started/run_finished wiring), so a live click cannot land here under
            # ordinary use. But a queued click delivered right on the busy/idle boundary,
            # or a future caller that skips the button, must not vanish without a trace —
            # see D24's ticket for two concrete cases (a Preview breath-exclusion click and
            # "Process & write this file") that used to look like they worked while a run
            # silently ignored them.
            self._set_status("A run is already in progress.")
            return
        ok, why = self._settings_ok()
        if not ok:
            self._set_status(f"Cannot run: {why}")
            return
        # Real filesystem check, INCLUDING an actual write probe (create+write+delete a
        # temp file) for both Run and Dry run — ticket A06 point 6: a folder that looked
        # fine to path_problem's earlier checks could still not be writable (a read-only
        # network share), which used to surface only after the whole batch had finished
        # computing, deep inside write_batch's os.makedirs.
        problem = path_problem(self.state.settings, probe_write=True)
        if problem:
            self._set_status(f"Cannot run: {problem}")
            return
        # ticket C03 point 8: only a REAL write risks losing results to OS temp cleanup —
        # a dry run touches no disk, so there is nothing here for it to protect.
        if write and not self._confirm_temp_output():
            self._set_status("Run cancelled — output folder is under the system temp directory.")
            return
        # overwrite guard: a real run into a folder that already holds results asks first
        clear_failed = 0
        if write:
            existing = self._existing_output()
            if existing:
                is_subset = self._safe_is_subset_run(only_files)
                if is_subset:
                    names = [os.path.basename(f) for f in only_files]
                    confirmed = self._confirm_overwrite_subset(names)
                else:
                    confirmed = self._confirm_overwrite(existing)
                if not confirmed:
                    self._set_status("Run cancelled — existing results kept.")
                    return
                # D17: only ever fires for a full run whose dialog checkbox was ticked —
                # the subset dialog has no such checkbox, and deliberately never clears
                # anything (a subset write's whole point is to leave the rest of the
                # study's results alone).
                if not is_subset and self._clear_existing_first:
                    clear_failed = self._clear_existing_output()
                    self._clear_existing_first = False
        self._only_files = list(only_files) if only_files else None
        # D20: the whole-run clock starts here — after every guard dialog the user may have
        # sat on (overwrite confirmation, temp-output warning) — not at the top of _start,
        # so its elapsed time reflects the run itself, not how long a dialog sat open.
        self._run_clock.restart()
        self.log.clear(); self.progress.setRange(0, 0)  # busy until first file
        if clear_failed:
            # self-review finding: a silent partial clear would say nothing while leaving
            # stale files behind — exactly the dishonesty D17 exists to stop. Logged here
            # (after the clear, before the pre-flight plan) rather than via _set_status,
            # which the very next progress event would immediately overwrite.
            self._append(f"NOTE: {clear_failed} file{'s' if clear_failed != 1 else ''} in "
                         "data/ or diagnostics/ could not be removed before this run "
                         "(permission denied or file in use) — they may still be present "
                         "alongside this run's own output.")
        self._fatal_msg = None
        self._last_write = write
        self._last_result = None
        self._write_target_folder = None      # a new run invalidates any earlier redirect
        self.btn_write_elsewhere.setEnabled(False)
        # B03 self-review: a stale "Show run report" must not survive into a run that could
        # fail before writing anything (or into a different output folder) — the button, and
        # the path it would open, are only ever trusted for the run that just enabled them.
        self._last_report_path = None
        self.btn_show_report.setEnabled(False)
        # A07: fresh per-run progress/cancel state. _file_index counts file_start events
        # (1-based, "file i of N"); _ok_file_count counts file_done events and becomes the
        # figure step's denominator, since the write phase has no BatchResult yet to read
        # len(result.ok_files) from — only the compute phase's own events can tell us.
        self._file_index = 0
        self._ok_file_count = 0
        self._cancel_requested = False
        self._figs_total = None
        self._figs_done = 0
        self._figs_current_file = ""
        # Snapshot the settings ONCE, used for both the pre-flight plan below and the
        # worker — so a "write elsewhere" retry later (Settings stays editable once a run
        # finishes) reuses exactly what produced the result, never a live edit made since.
        self._run_settings_snapshot = copy.deepcopy(self.state.settings)
        self._run_files = self._append_plan(write)      # pre-flight: what will be read/written
        # D20: the status line said nothing at all at the start of a run — whatever the
        # PREVIOUS run left behind (including a stale "Run cancelled…") stayed on screen
        # through the whole silent pre-file phase (shared noise profile + first file load).
        n = len(self._run_files)
        self._set_status(f"Starting — analysing {n} file{'s' if n != 1 else ''}…")
        self._set_running(True)
        self.run_started.emit()

        self._thread = QThread()
        self._worker = BatchWorker(self._run_settings_snapshot, write=write,
                                   only_files=self._only_files)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Explicitly queued (see preview_screen's worker connects): AutoConnection can
        # resolve to DIRECT under a fast-emit race, which would drive the progress bar
        # and status bar from the batch worker thread — a cross-thread QBasicTimer on
        # cocoa — and run _on_finished's thread.wait() ON the thread being waited for.
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.file_failed.connect(self._on_file_failed, Qt.QueuedConnection)
        self._worker.failed.connect(self._on_fatal, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._thread.start()

    def _on_file_failed(self, f, e):
        self._append(f"  {f}: FAILED — {e}")

    def _on_fatal(self, msg):
        self._fatal_msg = msg
        # ticket D16: `msg` is the WHOLE traceback.format_exc() — appending it here used to
        # dump ~20 lines of stack frames and carets into the always-visible log for what is
        # usually a data problem, not a program crash. The log now gets the one sentence a
        # physiologist can act on; the full trace is still one click away in the copyable
        # TextViewerDialog _show_error_window opens from self._fatal_msg once the run ends.
        self._append(f"ERROR: {short_error(msg)} (full technical detail: see the error window)")

    def closeEvent(self, ev):
        """Self-cleanup for a ``RunScreen`` closed WITHOUT a ``MainWindow`` around it (point
        6 durability). Qt never delivers ``closeEvent`` to a child widget when its PARENT
        window closes — ``MainWindow.closeEvent`` orchestrates ``shutdown()`` explicitly for
        that reason — so this override only fires when a ``RunScreen`` is itself the
        top-level widget being closed: standalone in a test (6 of them, per the point 6
        investigation), or any future composition without a ``MainWindow``. ``shutdown()``
        is safe to call twice (``self._worker``/``self._thread``/``self._write_thread`` are
        already ``None`` on a second pass), so a screen that is BOTH orchestrated by a
        parent's closeEvent AND later closed directly stays safe."""
        self.shutdown()
        super().closeEvent(ev)

    def shutdown(self, wait_ms=5000):
        """Cancel and join a running batch (called from MainWindow.closeEvent) so
        the worker thread is never destroyed while running.

        Stops the write-phase heartbeat FIRST (self-review, point 6 durability): a
        screen closed WITHOUT a MainWindow around it (this method is now also reachable
        via closeEvent, not only via MainWindow's orchestration) could otherwise leave a
        1 s ``QTimer`` armed on an instance that is closed but, per this suite's own
        never-delete-a-closed-window policy, never destroyed — ticking ``_tick_write``
        against a screen nothing shows again. Same hazard class
        ``PreviewScreen.shutdown()`` already disarms its own debounce timer for first."""
        self._heartbeat.stop()
        if self._worker is not None:
            try:
                self._worker.cancel()
            except Exception:                       # pragma: no cover
                pass
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(wait_ms):
                _ORPHANED_THREADS.append((self._thread, self._worker))
        self._thread = None
        self._worker = None
        # "Write elsewhere" (A06) has no cancel() — like write_batch's own write phase, it
        # is not cancellable mid-figure — but it must still be joined, not abandoned.
        if self._write_thread is not None:
            self._write_thread.quit()
            if not self._write_thread.wait(wait_ms):
                _ORPHANED_THREADS.append((self._write_thread, self._write_worker))
        self._write_thread = None
        self._write_worker = None

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self._cancel_requested = True
            # Read the worker's OWN flag, not self._heartbeat.isActive(): the heartbeat only
            # starts once the "writing" ProgressEvent has made its way through Qt's
            # QueuedConnection from the worker thread, which is unavoidably asynchronous
            # (this thread must never touch widgets directly). A click landing in that gap —
            # after the worker has already committed to the uninterruptible write, before the
            # GUI has been told — would otherwise still get the plain "cancelling…" line this
            # ticket exists to stop showing for a click that arrived too late to cancel
            # anything. _worker._writing is a plain attribute the worker sets as literally its
            # first action in that phase (see workers.py), so this is right the moment the
            # worker commits, not whenever Qt gets around to delivering the signal for it.
            if getattr(self._worker, "_writing", False):
                # The heartbeat itself may not have started ticking yet in that same gap, so
                # an elapsed time is only meaningful once it has; the wording still stands
                # without one, and the very next heartbeat tick (at most ~1s later) will show
                # it as soon as it exists.
                if self._heartbeat.isActive():
                    secs = self._write_clock.elapsed() // 1000
                    self._set_status(
                        "Cancel received; finishing the output write "
                        f"(this cannot be interrupted)… {secs}s")
                else:
                    self._set_status(
                        "Cancel received; finishing the output write (this cannot be interrupted)…")
            else:
                self._append("cancelling…")

    def _tick_write(self):
        """Once-a-second heartbeat during the write phase — see __init__."""
        self._set_status(self._write_status_text())

    def _write_status_text(self) -> str:
        """The write-phase status line — shared by the heartbeat tick and the per-file
        figure-progress event, so neither can show a line the other would immediately
        overwrite. Determinate once the first "figures — <file>" stage event has arrived
        AND the compute phase told us how many files there are to draw for
        (``self._ok_file_count``, ``self._figs_total``'s source); otherwise the same
        generic elapsed-time line as before this ticket (A07)."""
        secs = self._write_clock.elapsed() // 1000
        if self._figs_total:
            n, done = self._figs_total, self._figs_done
            eta = ""
            # A rate extrapolated from a SINGLE file is noise, not an estimate — one unusually
            # slow or fast figure would swing it wildly. Two or more gives at least a rough
            # average, matching A07's "only once there are enough files for it to mean anything".
            if 1 < done < n:
                rate = self._write_clock.elapsed() / 1000.0 / done
                eta = f" · {_fmt_eta(rate * (n - done))} left"
            return (f"Writing diagnostic figures — {done} of {n} files "
                    f"({self._figs_current_file}) · {_fmt_duration(secs)} elapsed{eta}")
        stage = f" — {self._write_stage}" if self._write_stage else ""
        return f"Writing output… {secs}s{stage}, cannot be interrupted"

    def _on_progress(self, ev):
        if ev.kind == "file_start":
            self._file_index += 1
            self._append(f"\n{ev.file}: {ev.message}")
        elif ev.kind == "stage":
            self._append(f"  {ev.message}…")
            # D20: the silent pre-file phase (shared noise profile, first file load) used to
            # leave the status label showing whatever the previous file_start/breath event —
            # or the previous run entirely — had last written, for up to a second and a half
            # on real data. Set unconditionally; the heartbeat block below (unchanged) then
            # immediately overwrites this with its own, more specific write-phase text
            # whenever it is active, so the heartbeat still owns the label during writing.
            self._set_status(f"{ev.file or 'Batch'}: {ev.message}…")
            if self._heartbeat.isActive():        # keep the heartbeat's label current
                self._write_stage = ev.message
                if ev.message.startswith("figures — "):
                    # One event per file (writers._emit / plots._write_figures_impl). On a
                    # spawned child this now arrives via the Manager().Queue() relay in
                    # _figure_process.py, not just on the in-process path — UNLESS building
                    # that Manager itself failed, in which case the child still writes every
                    # figure but this branch simply never fires for it (see write_figures'
                    # docstring: a failing progress channel costs only the per-file names,
                    # never the write). The bar then stays the plain busy state below.
                    self._figs_current_file = ev.file or ev.message[len("figures — "):]
                    if self._figs_total is None and self._ok_file_count:
                        self._figs_total = self._ok_file_count
                        self.progress.setRange(0, self._figs_total)
                    # Clamped at the total, not just at the progress-bar value: a timed-out or
                    # crashed isolated child falls back to _in_process (_figure_process.py),
                    # which RE-ITERATES every ok_file and re-emits this same event for files
                    # the child had already reported — without the clamp here too, the STATUS
                    # TEXT (built from _figs_done directly) could read "5 of 3 files" even
                    # though the bar itself was already capped.
                    self._figs_done = (min(self._figs_done + 1, self._figs_total)
                                       if self._figs_total else self._figs_done + 1)
                    if self._figs_total:
                        self.progress.setValue(self._figs_done)
                self._tick_write()
        elif ev.kind == "breath":
            # Batch-wide, not per-file (A07): a bar that fills from 0 to 100% once per file
            # reads as "done" N times over on a multi-file run — measured at 1.6s for 12
            # fills on a 12-file batch. Held indeterminate until the FIRST file_start (the
            # shared-noise-profile stage before it has no per-file information to show, and
            # a determinate bar stuck at zero reads worse than the animated one).
            if ev.total_breaths and self._run_files:
                n = len(self._run_files)
                frac = (self._file_index - 1 + ev.breath / ev.total_breaths) / n
                self.progress.setRange(0, 1000)
                self.progress.setValue(int(1000 * max(0.0, min(1.0, frac))))
                self._set_status(f"File {self._file_index} of {n} — "
                                 f"{ev.file}: breath {ev.breath}/{ev.total_breaths}")
            else:                # no run-file list (e.g. a direct test of this handler) — old behaviour
                pct = ""
                if ev.total_breaths:
                    self.progress.setRange(0, ev.total_breaths)
                    self.progress.setValue(ev.breath)
                    pct = f" — {ev.breath / ev.total_breaths:.0%}"
                self._set_status(f"{ev.file}: breath {ev.breath}/{ev.total_breaths}{pct}")
        elif ev.kind == "file_done":
            self._ok_file_count += 1
            self._append(f"  done ({ev.message})")
        elif ev.kind == "writing":
            # The compute phase left the bar at a static, determinate 100%; writing (figures
            # especially) can take longer than the compute did. Return to the animated busy
            # state — it stays that way until the first per-file figure event turns it
            # determinate (see the "stage" branch above) — and start the elapsed-time
            # heartbeat so the window visibly keeps working even during the long, quiet
            # figure step. Cancel cannot reach this phase (A07): write_batch takes no
            # cancel control, and the slow part (figures) runs in a child process the
            # parent can only wait on — so the control is disabled here, honestly, instead
            # of staying lit for a click that will be silently ignored.
            self.progress.setRange(0, 0)
            self._append(f"\n{ev.message}…")
            self._write_stage = ""
            self._figs_total = None
            self._figs_done = 0
            self._figs_current_file = ""
            self.btn_cancel.setEnabled(False)
            self.btn_cancel.setToolTip(
                "Writing the output cannot be interrupted; the analysis itself is already complete.")
            self._write_clock.restart()
            self._heartbeat.start()
            self._tick_write()
        elif ev.kind == "finished":
            self._set_status(ev.message)

    def _on_finished(self, result):
        self._heartbeat.stop()                    # end the write-phase heartbeat, any outcome
        # D20: the write phase's last stage name must never survive as the terminal status —
        # without this, a run whose success branch had nothing else to say (see below) left
        # e.g. "Writing output… 23s — figures — Subject04.csv" on screen, a stopped clock
        # over a finished run that reads as still working.
        self._write_stage = ""
        # D20: read before self._worker is discarded a few lines down — the only place this
        # screen can learn what write_batch actually wrote (BatchWorker.written; empty for a
        # dry run, or a run whose write raised before returning).
        written = list(getattr(self._worker, "written", None) or [])
        self._set_running(False)                  # always re-enable first
        fatal = self._fatal_msg is not None
        cancelled = result is None and not fatal
        if result is not None:
            try:
                self._fill_table(result)
                self._fill_file_rail(result)          # P20/B02 per-file status
                self._append_summary(result)
                self._last_failed = list(result.failed_files)   # P18 re-run set
            except Exception as e:                # noqa: BLE001
                self._append(f"\n(results table error: {e})")
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(2000):       # never GC a still-running thread
                _ORPHANED_THREADS.append((self._thread, self._worker))
        self._thread = None; self._worker = None
        # a fatal message with a result present means the ANALYSIS succeeded but the
        # WRITE failed — a distinct outcome from a run that never completed
        write_failed = fatal and result is not None
        if write_failed and self._last_write:
            # the computed result is the only thing "write elsewhere" needs — keep it, and
            # the exact settings that produced it (self._run_settings_snapshot, set in
            # _start), so a retry never re-runs the analysis.
            self._last_result = result
        self.btn_write_elsewhere.setEnabled(bool(write_failed and self._last_write))
        # enablement first (never lets a stale hint win over the outcome status)
        self.refresh_actions()
        self.progress.setRange(0, 1)
        if cancelled:
            self.progress.setValue(0)
            self._set_status("Run cancelled — no output written.")
        elif write_failed:
            self.progress.setValue(1)     # analysis finished; only the write failed
            self._set_status(f"Analysis complete, but writing failed — {short_error(self._fatal_msg)}")
        elif fatal:
            self.progress.setValue(0)
            self._set_status(f"Run failed — {short_error(self._fatal_msg)}")
        else:
            self.progress.setValue(1)
            ok_count = len(result.ok_files) if result is not None else 0
            fail_count = len(result.failed_files) if result is not None else 0
            if self._cancel_requested:
                # A07: the cancel click landed during the (uninterruptible) write phase, and
                # the run went on to finish normally. Neither "Run cancelled — no output
                # written." (false — the output IS complete) nor plain, silent success (the
                # click is simply never acknowledged) is honest here.
                self._set_status(
                    "Finished writing; the cancel arrived after the analysis, so the output is complete.")
            elif ok_count == 0:
                # D20: neither of the two texts below fits a run that finished (not fatal,
                # not cancelled) but could not actually process anything — checked first
                # because it can happen for a real write OR a dry run alike, and "nothing
                # usable was written" is honest either way.
                self._set_status("Finished — no file could be processed; nothing usable was written.")
            elif not self._last_write:
                self._set_status("Dry run complete — nothing written.")
            else:
                # D20: this used to be the one outcome this screen said NOTHING about — the
                # status label simply kept whatever the write phase's last event (or the
                # previous run entirely) had left on screen. Names what a user actually
                # needs without opening the log, a table or a button: how many files were
                # processed/failed, how many files landed on disk, where, and how long it took.
                elapsed = _fmt_duration(self._run_clock.elapsed() / 1000.0)
                out = (self._write_target_folder
                      or (self._run_settings_snapshot or self.state.settings).output.folder or "")
                self._set_status(
                    f"Finished: {ok_count} ok, {fail_count} failed — wrote {len(written)} "
                    f"file{'s' if len(written) != 1 else ''} to {out} in {elapsed}.")
        # a clean real-write run leaves results on disk — offer to open them
        if self._last_write and result is not None and not fatal and not cancelled:
            # D17: a run where every file failed also reaches this "clean" branch (no
            # fatal message, not cancelled) but wrote nothing usable to open — Open
            # output folder must require actual output, not just the absence of a fatal
            # error crossing the batch as a whole. Set UNCONDITIONALLY (never only when
            # true): found in self-review as a real bug — a run that enables this button
            # leaves it enabled forever if a LATER run in the same session reaches this
            # branch with an empty ok_files, since a merely conditional `if ok_files:
            # setEnabled(True)` never turns it back off.
            self.btn_open.setEnabled(bool(result.ok_files))
            report = self._run_report_path()
            if report:
                self._last_report_path = report                # B03: resolved once, not re-read live
                self.btn_show_report.setEnabled(True)
            # A05: a subset write leaves the study's cohort spreadsheets exactly as they
            # were — say so here, permanently in the log, not just in the pre-write dialog,
            # so a "Process & write this file" click doesn't quietly leave them stale.
            if self._safe_is_subset_run(self._only_files):
                self._append(f"\n{self._subset_cohort_note(self._only_files)}")
            elif getattr(result, "average_table", None) is not None:
                # B03: the completion message names the main deliverable, not just a
                # generic "finished" — the file a user reaching for their stats package
                # should open first. getattr, not a direct attribute read: some fixtures
                # (and a fatal-but-with-result outcome) hand in a result with no such field.
                n = len(result.average_table)
                self._append(f"\nAverage breathdata.xlsx ({n} row{'s' if n != 1 else ''}) "
                            "is the file to open first.")
            # D20: only run-report.txt used to list what a run actually wrote, and nothing
            # in the app ever opened it — a user scrolling back through the log to see what
            # they got found only per-file "done" lines, never a final account. Grouped by
            # the same three areas the overwrite-guard dialog already names (D17's
            # ``_existing_output_categories``): data workbook/CSV files, diagnostic
            # figures/audio, and the run's own provenance (analysis-used.toml/run-report.txt).
            if written and result.ok_files:
                out_root = (self._write_target_folder
                           or (self._run_settings_snapshot or self.state.settings).output.folder or "")
                # relpath's first path component, not a substring search on the full path:
                # the output folder itself can legitimately contain "data" or "diagnostics"
                # as a directory name somewhere in its OWN path (e.g. an output folder under
                # "~/data/study1"), which a substring check would misclassify as this run's
                # own data/diagnostics subfolder.
                data_n = fig_n = prov_n = 0
                for p in written:
                    try:
                        head = os.path.relpath(p, out_root).split(os.sep, 1)[0] if out_root else ""
                    except ValueError:          # different drives on Windows
                        head = ""
                    if head == "data":
                        data_n += 1
                    elif head == "diagnostics":
                        fig_n += 1
                    else:
                        prov_n += 1
                self._append(f"\nWrote {len(written)} file{'s' if len(written) != 1 else ''} "
                             f"to {out_root}:")
                if data_n:
                    self._append(f"    • {data_n} data workbook/CSV file{'s' if data_n != 1 else ''}")
                if fig_n:
                    self._append(f"    • {fig_n} diagnostic figure/audio file{'s' if fig_n != 1 else ''}")
                if prov_n:
                    self._append(f"    • {prov_n} provenance file{'s' if prov_n != 1 else ''}")
        self.btn_rerun.setEnabled(bool(self._last_failed))    # P18: re-run only if something failed
        self.run_finished.emit()
        # surface a copyable error-log window if anything failed
        failed = getattr(result, "failed_files", {}) if result is not None else {}
        if failed or self._fatal_msg:
            self._show_error_window(self._error_report(result, failed or {}),
                                    write_failed=write_failed, fatal=fatal)
        elif self._error_dialog is not None:
            # D17: a run that finishes clean must not leave an earlier, now-superseded
            # run's error window still open on screen — _show_error_window only ever
            # replaces it when THIS run has something to show, which a clean run never does.
            self._error_dialog.close()
            self._error_dialog.deleteLater()
            self._error_dialog = None

    def _fatal_fix_hint(self):
        """The `_FIX_HINTS` entry for `self._fatal_msg`'s own exception type, if any —
        ticket D16. A fatal traceback has no `FileResult.error_kind` (there is no
        FileResult; the run never got that far), so the type name is parsed straight out
        of the traceback's last line instead, the same way `short_error` reads it."""
        kind = exception_type_name(self._fatal_msg) if self._fatal_msg else None
        return f"Where to fix {kind}: {_FIX_HINTS[kind]}\n" if kind in _FIX_HINTS else None

    def _error_report(self, result, failed):
        parts = []
        if result is None:
            parts.append("The batch run did not complete.\n")
            if self._fatal_msg:
                parts.append(self._fatal_msg)
                hint = self._fatal_fix_hint()
                if hint:
                    parts.append(hint)
        else:
            if failed:
                parts.append(f"{len(failed)} file{'s' if len(failed) != 1 else ''} "
                             f"failed during the run:\n")
                for fname, fr in failed.items():
                    parts.append(f"── {fname} ──\n{getattr(fr, 'error', '') or '(no detail)'}\n")
                # the message above says what is wrong; this says where to go and change it.
                # dict.fromkeys de-duplicates while keeping the order files failed in.
                kinds = dict.fromkeys(getattr(f, "error_kind", None) for f in failed.values())
                for kind in kinds:
                    if kind in _FIX_HINTS:
                        parts.append(f"Where to fix {kind}: {_FIX_HINTS[kind]}\n")
            if self._fatal_msg:
                if parts:
                    parts.append("")
                parts.append(self._fatal_msg)
                hint = self._fatal_fix_hint()
                if hint:
                    parts.append(hint)
        return "\n".join(parts).strip()

    def _show_error_window(self, text, write_failed=False, fatal=False):
        if self._error_dialog is not None:   # replace, don't accumulate windows
            self._error_dialog.close()
            self._error_dialog.deleteLater()
            self._error_dialog = None
        if write_failed:
            intro = "The analysis completed, but writing the output failed. The full log below is copyable."
        elif fatal:
            intro = "The run stopped with an error before completing. The full log below is copyable."
        else:
            intro = "One or more files failed during the run. The full log is copyable."
        dlg = TextViewerDialog("Batch run — error log", text, self, intro=intro)
        self._error_dialog = dlg          # keep a reference so it isn't GC'd
        dlg.show()
        dlg.raise_()

    def _append_summary(self, result):
        self._append(f"\nFinished: {len(result.ok_files)} ok, {len(result.failed_files)} failed")
        nr = getattr(result, "noise_report", None)
        if nr:
            self._append(f"  noise: prop_decrease {nr.get('prop_decrease')} "
                         f"(fidelity target {nr.get('fidelity_target')})")
        # B02: failures first — a summary that lists nine successes before the one
        # failure buries the thing a reader actually needs to act on.
        for fname, fr in result.failed_files.items():
            self._append(f"  {fname}: FAILED — {getattr(fr, 'error', '') or '(no detail)'}")
        for fname, fr in result.ok_files.items():
            n = len(fr.breaths_table) if fr.breaths_table is not None else 0
            line = f"  {fname}: {n} breaths"
            ecg = getattr(fr, "ecg", None)
            if ecg:
                line += f", ECG suppression {ecg.get('suppression', float('nan')):.0%}"
            self._append(line)

    def _fill_table(self, result):
        if result.average_table is None:
            return
        self._table_model.set_dataframe(result.average_table)
        resize_result_table(self.table)

    def _set_running(self, running):
        self.btn_run.setEnabled(not running)
        self.btn_dry.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        # B03: the bar is the one thing on this screen that should look "busy" — visible
        # only WHILE something is actually in flight, hidden again the instant _on_finished
        # calls this with False, whatever the outcome. Leaving it up at a static 0%/100%
        # fill after a run used to be the loudest object on an otherwise idle screen.
        self.progress.setVisible(running)
        # B03: a run starting is exactly the moment the collapsed log/results section
        # stops being optional — auto-expand it so the user is never left watching a
        # collapsed toggle button while their batch runs.
        if running and not self.btn_toggle_results.isChecked():
            self.btn_toggle_results.setChecked(True)
        if running:
            self.btn_rerun.setEnabled(False)      # can't re-run while a run is in flight
            # A fresh run starts in the (interruptible) compute phase — clear any
            # "cannot be interrupted" tooltip a PREVIOUS run's write phase left behind, or
            # Cancel would look disabled-by-explanation while it is actually live again.
            self.btn_cancel.setToolTip("")

    # -- per-file results, subset re-run, single-file write (P18/P19/P20/B02) -
    def _fill_file_rail(self, result):
        """One rail row per file this run touched, marked with its verdict + breath
        count, then failures brought to the top — so reaching the one failed file in a
        large batch takes a glance and a click instead of scrolling past every success.

        A FULL run's file set (``self._only_files is None``) REPLACES the rail outright:
        it is the complete truth about what the current input folder/mask contains, and a
        stale row from a folder the user has since changed must never linger. A SUBSET
        run (P18 "Re-run failed", P19 "Process & write this file") instead unions with
        the rail's existing filenames — it must not shrink the rail down to just the
        retried files, since the other rows' verdicts from the full run that preceded it
        are still current information. This also keeps the method self-contained and
        directly testable without going through ``_start()`` first, exactly like the
        ``files_table`` it replaces was.

        The failed-first sort is likewise judged from the RAIL'S OWN current state
        (``FileRail.any_failed()``), not just this run's ``result.failed_files``: a
        subset re-run that fixes file A must not silently un-sort file D, which this run
        never touched and which is still failed."""
        files = getattr(result, "files", {}) or {}
        folder = (self._run_settings_snapshot or self.state.settings).input.folder
        if self._only_files is None:
            names = sorted(files.keys())
        else:
            names = sorted(set(self.file_rail.filenames()) | set(files.keys()))
        # Self-review finding: manifest_from_filenames is deliberately caveat-free (it knows
        # only the resolved run file list, not column/frequency probing), but FileRail.
        # set_manifest ALWAYS recomputes caveat from whatever manifest it is given — so a
        # naive call here would silently erase Preview & QC's own outlier/frequency-mismatch
        # warnings from the ONE shared rail the instant any run (even a dry run) finishes.
        # Capture what each row already knew and hand it straight back afterwards; the rail
        # itself defers entirely to the caller for what a caveat should be.
        old_caveats = {name: e.caveat for name in self.file_rail.filenames()
                       if (e := self.file_rail.entry(name)) is not None and e.caveat}
        self.file_rail.set_manifest(manifest_from_filenames(folder, names))
        for name, caveat in old_caveats.items():
            if name in names:
                self.file_rail.set_caveat(name, caveat)
        for fname, fr in files.items():
            err = getattr(fr, "error", None)
            bt = getattr(fr, "breaths_table", None)
            self.file_rail.mark_result(fname, ok=err is None,
                                       breaths=None if err else (len(bt) if bt is not None else 0),
                                       error=str(err) if err else None)
        self.file_rail.sort_failed_first(self.file_rail.any_failed())

    def _rerun_failed(self):
        """Re-run only the files that failed in the last run (P18)."""
        if not self._last_failed:
            self._set_status("No failed files to re-run.")
            return
        n = len(self._last_failed)
        self._append(f"\n── Re-running {n} failed file{'s' if n != 1 else ''} ──")
        self._start(write=self._last_write, only_files=list(self._last_failed))

    def run_single_file(self, filename: str):
        """Process AND write just one file (P19 'Process & write this file' from Preview).
        Reuses the full run machinery (overwrite guard, provenance) restricted to one
        file. The shared noise profile is still built from the whole test."""
        self._start(write=True, only_files=[filename])

    # -- write elsewhere, no re-analysis (ticket A06 point 7) ----------------
    def _write_elsewhere(self):
        """The analysis succeeded but writing it failed (e.g. the output folder turned out
        to be read-only). Re-uses the already-computed ``self._last_result`` and the exact
        settings snapshot that produced it — never re-runs the analysis — and writes to a
        NEW folder the user picks. Always on a worker thread: the figure step alone can
        take many seconds, and running it inline here would freeze the window exactly the
        way ``_heartbeat``/the "writing" progress event exist to prevent for a normal run."""
        if self._last_result is None or self._run_settings_snapshot is None or self._last_plan is None:
            return
        if self._thread is not None or self._write_thread is not None:
            return
        from PySide6.QtWidgets import QFileDialog
        from respmech.ui import prefs
        start = prefs.last_folder("browse", ".")
        folder = QFileDialog.getExistingDirectory(self, "Write results to folder", start)
        if not folder:
            return
        prefs.set_last_folder("browse", folder)
        self._append(f"\n── Writing results to {folder} (no re-analysis) ──")
        self._set_status(f"Writing to {folder}…")
        self.btn_write_elsewhere.setEnabled(False)
        self._pending_write_folder = folder      # committed to _write_target_folder on success

        self._write_thread = QThread()
        self._write_worker = WriteWorker(self._last_result, self._run_settings_snapshot,
                                         self._last_plan, folder)
        self._write_worker.moveToThread(self._write_thread)
        self._write_thread.started.connect(self._write_worker.run)
        # Bound methods, not lambdas: a lambda has no QObject identity of its own, so
        # PySide6 cannot resolve which thread a QueuedConnection should deliver it on —
        # found in self-review as a reproducible 'QThread::wait: Thread tried to wait on
        # itself' / segfault when this was first wired to a lambda.
        self._write_worker.finished.connect(self._on_write_elsewhere_finished, Qt.QueuedConnection)
        self._write_worker.failed.connect(self._on_write_elsewhere_failed, Qt.QueuedConnection)
        # refresh_actions() AFTER self._write_thread is assigned, not before — found in
        # self-review: calling it while self._write_thread was still None left Run/Dry-run
        # enabled for the whole write, and a click landed silently on _start's busy guard.
        self.refresh_actions()
        # MainWindow locks Settings/the Analysis menu on run_started/run_finished for a
        # normal run (see main_window._on_run_started); a write-elsewhere never emitted
        # them (self-review finding), so Settings stayed editable and the window's status
        # bar stayed silent about a write that was still genuinely in progress in the
        # background. It shares self._run_settings_snapshot (frozen), never live settings,
        # so this lock is about the WINDOW's honesty, not about protecting the write itself.
        self.run_started.emit()
        self._write_thread.start()

    def _cleanup_write_thread(self):
        if self._write_thread is not None:
            self._write_thread.quit()
            if not self._write_thread.wait(5000):
                _ORPHANED_THREADS.append((self._write_thread, self._write_worker))
        self._write_thread = None
        self._write_worker = None

    def _on_write_elsewhere_finished(self, written):
        target = self._pending_write_folder
        self._cleanup_write_thread()
        self._write_target_folder = target
        n = len(written)
        self._append(f"Wrote {n} file{'s' if n != 1 else ''} to {target}.")
        self._set_status(f"Wrote {n} file{'s' if n != 1 else ''} to the new folder.")
        self.btn_open.setEnabled(True)
        report = self._run_report_path()               # self._write_target_folder is set above
        if report:
            self._last_report_path = report            # B03: this write has its own report too
            self.btn_show_report.setEnabled(True)
        # The same result can still be written to a THIRD folder if this one turns out to
        # be wrong too — self._last_result/_run_settings_snapshot/_last_plan are untouched.
        self.btn_write_elsewhere.setEnabled(True)
        self.refresh_actions()
        self.run_finished.emit()

    def _on_write_elsewhere_failed(self, msg):
        self._cleanup_write_thread()
        self._append(f"ERROR writing to the new folder: {short_error(msg)}")
        self._set_status(f"Writing to the new folder failed — {short_error(msg)}")
        self.btn_write_elsewhere.setEnabled(True)      # still available — try another folder
        self.refresh_actions()
        self.run_finished.emit()

    def _append(self, text):
        self.log.appendPlainText(text)

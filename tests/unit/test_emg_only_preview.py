"""``ui.workers.stage_emg_segments_preview`` and the 'segments' reactive job:
the EMG-only counterpart of ``stage_mechanics_preview``/'mech' — no flow/pressure
channel to split into breaths, so a configured method (``whole_file``/``separators``)
splits the whole recording into segments instead. This file covers the Qt-free
worker directly (spans, soft error) and the job-dispatch/render wiring through a real
``PreviewScreen`` (auto-run gating, ``pv._breaths``, the 'Not processed' status line).
"""
import os

import numpy as np

from _helpers import INPUT, requires_synth, synth_settings

from respmech.core.settings import SeparatorEntry
from respmech.ui.workers import stage_emg_segments_preview

pytestmark = requires_synth()

FILENAME = "synth_case_A.csv"


def _emg_only_settings(tmp_path, method="whole_file", separator_times=None):
    s = synth_settings(tmp_path, channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})
    s.processing.segmentation.method = method
    if separator_times is not None:
        s.processing.segmentation.separators.append(
            SeparatorEntry(file=FILENAME, times_s=separator_times))
    return s


def _path(settings):
    return os.path.join(settings.input.folder, FILENAME)


# -- the worker, called directly (no Qt) -----------------------------------------------

def test_whole_file_gives_one_span_covering_the_whole_recording(tmp_path):
    s = _emg_only_settings(tmp_path, method="whole_file")
    data = stage_emg_segments_preview(s, _path(s))
    assert data["segment_error"] is None
    assert data["startix"] == 0
    assert data["endix"] == data["emg"].shape[0]
    assert len(data["spans"]) == 1
    num, t0, t1, ignored, kind = data["spans"][0]
    assert num == 1
    assert t0 == 0.0
    assert ignored is False
    assert kind is None
    assert t1 == data["emg"].shape[0] / data["fs"]


def test_spans_equal_the_segmentation_s_own_segments():
    """The ticket's own literal acceptance criterion: the worker's spans match what
    ``compute.separateintobreaths``/``core.analysis.segments`` itself produces for the
    same file — recomputed independently here, not merely re-derived from the worker."""
    from respmech.core import compute
    from respmech.core._legacy_ns import to_legacy_ns

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        s = _emg_only_settings(tmp, method="separators", separator_times=[1.0, 2.0])
        data = stage_emg_segments_preview(s, _path(s))
        ls = to_legacy_ns(s)
        expected = compute.separateintobreaths(
            "separators", FILENAME, data["t"], np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), [], data["emg_conditioned"], ls)
        assert len(data["spans"]) == len(expected) == 3
        for (num, t0, t1, ignored, kind), (exp_num, exp_seg) in zip(data["spans"], expected.items()):
            assert num == exp_num
            assert ignored == exp_seg["ignored"]
            assert kind == exp_seg["kind"]
            exp_time = np.atleast_1d(exp_seg["time"])
            assert t0 == exp_time[0]
            assert t1 == t0 + exp_time.size / data["fs"]


def test_separators_split_into_the_right_number_of_segments_with_contiguous_spans(tmp_path):
    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0, 3.5])
    data = stage_emg_segments_preview(s, _path(s))
    assert data["segment_error"] is None
    spans = data["spans"]
    assert len(spans) == 4
    assert [n for (n, *_r) in spans] == [1, 2, 3, 4]
    # half-open [t0, t1) per span, contiguous end-to-end with no gap/overlap
    for (_n1, _a1, t1, *_r1), (_n2, t0, _b2, *_r2) in zip(spans, spans[1:]):
        assert t1 == t0
    assert spans[0][1] == 0.0
    assert spans[-1][2] == data["emg"].shape[0] / data["fs"]


def test_a_separator_outside_the_recording_is_a_soft_segment_error_not_a_raise(tmp_path):
    """The ticket's other literal acceptance criterion: a bad separator is reported via
    ``segment_error`` in the returned dict, never an exception the worker lets escape —
    the render layer turns this into a status line, not a copyable error card."""
    s = _emg_only_settings(tmp_path, method="separators", separator_times=[999999.0])
    data = stage_emg_segments_preview(s, _path(s))
    assert data["spans"] == []
    assert data["segment_error"]
    assert FILENAME in data["segment_error"]
    # the raw EMG matrix is still staged -- the raw stack can still show the channels
    assert data["emg"].shape[0] > 0


def test_an_excluded_segment_is_flagged_ignored_in_its_own_span(tmp_path):
    """Every span carries its OWN ignored flag (processing.exclude_breaths, keyed by
    segment number exactly like a breath) -- not folded into a status count alone."""
    from respmech.core.settings import ExcludeEntry

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0])
    s.processing.exclude_breaths.append(ExcludeEntry(file=FILENAME, breaths=[2]))
    data = stage_emg_segments_preview(s, _path(s))
    assert data["segment_error"] is None
    ignored_by_num = {num: ignored for (num, _t0, _t1, ignored, _kind) in data["spans"]}
    assert ignored_by_num == {1: False, 2: True, 3: False}


# -- job dispatch + render, through a real PreviewScreen -------------------------------
# (the ticket's "test_startup_imports uændret" criterion is exercised by re-running the
# app's own tests/unit/test_startup_imports.py, not duplicated here -- it asserts nothing
# in this module's own worker/dispatch surface, only that GUI startup stays import-light.)

def test_selecting_an_emg_only_file_starts_a_segments_job_and_ends_with_breaths(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from PySide6.QtCore import QEventLoop, QElapsedTimer, QTimer

    def _pump_until(predicate, timeout=30.0):
        if predicate():
            return True
        loop = QEventLoop()
        clock = QElapsedTimer(); clock.start()
        state = {"ok": False}
        timer = QTimer(); timer.setInterval(10)

        def _tick():
            if predicate():
                state["ok"] = True
                loop.quit()
            elif clock.elapsed() > timeout * 1000:
                loop.quit()
        timer.timeout.connect(_tick)
        timer.start()
        loop.exec()
        timer.stop()
        return state["ok"] or predicate()

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    pv._schedule("segments")
    assert "segments" in pv._jobs                       # a real job was actually launched
    assert _pump_until(lambda: not pv._jobs and not pv._draining)
    assert pv._breaths                                    # ends non-empty (ticket's own wording)
    assert pv._breaths[0][:2] == (1, 0.0)
    assert pv.panel_error("segstack") is None            # a real result, not a crash card
    win.close()


def test_segments_is_gated_off_for_a_flow_bearing_set_and_mech_is_gated_off_for_emg_only(tmp_path):
    """The dispatch gate this ticket adds: 'mech' only when caps.flow, 'segments' only when
    caps.mode == 'emg_only' — the two are each other's counterpart, never both live for
    the same signal set."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    flow_settings = synth_settings(tmp_path / "flow")
    win = MainWindow(AppState(flow_settings))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    pv._schedule("segments")
    assert "segments" not in pv._jobs
    win.close()

    emg_settings = _emg_only_settings(tmp_path / "emg", method="whole_file")
    win2 = MainWindow(AppState(emg_settings))
    pv2 = win2.preview_screen
    pv2._refresh_files()
    pv2.file_rail.select_filename(FILENAME)
    pv2._schedule("mech")
    assert "mech" not in pv2._jobs
    win2.close()


def test_a_bad_separator_shows_not_processed_in_the_status_line_not_an_error_card(tmp_path):
    """The ticket's third literal acceptance criterion, at the render layer this time
    (the worker-level version is covered above): 'Not processed — ...' in the status
    line, and no error card on the panel this job owns."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[999999.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    data = stage_emg_segments_preview(pv.state.settings, _path(s))
    pv._render_segments_preview(data)
    assert "Not processed" in pv.status.text()
    assert pv.panel_error("segstack") is None
    win.close()


def test_an_excluded_segment_shows_in_the_status_line_count(tmp_path):
    """The other half of the status-line wording (nseg/nign) -- covered above at the
    worker level (each span's own ignored flag); this exercises the render layer's own
    'excluded' count text, the branch none of the other render-layer tests touch."""
    from respmech.core.settings import ExcludeEntry
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0])
    s.processing.exclude_breaths.append(ExcludeEntry(file=FILENAME, breaths=[2]))
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    data = stage_emg_segments_preview(pv.state.settings, _path(s))
    pv._render_segments_preview(data)
    assert "3 segments" in pv.status.text()
    assert "1 excluded" in pv.status.text()
    win.close()


# -- M-26: the dedicated tab, its action band, click-to-exclude/type and the batch table --

def test_emg_only_subtab_titles_are_segments_ecg_noise_in_order(tmp_path):
    """The ticket's own literal acceptance criterion."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    titles = [pv.subtabs.tabText(i) for i in range(pv.subtabs.count())]
    assert titles == ["EMG – segments", "› EMG – ECG reduction", "› EMG – noise reduction"]
    assert pv.subtabs.widget(0) is pv._segments_tab
    win.close()


def test_flow_bearing_subtab_titles_are_unchanged_mechanics_first(tmp_path):
    """The counterpart shape must still get Mechanics first, not the segments tab."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    titles = [pv.subtabs.tabText(i) for i in range(pv.subtabs.count())]
    assert titles == ["Mechanics", "› EMG – ECG reduction", "› EMG – noise reduction"]
    assert pv.subtabs.widget(0) is pv._mech_tab
    win.close()


def test_a_mid_session_shape_flip_keeps_the_user_on_the_replacement_first_tab(tmp_path):
    """Self-review finding: removing the OLD first-slot tab (Mechanics or segments)
    while it is the CURRENT one used to let Qt auto-select whatever tab shifted into
    that slot (ECG-reduction) instead of the tab that replaced it — a user parked on
    Mechanics at the moment a loaded analysis flips to EMG-only (or back) was silently
    dropped onto an unrelated sub-tab. Exercises _update_subtabs() via a REAL
    sync_from_settings() shape change, both directions, on the actual PreviewScreen a
    user would be looking at."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv.subtabs.setCurrentWidget(pv._mech_tab)
    for role in ("flow", "poes", "pgas", "pdi", "volume"):
        setattr(pv.state.settings.input.channels, role, None)
    pv.sync_from_settings()
    assert pv.subtabs.currentWidget() is pv._segments_tab, (
        "flipping to EMG-only while parked on Mechanics must land on the segments tab, "
        "not silently fall through to whatever tab shifted into that slot")

    canonical = synth_settings(tmp_path).input.channels
    for role in ("flow", "poes", "pgas", "pdi", "volume"):
        setattr(pv.state.settings.input.channels, role, getattr(canonical, role))
    pv.sync_from_settings()
    assert pv.subtabs.currentWidget() is pv._mech_tab, (
        "flipping back to flow-bearing while parked on the segments tab must land on "
        "Mechanics, symmetrically")
    win.close()


def test_clicking_a_segment_toggles_exclusion_via_the_shared_toggle_breath_funnel(tmp_path):
    """Acceptance criterion: 'left-click toggles exclusion via _toggle_breath' — M-25's
    provisional renderer left ``_breath_spans`` empty for exactly this reason (no click
    surface existed yet); this is what M-26 fixes."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    data = stage_emg_segments_preview(pv.state.settings, _path(s))
    pv._render_segments_preview(data)
    assert pv._breath_spans, "no click surface -- the exact gap this ticket exists to close"

    now_excluded = pv._toggle_breath(1)
    assert now_excluded is True
    entry = next(e for e in pv.state.settings.processing.exclude_breaths if e.file == FILENAME)
    assert entry.breaths == [1]
    win.close()


def test_a_real_click_on_the_segments_stack_toggles_exclusion(tmp_path):
    """The test above calls ``_toggle_breath`` directly, which proves the shared funnel
    works but never touches THIS ticket's own new wiring: ``segments_plots.scene().
    sigMouseClicked`` -> ``_on_segments_clicked`` -> ``_toggle_from_emg_click`` (hit-
    testing ``self._segments_subplots`` via ``sceneBoundingRect``/``mapSceneToView``).
    A bug that wires the wrong plot list, forgets the ``.connect()`` call, or breaks
    ``_segments_subplots``'s population would not be caught by the test above (found in
    self-review, mutation-verified).

    Fake event/viewbox objects exercising the REAL hit-test math, not real Qt scene
    geometry — the same technique ``test_gui_interactive.py::
    test_legend_click_does_not_toggle_breath`` already uses for the raw/detail/result
    views' equivalent click handlers, since a headless render has no real laid-out
    scene coordinates to click on."""
    from PySide6.QtCore import QPointF, Qt
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    data = stage_emg_segments_preview(pv.state.settings, _path(s))
    pv._render_segments_preview(data)
    num, t0, t1, _kind = pv._breaths[1]            # segment 2 of 3
    off = pv._trim_offset_s
    xc = (t0 + t1) / 2.0 + off                     # absolute-time centre, as the real stack is drawn

    class _Rect:
        def contains(self, _p):
            return True

    class _View:
        def sceneBoundingRect(self):
            return _Rect()

        def mapSceneToView(self, _pos):
            return QPointF(xc, 0.0)

    class _Plot:
        def getViewBox(self):
            return _View()

    class _Ev:
        def isAccepted(self):
            return False

        def button(self):
            return Qt.LeftButton

        def scenePos(self):
            return QPointF(0.0, 0.0)

    # exercises _on_segments_clicked itself (the actual .connect() target), reading
    # self._segments_subplots -- not _toggle_from_emg_click called directly.
    pv._segments_subplots = [_Plot()]
    pv._on_segments_clicked(_Ev())

    entry = next(e for e in pv.state.settings.processing.exclude_breaths if e.file == FILENAME)
    assert entry.breaths == [num]
    win.close()


def test_typerequested_signal_from_a_real_segment_item_reaches_the_shared_handler(tmp_path):
    """Acceptance criterion: 'højreklik typer via M-20's primitiv (rest tilbudt)'.
    ``_draw_breath_overlays`` wires ``BreathSpansItem.typeRequested ->
    _handle_type_requested`` identically regardless of which ``plots`` list it was
    given — but until this test, nothing rendered the segments stack and then actually
    emitted that signal from one of ITS OWN items (test_breath_typing_ui.py covers the
    primitive generically and end-to-end for Mechanics only). Uses a REAL emitted
    signal from a REAL ``BreathSpansItem`` drawn on ``segments_plots`` — not a direct
    ``_handle_type_requested(...)`` call — so a broken/missing ``.connect()`` for the
    segments-stack render specifically would be caught."""
    from PySide6.QtCore import QPointF
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    data = stage_emg_segments_preview(pv.state.settings, _path(s))
    pv._render_segments_preview(data)
    num = next(iter(pv._breath_spans))
    item, _idx = pv._breath_regions[num][0]      # the REAL BreathSpansItem drawn on segments_plots

    item.typeRequested.emit(num, QPointF(0.0, 0.0))   # must not raise -- same primitive as Mechanics

    # the menu this just popped applies through the SAME funnel a click on it would —
    # verify 'rest' really is reachable end to end for a segment, not just for a breath.
    result = pv._set_breath_type(num, "rest")
    assert result == "rest"
    entry = next(t for t in pv.state.settings.processing.breath_types
                if t.file == FILENAME and t.breath == num)
    assert entry.kind == "rest"
    win.close()


def test_segments_action_band_shows_only_on_the_segments_tab(qapp, tmp_path):
    """Mirrors test_mech_short_screen.py's own Mechanics-band test, for the segments
    tab's twin band (M-26 generalised _update_mech_action_band_visibility to cover both,
    mutually exclusively). ``win.show()`` is required: a widget's own ``isVisible()``
    stays False regardless of ``setVisible(True)`` until its top-level window is shown
    (same reason test_mech_short_screen.py's ``_window()`` helper calls it)."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    win.show()
    win.tabs.setCurrentWidget(win.preview_screen)   # Preview & QC must be the shown top-level tab
    for _ in range(6):
        qapp.processEvents()
    pv = win.preview_screen
    assert pv.subtabs.currentWidget() is pv._segments_tab
    assert pv._segments_action_band.isVisible()
    assert not pv._mech_action_band.isVisible()

    other = next(i for i in range(pv.subtabs.count()) if pv.subtabs.widget(i) is not pv._segments_tab)
    pv.subtabs.setCurrentIndex(other)
    for _ in range(6):
        qapp.processEvents()
    assert not pv._segments_action_band.isVisible()

    pv.subtabs.setCurrentIndex(pv.subtabs.indexOf(pv._segments_tab))
    for _ in range(6):
        qapp.processEvents()
    assert pv._segments_action_band.isVisible()
    win.close()


def test_process_segments_button_label_has_a_literal_double_ampersand(tmp_path):
    """The app-wide convention (RespMech's ui/CLAUDE.md): a literal '&' must be doubled
    or Qt eats it as a mnemonic — covered generically by test_ui_wording.py's full-window
    scan, asserted directly here too since it is this ticket's own new button."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    assert pv.btn_process_segments_file.text() == "Process && write this file"
    win.close()


def test_the_batch_test_run_computes_from_emg_and_fills_the_segments_table_for_an_emg_only_set(
        qapp, tmp_path):
    """Found in self-review (M-25), not a literal acceptance criterion: 'batch' (the
    auto-run 'Test run' job) has always been dispatched for every signal set, EMG-only
    included — but until M-25's fix its dispatch unconditionally stripped
    input.channels.emg (meant only for a flow-bearing set, where the EMG work is shown
    separately). For an EMG-only set that left NO channel assigned at all, so
    Settings.validate() refused the snapshot and BatchWorker's fatal-exception path
    painted a raw 'Test run failed' card.

    M-26 strengthening (self-review): this test now goes all the way through the REAL
    dispatch path (``_schedule`` -> ``_launch`` -> a real worker thread -> ``_on_job_done``
    -> ``_RENDER['batch']``) and checks ``segtable``/its rowcount, not just that
    ``table``/``campbell`` show no error. Those two panels are never touched by an
    EMG-only 'batch' result any more (M-26 routes it to the segments tab's own table
    instead), so asserting only `panel_error("table"/"campbell") is None` would pass
    vacuously regardless of whether the run — or the segtable render — actually works;
    a manual, ``_schedule``-bypassing call to ``BatchWorker``/``_on_batch_result`` would
    equally miss a regression in ``_schedule``'s own EMG-only 'batch' gating (the exact
    bug this test exists to catch in the first place). Both gaps were found and verified
    by mutation-testing in this ticket's self-review."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from PySide6.QtCore import QEventLoop, QElapsedTimer, QTimer

    def _pump_until(predicate, timeout=30.0):
        if predicate():
            return True
        loop = QEventLoop()
        clock = QElapsedTimer(); clock.start()
        state = {"ok": False}
        timer = QTimer(); timer.setInterval(10)

        def _tick():
            if predicate():
                state["ok"] = True
                loop.quit()
            elif clock.elapsed() > timeout * 1000:
                loop.quit()
        timer.timeout.connect(_tick)
        timer.start()
        loop.exec()
        timer.stop()
        return state["ok"] or predicate()

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    pv._schedule("batch")
    assert "batch" in pv._jobs
    assert _pump_until(lambda: not pv._jobs and not pv._draining)
    assert pv.panel_error("segtable") is None
    assert pv.panel_error("table") is None       # the Mechanics table is never touched
    assert pv.panel_error("campbell") is None
    assert pv._segtable_model.rowCount() == 1
    win.close()


# NOTE (M-26): a prior ticket added
# test_a_stale_job_of_one_kind_does_not_stop_the_other_kind_s_live_spinner_on_raw here,
# for a scenario that existed only because 'segments' provisionally shared the 'raw'
# panel with 'mech' (see _jobs.py's old _PANELS['segments'] comment). M-26 gives the
# segments tab its own dedicated panel ('segstack'), so no two _AUTO_KINDS entries share
# a panel any more and that specific scenario can no longer arise — removed rather than
# force-fit onto an unrelated pairing. The general defensive check it exercised
# (_on_job_done's `owned_elsewhere`) is still live code, generalised in this ticket to
# read each job's own frozen `job.panels` instead of the (now capability-varying)
# `_PANELS[job.kind]` — see screen.py's own comment at the call site.


def test_a_stale_batch_job_clears_the_panels_it_was_actually_dispatched_with(tmp_path):
    """A same-kind analogue of the removed test above, exercising the NEW
    capability-varying panels_for('batch', caps) via job.panels — and specifically
    distinguishing it from the OLD, static ``_PANELS[job.kind]`` lookup it replaces.

    Self-review (mutation-verified): an earlier version of this test gave the stale
    job ``panels=('table', 'campbell')`` — which is ALSO ``_PANELS['batch']``'s static
    default, so a regression that reverted ``_on_job_done`` back to reading
    ``_PANELS[job.kind]`` instead of ``job.panels`` would have passed this test just as
    well as the fix does. Here the stale job's OWN frozen panels are ``('segtable',)``
    — DIFFERENT from the static default — so only a correct, ``job.panels``-driven
    implementation stops the right ('segtable') overlay and leaves the static
    default's ('table'/'campbell') alone; the old, reverted behaviour would try to
    stop table/campbell instead and this test would fail."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    from respmech.ui.state import AppState

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._overlays["table"].start("Running test…")
    pv._overlays["campbell"].start("Running test…")
    pv._overlays["segtable"].start("Running test…")
    # a stale 'batch' job whose OWN frozen panels are 'segtable' -- deliberately NOT
    # equal to the static _PANELS['batch'] default, so the assertions below can only
    # pass if _on_job_done reads job.panels, never the static per-kind dict.
    pv._tokens["batch"] += 1
    stale = _Job("batch", pv._tokens["batch"] - 1, QThread(), object(),
                panels=("segtable",))
    pv._on_job_done(stale, None)
    assert pv._overlays["segtable"].busy is False    # the job's OWN panel, correctly stopped
    assert pv._overlays["table"].busy is True         # NOT touched -- proves job.panels drives this
    assert pv._overlays["campbell"].busy is True
    win.close()

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
    pv._render_emg_segments_preview(data)
    assert "Not processed" in pv.status.text()
    assert pv.panel_error("raw") is None
    win.close()


def test_the_batch_test_run_computes_from_emg_for_an_emg_only_set_not_a_crash_card(
        qapp, tmp_path):
    """Found in self-review, not a literal acceptance criterion: 'batch' (the auto-run
    'Test run' table+Campbell panel) has always been dispatched for every signal set,
    EMG-only included — but until this ticket its dispatch unconditionally stripped
    input.channels.emg (meant only for a flow-bearing set, where the EMG work is shown
    separately). For an EMG-only set that left NO channel assigned at all, so
    Settings.validate() refused the snapshot and BatchWorker's fatal-exception path
    painted a raw 'Test run failed' card — for the exact signal-set shape this ticket
    exists to support. Fixed alongside the 'segments'/'mech' gating this ticket already
    adds, in the same function, from the same caps/has_flow values."""
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
    assert pv.panel_error("table") is None
    assert pv.panel_error("campbell") is None
    win.close()

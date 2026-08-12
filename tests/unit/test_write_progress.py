"""Progress feedback for the (previously silent) write phase.

On a batch with diagnostics on, writing — figures especially — can take longer than the
compute that precedes it, and it used to emit nothing: the window looked frozen from the
last file's "done" to "Finished". These tests pin the two halves of the fix:

* ``core.io.writers.write_batch`` now emits ``stage`` progress events per phase and per
  file, and the figure step reports per file on the in-process path;
* the Run screen switches its progress bar to the animated busy state and runs a
  once-a-second elapsed-time heartbeat while writing, so activity is visible even if the
  styled busy bar happens not to animate on a given platform.
"""
import os

import pandas as pd
import pytest

from _helpers import requires_synth, synth_settings

pytestmark = requires_synth()


# --- core: write_batch emits progress ------------------------------------------------
def test_write_batch_emits_stage_events(tmp_path, monkeypatch):
    from respmech.core.pipeline import run_batch
    from respmech.core.io import writers, _figure_process

    monkeypatch.setenv("RESPMECH_NO_FIGURE_SUBPROCESS", "1")   # in-process → per-file figs
    # _can_spawn() caches its verdict process-wide; a prior test may have warmed it to True,
    # which would send figures to the child and skip the per-file callback. Reset it so the
    # env var above actually takes effect.
    monkeypatch.setattr(_figure_process, "_CAN_SPAWN", None)
    settings = synth_settings(str(tmp_path))
    settings.output.folder = str(tmp_path / "out")
    os.makedirs(settings.output.folder, exist_ok=True)
    result = run_batch(settings)

    events = []
    written = writers.write_batch(result, settings, settings.output.folder,
                                  progress=lambda ev: events.append(ev))
    assert written                                              # files really were written
    kinds = {e.kind for e in events}
    assert kinds == {"stage"}                                  # writing emits only stages
    messages = [e.message for e in events]
    assert any("breath-by-breath" in m for m in messages)
    assert any("diagnostic figures" in m for m in messages)
    # in-process figure path reports each file it draws
    assert any(m.startswith("figures — ") for m in messages)


def test_write_batch_progress_is_optional(tmp_path):
    """The CLI calls write_batch without progress — the default must stay silent, not crash."""
    from respmech.core.pipeline import run_batch
    from respmech.core.io import writers

    settings = synth_settings(str(tmp_path))
    settings.output.folder = str(tmp_path / "out")
    os.makedirs(settings.output.folder, exist_ok=True)
    result = run_batch(settings)
    assert writers.write_batch(result, settings, settings.output.folder)   # no progress arg


# --- GUI: the Run screen reacts to the write phase -----------------------------------
def _run_screen(tmp):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    return MainWindow(AppState(synth_settings(tmp))).run_screen


def test_writing_event_starts_busy_bar_and_heartbeat(qapp, tmp_path):
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    assert not rn._heartbeat.isActive()

    rn._on_progress(ProgressEvent("writing", message="writing output"))
    assert rn.progress.minimum() == 0 and rn.progress.maximum() == 0   # busy/indeterminate
    assert rn._heartbeat.isActive()
    assert "Writing output" in rn.status.text()


def test_stage_events_during_write_scroll_and_update_status(qapp, tmp_path):
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._on_progress(ProgressEvent("writing", message="writing output"))
    rn._on_progress(ProgressEvent("stage", message="writing diagnostic figures (the slow step)"))

    assert "diagnostic figures" in rn.log.toPlainText()
    assert "diagnostic figures" in rn.status.text()             # heartbeat label reflects stage


def test_heartbeat_stops_when_finished(qapp, tmp_path):
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._on_progress(ProgressEvent("writing", message="writing output"))
    assert rn._heartbeat.isActive()

    rn._on_finished(None)                                       # cancelled/none result path
    assert not rn._heartbeat.isActive()


# --------------------------------------------------------------------------------------- #
# A07 — Cancel must be honest during the write phase, and the compute/write bars must show
# real progress: a batch-wide compute bar (not a per-file one that "finishes" N times), and
# a determinate write-phase bar once the figure step's per-file events start arriving.
# --------------------------------------------------------------------------------------- #
def test_writing_event_disables_cancel_with_explanation(qapp, tmp_path):
    """Cancel is armed but useless once writing starts — write_batch takes no cancel
    control and the slow step (figures) runs in a child process the parent can only wait
    on. The control must say so, not sit lit for a click that is silently ignored."""
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._set_running(True)
    assert rn.btn_cancel.isEnabled()

    rn._on_progress(ProgressEvent("writing", message="writing output"))
    assert not rn.btn_cancel.isEnabled()
    assert "cannot be interrupted" in rn.btn_cancel.toolTip()


def test_a_fresh_run_clears_a_stale_cannot_interrupt_tooltip(qapp, tmp_path):
    """The tooltip set during a PREVIOUS run's write phase must not linger once a new run's
    (interruptible) compute phase begins — it would say Cancel cannot work when it can."""
    rn = _run_screen(str(tmp_path))
    rn.btn_cancel.setToolTip("Writing the output cannot be interrupted; stale from last run.")
    rn._set_running(True)
    assert rn.btn_cancel.toolTip() == ""


class _FakeWritingWorker:
    """Stands in for BatchWorker in these tests: real behaviour of setting ``_writing = True``
    as the worker's first action on entering the write phase (see workers.py), WITHOUT a real
    QThread — so tests can drive ``_cancel()`` deterministically at any point relative to
    whether the GUI has processed the (always-asynchronous) "writing" ProgressEvent yet."""
    def __init__(self, writing=False):
        self.cancelled = False
        self._writing = writing

    def cancel(self):
        self.cancelled = True


def test_cancel_during_write_phase_reports_uninterruptible_not_cancelling(qapp, tmp_path):
    """A Cancel click that lands during writing must not silently do nothing, and must not
    claim the click stopped anything — it announces that it landed and that the write is
    finishing anyway."""
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._worker = _FakeWritingWorker(writing=True)
    rn._on_progress(ProgressEvent("writing", message="writing output"))
    log_before = rn.log.toPlainText()

    rn._cancel()
    assert rn._worker.cancelled
    assert rn._cancel_requested
    assert "cannot be interrupted" in rn.status.text()
    assert rn.log.toPlainText() == log_before, "a write-phase cancel must not append 'cancelling…'"


def test_cancel_reads_the_workers_own_flag_not_the_heartbeat(qapp, tmp_path):
    """The race three independent reviewers flagged: the 'writing' ProgressEvent reaches the
    GUI thread only via a QueuedConnection, asynchronously, so a click landing in the gap
    between the WORKER committing to the write phase and the GUI processing that event must
    still be recognised as write-phase — it must not fall back to the plain 'cancelling…'
    line just because the heartbeat (started by that same, not-yet-processed event) hasn't
    ticked on yet. This is exactly what _worker._writing (set directly by the worker thread,
    read directly here) exists to fix."""
    rn = _run_screen(str(tmp_path))
    rn._worker = _FakeWritingWorker(writing=True)      # worker committed...
    assert not rn._heartbeat.isActive()                # ...but the GUI hasn't been told yet

    rn._cancel()
    assert "cannot be interrupted" in rn.status.text()
    assert "cancelling…" not in rn.log.toPlainText()


def test_cancel_during_compute_still_appends_cancelling(qapp, tmp_path):
    """Unchanged behaviour: a cancel click during the (cooperative, interruptible) compute
    phase still gets the old, simple log line — only the write phase gets the new message."""
    rn = _run_screen(str(tmp_path))
    rn._worker = _FakeWritingWorker(writing=False)
    rn._cancel()
    assert "cancelling…" in rn.log.toPlainText()
    assert rn._cancel_requested


def test_finished_after_cancel_during_write_reports_completion(tmp_path, qapp):
    """A cancel that arrives during writing, on a run that goes on to finish, must say the
    output is complete: neither the plain cancelled message (false — output WAS written)
    nor silent success (the click vanished)."""
    import os
    from respmech.core.pipeline import ProgressEvent, run_batch

    rn = _run_screen(str(tmp_path))
    settings = synth_settings(str(tmp_path))
    settings.output.folder = str(tmp_path / "out")
    os.makedirs(settings.output.folder, exist_ok=True)
    result = run_batch(settings)

    rn._worker = _FakeWritingWorker(writing=True)
    rn._on_progress(ProgressEvent("writing", message="writing output"))
    rn._cancel()                      # cancel lands during the uninterruptible write phase
    rn._on_finished(result)           # ...but the run finishes normally regardless

    text = rn.status.text().lower()
    assert "complete" in text
    assert "cancelled" not in text
    assert "no output written" not in text


def test_batch_wide_compute_progress_bar_reflects_file_and_breath(qapp, tmp_path):
    """With simulated progress for three files, half-way through file 2 the bar reads about
    half of the WHOLE batch (not half of one file), and the status names which file of
    how many."""
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._run_files = ["a.csv", "b.csv", "c.csv"]
    rn._file_index = 0

    rn._on_progress(ProgressEvent("file_start", file="a.csv", message="loading"))
    rn._on_progress(ProgressEvent("breath", file="a.csv", breath=9, total_breaths=9))
    rn._on_progress(ProgressEvent("file_start", file="b.csv", message="loading"))
    rn._on_progress(ProgressEvent("breath", file="b.csv", breath=5, total_breaths=10))

    assert "File 2 of 3" in rn.status.text()
    frac = rn.progress.value() / rn.progress.maximum()
    assert 0.45 < frac < 0.55, frac


def test_compute_progress_bar_does_not_refill_per_file(qapp, tmp_path):
    """The old per-file bar filled 0->100% once per file — a false 'done' signal repeated
    N times. The batch-wide bar must only ever increase across the whole run."""
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._run_files = ["a.csv", "b.csv"]
    rn._file_index = 0

    values = []
    rn._on_progress(ProgressEvent("file_start", file="a.csv", message="loading"))
    for b in range(1, 10):
        rn._on_progress(ProgressEvent("breath", file="a.csv", breath=b, total_breaths=9))
        values.append(rn.progress.value())
    rn._on_progress(ProgressEvent("file_start", file="b.csv", message="loading"))
    for b in range(1, 5):
        rn._on_progress(ProgressEvent("breath", file="b.csv", breath=b, total_breaths=4))
        values.append(rn.progress.value())

    assert values == sorted(values), "the bar went backwards moving from file 1 to file 2"
    assert values[-1] > values[0]


def test_write_phase_progress_bar_becomes_determinate_for_figures(qapp, tmp_path):
    """Once the figure step's first per-file event arrives, the write-phase bar switches
    from indeterminate to a determinate count of files-with-figures, and the status names
    the file and shows an ETA only once there is a rate to extrapolate from."""
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._ok_file_count = 3
    rn._on_progress(ProgressEvent("writing", message="writing output"))
    rn._on_progress(ProgressEvent("stage", message="writing breath-by-breath data"))
    assert rn.progress.minimum() == 0 and rn.progress.maximum() == 0   # still busy

    rn._on_progress(ProgressEvent("stage", message="writing diagnostic figures (the slow step)"))
    assert rn.progress.minimum() == 0 and rn.progress.maximum() == 0   # group header, no file yet

    rn._on_progress(ProgressEvent("stage", file="a.csv", message="figures — a.csv"))
    assert rn.progress.maximum() == 3
    assert rn.progress.value() == 1
    assert "Writing diagnostic figures — 1 of 3 files (a.csv)" in rn.status.text()
    assert "~" not in rn.status.text()      # nothing to extrapolate from yet

    rn._on_progress(ProgressEvent("stage", file="b.csv", message="figures — b.csv"))
    assert rn.progress.value() == 2
    assert "2 of 3 files (b.csv)" in rn.status.text()
    assert "~" in rn.status.text()          # an estimate now exists, and always reads as one

    rn._on_progress(ProgressEvent("stage", file="c.csv", message="figures — c.csv"))
    assert rn.progress.value() == 3
    assert "3 of 3 files (c.csv)" in rn.status.text()
    assert "~" not in rn.status.text()      # nothing left to wait for


def test_write_phase_bar_stays_indeterminate_without_a_file_count(qapp, tmp_path):
    """If the compute phase never told us how many files succeeded (should not happen in
    practice, but must never crash or divide by zero), the figure step's per-file event must
    not blow up and must leave the bar exactly as an ordinary busy write phase would."""
    from respmech.core.pipeline import ProgressEvent

    rn = _run_screen(str(tmp_path))
    rn._ok_file_count = 0
    rn._on_progress(ProgressEvent("writing", message="writing output"))
    rn._on_progress(ProgressEvent("stage", file="a.csv", message="figures — a.csv"))
    assert rn.progress.minimum() == 0 and rn.progress.maximum() == 0
    assert "Writing output" in rn.status.text()


def test_fmt_duration_and_eta():
    from respmech.ui.screens.run_screen import _fmt_duration, _fmt_eta

    assert _fmt_duration(0) == "0s"
    assert _fmt_duration(45) == "45s"
    assert _fmt_duration(192) == "3m 12s"
    assert _fmt_duration(-5) == "0s"

    assert _fmt_eta(0) == "~1s"
    assert _fmt_eta(30) == "~30s"
    assert _fmt_eta(59.9) == "~1m"          # rounds to a whole minute -> the minute bucket
    assert _fmt_eta(125) == "~2m"

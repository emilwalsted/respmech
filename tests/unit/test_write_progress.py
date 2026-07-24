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

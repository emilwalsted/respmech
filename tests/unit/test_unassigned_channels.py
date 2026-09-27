"""An incomplete or out-of-range channel mapping must be reported, never guessed at.

Two defects lived here. A channel left unassigned reached pandas as ``None`` and surfaced
as "unsupported operand type(s) for -: 'NoneType' and 'int'" on a worker thread — and, far
worse, column 0 became ``iloc[:, -1]``, so the analysis silently ran on the LAST column of
the recording and reported nothing at all. ``Settings.validate`` never checked either.

Both were masked in the GUI by the Setup spin boxes, whose minimum of 1 meant a required
channel could not be unset. The channel-assignment dialog is becoming the only way to
assign channels, so they can be — hence the loader-level resolution plus a gate covering
every preview job, not only the test run.
"""
import os

import pytest

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_FILE = os.path.join(INPUT, "synth_case_A.csv")


def _load(**channels):
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core.io.loaders import load
    s = synth_settings("")
    for k, v in channels.items():
        setattr(s.input.channels, k, v)
    return load(_FILE, to_legacy_ns(s))


# -- the loader names the setting instead of leaking a numpy/pandas error ------
def test_an_unassigned_channel_is_named(tmp_path):
    from respmech.core.io.loaders import DataValidationError
    with pytest.raises(DataValidationError) as e:
        _load(flow=None)
    assert "Flow channel" in str(e.value) and "not assigned" in str(e.value)


def test_column_zero_is_rejected_rather_than_wrapping_to_the_last_column(tmp_path):
    """The dangerous one: iloc[:, -1] produced a complete, plausible analysis of an
    unrelated channel. A crash would have been kinder than what this did."""
    from respmech.core.io.loaders import DataValidationError
    with pytest.raises(DataValidationError) as e:
        _load(flow=0)
    assert "numbered from 1" in str(e.value)


def test_a_column_past_the_end_of_the_file_says_so(tmp_path):
    from respmech.core.io.loaders import DataValidationError
    with pytest.raises(DataValidationError) as e:
        _load(flow=99)
    msg = str(e.value)
    assert "column 99" in msg and "synth_case_A.csv" in msg and "12 column" in msg


@pytest.mark.parametrize("field, value", [("entropy", [99]), ("emg", [99])])
def test_out_of_range_emg_and_entropy_columns_are_caught_too(field, value):
    """Settings.validate never inspects these two at all, so the loader is the only place
    the check can happen — it is the only layer that knows the file's width."""
    from respmech.core.io.loaders import DataValidationError
    with pytest.raises(DataValidationError) as e:
        _load(**{field: value})
    assert "column 99" in str(e.value)


def test_a_valid_mapping_still_loads(tmp_path):
    flow, *_ = _load()
    assert len(flow) > 1000


# -- the preview gates EVERY job on the mapping, not just the test run --------

# -- The preview gate reads Settings.validate(), which is itself generic over
# the DECLARED signal set (core.analysis.signals) so it must gate
# correctly whichever shape an analysis declares, not only the historical "full" set
# with every channel assigned. Every test below is parametrized over the three shapes
# a real analysis can currently declare (``signals=None`` derives 'full' from the
# synthetic default's fully-assigned channels, exactly as before this ticket) and
# always breaks 'flow' -- the one channel every one of the three shapes requires --
# so the SAME assertions the original, unparametrized tests made are now proven to
# hold for flow-only and flow+poes as well, not just for the full set.
_SIGNAL_SETS = pytest.mark.parametrize(
    "signals", [None, ["flow"], ["flow", "poes"]], ids=["full", "flow_only", "poes_only"])


def _gated_settings(tmp_path, signals):
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    if signals is not None:
        s.analysis.signals = list(signals)
    return s


@_SIGNAL_SETS
def test_every_preview_job_is_gated_on_a_valid_mapping(qapp, tmp_path, signals):
    """Regression: only "batch" consulted _settings_ok, so an unassigned channel left the
    other five kinds to launch, load, and paint a raw traceback into their panels."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = _gated_settings(tmp_path, signals)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv.state.settings.input.channels.flow = None            # as the dialog can now leave it

    # NOT "the tokens did not move": the gate cancels in-flight work, which bumps them by
    # design. What must not happen is a worker being STARTED.
    for kind in ("mech", "batch", "ecg", "emg_all", "emg_detail", "noise"):
        pv._schedule(kind)
        assert kind not in pv._jobs, f"{kind} launched a worker with no flow channel"
    assert not pv._launch_queue, "a worker was queued with no flow channel"
    assert "incomplete" in pv.status.text().lower()
    # a non-default (explicit) declared set names ITSELF as the reason (Settings.validate:
    # "... is required by analysis.signals"), proving the gate actually consulted the
    # declared set here rather than a hardcoded "flow" check that would read identically
    # for every one of these three parametrizations
    if signals is not None:
        assert "analysis.signals" in pv.status.text()
    win.close()


@_SIGNAL_SETS
def test_a_valid_mapping_still_schedules(qapp, tmp_path, signals):
    """The gate must not be so wide that it blocks ordinary work, for any declared set."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = _gated_settings(tmp_path, signals)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    before = pv._tokens["mech"]
    pv._schedule("mech")
    assert pv._tokens["mech"] == before + 1
    pv.shutdown()
    win.close()


@pytest.mark.parametrize("signals, poes_required", [
    (["flow"], False),
    (["flow", "poes"], True),
], ids=["flow_only-poes_unused", "poes_only-poes_required"])
def test_a_channel_not_required_by_the_declared_set_does_not_gate_the_preview(
        qapp, tmp_path, signals, poes_required):
    """The DECLARED set, not merely 'is every channel on the form filled in', decides
    what the gate requires: a flow-only analysis still carries the synthetic default's
    poes channel (nothing clears it when the set narrows), and clearing THAT channel
    must not block the preview, since analysis.signals=['flow'] never required it in the
    first place. Without this, gating on "every assigned channel must stay assigned"
    would needlessly block a perfectly runnable flow-only analysis the moment poes was
    dropped from Setup for an unrelated reason.

    Parametrized against its own contrasting case (self-review finding): clearing the
    SAME channel under signals=['flow','poes'] -- where poes IS required -- must still
    block, or this test would pass just as well if the gate ignored the declared set
    entirely and always let a narrow-enough state through."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = _gated_settings(tmp_path, signals)
    assert s.input.channels.poes is not None, "nothing clears it -- the point is it's unused"
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv.state.settings.input.channels.poes = None
    before = pv._tokens["mech"]
    pv._schedule("mech")
    if poes_required:
        # NOT "the token did not move" -- gating cancels in-flight work, which bumps the
        # token by design (see test_every_preview_job_is_gated_on_a_valid_mapping above);
        # what must not happen is a worker being STARTED.
        assert "mech" not in pv._jobs, "a required channel's removal was not gated"
    else:
        assert pv._tokens["mech"] == before + 1, "an irrelevant channel blocked the preview"
        assert "mech" in pv._jobs
        pv.shutdown()
    win.close()


@_SIGNAL_SETS
def test_clearing_the_mapping_blanks_the_plots_not_just_the_spinners(qapp, tmp_path, signals):
    """Regression: the Mechanics tab is never removed the way the EMG sub-tabs are, and
    sync_from_settings does not clear panels — so clearing the channel mapping after a
    preview left the previous mapping's traces on screen, reading as a current result for
    settings that can no longer produce one."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.workers import stage_mechanics_preview
    s = _gated_settings(tmp_path, signals)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(pv.state.settings, _FILE))
    qapp.processEvents()
    assert pv._channel_plots, "the fixture never drew anything, so this proves nothing"

    pv.state.settings.input.channels.flow = None      # as cancelling the picker leaves it
    pv._schedule("mech")
    qapp.processEvents()
    assert not pv._channel_plots, "stale traces from the previous mapping stayed on screen"
    assert "incomplete" in pv.status.text().lower()
    win.close()


@_SIGNAL_SETS
def test_the_blank_re_arms_when_the_settings_become_valid_again(qapp, tmp_path, signals):
    """It blanks once on the way into an invalid state; a later invalid transition must
    blank again rather than being suppressed by a latched flag."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = _gated_settings(tmp_path, signals)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    pv.state.settings.input.channels.flow = None
    pv._schedule("mech")
    assert pv._blanked_for_invalid is True
    pv.state.settings.input.channels.flow = 5         # valid again
    pv._schedule("mech")
    assert pv._blanked_for_invalid is False, "the blank never re-armed"
    pv.shutdown()
    win.close()


@_SIGNAL_SETS
def test_a_job_launched_while_valid_cannot_repaint_the_blanked_panels(qapp, tmp_path, signals):
    """Regression: blanking does not bump the kind tokens, so a worker started while the
    settings were still valid completed afterwards, passed the acceptance check in
    _on_job_done, and repainted the traces we had just cleared — over a status line claiming
    success. The blank cancels every kind first."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.screens.preview_screen import _Job
    from respmech.ui.workers import stage_mechanics_preview
    s = _gated_settings(tmp_path, signals)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    result = stage_mechanics_preview(pv.state.settings, _FILE)

    token = pv._tokens["mech"] + 1                 # a job launched while settings were valid
    pv._tokens["mech"] = token
    job = _Job("mech", token, QThread(), object())
    pv._jobs["mech"] = job

    pv.state.settings.input.channels.flow = None   # ...and now they are not
    pv._schedule("mech")
    qapp.processEvents()
    pv._on_job_done(job, result)                   # the in-flight result lands late
    qapp.processEvents()
    assert not pv._channel_plots, "a stale result repainted the blanked panel"
    assert "incomplete" in pv.status.text().lower(), "the stale result overwrote the reason"
    win.close()


@_SIGNAL_SETS
def test_recovering_from_invalid_settings_rebuilds_every_panel(qapp, tmp_path, signals):
    """The blank is global, so the recovery must be too. sync_from_settings scopes a rebuild
    to the kinds the repairing edit touched — right for an ordinary edit, wrong after a
    blank: repairing a field that scopes to {"batch"} alone would leave five panels empty
    for good, under a status line reading success."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.screens.preview_screen import _AUTO_KINDS
    s = _gated_settings(tmp_path, signals)
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")

    pv.state.settings.input.channels.flow = None
    pv._schedule("mech")
    assert pv._blanked_for_invalid is True

    pv.state.settings.input.channels.flow = 5          # repaired
    pv._pending_kinds = set()
    pv._schedule("mech")
    assert pv._blanked_for_invalid is False
    # every kind the blank cleared is queued again, not just the one that was scheduled
    assert pv._pending_kinds == set(_AUTO_KINDS), pv._pending_kinds
    pv.shutdown()
    win.close()


def test_the_ecg_capture_channel_cannot_point_past_the_assigned_emg(qapp, tmp_path):
    """detect_channel is an INDEX into input.channels.emg, not a column number, so
    re-assigning the EMG channels re-points it — and a shorter list points it past the end.
    The combo clamped for display only, so the screen showed one electrode while the model
    held an index the core would crash on (emgcols[:, detect]). Measured: index 2 survived a
    re-assignment to two channels."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    s.processing.emg.remove_ecg = True
    s.processing.emg.detect_channel = 2            # the third of three EMG channels
    win = MainWindow(AppState(s))
    win.settings_screen._apply_channel_mapping(
        {"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9, "emg": [2, 3], "entropy": []})
    win.preview_screen.sync_from_settings()
    qapp.processEvents()

    e, cols = s.processing.emg, s.input.channels.emg
    assert 0 <= e.detect_channel < len(cols), "the capture channel points past the EMG list"
    # and the user is told, because which electrode the heartbeats come from is a scientific
    # choice rather than a detail to repair quietly
    assert "capture channel" in win.preview_screen.status.text().lower()
    win.close()

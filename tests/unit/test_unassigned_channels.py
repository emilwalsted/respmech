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
def test_every_preview_job_is_gated_on_a_valid_mapping(qapp, tmp_path):
    """Regression: only "batch" consulted _settings_ok, so an unassigned channel left the
    other five kinds to launch, load, and paint a raw traceback into their panels."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")
    pv.state.settings.input.channels.flow = None            # as the dialog can now leave it

    # NOT "the tokens did not move": the gate cancels in-flight work, which bumps them by
    # design. What must not happen is a worker being STARTED.
    for kind in ("mech", "batch", "ecg", "emg_all", "emg_detail", "noise"):
        pv._schedule(kind)
        assert kind not in pv._jobs, f"{kind} launched a worker with no flow channel"
    assert not pv._launch_queue, "a worker was queued with no flow channel"
    assert "incomplete" in pv.status.text().lower()
    win.close()


def test_a_valid_mapping_still_schedules(qapp, tmp_path):
    """The gate must not be so wide that it blocks ordinary work."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")
    before = pv._tokens["mech"]
    pv._schedule("mech")
    assert pv._tokens["mech"] == before + 1
    pv.shutdown()
    win.close()


def test_clearing_the_mapping_blanks_the_plots_not_just_the_spinners(qapp, tmp_path):
    """Regression: the Mechanics tab is never removed the way the EMG sub-tabs are, and
    sync_from_settings does not clear panels — so clearing the channel mapping after a
    preview left the previous mapping's traces on screen, reading as a current result for
    settings that can no longer produce one."""
    import os
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(pv.state.settings, _FILE))
    qapp.processEvents()
    assert pv._channel_plots, "the fixture never drew anything, so this proves nothing"

    pv.state.settings.input.channels.flow = None      # as cancelling the picker leaves it
    pv._schedule("mech")
    qapp.processEvents()
    assert not pv._channel_plots, "stale traces from the previous mapping stayed on screen"
    assert "incomplete" in pv.status.text().lower()
    win.close()


def test_the_blank_re_arms_when_the_settings_become_valid_again(qapp, tmp_path):
    """It blanks once on the way into an invalid state; a later invalid transition must
    blank again rather than being suppressed by a latched flag."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")
    pv.state.settings.input.channels.flow = None
    pv._schedule("mech")
    assert pv._blanked_for_invalid is True
    pv.state.settings.input.channels.flow = 5         # valid again
    pv._schedule("mech")
    assert pv._blanked_for_invalid is False, "the blank never re-armed"
    pv.shutdown()
    win.close()


def test_a_job_launched_while_valid_cannot_repaint_the_blanked_panels(qapp, tmp_path):
    """Regression: blanking does not bump the kind tokens, so a worker started while the
    settings were still valid completed afterwards, passed the acceptance check in
    _on_job_done, and repainted the traces we had just cleared — over a status line claiming
    success. The blank cancels every kind first."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.screens.preview_screen import _Job
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")
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


def test_recovering_from_invalid_settings_rebuilds_every_panel(qapp, tmp_path):
    """The blank is global, so the recovery must be too. sync_from_settings scopes a rebuild
    to the kinds the repairing edit touched — right for an ordinary edit, wrong after a
    blank: repairing a field that scopes to {"batch"} alone would leave five panels empty
    for good, under a status line reading success."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.screens.preview_screen import _AUTO_KINDS
    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True,
                                                "savebreathbybreathdata": True})
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_combo.setCurrentText("synth_case_A.csv")

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

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

    for kind in ("mech", "batch", "ecg", "emg_all", "emg_detail", "noise"):
        before = dict(pv._tokens)
        pv._schedule(kind)
        assert pv._tokens == before, f"{kind} launched a worker with no flow channel"
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

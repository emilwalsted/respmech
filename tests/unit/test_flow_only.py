"""Compute-level guards for a flow-only signal set (no Poes/Pgas/Pdi).

These call ``calculatemechanics``/``calculateaveragebreaths`` directly on segments
built from the real synthetic recording (``_helpers.segment_synth_case``), bypassing
``core.pipeline.run_batch``/``core.results`` — narrower and faster than a full batch
run, and it isolates a failure to these two functions specifically. ``run_batch``
itself already succeeds end to end for a flow-only settings object once these guards
are in place (``core/pipeline.py``/``core/results.py`` are written generically over
whatever the mechanics table ends up containing, rather than hardcoding pressure
columns) — see ``test_flow_only_reaches_run_batch_end_to_end`` below, which pins
exactly that."""
import os

import numpy as np

from _helpers import compute_all_breaths, requires_synth, segment_synth_case, synth_settings


def _run(tmp_path, channels):
    settings = synth_settings(tmp_path, channels=channels)
    s, breaths, n_trimmed = segment_synth_case(settings)
    compute_all_breaths(s, breaths, n_trimmed)
    return s, breaths


@requires_synth()
def test_mechanics_columns_equal_full_channel_call(tmp_path):
    """Flow-only (poes/pgas/pdi absent): the columns that DO get computed are byte-identical
    (rtol 1e-9) to the same columns from a full-channel run on the same recording — every
    ``if caps.x:`` guard in ``calculatemechanics`` wraps the exact same statements that ran
    unconditionally before this ticket, so the full-channel path is untouched. Also pins the
    acceptance criteria's shape: ``volumeavg`` present, ``eilv[0]`` finite/``eilv[1]`` NaN,
    and only the pure-timing group of keys (no poes/pgas/pdi/vmr/tlr_insp/wob)."""
    _, full_breaths = _run(tmp_path / "full", channels={"emg": [], "entropy": []})
    _, fo_breaths = _run(
        tmp_path / "flow_only",
        channels={"poes": None, "pgas": None, "pdi": None, "emg": [], "entropy": []},
    )

    assert set(full_breaths) == set(fo_breaths)
    for bno, full_breath in full_breaths.items():
        fo_breath = fo_breaths[bno]
        fmechs, romechs = full_breath["mechanics"], fo_breath["mechanics"]
        for key, value in romechs.items():
            assert key in fmechs, f"breath {bno}: {key!r} not in full-channel mechanics"
            assert np.allclose(fmechs[key], value, rtol=1e-9, atol=0), (
                f"breath {bno}: {key!r} differs from the full-channel call "
                f"({fmechs[key]!r} vs {value!r})"
            )

        assert "volumeavg" in fo_breath
        assert np.isfinite(fo_breath["eilv"][0])
        assert np.isnan(fo_breath["eilv"][1])
        assert np.isfinite(fo_breath["eelv"][0])
        assert np.isnan(fo_breath["eelv"][1])
        assert "wob" not in fo_breath

    # The golden-locked timing-group quirks (documented, not fixed by this ticket): the
    # legacy names swap sign/expiratory-vs-inspiratory framing relative to what they sound
    # like at first glance. Both survive unchanged on the flow-only path.
    for breath in fo_breaths.values():
        m = breath["mechanics"]
        assert m["in_flow_midvol"] == m["flow_midvolinsp"]
        assert m["flow_midvolexp"] == -m["ex_flow_midvol"]

    expected_keys = {
        "flow_midvolexp", "flow_midvolinsp", "vol_endinsp", "vol_endexp",
        "max_in_flow", "max_ex_flow", "in_flow_midvol", "ex_flow_midvol",
        "ti", "te", "ttot", "ti_ttot", "vt", "bf", "ve",
    }
    for breath in fo_breaths.values():
        assert set(breath["mechanics"]) == expected_keys


@requires_synth()
def test_flow_only_reaches_run_batch_end_to_end(tmp_path):
    """The real public entry point, not the compute-level bypass the other tests in this
    file use: ``run_batch`` on a flow-only ``Settings`` object completes with no failed
    files, and the resulting breath table carries exactly the pure-timing columns (no
    poes/pgas/pdi/vmr/wob) -- ``core/pipeline.py``/``core/results.py`` needed no change of
    their own for this, since they read whatever columns ``calculatemechanics`` happens to
    produce rather than hardcoding the pressure family. Before this ticket's guards, this
    exact call raised inside ``calculateaveragebreaths`` (``ValueError: ... too short to
    average``, from resampling an empty Poes array) -- this test would have failed on
    ``origin/emil/vigilant-dijkstra-eui21m`` before this branch."""
    from respmech.core.pipeline import run_batch

    settings = synth_settings(
        tmp_path, channels={"poes": None, "pgas": None, "pdi": None, "emg": [], "entropy": []}
    )
    result = run_batch(settings, only_files=["synth_case_A.csv"])

    assert result.failed_files == {}
    assert "synth_case_A.csv" in result.ok_files
    cols = set(result.ok_files["synth_case_A.csv"].breaths_table.columns)
    assert "flow_midvolexp" in cols and "ti" in cols
    assert not (cols & {"poes_mininsp", "pgas_endinsp", "pdi_endinsp", "vmr", "wobtotal"})


@requires_synth()
def test_outlier_guard_unchanged(tmp_path):
    """K-035/K-204's outlier guard (``core/results.py::processoutliers``, owned by an
    earlier ticket) is gated on ``'poes_mininsp' in mechs.columns`` — this pins, at the
    compute level, that a flow-only breath's ``mechanics`` dict really does NOT carry that
    key (so the results-layer guard correctly skips outlier filtering instead of raising a
    KeyError trying to build ``rms_poes = rms_max / poes_mininsp``)."""
    _, breaths = _run(
        tmp_path,
        channels={"poes": None, "pgas": None, "pdi": None, "emg": [], "entropy": []},
    )
    for breath in breaths.values():
        assert "poes_mininsp" not in breath["mechanics"]
        assert "pgas_endinsp" not in breath["mechanics"]
        assert "pdi_endinsp" not in breath["mechanics"]
        assert "vmr" not in breath["mechanics"]
        assert "tlr_insp" not in breath["mechanics"]


@requires_synth()
def test_entropy_columns_equal_full_channel_run(tmp_path):
    """R8: sample entropy is a signal-set-independent capability, assignable to any
    channel regardless of which pressure/flow signals are declared — never an EMG (or
    anything else) companion. A flow-only run's entropy columns are therefore expected
    to be byte-identical (rtol 1e-9) to the same columns from a full-channel run on the
    same recording, through the real ``run_batch`` entry point (not the compute-level
    bypass the other tests in this file use)."""
    from respmech.core.pipeline import run_batch

    settings_full = synth_settings(tmp_path / "full", channels={"emg": []})
    result_full = run_batch(settings_full, only_files=["synth_case_A.csv"])

    settings_fo = synth_settings(
        tmp_path / "flow_only", channels={"poes": None, "pgas": None, "pdi": None, "emg": []}
    )
    result_fo = run_batch(settings_fo, only_files=["synth_case_A.csv"])

    bt_full = result_full.ok_files["synth_case_A.csv"].breaths_table
    bt_fo = result_fo.ok_files["synth_case_A.csv"].breaths_table
    entropy_cols = [c for c in bt_fo.columns if "entropy" in c]
    assert entropy_cols, "no entropy columns were produced at all"
    for col in entropy_cols:
        assert col in bt_full.columns
        assert np.allclose(bt_full[col].to_numpy(), bt_fo[col].to_numpy(), rtol=1e-9, atol=0), (
            f"{col!r} differs from the full-channel run"
        )


@requires_synth()
def test_processed_csv_has_only_present_channels(tmp_path):
    """The processed-data sheet (``core/results.py::build_processed_data``) already skips
    an absent channel by its empty-array convention (``if len(breath[key]) == 0: continue``,
    an earlier ticket's fix) — this pins it through the real ``run_batch`` entry point for a
    flow-only recording: Flow and Volume present, no Poes/Pgas/Pdi column."""
    from respmech.core.pipeline import run_batch

    settings = synth_settings(
        tmp_path, channels={"poes": None, "pgas": None, "pdi": None, "emg": []}
    )
    settings.output.data.save_processed = True
    result = run_batch(settings, only_files=["synth_case_A.csv"])

    processed = result.ok_files["synth_case_A.csv"].processed
    assert processed is not None and len(processed) > 0
    cols = set(processed.columns)
    assert {"Flow", "Volume"} <= cols
    assert not (cols & {"Poes", "Pgas", "Pdi"})


@requires_synth()
def test_mechanics_preview_series_without_pressures(tmp_path):
    """``stage_mechanics_preview``'s ``series`` dict now mirrors R1's "documented absence,
    not a hidden empty array" principle: a flow-only settings object gets back a ``series``
    with only ``flow``/``volume`` and no exception, on both the normal path and the
    ``TrimError`` fallback (mocked here — a real trim failure needs a broken recording, not
    this settings object)."""
    from unittest import mock

    from respmech.core import compute
    from respmech.ui.workers import stage_mechanics_preview

    settings = synth_settings(
        tmp_path, channels={"poes": None, "pgas": None, "pdi": None, "emg": []}
    )
    path = os.path.join(settings.input.folder, "synth_case_A.csv")

    result = stage_mechanics_preview(settings, path)
    assert result["trim_error"] is None
    assert set(result["series"]) == {"flow", "volume"}

    with mock.patch.object(compute, "trim", side_effect=compute.TrimError("boom")):
        fallback = stage_mechanics_preview(settings, path)
    assert fallback["trim_error"] == "boom"
    assert set(fallback["series"]) == {"flow", "volume"}


@requires_synth()
def test_no_campbell_paths_in_plan_and_none_written(tmp_path):
    """A flow-only signal set has no Poes, so the Campbell (per-file average,
    per-file individual, and cross-file cohort) figure jobs must never appear in
    ``plan_outputs``' ceiling, and a real write must not produce any of those paths
    either — the plan and the writer read the exact same
    ``core.plots.per_file_figure_jobs`` list, so this also proves the writer itself
    never tries to draw an empty/absent Poes trace. 'volume endpoints.pdf' (the drift
    diagnostic, gated on ``Capabilities.volume`` — always True here, flow implies
    volume) is still produced."""
    from respmech.core.io.plan import plan_outputs
    from respmech.core.io.writers import write_batch
    from respmech.core.pipeline import run_batch

    settings = synth_settings(
        tmp_path, channels={"poes": None, "pgas": None, "pdi": None, "emg": [], "entropy": []}
    )
    for flag in ("save_pv_average", "save_pv_individual", "save_raw", "save_trimmed",
                "save_drift"):
        setattr(settings.output.diagnostics, flag, True)

    files = [os.path.join(settings.input.folder, "synth_case_A.csv"),
            os.path.join(settings.input.folder, "synth_case_B.csv")]
    plan = plan_outputs(settings, files)
    assert not any("Campbell" in p for p in plan.all_paths())
    assert any("volume endpoints.pdf" in p for p in plan.all_paths())

    result = run_batch(settings)
    written = write_batch(result, settings, str(tmp_path))
    assert not any("Campbell" in os.path.basename(p) for p in written)
    assert any("volume endpoints.pdf" in os.path.basename(p) for p in written)


@requires_synth()
def test_mechanics_stack_renders_without_crashing_on_a_reduced_signal_set(qapp, tmp_path):
    """Regression, found by self-review before this ticket closed: a reduced signal set
    is already reachable today via ``channel_setup_dialog.py``'s declared-roles gating
    (an earlier ticket) — Preview & QC's Mechanics tab must not crash on it just because
    ``stage_mechanics_preview``'s ``series`` dict now OMITS absent pressure keys entirely
    instead of carrying them as empty arrays. ``ui/screens/preview/_mechanics.py``'s
    channel-stack render loop indexed ``series[key]`` unconditionally over a hardcoded
    5-channel list before this fix — a real ``KeyError`` on exactly this configuration,
    reproduced here through the actual render path, not a synthetic dict check."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.workers import stage_mechanics_preview

    settings = synth_settings(
        str(tmp_path), channels={"poes": None, "pgas": None, "pdi": None, "emg": [], "entropy": []}
    )
    win = MainWindow(AppState(settings))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    data = stage_mechanics_preview(
        pv.state.settings, os.path.join(settings.input.folder, "synth_case_A.csv")
    )
    pv._render_preview(data)   # must not raise KeyError
    assert len(pv._channel_plots) == 2   # flow + volume only -- no gap, no stale row
    win.close()

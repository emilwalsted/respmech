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

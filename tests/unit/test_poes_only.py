"""Compute-level guards for a poes-only signal set (flow + Poes, no Pgas/Pdi).

Same intersection test as ``test_flow_only.py`` (see its module docstring for why these
call ``calculatemechanics``/``calculateaveragebreaths`` directly instead of going through
``core.pipeline.run_batch``), but with Poes present: this pins that WOB survives (it only
needs flow + poes) while the pgas/pdi-only keys and ``vmr`` (which needs both poes AND
pgas) disappear."""
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
    """Poes-only (pgas/pdi absent): the columns that DO get computed are byte-identical
    (rtol 1e-9) to the same columns from a full-channel run. WOB is present (it only needs
    flow + poes); pgas/pdi-only keys and vmr (needs poes AND pgas) are absent; eilv/eelv
    are both finite (poes is present, unlike the flow-only case)."""
    _, full_breaths = _run(tmp_path / "full", channels={"emg": [], "entropy": []})
    _, po_breaths = _run(
        tmp_path / "poes_only", channels={"pgas": None, "pdi": None, "emg": [], "entropy": []}
    )

    assert set(full_breaths) == set(po_breaths)
    for bno, full_breath in full_breaths.items():
        po_breath = po_breaths[bno]
        fmechs, pomechs = full_breath["mechanics"], po_breath["mechanics"]
        for key, value in pomechs.items():
            assert key in fmechs, f"breath {bno}: {key!r} not in full-channel mechanics"
            assert np.allclose(fmechs[key], value, rtol=1e-9, atol=0), (
                f"breath {bno}: {key!r} differs from the full-channel call "
                f"({fmechs[key]!r} vs {value!r})"
            )

        assert "wob" in po_breath
        assert np.allclose(po_breath["wob"]["wobtotal"], full_breath["wob"]["wobtotal"],
                            rtol=1e-9, atol=0)
        assert np.isfinite(po_breath["eilv"][0])
        assert np.isfinite(po_breath["eilv"][1])
        assert np.isfinite(po_breath["eelv"][0])
        assert np.isfinite(po_breath["eelv"][1])

        assert "pgas_endinsp" not in pomechs
        assert "pdi_endinsp" not in pomechs
        assert "vmr" not in pomechs
        assert "tlr_insp" in pomechs   # needs only flow + poes, present here


@requires_synth()
def test_pgas_pdi_and_vmr_keys_absent(tmp_path):
    """``calculatemechanics``'s own ``if caps.x:`` guards (not ``registry.resolve()``,
    which ``compute.py`` does not call — only ``LEGACY_MECHANICS_ORDER`` itself is used,
    to pick the final key ORDER, not to decide which keys are computed): every pgas-only
    and pdi-only column, plus ``vmr``, is absent from a poes-only breath's mechanics, and
    every poes-only column IS present."""
    _, breaths = _run(
        tmp_path, channels={"pgas": None, "pdi": None, "emg": [], "entropy": []}
    )
    pgas_only_keys = {
        "pgas_endinsp", "pgas_endexp", "pgas_maxexp", "pgas_minexp",
        "exp_pgas_rise", "int_pgasexp", "ptp_pgasexp", "pgas_tidal_swing",
    }
    pdi_only_keys = {
        "int_pdiinsp", "ptp_pdiinsp", "pdi_minexp", "pdi_maxinsp",
        "pdi_endinsp", "pdi_endexp", "insp_pdi_rise", "pdi_tidal_swing",
    }
    poes_keys = {
        "poes_maxexp", "poes_mininsp", "poes_endinsp", "poes_endexp",
        "poes_midvolexp", "poes_midvolinsp", "int_oesinsp", "ptp_oesinsp",
        "poes_tidal_swing", "tlr_insp",
    }
    for breath in breaths.values():
        keys = set(breath["mechanics"])
        assert not (keys & pgas_only_keys), keys & pgas_only_keys
        assert not (keys & pdi_only_keys), keys & pdi_only_keys
        assert "vmr" not in keys
        assert poes_keys <= keys


@requires_synth()
def test_poes_only_reaches_run_batch_end_to_end(tmp_path):
    """The real public entry point (mirrors ``test_flow_only.py``'s equivalent test):
    ``run_batch`` on a poes-only ``Settings`` object completes with no failed files, WOB
    columns are present, and pgas/pdi/vmr columns are absent."""
    from respmech.core.pipeline import run_batch

    settings = synth_settings(
        tmp_path, channels={"pgas": None, "pdi": None, "emg": [], "entropy": []}
    )
    result = run_batch(settings, only_files=["synth_case_A.csv"])

    assert result.failed_files == {}
    cols = set(result.ok_files["synth_case_A.csv"].breaths_table.columns)
    assert "wobtotal" in cols and "poes_mininsp" in cols
    assert not (cols & {"pgas_endinsp", "pdi_endinsp", "vmr"})


@requires_synth()
def test_processed_csv_has_only_present_channels(tmp_path):
    """Mirrors ``test_flow_only.py``'s equivalent test: the processed-data sheet keeps
    Poes (present) but drops Pgas/Pdi (absent) for a poes-only recording, through the
    real ``run_batch`` entry point."""
    from respmech.core.pipeline import run_batch

    settings = synth_settings(
        tmp_path, channels={"pgas": None, "pdi": None, "emg": []}
    )
    settings.output.data.save_processed = True
    result = run_batch(settings, only_files=["synth_case_A.csv"])

    processed = result.ok_files["synth_case_A.csv"].processed
    assert processed is not None and len(processed) > 0
    cols = set(processed.columns)
    assert {"Flow", "Volume", "Poes"} <= cols
    assert not (cols & {"Pgas", "Pdi"})


@requires_synth()
def test_mechanics_preview_series_without_pgas_pdi(tmp_path):
    """Mirrors ``test_flow_only.py``'s equivalent test: a poes-only settings object gets
    back a ``series`` with ``flow``/``volume``/``poes`` but no ``pgas``/``pdi``, on both
    the normal path and the ``TrimError`` fallback (mocked here)."""
    from unittest import mock

    from respmech.core import compute
    from respmech.ui.workers import stage_mechanics_preview

    settings = synth_settings(tmp_path, channels={"pgas": None, "pdi": None, "emg": []})
    path = os.path.join(settings.input.folder, "synth_case_A.csv")

    result = stage_mechanics_preview(settings, path)
    assert result["trim_error"] is None
    assert set(result["series"]) == {"flow", "volume", "poes"}

    with mock.patch.object(compute, "trim", side_effect=compute.TrimError("boom")):
        fallback = stage_mechanics_preview(settings, path)
    assert fallback["trim_error"] == "boom"
    assert set(fallback["series"]) == {"flow", "volume", "poes"}


@requires_synth()
def test_mechanics_stack_renders_without_crashing_on_a_reduced_signal_set(qapp, tmp_path):
    """Mirrors ``test_flow_only.py``'s equivalent regression test: Preview & QC's
    Mechanics tab must not crash on a poes-only signal set either — three rows
    (flow/volume/poes), no gap, no ``KeyError`` on the omitted pgas/pdi keys."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    from respmech.ui.workers import stage_mechanics_preview

    settings = synth_settings(
        str(tmp_path), channels={"pgas": None, "pdi": None, "emg": [], "entropy": []}
    )
    win = MainWindow(AppState(settings))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename("synth_case_A.csv")
    data = stage_mechanics_preview(
        pv.state.settings, os.path.join(settings.input.folder, "synth_case_A.csv")
    )
    pv._render_preview(data)   # must not raise KeyError
    assert len(pv._channel_plots) == 3   # flow + volume + poes
    win.close()

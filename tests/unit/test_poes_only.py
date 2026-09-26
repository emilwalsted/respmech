"""Compute-level guards for a poes-only signal set (flow + Poes, no Pgas/Pdi).

Same intersection test as ``test_flow_only.py`` (see its module docstring for why these
call ``calculatemechanics``/``calculateaveragebreaths`` directly instead of going through
``core.pipeline.run_batch``), but with Poes present: this pins that WOB survives (it only
needs flow + poes) while the pgas/pdi-only keys and ``vmr`` (which needs both poes AND
pgas) disappear."""
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
    """The registry-driven skip (`resolve()`'s own capability gate, exercised here at the
    compute level): every pgas-only and pdi-only column, plus ``vmr``, is absent from a
    poes-only breath's mechanics, and every poes-only column IS present."""
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

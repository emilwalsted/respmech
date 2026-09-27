"""Compute-level guards for a pdi-only signal set (flow + Pdi, no Poes/Pgas).

Mirrors ``test_poes_only.py``/``test_pgas_only.py`` for the third single-pressure-channel
case — see ``test_pgas_only.py``'s module docstring for why this combination, though not
named in the ticket's own acceptance criteria, is worth a regression test rather than
relying on "the guard shape is the same for all three" as an argument alone."""
import numpy as np

from _helpers import compute_all_breaths, requires_synth, segment_synth_case, synth_settings


def _run(tmp_path, channels):
    settings = synth_settings(tmp_path, channels=channels)
    s, breaths, n_trimmed = segment_synth_case(settings)
    compute_all_breaths(s, breaths, n_trimmed)
    return s, breaths


@requires_synth()
def test_mechanics_columns_equal_full_channel_call(tmp_path):
    """Pdi-only (poes/pgas absent): the columns that DO get computed are byte-identical
    (rtol 1e-9) to the same columns from a full-channel run. WOB, tlr_insp and vmr are
    absent (all need poes and/or pgas); poes/pgas-only keys are absent."""
    _, full_breaths = _run(tmp_path / "full", channels={"emg": [], "entropy": []})
    _, pdo_breaths = _run(
        tmp_path / "pdi_only", channels={"poes": None, "pgas": None, "emg": [], "entropy": []}
    )

    assert set(full_breaths) == set(pdo_breaths)
    for bno, full_breath in full_breaths.items():
        pdo_breath = pdo_breaths[bno]
        fmechs, pdomechs = full_breath["mechanics"], pdo_breath["mechanics"]
        for key, value in pdomechs.items():
            assert key in fmechs, f"breath {bno}: {key!r} not in full-channel mechanics"
            assert np.allclose(fmechs[key], value, rtol=1e-9, atol=0), (
                f"breath {bno}: {key!r} differs from the full-channel call "
                f"({fmechs[key]!r} vs {value!r})"
            )

        assert "wob" not in pdo_breath
        assert np.isfinite(pdo_breath["eilv"][0])
        assert np.isnan(pdo_breath["eilv"][1])   # poes-derived, poes absent here


@requires_synth()
def test_poes_pgas_and_vmr_and_wob_keys_absent(tmp_path):
    """Only the pdi-only column family (plus the always-present timing group) survives;
    every poes-only and pgas-only column, ``vmr`` and ``tlr_insp`` (both need poes) are
    gone."""
    _, breaths = _run(
        tmp_path, channels={"poes": None, "pgas": None, "emg": [], "entropy": []}
    )
    poes_only_keys = {
        "poes_maxexp", "poes_mininsp", "poes_endinsp", "poes_endexp",
        "poes_midvolexp", "poes_midvolinsp", "int_oesinsp", "ptp_oesinsp",
        "poes_tidal_swing", "tlr_insp",
    }
    pgas_only_keys = {
        "pgas_endinsp", "pgas_endexp", "pgas_maxexp", "pgas_minexp",
        "exp_pgas_rise", "int_pgasexp", "ptp_pgasexp", "pgas_tidal_swing",
    }
    pdi_keys = {
        "int_pdiinsp", "ptp_pdiinsp", "pdi_minexp", "pdi_maxinsp",
        "pdi_endinsp", "pdi_endexp", "insp_pdi_rise", "pdi_tidal_swing",
    }
    for breath in breaths.values():
        keys = set(breath["mechanics"])
        assert not (keys & poes_only_keys), keys & poes_only_keys
        assert not (keys & pgas_only_keys), keys & pgas_only_keys
        assert "vmr" not in keys
        assert pdi_keys <= keys

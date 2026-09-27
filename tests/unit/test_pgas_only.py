"""Compute-level guards for a pgas-only signal set (flow + Pgas, no Poes/Pdi).

Mirrors ``test_poes_only.py`` for the other single-pressure-channel case: this
combination is not named in the ticket's own acceptance criteria (only flow-only and
poes-only are), but the guard shape in ``calculatemechanics`` treats poes/pgas/pdi as
three structurally independent capabilities (the only cross-capability coupling is
``vmr``, which needs poes AND pgas together, and ``tlr_insp``, which needs only poes) —
so this closes a real regression-coverage gap for the "any subset of the three is
optional" claim, found in self-review, rather than testing something structurally new."""
import numpy as np

from _helpers import compute_all_breaths, requires_synth, segment_synth_case, synth_settings


def _run(tmp_path, channels):
    settings = synth_settings(tmp_path, channels=channels)
    s, breaths, n_trimmed = segment_synth_case(settings)
    compute_all_breaths(s, breaths, n_trimmed)
    return s, breaths


@requires_synth()
def test_mechanics_columns_equal_full_channel_call(tmp_path):
    """Pgas-only (poes/pdi absent): the columns that DO get computed are byte-identical
    (rtol 1e-9) to the same columns from a full-channel run. WOB and tlr_insp are absent
    (both need poes); pgas columns and vmr are absent too (vmr also needs poes); poes/pdi
    -only keys are absent."""
    _, full_breaths = _run(tmp_path / "full", channels={"emg": [], "entropy": []})
    _, pgo_breaths = _run(
        tmp_path / "pgas_only", channels={"poes": None, "pdi": None, "emg": [], "entropy": []}
    )

    assert set(full_breaths) == set(pgo_breaths)
    for bno, full_breath in full_breaths.items():
        pgo_breath = pgo_breaths[bno]
        fmechs, pgomechs = full_breath["mechanics"], pgo_breath["mechanics"]
        for key, value in pgomechs.items():
            assert key in fmechs, f"breath {bno}: {key!r} not in full-channel mechanics"
            assert np.allclose(fmechs[key], value, rtol=1e-9, atol=0), (
                f"breath {bno}: {key!r} differs from the full-channel call "
                f"({fmechs[key]!r} vs {value!r})"
            )

        assert "wob" not in pgo_breath
        assert np.isfinite(pgo_breath["eilv"][0])
        assert np.isnan(pgo_breath["eilv"][1])   # poes-derived, poes absent here


@requires_synth()
def test_poes_pdi_and_vmr_and_wob_keys_absent(tmp_path):
    """Only the pgas-only column family (plus the always-present timing group) survives;
    every poes-only and pdi-only column, ``vmr`` and ``tlr_insp`` (both need poes) are
    gone."""
    _, breaths = _run(
        tmp_path, channels={"poes": None, "pdi": None, "emg": [], "entropy": []}
    )
    poes_only_keys = {
        "poes_maxexp", "poes_mininsp", "poes_endinsp", "poes_endexp",
        "poes_midvolexp", "poes_midvolinsp", "int_oesinsp", "ptp_oesinsp",
        "poes_tidal_swing", "tlr_insp",
    }
    pdi_only_keys = {
        "int_pdiinsp", "ptp_pdiinsp", "pdi_minexp", "pdi_maxinsp",
        "pdi_endinsp", "pdi_endexp", "insp_pdi_rise", "pdi_tidal_swing",
    }
    pgas_keys = {
        "pgas_endinsp", "pgas_endexp", "pgas_maxexp", "pgas_minexp",
        "exp_pgas_rise", "int_pgasexp", "ptp_pgasexp", "pgas_tidal_swing",
    }
    for breath in breaths.values():
        keys = set(breath["mechanics"])
        assert not (keys & poes_only_keys), keys & poes_only_keys
        assert not (keys & pdi_only_keys), keys & pdi_only_keys
        assert "vmr" not in keys
        assert pgas_keys <= keys

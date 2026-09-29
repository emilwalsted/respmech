"""Coverage for the vendored entropy helpers (pyentrp-derived). Only ``sample_entropy``
is on the analysis path; the rest (multiscale / permutation / composite) shipped
untested (audit #30). These are characterisation tests — they pin basic invariants so
the helpers are exercised, not silently rotting."""
import numpy as np

from respmech.core import entropy as ent


def _rng():
    # deterministic pseudo-signal without Math.random / global seed leakage
    x = np.linspace(0, 20 * np.pi, 2000)
    return np.sin(x) + 0.1 * np.sin(7.3 * x)


def test_shannon_entropy_nonnegative_and_higher_for_more_symbols():
    assert ent.shannon_entropy([1, 1, 1, 1]) == 0.0                 # one symbol → zero
    assert ent.shannon_entropy([1, 2, 3, 4]) > ent.shannon_entropy([1, 1, 2, 2])


def test_sample_entropy_finite_and_length_matches_sample_length():
    se = ent.sample_entropy(_rng(), 3, tolerance=0.2)
    assert len(se) == 3 and np.all(np.isfinite(se))


def test_permutation_entropy_bounds():
    pe = ent.permutation_entropy(_rng(), order=3, delay=1, normalize=True)
    assert 0.0 <= pe <= 1.0                                          # normalised → [0, 1]
    assert ent.permutation_entropy(np.arange(100), order=3) < 0.1    # monotonic → near-zero


def test_multiscale_permutation_entropy_returns_one_value_per_scale():
    mspe = ent.multiscale_permutation_entropy(_rng(), m=3, delay=1, scale=4)
    assert len(mspe) == 4 and np.all(np.isfinite(mspe))


def test_multiscale_entropy_runs_over_scales():
    mse = ent.multiscale_entropy(_rng(), sample_length=2, tolerance=0.2, maxscale=3)
    assert len(mse) == 3


# --- entropy on the conditioned volume (coincidence rule per role) -------------------------

import os
import dataclasses

import pytest

from respmech.core import entropy as entlib
from respmech.core.io import writers
from respmech.core.pipeline import run_batch
from respmech.core.settings import Settings, SettingsError
from respmech.settingsio.toml_io import load_toml

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "..", "golden")
_SCENARIO = os.path.join(_GOLDEN, "scenarios", "entropy_on_volume.toml")


def _settings(tmp_path, *, entropy, derived=(), integrate=False, volume=6):
    s = load_toml(_SCENARIO)
    s.input.folder = os.path.join(_GOLDEN, "input")
    s.input.files = "synth_case_A.csv"
    s.output.folder = str(tmp_path)
    s.input.channels.entropy = list(entropy)
    s.input.channels.entropy_derived = list(derived)
    s.input.channels.volume = volume
    s.processing.volume.integrate_from_flow = integrate
    return s


def _run(settings):
    res = run_batch(settings)
    fr = next(iter(res.ok_files.values()))
    return fr


def _raw_column(col):
    import pandas as pd
    df = pd.read_csv(os.path.join(_GOLDEN, "input", "synth_case_A.csv"))
    return df.iloc[:, col - 1].to_numpy()


def test_volume_column_entropy_uses_the_conditioned_volume(tmp_path):
    s = _settings(tmp_path, entropy=[6, 10])
    fr = _run(s)
    m, tol = s.processing.entropy.epochs, s.processing.entropy.tolerance
    first = fr.breaths[min(fr.breaths)]
    # the entropy window is the breath's own (inspiration + the joint sample + expiration),
    # sliced from the CONDITIONED volume: its two phases ARE the breath's phase volumes
    win = np.asarray(first["entcols"], dtype=float)[:, 0]
    n_in = len(first["inspiration"]["volume"])
    n_ex = len(first["expiration"]["volume"])
    assert np.array_equal(win[:n_in], np.asarray(first["inspiration"]["volume"], dtype=float))
    assert np.array_equal(win[-n_ex:], np.asarray(first["expiration"]["volume"], dtype=float))
    expected = entlib.sample_entropy(win, m, tol * np.std(win))[-1]
    got = fr.breaths_table["sample_entropy_col_6"].iloc[0]
    assert got == pytest.approx(expected, abs=1e-9)
    # ... and it is NOT the entropy of the raw file column (drift correction is on)
    raw = _raw_column(6)
    rawwin = raw[: win.size]
    assert not np.allclose(win, rawwin)
    raw_entropy = entlib.sample_entropy(rawwin, m, tol * np.std(rawwin))[-1]
    assert abs(got - raw_entropy) > 1e-6


def test_volume_column_entropy_phases_use_the_phase_volume(tmp_path):
    s = _settings(tmp_path, entropy=[6])
    fr = _run(s)
    m, tol = s.processing.entropy.epochs, s.processing.entropy.tolerance
    first = fr.breaths[min(fr.breaths)]
    for phase, col in (("inspiration", "sample_entropy_insp_col_6"),
                       ("expiration", "sample_entropy_exp_col_6")):
        vol = np.asarray(first[phase]["volume"], dtype=float)
        expected = entlib.sample_entropy(vol, m, tol * np.std(vol))[-1]
        assert fr.breaths_table[col].iloc[0] == pytest.approx(expected, abs=1e-9)


def test_other_roles_keep_their_rule(tmp_path):
    """A flow column stays the raw trimmed column, and an ordinary entropy column is
    untouched by the volume rule."""
    with_volume = _run(_settings(tmp_path / "a", entropy=[10, 5]))
    without = _run(_settings(tmp_path / "b", entropy=[10, 5], volume=6))
    for col in ("sample_entropy_col_10", "sample_entropy_col_5"):
        assert list(with_volume.breaths_table[col]) == list(without.breaths_table[col])
    # flow (column 5): entropy of the raw trimmed flow, i.e. the breath's own flow array
    s = _settings(tmp_path / "c", entropy=[5])
    fr = _run(s)
    first = fr.breaths[min(fr.breaths)]
    flow = np.asarray(first["entcols"], dtype=float)[:, 0]     # the raw trimmed flow window
    assert np.array_equal(flow[: len(first["inspiration"]["flow"])],
                          np.asarray(first["inspiration"]["flow"], dtype=float))
    m, tol = s.processing.entropy.epochs, s.processing.entropy.tolerance
    expected = entlib.sample_entropy(flow, m, tol * np.std(flow))[-1]
    assert fr.breaths_table["sample_entropy_col_5"].iloc[0] == pytest.approx(expected, abs=1e-9)


def test_derived_volume_gives_three_columns_and_leaves_the_summary_alone(tmp_path):
    base = _run(_settings(tmp_path / "a", entropy=[10], integrate=True))
    with_derived = _run(_settings(tmp_path / "b", entropy=[10], derived=["volume"], integrate=True))
    new = [c for c in with_derived.breaths_table.columns if c not in base.breaths_table.columns]
    assert new == ["sample_entropy_col_volume", "sample_entropy_insp_col_volume",
                   "sample_entropy_exp_col_volume"]
    for c in ("sample_entropy_max", "sample_entropy_min", "sample_entropy_mean",
              "sample_entropy_col_10"):
        assert list(base.breaths_table[c]) == list(with_derived.breaths_table[c])
    # the derived volume is the same conditioned volume a volume COLUMN would give
    onvol = _run(_settings(tmp_path / "c", entropy=[6], integrate=True))
    for whole, ref in (("sample_entropy_col_volume", "sample_entropy_col_6"),
                       ("sample_entropy_insp_col_volume", "sample_entropy_insp_col_6"),
                       ("sample_entropy_exp_col_volume", "sample_entropy_exp_col_6")):
        assert list(with_derived.breaths_table[whole]) == list(onvol.breaths_table[ref])


def test_derived_volume_alone_computes_entropy_without_any_entropy_column(tmp_path):
    fr = _run(_settings(tmp_path, entropy=[], derived=["volume"], integrate=True))
    cols = [c for c in fr.breaths_table.columns if c.startswith("sample_entropy")]
    assert cols == ["sample_entropy_col_volume", "sample_entropy_insp_col_volume",
                    "sample_entropy_exp_col_volume"]


def test_empty_derived_list_adds_no_columns(tmp_path):
    fr = _run(_settings(tmp_path, entropy=[10], derived=[], integrate=True))
    assert not any("volume" in c and c.startswith("sample_entropy") for c in fr.breaths_table.columns)


def test_entropy_derived_validation(tmp_path):
    s = _settings(tmp_path, entropy=[10], derived=["flow"])
    with pytest.raises(SettingsError, match="unknown signal"):
        s.validate()
    # an EMG-only signal set has no volume for "volume" to mean
    s = _settings(tmp_path, entropy=[10], derived=["volume"], integrate=False, volume=None)
    s.analysis.signals = ["emg"]
    s.input.channels.flow = None
    s.input.channels.emg = [2, 3]
    s.processing.segmentation.method = "whole_file"
    with pytest.raises(SettingsError, match="needs a volume"):
        s.validate()
    _settings(tmp_path, entropy=[10], derived=["volume"], integrate=True, volume=None).validate()


def test_entropy_derived_round_trips_through_toml(tmp_path):
    from respmech.settingsio.toml_io import dumps_toml
    s = _settings(tmp_path, entropy=[10], derived=["volume"], integrate=True)
    path = tmp_path / "a.toml"
    path.write_text(dumps_toml(s), encoding="utf-8")
    assert load_toml(path).input.channels.entropy_derived == ["volume"]


def test_provenance_names_the_conditioned_volume_rule(tmp_path):
    def rows(**kw):
        df = writers._provenance_rows(_settings(tmp_path, **kw), None)
        return dict(zip(df["Key"], df["Value"]))
    derived = rows(entropy=[10], derived=["volume"], integrate=True, volume=None)
    assert derived["Entropy on derived volume"] == "conditioned (zero/drift/trend as configured)"
    assert "Sample entropy" in derived
    onvol = rows(entropy=[6, 10])
    assert "conditioned volume" in onvol["Entropy on the volume column"]
    none = rows(entropy=[10])
    assert "Entropy on derived volume" not in none and "Entropy on the volume column" not in none

"""An absent channel loads as an empty float64 array, never a Python list or a crash
(M-03, fase 0 of the modular-analysis program).

Volume was already optional (``np.isnan(column_volume)`` means "absent"); this extends
the SAME idiom to poes/pgas/pdi (``core/io/loaders.py``, ``core/_legacy_ns.py``'s
None -> ``math.nan`` mapping) so a later ticket (M-08) can let ``Settings.validate()``
allow a signal set without one of them, without touching the loader again. Flow keeps
its own unconditional, always-required resolution: an absent flow still raises today
(see ``test_unassigned_channels.py``'s ``test_an_unassigned_channel_is_named``, pinned
unchanged by this ticket's own acceptance criteria), and that path is deliberately not
touched here — see ``_absent``'s docstring in ``loaders.py`` for why.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.io.loaders import (
    DataValidationError, load, probe_constant_channels, validatedata,
)
from respmech.core.settings import Settings

_N = 20


def _settings(*, fs=100, pgas=4):
    """flow=1, volume=2, poes=3, pdi=5 always assigned; pgas is the channel under test
    (column 4 in the 5-column fixtures below, or unassigned when ``pgas=None``)."""
    s = Settings()
    s.input.format.sampling_frequency = fs
    c = s.input.channels
    c.flow, c.volume, c.poes, c.pdi = 1, 2, 3, 5
    c.pgas = pgas
    return s


def _series():
    return [np.linspace(0.0, 1.0, _N) + i for i in range(1, 6)]


def _write_csv(tmp_path):
    df = pd.DataFrame({f"c{i}": col for i, col in enumerate(_series(), start=1)})
    path = tmp_path / "rec.csv"
    df.to_csv(path, index=False)
    return str(path)


def _write_txt(tmp_path):
    df = pd.DataFrame({f"c{i}": col for i, col in enumerate(_series(), start=1)})
    path = tmp_path / "rec.txt"
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def _write_mat(tmp_path):
    # matlab_variant defaults to "mac" (matlabfileformat=2), which reads every top-level
    # variable in file order via data.items() — scipy.io always adds three metadata keys
    # (__header__/__version__/__globals__) before user data, so "windows" (format 1,
    # data["data_block1"], shape (channels, samples)) is the simpler, unambiguous fixture
    # to hand-build; it needs no assumption about scipy's own key ordering.
    block = np.array(_series())  # shape (5, _N): one row per channel
    path = tmp_path / "rec.mat"
    sio.savemat(str(path), {"data_block1": block})
    return str(path)


def _mat_settings(**kw):
    s = _settings(**kw)
    s.input.format.matlab_variant = "windows"
    return s


@pytest.mark.parametrize("write_fixture,build_settings", [
    (_write_csv, _settings),
    (_write_txt, _settings),
    (_write_mat, _mat_settings),
], ids=["csv", "txt", "mat"])
def test_an_absent_pgas_column_loads_as_an_empty_float_array(tmp_path, write_fixture, build_settings):
    filepath = write_fixture(tmp_path)
    s = build_settings(pgas=None)
    flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns = load(filepath, to_legacy_ns(s))

    assert isinstance(pgas, np.ndarray)
    assert pgas.shape == (0,)
    assert pgas.dtype == np.float64

    # the other four channels are unaffected by pgas being absent
    for name, arr in (("flow", flow), ("volume", volume), ("poes", poes), ("pdi", pdi)):
        assert len(arr) == _N, name
        assert np.asarray(arr).dtype == np.float64, name


def test_a_valid_pgas_mapping_still_loads_unaffected(tmp_path):
    """Baseline: the optional-channel path must not disturb the ordinary, all-assigned
    case it sits alongside."""
    filepath = _write_csv(tmp_path)
    s = _settings(pgas=4)
    flow, volume, poes, pgas, pdi, *_ = load(filepath, to_legacy_ns(s))
    assert len(pgas) == _N
    assert pgas.dtype == np.float64


def test_an_out_of_range_assigned_pgas_still_raises_unchanged(tmp_path):
    """A channel that IS assigned, just to a column past the end of the file, must still
    be reported precisely — the new absent-channel branch must never swallow this."""
    filepath = _write_csv(tmp_path)
    s = _settings(pgas=99)
    with pytest.raises(DataValidationError) as e:
        load(filepath, to_legacy_ns(s))
    assert "column 99" in str(e.value) and "5 column" in str(e.value)


def test_an_unassigned_flow_still_raises_unchanged(tmp_path):
    """Regression guard for the deliberate asymmetry: flow's own resolution in loaders.py
    is untouched by this ticket (see _absent's docstring) — only volume/poes/pgas/pdi
    gained the tolerant path. This mirrors test_unassigned_channels.py's own flow test,
    reproduced here against this file's own fixture rather than the committed synthetic
    recording."""
    filepath = _write_csv(tmp_path)
    s = _settings(pgas=4)
    s.input.channels.flow = None
    with pytest.raises(DataValidationError) as e:
        load(filepath, to_legacy_ns(s))
    assert "Flow channel" in str(e.value) and "not assigned" in str(e.value)


def test_validatedata_does_not_raise_a_length_mismatch_for_an_empty_pgas():
    """The acceptance criterion in prose: an empty pgas must not be compared for length
    against the other, genuinely populated channels."""
    flow = volume = poes = pdi = np.linspace(0.0, 1.0, _N)
    pgas = np.asarray([], dtype=float)
    validatedata(flow, volume, poes, pgas, pdi, np.asarray([]), np.asarray([]), settings=None)


def test_validatedata_still_catches_a_real_length_mismatch():
    """The skip is specific to genuinely absent (zero-length) channels — two channels
    that are both PRESENT but disagree in length must still be caught."""
    flow = np.linspace(0.0, 1.0, _N)
    volume = poes = pdi = flow
    pgas = np.linspace(0.0, 1.0, _N - 1)          # present, but one sample short
    with pytest.raises(DataValidationError) as e:
        validatedata(flow, volume, poes, pgas, pdi, np.asarray([]), np.asarray([]), settings=None)
    assert "lengths differ" in str(e.value)


def test_probe_constant_channels_tolerates_an_unassigned_pgas(tmp_path):
    """Already tolerant via getattr(ch, attr, None) per the ticket's own note — this pins
    it: a probe over a recording with Pgas left unassigned must not raise, and must not
    report a channel that was never there as constant."""
    df = pd.DataFrame({
        "c1": np.linspace(0.0, 1.0, 50),                    # flow: varying, not constant
        "c2": np.linspace(0.0, 1.0, 50),                    # volume
        "c3": np.linspace(0.0, 1.0, 50),                    # poes
        "c4": np.full(50, 7.0),                             # unused column, deliberately constant
        "c5": np.linspace(0.0, 1.0, 50),                    # pdi
    })
    path = tmp_path / "rec.csv"
    df.to_csv(path, index=False)
    s = _settings(pgas=None)
    result = probe_constant_channels(s, str(path))
    assert result == ()

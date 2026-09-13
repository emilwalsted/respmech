"""A channel written as an all-integer column in the source file must not reach
downstream arithmetic as int64 (ticket 20260913-2053).

A real recording had two pressure channels wired to unused ("dummy") ports and
written as a literal ``0`` throughout, so pandas inferred ``int64`` for those
columns. With Pgas pointed at such a column, ``compute.compute_breath``'s VMR
division crashed with ``UFuncTypeError: Cannot cast ufunc 'divide' output from
dtype('float64') to dtype('int64')`` — a precondition failure surfacing as a raw
numpy internals error instead of a result. The same happened to trigger with an
all-integer Pdi column too, and which channel is affected is essentially
arbitrary: it depends only on which computation happens to combine that column
with a ``np.zeros_like``/``np.divide`` pair somewhere downstream. The fix is
general: the loader casts every channel to float64 once, so no future
computation needs to care about the input file's original dtype. This test
pins that general dtype guarantee directly at the loader, for every one of the
five main channels, rather than only the one channel a division happens to
touch today.
"""
import os

import numpy as np
import pandas as pd
import pytest

from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.io.loaders import load
from respmech.core.settings import Settings

_CHANNELS = ["flow", "volume", "poes", "pgas", "pdi"]


def _settings(fs=100):
    s = Settings()
    s.input.format.sampling_frequency = fs
    c = s.input.channels
    c.flow, c.volume, c.poes, c.pgas, c.pdi = 1, 2, 3, 4, 5
    return to_legacy_ns(s)


def _write_csv(tmp_path, integer_channel):
    """Five plausible-looking pressure/flow columns; the one under test is written
    as literal integers (no decimal point), so pandas reads it back as int64 —
    exactly how an all-zero "dummy" channel round-trips through a real CSV export."""
    n = 20
    cols = {}
    for i, name in enumerate(_CHANNELS, start=1):
        if name == integer_channel:
            cols[name] = np.zeros(n, dtype=np.int64)
        else:
            cols[name] = np.linspace(0.0, 1.0, n) + i  # distinct, non-trivial floats
    df = pd.DataFrame(cols)
    path = tmp_path / "rec.csv"
    df.to_csv(path, index=False)
    # sanity: the fixture really does round-trip as int64 for the chosen column
    assert pd.read_csv(path)[integer_channel].dtype == np.int64
    return str(path)


@pytest.mark.parametrize("integer_channel", _CHANNELS)
def test_every_channel_is_cast_to_float64_regardless_of_source_dtype(tmp_path, integer_channel):
    filepath = _write_csv(tmp_path, integer_channel)
    flow, volume, poes, pgas, pdi = load(filepath, _settings())[:5]
    for name, arr in (("flow", flow), ("volume", volume), ("poes", poes),
                     ("pgas", pgas), ("pdi", pdi)):
        assert np.asarray(arr).dtype == np.float64, (
            f"{name} channel is {np.asarray(arr).dtype}, expected float64 "
            f"(source column under test: {integer_channel})")

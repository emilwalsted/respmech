"""End-to-end regression for ticket 20260913-2053: a real recording with an all-zero,
integer-typed Pgas column crashed ``run_batch`` with ``UFuncTypeError`` from the VMR
division in ``compute.compute_breath`` instead of producing a result -- the only
``np.zeros_like``/``np.divide`` combination anywhere in the codebase that depends on
channel dtype, and it depends only on Pgas (the VMR numerator). Pdi and Poes are
included here too, parametrized, as a plain "does an all-integer column of THIS
channel crash the batch" regression -- true today only for Pgas, but worth pinning
for all three now that the fix (the loader casting every channel to float64) makes no
channel-specific distinction. Uses the committed synthetic golden recording with one
column overwritten to round-trip as int64, exactly as a real "dummy" pressure channel
does.
"""
import os

import numpy as np
import pandas as pd
import pytest

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _copy_with_integer_channel(tmp_path, channel):
    df = pd.read_csv(os.path.join(INPUT, "synth_case_A.csv"))
    df[channel] = np.zeros(len(df), dtype=np.int64)
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    dest = in_dir / "synth_case_A.csv"
    df.to_csv(dest, index=False)
    assert pd.read_csv(dest)[channel].dtype == np.int64  # sanity: fixture is really int64
    return str(in_dir)


@pytest.mark.parametrize("channel", ["pgas", "pdi", "poes"])
def test_all_zero_integer_pressure_channel_does_not_crash_the_batch(tmp_path, channel):
    from respmech.core.pipeline import run_batch
    in_dir = _copy_with_integer_channel(tmp_path, channel)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    s = synth_settings(str(out_dir))
    s.input.folder = in_dir
    result = run_batch(s)
    assert result.failed_files == {}, result.failed_files
    assert result.ok_files, "no file produced a result"
    table = result.ok_files["synth_case_A.csv"].breaths_table
    assert len(table) > 0
    assert "vmr" in table.columns

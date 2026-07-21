"""A recording with exactly ONE EMG or entropy column.

Every fixture in the suite used [] or three columns, so the single-column layout was never
exercised — and it was broken from the very first step. The DataFrame loaders squeezed a
(N,1) selection to 1-D while every consumer indexes (samples, channels), so validatedata's
``.shape[1]`` raised "IndexError: tuple index out of range" before the recording finished
loading. The .mat path built its arrays with np.column_stack and was always right, but it
then hit the SAME defect one layer down, where the per-breath phase slices were squeezed.

These tests pin the 2-D contract at both layers and run a real batch end to end.
"""
import numpy as np

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _one_column(tmp_path, *, emg=(2,), entropy=(10,)):
    s = synth_settings(str(tmp_path), data_out=_OUT)
    s.input.channels.emg = list(emg)
    s.input.channels.entropy = list(entropy)
    return s


# -- layer 1: the loader keeps (samples, channels) -----------------------------
def test_loader_returns_2d_for_a_single_column(tmp_path):
    import os
    from respmech.core.io.loaders import load
    from respmech.core._legacy_ns import to_legacy_ns
    s = _one_column(tmp_path)
    *_, ent, emg = load(os.path.join(INPUT, "synth_case_A.csv"), to_legacy_ns(s))
    assert np.asarray(ent).ndim == 2 and np.asarray(ent).shape[1] == 1
    assert np.asarray(emg).ndim == 2 and np.asarray(emg).shape[1] == 1


def test_loader_shape_is_unchanged_for_several_columns(tmp_path):
    """The fix must be invisible to every analysis that already worked."""
    import os
    from respmech.core.io.loaders import load
    from respmech.core._legacy_ns import to_legacy_ns
    s = _one_column(tmp_path, emg=(2, 3, 4), entropy=(10, 11, 12))
    *_, ent, emg = load(os.path.join(INPUT, "synth_case_A.csv"), to_legacy_ns(s))
    assert np.asarray(ent).shape[1] == 3 and np.asarray(emg).shape[1] == 3


# -- layer 2: the per-breath phase slices stay 2-D too -------------------------
def test_phase_slices_keep_their_channel_axis(tmp_path):
    """Regression for the second squeeze: even when the loader got it right (the .mat path
    always did), the inspiration/expiration slices collapsed to 1-D and calculate_rms then
    iterated a 1-D array, raising "object of type 'numpy.float64' has no len()"."""
    from respmech.core.compute import _phase_dicts
    n = 100
    emg = np.random.RandomState(0).randn(n, 1)
    ent = np.random.RandomState(1).randn(n, 1)
    one = np.linspace(0.0, 1.0, n)
    exp, insp, _, _ = _phase_dicts((0, 50), (50, n), one, one, one, one, one, one, ent, emg)
    for d in (exp, insp):
        assert d["emgcols"].ndim == 2 and d["emgcols"].shape[1] == 1
        assert d["entcols"].ndim == 2 and d["entcols"].shape[1] == 1


# -- end to end ----------------------------------------------------------------
def test_batch_runs_with_a_single_emg_and_entropy_column(tmp_path):
    from respmech.core.pipeline import run_batch
    result = run_batch(_one_column(tmp_path))
    assert result.failed_files == {}, result.failed_files
    assert result.ok_files, "no file produced a result"
    table = next(iter(result.ok_files.values())).breaths_table
    assert any(c.startswith("rms_col_") for c in table.columns)
    assert any(c.startswith("sample_entropy_col_") for c in table.columns)


def test_batch_runs_when_the_single_entropy_column_is_also_the_emg_column(tmp_path):
    """Sample entropy is legitimately computed on a channel that also carries EMG — the
    shipped example config does exactly that. With one column it took a separate 2-D read
    (compute.py's entropy-from-EMG path) through the squeezed phase slice."""
    from respmech.core.pipeline import run_batch
    result = run_batch(_one_column(tmp_path, emg=(2,), entropy=(2,)))
    assert result.failed_files == {}, result.failed_files
    table = next(iter(result.ok_files.values())).breaths_table
    assert any(c.startswith("sample_entropy_col_") for c in table.columns)

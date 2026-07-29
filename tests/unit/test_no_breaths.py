"""Unit tests for the zero-breath precondition.

Background — the bug these lock. When a file yielded no breath to analyse,
``build_breath_table`` left ``mechs`` as the ``[]`` literal it starts as, and the average
row was then built with ``mechs.mean()`` — so the batch reported the file's error as
``AttributeError: 'list' object has no attribute 'mean'``, naming neither the file's real
problem nor anything the user could change. (With ``outlier_rms_sd_limit`` set it failed
even earlier, inside ``processoutliers``, with a different internal message.)

Two distinct causes reach it and are reported separately: segmentation detected nothing,
and every detected breath is excluded.
"""
import numpy as np
import pytest

from respmech.core import compute, pipeline
from respmech.core.compute import NoBreathsError
from respmech.core.results import build_breath_table
from respmech.core.settings import ExcludeEntry, Settings
from respmech.core._legacy_ns import to_legacy_ns

FS = 200


def _write(path, *, flat_volume=False, n_breaths=6, vt=0.6, period_s=4.0):
    """A minimal LabChart-shaped CSV; ``flat_volume`` gives a volume channel with no
    peaks at all, which is what makes volume segmentation find nothing."""
    per = int(period_s * FS)
    t = np.arange(n_breaths * per + 1) / FS
    amp = vt * np.pi / period_s
    flow = -amp * np.sin(2 * np.pi * t / period_s)
    one = (1 - np.cos(2 * np.pi * (t % period_s) / period_s)) / 2
    volume = np.zeros_like(t) if flat_volume else one * vt
    poes = -5.0 * np.sin(2 * np.pi * t / period_s) - 5.0
    pgas = 2.0 * np.sin(2 * np.pi * t / period_s) + 8.0
    np.savetxt(path, np.column_stack(
        [t, t * 0, t * 0, t * 0, flow, volume, poes, pgas, pgas - poes]), delimiter=",")


def _settings(src, out, *, method="flow", exclude=None):
    s = Settings()
    s.input.folder, s.input.files = str(src), "*.csv"
    s.input.format.sampling_frequency = FS
    c = s.input.channels
    c.flow, c.volume, c.poes, c.pgas, c.pdi = 5, 6, 7, 8, 9
    s.processing.segmentation.method = method
    if exclude is not None:
        s.processing.exclude_breaths = [ExcludeEntry(file=exclude[0], breaths=exclude[1])]
    s.output.folder = str(out)
    return s.validate()


def _dirs(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(); out.mkdir()
    return src, out


def test_a_file_with_no_detectable_breath_names_the_problem_not_a_pandas_internal(tmp_path):
    src, out = _dirs(tmp_path)
    _write(src / "flat.csv", flat_volume=True)
    result = pipeline.run_batch(_settings(src, out, method="volume"))
    fr = result.failed_files["flat.csv"]
    assert fr.error_kind == "NoBreathsError"
    assert "has no attribute" not in fr.error          # the old, internal message
    assert "No breaths were detected in flat.csv" in fr.error
    assert "volume" in fr.error                        # names the signal it split on
    assert "processing.segmentation" in fr.error       # …and the setting to change


def test_excluding_every_breath_is_reported_as_such(tmp_path):
    """A different cause with a different remedy: the breaths ARE there, the user has
    just excluded all of them."""
    src, out = _dirs(tmp_path)
    _write(src / "rec.csv", n_breaths=6)
    s = _settings(src, out, exclude=("rec.csv", list(range(1, 20))))
    fr = pipeline.run_batch(s).failed_files["rec.csv"]
    assert fr.error_kind == "NoBreathsError"
    assert "are excluded" in fr.error
    assert "exclude_breaths" in fr.error
    assert "No breaths were detected" not in fr.error   # not the other cause's message


def test_the_rest_of_the_batch_still_runs(tmp_path):
    """A per-file precondition failure must not take the run down with it."""
    src, out = _dirs(tmp_path)
    _write(src / "flat.csv", flat_volume=True)
    _write(src / "good.csv")
    result = pipeline.run_batch(_settings(src, out, method="volume"))
    assert set(result.failed_files) == {"flat.csv"}
    assert set(result.ok_files) == {"good.csv"}
    assert len(result.ok_files["good.csv"].breaths_table) > 0


@pytest.mark.parametrize("limit", [0.0, 3.0])
def test_build_breath_table_refuses_an_empty_breath_set(tmp_path, limit):
    """The guard has to sit ahead of the outlier pass too — with a non-zero limit the
    empty table used to fail inside processoutliers instead, with a third message."""
    src, out = _dirs(tmp_path)
    s = _settings(src, out)
    s.processing.emg.outlier_rms_sd_limit = limit
    with pytest.raises(NoBreathsError):
        build_breath_table("x.csv", {}, to_legacy_ns(s))


def test_check_breaths_passes_when_one_breath_survives():
    """The guard is about the USED breaths, not the detected ones — one included breath
    among excluded ones is a perfectly valid (if small) analysis."""
    breaths = {1: {"ignored": True}, 2: {"ignored": False}, 3: {"ignored": True}}
    s = to_legacy_ns(Settings())
    compute.check_breaths(breaths, "rec.csv", s)        # must not raise


def test_no_breaths_error_is_a_valueerror():
    """Same family as TrimError / VolumeTrendError, so the batch catches it per file
    rather than aborting the run."""
    assert issubclass(NoBreathsError, ValueError)

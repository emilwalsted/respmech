"""EMG-only segmentation (``core.analysis.segments`` + the ``whole_file``/``separators``
dispatch in ``compute.separateintobreaths``): a recording with no flow/pressure channel
is split into segments a downstream EMG/entropy analysis can run against, instead of the
usual inspiration/expiration breath split. See ``core.analysis.segments``'s own module
docstring for the two methods' shapes.
"""
import numpy as np
import pytest

from _helpers import requires_synth, synth_settings

from respmech.core import compute
from respmech.core import emg as emglib
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.analysis.segments import EmgSegmentationError, separators, whole_file
from respmech.core.settings import Settings

FS = 200


def _emg_only_synth_settings(tmp_path):
    return synth_settings(tmp_path, channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})


def _base_settings(*, emg=(2, 3), entropy=()):
    s = Settings()
    s.input.format.sampling_frequency = FS
    s.input.channels.emg = list(emg)
    s.input.channels.entropy = list(entropy)
    return s


def _emg_data(n_samples, n_channels=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.2, size=(n_samples, n_channels))


# -- core.analysis.segments, called directly ------------------------------------------

def test_whole_file_gives_one_segment_with_rms_equal_to_calculate_rms():
    """The ticket's own literal acceptance criterion."""
    n = 4 * FS
    timecol = np.arange(n) / FS
    emgcols = _emg_data(n)
    seg = whole_file("f.csv", timecol, emgcols, [], rms_s=0.05, fs=FS,
                     ignored_breaths=set(), kinds={})
    assert list(seg.keys()) == [1]
    segment = seg[1]
    assert segment["has_phases"] is False
    assert segment["number"] == 1
    assert segment["ignored"] is False
    assert segment["kind"] is None
    assert len(segment["time"]) == n
    for key in ("flow", "volume", "poes", "pgas", "pdi"):
        assert len(segment[key]) == 0
    expected_rms, expected_intemg = emglib.calculate_rms(emgcols, 0.05, FS)
    got_rms, got_intemg = emglib.calculate_rms(segment["emgcols"], 0.05, FS)
    assert got_rms == expected_rms
    assert got_intemg == expected_intemg


def test_three_separators_give_four_segments():
    """The ticket's other literal acceptance criterion."""
    n = 10 * FS
    timecol = np.arange(n) / FS
    emgcols = _emg_data(n)
    times_s = [2.0, 5.0, 8.0]
    segs = separators("f.csv", times_s, timecol, emgcols, [], FS,
                      ignored_breaths=set(), kinds={})
    assert list(segs.keys()) == [1, 2, 3, 4]
    bounds = [0] + [int(round(t * FS)) for t in times_s] + [n]
    for number, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        seg = segs[number]
        assert len(seg["time"]) == end - start
        assert np.allclose(np.asarray(seg["emgcols"]), emgcols[start:end])
        assert seg["has_phases"] is False


def test_zero_separators_is_the_same_shape_as_whole_file():
    n = 3 * FS
    timecol = np.arange(n) / FS
    emgcols = _emg_data(n)
    segs = separators("f.csv", [], timecol, emgcols, [], FS,
                      ignored_breaths=set(), kinds={})
    assert list(segs.keys()) == [1]
    assert len(segs[1]["time"]) == n


def test_a_separator_time_outside_the_recording_raises_named_error():
    n = 2 * FS
    timecol = np.arange(n) / FS
    emgcols = _emg_data(n)
    with pytest.raises(EmgSegmentationError, match="f.csv"):
        separators("f.csv", [5.0], timecol, emgcols, [], FS,
                  ignored_breaths=set(), kinds={})


def test_two_separator_times_rounding_to_the_same_sample_raises():
    n = 2 * FS
    timecol = np.arange(n) / FS
    emgcols = _emg_data(n)
    with pytest.raises(EmgSegmentationError):
        separators("f.csv", [1.0, 1.0 + 1e-9], timecol, emgcols, [], FS,
                  ignored_breaths=set(), kinds={})


def test_ignored_breaths_and_kinds_apply_by_segment_number():
    n = 6 * FS
    timecol = np.arange(n) / FS
    emgcols = _emg_data(n)
    segs = separators("f.csv", [2.0, 4.0], timecol, emgcols, [], FS,
                      ignored_breaths={2}, kinds={3: "rest"})
    assert segs[1]["ignored"] is False
    assert segs[2]["ignored"] is True
    assert segs[3]["kind"] == "rest"
    assert segs[1]["kind"] is None


def test_whole_file_rms_diagnostics_locate_the_peak_and_top3():
    """rms_file_max/t_rms_file_max/rms_file_top3: one value per EMG channel, the max
    equal to calculate_rms's own per-channel max, and the top3 mean bounded by it."""
    n = 5 * FS
    timecol = np.arange(n) / FS
    emgcols = np.zeros((n, 1))
    # A single, unambiguous burst placed well away from the edges so the rolling-RMS
    # window (rms_s=0.05 -> 10 samples at FS=200) is centred on it cleanly.
    burst_start = 2 * FS
    emgcols[burst_start:burst_start + 20, 0] = 5.0
    seg = whole_file("f.csv", timecol, emgcols, [], rms_s=0.05, fs=FS,
                     ignored_breaths=set(), kinds={})[1]
    expected_rms, _ = emglib.calculate_rms(emgcols, 0.05, FS)
    assert seg["rms_file_max"][0] == pytest.approx(expected_rms[0])
    assert 1.9 < seg["t_rms_file_max"][0] < 2.2          # lands inside the burst
    assert seg["rms_file_top3"][0] <= seg["rms_file_max"][0]
    assert seg["rms_file_top3"][0] > 0


def test_whole_file_rms_diagnostics_absent_without_emg_columns():
    n = 2 * FS
    timecol = np.arange(n) / FS
    seg = whole_file("f.csv", timecol, [], [], rms_s=0.05, fs=FS,
                     ignored_breaths=set(), kinds={})[1]
    assert "rms_file_max" not in seg


# -- compute.separateintobreaths dispatch, and the run through compute_segment_emg ----

def test_separateintobreaths_dispatches_whole_file_and_separators():
    s = to_legacy_ns(_base_settings())
    n = 4 * FS
    timecol = np.arange(n) / FS
    empty = np.array([])
    emgcols = _emg_data(n)
    breaths = compute.separateintobreaths(
        "whole_file", "f.csv", timecol, empty, empty, empty, empty, empty, [], emgcols, s)
    assert list(breaths.keys()) == [1]

    s2 = _base_settings()
    s2.processing.segmentation.method = "separators"
    from respmech.core.settings import SeparatorEntry
    s2.processing.segmentation.separators.append(
        SeparatorEntry(file="f.csv", times_s=[1.0, 2.0]))
    legacy2 = to_legacy_ns(s2)
    breaths2 = compute.separateintobreaths(
        "separators", "f.csv", timecol, empty, empty, empty, empty, empty, [], emgcols, legacy2)
    assert list(breaths2.keys()) == [1, 2, 3]


def test_entropy_and_ignore_are_read_from_settings_by_the_dispatch():
    """The dispatch (not the pure segments.py functions themselves, tested directly
    above) is responsible for resolving ignorebreaths/breathkinds/separator times from
    settings -- exercised here end to end through separateintobreaths."""
    from respmech.core.settings import ExcludeEntry, SeparatorEntry

    s = _base_settings()
    s.processing.segmentation.method = "separators"
    s.processing.segmentation.separators.append(
        SeparatorEntry(file="f.csv", times_s=[1.0, 2.0]))
    s.processing.exclude_breaths.append(ExcludeEntry(file="f.csv", breaths=[2]))
    legacy = to_legacy_ns(s)

    n = 3 * FS
    timecol = np.arange(n) / FS
    empty = np.array([])
    emgcols = _emg_data(n)
    breaths = compute.separateintobreaths(
        "separators", "f.csv", timecol, empty, empty, empty, empty, empty, [], emgcols, legacy)
    assert breaths[2]["ignored"] is True
    assert breaths[1]["ignored"] is False
    assert breaths[3]["ignored"] is False


def test_compute_segment_emg_runs_end_to_end_on_a_dispatched_segment():
    s = _base_settings(emg=(2, 3), entropy=(2,))
    legacy = to_legacy_ns(s)
    n = 3 * FS
    timecol = np.arange(n) / FS
    empty = np.array([])
    emgcols = _emg_data(n)
    entropycols = _emg_data(n, n_channels=1, seed=1)
    breaths = compute.separateintobreaths(
        "whole_file", "f.csv", timecol, empty, empty, empty, empty, empty,
        entropycols, emgcols, legacy)
    breath = breaths[1]
    compute.compute_segment_emg(breath, breath, legacy, None, None, True, "",
                               phases=breath["has_phases"])
    assert "rms" in breath and "rms_insp" not in breath
    assert "entropy" in breath and "entropy_insp" not in breath


# -- end to end through run_batch, on a real (committed) synthetic recording ----------

pytestmark = requires_synth()


def test_run_batch_whole_file_end_to_end(tmp_path):
    """The full path an EMG-only analysis actually takes: no flow channel assigned at
    all (the loader must not require one — see core/io/loaders.py), whole_file
    segmentation, a breath table with seg_start_s/seg_end_s/seg_duration_s and no
    flow-derived timing columns, and a processed-data CSV with no Flow/Volume/Poes/
    Pgas/Pdi columns (all four channels genuinely absent, not just zero-filled)."""
    from respmech.core.pipeline import run_batch

    s = _emg_only_synth_settings(tmp_path)
    s.processing.segmentation.method = "whole_file"
    s.input.files = "synth_case_A.csv"
    s.output.data.save_processed = True
    result = run_batch(s)
    fr = result.ok_files["synth_case_A.csv"]
    assert fr.error is None
    breath = next(iter(fr.breaths.values()))
    assert breath["has_phases"] is False
    assert list(breath["mechanics"].keys()) == ["seg_start_s", "seg_end_s", "seg_duration_s"]
    assert breath["mechanics"]["seg_duration_s"] > 0
    assert "rms" in breath
    cols = list(fr.breaths_table.columns)
    assert "seg_duration_s" in cols
    assert not any(c in cols for c in ("ti", "te", "ttot", "vt", "bf", "ve"))
    assert any(c.startswith("rms_file_max_col_") for c in cols)
    processed_cols = list(fr.processed.columns) if fr.processed is not None else []
    assert not any(c in processed_cols for c in ("Flow", "Volume", "Poes", "Pgas", "Pdi"))


def test_run_batch_separators_end_to_end(tmp_path):
    from respmech.core.pipeline import run_batch
    from respmech.core.settings import SeparatorEntry

    s = _emg_only_synth_settings(tmp_path)
    s.processing.segmentation.method = "separators"
    s.processing.segmentation.separators.append(
        SeparatorEntry(file="synth_case_A.csv", times_s=[1.0, 2.0]))
    s.input.files = "synth_case_A.csv"
    result = run_batch(s)
    fr = result.ok_files["synth_case_A.csv"]
    assert fr.error is None
    assert len(fr.breaths) == 3


def test_run_batch_reports_a_separator_out_of_range_as_a_soft_per_file_error(tmp_path):
    """The ticket's own acceptance criterion: a bad separator placement is a per-file
    FileResult error (EmgSegmentationError), never a batch-stopping crash."""
    from respmech.core.pipeline import run_batch
    from respmech.core.settings import SeparatorEntry

    s = _emg_only_synth_settings(tmp_path)
    s.processing.segmentation.method = "separators"
    s.processing.segmentation.separators.append(
        SeparatorEntry(file="synth_case_A.csv", times_s=[999999.0]))
    s.input.files = "synth_case_A.csv"
    result = run_batch(s)
    fr = result.failed_files["synth_case_A.csv"]
    assert fr.error_kind == "EmgSegmentationError"

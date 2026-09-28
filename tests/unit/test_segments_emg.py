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


def test_whole_file_rms_diagnostics_ignore_a_nan_sample():
    """A NaN sample (e.g. from upstream noise reduction) must never be silently picked
    as "the peak": rolling_rms's cumulative-sum grid means a NaN poisons every window
    from that sample onward (not just the ones directly overlapping it), so a NaN
    placed AFTER the real burst leaves the burst's own peak recoverable, while
    everything past the NaN correctly reads as NaN too -- never a wrong, plausible-
    looking timestamp paired with a NaN value (the bug this fix closes)."""
    n = 5 * FS
    timecol = np.arange(n) / FS
    emgcols = np.zeros((n, 1))
    burst_start = 1 * FS
    emgcols[burst_start:burst_start + 20, 0] = 5.0
    emgcols[3 * FS, 0] = np.nan          # after the burst -> the burst itself stays clean
    seg = whole_file("f.csv", timecol, emgcols, [], rms_s=0.05, fs=FS,
                     ignored_breaths=set(), kinds={})[1]
    assert np.isfinite(seg["rms_file_max"][0])
    assert 0.9 < seg["t_rms_file_max"][0] < 1.2          # lands inside the real, clean burst
    assert np.isfinite(seg["rms_file_top3"][0])
    # the mismatch this fix specifically closes: value and time must never disagree on
    # whether the peak is trustworthy.
    assert np.isnan(seg["rms_file_max"][0]) == np.isnan(seg["t_rms_file_max"][0])


def test_whole_file_rms_diagnostics_all_nan_channel_reports_nan_not_a_crash():
    n = 3 * FS
    timecol = np.arange(n) / FS
    emgcols = np.full((n, 1), np.nan)
    seg = whole_file("f.csv", timecol, emgcols, [], rms_s=0.05, fs=FS,
                     ignored_breaths=set(), kinds={})[1]
    assert np.isnan(seg["rms_file_max"][0])
    assert np.isnan(seg["t_rms_file_max"][0])
    assert np.isnan(seg["rms_file_top3"][0])


def test_one_sample_segment_keeps_a_length_one_time_array_not_a_scalar():
    """A separator placement that produces a 1-sample segment must not collapse
    breath["time"] to a 0-d scalar (no len(), no indexing) -- every downstream reader
    (build_processed_data, the diagnostic plots) assumes at least 1-D."""
    n = 5 * FS
    timecol = np.arange(n) / FS
    emgcols = _emg_data(n)
    segs = separators("f.csv", [1.0, 1.0 + 1.0 / FS], timecol, emgcols, [], FS,
                      ignored_breaths=set(), kinds={})
    one_sample = segs[2]
    assert len(one_sample["time"]) == 1
    assert one_sample["time"].shape == (1,)


# -- remap_segment_number (M-27's renumbering rule for a placed/removed separator) ----

def test_remap_is_the_identity_when_the_boundaries_do_not_change():
    from respmech.core.analysis.segments import remap_segment_number
    bounds = [0.0, 1.0, 2.0, 3.5]        # 4 segments
    for n in (1, 2, 3, 4):
        assert remap_segment_number(bounds, bounds, n) == n


def test_insertion_after_a_segment_leaves_its_own_number_and_start_unchanged():
    """Segments strictly BEFORE the inserted boundary keep both their number and
    their start time — nothing about them changed."""
    from respmech.core.analysis.segments import remap_segment_number
    old = [0.0, 1.0, 2.0]                # segments 1, 2, 3
    new = [0.0, 1.0, 1.5, 2.0]           # a new separator at 1.5 splits old segment 2
    assert remap_segment_number(old, new, 1) == 1


def test_insertion_before_a_segment_shifts_its_number_but_not_its_start():
    """The literal acceptance sequence: 3 separators (4 segments), exclude segment 3,
    insert a separator BEFORE it -- the exclusion must follow to the new segment 4.
    Segment 3 started at 2.0 in the old list; that instant is STILL an exact boundary
    in the new list, just with one more boundary ahead of it now."""
    from respmech.core.analysis.segments import remap_segment_number
    old = [0.0, 1.0, 2.0, 3.5]            # segments 1..4 (3 separators)
    new = [0.0, 1.0, 1.5, 2.0, 3.5]       # inserted at 1.5, strictly before segment 3's start
    assert remap_segment_number(old, new, 3) == 4
    # segment 4 (started at 3.5, after the inserted boundary) shifts the same way
    assert remap_segment_number(old, new, 4) == 5
    # segment 1 (started at 0.0, before everything) is untouched
    assert remap_segment_number(old, new, 1) == 1


def test_removing_a_boundary_merges_into_the_preceding_segment():
    """The removal/merge case: the boundary at 2.0 disappears, so what was segment 3
    (started at 2.0) now falls INSIDE segment 2 (which now spans 1.0..3.5)."""
    from respmech.core.analysis.segments import remap_segment_number
    old = [0.0, 1.0, 2.0, 3.5]             # segments 1..4
    new = [0.0, 1.0, 3.5]                  # the separator at 2.0 was removed
    assert remap_segment_number(old, new, 3) == 2
    # the segment before the removed boundary is unaffected
    assert remap_segment_number(old, new, 2) == 2
    # and the one after just shifts down by one, exactly like the merged segment
    assert remap_segment_number(old, new, 4) == 3


def test_remap_clamps_a_stale_number_beyond_the_old_bounds_instead_of_raising():
    """Stale data (a breath number the current separators no longer produce) has no
    better instant to compare with than the recording's own last known boundary --
    a renumbering aid, not a validator, so it clamps rather than raises IndexError."""
    from respmech.core.analysis.segments import remap_segment_number
    old = [0.0, 1.0, 2.0]
    new = [0.0, 1.0, 2.0, 2.5]
    assert remap_segment_number(old, new, 99) == remap_segment_number(old, new, 3)


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


def test_run_batch_refuses_unresolved_emg_only_noise_reduction_cleanly(tmp_path):
    """EMG-only noise reduction now has a real reference-resolution rule
    (resolve_noise_reference_mode) -- but a settings object with no rest-typed
    reference segment and no explicit reference_intervals is still 'unresolved', and
    Settings.validate() must still refuse it up front with a named, clean
    SettingsError, rather than let it reach the pipeline's unguarded machinery."""
    from respmech.core.pipeline import run_batch
    from respmech.core.settings import SettingsError

    s = _emg_only_synth_settings(tmp_path)
    s.processing.segmentation.method = "whole_file"
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.input.files = "synth_case_A.csv"
    with pytest.raises(SettingsError, match="no usable rest reference"):
        run_batch(s)


def test_run_batch_refuses_emg_only_auto_prop_cleanly(tmp_path):
    """A resolved reference (a rest-typed segment) is not enough on its own: auto_prop
    (choose the noise-reduction strength from active/quiet EMG pooled across the whole
    test) is built on inspiration/expiration PHASES and has no EMG-only
    implementation -- left unguarded, this would reach _build_noise_set's flow-only
    gather loop and crash the whole batch the same way the unresolved case above
    would. auto_prop defaults to True, so this is refused unless turned off."""
    from respmech.core.pipeline import run_batch
    from respmech.core.settings import BreathTypeEntry, SettingsError

    s = _emg_only_synth_settings(tmp_path)
    s.processing.segmentation.method = "whole_file"
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=1, kind="rest"))
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    assert s.processing.emg.noise.auto_prop is True
    s.input.files = "synth_case_A.csv"
    with pytest.raises(SettingsError, match="auto_prop"):
        run_batch(s)


# -- automatic methods: fixed_windows and emg_burst ------------------------------------

from respmech.core.analysis import segments as seglib          # noqa: E402
from respmech.core.analysis.segments import (                  # noqa: E402
    BURST_QC_COLUMNS, NEURAL_TIMING_COLUMNS, burst_masks, detect_bursts, emg_burst,
    fixed_windows)

BFS = 2000                          # sampling rate of the synthetic burst trains
_DETECT = dict(threshold_frac=0.3, min_s=0.2, smooth_s=0.1, min_contrast=1.5)


def _burst_train(onsets_s, dur_s, n_s=60, rest=0.02, seed=None, channels=1):
    """A deterministic carrier whose amplitude steps between ``rest`` and 1 (so its RMS
    is exactly known, unlike a noise burst, whose envelope fluctuates by several
    percent): returns the signal and the TRUE ``(onset, offset)`` sample pairs."""
    n = int(n_s * BFS)
    t = np.arange(n) / BFS
    env = np.full(n, rest)
    truth = []
    for o in onsets_s:
        i0, i1 = int(round(o * BFS)), int(round((o + dur_s) * BFS))
        env[i0:i1] = 1.0
        truth.append((i0, i1))
    sig = env * np.sin(2 * np.pi * 97 * t + 0.3)
    if seed is not None:
        sig = sig + np.random.default_rng(seed).normal(0, rest / 4, n)
    if channels > 1:
        sig = np.column_stack([sig] + [np.random.default_rng(i).normal(0, 0.01, n)
                                       for i in range(1, channels)])
    return sig, truth


def test_detect_bursts_recovers_a_known_train_within_two_samples():
    """The ticket's acceptance criterion: a synthetic burst train with a known onset and
    offset gives segment times within 2 samples."""
    onsets = np.arange(2.0, 55.0, 4.0)
    sig, truth = _burst_train(onsets, 1.5)
    det = detect_bursts(sig, BFS, **_DETECT)
    assert len(det.onsets) == len(truth)
    assert np.max(np.abs(det.onsets - np.array([a for a, _ in truth]))) <= 2
    assert np.max(np.abs(det.offsets - np.array([b for _, b in truth]))) <= 2
    assert det.n_channels_used == 1 and det.contrast > 10


def test_detect_bursts_locates_noisy_bursts_to_within_a_few_hundredths_of_a_second():
    """Stochastic EMG has an envelope that fluctuates, so the half-power edge is only as
    good as that fluctuation: measured 26 samples (13 ms) worst case at 2 kHz on the
    train below. Pinned loosely on purpose -- this is the honest precision, not 2 samples."""
    rng = np.random.default_rng(1)
    n = 60 * BFS
    sig = rng.normal(0, 0.05, n)
    truth = []
    for o in np.arange(2.0, 55.0, 4.0):
        i0, i1 = int(o * BFS), int((o + 1.5) * BFS)
        sig[i0:i1] += rng.normal(0, 0.5, i1 - i0)
        truth.append((i0, i1))
    det = detect_bursts(sig, BFS, **_DETECT)
    assert len(det.onsets) == len(truth)
    assert np.max(np.abs(det.onsets - np.array([a for a, _ in truth]))) <= 0.03 * BFS
    assert np.max(np.abs(det.offsets - np.array([b for _, b in truth]))) <= 0.03 * BFS


def test_pure_noise_raises_emg_segmentation_error():
    """The other half of the acceptance criterion: nothing to find is an error, never a
    segmentation of noise."""
    noise = np.random.default_rng(3).normal(0, 1, 60 * BFS)
    with pytest.raises(EmgSegmentationError, match="No EMG bursts found in f.csv"):
        detect_bursts(noise, BFS, filename="f.csv", **_DETECT)


def test_a_silent_recording_raises_instead_of_dividing_by_zero():
    with pytest.raises(EmgSegmentationError):
        detect_bursts(np.zeros(10 * BFS), BFS, **_DETECT)


def test_a_burst_shorter_than_burst_min_s_is_dropped_and_a_short_gap_is_bridged():
    # 0.6 s burst, 0.1 s gap (< 0.2 s), 0.6 s burst -> ONE burst; a lone 0.1 s blip -> none
    sig, _ = _burst_train([2.0], 0.6, n_s=8)
    sig2, _ = _burst_train([2.7], 0.6, n_s=8)
    bridged = sig + sig2 - np.sin(2 * np.pi * 97 * np.arange(sig.size) / BFS + 0.3) * 0.02
    det = detect_bursts(bridged, BFS, **_DETECT)
    assert len(det.onsets) == 1
    assert (det.offsets[0] - det.onsets[0]) / BFS > 1.2          # the two merged
    train, _ = _burst_train([2.0, 8.0], 1.0, n_s=12)
    blip, _ = _burst_train([5.0], 0.1, n_s=12)
    both = train + blip - np.sin(2 * np.pi * 97 * np.arange(train.size) / BFS + 0.3) * 0.02
    assert len(detect_bursts(both, BFS, **_DETECT).onsets) == 2   # the blip never counts


def test_a_channel_without_bursts_is_left_out_of_the_detection():
    sig, truth = _burst_train(np.arange(2.0, 55.0, 4.0), 1.5, channels=2)
    det = detect_bursts(sig, BFS, **_DETECT)
    assert det.n_channels_used == 1
    assert len(det.onsets) == len(truth)


def test_a_non_finite_sample_does_not_poison_the_envelope_after_it():
    sig, truth = _burst_train(np.arange(2.0, 55.0, 4.0), 1.5)
    sig[int(1.0 * BFS)] = np.nan
    det = detect_bursts(sig, BFS, **_DETECT)
    assert len(det.onsets) == len(truth)


def _segment_burst_train(**kw):
    onsets = [2.0, 6.0, 10.5, 14.0]
    sig, truth = _burst_train(onsets, 1.5, n_s=20)
    n = sig.size
    timecol = np.arange(n) / BFS
    segs = emg_burst("f.csv", timecol, sig[:, None], [], sig[:, None], BFS,
                     ignored_breaths=kw.get("ignored", set()), kinds=kw.get("kinds", {}),
                     **_DETECT)
    return segs, truth, n


def test_emg_burst_cuts_at_each_onset_and_the_last_segment_runs_to_the_end():
    segs, truth, n = _segment_burst_train()
    assert list(segs) == [1, 2, 3, 4]
    starts = [truth[k][0] for k in range(4)]
    for k, number in enumerate(segs):
        seg = segs[number]
        assert abs(int(round(seg["time"][0] * BFS)) - starts[k]) <= 2
        end = starts[k + 1] if k + 1 < 4 else n
        assert abs(len(seg["time"]) - (end - starts[k])) <= 3
        assert seg["has_phases"] is False
        assert seg["burst_span"][1] > seg["burst_span"][0]
        for key in ("flow", "volume", "poes", "pgas", "pdi"):
            assert len(seg[key]) == 0


def test_neural_timing_is_the_bursts_own_arithmetic():
    segs, truth, _ = _segment_burst_train()
    first = segs[1]["neural_timing"]
    assert list(first) == list(NEURAL_TIMING_COLUMNS)
    ti = (truth[0][1] - truth[0][0]) / BFS
    ttot = (truth[1][0] - truth[0][0]) / BFS
    assert first["ti_emg"] == pytest.approx(ti, abs=2 / BFS)
    assert first["ttot_emg"] == pytest.approx(ttot, abs=3 / BFS)
    assert first["te_emg"] == pytest.approx(first["ttot_emg"] - first["ti_emg"])
    assert first["ti_ttot_emg"] == pytest.approx(first["ti_emg"] / first["ttot_emg"])
    assert first["bf_emg"] == pytest.approx(60.0 / first["ttot_emg"])
    last = segs[4]["neural_timing"]                       # no following onset: not a short cycle
    assert last["ti_emg"] == pytest.approx(1.5, abs=2 / BFS)
    assert all(np.isnan(last[k]) for k in ("te_emg", "ttot_emg", "ti_ttot_emg", "bf_emg"))


def test_burst_qc_is_identical_on_every_segment_of_a_file():
    segs, truth, n = _segment_burst_train()
    qcs = [dict(s["emg_seg_qc"]) for s in segs.values()]
    assert all(q == qcs[0] for q in qcs)
    assert list(qcs[0]) == list(BURST_QC_COLUMNS)
    assert qcs[0]["emg_seg_n_bursts"] == 4
    assert qcs[0]["emg_seg_burst_frac"] == pytest.approx(4 * 1.5 * BFS / n, rel=0.01)


def test_emg_burst_ignored_and_kinds_apply_by_segment_number():
    segs, _, _ = _segment_burst_train(ignored={2}, kinds={3: "rest"})
    assert [s["ignored"] for s in segs.values()] == [False, True, False, False]
    assert segs[3]["kind"] == "rest" and segs[1]["kind"] is None


def test_burst_masks_are_the_bursts_and_the_guarded_periods_between_them():
    segs, _, n = _segment_burst_train()
    spans = [s["burst_span"] for s in segs.values()]
    burst, inter = burst_masks(segs, n, BFS, 0.1)
    guard = int(round(0.1 * BFS))
    expect_burst = np.zeros(n, bool)
    expect_inter = np.zeros(n, bool)
    for k, (on, off) in enumerate(spans):
        expect_burst[on:off] = True
        if k + 1 < len(spans):
            expect_inter[off + guard:spans[k + 1][0] - guard] = True
    assert np.array_equal(burst, expect_burst)
    assert np.array_equal(inter, expect_inter)
    assert not (burst & inter).any()
    assert not inter[:spans[0][0]].any() and not inter[spans[-1][1]:].any()   # nothing outside


def test_fixed_windows_tile_the_recording_and_drop_a_trailing_piece():
    n = int(23.0 * FS)
    timecol = np.arange(n) / FS
    emg = _emg_data(n)
    segs = fixed_windows("f.csv", timecol, emg, [], FS, window_s=5.0, hop_s=5.0,
                         ignored_breaths=set(), kinds={})
    assert list(segs) == [1, 2, 3, 4]                     # 3 s left over is dropped, not padded
    assert all(len(s["time"]) == 5 * FS and s["has_phases"] is False for s in segs.values())
    assert np.allclose(np.asarray(segs[2]["emgcols"]), emg[5 * FS:10 * FS])


def test_fixed_windows_hop_shorter_than_window_overlaps_and_longer_leaves_gaps():
    n = 20 * FS
    timecol = np.arange(n) / FS
    emg = _emg_data(n)
    over = fixed_windows("f.csv", timecol, emg, [], FS, window_s=4.0, hop_s=2.0,
                         ignored_breaths=set(), kinds={})
    assert len(over) == 9 and over[2]["time"][0] == pytest.approx(2.0)
    gaps = fixed_windows("f.csv", timecol, emg, [], FS, window_s=2.0, hop_s=5.0,
                         ignored_breaths=set(), kinds={})
    assert len(gaps) == 4 and gaps[2]["time"][0] == pytest.approx(5.0)


def test_fixed_windows_on_a_recording_shorter_than_one_window_raises_named_error():
    n = 3 * FS
    with pytest.raises(EmgSegmentationError, match="f.csv"):
        fixed_windows("f.csv", np.arange(n) / FS, _emg_data(n), [], FS, window_s=5.0,
                      hop_s=5.0, ignored_breaths=set(), kinds={})


def test_the_eight_new_columns_resolve_to_their_registry_units():
    from _helpers import assert_units
    assert_units({"ti_emg": "s", "te_emg": "s", "ttot_emg": "s", "ti_ttot_emg": "—",
                  "bf_emg": "min⁻¹", "emg_seg_n_bursts": "", "emg_seg_burst_frac": "—",
                  "emg_seg_contrast": "—"})


# -- the dispatch and the whole run ------------------------------------------------------

BURST_INPUT = "synth_emgburst_A.csv"


def _burst_run_settings(tmp_path, method="emg_burst"):
    s = _emg_only_synth_settings(tmp_path)
    s.input.folder = seglib.__file__ and __import__("_helpers").INPUT
    s.input.files = BURST_INPUT
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = method
    s.input.channels.entropy = []
    s.input.channels.emg = [2, 3, 4]
    return s


@requires_synth()
def test_run_batch_emg_burst_end_to_end_reports_timing_and_qc_columns(tmp_path):
    from respmech.core.pipeline import run_batch

    s = _burst_run_settings(tmp_path)
    result = run_batch(s)
    fr = result.ok_files[BURST_INPUT]
    assert fr.error is None
    assert len(fr.breaths) == 6
    cols = list(fr.breaths_table.columns)
    for c in NEURAL_TIMING_COLUMNS + BURST_QC_COLUMNS + ("seg_duration_s",):
        assert c in cols
    assert not any(c in cols for c in ("ti", "te", "ttot", "bf"))
    row = fr.breaths_table.iloc[0]
    assert 1.0 < row["ti_emg"] < 1.6 and 3.5 < row["ttot_emg"] < 4.5
    assert 13 < row["bf_emg"] < 17 and row["emg_seg_n_bursts"] == 6
    assert "rms_col_2" in cols


@requires_synth()
def test_run_batch_fixed_windows_end_to_end(tmp_path):
    from respmech.core.pipeline import run_batch

    s = _burst_run_settings(tmp_path, method="fixed_windows")
    s.processing.segmentation.emg.window_s = 4.0
    s.processing.segmentation.emg.hop_s = 4.0
    result = run_batch(s)
    fr = result.ok_files[BURST_INPUT]
    assert fr.error is None
    assert len(fr.breaths) == 7                           # 28.4 s -> seven whole 4 s windows
    assert not any(c in fr.breaths_table.columns for c in NEURAL_TIMING_COLUMNS)


@requires_synth()
def test_run_batch_reports_a_recording_without_bursts_as_a_soft_per_file_error(tmp_path):
    from respmech.core.pipeline import run_batch

    s = _burst_run_settings(tmp_path)
    s.input.files = "synth_case_A.csv"        # flow-bearing file read as EMG only: steady EMG, no gaps
    s.processing.segmentation.emg.burst_min_contrast = 50.0
    result = run_batch(s)
    fr = result.failed_files["synth_case_A.csv"]
    assert fr.error_kind == "EmgSegmentationError"


@requires_synth()
def test_bursts_are_detected_on_the_ecg_removed_signal_not_the_noise_reduced_one(tmp_path):
    """The noise profile is itself cut from the periods between bursts, so segmenting the
    already-reduced signal would be circular. ``segment_file`` therefore hands the
    detection the ECG-stage signal (``stages['detect']``), whatever the noise setting."""
    from respmech.core.pipeline import _process_emg, segment_file
    from respmech.core._legacy_ns import to_legacy_ns

    s = _burst_run_settings(tmp_path)
    ns = to_legacy_ns(s)
    raw = np.random.default_rng(0).normal(size=(500, 3))

    class _Reduce:
        def apply_columns(self, x):
            return np.asarray(x) * 0.5

    _emg, _diag, stages = _process_emg(ns, raw, 0, 500, noise_set=_Reduce())
    assert np.array_equal(stages["detect"], raw)          # remove_ecg is off: ECG stage == raw
    assert np.array_equal(stages["noise_reduced"], raw * 0.5)


@requires_synth()
def test_segmentation_settings_round_trip_through_toml(tmp_path):
    from respmech.settingsio import toml_io

    s = _burst_run_settings(tmp_path)
    e = s.processing.segmentation.emg
    e.window_s, e.hop_s, e.burst_threshold_frac = 2.5, 1.25, 0.4
    e.burst_min_s, e.burst_smooth_s, e.burst_min_contrast = 0.3, 0.05, 2.0
    path = tmp_path / "a.toml"
    toml_io.save_toml(s, str(path))
    back = toml_io.load_toml(str(path))
    assert back.processing.segmentation.method == "emg_burst"
    assert back.processing.segmentation.emg == e


@pytest.mark.parametrize("field,value", [
    ("window_s", 0), ("hop_s", -1.0), ("burst_min_s", 0), ("burst_smooth_s", float("nan")),
    ("burst_threshold_frac", 0.0), ("burst_threshold_frac", 1.0), ("burst_min_contrast", 1.0)])
def test_validate_rejects_nonsense_emg_segmentation_parameters(field, value):
    from respmech.core.settings import SettingsError
    s = _base_settings()
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "emg_burst"
    setattr(s.processing.segmentation.emg, field, value)
    with pytest.raises(SettingsError, match=f"segmentation.emg.{field}"):
        s.validate()


@requires_synth()
def test_run_report_and_provenance_name_the_automatic_methods(tmp_path):
    from respmech.core.io import writers
    from respmech.core.pipeline import run_batch

    s = _burst_run_settings(tmp_path)
    s.output.folder = str(tmp_path)
    writers.write_batch(run_batch(s), s, str(tmp_path))
    report = open(tmp_path / "run-report.txt", encoding="utf-8").read()
    assert "Segmentation:            EMG bursts (6)" in report
    frame = writers._provenance_rows(s, None)
    rows = dict(zip(frame["Key"], frame["Value"]))
    assert rows["Segmentation"] == "EMG bursts"
    assert "threshold 0.3" in rows["EMG burst detection"]
    s.processing.segmentation.method = "fixed_windows"
    assert writers._segmentation_provenance_value(s) == "fixed windows 5.0/5.0 s"
    assert "EMG burst detection" not in list(writers._provenance_rows(s, None)["Key"])

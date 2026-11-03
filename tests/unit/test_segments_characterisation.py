"""Characterisation of the EMG-only segmentation (``core.analysis.segments``).

Mutation testing showed that the existing segmentation tests check the shape of the
result (how many segments, how long) but not several of the values a downstream
analysis reads: the keys of a segment dict, the per-segment ``ignored``/``kind``
lookups for segments other than number 1, the slicing of the entropy columns, the
handling of non-finite samples in the envelope and the detrending, the exact
boundaries of the fixed-window and separator rules, and the onsets/offsets the burst
detector returns. These tests pin all of that against the current behaviour; none of
them changes a number.
"""
import numpy as np
import pytest

from respmech.core.analysis import segments as sg
from respmech.core.analysis.segments import EmgSegmentationError

FS = 500


# -- reference implementations, written as plain per-sample loops --------------------

def _ref_window(i, n, window):
    lo = max(i - window // 2, 0)
    hi = min(i - window // 2 + window, n)
    return lo, hi


def _ref_rms_envelope(x, window):
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    window = max(1, min(int(window), n))
    out = np.empty(n)
    for i in range(n):
        lo, hi = _ref_window(i, n, window)
        vals = x[lo:hi]
        good = vals[np.isfinite(vals)]
        if 2 * good.size >= hi - lo:
            out[i] = np.sqrt(np.sum(good * good) / max(good.size, 1))
        else:
            out[i] = np.nan
    return out


def _ref_detrend(x, fs, window_s=5.0):
    x = np.asarray(x, dtype=float)
    n = x.size
    w = max(1, int(round(window_s * fs)))
    out = np.empty(n)
    for i in range(n):
        lo, hi = _ref_window(i, n, w)
        vals = x[lo:hi]
        good = vals[np.isfinite(vals)]
        mean = good.sum() / good.size if good.size else 0.0
        out[i] = x[i] - mean
    return out


def _signal_with_gaps(n=61, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.4, 1.0, size=n)
    x[0] = np.nan                 # a dropout at the very first sample
    x[7:10] = np.nan              # a short run inside
    x[20:31] = np.nan             # a run longer than the small windows
    x[-1] = np.inf                # and a non-finite last sample
    return x


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5, 8, 13, 61, 200])
def test_rms_envelope_matches_the_per_sample_definition(window):
    x = _signal_with_gaps()
    got = sg.rms_envelope(x, window)
    np.testing.assert_allclose(got, _ref_rms_envelope(x, window), rtol=1e-12, atol=0,
                               equal_nan=True)
    assert got.shape == x.shape


def test_rms_envelope_reads_nan_exactly_when_fewer_than_half_the_window_is_finite():
    # window 4 centred on sample 3 covers samples 1..4; two of four finite is "half"
    x = np.array([1.0, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0])
    env = sg.rms_envelope(x, 4)
    assert np.isfinite(env[3])            # samples 1..4: 2 of 4 finite, exactly half, kept
    assert np.isfinite(env[2])            # samples 0..3: 2 of 4 finite, kept
    assert np.isnan(env[1])               # samples 0..2 at the edge: 1 of 3 finite
    np.testing.assert_allclose(env, _ref_rms_envelope(x, 4), equal_nan=True)


def test_rms_envelope_accepts_a_column_vector():
    x = np.arange(1.0, 11.0)
    np.testing.assert_allclose(sg.rms_envelope(x[:, None], 3), sg.rms_envelope(x, 3))


@pytest.mark.parametrize("window_s", [0.002, 0.004, 0.01, 0.05, 5.0])
def test_detrend_matches_the_per_sample_definition(window_s):
    x = _signal_with_gaps()
    got = sg._detrend(x, FS, window_s)
    np.testing.assert_allclose(got, _ref_detrend(x, FS, window_s), rtol=1e-12, atol=1e-15,
                               equal_nan=True)


def test_detrend_of_an_all_nan_window_leaves_the_samples_as_they_are():
    x = np.array([np.nan, np.nan, np.nan, 3.0])
    got = sg._detrend(x, FS, 0.002)          # one-sample window
    assert np.isnan(got[:3]).all()
    assert got[3] == 0.0


# -- the segment dict ------------------------------------------------------------------

def _data(n, n_emg=2, n_ent=2, seed=0):
    rng = np.random.default_rng(seed)
    timecol = np.arange(n) / FS
    emg = rng.normal(size=(n, n_emg))
    ent = rng.normal(size=(n, n_ent))
    return timecol, emg, ent


def _check_segment(seg, number, start, end, timecol, emg, ent, filename, ignored, kind):
    assert seg["number"] == number
    assert seg["name"] == f"Segment #{number}"
    assert seg["breathcnt"] == number
    assert seg["filename"] == filename
    assert seg["ignored"] is ignored
    assert seg["kind"] == kind
    np.testing.assert_array_equal(seg["time"], timecol[start:end])
    np.testing.assert_array_equal(seg["emgcols"], emg[start:end, :])
    np.testing.assert_array_equal(seg["entcols"], ent[start:end, :])


def test_slice_cols_handles_an_absent_a_one_d_and_a_two_d_column_set():
    assert sg._slice_cols([], 2, 5) == []
    one = np.arange(10.0)
    np.testing.assert_array_equal(sg._slice_cols(one, 2, 5), one[2:5])
    two = np.arange(20.0).reshape(10, 2)
    np.testing.assert_array_equal(sg._slice_cols(two, 2, 5), two[2:5, :])
    single = np.array([[7.0, 8.0]])
    np.testing.assert_array_equal(sg._slice_cols(single, 0, 1), single)


def test_whole_file_segment_carries_every_field_and_its_own_lookups():
    timecol, emg, ent = _data(3 * FS)
    seg = sg.whole_file("rec.csv", timecol, emg, ent, rms_s=0.05, fs=FS,
                        ignored_breaths={1}, kinds={1: "rest", 2: "sniff"})[1]
    _check_segment(seg, 1, 0, 3 * FS, timecol, emg, ent, "rec.csv", True, "rest")
    seg = sg.whole_file("rec.csv", timecol, emg, ent, rms_s=0.05, fs=FS,
                        ignored_breaths={2}, kinds={2: "sniff"})[1]
    assert seg["ignored"] is False
    assert seg["kind"] is None


def test_whole_file_rms_diagnostics_give_nan_for_an_absent_and_an_all_nan_channel():
    n = 2 * FS
    timecol, emg, _ = _data(n, n_emg=3)
    emg[:, 1] = np.nan
    seg = sg.whole_file("rec.csv", timecol, emg, [], rms_s=0.05, fs=FS,
                        ignored_breaths=set(), kinds={})[1]
    for key in ("rms_file_max", "t_rms_file_max", "rms_file_top3"):
        values = seg[key]
        assert len(values) == 3
        assert np.isfinite(values[0]) and np.isnan(values[1]) and np.isfinite(values[2])


def test_fixed_windows_numbers_slices_and_looks_up_every_window():
    n = 10
    timecol, emg, ent = _data(n)
    segs = sg.fixed_windows("w.csv", timecol, emg, ent, FS, window_s=4 / FS, hop_s=1 / FS,
                            ignored_breaths={3, 5}, kinds={2: "rest", 6: "other"})
    assert list(segs) == [1, 2, 3, 4, 5, 6, 7]       # n - win + 1 starts, all complete
    for number, seg in segs.items():
        start = number - 1
        _check_segment(seg, number, start, start + 4, timecol, emg, ent, "w.csv",
                       number in {3, 5}, {2: "rest", 6: "other"}.get(number))


def test_fixed_windows_with_a_window_exactly_the_recording_gives_one_segment():
    timecol, emg, ent = _data(FS)
    segs = sg.fixed_windows("w.csv", timecol, emg, ent, FS, window_s=1.0, hop_s=0.5,
                            ignored_breaths=set(), kinds={})
    assert list(segs) == [1]
    assert len(segs[1]["time"]) == FS


def test_fixed_windows_shorter_recording_names_its_duration():
    timecol, emg, ent = _data(FS // 2)
    with pytest.raises(EmgSegmentationError,
                       match=r"w\.csv is 0\.50 s long, shorter than one 1 s window "
                             r"\(processing\.segmentation\.emg\.window_s\)\."):
        sg.fixed_windows("w.csv", timecol, emg, ent, FS, window_s=1.0, hop_s=0.5,
                         ignored_breaths=set(), kinds={})


def test_fixed_windows_with_one_sample_windows():
    timecol, emg, ent = _data(3)
    segs = sg.fixed_windows("w.csv", timecol, emg, ent, FS, window_s=0.0001, hop_s=0.0001,
                            ignored_breaths=set(), kinds={})
    assert list(segs) == [1, 2, 3]
    assert [len(s["time"]) for s in segs.values()] == [1, 1, 1]


def test_separators_slice_and_look_up_every_segment():
    n = 4 * FS
    timecol, emg, ent = _data(n)
    segs = sg.separators("s.csv", [1.0, 2.5], timecol, emg, ent, FS,
                         ignored_breaths={2}, kinds={3: "rest"})
    bounds = [0, FS, int(2.5 * FS), n]
    for number in (1, 2, 3):
        _check_segment(segs[number], number, bounds[number - 1], bounds[number], timecol,
                       emg, ent, "s.csv", number == 2, {3: "rest"}.get(number))


def test_a_separator_on_the_last_sample_boundary_is_rejected_with_the_duration():
    n = 2 * FS
    timecol, emg, ent = _data(n)
    with pytest.raises(EmgSegmentationError,
                       match=r"A separator at 2 s in s\.csv does not fall strictly inside "
                             r"the recording \(2\.00 s long\) after the previous boundary"
                             r" .* check Preview & QC ▸ EMG – segments\.$"):
        sg.separators("s.csv", [2.0], timecol, emg, ent, FS, ignored_breaths=set(), kinds={})
    # one sample earlier is still inside
    segs = sg.separators("s.csv", [(n - 1) / FS], timecol, emg, ent, FS,
                         ignored_breaths=set(), kinds={})
    assert [len(s["time"]) for s in segs.values()] == [n - 1, 1]


@pytest.mark.parametrize("old,new,number,expected", [
    ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 3.0], 3, 2),
    ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 3.0], 1, 1),
])
def test_remap_segment_number_stays_inside_the_new_segments(old, new, number, expected):
    assert sg.remap_segment_number(old, new, number) == expected


# -- burst detection -------------------------------------------------------------------

def _burst_train(n_bursts=6, period_s=3.0, burst_s=1.0, seed=3, ramp=False):
    rng = np.random.default_rng(seed)
    n = int(n_bursts * period_s * FS) + FS
    t = np.arange(n) / FS
    amp = np.full(n, 0.05)
    for k in range(n_bursts):
        on = 0.7 + k * period_s + 0.13 * k
        idx = (t >= on) & (t < on + burst_s)
        amp[idx] = 0.05 + (t[idx] - on) / burst_s if ramp else 1.0
    x = amp * np.sin(2 * np.pi * 80 * t) + 0.3 + 0.02 * t
    return x + 0.01 * rng.normal(size=n)


def _channels():
    x = _burst_train()
    noise = np.random.default_rng(9).normal(0, 0.2, size=x.size)
    return np.column_stack([np.full(x.size, np.nan), noise, x, _burst_train(seed=4, ramp=True)])


def test_detect_bursts_pinned_on_a_dropout_a_noise_and_two_burst_channels():
    det = sg.detect_bursts(_channels(), FS, threshold_frac=0.3, min_s=0.2, smooth_s=0.1,
                           min_contrast=3.0)
    assert det.onsets.tolist() == [362, 1925, 3490, 5055, 6621, 8186]
    assert det.offsets.tolist() == [862, 2427, 3990, 5556, 7121, 8687]
    assert det.coarse_onsets.tolist() == [331, 1896, 3461, 5025, 6590, 8156]
    assert det.coarse_offsets.tolist() == [874, 2440, 4003, 5568, 7134, 8699]
    assert det.n_channels_used == 2
    assert det.contrast == pytest.approx(18.2622320898493, rel=1e-9)
    for arr in (det.onsets, det.offsets, det.coarse_onsets, det.coarse_offsets):
        assert arr.dtype.kind == "i"


def test_detect_bursts_on_one_channel_alone_is_the_same_as_with_the_dropped_ones():
    cols = _channels()
    alone = sg.detect_bursts(cols[:, 2:], FS, threshold_frac=0.3, min_s=0.2, smooth_s=0.1,
                             min_contrast=3.0)
    full = sg.detect_bursts(cols, FS, threshold_frac=0.3, min_s=0.2, smooth_s=0.1,
                            min_contrast=3.0)
    np.testing.assert_array_equal(alone.onsets, full.onsets)
    np.testing.assert_array_equal(alone.offsets, full.offsets)


def test_no_contrast_names_the_file():
    noise = np.random.default_rng(2).normal(size=(4 * FS, 2))
    with pytest.raises(EmgSegmentationError, match=r"^No EMG bursts found in n\.csv: the "
                       r"envelope of no channel rises to 3 times its resting level"):
        sg.detect_bursts(noise, FS, threshold_frac=0.3, min_s=0.2, smooth_s=0.1,
                         min_contrast=3.0, filename="n.csv")
    with pytest.raises(EmgSegmentationError, match=r"^No EMG bursts found: the envelope"):
        sg.detect_bursts(noise, FS, threshold_frac=0.3, min_s=0.2, smooth_s=0.1,
                         min_contrast=3.0)


def test_too_short_bursts_name_the_minimum_duration():
    x = _burst_train(burst_s=0.4)
    with pytest.raises(EmgSegmentationError,
                       match=r"^No EMG bursts found in b\.csv: activity above the threshold "
                             r"never lasts 0\.5 s \(processing\.segmentation\.emg\."
                             r"burst_min_s\)\.$"):
        sg.detect_bursts(x, FS, threshold_frac=0.3, min_s=0.5, smooth_s=0.05,
                         min_contrast=3.0, filename="b.csv")


def test_emg_burst_segments_carry_the_file_and_the_lookups():
    x = _burst_train()
    n = x.size
    timecol = np.arange(n) / FS
    emg = np.column_stack([x, x])
    ent = np.column_stack([x * 2.0])
    segs = sg.emg_burst("e.csv", timecol, emg, ent, x, FS, threshold_frac=0.3, min_s=0.2,
                        smooth_s=0.1, min_contrast=3.0, ignored_breaths={2},
                        kinds={4: "rest"})
    det = sg.detect_bursts(x, FS, threshold_frac=0.3, min_s=0.2, smooth_s=0.1,
                           min_contrast=3.0)
    onsets = det.onsets.tolist()
    assert len(onsets) == 6
    assert [s["burst_span"] for s in segs.values()] == list(zip(onsets, det.offsets.tolist()))
    ends = onsets[1:] + [n]
    for number, seg in segs.items():
        _check_segment(seg, number, onsets[number - 1], ends[number - 1], timecol, emg, ent,
                       "e.csv", number == 2, {4: "rest"}.get(number))
        assert seg["emg_seg_qc"]["emg_seg_n_bursts"] == 6.0


def _crowded():
    """Bursts of 0.6 s in pairs 0.15 s apart, a burst from the very first sample, an
    augmenting (ramp) burst and a 50 ms spike, on a negative offset with a slow drift."""
    rng = np.random.default_rng(11)
    n = 14 * FS
    t = np.arange(n) / FS
    amp = np.full(n, 0.04)
    spans = [(0.0, 0.8), (2.0, 2.6), (2.75, 3.35), (5.0, 5.6), (5.7, 6.3), (8.0, 8.05),
             (9.0, 10.2), (12.0, 12.9)]
    for k, (a, b) in enumerate(spans):
        idx = (t >= a) & (t < b)
        amp[idx] = 0.04 + (t[idx] - a) / (b - a) if k == 6 else 1.0
    x = amp * np.sin(2 * np.pi * 70 * t) - 0.2 + 0.03 * np.sin(2 * np.pi * 0.05 * t)
    return x + 0.01 * rng.normal(size=n)


@pytest.mark.parametrize("min_s,onsets,offsets,coarse_on,coarse_off", [
    # the 50 ms spike is exactly min_s long at 0.1 s and is kept; at 0.2 s it is dropped
    (0.1, [0, 1002, 2500, 3988, 4749, 6002], [400, 1677, 3150, 4038, 5117, 6452],
     [0, 978, 2478, 3978, 4589, 5978], [424, 1699, 3174, 4049, 5124, 6474]),
    (0.2, [0, 1002, 2500, 4749, 6002], [400, 1677, 3150, 5117, 6452],
     [0, 978, 2478, 4589, 5978], [424, 1699, 3174, 5124, 6474]),
])
def test_detect_bursts_pinned_on_crowded_bursts(min_s, onsets, offsets, coarse_on, coarse_off):
    det = sg.detect_bursts(_crowded(), FS, threshold_frac=0.3, min_s=min_s, smooth_s=0.1,
                           min_contrast=3.0)
    assert det.onsets.tolist() == onsets
    assert det.offsets.tolist() == offsets
    assert det.coarse_onsets.tolist() == coarse_on
    assert det.coarse_offsets.tolist() == coarse_off
    assert det.contrast == pytest.approx(23.879218848356423, rel=1e-9)


def test_no_contrast_message_in_full():
    noise = np.random.default_rng(2).normal(size=(4 * FS, 1))
    with pytest.raises(EmgSegmentationError) as err:
        sg.detect_bursts(noise, FS, threshold_frac=0.3, min_s=0.2, smooth_s=0.1,
                         min_contrast=3.0, filename="n.csv")
    assert str(err.value) == (
        "No EMG bursts found in n.csv: the envelope of no channel rises to 3 times its "
        "resting level, or that resting level cannot be separated from the bursts, so "
        "there is nothing to segment (lower processing.segmentation.emg.burst_min_contrast "
        "only if the bursts are real, or use whole_file/separators/fixed_windows instead).")

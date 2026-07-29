"""Unit tests for the end-expiratory volume trend correction.

Background — the bug these lock. ``trend_peak_min_height`` was an ABSOLUTE depth (v1
default 0.8) that a trough had to reach below the recording's GLOBAL volume maximum. On
ordinary tidal breathing (range < 0.8) no trough qualified, ``find_peaks`` returned an
empty array, and ``interp1d`` raised numpy's ``cannot reshape array of size 0 into shape
(0,newaxis)`` — a crash that named neither the setting nor the recording. One anchor was
worse still: ``interp1d`` accepted it and returned an all-NaN envelope, so the run
"succeeded" with NaN in every result cell.

The default is now a per-trough PROMINENCE expressed as a fraction of the recording's own
volume range, so it is invariant to tidal volume and to the volume unit. An explicit
``trend_peak_min_height`` still selects the old absolute gate, unchanged, so an older
analysis reproduces exactly (that equivalence is pinned below at ``np.array_equal``).

Arrays are built in memory rather than committed as a CSV: ``tests/golden/make_golden.py``
masks ``synth_case_*.csv``, but a new file under ``tests/golden/input/`` is still input
data the golden builder would see.
"""
import os

import numpy as np
import pytest
import scipy as sp
import scipy.interpolate
from scipy import signal
from types import SimpleNamespace

from respmech.core import compute
from respmech.core.compute import VolumeTrendError, correcttrend, trend_anchors

FS = 100.0


def _breathing(n_breaths=10, vt=0.5, period_s=4.0, fs=FS, eelv_drift=0.0, noise=0.0):
    """A tidal volume trace: raised-cosine breaths on a linear end-expiratory trend.

    Starts and ends at end-expiration, exactly as ``trim`` guarantees for real input.
    """
    per = int(period_s * fs)
    one = (1.0 - np.cos(np.linspace(0, 2 * np.pi, per, endpoint=False))) / 2.0
    v = np.tile(one, n_breaths) * vt
    v = np.concatenate([v, [0.0]])                       # end on an end-expiratory sample
    v += np.linspace(0.0, eelv_drift, v.size)            # the trend to be removed
    if noise:
        v += np.sin(np.arange(v.size) * 1.7) * noise     # deterministic ripple, no RNG
    return v


def _settings(*, height=None, frac=compute.TREND_MIN_PROMINENCE_FRAC, distance=0.4,
              method="linear", fs=FS):
    """The legacy attribute shape the compute layer reads (see core/_legacy_ns)."""
    return SimpleNamespace(
        input=SimpleNamespace(format=SimpleNamespace(samplingfrequency=fs)),
        processing=SimpleNamespace(mechanics=SimpleNamespace(
            volumetrendpeakminheight=height,
            trend_peak_min_prominence_frac=frac,
            volumetrendpeakmindistance=distance,
            volumetrendadjustmethod=method)))


# --- the reported bug ------------------------------------------------------

def test_auto_rule_anchors_every_breath_where_the_legacy_gate_found_none():
    """The reporter's case: tidal volume well under the retired 0.8 L threshold."""
    v = _breathing(n_breaths=10, vt=0.5)
    assert np.ptp(v) < 0.8                                   # the whole excursion
    legacy = trend_anchors(v, FS, min_height=0.8)
    assert legacy.size == 0                                  # what used to crash
    auto = trend_anchors(v, FS)
    assert auto.size == 11                                   # one per breath, + both ends


def test_zero_anchors_raises_an_actionable_error_not_a_numpy_reshape():
    v = _breathing(n_breaths=10, vt=0.5)
    with pytest.raises(VolumeTrendError) as e:
        correcttrend(v, _settings(height=0.8))
    msg = str(e.value)
    assert "reshape" not in msg                       # the old, meaningless message
    assert "trend_peak_min_height" in msg             # names the setting…
    assert "0.8" in msg                               # …its value…
    assert f"{np.ptp(v):.2f}" in msg                  # …and the measured range


@pytest.mark.filterwarnings("ignore::RuntimeWarning")   # the all-NaN interp1d is the point
def test_one_anchor_raises_instead_of_returning_an_all_nan_volume():
    """interp1d ACCEPTS a single point and returns an all-NaN envelope, so before the
    guard this produced a full workbook of NaN with no error anywhere. The most important
    test here: a crash is loud, this was silent."""
    v = _breathing(n_breaths=10, vt=0.5, eelv_drift=1.0)
    height = 1.30                                    # keeps exactly the deepest trough
    assert trend_anchors(v, FS, min_height=height).size == 1
    # what the old code did with that single anchor:
    one = trend_anchors(v, FS, min_height=height)
    envelope = sp.interpolate.interp1d(one, v[one], "linear", fill_value="extrapolate")(
        np.linspace(0, v.size - 1, v.size))
    assert np.isnan(envelope).all()                  # every result cell would be NaN
    with pytest.raises(VolumeTrendError):
        correcttrend(v, _settings(height=height))


# --- the auto rule is scale-free -------------------------------------------

@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_auto_rule_is_invariant_to_the_volume_unit(scale):
    v = _breathing(n_breaths=8, vt=0.6)
    assert np.array_equal(trend_anchors(v, FS), trend_anchors(v * scale, FS))


@pytest.mark.parametrize("vt", [0.05, 0.5, 3.0])
def test_auto_rule_is_invariant_to_tidal_volume(vt):
    base = trend_anchors(_breathing(n_breaths=8, vt=1.0), FS)
    assert np.array_equal(trend_anchors(_breathing(n_breaths=8, vt=vt), FS), base)


def test_auto_rule_survives_a_rising_end_expiratory_trend():
    """The case the feature exists for — and the one a threshold measured from the
    file's global maximum fails, because a rising baseline lifts late troughs out of
    range exactly when the trend is largest."""
    v = _breathing(n_breaths=10, vt=0.5, eelv_drift=1.5)
    assert trend_anchors(v, FS).size == 11
    corrected = correcttrend(v, _settings())
    # every breath now ends at ~0 instead of climbing to +1.5
    ends = corrected[::int(4.0 * FS)]
    assert np.max(np.abs(ends)) < 0.02


@pytest.mark.parametrize("frac", [0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
def test_the_prominence_fraction_is_a_noise_gate_not_a_tuning_knob(frac):
    """Measured on 13 real recordings: the anchor set does not move anywhere in this
    range. A default that needed tuning per recording would be the old bug again."""
    v = _breathing(n_breaths=10, vt=0.5, eelv_drift=0.8, noise=0.002)
    assert np.array_equal(trend_anchors(v, FS, min_prominence_frac=frac),
                          trend_anchors(v, FS, min_prominence_frac=0.05))


def test_auto_rule_ignores_a_shallow_plateau_inflection():
    """A near-zero-prominence wiggle is not an end-expiratory trough; anchoring on one
    drags the envelope through the middle of a breath."""
    v = _breathing(n_breaths=6, vt=1.0)
    v[int(2.0 * FS)] += 0.0005                    # a dimple at peak inspiration
    assert int(2.0 * FS) not in set(trend_anchors(v, FS).tolist())


def test_both_ends_anchor_when_the_recording_starts_and_ends_at_end_expiration():
    """An edge is never a local maximum, so find_peaks can never return one; without
    testing them separately every recording loses its two outermost anchors and the
    envelope extrapolates past them (which is what leaves a NaN head/tail under
    'previous'/'next')."""
    v = _breathing(n_breaths=6, vt=0.5, eelv_drift=0.5)
    a = trend_anchors(v, FS)
    assert a[0] == 0 and a[-1] == v.size - 1


def _cut(v, frac, period_s=4.0, fs=FS):
    return v[:-int(frac * period_s * fs)]


def test_an_end_left_mid_breath_is_not_anchored():
    """trim() ends the window at the last sample with flow >= 0 — "in expiration", not
    "at end-expiration". A recording stopped mid-exhalation has its final sample part way
    down a breath; anchoring there pins that sample to zero and invents a trend."""
    v = _cut(_breathing(n_breaths=8, vt=0.6), 0.5)     # cut at peak inspiration
    a = trend_anchors(v, FS)
    assert a[-1] != v.size - 1
    # and the last sample keeps its real level instead of being dragged to 0
    assert abs(correcttrend(v, _settings())[-1]) > 0.1 * 0.6


def test_a_start_mid_inspiration_is_not_anchored():
    """`startix = argmax(flow < 0)` returns 0 for a file that already begins mid-breath,
    so index 0 is not end-expiratory by construction either."""
    v = _breathing(n_breaths=8, vt=0.6)[int(0.25 * 4.0 * FS):]
    assert trend_anchors(v, FS)[0] != 0


def test_a_trending_recording_still_anchors_its_ends():
    """The end test compares against the level the interior troughs PREDICT there, not a
    fixed level — end-expiratory volume is exactly what trends, so a start well above a
    later trough is still a good anchor. Measured: a stricter rule rejects legitimate ends
    on RIU_H5_IC and RIU_H6_60W and doubles the residual end-expiratory error."""
    for drift in (1.5, -0.6):
        v = _breathing(n_breaths=10, vt=0.5, eelv_drift=drift)
        a = trend_anchors(v, FS)
        assert a[0] == 0 and a[-1] == v.size - 1, f"drift={drift}"


def test_no_detectable_trough_is_refused_rather_than_silently_doing_nothing():
    """With both ends accepted unconditionally a recording without a single breath-sized
    trough still fitted a 2-point line, so the run 'corrected the trend', subtracted
    almost nothing, and the run report said it had."""
    v = np.linspace(0.0, 1.0, 2000)                    # a ramp: no troughs at all
    assert trend_anchors(v, FS).size == 0
    with pytest.raises(VolumeTrendError):
        correcttrend(v, _settings())


@pytest.mark.parametrize("method", ["linear", "nearest", "previous", "next", "slinear"])
def test_no_interpolation_kind_leaves_undefined_samples_under_the_auto_rule(method):
    """'previous'/'next' cannot cover anything outside the outermost anchor; anchoring
    both ends is what makes every offered kind safe."""
    v = _breathing(n_breaths=8, vt=0.5, eelv_drift=0.4)
    assert np.isfinite(correcttrend(v, _settings(method=method))).all()


# --- the legacy gate is untouched ------------------------------------------

def test_explicit_height_is_bit_identical_to_the_pre_change_formula():
    """Unit-level proof of the golden argument: an analysis that pins a threshold takes
    the old code path exactly. Deliberately array_equal, not allclose."""
    v = _breathing(n_breaths=8, vt=2.0, eelv_drift=0.3)
    for height in (0.5, 0.8, 1.2):
        old = signal.find_peaks((v * -1) + max(v), height=height, distance=0.4 * FS)[0]
        assert np.array_equal(trend_anchors(v, FS, min_height=height), old)
        f = sp.interpolate.interp1d(old, v[old], "linear", fill_value="extrapolate")
        expected = v - f(np.linspace(0, v.size - 1, v.size))
        assert np.array_equal(correcttrend(v, _settings(height=height)), expected)


def test_explicit_height_does_not_gain_the_boundary_anchors():
    """The end anchors belong to the new rule only — adding them to the legacy gate
    would silently change every existing analysis."""
    v = _breathing(n_breaths=8, vt=2.0)
    assert trend_anchors(v, FS, min_height=0.5)[0] != 0


# --- degenerate input ------------------------------------------------------

@pytest.mark.parametrize("method,need", [("quadratic", 3), ("cubic", 4)])
def test_higher_order_kinds_name_the_shortfall_not_scipy_derivatives(method, need):
    """scipy says 'The number of derivatives at boundaries does not match: expected 2,
    got 0+0', which names neither the recording nor the setting."""
    v = _breathing(n_breaths=1, vt=0.5, period_s=2.0)
    with pytest.raises(VolumeTrendError) as e:
        correcttrend(v, _settings(method=method))
    assert method in str(e.value) and str(need) in str(e.value)


def test_non_finite_volume_is_reported_as_a_precondition_failure():
    v = _breathing(n_breaths=6, vt=0.5)
    v[100] = np.nan
    with pytest.raises(VolumeTrendError) as e:
        correcttrend(v, _settings())
    assert "missing or infinite" in str(e.value)


def test_a_flat_volume_signal_fails_cleanly():
    with pytest.raises(VolumeTrendError):
        correcttrend(np.zeros(500), _settings())


def test_sub_sample_trough_spacing_does_not_reach_scipy():
    """find_peaks rejects distance < 1 with a message naming none of our settings."""
    v = _breathing(n_breaths=6, vt=0.5)
    assert trend_anchors(v, FS, min_distance_s=1e-9).size > 0


# --- one detector, two consumers -------------------------------------------

def test_the_diagnostic_figure_uses_the_same_anchors_as_the_computation(monkeypatch, tmp_path):
    """The plots layer used to re-implement the detection, so the figure could mark
    anchors that were never subtracted — or be skipped for a run that computed fine."""
    from respmech.core import plots
    seen = {}
    real = compute.trend_anchors

    def spy(vol, fs, **kw):
        out = real(vol, fs, **kw)
        seen["n"] = out.size
        return out

    monkeypatch.setattr(compute, "trend_anchors", spy)
    v = _breathing(n_breaths=8, vt=0.5, eelv_drift=0.5)
    fr = SimpleNamespace(signals={"trend_on": True, "vol_drift": v, "fs": FS,
                                  "time": np.arange(v.size) / FS})
    from respmech.core.settings import Settings
    s = Settings()
    s.processing.volume.correct_trend = True
    out = plots._trend(fr, "f.txt", str(tmp_path / "volume trend.pdf"), s)
    assert out is not None                            # the figure was actually produced
    assert seen["n"] == trend_anchors(v, FS).size


# --- the reported bug, end to end ------------------------------------------

def _write_recording(path, n_breaths=10, vt=0.6, period_s=4.0, fs=200):
    """A minimal LabChart-shaped CSV: whole breaths, tidal volume BELOW the retired
    0.8 threshold — i.e. the shape of recording that crashed the run."""
    per = int(period_s * fs)
    t = np.arange(n_breaths * per + 1) / fs
    amp = vt * np.pi / period_s                       # so ∫|flow| over an inspiration = vt
    flow = -amp * np.sin(2 * np.pi * t / period_s)    # inspiration first (negative)
    poes = -5.0 * np.sin(2 * np.pi * t / period_s) - 5.0
    pgas = 2.0 * np.sin(2 * np.pi * t / period_s) + 8.0
    cols = np.column_stack([t, poes * 0, poes * 0, poes * 0, flow, flow * 0,
                            poes, pgas, pgas - poes])
    np.savetxt(path, cols, delimiter=",")
    return vt


def _batch_settings(folder, out, *, height=None, fs=200):
    from respmech.core.settings import Settings
    s = Settings()
    s.input.folder, s.input.files = str(folder), "*.csv"
    s.input.format.sampling_frequency = fs
    c = s.input.channels
    c.flow, c.volume, c.poes, c.pgas, c.pdi = 5, 6, 7, 8, 9
    v = s.processing.volume
    v.integrate_from_flow = True          # the reporter's 'Calculate volume from flow'
    v.correct_drift = True
    v.correct_trend = True                # …and 'Correct end-expiratory trend'
    v.trend_peak_min_height = height
    s.output.folder = str(out)
    return s.validate()


def test_batch_run_completes_on_a_low_excursion_recording(tmp_path):
    """The reported scenario at the level it was reported: a whole batch run over a
    recording whose entire volume range is under the retired threshold."""
    from respmech.core import pipeline
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(); out.mkdir()
    vt = _write_recording(src / "rec.csv")
    result = pipeline.run_batch(_batch_settings(src, out))
    assert list(result.failed_files) == []
    fr = result.ok_files["rec.csv"]
    assert len(fr.breaths_table) >= 8
    for col in ("vt", "ve", "wobtotal"):
        assert np.isfinite(fr.breaths_table[col].to_numpy(float)).all()
    assert np.ptp(fr.signals["vol_final"]) < 0.8      # still the low-excursion case
    assert abs(float(np.median(fr.breaths_table["vt"])) - vt) < 0.05


def test_batch_run_on_the_same_recording_fails_actionably_with_the_legacy_threshold(tmp_path):
    from respmech.core import pipeline
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(); out.mkdir()
    _write_recording(src / "rec.csv")
    result = pipeline.run_batch(_batch_settings(src, out, height=0.8))
    fr = result.failed_files["rec.csv"]
    assert fr.error_kind == "VolumeTrendError"        # a precondition failure, not a fault
    assert "reshape" not in fr.error                  # the old numpy message is gone
    assert "trend_peak_min_height" in fr.error


def test_the_trend_figure_is_written_for_a_low_excursion_recording(tmp_path):
    """The diagnostic figure used to be silently skipped for exactly the recordings
    whose correction failed — no picture of the thing that went wrong."""
    from respmech.core import pipeline
    from respmech.core.io import writers
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(); out.mkdir()
    _write_recording(src / "rec.csv")
    s = _batch_settings(src, out)
    s.output.diagnostics.save_drift = True
    written = writers.write_batch(pipeline.run_batch(s), s, str(out))
    assert any("trend" in os.path.basename(p) for p in written)


def test_the_mechanics_preview_lands_softly_when_the_trend_cannot_be_fitted(tmp_path):
    """The preview showed a hard 'Channel preview failed' card. It must now keep the
    channels readable — on the DRIFT-corrected volume — and explain itself."""
    from respmech.ui import workers
    src = tmp_path / "in"
    src.mkdir()
    _write_recording(src / "rec.csv")
    ok = workers.stage_mechanics_preview(
        _batch_settings(src, tmp_path / "out"), str(src / "rec.csv"))
    assert ok["trend_error"] is None and np.isfinite(ok["series"]["volume"]).all()

    soft = workers.stage_mechanics_preview(
        _batch_settings(src, tmp_path / "out", height=0.8), str(src / "rec.csv"))
    assert soft["trend_error"] and "trend_peak_min_height" in soft["trend_error"]
    assert np.isfinite(soft["series"]["volume"]).all()   # drift-corrected, still readable
    assert soft["nbreaths"] > 0                          # channels + breaths still drawn

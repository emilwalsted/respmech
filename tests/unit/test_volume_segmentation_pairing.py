"""Volume-based breath segmentation must not crash with a bare IndexError when it
cannot pair every inspiratory peak with an expiratory one (13-09-2026).

A real ~33 s recording with slow breathing and pauses at zero flow between phases
found more inspiratory than the expected number of expiratory peaks and crashed
``separateintobreathsbyvolume`` with ``IndexError: index 6 is out of bounds for axis
0 with size 6``. Rather than hand-tuning peak-detection thresholds to reproduce that
exact pause shape (a fragile, indirect way to pin the bug), these tests simulate the
mismatch directly by dropping one real expiratory peak scipy's own ``find_peaks``
found on a clean synthetic volume signal -- the precise precondition the function's
index arithmetic depends on (every inspiratory peak but the last needs one
expiratory peak between it and the next).
"""
import numpy as np
import pytest
import scipy.signal as _signal

from respmech.core import compute
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.settings import Settings

FS = 100


def _volume_recording(n_breaths=4, period_s=4.0, vt=0.6, fs=FS):
    """A clean sinusoidal-volume recording with exactly ``n_breaths`` inspiratory
    peaks and ``n_breaths - 1`` expiratory peaks under the default peak settings."""
    n = int(n_breaths * period_s * fs) + 1
    t = np.arange(n) / fs
    volume = vt * (1 - np.cos(2 * np.pi * t / period_s)) / 2
    return t, volume


def _settings(fs=FS):
    s = Settings()
    s.input.format.sampling_frequency = fs
    return to_legacy_ns(s)


def _drop_one_expiratory_peak(monkeypatch, invol):
    """Patch scipy's find_peaks so the call on anything OTHER than ``invol`` itself
    (i.e. separateintobreathsbyvolume's expeaks call, on its derived exvol array)
    returns one fewer peak than it really found -- simulating an expiratory peak
    suppressed below threshold by a zero-flow pause, without needing to hand-tune a
    waveform to reproduce that shape exactly. Identifying the expeaks call by object
    identity (rather than "the second call") keeps this robust to an unrelated
    future find_peaks call (e.g. a trend-anchor probe) being added earlier in the
    same code path -- it would not match ``invol`` and so would not be mistaken for
    the inpeaks call either."""
    real_find_peaks = _signal.find_peaks

    def fake_find_peaks(x, **kw):
        peaks, props = real_find_peaks(x, **kw)
        if x is not invol:
            peaks = peaks[:-1]
        return peaks, props

    monkeypatch.setattr(_signal, "find_peaks", fake_find_peaks)


def test_sanity_clean_recording_pairs_every_breath():
    """Baseline: on an unmodified recording (real find_peaks), every inspiratory peak
    pairs with an expiratory one and no error is raised."""
    t, volume = _volume_recording(n_breaths=4)
    zeros = np.zeros(len(volume))
    breaths = compute.separateintobreathsbyvolume(
        "rec.csv", t, zeros, volume, zeros, zeros, zeros, [], [], _settings())
    assert len(breaths) == 4


def test_too_few_expiratory_peaks_raises_a_named_error_not_indexerror(monkeypatch):
    t, volume = _volume_recording(n_breaths=4)
    zeros = np.zeros(len(volume))
    _drop_one_expiratory_peak(monkeypatch, volume)
    with pytest.raises(compute.VolumeSegmentationError) as exc_info:
        compute.separateintobreathsbyvolume(
            "rec.csv", t, zeros, volume, zeros, zeros, zeros, [], [], _settings())
    msg = str(exc_info.value)
    assert "rec.csv" in msg
    assert "inspiratory" in msg.lower()
    assert "expiratory" in msg.lower()


def test_batch_reports_a_named_error_kind_not_indexerror(tmp_path, monkeypatch):
    """End to end through run_batch (settings.processing.segmentation.method =
    'volume'): the per-file error surfaces with error_kind naming the real problem,
    never 'IndexError'."""
    from respmech.core.pipeline import run_batch

    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir(); out.mkdir()
    t, volume = _volume_recording(n_breaths=4)
    zeros = np.zeros(len(volume))
    poes = -5.0 + zeros
    pgas = 8.0 + zeros
    np.savetxt(src / "rec.csv", np.column_stack([t, zeros, zeros, zeros, zeros, volume, poes, pgas, pgas - poes]),
              delimiter=",")

    s = Settings()
    s.input.folder, s.input.files = str(src), "*.csv"
    s.input.format.sampling_frequency = FS
    c = s.input.channels
    c.flow, c.volume, c.poes, c.pgas, c.pdi = 5, 6, 7, 8, 9
    s.output.folder = str(out)
    s.processing.segmentation.method = "volume"
    s = s.validate()

    # Through run_batch the pipeline reloads and re-derives its own volume array, so
    # the identity-based patch above (which needs a reference to that exact array)
    # isn't available here; fall back to call order, which is safe for this specific
    # settings combination (segmentation.method="volume", correct_trend=False, no
    # EMG/ECG processing) where separateintobreathsbyvolume's inpeaks/expeaks calls
    # are the only two find_peaks calls in the whole run.
    real_find_peaks = _signal.find_peaks
    calls = {"n": 0}

    def fake_find_peaks(x, **kw):
        calls["n"] += 1
        peaks, props = real_find_peaks(x, **kw)
        if calls["n"] == 2:
            peaks = peaks[:-1]
        return peaks, props

    monkeypatch.setattr(_signal, "find_peaks", fake_find_peaks)
    result = run_batch(s)
    fr = result.failed_files["rec.csv"]
    assert fr.error_kind == "VolumeSegmentationError"
    assert "IndexError" not in fr.error

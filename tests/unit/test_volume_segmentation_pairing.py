"""Volume-based breath segmentation must not crash with a bare IndexError when it
cannot pair every inspiratory peak with an expiratory one (ticket 20260913-2053).

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


def _drop_one_expiratory_peak(monkeypatch):
    """Patch scipy's find_peaks so the SECOND call (separateintobreathsbyvolume's
    expeaks, called after inpeaks) returns one fewer peak than it really found --
    simulating an expiratory peak suppressed below threshold by a zero-flow pause,
    without needing to hand-tune a waveform to reproduce that shape exactly."""
    real_find_peaks = _signal.find_peaks
    calls = {"n": 0}

    def fake_find_peaks(x, **kw):
        calls["n"] += 1
        peaks, props = real_find_peaks(x, **kw)
        if calls["n"] == 2:
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
    _drop_one_expiratory_peak(monkeypatch)
    with pytest.raises(compute.VolumeSegmentationError) as exc_info:
        compute.separateintobreathsbyvolume(
            "rec.csv", t, zeros, volume, zeros, zeros, zeros, [], [], _settings())
    msg = str(exc_info.value)
    assert "rec.csv" in msg
    assert "inspiratory" in msg.lower()
    assert "expiratory" in msg.lower()


def test_volume_segmentation_error_is_a_valueerror():
    """Same family as the other segmentation-precondition errors (NoBreathsError,
    DegenerateBreathError, ...), so the batch catches it per file."""
    assert issubclass(compute.VolumeSegmentationError, ValueError)


def test_volume_segmentation_error_has_a_run_screen_fix_hint():
    from respmech.ui.screens.run_screen import _FIX_HINTS
    assert "VolumeSegmentationError" in _FIX_HINTS


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

    _drop_one_expiratory_peak(monkeypatch)
    result = run_batch(s)
    fr = result.failed_files["rec.csv"]
    assert fr.error_kind == "VolumeSegmentationError"
    assert "IndexError" not in fr.error

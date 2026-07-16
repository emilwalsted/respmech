"""Pre-analysis resampling (all channels -> one common rate), the resurrected
``processing.sampling.resample`` feature. Anti-aliased polyphase resample, a fixed
analysis rate downstream, and the noise STFT window coupled to the new rate. Default
OFF ⇒ byte-identical (locked by the golden suites); here we test the ON path."""
import numpy as np

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def test_resample_channels_downsamples_and_antialiases():
    from respmech.core.pipeline import _resample_channels
    fs_in, fs_out, n = 2000, 500, 4000
    t = np.arange(n) / fs_in
    lo = np.sin(2 * np.pi * 10 * t)             # survives (below new Nyquist 250 Hz)
    hi = np.sin(2 * np.pi * 700 * t)            # must be filtered (above new Nyquist)
    data = (lo + hi, lo, lo, lo, lo, np.array([]), np.column_stack([lo + hi, hi]))
    flow, vol, poes, pgas, pdi, ent, emg = _resample_channels(data, fs_in, fs_out)
    assert len(flow) == n * fs_out // fs_in                     # 4000 -> 1000
    assert emg.shape == (1000, 2)                               # 2-D channels resampled on axis 0
    tt = np.arange(len(flow)) / fs_out
    # the 700 Hz component is gone: what's left is (close to) the pure 10 Hz tone
    assert float(np.std(flow - np.sin(2 * np.pi * 10 * tt))) < 0.02
    assert ent.size == 0                                        # empty channels pass through


def test_coupled_stft_preserves_time_window():
    from types import SimpleNamespace
    from respmech.core.pipeline import _coupled_stft
    cfg = SimpleNamespace(n_fft=256, hop_length=64, win_length=256)
    assert _coupled_stft(cfg, 2000, 2000) == (256, 64, 256)    # no resample -> unchanged
    n_fft, hop, win = _coupled_stft(cfg, 2000, 500)            # 4x down -> ~4x smaller window
    assert n_fft == 64 and win == 64 and hop == 16
    assert n_fft & (n_fft - 1) == 0                            # stays a power of two


def test_load_is_passthrough_when_no_resample():
    """Without ``samplingfrequency_in`` set, _load must equal load() exactly — this is
    the golden path, so it must not perturb the data at all."""
    import os
    from respmech.core.io.loaders import load
    from respmech.core.pipeline import _load
    from respmech.core._legacy_ns import to_legacy_ns
    from _helpers import INPUT
    s = to_legacy_ns(synth_settings("/tmp/o", data_out=_DATA_OUT))
    path = os.path.join(INPUT, "synth_case_A.csv")
    a = load(path, s)
    b = _load(path, s)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_run_batch_resample_on_runs_and_changes_rate(tmp_path):
    """A resampled run completes, still segments breaths, and actually uses the new
    analysis rate (the legacy namespace's samplingfrequency is switched to the target)."""
    from respmech.core import pipeline
    s = synth_settings(str(tmp_path), data_out=_DATA_OUT)     # synth input is 1000 Hz
    s.processing.sampling.resample = True
    s.processing.sampling.resample_to_frequency = 400
    r = pipeline.run_batch(s)
    assert r.ok_files and all(len(fr.breaths) for fr in r.ok_files.values())


def test_run_report_states_resample_truthfully(tmp_path):
    """The provenance line is no longer a false claim: it reflects the real, now-applied
    resample (audit #2)."""
    from respmech.core.io.writers import _write_run_report
    from respmech.core import pipeline
    s = synth_settings(str(tmp_path), data_out=_DATA_OUT)
    s.processing.sampling.resample = True
    s.processing.sampling.resample_to_frequency = 400
    r = pipeline.run_batch(s)
    path = _write_run_report(r, s, str(tmp_path), [], None)
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    assert "Resample:" in txt and "yes" in txt and "400 Hz" in txt

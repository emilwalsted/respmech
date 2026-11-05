"""ECG removal: suppression metric, per-test-consistent detection, and the
peak-window RMS helper. Uses a small synthetic ECG-contaminated signal (fast)."""
import numpy as np
import pytest

from respmech.core import emg as E


def _ecg_spike(w):
    """A short biphasic spike ~ an R-wave."""
    t = np.linspace(-1, 1, w)
    return np.exp(-(t ** 2) / 0.02) * np.sin(2 * np.pi * 1.5 * t)


def _make_signal(fs=2000, dur=12.0, hr_period=1.0, n_ch=3, seed=0, amp=1.0,
                 invert=False, dc_offset=0.0):
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    emg = rng.normal(0, 0.05, (n, n_ch)) + dc_offset   # baseline EMG-like noise
    w = int(0.06 * fs)
    spike = _ecg_spike(w) * amp * (-1.0 if invert else 1.0)
    peaks = []
    p = int(0.5 * fs)
    while p + w < n:
        for c in range(n_ch):
            emg[p:p + w, c] += spike * (1.0 - 0.15 * c)   # ECG on every channel
        peaks.append(p + w // 2)
        p += int(hr_period * fs)
    return emg, np.array(peaks), fs


def test_peak_window_rms_helper():
    fs = 2000
    x = np.zeros(fs)
    x[100:140] = 1.0
    peaks = np.array([120])
    r = E.peak_window_rms(x, peaks, fs, halfwidth_s=0.04)
    assert r > 0
    assert np.isnan(E.peak_window_rms(x, np.array([], dtype=int), fs))


def _amplitudes(pos, pos_amp, neg, neg_amp, n=12_000):
    """A signal whose value is pos_amp at the positive and -neg_amp at the negative peaks."""
    x = np.zeros(n)
    x[pos] = pos_amp
    x[neg] = -np.asarray(neg_amp, dtype=float)
    return x


def test_choose_polarity(monkeypatch):
    """Direct unit test for the per-channel polarity rule behind remove_ecg's two-polarity
    search (ticket 5.7 / K-191), with the QRS band filter replaced by the identity so the
    amplitudes at the peaks are the values written here. The result is always ONE polarity,
    never a per-beat mix."""
    from respmech.core.emg import _MIN_SHARE, _NEGATIVE_DOMINANCE, _choose_polarity
    monkeypatch.setattr(E, "_qrs_band", lambda xr, fs: np.asarray(xr, dtype=float))
    fs = 2000.0

    def choose(pos, neg, x):
        return list(_choose_polarity(pos, neg, x, fs))

    pos, neg = np.array([100, 1100, 2100]), np.array([130, 1130, 2130])
    empty = np.array([], dtype=int)
    # empty polarities: whichever one found something
    assert choose(pos, empty, _amplitudes(pos, 1.0, empty, [])) == list(pos)
    assert choose(empty, neg, _amplitudes(empty, [], neg, 1.0)) == list(neg)
    # comparable amplitudes (the production recordings measured 0.71-1.35 in the QRS band):
    # positive, every beat, even where an individual negative peak is the larger one
    assert choose(pos, neg, _amplitudes(pos, [1.0, 0.8, 1.0], neg, [0.9, 1.2, 1.1])) == list(pos)
    # the threshold is twice the positive amplitude: just under it still positive, at it
    # negative (an inverted R-wave)
    assert _NEGATIVE_DOMINANCE == 2.0
    assert choose(pos, neg, _amplitudes(pos, 1.0, neg, 1.98)) == list(pos)
    assert choose(pos, neg, _amplitudes(pos, 1.0, neg, 2.0)) == list(neg)
    # the median decides, not the mean or an extreme: one huge or one tiny negative peak
    # among the others does not move it
    many_p = np.arange(100, 100 + 1000 * 10, 1000)
    many_n = many_p + 30
    assert choose(many_p, many_n, _amplitudes(many_p, 1.0, many_n, [1.0] * 9 + [30.0])) == list(many_p)
    assert choose(many_p, many_n, _amplitudes(many_p, 1.0, many_n, [2.5] * 9 + [0.1])) == list(many_n)
    p5, n5 = many_p[:5], many_n[:5]
    assert choose(p5, n5, _amplitudes(p5, 1.0, n5, [1.0, 1.5, 2.0, 2.5, 3.0])) == list(n5)
    assert choose(p5, n5, _amplitudes(p5, 1.0, n5, [1.0, 1.5, 1.9, 2.5, 3.0])) == list(p5)
    # positive peaks with a negative QRS-band value (a pacing spike before a broad QRS): the
    # negative peaks need twice its size, not just a positive value of their own
    assert choose(pos, neg, _amplitudes(pos, -0.1, neg, 0.25)) == list(neg)
    assert choose(pos, neg, _amplitudes(pos, -0.1, neg, 0.15)) == list(pos)
    assert choose(pos, neg, _amplitudes(pos, -0.1, neg, 0.0)) == list(pos)
    assert choose(pos, neg, _amplitudes(pos, -0.1, neg, -0.2)) == list(pos)
    assert choose(pos, neg, _amplitudes(pos, 0.0, neg, 0.01)) == list(neg)
    assert choose(pos, neg, _amplitudes(pos, 0.0, neg, 0.0)) == list(pos)
    # the count guard, both ways: a polarity with at most _MIN_SHARE as many peaks as the
    # other never wins, whatever its amplitude (one artefact, a few ectopic beats)
    half = many_p[::2] + 500
    assert half.size == _MIN_SHARE * many_p.size
    assert choose(many_p, half, _amplitudes(many_p, 1.0, half, 5.0)) == list(many_p)
    assert choose(half, many_n, _amplitudes(half, 5.0, many_n, 1.0)) == list(many_n)
    # one peak more than that, and the amplitude decides again
    more = np.append(half, 9950)
    assert choose(many_p, more, _amplitudes(many_p, 1.0, more, 5.0)) == list(more)
    assert choose(more, many_n, _amplitudes(more, 1.0, many_n, 5.0)) == list(many_n)


def test_qrs_band_keeps_the_qrs_band_and_drops_slow_waves():
    """The amplitude comparison runs in 8-30 Hz: a 15 Hz wave passes, a 1 Hz wave (T-wave or
    drift speed) does not. Too low a sampling rate for the band (66 Hz or less), or too short
    a channel to filter, leaves the signal as it is, and a non-finite sample counts as 0."""
    fs = 2000.0
    t = np.arange(int(4 * fs)) / fs
    slow, fast = np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 15.0 * t)
    middle = slice(int(fs), int(3 * fs))
    assert np.max(np.abs(E._qrs_band(slow, fs)[middle])) < 0.01
    assert np.max(np.abs(E._qrs_band(fast, fs)[middle])) > 0.9
    for f, low, high in ((5.0, 0.0, 0.1), (10.0, 0.8, 1.0), (25.0, 0.75, 1.0), (60.0, 0.0, 0.05)):
        gain = np.max(np.abs(E._qrs_band(np.sin(2 * np.pi * f * t), fs)[middle]))
        assert low <= gain <= high, (f, gain)
    s = np.sin(2 * np.pi * np.arange(400) / 67.0)
    assert not np.array_equal(E._qrs_band(s, 67.0), s)
    assert np.array_equal(E._qrs_band(s, 66.0), s)
    assert np.array_equal(E._qrs_band(slow[:15], fs), slow[:15])
    assert not np.array_equal(E._qrs_band(slow[:16], fs), slow[:16])
    spiked = fast.copy()
    spiked[1000], spiked[5000], spiked[6500] = np.inf, -np.inf, np.nan
    band = E._qrs_band(spiked, fs)
    assert np.all(np.isfinite(band))
    assert np.array_equal(band, E._qrs_band(np.where(np.isfinite(spiked), spiked, 0.0), fs))
    assert np.max(np.abs(band[6800:7600] - E._qrs_band(fast, fs)[6800:7600])) < 0.05


def test_detect_r_peaks_uses_seconds_and_the_sampling_rate(monkeypatch):
    """detect_r_peaks takes the minimum width and distance in seconds and hands the QRS band
    the channel's own sampling rate: a minimum width of 4 ms keeps one-sample spikes between
    the beats out, and the band filter sees fs, not a fraction of it."""
    fs, w = 2000, 120
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.02, int(12 * fs))
    t = np.arange(w)
    r_idx = []
    for k in range(11):
        p = int((0.5 + k) * fs)
        x[p:p + w] += np.exp(-((t - w // 3) ** 2) / (2 * 36.0))
        r_idx.append(p + w // 3)
        if k < 10:
            x[p + w // 3 + fs // 2] += 1.5                   # a one-sample spike between beats
    peaks = E.detect_r_peaks(x, fs, 0.3, 0.3, 0.004)
    assert len(peaks) == 11 and np.all(np.abs(peaks - np.array(r_idx)) <= 4)

    seen = []
    band = E._qrs_band
    monkeypatch.setattr(E, "_qrs_band", lambda xr, f: (seen.append(f), band(xr, f))[1])
    y = rng.normal(0, 0.02, int(6 * fs))
    for k in range(5):                                      # biphasic: both searches find every beat
        p = int((0.5 + k) * fs)
        y[p:p + w] += _qrs(w, 1.0, 0.9)
    E.detect_r_peaks(y, fs, 0.3, 0.5, 0.001)
    assert seen == [fs]


def _qrs(w, r_amp, s_amp):
    """An R lobe followed 20 ms later (at 2 kHz) by an S lobe -- a biphasic QRS whose
    lobes can be of comparable size, the shape the production recordings showed."""
    t = np.arange(w)
    r = r_amp * np.exp(-((t - w // 3) ** 2) / (2 * 6.0 ** 2))
    s = -s_amp * np.exp(-((t - w // 3 - 40) ** 2) / (2 * 6.0 ** 2))
    return r + s


def test_biphasic_complex_keeps_every_beat_on_the_r_lobe():
    """Regression for 2.5.0: with R and S lobes of comparable size, taking the taller lobe
    beat by beat put some beats on the S lobe and the averaged template was built from
    misaligned windows. Every detected peak must now sit on the R lobe, and the beats
    must all be found."""
    fs, n_beats, w = 2000, 11, 120
    rng = np.random.default_rng(7)
    emg = rng.normal(0, 0.02, (int(12 * fs), 2))
    r_idx = []
    for k in range(n_beats):
        p = int((0.5 + k) * fs)
        s_amp = 1.15 if k % 2 else 0.9        # the S lobe is the taller one on every other beat
        emg[p:p + w, :] += _qrs(w, 1.0, s_amp)[:, None]
        r_idx.append(p + w // 3)
    _proc, _win, peaks_s = E.remove_ecg(emg.copy(), emg[:, 0], samplingfrequency=fs,
                                        ecgminheight=0.3, ecgmindistance=0.5,
                                        ecgminwidth=0.001, windowsize=0.4)
    peaks = np.round(np.asarray(peaks_s) * fs).astype(int)
    assert len(peaks) == n_beats
    assert np.all(np.abs(peaks - np.array(r_idx)) <= 4)     # on R, never 40 samples later on S


def _upright_channel(seed, n_beats=11, fs=2000, w=120):
    """Upright QRS (R 1.0, a small S below the height threshold) on two channels."""
    rng = np.random.default_rng(seed)
    emg = rng.normal(0, 0.02, (int(12 * fs), 2))
    r_idx = []
    for k in range(n_beats):
        p = int((0.5 + k) * fs)
        emg[p:p + w, :] += _qrs(w, 1.0, 0.1)[:, None]
        r_idx.append(p + w // 3)
    return emg, np.array(r_idx), fs


@pytest.mark.parametrize("n_beats", [2, 11])
def test_a_single_negative_artefact_does_not_switch_an_upright_channel(n_beats):
    """Found by review: comparing median heights alone let ONE large negative artefact
    (a movement or swallowing deflection) switch an upright channel onto that single peak,
    so 11 beats became 1 and the ECG was not removed at all. The count guard keeps the
    positive R-peaks, even in a recording of two beats."""
    emg, r_idx, fs = _upright_channel(seed=11, n_beats=n_beats)
    t = np.arange(emg.shape[0])
    emg += (-3.0 * np.exp(-((t - int(6.75 * fs)) ** 2) / (2 * 10.0 ** 2)))[:, None]
    _proc, _win, peaks_s = E.remove_ecg(emg.copy(), emg[:, 0], samplingfrequency=fs,
                                        ecgminheight=0.3, ecgmindistance=0.5,
                                        ecgminwidth=0.001, windowsize=0.4)
    peaks = np.round(np.asarray(peaks_s) * fs).astype(int)
    assert len(peaks) == len(r_idx)
    assert np.all(np.abs(peaks - r_idx) <= 4)


def test_a_few_negative_ectopic_beats_do_not_switch_an_upright_channel():
    """Two ectopic beats with a negative deflection 2.5 times the R-wave, in an otherwise
    upright channel: the channel stays on the positive R-peaks of the normal beats."""
    emg, r_idx, fs = _upright_channel(seed=12)
    w = 120
    for p in (int(3.0 * fs), int(8.0 * fs)):            # between normal beats
        emg[p:p + w, :] += -_qrs(w, 2.5, 0.0)[:, None]
    _proc, _win, peaks_s = E.remove_ecg(emg.copy(), emg[:, 0], samplingfrequency=fs,
                                        ecgminheight=0.3, ecgmindistance=0.4,
                                        ecgminwidth=0.001, windowsize=0.4)
    peaks = np.round(np.asarray(peaks_s) * fs).astype(int)
    assert len(peaks) == len(r_idx)
    assert np.all(np.abs(peaks - r_idx) <= 4)


def _beats_with_t_wave(seed, r_amp, t_amp, n_beats=11, fs=2000, w=120, noise=0.02):
    """A QRS of R amplitude r_amp (negative = inverted, no S) and a T-wave of t_amp (a 160 ms
    half-sine 250 ms after the R-wave) on two channels."""
    rng = np.random.default_rng(seed)
    emg = rng.normal(0, noise, (int(12 * fs), 2))
    r_idx = []
    for k in range(n_beats):
        p = int((0.5 + k) * fs)
        emg[p:p + w, :] += (np.sign(r_amp) * _qrs(w, abs(r_amp), 0.0))[:, None]
        t0 = p + w // 3 + int(0.25 * fs)
        tt = np.arange(int(0.16 * fs))
        emg[t0:t0 + tt.size, :] += (t_amp * np.sin(np.pi * tt / tt.size))[:, None]
        r_idx.append(p + w // 3)
    return emg, np.array(r_idx), fs


def _detected(emg, fs, height=0.3):
    _proc, _win, peaks_s = E.remove_ecg(emg.copy(), emg[:, 0], samplingfrequency=fs,
                                        ecgminheight=height, ecgmindistance=0.5,
                                        ecgminwidth=0.001, windowsize=0.4)
    return np.round(np.asarray(peaks_s) * fs).astype(int)


@pytest.mark.parametrize("r_over_t", [1.1, 2.9])
def test_inverted_r_wave_with_upright_t_wave_uses_the_negative_polarity(r_over_t):
    """Ticket 5.7 / K-191 with a realistic inverted lead: a negative R-wave and an upright
    T-wave that clears the height threshold, so both searches find every beat. In the QRS band
    the R-waves dominate the T-waves, so the whole channel uses the negative polarity and every
    beat lands on the R-wave, also when the T-wave is nearly as tall as the R-wave (comparing
    raw peak heights used the T-waves there)."""
    emg, r_idx, fs = _beats_with_t_wave(seed=3, r_amp=-1.0, t_amp=1.0 / r_over_t)
    peaks = _detected(emg, fs)
    assert len(peaks) == len(r_idx)
    assert np.all(np.abs(peaks - r_idx) <= 4)


@pytest.mark.parametrize("t_over_r", [1.7, 3.0])
def test_a_deep_negative_t_wave_does_not_take_over_an_upright_channel(t_over_r):
    """Found by review: comparing raw peak heights switched an upright channel whose
    negative T-waves are deeper than its R-waves onto the T-waves, a third of a second after
    the R-waves, so the ECG was not removed at all. In the QRS band the T-waves are small, and
    the channel stays on its R-waves."""
    emg, r_idx, fs = _beats_with_t_wave(seed=5, r_amp=0.5, t_amp=-0.5 * t_over_r)
    peaks = _detected(emg, fs)
    assert len(peaks) == len(r_idx)
    assert np.all(np.abs(peaks - r_idx) <= 4)


@pytest.mark.parametrize("artefacts", [1, 2])
def test_a_few_positive_artefacts_do_not_hold_an_inverted_channel(artefacts):
    """Found by review, the mirror of the negative artefact: in a clean inverted channel the
    positive search finds only an artefact or two, which comparing heights kept, so 11 beats
    became 1 or 2. The count guard works both ways, and the channel uses its R-waves."""
    emg, r_idx, fs = _beats_with_t_wave(seed=13, r_amp=-1.0, t_amp=0.2)
    t = np.arange(emg.shape[0])
    for c in (6.75, 9.75)[:artefacts]:
        emg += (1.0 * np.exp(-((t - int(c * fs)) ** 2) / (2 * 10.0 ** 2)))[:, None]
    peaks = _detected(emg, fs)
    assert len(peaks) == len(r_idx)
    assert np.all(np.abs(peaks - r_idx) <= 4)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("where_s", [11.6, 6.9])
def test_a_non_finite_sample_does_not_hide_an_inverted_r_wave(value, where_s):
    """Found by review: one inf sample turned the whole QRS band to NaN, so an inverted channel
    fell back to its upright T-waves (11 beats, none on the R-wave); one NaN made the median NaN
    and no peak was found at all; and an inf within the minimum distance of an R-wave crowded
    it out. Non-finite samples now count as the median, after the last beat (11.6 s) or 0.38 s
    after an R-wave (6.9 s) alike."""
    emg, r_idx, fs = _beats_with_t_wave(seed=3, r_amp=-1.0, t_amp=0.5)
    emg[int(where_s * fs), 0] = value
    peaks = E.detect_r_peaks(emg[:, 0], fs, 0.3, 0.5, 0.001)
    assert len(peaks) == len(r_idx)
    assert np.all(np.abs(peaks - r_idx) <= 4)
    assert E.detect_r_peaks(np.full(1000, np.nan), fs, 0.3, 0.5, 0.001).size == 0


def _paced_channel(seed, spike, t_amp=-0.4, tail=0.0, fs=2000, n_beats=11):
    """A paced rhythm on two channels: a 1 ms pacing spike 55 ms before a broad upright QRS
    (43 ms FWHM) with a small S, a discordant negative T-wave, and optionally the
    pacemaker's recharge tail after the spike."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, (int(12 * fs), 2))
    t = np.arange(int(0.6 * fs)) / fs

    def lobe(c, fwhm, a):
        return a * np.exp(-0.5 * ((t - c) / (fwhm / 2.355)) ** 2)
    sp = 0.025
    beat = (lobe(sp, 0.001, spike) + lobe(sp + 0.055, 0.043, 1.0) + lobe(sp + 0.105, 0.03, -0.25)
            + lobe(sp + 0.335, 0.18, t_amp))
    d = t - (sp + 0.001)
    beat += np.where(d > 0, -tail * spike * np.exp(-d / 0.03) * (1 - np.exp(-d / 0.002)), 0.0)
    spikes = []
    for k in range(n_beats):
        p = int((0.5 + k) * fs)
        x[p:p + t.size, :] += beat[:, None]
        spikes.append(p + int(sp * fs))
    return x, np.array(spikes), fs


@pytest.mark.parametrize("spike, tail", [(2.0, 0.0), (3.0, 0.05)])
def test_a_paced_channel_stays_on_its_pacing_spikes(spike, tail):
    """Found by review: a pacing spike has almost no QRS-band amplitude, and the broad QRS after
    it makes the band negative at the spike, so with a signed comparison the small positive
    band value of the discordant T-waves won and every peak sat on a T-wave (spike twice the
    QRS). The positive side now counts by size, and the channel stays on the spikes."""
    x, spikes, fs = _paced_channel(seed=21, spike=spike, tail=tail)
    peaks = E.detect_r_peaks(x[:, 0], fs, 0.3, 0.4, 0.001)
    assert len(peaks) == len(spikes)
    assert np.all(np.abs(peaks - spikes) <= int(0.06 * fs))


def test_a_deep_s_lobe_keeps_the_channel_on_the_r_lobe():
    """The margin behind _NEGATIVE_DOMINANCE: the production recordings measured 0.71-1.35
    as negative/positive QRS-band ratio. A biphasic channel whose S lobe is 2.5 times its R
    lobe measures about 1.45 in the band and stays on the R lobe, every beat; a threshold of
    1.4 would move it onto S, and comparing raw heights (2.5) did."""
    fs, n_beats, w = 2000, 11, 120
    rng = np.random.default_rng(17)
    emg = rng.normal(0, 0.02, (int(12 * fs), 2))
    r_idx = []
    for k in range(n_beats):
        p = int((0.5 + k) * fs)
        emg[p:p + w, :] += _qrs(w, 1.0, 2.5)[:, None]
        r_idx.append(p + w // 3)
    peaks = _detected(emg, fs)
    assert len(peaks) == n_beats
    assert np.all(np.abs(peaks - np.array(r_idx)) <= 4)


def test_ecg_removal_suppresses_contamination():
    emg, true_peaks, fs = _make_signal()
    detect = emg[:, 0]
    before = E.peak_window_rms(detect, true_peaks, fs)
    processed, _win, peaks_s = E.remove_ecg(
        emg.copy(), detect, samplingfrequency=fs,
        ecgminheight=0.2, ecgmindistance=0.5, ecgminwidth=0.001, windowsize=0.4)
    peaks_samp = (np.asarray(peaks_s) * fs).astype(int)
    after = E.peak_window_rms(np.asarray(processed)[:, 0], peaks_samp, fs)
    suppression = 1 - after / before
    assert len(peaks_samp) >= 8                   # detected most beats
    assert suppression > 0.3                      # meaningfully reduced ECG


def test_detection_is_deterministic_and_param_driven():
    # Same params + same signal -> identical detected peaks (consistent per test).
    emg, _tp, fs = _make_signal(seed=1)
    kw = dict(samplingfrequency=fs, ecgminheight=0.2, ecgmindistance=0.5,
              ecgminwidth=0.001, windowsize=0.4)
    _p1, _w1, peaks1 = E.remove_ecg(emg.copy(), emg[:, 0], **kw)
    _p2, _w2, peaks2 = E.remove_ecg(emg.copy(), emg[:, 0], **kw)
    assert np.array_equal(peaks1, peaks2)
    # A higher threshold detects fewer/equal peaks (parameter actually drives it).
    _p3, _w3, peaks3 = E.remove_ecg(emg.copy(), emg[:, 0], **{**kw, "ecgminheight": 5.0})
    assert len(peaks3) <= len(peaks1)


def test_inverted_r_wave_is_detected_too():
    """Ticket 5.7 / K-191: the detector used to find positive peaks only (find_peaks'
    height= convention), so a channel whose R-wave is inverted (negative-going) could
    not be used for detection at all. It must now find the beats regardless of
    polarity, and count the same as the upright case on the same recording."""
    kw = dict(samplingfrequency=2000, ecgminheight=0.2, ecgmindistance=0.5,
              ecgminwidth=0.001, windowsize=0.4)
    up, true_peaks, fs = _make_signal(seed=2, invert=False)
    down, _tp2, _fs2 = _make_signal(seed=2, invert=True)
    _pu, _wu, peaks_up = E.remove_ecg(up.copy(), up[:, 0], **kw)
    _pd, _wd, peaks_down = E.remove_ecg(down.copy(), down[:, 0], **kw)
    assert len(peaks_down) == len(peaks_up) == len(true_peaks)
    # detected instants line up between the two polarities (same underlying beats;
    # the exact sample can jitter a bit since the added noise differs by sign too)
    assert np.allclose(np.sort(peaks_up), np.sort(peaks_down), atol=15.0 / fs)


def test_detection_matches_dc_removed_signal_like_auto_suggest():
    """Ticket 5.7 / K-191: remove_ecg used to run find_peaks on the raw channel while
    suggest_ecg_settings derives ecgminheight from a median-subtracted copy, so a
    channel with a DC offset made a suggested height wrong for what remove_ecg
    actually saw. Detection must now be identical regardless of a constant DC offset
    added to the channel (median-removal cancels it exactly)."""
    kw = dict(samplingfrequency=2000, ecgminheight=0.2, ecgmindistance=0.5,
              ecgminwidth=0.001, windowsize=0.4)
    plain, _tp, fs = _make_signal(seed=4, dc_offset=0.0)
    offset, _tp2, _fs2 = _make_signal(seed=4, dc_offset=3.7)
    _p1, _w1, peaks_plain = E.remove_ecg(plain.copy(), plain[:, 0], **kw)
    _p2, _w2, peaks_offset = E.remove_ecg(offset.copy(), offset[:, 0], **kw)
    assert np.array_equal(peaks_plain, peaks_offset)


@pytest.mark.parametrize("import_ok", [True])
def test_pipeline_reports_ecg_suppression(import_ok, tmp_path):
    """The pipeline attaches ECG diagnostics to each FileResult when remove_ecg is on."""
    import os
    INPUT = os.path.join(os.path.dirname(__file__), "..", "golden", "input")
    if not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")):
        pytest.skip("synthetic input not present")
    # synthetic data has no ECG, but the diagnostic plumbing must still populate.
    from respmech.settingsio.migrate import migrate_dict
    from respmech.core.pipeline import run_batch
    legacy = {"input": {"inputfolder": os.path.abspath(INPUT), "files": "synth_case_A.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": True, "column_detect": 0,
                                     "minheight": 0.5, "mindistance": 0.3, "minwidth": 0.001}},
              "output": {"outputfolder": str(tmp_path), "data": {"savebreathbybreathdata": True}}}
    s, _ = migrate_dict(legacy)
    res = run_batch(s)
    fr = res.ok_files["synth_case_A.csv"]
    assert fr.ecg is not None
    assert "n_peaks" in fr.ecg and "suppression" in fr.ecg


def test_stage_ecg_reduction_reports_suppression_matching_the_pipeline(tmp_path):
    """The 'EMG - ECG reduction' tab (ui.workers.stage_ecg_reduction) is a live TUNING
    surface, but until the 04-08-2026 fix it never computed the peak-window-RMS
    suppression core.pipeline._ecg_remove reports for a real run — so 'Min height'/'Min
    gap' were dialled in on nothing but eyeballing the traces. It must now report a
    positive suppression, on a syntetic signal with known R-peaks, of the same order of
    magnitude as the pipeline computes on the identically-loaded raw matrix, using the
    SAME core.emg.peak_window_rms helper (no new variant, no golden-affecting change)."""
    import pandas as pd
    from respmech.settingsio.migrate import migrate_dict
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core.io.loaders import load
    from respmech.core.pipeline import _ecg_remove
    from respmech.ui.workers import stage_ecg_reduction

    emg, _true_peaks, fs = _make_signal(seed=7)         # same synthetic ECG-on-EMG builder
    n = emg.shape[0]
    t = np.arange(n) / fs
    path = tmp_path / "ecg_contaminated.csv"
    zeros = np.zeros(n)
    pd.DataFrame({"time": t, "c2": emg[:, 0], "c3": emg[:, 1], "c4": emg[:, 2],
                 "flow": zeros, "volume": zeros,
                 "poes": zeros, "pgas": zeros, "pdi": zeros}).to_csv(path, index=False)

    legacy = {"input": {"inputfolder": str(tmp_path), "files": path.name,
                        "format": {"samplingfrequency": fs},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300},
                             "emg": {"remove_ecg": True, "column_detect": 0,
                                     "minheight": 0.2, "mindistance": 0.5, "minwidth": 0.001,
                                     "windowsize": 0.4}},
              "output": {"outputfolder": str(tmp_path), "data": {}}}
    s, _ = migrate_dict(legacy)
    s.input.folder = str(tmp_path)

    data = stage_ecg_reduction(s, str(path))
    assert data["ecg_applied"] is True
    supp = data["suppression"]
    assert supp is not None and supp == supp            # not None, not NaN
    assert supp > 0.3                                    # meaningfully reduced (test_ecg_removal_* threshold)

    # Cross-check against what a REAL run computes, on the identically-loaded raw matrix,
    # via the same helper the ticket requires (core.emg.peak_window_rms, no new variant).
    ls = to_legacy_ns(s)
    _flow, _v, _p, _g, _d, _e, raw_emg = load(str(path), ls)
    _emgcols, diag = _ecg_remove(ls, raw_emg)
    assert diag is not None
    assert abs(supp - diag["suppression"]) < 0.05        # same order of magnitude, same detector

    # Removal OFF: nothing to suppress, so no number at all (None, not NaN) -- distinct
    # from "computed but degenerate", which stays NaN.
    s_off, _ = migrate_dict({**legacy, "processing": {**legacy["processing"],
                             "emg": {**legacy["processing"]["emg"], "remove_ecg": False}}})
    s_off.input.folder = str(tmp_path)
    data_off = stage_ecg_reduction(s_off, str(path))
    assert data_off["ecg_applied"] is False
    assert data_off["suppression"] is None


def test_stage_ecg_reduction_marks_the_same_r_peaks_with_removal_off(tmp_path):
    """Found by review: with ECG removal off, the 'EMG - ECG reduction' tab marked R-peaks with
    find_peaks on the raw channel, positive peaks only, while a run with removal on uses
    remove_ecg's detector, so on an inverted channel the markers sat on the upright T-waves
    the run does not use. Both now come from detect_r_peaks."""
    import pandas as pd
    from respmech.settingsio.migrate import migrate_dict
    from respmech.ui.workers import stage_ecg_reduction

    emg, r_idx, fs = _beats_with_t_wave(seed=3, r_amp=-1.0, t_amp=0.5)
    n = emg.shape[0]
    path = tmp_path / "inverted.csv"
    zeros = np.zeros(n)
    pd.DataFrame({"time": np.arange(n) / fs, "c2": emg[:, 0], "c3": emg[:, 1],
                  "flow": zeros, "volume": zeros, "poes": zeros, "pgas": zeros,
                  "pdi": zeros}).to_csv(path, index=False)

    def settings(remove):
        legacy = {"input": {"inputfolder": str(tmp_path), "files": path.name,
                            "format": {"samplingfrequency": fs},
                            "data": {"column_poes": 6, "column_pgas": 7, "column_pdi": 8,
                                     "column_volume": 5, "column_flow": 4, "columns_emg": [2, 3]}},
                  "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                               "avgresamplingobs": 300},
                                 "emg": {"remove_ecg": remove, "column_detect": 0,
                                         "minheight": 0.3, "mindistance": 0.5, "minwidth": 0.001,
                                         "windowsize": 0.4}},
                  "output": {"outputfolder": str(tmp_path), "data": {}}}
        s, _ = migrate_dict(legacy)
        s.input.folder = str(tmp_path)
        return s

    on = stage_ecg_reduction(settings(True), str(path))
    off = stage_ecg_reduction(settings(False), str(path))
    assert on["ecg_applied"] is True and off["ecg_applied"] is False
    assert np.array_equal(off["peaks"], on["peaks"])
    peaks = np.round(np.asarray(off["peaks"]) * fs).astype(int)
    assert len(peaks) == len(r_idx)
    assert np.all(np.abs(peaks - r_idx) <= 4)


def test_remove_ecg_returns_one_c_ordered_array_matching_the_row_list():
    # remove_ecg used to return a list of per-sample rows that every caller re-stacked
    # with np.array; it now returns that stacked array itself. Pin that the values,
    # dtype, shape and C memory layout are exactly what the old round trip produced.
    emg, _tp, fs = _make_signal(seed=2)
    kw = dict(samplingfrequency=fs, ecgminheight=0.2, ecgmindistance=0.5,
              ecgminwidth=0.001, windowsize=0.4)
    processed, _w, _p = E.remove_ecg(emg.copy(), emg[:, 0], **kw)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == emg.shape
    assert processed.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("n_samples, n_channels", [(0, 1), (0, 3), (1, 1), (5, 1), (7, 3)])
def test_stack_channels_matches_the_former_list_of_rows_round_trip(dtype, n_samples, n_channels):
    # Reference: what remove_ecg returned before (a list of per-sample rows) after the
    # np.array every caller applies. Values, dtype, shape and bytes must be identical,
    # including the shape of a recording without samples.
    rng = np.random.default_rng(n_samples * 10 + n_channels)
    retch = [(rng.standard_normal(n_samples) * 100).astype(dtype) for _ in range(n_channels)]
    old = np.array(list(np.array(retch).T))
    new = np.array(E._stack_channels(retch))
    assert new.dtype == old.dtype
    assert new.shape == old.shape
    assert new.tobytes() == old.tobytes()

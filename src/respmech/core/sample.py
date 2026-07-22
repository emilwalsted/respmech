"""A small but physiologically realistic synthetic recording, generated on demand for
the "Explore sample data" onboarding door (feature P23).

Real recordings are large (and are patient data), so instead we synthesise a short
breathing trace from a respiratory-mechanics model whenever a first-time user asks to
explore. The layout matches a typical LabChart export so the standard channel mapping
applies:

    col 1 time · 2–4 diaphragm EMG · 5 flow · 6 volume · 7 Poes · 8 Pgas · 9 Pdi

The model's parameters are anchored to statistics measured from real oesophageal
recordings (moderate loaded breathing), so the traces *look* real without reproducing
any patient data:

* **Mechanics.** Oesophageal pressure is an **elastic** term (rising with lung volume,
  with a gentle curvature) plus a **resistive** term (proportional to flow):
  ``Poes = Poes₀ − E·V − E₂·V² + R·flow``. Because inspiratory and expiratory flow have
  opposite sign, at any given volume the two limbs sit ``2·R·flow`` apart, so the
  Campbell (Poes–volume) loop **opens** into a leaf whose area is the resistive work of
  breathing. Elastance and tidal volume are chosen so the loop is a realistic open leaf
  rather than a steep sliver. Gastric pressure rises with inspiration (diaphragm
  descent); Pdi = Pgas − Poes. A slow downward **volume drift** (integration bias) is
  baked in for the drift-correction demo.

* **Diaphragm EMG.** Each channel is **band-limited noise** centred near 30 Hz (the
  measured diaphragm-EMG band, not a pure tone), amplitude-modulated by an
  inspiratory-weighted burst over a low tonic floor, plus a coloured noise floor. A
  **heartbeat (ECG) artefact** — a sharp triphasic QRS at ~72 bpm — is superimposed,
  strongest on the middle electrode (its natural detection channel) and weakest on the
  outer ones, so the sample exercises ECG removal and spectral noise reduction. The same
  cardiac clock ripples the pressures.

Deterministic (fixed seed) so the sample is identical every time. Computation only —
no file I/O beyond the single CSV it writes, and no Qt.
"""
from __future__ import annotations

import os

import numpy as np

FS = 1000                      # Hz — covers the diaphragm-EMG band (<~250 Hz) and a crisp QRS
N_BREATHS = 9
PERIOD_S = 3.2                 # ~19 breaths/min
VT_L = 1.0                     # tidal volume (L)
RESISTANCE = 2.0               # cmH₂O/(L/s) — opens the loop to ~half its elastic height
ELASTANCE = 6.0                # cmH₂O/L — pleural pressure per litre (real loaded ~4–6)
ELASTANCE_CURV = 1.5           # cmH₂O/L² — gentle bow on the elastic axis
POES_EE = -5.0                 # end-expiratory pleural pressure (cmH₂O)
PGAS_EE = 8.0                  # end-expiratory gastric pressure (cmH₂O)
PGAS_SWING = 6.0               # gastric-pressure rise over a tidal breath (cmH₂O)
VOL_DRIFT_L = -0.4             # slow downward integration drift over the record (L)
HR_HZ = 1.2                    # heart rate (72 bpm)
LEAD_S = 1.0                   # quiet rest lead-in (EMG- and ECG-free) for the noise reference
N_EMG = 3
EMG_SCALE = (0.85, 1.0, 0.90)  # per-channel EMG amplitude (centre-weighted)
ECG_SCALE = (0.55, 1.0, 0.70)  # per-channel ECG prominence (strongest on the middle electrode)
ECG_R_OVER_EMG = 5.0           # R-wave peak as a multiple of the EMG burst peak (real: 5–20×);
                               # this is why removal matters — an R-wave inflates the EMG RMS
EMG_CARRIER_WEIGHT = 0.45      # weight of the smooth band-limited interference base (the
                               # unresolvable many-small-units background)
EMG_TEXTURE_WEIGHT = 0.20      # weight of the superimposed MUAP-spike texture (the ragged,
                               # drive-recruited morphology). Together they keep the burst peak
                               # ~3·burst_peak while the spikes give the interference pattern.
FILENAME = "sample_recording.csv"
DETECT_CHANNEL = 1             # 0-based index of the strongest-ECG EMG channel


def _physio_noise(n, rng, amp, smooth_s=0.35):
    """Low-frequency, physiological-looking wander (not white fuzz): white noise
    low-passed by a Hann window so it varies over ~a third of a second, scaled to
    ``amp`` (cmH₂O). Gives the Campbell loops a natural breath-to-breath spread."""
    w = np.hanning(max(3, int(smooth_s * FS)))
    w = w / w.sum()
    x = np.convolve(rng.standard_normal(n + w.size), w, mode="same")[:n]
    return amp * x / (x.std() or 1.0)


def _instrument_noise(n, rng, amp):
    """Broadband instrumentation noise weighted ABOVE the EMG's dominant band (it ramps in
    from ~120 Hz), scaled to std ``amp``. Real diaphragm-EMG noise sits mostly higher than
    the ~30 Hz signal peak, which is exactly what makes it separable — spectral noise
    reduction gates it out while leaving the low-frequency burst, a clear before/after."""
    x = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    h = np.clip((f - 120.0) / 130.0, 0.0, 1.0)                    # ramp 120→250 Hz, flat above
    y = np.fft.irfft(x * h, n=n)
    return amp * y / (y.std() or 1.0)


def _emg_carrier(n, rng):
    """Unit-RMS band-limited noise shaped to the diaphragm-EMG spectrum (peak ~30 Hz,
    rolling off above ~50 Hz, band-limited 20–250 Hz) — the measured shape, not a tone."""
    x = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    h = np.where((f >= 20.0) & (f <= 250.0), (f / 30.0) * np.exp(-f / 45.0), 0.0)
    y = np.fft.irfft(x * h, n=n)
    return y / (y.std() or 1.0)


def _muap_wavelet(dur_s=0.028, f0=38.0):
    """A single surface motor-unit action potential (MUAP) template. Real needle MUAPs are
    sharp triphasic spikes, but volume conduction through chest-wall tissue low-passes the
    *surface* MUAP, so its energy sits in the diaphragm-EMG band and rolls off well before
    the removable >120 Hz instrument floor. Built as the first derivative of a Gaussian (the
    dominant biphasic swing) plus a small second-derivative lobe (the third phase); the
    Gaussian width sets the spectral centre (~f0). Zero-mean (no per-spike DC step), then
    peak-normalised so the per-spike amplitude is carried by the spike train, not the shape."""
    m = int(round(dur_s * FS))
    t = (np.arange(m) - m // 2) / FS
    sig = 1.0 / (2.0 * np.pi * f0)
    g = np.exp(-0.5 * (t / sig) ** 2)
    d1 = -(t / sig ** 2) * g                                     # biphasic (1st derivative)
    d2 = ((t ** 2 - sig ** 2) / sig ** 4) * g                    # triphasic (2nd derivative)
    w = d1 + 0.35 * sig * d2
    return w / (np.max(np.abs(w)) or 1.0)


def _muap_texture(n, rng, drive, wavelet, rate_max=380.0, size_k=1.4, amp_jitter=0.28):
    """The recruited-motor-unit interference pattern: a drive-gated Poisson train of MUAPs
    convolved with the finite ``wavelet``. As neural ``drive`` rises through an inspiration,
    more units fire (spike DENSITY ∝ ``rate_max·drive``, thinned Bernoulli) AND larger units
    recruit (per-spike SIZE ∝ ``1 + size_k·drive``, the size principle), so the pattern fills
    in — the ragged, biphasic-spiky texture of real surface EMG rather than smooth noise.
    ``drive`` is 0 in the quiet lead-in, so no MUAPs land there. Uses only the seeded ``rng``
    (deterministic). Random per-spike polarity + log-normal jitter give the interference look."""
    p = np.clip(rate_max * drive / FS, 0.0, 1.0)                 # per-sample firing probability
    idx = np.flatnonzero(rng.random(n) < p)                      # MUAP onset samples
    imp = np.zeros(n)
    if idx.size:
        mag = 1.0 + size_k * drive[idx]                         # size principle: bigger with drive
        jit = np.exp(amp_jitter * rng.standard_normal(idx.size))
        sign = np.where(rng.random(idx.size) < 0.5, -1.0, 1.0)  # mixed-polarity units
        imp[idx] = sign * mag * jit
    return np.convolve(imp, wavelet, mode="same")


def _qrs():
    """A triphasic QRS complex (~50 ms): small Q dip, tall R spike, deep S trough — the
    measured morphology of the cardiac artefact on oesophageal EMG. The R is given a
    realistic ~14 ms width (not a 1–2 sample spike) so averaged-template ECG removal can
    align and cancel it cleanly, as it does on real recordings."""
    m = int(0.06 * FS)
    t = (np.arange(m) - m // 2) / FS
    def g(a, mu, sig):
        return a * np.exp(-0.5 * ((t - mu) / sig) ** 2)
    return g(1.0, 0.0, 0.006) + g(-0.22, -0.013, 0.004) + g(-0.55, 0.019, 0.007)


def _heartbeat_impulses(n, rng):
    """An impulse train at ~HR (mild beat-to-beat variation), with no beats during the
    quiet lead-in. One shared clock so the EMG artefact and the pressure ripple align."""
    train = np.zeros(n + int(0.05 * FS))
    t = LEAD_S + 0.15 + 0.02 * rng.standard_normal()
    while int(t * FS) < n:
        train[int(t * FS)] += 1.0
        t += (1.0 / HR_HZ) * (1.0 + 0.08 * rng.standard_normal())
    return train


def _unit(x):
    peak = np.max(np.abs(x))
    return x / (peak or 1.0)


def _ecg_signal(train, n):
    """The sharp triphasic QRS artefact seen on the EMG (unit R-amplitude)."""
    return _unit(np.convolve(train, _qrs(), mode="same")[:n])


def _cardiac_pressure(train, n):
    """The smooth pulsatile cardiac oscillation seen on the PRESSURES — the heart pushing
    on the oesophageal balloon, not a sharp QRS — so the Poes/Pgas ripple is rounded."""
    m = int(0.12 * FS)
    t = (np.arange(m) - m // 2) / FS
    pulse = np.exp(-0.5 * (t / 0.03) ** 2) - 0.35 * np.exp(-0.5 * ((t - 0.05) / 0.035) ** 2)
    return _unit(np.convolve(train, pulse, mode="same")[:n])


def _one_breath(rng, T, vt):
    """One breath on the sin² volume model (segments reliably): volume 0→VT→0, flow the
    negative derivative (negative on inspiration), and the elastic Poes/Pgas terms. The
    resistive Poes term and the EMG are added later, continuously across the record."""
    n = int(round(T * FS))
    t = np.arange(n) / FS
    vol = vt * np.sin(np.pi * t / T) ** 2                          # 0 → VT → 0
    dvdt = vt * (np.pi / T) * np.sin(2 * np.pi * t / T)            # +insp … −exp
    flow = -dvdt                                                   # inspiration → flow negative
    poes_el = POES_EE - ELASTANCE * vol - ELASTANCE_CURV * vol ** 2
    pgas = PGAS_EE + (PGAS_SWING / vt) * vol
    # EMG burst confined to INSPIRATION (the negative-flow first half): it rises from
    # inspiration onset, crescendos to a peak late in inspiration, decrescendos back to
    # near-silent by end-inspiration, and is quiet through expiration — so successive
    # bursts sit over the inspirations with a clear gap (the expiration) between them,
    # coordinated with flow (hence with the Poes/Pgas swings built from the same breath).
    u = t / (0.5 * T)                                             # 0..1 over inspiration, >1 after
    up = 0.65                                                     # burst peaks late in inspiration
    rise = np.clip(u / up, 0, 1) ** 1.3
    fall = np.clip((1.0 - u) / (1.0 - up), 0, 1) ** 1.6
    env = np.where(u <= 1.0, np.where(u < up, rise, fall), 0.0)
    return vol, flow, poes_el, pgas, env


def _signals():
    rng = np.random.default_rng(20260711)
    nlead = int(LEAD_S * FS)
    tl = np.arange(nlead) / FS
    # a gentle resting expiratory lead-in (EMG- and ECG-free) so the first breath starts
    # cleanly and [0, ~LEAD_S] is a quiet reference for noise reduction
    vol_all = [np.zeros(nlead)]
    flow_all = [0.15 * np.sin(np.pi * tl / LEAD_S)]
    poesel_all = [np.full(nlead, POES_EE)]
    pgas_all = [np.full(nlead, PGAS_EE)]
    env_all = [np.zeros(nlead)]
    for b in range(N_BREATHS):
        T = PERIOD_S * (1.0 + 0.10 * np.sin(1.3 * b + 0.4))         # breath-to-breath variation
        vt = VT_L * (1.0 + 0.10 * np.sin(0.7 * b + 0.5))
        vol, flow, poes_el, pgas, env = _one_breath(rng, T, vt)
        vol_all.append(vol); flow_all.append(flow); poesel_all.append(poes_el)
        pgas_all.append(pgas); env_all.append(env)
    vol = np.concatenate(vol_all); flow = np.concatenate(flow_all)
    poes_el = np.concatenate(poesel_all); pgas = np.concatenate(pgas_all)
    env = np.concatenate(env_all)
    n = len(flow)
    # the neural drive leads the flow slightly — advance the burst envelope by ~50 ms
    lead = int(0.05 * FS)
    env = np.roll(env, -lead); env[-lead:] = 0.0
    # the tonic (between-burst) EMG floor is present only while breathing; the rest lead-in
    # stays pure noise so [0, LEAD_S] is a clean reference for spectral noise reduction
    breathing = np.ones(n); breathing[:nlead] = 0.0

    beats = _heartbeat_impulses(n, rng)                            # one shared cardiac clock
    ecg = _ecg_signal(beats, n)                                    # sharp QRS on the EMG
    cardiac = _cardiac_pressure(beats, n)                          # smooth pulse on the pressures

    # EMG channels: a smooth band-limited-noise carrier (the unresolvable interference base)
    # PLUS a drive-gated train of MUAP-like spikes (the ragged recruited-motor-unit texture),
    # an inspiratory burst over a tonic floor, + the heartbeat artefact (centre-weighted) +
    # a coloured noise floor.
    burst_peak = 0.06
    tonic = 0.15
    # The carrier peaks at ~3× its RMS and the MUAP spikes add sharp deflections on top; the
    # two weights (EMG_CARRIER_WEIGHT/EMG_TEXTURE_WEIGHT) keep the instantaneous EMG burst peak
    # on the strongest channel ~3·burst_peak. The R-wave is ECG_R_OVER_EMG× that, i.e. it dwarfs
    # the EMG (the whole point of ECG removal).
    r_peak = ECG_R_OVER_EMG * 3.0 * burst_peak
    muap = _muap_wavelet()                                        # one surface-MUAP template, shared
    # A DEDICATED stream for the MUAP texture, so adding it does not perturb the carrier/floor/
    # pressure draws from the main rng: everything except the EMG burst morphology stays
    # byte-identical to the pre-texture sample (same noise reference, same Campbell/drift figures).
    tex_rng = np.random.default_rng(20260712)
    emg = []
    for ch in range(N_EMG):
        # neural drive: 0 in the quiet lead-in, a low tonic floor while breathing, rising to ~1
        # at the inspiratory peak. It modulates BOTH the carrier amplitude AND the MUAP firing
        # density/size, so amplitude and spike density grow together — recruitment fill-in.
        drive = tonic * breathing + (1.0 - tonic) * env
        carrier = _emg_carrier(n, rng)
        texture = _muap_texture(n, tex_rng, drive, muap)         # drive-gated Poisson MUAP train
        texture[:nlead] = 0.0                                    # lead-in stays a pure noise reference
        amp = EMG_SCALE[ch] * burst_peak * drive
        signal = (EMG_CARRIER_WEIGHT * amp * carrier
                  + EMG_TEXTURE_WEIGHT * EMG_SCALE[ch] * burst_peak * texture)
        floor = _instrument_noise(n, rng, amp=0.35 * EMG_SCALE[ch] * burst_peak)
        emg.append(signal + floor + ECG_SCALE[ch] * r_peak * ecg)

    # pressures: elastic + resistive, a smooth cardiac ripple, and low-frequency wander
    poes = poes_el + RESISTANCE * flow + 0.7 * cardiac + _physio_noise(n, rng, amp=0.6)
    pgas = pgas + 0.4 * cardiac + _physio_noise(n, rng, amp=0.4)
    vol = vol + np.linspace(0.0, VOL_DRIFT_L, n)                   # slow integration drift
    pdi = pgas - poes                                             # transdiaphragmatic
    return flow, vol, poes, pgas, pdi, emg


def write_sample_recording(folder: str) -> dict:
    """Write the sample CSV into ``folder`` and return a descriptor with the file path,
    the 1-based channel mapping, the sampling frequency and the strongest-ECG channel,
    ready to build Settings."""
    os.makedirs(folder, exist_ok=True)
    flow, vol, poes, pgas, pdi, emg = _signals()
    n = len(flow)
    time = np.arange(n) / FS
    cols = [time] + list(emg) + [flow, vol, poes, pgas, pdi]
    emg_hdr = ",".join(f"EMG{i + 1}" for i in range(N_EMG))
    header = f"time,{emg_hdr},flow,volume,poes,pgas,pdi"
    path = os.path.join(folder, FILENAME)
    # %.5g keeps the CSV a few MB at 1 kHz (full precision would be >10 MB)
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=header,
               comments="", fmt="%.5g")
    emg_cols = list(range(2, 2 + N_EMG))
    return {"path": path, "folder": folder, "filename": FILENAME,
            "sampling_frequency": FS,
            "detect_channel": DETECT_CHANNEL,
            "reference_interval": [0.1, LEAD_S - 0.1],
            "mapping": {"flow": 2 + N_EMG, "volume": 3 + N_EMG, "poes": 4 + N_EMG,
                        "pgas": 5 + N_EMG, "pdi": 6 + N_EMG,
                        "emg": emg_cols, "entropy": []}}


def build_sample_settings(desc: dict, output_folder: str):
    """A ready ``Settings`` for the sample recording, with the full EMG pipeline — ECG
    removal (on the strongest-ECG channel) and spectral noise reduction (referenced to
    the quiet rest lead-in) — enabled, so "Explore sample data" demonstrates them out of
    the box. Shared by the onboarding door and the README-figure generator."""
    from respmech.core.settings import Settings  # noqa: PLC0415
    s = Settings()
    s.input.folder = desc["folder"]
    s.input.files = desc["filename"]
    s.input.format.sampling_frequency = desc["sampling_frequency"]
    m, ch = desc["mapping"], s.input.channels
    ch.flow, ch.volume = m["flow"], m["volume"]
    ch.poes, ch.pgas, ch.pdi = m["poes"], m["pgas"], m["pdi"]
    ch.emg, ch.entropy = list(m["emg"]), list(m["entropy"])
    s.processing.segmentation.buffer = 200
    e = s.processing.emg
    e.remove_ecg = True
    e.detect_channel = desc["detect_channel"]
    e.ecg_min_height = 0.4      # between the EMG/noise (~0.2) and the R-wave (~0.9) so the
                                # detector locks onto the heartbeats, not EMG or noise peaks
    e.noise.enabled = True
    e.noise.reference_file = desc["filename"]
    e.noise.use_expiration = False
    e.noise.reference_intervals = [desc["reference_interval"]]
    e.noise.auto_prop = True
    e.plot_yscale = []          # auto-scale the diagnostic EMG figures: the raw stage is
                                # dominated by the R-waves, the conditioned stages are ~5× smaller
    s.output.folder = output_folder
    return s

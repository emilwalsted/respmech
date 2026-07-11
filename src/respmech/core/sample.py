"""A small but physiologically realistic synthetic recording, generated on demand for
the "Explore sample data" onboarding door (feature P23).

Real recordings are large and cannot ship in the package, so instead we synthesise a
short breathing trace from a simple respiratory-mechanics model whenever a first-time
user asks to explore. The layout matches a typical LabChart export so the standard
channel mapping applies:

    col 1 time · 2–4 EMG · 5 flow · 6 volume · 7 Poes · 8 Pgas · 9 Pdi

The model is what makes the Campbell (Poes–Volume) loop look real. Oesophageal
pressure is the sum of an **elastic** term (proportional to lung volume) and a
**resistive** term (proportional to flow):

    Poes(t) = Poes₀ − V/C + R·flow

Because inspiratory and expiratory flow have opposite sign, at any given volume the
two limbs sit at different pressures, so the loop **opens** — its enclosed area is the
resistive work of breathing. A modest airway resistance (as in mildly loaded
breathing) is used so the loop is clearly visible. Gastric pressure rises with
inspiration (diaphragm descent); transdiaphragmatic pressure Pdi = Pgas − Poes.
Diaphragm EMG bursts during inspiration, band-limited to the EMG range.

Deterministic (fixed seed) so the sample is identical every time. Computation only —
no file I/O beyond the single CSV it writes, and no Qt.
"""
from __future__ import annotations

import os

import numpy as np

FS = 500                       # Hz
N_BREATHS = 8
PERIOD_S = 4.0                 # ~15 breaths/min
VT_L = 0.6                     # tidal volume (L)
RESISTANCE = 6.5               # cmH₂O/(L/s) → a visibly open (mildly loaded) loop
POES_EE = -5.0                 # end-expiratory pleural pressure (cmH₂O)
PGAS_EE = 8.0                  # end-expiratory gastric pressure (cmH₂O)
ELASTIC_SWING = 8.0            # cmH₂O elastic recoil over a tidal breath
FILENAME = "sample_recording.csv"


def _physio_noise(n, rng, amp, smooth_s=0.35):
    """Low-frequency, physiological-looking noise (a smooth wander), not white fuzz:
    white noise low-passed by a Hann window so it varies over ~a third of a second
    rather than sample-to-sample, then scaled to ``amp`` (cmH₂O). This is what gives the
    Campbell loops natural breath-to-breath spread instead of a hairline of jitter."""
    w = np.hanning(max(3, int(smooth_s * FS)))
    w = w / w.sum()
    x = np.convolve(rng.standard_normal(n + w.size), w, mode="same")[:n]
    return amp * x / (x.std() or 1.0)


def _one_breath(rng, T, vt):
    """One breath on the sin² volume model (segments reliably), with Poes built from an
    elastic term (∝ volume) plus a resistive term (∝ flow) so the loop opens. Pressure
    noise is added later, continuously across the whole recording (see _signals); the EMG
    carrier is genuinely high-frequency, so its measurement noise stays white here."""
    n = int(round(T * FS))
    t = np.arange(n) / FS
    vol = vt * np.sin(np.pi * t / T) ** 2                          # 0 → VT → 0
    dvdt = vt * (np.pi / T) * np.sin(2 * np.pi * t / T)            # +insp… −exp
    flow = -dvdt                                                   # inspiration → flow negative

    # Poes = baseline − elastic recoil (∝ volume) + resistive (∝ flow). The resistive
    # term has opposite sign on the two limbs, which is what opens the Campbell loop.
    poes = POES_EE - (ELASTIC_SWING / vt) * vol + RESISTANCE * flow
    pgas = PGAS_EE + (3.0 / vt) * vol

    env = np.clip(np.sin(np.pi * t / T), 0, None)                 # inspiratory-weighted burst
    emg = []
    for ch in range(3):
        carrier = np.sin(2 * np.pi * (70 + 20 * ch) * t) + 0.6 * np.sin(2 * np.pi * (140 + 25 * ch) * t)
        emg.append(0.03 * (1.0 + 0.2 * ch) * env * carrier + 0.006 * rng.standard_normal(n))
    return vol, flow, poes, pgas, emg


def _signals():
    rng = np.random.default_rng(20260711)
    vol_all, flow_all, poes_all, pgas_all, emg_all = [], [], [], [], [[], [], []]
    # a short resting expiratory lead-in so the first breath starts cleanly
    nlead = int(0.6 * FS)
    tl = np.arange(nlead) / FS
    vol_all.append(np.zeros(nlead))
    flow_all.append(0.3 * np.sin(np.pi * tl / (nlead / FS)))       # gentle expiratory hump
    poes_all.append(np.full(nlead, POES_EE))
    pgas_all.append(np.full(nlead, PGAS_EE))
    for ch in range(3):
        emg_all[ch].append(0.006 * rng.standard_normal(nlead))
    for b in range(N_BREATHS):
        T = PERIOD_S * (1.0 + 0.06 * np.sin(1.3 * b))               # breath-to-breath variation
        vt = VT_L * (1.0 + 0.10 * np.sin(0.7 * b + 0.5))
        vol, flow, poes, pgas, emg = _one_breath(rng, T, vt)
        vol_all.append(vol); flow_all.append(flow); poes_all.append(poes); pgas_all.append(pgas)
        for ch in range(3):
            emg_all[ch].append(emg[ch])
    vol = np.concatenate(vol_all); flow = np.concatenate(flow_all)
    poes = np.concatenate(poes_all); pgas = np.concatenate(pgas_all)
    emg = [np.concatenate(c) for c in emg_all]
    vol = vol + np.linspace(0, 0.12, len(vol))                     # slow baseline drift to correct
    # continuous low-frequency wander on the pressures (physiological variation, not fuzz)
    poes = poes + _physio_noise(len(poes), rng, amp=0.6)
    pgas = pgas + _physio_noise(len(pgas), rng, amp=0.4)
    pdi = pgas - poes                                              # transdiaphragmatic
    return flow, vol, poes, pgas, pdi, emg


def write_sample_recording(folder: str) -> dict:
    """Write the sample CSV into ``folder`` and return a descriptor with the file path,
    the 1-based channel mapping and the sampling frequency, ready to build Settings."""
    os.makedirs(folder, exist_ok=True)
    flow, vol, poes, pgas, pdi, emg = _signals()
    n = len(flow)
    time = np.arange(n) / FS
    cols = [time, emg[0], emg[1], emg[2], flow, vol, poes, pgas, pdi]
    header = "time,EMG1,EMG2,EMG3,flow,volume,poes,pgas,pdi"
    path = os.path.join(folder, FILENAME)
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="")
    return {"path": path, "folder": folder, "filename": FILENAME,
            "sampling_frequency": FS,
            "mapping": {"flow": 5, "volume": 6, "poes": 7, "pgas": 8, "pdi": 9,
                        "emg": [2, 3, 4], "entropy": []}}

"""A small synthetic recording, generated on demand for the "Explore sample data"
onboarding door (feature P23).

Real recordings are large and cannot ship in the package, so instead we synthesise a
short, physiologically-shaped breathing trace (a sin² volume model, its analytic flow,
plausible oesophageal/gastric pressures, and band-limited EMG) whenever a first-time
user asks to explore. The layout matches a typical LabChart export so the standard
channel mapping applies:

    col 1 time · 2–4 EMG · 5 flow · 6 volume · 7 Poes · 8 Pgas · 9 Pdi

Deterministic (fixed seed) so the sample is identical every time. Computation only —
no file I/O beyond the single CSV it writes, and no Qt.
"""
from __future__ import annotations

import os

import numpy as np

FS = 500                      # Hz — modest, so the sample file stays small
N_BREATHS = 6
PERIOD_S = 3.0
FILENAME = "sample_recording.csv"


def _signals():
    rng = np.random.default_rng(20260711)
    flow_all, vol_all, poes_all, pgas_all, emg_all = [], [], [], [], [[], [], []]
    for b in range(N_BREATHS):
        amp = 1.2 * (1.0 + 0.08 * np.sin(0.7 * b))       # mild per-breath variation
        T = PERIOD_S * (1.0 + 0.05 * np.sin(1.3 * b))
        n = int(round(T * FS))
        t = np.arange(n) / FS
        vol = amp * np.sin(np.pi * t / T) ** 2
        dvdt = amp * (np.pi / T) * np.sin(2 * np.pi * t / T)
        flow = -dvdt                                     # inspiration: volume up → flow negative
        poes = -8.0 * np.sin(np.pi * t / T) ** 2 - 5.0 + 0.15 * np.sin(3.1 * np.pi * t / T)
        pgas = 6.0 * np.sin(np.pi * t / T) ** 2 + 8.0 + 0.10 * np.cos(2.3 * np.pi * t / T)
        insp_env = np.clip(np.sin(np.pi * t / T), 0, None)
        flow_all.append(flow); vol_all.append(vol); poes_all.append(poes); pgas_all.append(pgas)
        for ch in range(3):
            carrier = np.sin(2 * np.pi * (80 + 15 * ch) * t)
            noise = 0.05 * rng.standard_normal(n)
            emg_all[ch].append(0.02 * (1 + ch * 0.3) * insp_env * carrier + noise * 0.01)
    flow = np.concatenate(flow_all); vol = np.concatenate(vol_all)
    poes = np.concatenate(poes_all); pgas = np.concatenate(pgas_all)
    emg = [np.concatenate(c) for c in emg_all]
    # a short expiratory lead-in so the first breath has a clean start
    nlead = int(0.6 * FS)
    tl = np.arange(nlead) / FS
    lead = lambda base, amp2: base + amp2 * np.sin(np.pi * tl / 0.6)   # noqa: E731
    flow = np.concatenate([0.4 * np.sin(np.pi * tl / 0.6), flow])
    vol = np.concatenate([np.zeros(nlead), vol])
    poes = np.concatenate([np.full(nlead, -5.0), poes])
    pgas = np.concatenate([np.full(nlead, 8.0), pgas])
    emg = [np.concatenate([0.01 * np.random.default_rng(ch).standard_normal(nlead), e])
           for ch, e in enumerate(emg)]
    vol = vol + np.linspace(0, 0.15, len(vol))           # a little drift to correct
    pdi = pgas - poes                                    # transdiaphragmatic = gastric − oesophageal
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

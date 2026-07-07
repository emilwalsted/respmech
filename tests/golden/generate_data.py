#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic synthetic data generator for RespMech golden/characterisation tests.

This produces reproducible, physiologically-plausible multi-channel respiratory
recordings (flow, volume, oesophageal/gastric/transdiaphragmatic pressure and a
few EMG channels) as CSV files. The data is NOT clinical data — its only purpose
is to exercise the real calculation paths in respmech.py so that the exact numeric
output can be frozen as a golden reference. A future refactor must reproduce the
same numbers (within a documented float tolerance) on the same input.

Everything here is fully deterministic (fixed RNG seed baked into the committed
CSV), so the golden reference is stable across machines.

Channel layout (1-based column numbers, as RespMech settings expect):
    1:  time (seconds)     - informational, not read by RespMech
    2:  EMG1
    3:  EMG2
    4:  EMG3
    5:  flow               - L/s, negative = inspiration, positive = expiration
    6:  volume             - L, inspired volume (positive)
    7:  poes               - cmH2O, oesophageal pressure
    8:  pgas               - cmH2O, gastric pressure
    9:  pdi                - cmH2O, transdiaphragmatic pressure (= pgas - poes)
    10: ENT1               - independent channel for entropy (disjoint from EMG)
    11: ENT2
    12: ENT3

Entropy channels are kept DISJOINT from the EMG channels on purpose: the current
respmech.py leaves entropycolumns untrimmed while indexing them with trimmed-
coordinate breath boundaries, and additionally overwrites overlapping entropy/EMG
columns. Overlapping the two therefore crashes / misaligns in the current code
(documented as a known issue). Disjoint channels exercise the entropy path
deterministically without tripping that latent bug.
"""
import os
import numpy as np

FS = 1000  # sampling frequency (Hz)


def _breath_waveforms(rng, n_breaths, period_s, vt_l, drift_l):
    """Build concatenated multi-breath signals using a sin^2 volume model.

    Volume over one breath of length T:  V(t) = VT * sin^2(pi * t / T)
      -> V=0 at t=0, peak VT at T/2, back to 0 at T.
    Flow = -dV/dt  (inspiration: volume rising -> flow negative).
    """
    flow_all, vol_all, poes_all, pgas_all = [], [], [], []
    emg_all = [[], [], []]
    ent_all = [[], [], []]

    for b in range(n_breaths):
        # Mild deterministic per-breath variation so breaths are not identical.
        amp = vt_l * (1.0 + 0.08 * np.sin(0.7 * b))
        T = period_s * (1.0 + 0.05 * np.sin(1.3 * b))
        n = int(round(T * FS))
        t = np.arange(n) / FS

        vol = amp * np.sin(np.pi * t / T) ** 2
        # analytic derivative of amp*sin^2(pi t/T)
        dvdt = amp * (np.pi / T) * np.sin(2 * np.pi * t / T)
        flow = -dvdt  # inspiration negative

        # Oesophageal pressure: falls (more negative) during inspiration.
        # Base swing ~ -8 cmH2O at peak inspiration, small end-expiratory level.
        poes = -8.0 * np.sin(np.pi * t / T) ** 2 - 5.0
        poes = poes + 0.15 * np.sin(3.1 * np.pi * t / T)  # small ripple

        # Gastric pressure: rises during expiration.
        pgas = 6.0 * np.sin(np.pi * t / T) ** 2 + 8.0
        pgas = pgas + 0.10 * np.cos(2.3 * np.pi * t / T)

        # Small deterministic measurement noise for realistic entropy/RMS.
        poes = poes + rng.normal(0, 0.05, n)
        pgas = pgas + rng.normal(0, 0.05, n)

        # EMG channels: activity bursts during inspiration + baseline noise.
        insp_env = np.clip(np.sin(np.pi * t / T), 0, None)
        for ch in range(3):
            carrier = np.sin(2 * np.pi * (80 + 15 * ch) * t)
            burst = (0.02 + 0.004 * ch) * insp_env * carrier
            noise = rng.normal(0, 0.002 + 0.0005 * ch, n)
            emg_all[ch].append(burst + noise)

        # Independent entropy channels: structured oscillation + noise so that
        # sample entropy is finite and non-trivial.
        for ch in range(3):
            struct = 0.5 * np.sin(2 * np.pi * (5 + 2 * ch) * t) \
                + 0.3 * np.sin(2 * np.pi * (11 + 3 * ch) * t)
            ent_all[ch].append(struct + rng.normal(0, 0.1, n))

        flow_all.append(flow)
        vol_all.append(vol)
        poes_all.append(poes)
        pgas_all.append(pgas)

    flow = np.concatenate(flow_all)
    vol = np.concatenate(vol_all)
    poes = np.concatenate(poes_all)
    pgas = np.concatenate(pgas_all)
    emg = [np.concatenate(c) for c in emg_all]
    ent = [np.concatenate(c) for c in ent_all]

    # Add a slow linear volume drift so correctvolumedrift has something to do.
    N = len(vol)
    vol = vol + np.linspace(0, drift_l, N)

    return flow, vol, poes, pgas, emg, ent


def make_file(path, seed, n_breaths, period_s=3.0, vt_l=1.2, drift_l=0.15,
              lead_expiration_s=0.3):
    """Write one synthetic recording to CSV.

    A short positive-flow lead-in (tail of an expiration) is prepended so the
    RespMech trim() step (which starts at the first flow<=0 and ends at the last
    flow>=0) has a clean leading expiration to trim away, per the tool's
    'start on last part of an expiration' data requirement.
    """
    rng = np.random.default_rng(seed)
    flow, vol, poes, pgas, emg, ent = _breath_waveforms(rng, n_breaths, period_s, vt_l, drift_l)

    # Lead-in: short expiration (positive flow, decaying volume back toward 0).
    nlead = int(round(lead_expiration_s * FS))
    tl = np.arange(nlead) / FS
    lead_flow = 0.4 * np.sin(np.pi * tl / lead_expiration_s)  # positive hump
    lead_vol = 0.05 * np.cos(np.pi * tl / (2 * lead_expiration_s))
    lead_poes = -5.0 + 0.05 * rng.normal(0, 1, nlead)
    lead_pgas = 8.0 + 0.05 * rng.normal(0, 1, nlead)
    lead_emg = [rng.normal(0, 0.002 + 0.0005 * ch, nlead) for ch in range(3)]
    lead_ent = [rng.normal(0, 0.1, nlead) for ch in range(3)]

    flow = np.concatenate([lead_flow, flow])
    vol = np.concatenate([lead_vol, vol])
    poes = np.concatenate([lead_poes, poes])
    pgas = np.concatenate([lead_pgas, pgas])
    emg = [np.concatenate([lead_emg[ch], emg[ch]]) for ch in range(3)]
    ent = [np.concatenate([lead_ent[ch], ent[ch]]) for ch in range(3)]

    pdi = pgas - poes
    N = len(flow)
    time = np.arange(N) / FS

    header = "time,EMG1,EMG2,EMG3,flow,volume,poes,pgas,pdi,ENT1,ENT2,ENT3"
    data = np.column_stack([time, emg[0], emg[1], emg[2], flow, vol, poes, pgas, pdi,
                            ent[0], ent[1], ent[2]])
    np.savetxt(path, data, delimiter=",", header=header, comments="",
               fmt="%.10g")
    return N


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    indir = os.path.join(here, "input")
    os.makedirs(indir, exist_ok=True)

    n1 = make_file(os.path.join(indir, "synth_case_A.csv"), seed=12345, n_breaths=8)
    n2 = make_file(os.path.join(indir, "synth_case_B.csv"), seed=67890, n_breaths=6)
    print(f"Wrote synth_case_A.csv ({n1} samples), synth_case_B.csv ({n2} samples)")

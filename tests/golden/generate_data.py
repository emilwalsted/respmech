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


def _emg_only_waveforms(rng, n_segments, period_s):
    """Concatenated EMG-only bursts (no flow/pressure channel at all — there is
    nothing to segment BY, unlike ``_breath_waveforms`` above, which is exactly the
    shape the ``whole_file``/``separators`` EMG-only segmentation methods exist for).
    One smooth burst per segment over a low tonic floor, on 3 channels, so RMS/integrated-
    EMG have something non-trivial to measure. Returns the concatenated channels and
    the ``n_segments - 1`` internal segment-to-segment boundaries (seconds, relative
    to the start of THIS waveform — the caller adds the lead-in offset)."""
    emg_all = [[], [], []]
    onsets = []
    t_cursor = 0.0
    for b in range(n_segments):
        T = period_s * (1.0 + 0.06 * np.sin(1.1 * b))
        n = int(round(T * FS))
        t = np.arange(n) / FS
        env = np.sin(np.pi * t / T) ** 2                # smooth burst, peak mid-segment
        for ch in range(3):
            carrier = np.sin(2 * np.pi * (70 + 10 * ch) * t)
            burst = (0.03 + 0.005 * ch) * env * carrier
            tonic = (0.004 + 0.0005 * ch) * (1.0 - env) * carrier   # low floor between bursts
            noise = rng.normal(0, 0.003 + 0.0005 * ch, n)
            emg_all[ch].append(burst + tonic + noise)
        t_cursor += T
        if b < n_segments - 1:
            onsets.append(t_cursor)
    emg = [np.concatenate(c) for c in emg_all]
    return emg, onsets


def make_emgonly_file(path, seed, n_segments, period_s=2.0, lead_s=0.4):
    """Write one EMG-only synthetic recording (time + 3 EMG channels, no flow/pressure
    columns at all) — a DEDICATED input for the ``emg_only_whole_file``/
    ``emg_only_separators`` golden scenarios, on its OWN RNG stream (``seed`` is never
    one of ``make_file``'s own seeds above) and its own ``synth_emgonly_*.csv`` naming,
    so it can never be mistaken for (or accidentally match the glob of) the flow-
    bearing ``synth_case_*.csv`` files every other scenario uses.

    A short quiet lead-in (noise only, no burst) precedes the first segment, mirroring
    ``make_file``'s own lead-in convention above — though here it is not trimmed away
    (EMG-only segmentation has no ``trim()`` step at all), it just makes the first
    segment's own burst start visibly after silence, like a real recording's settling
    period. Returns the sample count and the internal segment-boundary times (seconds,
    already offset by the lead-in) for the caller to use as ``separators`` times."""
    rng = np.random.default_rng(seed)
    emg, onsets = _emg_only_waveforms(rng, n_segments, period_s)
    nlead = int(round(lead_s * FS))
    lead_emg = [rng.normal(0, 0.003 + 0.0005 * ch, nlead) for ch in range(3)]
    emg = [np.concatenate([lead_emg[ch], emg[ch]]) for ch in range(3)]
    onsets = [round(lead_s + o, 6) for o in onsets]
    N = len(emg[0])
    time = np.arange(N) / FS
    header = "time,EMG1,EMG2,EMG3"
    data = np.column_stack([time, emg[0], emg[1], emg[2]])
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")
    return N, onsets


def _manoeuvre_breath(*, insp_ramp_n, insp_ramp_flow, insp_plateau_n, insp_plateau_flow,
                      vol_base, vol_peak, exp_ramp_n, exp_ramp_flow,
                      exp_plateau_n, exp_plateau_flow, poes_swing, pgas_bump):
    """Build one breath's flow/volume/poes/pgas arrays as explicit TRAPEZOIDS
    (piecewise-constant flow, linearly-ramped volume/poes/pgas) — a deliberately
    different shape from ``_breath_waveforms``' smooth sin^2 model above, built for the
    ``typed_ic_fvc_same_file`` golden scenario, whose own test checks an
    extracted value (``vol_ic``) against a literal analytical constant, not merely
    against a previously committed reference.

    WHY a plateau, not just a ramp: ``compute.separateintobreathsbyflow`` (the sign-walk
    every ``method = "flow"`` golden scenario uses) drops exactly one sample at each
    phase boundary — ``inend = i - 1`` is the index of the true last inspiratory sample,
    but ``_phase_dicts`` slices ``volume[instart:inend]``, a Python half-open range that
    EXCLUDES index ``inend`` itself; the mirror-image drop happens at ``exend`` on the
    expiration side. Only a phase's LAST sample is ever dropped this way — its FIRST
    sample (``volume[instart]``/``volume[exstart]``) always survives untouched, since a
    half-open slice always includes its own start. This asymmetry is why the two
    plateaus below are not equally load-bearing:

    * ``vol_peak`` (the manoeuvre's peak volume, what ``vol_ic`` reads via
      ``max(breath['volume'])``) sits at the insp/exp SEAM: it survives bit-exact via
      the EXPIRATION phase's own leading sample (``np.linspace(vol_peak, vol_base,
      exp_ramp_n)[0] == vol_peak``, a leading sample that is never dropped) regardless
      of whether ``insp_plateau_n`` is 0 or not. ``insp_plateau_n`` (a short hold at
      ``vol_peak``, when present) is therefore a physiological-realism touch — a
      plausible momentary breath-hold at TLC, and it gives ``manoeuvres.py`` a genuine,
      non-zero ``ic_plateau_s`` to report — not what makes ``vol_ic`` exact.
    * ``vol_base`` (the level ``ic_eelv_pre`` reads back OUT of a PRECEDING tidal
      breath, via that breath's own ``_arr(b['volume'])[-1]`` — its own LAST retained
      sample) genuinely NEEDS ``exp_plateau_n`` >= 2: without it, the one sample that
      gets dropped from that breath's own expiration IS its ramp's exact-``vol_base``
      endpoint, leaving the ramp's second-to-last point (off by one ``linspace``
      increment, e.g. ~0.5 mL for a tidal breath) as the value ``ic_eelv_pre`` would
      average over. A plateau of >= 2 identical ``vol_base`` samples survives the drop
      with at least one copy intact.

    Both plateaus use a flow MAGNITUDE small enough (1e-3 L/s, well under
    ``IcSettings.plateau_flow_lps``'s 0.1 default) to register as a genuine plateau,
    but with the SIGN needed to keep the sign-walk inside the correct phase (negative
    throughout inspiration, positive throughout expiration) — the walk is driven by a
    plain ``flow[i] < 0``/``flow[i] > 0`` test, so this never depends on its
    forward-window mean.

    This is what lets the golden scenario assert ``vol_ic`` against a bare Python float
    (``3.0``) and ``ic_eelv_pre`` against ``0.0`` to ``abs_tol=1e-9``: neither value is
    merely "close to" its analytical target, each IS that same double — surviving via
    the mechanisms above — and the CSV round-trip (``%.10g``, ample precision for a
    round decimal literal like ``3.0``/``0.0``) and the trim/zero/drift-correct chain (a
    no-op here: ``compute.zero`` subtracts the first trimmed sample, this file's own
    breath #1 onset, which is 0.0; ``compute.correctdrift``'s slope is
    ``(volume[-1] - volume[0]) / n``, exactly 0 because both ends of the WHOLE
    recording sit on a ``vol_base`` plateau) never perturb either.

    ``insp_plateau_n``/``exp_plateau_n`` may be 0 (an ordinary tidal breath's own peak
    is never read exactly, and the FVC breath's inspiration — the actual FVC numerics
    are a later feature's scope, not this generator's — needs no exact peak either).
    """
    insp_n = insp_ramp_n + insp_plateau_n
    exp_n = exp_ramp_n + exp_plateau_n
    insp = {
        "flow": np.concatenate([np.full(insp_ramp_n, float(insp_ramp_flow)),
                                np.full(insp_plateau_n, float(insp_plateau_flow))]),
        "volume": np.concatenate([np.linspace(vol_base, vol_peak, insp_ramp_n),
                                  np.full(insp_plateau_n, float(vol_peak))]),
        "poes": np.linspace(0.0, -poes_swing, insp_n),
        "pgas": np.linspace(0.0, pgas_bump, insp_n),
    }
    exp = {
        "flow": np.concatenate([np.full(exp_ramp_n, float(exp_ramp_flow)),
                                np.full(exp_plateau_n, float(exp_plateau_flow))]),
        "volume": np.concatenate([np.linspace(vol_peak, vol_base, exp_ramp_n),
                                  np.full(exp_plateau_n, float(vol_base))]),
        "poes": np.linspace(-poes_swing, 0.0, exp_n),
        "pgas": np.linspace(pgas_bump, 0.0, exp_n),
    }
    return insp, exp


#: Breath recipes for ``make_manoeuvre_file`` below. A TIDAL breath has no peak
#: plateau (its own peak volume is never checked exactly) but DOES get the trailing
#: zero-plateau (its END-expiratory volume is what ``ic_eelv_pre`` averages over — see
#: ``_manoeuvre_breath``'s docstring for why that needs to survive the boundary drop
#: too). VT_IC = 3.0 L matches the convention already established by
#: ``tests/unit/test_manoeuvres.py``'s own analytical IC test ("EELV 0 til 3.0 L").
_MANOEUVRE_TIDAL = dict(insp_ramp_n=1000, insp_ramp_flow=-0.5, insp_plateau_n=0, insp_plateau_flow=0.0,
                        vol_base=0.0, vol_peak=0.5, exp_ramp_n=980, exp_ramp_flow=0.5,
                        exp_plateau_n=20, exp_plateau_flow=1e-3, poes_swing=5.0, pgas_bump=3.0)
_MANOEUVRE_IC = dict(insp_ramp_n=2000, insp_ramp_flow=-2.0, insp_plateau_n=20, insp_plateau_flow=-1e-3,
                     vol_base=0.0, vol_peak=3.0, exp_ramp_n=1500, exp_ramp_flow=1.5,
                     exp_plateau_n=20, exp_plateau_flow=1e-3, poes_swing=20.0, pgas_bump=10.0)
#: FVC breath: a tidal-ish inspiration (no peak plateau — the actual FVC/FEV1/PEF
#: numerics are a later feature's scope, not this generator's) followed by a long forced expiration
#: (2.5 s ramp + trailing plateau, comfortably over manoeuvres.py's own
#: ``_FVC_MIN_DURATION_S = 1.0`` threshold even after the one-sample boundary drop).
_MANOEUVRE_FVC = dict(insp_ramp_n=1000, insp_ramp_flow=-2.0, insp_plateau_n=0, insp_plateau_flow=0.0,
                      vol_base=0.0, vol_peak=2.0, exp_ramp_n=2500, exp_ramp_flow=3.0,
                      exp_plateau_n=20, exp_plateau_flow=1e-3, poes_swing=15.0, pgas_bump=8.0)

#: The engine numbers breaths sequentially from 1 as it segments a recording (see
#: ``compute.separateintobreathsbyflow``): 3 tidal breaths, then IC, then 2 tidal, then
#: FVC, then 2 tidal — so the IC breath is always #4 and the FVC breath always #7 in
#: ``synth_manoeuvre_A.csv``, matching ``scenarios/typed_ic_fvc_same_file.toml``'s
#: ``[[processing.breath_types]]`` entries. Keep the two in sync if this sequence ever
#: changes.
MANOEUVRE_IC_BREATH_NO = 4
MANOEUVRE_FVC_BREATH_NO = 7
_MANOEUVRE_SEQUENCE = (
    _MANOEUVRE_TIDAL, _MANOEUVRE_TIDAL, _MANOEUVRE_TIDAL,
    _MANOEUVRE_IC,
    _MANOEUVRE_TIDAL, _MANOEUVRE_TIDAL,
    _MANOEUVRE_FVC,
    _MANOEUVRE_TIDAL, _MANOEUVRE_TIDAL,
)


def make_manoeuvre_file(path, seed, lead_expiration_s=0.3):
    """Write the dedicated ``synth_manoeuvre_*.csv`` input for the
    ``typed_ic_fvc_same_file`` golden scenario — its own RNG stream (``seed`` is
    never one of ``make_file``'s/``make_emgonly_file``'s own seeds above), its own
    ``synth_manoeuvre_*.csv`` naming (never matching the ``synth_case_*.csv``/
    ``synth_emgonly_*.csv`` globs every other scenario's input uses), and its own
    TRAPEZOID breath shapes (see ``_manoeuvre_breath``) rather than ``make_file``'s
    smooth sin^2 model — a flow-bearing recording (full channel set: flow, volume,
    poes, pgas, pdi, 3 EMG, 3 entropy) with ``_MANOEUVRE_SEQUENCE`` above: 3 tidal
    breaths, an inspiratory-capacity manoeuvre (breath #4), 2 more tidal, a
    forced-vital-capacity manoeuvre (breath #7), then 2 trailing tidal breaths.

    Same lead-in convention as ``make_file`` (a short decaying positive-flow hump so
    ``compute.trim()`` has a clean leading expiration to cut away, leaving breath #1
    starting exactly at this file's own t=0) — its own volume/pressure values are
    irrelevant, since ``trim()`` discards the entire lead-in.
    """
    rng = np.random.default_rng(seed)

    nlead = int(round(lead_expiration_s * FS))
    tl = np.arange(nlead) / FS
    lead_flow = 0.4 * np.sin(np.pi * tl / lead_expiration_s)
    lead_vol = 0.05 * np.cos(np.pi * tl / (2 * lead_expiration_s))
    lead_poes = -5.0 + 0.05 * rng.normal(0, 1, nlead)
    lead_pgas = 8.0 + 0.05 * rng.normal(0, 1, nlead)

    flow_parts, vol_parts, poes_parts, pgas_parts = [lead_flow], [lead_vol], [lead_poes], [lead_pgas]
    for cfg in _MANOEUVRE_SEQUENCE:
        insp, exp = _manoeuvre_breath(**cfg)
        for parts, key in ((flow_parts, "flow"), (vol_parts, "volume"),
                          (poes_parts, "poes"), (pgas_parts, "pgas")):
            parts.append(insp[key])
            parts.append(exp[key])

    flow = np.concatenate(flow_parts)
    vol = np.concatenate(vol_parts)
    poes = np.concatenate(poes_parts)
    pgas = np.concatenate(pgas_parts)
    pdi = pgas - poes
    N = len(flow)
    emg = [rng.normal(0, 0.002 + 0.0005 * ch, N) for ch in range(3)]
    ent = [rng.normal(0, 0.1, N) for ch in range(3)]
    time = np.arange(N) / FS

    header = "time,EMG1,EMG2,EMG3,flow,volume,poes,pgas,pdi,ENT1,ENT2,ENT3"
    data = np.column_stack([time, emg[0], emg[1], emg[2], flow, vol, poes, pgas, pdi,
                            ent[0], ent[1], ent[2]])
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")
    return N


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    indir = os.path.join(here, "input")
    os.makedirs(indir, exist_ok=True)

    n1 = make_file(os.path.join(indir, "synth_case_A.csv"), seed=12345, n_breaths=8)
    n2 = make_file(os.path.join(indir, "synth_case_B.csv"), seed=67890, n_breaths=6)
    print(f"Wrote synth_case_A.csv ({n1} samples), synth_case_B.csv ({n2} samples)")

    n3, onsets_a = make_emgonly_file(
        os.path.join(indir, "synth_emgonly_A.csv"), seed=13579, n_segments=4)
    n4, onsets_b = make_emgonly_file(
        os.path.join(indir, "synth_emgonly_B.csv"), seed=24680, n_segments=3)
    print(f"Wrote synth_emgonly_A.csv ({n3} samples, separators {onsets_a}), "
         f"synth_emgonly_B.csv ({n4} samples, separators {onsets_b})")

    n5 = make_manoeuvre_file(os.path.join(indir, "synth_manoeuvre_A.csv"), seed=90210)
    print(f"Wrote synth_manoeuvre_A.csv ({n5} samples, IC=breath #{MANOEUVRE_IC_BREATH_NO}, "
         f"FVC=breath #{MANOEUVRE_FVC_BREATH_NO})")

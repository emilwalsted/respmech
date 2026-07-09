"""Shared-profile EMG noise reduction (spectral gating).

Design constraint (mathematically required): **the same transformation must be
applied to every file of a test**, because EMG is quantified against a
patient+test-specific maximum. So a single :class:`NoiseProfile` (fixed STFT
parameters + a precomputed per-frequency noise threshold) is built **once** from a
rest reference and applied **identically** to every file — never re-tuned per file.

This replaces the legacy `emg.reducenoise`, which (1) tied `n_fft`/`win_length` to
the noise-clip length (`len(noise)**2` — a degenerate, unstable estimate) and (2)
recomputed the noise statistics inside every call, so the transformation could
differ between files. See ``docs/NOISE_ECG_OPTIMIZATION.md``.

The gate math is the established Sainburg spectral-gating method (as used before),
but driven by a fixed, serialisable threshold spectrum.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy as sp
from scipy import signal as _sig

from respmech.core.emg import _amp_to_db, _db_to_amp, _stft, _istft
from respmech.core._compat import complex_sign_v1, trapezoid

# Default fixed STFT parameters (decoupled from the noise-clip length — the bug fix).
DEFAULT_N_FFT = 256
DEFAULT_HOP = 64
DEFAULT_WIN = 256
DEFAULT_N_STD = 1.0
DEFAULT_PROP = 0.6
DEFAULT_FIDELITY_TARGET = 0.8
EMG_BAND = (20.0, 250.0)


class NoiseProfileError(ValueError):
    pass


def _smoothing_filter(n_grad_freq: int, n_grad_time: int) -> np.ndarray:
    f = np.outer(
        np.concatenate([np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                        np.linspace(1, 0, n_grad_freq + 2)])[1:-1],
        np.concatenate([np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                        np.linspace(1, 0, n_grad_time + 2)])[1:-1],
    )
    return f / np.sum(f)


@dataclass
class NoiseProfile:
    """A fixed, serialisable per-channel noise-gate profile. Built once, applied
    identically to every file's matching channel."""
    n_fft: int
    hop_length: int
    win_length: int
    n_std_thresh: float
    n_grad_freq: int
    n_grad_time: int
    sr: int
    noise_thresh: np.ndarray = field(repr=False)   # per-frequency threshold (dB)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_clip(cls, noise_clip, sr, *, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP,
                  win_length=DEFAULT_WIN, n_std_thresh=DEFAULT_N_STD,
                  n_grad_freq=0, n_grad_time=4) -> "NoiseProfile":
        noise_clip = np.asarray(noise_clip, dtype=float)
        if len(noise_clip) < n_fft:
            raise NoiseProfileError(
                f"Noise reference is too short ({len(noise_clip)} samples) for a stable "
                f"estimate at n_fft={n_fft}. Provide a longer EMG-free reference "
                f"(≥ several STFT windows).")
        n_frames = 1 + (len(noise_clip) - win_length) // hop_length
        if n_frames < 8:
            warnings.warn(
                f"Noise reference gives only {n_frames} STFT frames; ≥ ~8 recommended "
                "for a stable per-frequency std. Use a longer EMG-free reference.")
        noise_db = _amp_to_db(np.abs(_stft(noise_clip, n_fft, hop_length, win_length)))
        thresh = np.mean(noise_db, axis=1) + np.std(noise_db, axis=1) * n_std_thresh
        return cls(n_fft, hop_length, win_length, n_std_thresh, n_grad_freq,
                   n_grad_time, sr, thresh)

    # -- application (deterministic; identical operator for every file) -----
    def apply(self, signal, prop_decrease: float):
        signal = np.asarray(signal, dtype=float)
        n = len(signal)
        sig_stft = _stft(signal, self.n_fft, self.hop_length, self.win_length)
        sig_db = _amp_to_db(np.abs(sig_stft))
        mask_gain_db = np.min(sig_db)
        smooth = _smoothing_filter(self.n_grad_freq, self.n_grad_time)
        db_thresh = np.repeat(np.reshape(self.noise_thresh, [1, len(self.noise_thresh)]),
                              np.shape(sig_db)[1], axis=0).T
        mask = sig_db < db_thresh
        mask = _sig.fftconvolve(mask, smooth, mode="same") * prop_decrease
        sig_db_masked = sig_db * (1 - mask) + np.ones(np.shape(mask_gain_db)) * mask_gain_db * mask
        sig_imag_masked = np.imag(sig_stft) * (1 - mask)
        sig_amp = (_db_to_amp(sig_db_masked) * complex_sign_v1(sig_stft)) + 1j * sig_imag_masked
        rec = np.asarray(_istft(sig_amp, self.n_fft, self.hop_length, self.win_length))
        if len(rec) < n:
            rec = np.pad(rec, (0, n - len(rec)))
        return rec[:n]

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"n_fft": self.n_fft, "hop_length": self.hop_length,
                "win_length": self.win_length, "n_std_thresh": self.n_std_thresh,
                "n_grad_freq": self.n_grad_freq, "n_grad_time": self.n_grad_time,
                "sr": self.sr, "noise_thresh": np.asarray(self.noise_thresh).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseProfile":
        return cls(d["n_fft"], d["hop_length"], d["win_length"], d["n_std_thresh"],
                   d["n_grad_freq"], d["n_grad_time"], d["sr"],
                   np.asarray(d["noise_thresh"], dtype=float))


@dataclass
class NoiseProfileSet:
    """One :class:`NoiseProfile` per EMG channel + the chosen `prop_decrease`, built
    once and applied identically to every file in the test."""
    profiles: list          # list[NoiseProfile], one per EMG channel
    prop_decrease: float

    def apply_columns(self, emgcols: np.ndarray) -> np.ndarray:
        emgcols = np.asarray(emgcols, dtype=float)
        if emgcols.shape[1] != len(self.profiles):
            raise NoiseProfileError(
                f"channel count {emgcols.shape[1]} != number of profiles {len(self.profiles)}")
        out = np.empty_like(emgcols)
        for i, prof in enumerate(self.profiles):
            out[:, i] = prof.apply(emgcols[:, i], self.prop_decrease)
        return out

    def to_dict(self) -> dict:
        return {"prop_decrease": self.prop_decrease,
                "profiles": [p.to_dict() for p in self.profiles]}

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseProfileSet":
        return cls([NoiseProfile.from_dict(p) for p in d["profiles"]], d["prop_decrease"])


# --- quality metrics --------------------------------------------------------

def inband_power(x, sr, band=EMG_BAND):
    x = np.asarray(x, dtype=float)
    f, p = _sig.welch(x, sr, nperseg=min(1024, len(x)))
    m = (f >= band[0]) & (f <= band[1])
    return float(trapezoid(p[m], f[m]))


def fidelity(raw_active, processed_active, sr, band=EMG_BAND):
    """Fraction of in-band inspiratory EMG power retained (over-subtraction guard)."""
    denom = inband_power(raw_active, sr, band)
    if denom <= 0:
        return 1.0
    return inband_power(processed_active, sr, band) / denom


def snr_db(active, quiet, sr, band=EMG_BAND):
    return 10.0 * np.log10((inband_power(active, sr, band) + 1e-30) /
                           (inband_power(quiet, sr, band) + 1e-30))


# --- build from reference + fidelity-driven prop selection ------------------

def build_profiles(reference_cols, sr, *, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP,
                   win_length=DEFAULT_WIN, n_std_thresh=DEFAULT_N_STD,
                   n_grad_freq=0, n_grad_time=4) -> list:
    """One NoiseProfile per column of an EMG-free reference clip (shape [n, nch])."""
    reference_cols = np.asarray(reference_cols, dtype=float)
    if reference_cols.ndim == 1:
        reference_cols = reference_cols[:, None]
    return [NoiseProfile.from_clip(reference_cols[:, i], sr, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, n_std_thresh=n_std_thresh,
                                   n_grad_freq=n_grad_freq, n_grad_time=n_grad_time)
            for i in range(reference_cols.shape[1])]


def select_prop_decrease(profiles, active_cols, quiet_cols, sr, *,
                         target=DEFAULT_FIDELITY_TARGET, grid=None, band=EMG_BAND):
    """Pick — ONCE per test — the highest ``prop_decrease`` whose worst-channel
    fidelity stays ≥ ``target`` on the test's active EMG. Returns
    (prop_decrease, report) where report lists per-channel fidelity/ΔSNR.

    ``active_cols`` / ``quiet_cols``: representative inspiratory / expiratory EMG
    (ECG-removed) with one column per profile."""
    active_cols = np.asarray(active_cols, dtype=float)
    quiet_cols = np.asarray(quiet_cols, dtype=float)
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.1, 1.001, 0.1)]
    best = None
    per_prop = {}
    for prop in sorted(grid):
        fids = []
        for i, prof in enumerate(profiles):
            proc = prof.apply(active_cols[:, i], prop)
            fids.append(fidelity(active_cols[:, i], proc, sr, band))
        per_prop[prop] = fids
        if min(fids) >= target:
            best = prop
    chosen = best if best is not None else min(grid)  # if none qualify, gentlest
    report = _prop_report(profiles, active_cols, quiet_cols, sr, chosen, band, per_prop, target)
    return chosen, report


def _prop_report(profiles, active_cols, quiet_cols, sr, prop, band, per_prop, target):
    rows = []
    for i, prof in enumerate(profiles):
        proc_a = prof.apply(active_cols[:, i], prop)
        proc_q = prof.apply(quiet_cols[:, i], prop)
        base_snr = snr_db(active_cols[:, i], quiet_cols[:, i], sr, band)
        proc_snr = snr_db(proc_a, proc_q, sr, band)
        rows.append({"channel": i, "fidelity": fidelity(active_cols[:, i], proc_a, sr, band),
                     "delta_snr_db": proc_snr - base_snr})
    return {"prop_decrease": prop, "fidelity_target": target, "channels": rows,
            "frontier": {p: [round(f, 3) for f in fids] for p, fids in per_prop.items()}}

"""Small cross-version numeric-library shims.

Keeps the code working on both the golden/reference stack (NumPy 1.x) and modern
installs (NumPy 2.x), with identical numerics.
"""
import numpy as np

# NumPy 2.0 renamed ``np.trapz`` -> ``np.trapezoid`` (same algorithm). Support both.
trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def complex_sign_v1(z):
    """``np.sign`` for complex input as NumPy < 2.0 defined it.

    NumPy 2.0 changed ``np.sign`` for complex numbers from ``sign(z.real)`` (the
    real-part sign, tie-broken by ``sign(z.imag)`` when the real part is exactly 0)
    to the true unit phasor ``z / |z|``. The Sainburg spectral-gating reconstruction
    (``noise.NoiseProfile.apply`` / ``emg.removeNoise``) was written and validated
    against the old behaviour — the phase term is deliberately the real-part sign,
    not the full phasor — so the canonical EMG golden only reproduces with it. This
    pins that behaviour on every NumPy, returning a real array (numerically identical
    downstream, where it is multiplied by a real magnitude and a ``1j`` term is added).
    """
    z = np.asarray(z)
    if not np.iscomplexobj(z):
        return np.sign(z)
    s = np.sign(z.real)
    return np.where(s != 0, s, np.sign(z.imag))

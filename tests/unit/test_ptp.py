"""Unit tests for the PTP baseline (window mean vs single sample).

Correctness is checked against an independent trapezoidal integral (the code uses
scipy.integrate.simpson; they agree closely on smooth signals). Uses the same
numpy-2-safe ``trapezoid`` shim as the core, so the test runs on numpy 1.x and 2.x."""
import numpy as np

from respmech.core._compat import trapezoid
from respmech.core.compute import calcptp

FS = 1000
Y = np.linspace(0.0, 10.0, 1001)      # smooth ramp
X = np.linspace(0.0, len(Y) / FS, len(Y))


def _trapz(sig):
    return trapezoid(sig, X)


def test_single_sample_baseline_matches_independent_integral():
    _, integral = calcptp(Y.copy(), 1, 1, samplingfreq=FS, baseline_samples=1)
    assert abs(integral - _trapz(Y - Y[0])) < 1e-3


def test_window_mean_baseline_matches_independent_integral():
    n = 100
    _, integ_win = calcptp(Y.copy(), 1, 1, samplingfreq=FS, baseline_samples=n)
    assert abs(integ_win - _trapz(Y - np.mean(Y[:n]))) < 1e-3


def test_wider_window_lowers_integral_on_rising_signal():
    _, one = calcptp(Y.copy(), 1, 1, FS, baseline_samples=1)
    _, win = calcptp(Y.copy(), 1, 1, FS, baseline_samples=100)
    # baseline moves up the ramp -> smaller area
    assert win < one


def test_window_clamped_to_length():
    y = np.array([2.0, 4.0, 6.0])
    _, integral = calcptp(y.copy(), 1, 1, samplingfreq=10, baseline_samples=999)
    assert np.isfinite(integral)

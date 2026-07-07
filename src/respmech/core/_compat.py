"""Small cross-version numeric-library shims.

Keeps the code working on both the golden/reference stack (NumPy 1.x) and modern
installs (NumPy 2.x), with identical numerics.
"""
import numpy as np

# NumPy 2.0 renamed ``np.trapz`` -> ``np.trapezoid`` (same algorithm). Support both.
trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

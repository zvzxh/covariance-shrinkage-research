"""Scaled-identity shrinkage from Ledoit and Wolf (2004), Eq. (14).

Rows are observations; columns are variables. Input means are known to be zero
unless ``center=True``. Normalization is 1/n, never the unbiased 1/(n-1).
This implementation was written from the equations, not copied from a package.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ShrinkageEstimate:
    covariance: NDArray[np.float64]
    sample_covariance: NDArray[np.float64]
    shrinkage: float
    target_variance: float


def ledoit_wolf_identity(x: ArrayLike, *, center: bool = False) -> ShrinkageEstimate:
    """Estimate covariance using a scalar-identity shrinkage target.

    The centered mode estimates a mean and retains the maximum-likelihood 1/n
    normalization. It is useful for oracle tests, but is not the paper's known-
    zero-mean simulation. A degenerate all-zero sample returns zero covariance;
    no blanket positive-definiteness guarantee is made for degenerate data.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("x must be a 2D array with at least 2 observations and 1 variable")
    if not np.isfinite(values).all():
        raise ValueError("x must contain only finite values")
    if center:
        values = values - values.mean(axis=0, keepdims=True)

    # Normalize internally to avoid squaring or taking fourth powers of the
    # original scale. Shrinkage is scale-invariant. Inputs are never modified.
    magnitude = float(np.max(np.abs(values)))
    n, p = values.shape
    if magnitude == 0.0:
        zeros = np.zeros((p, p), dtype=np.float64)
        return ShrinkageEstimate(zeros.copy(), zeros, 0.0, 0.0)
    scaled = values / magnitude
    sample = (scaled.T @ scaled) / n
    mu = float(np.trace(sample) / p)
    departure = sample - mu * np.eye(p)
    d2 = float(np.sum(departure * departure) / p)

    if p == 1 or d2 == 0.0:
        # Both endpoints coincide; the coefficient is unidentified. Choosing
        # zero preserves the covariance and matches sklearn's convention.
        intensity = 0.0
    else:
        squared_row_norm = np.einsum("ij,ij->i", scaled, scaled)
        # Equivalent to sum_k ||x_k x_k' - S||_F^2 / (p*n^2), using
        # sum_k x_k x_k' = n*S. Only O(np+p^2) memory is needed.
        b2_unclipped = float(
            (np.dot(squared_row_norm, squared_row_norm) / n
             - np.sum(sample * sample)) / (p * n)
        )
        intensity = float(np.clip(b2_unclipped / d2, 0.0, 1.0))

    covariance = (1.0 - intensity) * sample + intensity * mu * np.eye(p)
    with np.errstate(over="ignore", invalid="ignore"):
        # Multiply in stages: magnitude**2 itself may overflow even when the
        # final covariance (with its smaller coefficient) remains finite.
        covariance = (covariance * magnitude) * magnitude
        sample = (sample * magnitude) * magnitude
        target_variance = (mu * magnitude) * magnitude
    if not np.isfinite(covariance).all() or not np.isfinite(sample).all():
        raise ValueError("input magnitude exceeds representable covariance range")
    return ShrinkageEstimate(covariance, sample, intensity, target_variance)

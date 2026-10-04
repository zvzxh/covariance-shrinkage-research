"""Explicit spectrum construction for a paper-aligned simulation subset."""

import numpy as np
from numpy.typing import NDArray


def matched_lognormal_spectrum(
    p: int, dispersion: float, seed: int
) -> NDArray[np.float64]:
    """Return a positive, fixed spectrum with mean 1 and variance dispersion.

    A seeded normal vector z defines lambda_i = exp(t*z_i)/mean(exp(t*z)).
    Bisection sets its realized variance exactly. This is a disclosed design
    choice, not a claim to recover the authors' unpublished eigenvalue vector.
    The covariance is fixed across observation replications in a scenario.
    """
    if p < 2:
        raise ValueError("p must be at least 2")
    if not np.isfinite(dispersion) or not 0.0 <= dispersion < p - 1:
        raise ValueError("dispersion must satisfy 0 <= dispersion < p-1")
    if dispersion == 0.0:
        return np.ones(p)
    z = np.random.default_rng(seed).standard_normal(p)

    def spectrum(t: float) -> NDArray[np.float64]:
        exponent = t * z
        exponent -= exponent.max()
        weights = np.exp(exponent)
        return weights / weights.mean()

    left, right = 0.0, 1.0
    while np.var(spectrum(right)) < dispersion:
        right *= 2.0
        if right > 1024.0:
            raise RuntimeError("spectrum calibration failed to bracket the target")
    for _ in range(100):
        middle = (left + right) / 2.0
        if np.var(spectrum(middle)) < dispersion:
            left = middle
        else:
            right = middle
    eigenvalues = spectrum((left + right) / 2.0)
    if not (eigenvalues > 0.0).all():
        raise ValueError("requested spectrum is too extreme for float64")
    return eigenvalues


def gaussian_sample_risk(eigenvalues: NDArray[np.float64], n: int) -> float:
    """Exact E[||S-Sigma||_F^2/p] for iid N(0,Sigma) with known mean."""
    p = len(eigenvalues)
    return float((np.dot(eigenvalues, eigenvalues) + eigenvalues.sum() ** 2) / (p * n))

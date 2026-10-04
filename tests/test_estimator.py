import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.covariance import ledoit_wolf

from shrinkage_research import ledoit_wolf_identity


def test_hand_calculated_nontrivial_fixture():
    x = np.array([[3, 1], [-3, -1], [1, 0], [-1, 0]], dtype=float)
    estimate = ledoit_wolf_identity(x)
    sample = np.array([[5, 1.5], [1.5, 0.5]])
    intensity = 83 / 234
    assert_allclose(estimate.sample_covariance, sample, rtol=0, atol=1e-14)
    assert_allclose(estimate.shrinkage, intensity, rtol=0, atol=1e-14)
    assert_allclose(estimate.covariance, (1 - intensity) * sample + intensity * 2.75 * np.eye(2))


@pytest.mark.parametrize("n,p", [(40, 20), (20, 40), (8, 8), (2, 5), (50, 1)])
@pytest.mark.parametrize("center", [False, True])
def test_independent_sklearn_oracle(n, p, center):
    rng = np.random.default_rng(4723)
    x = rng.normal(size=(n, p)) * np.linspace(0.3, 3.0, p) + 0.7
    actual = ledoit_wolf_identity(x, center=center)
    expected_covariance, expected_intensity = ledoit_wolf(x, assume_centered=not center)
    assert_allclose(actual.covariance, expected_covariance, rtol=1e-11, atol=1e-12)
    assert_allclose(actual.shrinkage, expected_intensity, rtol=1e-11, atol=1e-12)


def test_literal_outer_product_formula_matches_fast_calculation():
    x = np.random.default_rng(888).normal(size=(23, 7)) * np.arange(1, 8)
    n, p = x.shape
    sample = x.T @ x / n
    mu = np.trace(sample) / p
    d2 = np.sum((sample - mu * np.eye(p)) ** 2) / p
    b2 = sum(np.sum((np.outer(row, row) - sample) ** 2) for row in x) / (p * n * n)
    actual = ledoit_wolf_identity(x)
    assert_allclose(actual.shrinkage, min(1, b2 / d2), rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-70, 1e-8, -2.0, 1e8, 1e70])
def test_scale_equivariance(scale):
    x = np.random.default_rng(127).normal(size=(60, 9)) * np.arange(1, 10)
    base = ledoit_wolf_identity(x)
    changed = ledoit_wolf_identity(scale * x)
    assert_allclose(changed.covariance / scale**2, base.covariance, rtol=1e-11, atol=1e-11)
    assert_allclose(changed.shrinkage, base.shrinkage, rtol=1e-11)


def test_orthogonal_equivariance_and_input_not_mutated():
    rng = np.random.default_rng(234)
    x = rng.normal(size=(30, 8)) * np.arange(1, 9)
    saved = x.copy()
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    base = ledoit_wolf_identity(x)
    rotated = ledoit_wolf_identity(x @ q)
    assert_allclose(rotated.covariance, q.T @ base.covariance @ q, rtol=1e-11, atol=1e-11)
    assert_allclose(rotated.shrinkage, base.shrinkage, rtol=1e-11)
    assert_allclose(x, saved, rtol=0, atol=0)


def test_rank_deficient_sample_becomes_positive_definite():
    x = np.random.default_rng(10).normal(size=(10, 30))
    estimate = ledoit_wolf_identity(x)
    assert np.linalg.matrix_rank(estimate.sample_covariance) <= 10
    assert np.linalg.eigvalsh(estimate.covariance).min() > 0
    assert_allclose(np.trace(estimate.covariance), np.trace(estimate.sample_covariance), rtol=1e-13)


def test_degenerate_samples_are_not_claimed_to_be_positive_definite():
    estimate = ledoit_wolf_identity(np.zeros((10, 3)))
    assert_allclose(estimate.covariance, 0)
    assert estimate.shrinkage == 0.0
    scalar = ledoit_wolf_identity(np.array([[2.0], [4.0]]))
    assert scalar.shrinkage == 0.0
    assert_allclose(scalar.covariance, [[10.0]])
    spherical = ledoit_wolf_identity(np.eye(3))
    assert_allclose(spherical.covariance, np.eye(3) / 3)


def test_representable_extreme_covariance_does_not_overflow_intermediate_scale():
    estimate = ledoit_wolf_identity([[1.5e154], [0.0]])
    assert_allclose(estimate.covariance, [[1.125e308]], rtol=1e-14)
    with pytest.raises(ValueError, match="representable covariance range"):
        ledoit_wolf_identity([[1.5e155], [0.0]])


@pytest.mark.parametrize("bad", [[], [1, 2], [[1, 2]], [[1, float('nan')], [2, 3]],
                                  [[1, float('inf')], [2, 3]], np.empty((3, 0))])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        ledoit_wolf_identity(bad)

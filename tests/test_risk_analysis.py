import copy
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.covariance import ledoit_wolf

from shrinkage_research.risk_analysis import analyze_risk, covariance_diagnostics, fit_covariances, portfolio_weights
from shrinkage_research.risk_data import RiskInput, read_risk_input
from shrinkage_research.risk_demo import synthetic_csv


@pytest.fixture
def risk_config():
    return json.loads((Path(__file__).parents[1] / "configs" / "risk_demo.json").read_text())


def test_euler_negative_contribution_matches_hand_calculation():
    covariance = np.array([[0.0004, -0.0003], [-0.0003, 0.0009]])
    result = covariance_diagnostics(covariance, np.array([0.9, 0.1]), 252)
    assert result["period_variance"] == pytest.approx(0.000279)
    assert result["annualized_volatility"] == pytest.approx(np.sqrt(252 * 0.000279))
    assert_allclose(result["risk_contribution_shares"], [33 / 31, -2 / 31])
    assert sum(result["volatility_components"]) == pytest.approx(result["annualized_volatility"])


def test_singular_covariance_still_has_risk_and_zero_risk_is_explicitly_undefined():
    same = covariance_diagnostics(np.ones((2, 2)), np.array([0.5, 0.5]), 12)
    assert same["singular"] is True
    assert same["condition_number"] is None
    assert same["annualized_volatility"] == pytest.approx(np.sqrt(12))
    hedge = covariance_diagnostics(np.array([[1, -1], [-1, 1]]), np.array([0.5, 0.5]), 12)
    assert hedge["annualized_volatility"] == 0
    assert hedge["volatility_components"] is None
    assert hedge["risk_contribution_shares"] is None


def test_zero_variance_asset_correlations_are_undefined_and_constants_stay_zero():
    matrices, _ = fit_covariances(np.tile([0.001, 0.002], (125, 1)))
    for matrix in matrices.values():
        assert_allclose(matrix, 0, atol=0)
    result = covariance_diagnostics(np.diag([0.0, 0.01]), np.array([0.5, 0.5]), 252)
    assert result["correlation"][0] == [None, None]
    assert result["correlation"][1][0] is None
    assert result["correlation"][1][1] == pytest.approx(1.0)


def test_covariance_and_correlation_normalization_match_independent_oracles():
    x = np.random.default_rng(412).normal(size=(35, 4)) * [0.01, 0.03, 0.004, 0.04] + 0.001
    matrices, intensity = fit_covariances(x)
    expected, delta = ledoit_wolf(x, assume_centered=False)
    assert_allclose(matrices["sample_mle"], np.cov(x, rowvar=False, ddof=0), rtol=1e-12, atol=1e-15)
    assert_allclose(matrices["ledoit_wolf"], expected, rtol=1e-12, atol=1e-15)
    assert intensity == pytest.approx(delta)
    assert_allclose(matrices["diagonal_sample_mle"], np.diag(np.diag(matrices["sample_mle"])))
    covariance = np.array([[2.0, 1.0], [1.0, 3.0]])
    expected_corr = [[1.0, 1 / np.sqrt(6)], [1 / np.sqrt(6), 1.0]]
    for scale in [1e-180, 1.0, 1e180]:
        result = covariance_diagnostics(scale * covariance, np.array([0.5, 0.5]), 1)
        assert_allclose(result["correlation"], expected_corr, rtol=1e-12)


def test_rolling_predictions_ignore_future_data_and_realized_proxy_uses_ddof1(risk_config):
    data = read_risk_input(synthetic_csv(), input_kind="returns")
    before = analyze_risk(data, risk_config)
    changed_values = data.returns.copy()
    changed_values[83:] = changed_values[83:] * 2 + 0.003
    changed = RiskInput(data.dates, data.assets, changed_values, data.audit)
    after = analyze_risk(changed, risk_config)
    cutoff = data.dates[83]
    for a, b in zip(before["rolling_rows"], after["rolling_rows"]):
        if a["train_end"] < cutoff:
            assert a["predicted_period_variance"] == b["predicted_period_variance"]
    first = before["rolling_rows"][0]
    assert first["train_end"] < first["test_start"]
    direct = np.var(data.returns[60:80] @ np.full(4, 0.25), ddof=1)
    assert first["realized_period_variance_ddof1"] == pytest.approx(direct)
    assert before["rolling_rows"][-1]["test_rows"] == 5
    assert before["rolling_rows"][-1]["partial_test"] is True


def test_singleton_final_block_is_disclosed_not_evaluated(risk_config):
    data = read_risk_input(synthetic_csv(rows=121), input_kind="returns")
    result = analyze_risk(data, risk_config)
    assert len(result["rolling_rows"]) == 9
    assert result["rolling_summary"]["skipped_test_blocks"][0]["rows"] == 1


@pytest.mark.parametrize("weights", [{"A": 1}, {"A": -0.1, "B": 1.1}, {"A": 0.2, "B": 0.2}, {"A": float('nan'), "B": 0.5}])
def test_invalid_or_misaligned_weights_rejected(weights):
    with pytest.raises(ValueError):
        portfolio_weights(("A", "B"), weights)

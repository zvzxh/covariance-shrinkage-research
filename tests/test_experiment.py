import copy
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from shrinkage_research import cli
from shrinkage_research.design import gaussian_sample_risk, matched_lognormal_spectrum
from shrinkage_research.experiment import config_digest, run_scenario, validate_config


@pytest.fixture
def config():
    path = Path(__file__).parents[1] / "configs" / "paper_subset.json"
    value = json.loads(path.read_text())
    value["replications"] = 100
    value["scenarios"] = value["scenarios"][:1]
    return value


@pytest.mark.parametrize("p,alpha2", [(20, 0.0), (20, 0.1), (20, 0.5), (20, 2.0), (40, 0.5)])
def test_positive_spectrum_has_requested_exact_moments(p, alpha2):
    eigenvalues = matched_lognormal_spectrum(p, alpha2, 912)
    assert (eigenvalues > 0).all()
    assert_allclose(eigenvalues.mean(), 1.0, rtol=0, atol=1e-13)
    assert_allclose(eigenvalues.var(), alpha2, rtol=0, atol=1e-12)
    assert_allclose(eigenvalues, matched_lognormal_spectrum(p, alpha2, 912), rtol=0, atol=0)


def test_gaussian_risk_known_cases():
    # S = X'X/n for known mean: identity risk is (p+1)/n, not p/n.
    assert gaussian_sample_risk(np.ones(10), 40) == pytest.approx(11 / 40)
    eigenvalues = matched_lognormal_spectrum(20, 0.5, 912)
    assert gaussian_sample_risk(eigenvalues, 40) == pytest.approx(0.5375)


def test_reproducibility_and_reported_paired_monte_carlo_se(config):
    first, rows = run_scenario(config, config["scenarios"][0])
    second, repeated_rows = run_scenario(config, config["scenarios"][0])
    assert first == second
    assert rows == repeated_rows
    losses = np.array([[r["sample_loss"], r["shrinkage_loss"]] for r in rows])
    a, b = losses.mean(axis=0)
    covariance_of_mean = np.cov(losses, rowvar=False, ddof=1) / len(rows)
    gradient = np.array([100 * b / a**2, -100 / a])
    independently_calculated_se = np.sqrt(gradient @ covariance_of_mean @ gradient)
    assert first["prial_mc_se_pp"] == pytest.approx(independently_calculated_se, rel=1e-12)
    assert first["paired_risk_reduction_mc_se"] == pytest.approx(
        np.std(losses[:, 0] - losses[:, 1], ddof=1) / np.sqrt(len(rows)))


def test_config_hash_is_order_independent_and_changes_with_seed(config):
    assert config_digest(config) == config_digest(dict(reversed(list(config.items()))))
    changed = copy.deepcopy(config)
    changed["observation_seed"] += 1
    assert config_digest(config) != config_digest(changed)


def test_changed_central_parameters_do_not_masquerade_as_table2(config):
    config["scenarios"][0]["alpha2"] = 0.1
    result, _ = run_scenario(config, config["scenarios"][0])
    assert "paper_table2_reference" not in result


@pytest.mark.parametrize("field,value", [("replications", 1), ("replications", True),
                                        ("observation_seed", -1), ("schema_version", 7)])
def test_invalid_protocol_rejected(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_cli_retains_failed_gate_and_does_not_overwrite(config, tmp_path, monkeypatch):
    original = cli.run_scenario

    def forced_failure(*args):
        result, rows = original(*args)
        result["checks"]["deliberately_failed_test_gate"] = False
        result["status"] = "fail"
        return result, rows

    monkeypatch.setattr(cli, "run_scenario", forced_failure)
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config))
    output = tmp_path / "run"
    assert cli.main(["--config", str(source), "--out", str(output)]) == 1
    failure = json.loads((output / "failure.json").read_text())
    assert failure["kind"] == "acceptance_gate_failure"
    assert (output / "replications.csv").is_file()
    assert (output / "loss_comparison.png").is_file()
    before = (output / "summary.json").read_bytes()
    with pytest.raises(SystemExit) as caught:
        cli.main(["--config", str(source), "--out", str(output)])
    assert caught.value.code == 2
    assert (output / "summary.json").read_bytes() == before


def test_cli_saves_invalid_config_failure(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text("{invalid JSON")
    output = tmp_path / "run"
    assert cli.main(["--config", str(source), "--out", str(output)]) == 1
    assert json.loads((output / "failure.json").read_text())["kind"] == "JSONDecodeError"
    assert json.loads((output / "manifest.json").read_text())["status"] == "error"

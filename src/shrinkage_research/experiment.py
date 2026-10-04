"""Seeded experiments with retained observations of estimator loss and gates."""

import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from .design import gaussian_sample_risk, matched_lognormal_spectrum
from .estimator import ledoit_wolf_identity


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported schema_version; expected 1")
    for key in ("replications", "observation_seed", "spectrum_seed"):
        if type(config.get(key)) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["replications"] < 100:
        raise ValueError("at least 100 replications are required for reported Monte Carlo SEs")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a nonempty list")
    ids: set[str] = set()
    for scenario in scenarios:
        name = scenario.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("scenario id must contain lowercase letters, digits, underscores")
        if name in ids:
            raise ValueError(f"duplicate scenario id: {name}")
        ids.add(name)
        for key in ("p", "n"):
            if type(scenario.get(key)) is not int or scenario[key] < 2:
                raise ValueError(f"{name}.{key} must be an integer >= 2")
        alpha2 = scenario.get("alpha2")
        if not isinstance(alpha2, (float, int)) or not math.isfinite(alpha2):
            raise ValueError(f"{name}.alpha2 must be finite")
        if not 0 <= alpha2 < scenario["p"] - 1:
            raise ValueError(f"{name}.alpha2 must satisfy 0 <= alpha2 < p-1")
    for key in ("analytic_risk_z_limit", "paired_improvement_z_limit",
                "spectrum_moment_tolerance", "matrix_relative_tolerance"):
        value = config.get("gates", {}).get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"gates.{key} must be positive and finite")


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def _stream_seed(base: int, scenario_id: str) -> np.random.SeedSequence:
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    return np.random.SeedSequence([base, int.from_bytes(digest[:8], "little")])


def run_scenario(config: dict[str, Any], scenario: dict[str, Any]) -> tuple[dict, list[dict]]:
    """Run a fixed-covariance Gaussian experiment and return every replication."""
    p, n, reps = scenario["p"], scenario["n"], config["replications"]
    # Equal p uses the same shape seed across scenarios. Calibration changes
    # dispersion without introducing a new underlying normal shape vector.
    spectrum_seed = np.random.SeedSequence([config["spectrum_seed"], p])
    eigenvalues = matched_lognormal_spectrum(p, scenario["alpha2"], spectrum_seed)
    truth = np.diag(eigenvalues)
    rng = np.random.default_rng(_stream_seed(config["observation_seed"], scenario["id"]))
    loss_sample = np.empty(reps)
    loss_shrunk = np.empty(reps)
    intensities = np.empty(reps)
    condition_sample = np.empty(reps)
    condition_shrunk = np.empty(reps)
    max_trace_error, max_symmetry_error = 0.0, 0.0
    min_relative_eigenvalue = math.inf
    rows = []
    for i in range(reps):
        x = rng.standard_normal((n, p)) * np.sqrt(eigenvalues)
        estimate = ledoit_wolf_identity(x)
        sample, shrunk = estimate.sample_covariance, estimate.covariance
        loss_sample[i] = np.sum((sample - truth) ** 2) / p
        loss_shrunk[i] = np.sum((shrunk - truth) ** 2) / p
        intensities[i] = estimate.shrinkage
        sample_eigenvalues = np.linalg.eigvalsh(sample)
        # Identity shrinkage preserves eigenvectors and transforms eigenvalues.
        shrunk_eigenvalues = ((1.0 - estimate.shrinkage) * sample_eigenvalues
                             + estimate.shrinkage * estimate.target_variance)
        if p > n or sample_eigenvalues[0] <= np.finfo(float).eps * sample_eigenvalues[-1]:
            condition_sample[i] = np.inf
        else:
            condition_sample[i] = sample_eigenvalues[-1] / sample_eigenvalues[0]
        condition_shrunk[i] = shrunk_eigenvalues[-1] / shrunk_eigenvalues[0]
        matrix_scale = max(float(np.linalg.norm(sample, ord="fro")), np.finfo(float).tiny)
        max_trace_error = max(max_trace_error, abs(float(np.trace(sample - shrunk))) / matrix_scale)
        max_symmetry_error = max(max_symmetry_error, float(np.linalg.norm(shrunk - shrunk.T)) / matrix_scale)
        min_relative_eigenvalue = min(min_relative_eigenvalue, float(shrunk_eigenvalues[0]) / matrix_scale)
        rows.append({"scenario": scenario["id"], "replication": i,
                     "sample_loss": float(loss_sample[i]), "shrinkage_loss": float(loss_shrunk[i]),
                     "shrinkage_intensity": float(intensities[i]),
                     "sample_condition_number": (float(condition_sample[i])
                                                 if np.isfinite(condition_sample[i]) else None),
                     "shrinkage_condition_number": float(condition_shrunk[i])})

    risk_sample, se_sample = _mean_se(loss_sample)
    risk_shrunk, se_shrunk = _mean_se(loss_shrunk)
    improvement, improvement_se = _mean_se(loss_sample - loss_shrunk)
    ratio = risk_shrunk / risk_sample
    # Paired delta-method SE for 100*(1-mean(Lw)/mean(Ls)). This retains the
    # covariance between losses computed on the same random observations.
    prial_se = 100.0 * _mean_se(loss_shrunk - ratio * loss_sample)[1] / risk_sample
    analytic = gaussian_sample_risk(eigenvalues, n)
    z = (risk_sample - analytic) / se_sample
    gates = config["gates"]
    mean_error = abs(float(eigenvalues.mean()) - 1.0)
    variance_error = abs(float(eigenvalues.var()) - scenario["alpha2"])
    checks = {
        "population_spectrum_matches_moments": max(mean_error, variance_error) <= gates["spectrum_moment_tolerance"],
        "sample_risk_matches_gaussian_identity": abs(z) <= gates["analytic_risk_z_limit"],
        "positive_paired_mean_improvement": improvement - gates["paired_improvement_z_limit"] * improvement_se > 0,
        "matrix_invariants": (max_trace_error <= gates["matrix_relative_tolerance"]
                              and max_symmetry_error <= gates["matrix_relative_tolerance"]
                              and min_relative_eigenvalue > 0
                              and bool(((intensities >= 0) & (intensities <= 1)).all())),
        "finite_shrinkage_condition_numbers": bool(np.isfinite(condition_shrunk).all()),
    }
    result = {
        **scenario, "replications": reps, "spectrum": eigenvalues.tolist(),
        "population_mean_eigenvalue": float(eigenvalues.mean()),
        "population_eigenvalue_variance": float(eigenvalues.var()),
        "sample_risk": risk_sample, "sample_risk_mc_se": se_sample,
        "shrinkage_risk": risk_shrunk, "shrinkage_risk_mc_se": se_shrunk,
        "analytic_sample_risk": analytic, "sample_risk_z_score": z,
        "paired_risk_reduction": improvement, "paired_risk_reduction_mc_se": improvement_se,
        "prial_percent": 100.0 * (1.0 - ratio), "prial_mc_se_pp": prial_se,
        "asymptotic_prial_reference_percent": 100.0 * (p / n) / (p / n + scenario["alpha2"]),
        "mean_shrinkage": float(intensities.mean()),
        "mean_sample_condition_number": (float(condition_sample.mean())
                                         if np.isfinite(condition_sample).all() else None),
        "sample_singular_fraction": float((~np.isfinite(condition_sample)).mean()),
        "mean_shrinkage_condition_number": float(condition_shrunk.mean()),
        "true_condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "max_relative_trace_error": max_trace_error,
        "max_relative_symmetry_error": max_symmetry_error,
        "min_relative_shrinkage_eigenvalue": min_relative_eigenvalue,
        "checks": checks, "status": "pass" if all(checks.values()) else "fail",
    }
    if (scenario["id"] == "central" and p == 20 and n == 40
            and scenario["alpha2"] == 0.5):
        # Descriptive comparison, not a gate: the original realized spectrum
        # and random seed are unavailable and may affect finite-sample LW risk.
        result["paper_table2_reference"] = {
            "sample_risk": 0.5372, "sample_risk_mc_se": 0.0033,
            "shrinkage_risk": 0.2723, "shrinkage_risk_mc_se": 0.0013,
            "prial_percent": 49.3, "replications": 1000,
            "comparison_is_acceptance_gate": False,
            "sample_risk_difference": risk_sample - 0.5372,
            "shrinkage_risk_difference": risk_shrunk - 0.2723,
            "prial_difference_pp": 100 * (1 - ratio) - 49.3,
        }
    return result, rows


def config_digest(config: dict) -> str:
    content = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

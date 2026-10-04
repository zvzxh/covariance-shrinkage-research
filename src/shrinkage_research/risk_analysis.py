"""V2 covariance diagnostics, Euler risk and out-of-sample variance checks."""

import math

import numpy as np

from .estimator import ledoit_wolf_identity
from .risk_data import RiskInput, iso_date

METHODS = ("sample_mle", "ledoit_wolf", "diagonal_sample_mle")


def validate_risk_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise ValueError("risk configuration schema_version must be 2")
    if not isinstance(config.get("study_id"), str) or not config["study_id"].strip():
        raise ValueError("study_id must be a nonempty string")
    if config.get("data_kind") not in {"synthetic", "user_supplied"}:
        raise ValueError("data_kind must be synthetic or user_supplied")
    if config.get("input_kind") not in {"returns", "adjusted_prices"}:
        raise ValueError("input_kind must be returns or adjusted_prices")
    for name in ("start", "end"):
        if config.get(name) is not None:
            iso_date(config[name])
    if config.get("start") is not None and config.get("end") is not None and config["start"] > config["end"]:
        raise ValueError("start must not be after end")
    n = config.get("last_n")
    if n is not None and (type(n) is not int or n < 2):
        raise ValueError("last_n must be null or an integer >= 2")
    a = config.get("periods_per_year")
    if type(a) is not int or a < 1:
        raise ValueError("periods_per_year must be a positive integer")
    weights = config.get("weights")
    if weights is not None and (not isinstance(weights, dict) or any(not isinstance(k, str) for k in weights)):
        raise ValueError("weights must be null for equal weight or an asset-to-weight JSON object")
    rolling = config.get("rolling")
    if not isinstance(rolling, dict) or type(rolling.get("enabled")) is not bool:
        raise ValueError("rolling.enabled must be true or false")
    for name in ("train_window", "test_window"):
        if type(rolling.get(name)) is not int or rolling[name] < 2:
            raise ValueError(f"rolling.{name} must be an integer >= 2")
    if type(rolling.get("include_partial")) is not bool:
        raise ValueError("rolling.include_partial must be true or false")


def portfolio_weights(assets: tuple[str, ...], supplied: dict | None) -> np.ndarray:
    if supplied is None:
        return np.full(len(assets), 1.0 / len(assets))
    if set(supplied) != set(assets):
        raise ValueError("explicit weights must name every input asset exactly once, with no extra assets")
    if any(type(supplied[a]) not in (int, float) for a in assets):
        raise ValueError("weights must be JSON numbers")
    weights = np.array([supplied[a] for a in assets], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or abs(float(weights.sum()) - 1) > 1e-10:
        raise ValueError("weights must be finite, nonnegative and sum to one within 1e-10")
    return weights


def fit_covariances(x: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("covariance fitting needs at least two finite observations and assets")
    # Exactly constant columns are centered to zero explicitly, so a repeating
    # nonzero decimal cannot acquire variance from floating-point mean rounding.
    clean = values.copy()
    clean[:, np.all(values == values[0], axis=0)] = 0.0
    estimate = ledoit_wolf_identity(clean, center=True)
    sample = estimate.sample_covariance
    return {"sample_mle": sample, "ledoit_wolf": estimate.covariance,
            "diagonal_sample_mle": np.diag(np.diag(sample))}, estimate.shrinkage


def covariance_diagnostics(covariance: np.ndarray, weights: np.ndarray, annualization: int) -> dict:
    covariance = np.asarray(covariance, dtype=float)
    p = len(weights)
    if covariance.shape != (p, p) or not np.isfinite(covariance).all():
        raise ValueError("covariance must be a finite square matrix aligned with weights")
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = scale * p * np.finfo(float).eps * 100
    if eigenvalues[0] < -tolerance:
        raise ValueError("covariance has a materially negative eigenvalue")
    rank = int(np.sum(eigenvalues > tolerance))
    condition = float(eigenvalues[-1] / eigenvalues[0]) if rank == p else None
    variance = float(weights @ covariance @ weights)
    if variance < -tolerance:
        raise ValueError("portfolio variance is materially negative")
    clipped = variance < 0
    variance = max(0.0, variance)
    annual_variance = annualization * variance
    if not math.isfinite(annual_variance):
        raise ValueError("annualized portfolio variance is unrepresentable")
    volatility = math.sqrt(annual_variance)
    components = annualization * weights * (covariance @ weights) / volatility if volatility > 0 else None
    shares = components / volatility if components is not None else None
    diagonal = np.diag(covariance)
    standard_deviations = np.sqrt(np.maximum(diagonal, 0))
    denominators = np.outer(standard_deviations, standard_deviations)
    correlation = np.full_like(covariance, np.nan)
    np.divide(covariance, denominators, out=correlation, where=denominators > 0)
    warnings = []
    if rank < p:
        warnings.append("Covariance is numerically singular; inverse-based use is unsafe, but portfolio variance can still be defined.")
    if (diagonal <= 0).any():
        warnings.append("Zero-variance asset: its correlations are undefined, including its correlation diagonal.")
    if volatility == 0:
        warnings.append("Portfolio variance is zero; volatility-contribution shares and Euler components are undefined.")
    if clipped:
        warnings.append("Tiny negative portfolio variance from roundoff was clipped to zero.")
    if rank == p and condition is not None and condition > 1e8:
        warnings.append("Covariance condition number exceeds 1e8; inverse-based calculations can be unstable.")
    return {"period_variance": variance, "annualized_variance": annual_variance,
            "annualized_volatility": volatility,
            "volatility_components": components.tolist() if components is not None else None,
            "risk_contribution_shares": shares.tolist() if shares is not None else None,
            "correlation": [[float(v) if np.isfinite(v) else None for v in row] for row in correlation],
            "eigenvalues": eigenvalues.tolist(), "numerical_rank": rank,
            "eigenvalue_tolerance": tolerance, "condition_number": condition,
            "singular": rank < p, "warnings": warnings}


def rolling_risk(data: RiskInput, weights: np.ndarray, settings: dict) -> tuple[list[dict], dict]:
    train, test = settings["train_window"], settings["test_window"]
    rows, skipped = [], []
    for start in range(train, len(data.dates), test):
        stop = min(start + test, len(data.dates))
        count = stop - start
        if count < 2 or (count < test and not settings["include_partial"]):
            skipped.append({"start_date": data.dates[start], "rows": count,
                            "reason": "at least 2 test observations required" if count < 2 else "partial test block disabled"})
            continue
        matrices, intensity = fit_covariances(data.returns[start - train:start])
        portfolio_observations = data.returns[start:stop] @ weights
        realized = 0.0 if np.all(portfolio_observations == portfolio_observations[0]) else float(np.var(portfolio_observations, ddof=1))
        if not math.isfinite(realized):
            raise ValueError("realized test variance is unrepresentable")
        for method, covariance in matrices.items():
            prediction = covariance_diagnostics(covariance, weights, 1)["period_variance"]
            rows.append({"fold": len(rows) // len(METHODS), "method": method,
                         "train_start": data.dates[start - train], "train_end": data.dates[start - 1],
                         "test_start": data.dates[start], "test_end": data.dates[stop - 1],
                         "training_rows": train, "test_rows": count, "partial_test": count < test,
                         "predicted_period_variance": prediction, "realized_period_variance_ddof1": realized,
                         "variance_error": prediction - realized, "squared_variance_error": (prediction - realized) ** 2,
                         "prediction_to_realized_ratio": prediction / realized if realized > 0 else None,
                         "training_shrinkage": intensity if method == "ledoit_wolf" else None})
    if not rows:
        raise ValueError("rolling validation needs a training window followed by at least two selected return rows")
    summary = {"fixed_weights": True, "realized_variance_ddof": 1,
               "interpretation": "Noisy sample-variance proxy of fixed-weight period returns, not true covariance or a drifted-holdings backtest.",
               "skipped_test_blocks": skipped, "methods": {}}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        errors = np.array([row["variance_error"] for row in selected])
        summary["methods"][method] = {"folds": len(selected), "mean_squared_variance_error": float(np.mean(errors ** 2)),
                                       "mean_absolute_variance_error": float(np.mean(np.abs(errors)))}
    return rows, summary


def analyze_risk(data: RiskInput, config: dict) -> dict:
    validate_risk_config(config)
    weights = portfolio_weights(data.assets, config.get("weights"))
    matrices, intensity = fit_covariances(data.returns)
    methods = {name: {**covariance_diagnostics(matrix, weights, config["periods_per_year"]),
                      "covariance": matrix.tolist(), "shrinkage_intensity": intensity if name == "ledoit_wolf" else None}
               for name, matrix in matrices.items()}
    rolling_rows, rolling_summary = (rolling_risk(data, weights, config["rolling"])
                                     if config["rolling"]["enabled"] else ([], None))
    return {"schema_version": 2, "study_id": config["study_id"], "data_kind": config["data_kind"],
            "assets": list(data.assets), "weights": weights.tolist(), "observations": len(data.dates),
            "periods_per_year": config["periods_per_year"], "training_covariance_ddof": 0,
            "normalization": "All three covariance estimators center observations and divide by n.",
            "input_audit": data.audit, "methods": methods,
            "rolling_rows": rolling_rows, "rolling_summary": rolling_summary}

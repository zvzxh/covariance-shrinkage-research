"""Portable V2 HTML/CSV/JSON artifacts with no remote report dependencies."""

import base64
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
from importlib import metadata
import io
import json
import platform
import zipfile

import numpy as np

from . import __version__
from .risk_data import RiskInput, selected_returns_csv


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv_bytes(header: list[str], rows: list) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def risk_figure(analysis: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(analysis["methods"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    assets = analysis["assets"]
    display_count = min(len(assets), 20)
    for j, (name, values) in enumerate(analysis["methods"].items()):
        components = values["volatility_components"]
        if components is not None:
            axes[0].bar(np.arange(display_count) + (j - 1) * 0.26, np.array(components[:display_count]) * 100,
                        width=0.26, label=name)
        eigenvalues = np.array(values["eigenvalues"])[::-1]
        axes[1].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, label=name, marker=".")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_xticks(np.arange(display_count), assets[:display_count], rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Contribution to annualized volatility (percentage points)")
    axes[0].set_title("Euler risk contributions" + (" (first 20 assets)" if len(assets) > 20 else ""))
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("Eigenvalue rank (largest first)")
    axes[1].set_ylabel("Period covariance eigenvalue")
    axes[1].set_title("Covariance spectrum")
    axes[1].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SYNTHETIC EXAMPLE — not observed market data" if analysis["data_kind"] == "synthetic"
                 else "Historical risk estimates — user-supplied data", fontsize=12)
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=160)
    plt.close(fig)
    return stream.getvalue()


def _formatted(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, (float, np.floating)):
        return f"{value:.7g}"
    return str(value)


def _table(headers: list[str], rows: list) -> str:
    # Escape every header and cell, including asset names, user study labels,
    # filenames and numeric-looking strings. No user text is interpreted as HTML.
    return ("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(_formatted(v))}</td>" for v in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def render_html(analysis: dict, config: dict, filename: str, figure: bytes) -> str:
    methods, assets, audit = analysis["methods"], analysis["assets"], analysis["input_audit"]
    badge = "SYNTHETIC EXAMPLE — generated observations, not market data" if analysis["data_kind"] == "synthetic" else "USER-SUPPLIED DATA — provenance and price adjustment are not certified"
    html = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>Portfolio risk analysis</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#162e38;background:#f6f8f9;margin:0}main{max-width:1150px;margin:auto;padding:28px}section{background:white;padding:24px;margin:20px 0;border:1px solid #dce5e8;border-radius:10px}h1,h2,h3{line-height:1.2}small{color:#506873}.badge{padding:12px;background:#fff2d4;border-left:4px solid #be8524}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #dce5e8;text-align:right;padding:8px;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}.warning{color:#854811}</style></head><body><main>",
            "<h1>Portfolio risk analysis</h1>", f"<p class='badge'>{escape(badge)}</p>",
            f"<p>Study: <strong>{escape(analysis['study_id'])}</strong><br>Input: {escape(filename)}</p>",
            "<section><h2>Selection and method</h2>",
            f"<p>{analysis['observations']} return periods, {len(assets)} assets, {escape(audit['selected_first_return_end'])} to {escape(audit['selected_last_return_end'])}. "
            f"Annualization: {analysis['periods_per_year']} periods/year.</p>",
            "<p><strong>All three estimators center returns and use 1/n (ddof=0).</strong> The full-sample estimates describe the selected history; rolling results below are fitted only on earlier data. No return forecast or buy/sell recommendation is made.</p>",
            _table(["Asset", "Portfolio weight"], list(zip(assets, analysis["weights"]))),
            "<p>Annualization scales variance by periods/year and assumes zero serial covariance between periods. A row is a user-declared observation period; missing trading sessions are not imputed.</p></section>",
            "<section><h2>Portfolio risk comparison</h2>",
            _table(["Method", "Annualized volatility", "Period variance", "Rank", "Condition number", "LW intensity"],
                   [[name, values["annualized_volatility"], values["period_variance"], values["numerical_rank"], values["condition_number"], values["shrinkage_intensity"]] for name, values in methods.items()]),
            "<p>Volatility and contributions are decimal units (0.10 = 10%). Euler components sum to portfolio volatility and can be negative. Diagonal covariance ignores cross-asset dependence. A singular matrix can still give a valid portfolio variance; an undefined condition number means it is not safely invertible.</p>",
            "<img alt='Risk contribution and eigenvalue diagnostics' src='data:image/png;base64," + base64.b64encode(figure).decode() + "'></section>"]
    for name, values in methods.items():
        html.append(f"<section><h2>{escape(name)}</h2>")
        html.extend(f"<p class='warning'>{escape(warning)}</p>" for warning in values["warnings"])
        components, shares = values["volatility_components"], values["risk_contribution_shares"]
        html.append(_table(["Asset", "Volatility contribution", "Contribution share"],
                           [[asset, components[i] if components is not None else None, shares[i] if shares is not None else None] for i, asset in enumerate(assets)]))
        html.append("<details><summary>Period covariance matrix</summary>" + _table(["Asset", *assets], [[asset, *values["covariance"][i]] for i, asset in enumerate(assets)]) + "</details>")
        html.append("<details><summary>Correlation matrix</summary>" + _table(["Asset", *assets], [[asset, *values["correlation"][i]] for i, asset in enumerate(assets)]) + "</details></section>")
    html.append("<section><h2>Rolling out-of-sample variance check</h2>")
    if analysis["rolling_summary"] is None:
        html.append("<p>Disabled in this run. Enable it with enough selected observations for a training block and at least two following test returns.</p>")
    else:
        html.append("<p>Predicted period variance uses only the previous training window. Realized variance is the test sample variance (ddof=1) of fixed-weight period returns: a noisy proxy, not true covariance. These are constant-exposure scalar observations, not a transaction-cost or drifting-holdings backtest. Errors below are unweighted averages across test blocks; short partial blocks have noisier variance proxies.</p>")
        html.append(_table(["Method", "Test blocks", "Mean absolute variance error", "Mean squared variance error"],
                           [[name, value["folds"], value["mean_absolute_variance_error"], value["mean_squared_variance_error"]] for name, value in analysis["rolling_summary"]["methods"].items()]))
        html.append(_table(["Fold", "Method", "Training ends", "Testing starts", "Test rows", "Predicted variance", "Realized proxy"],
                           [[r["fold"], r["method"], r["train_end"], r["test_start"], r["test_rows"], r["predicted_period_variance"], r["realized_period_variance_ddof1"]] for r in analysis["rolling_rows"]]))
        for skipped in analysis["rolling_summary"]["skipped_test_blocks"]:
            html.append(f"<p class='warning'>Skipped final block at {escape(skipped['start_date'])}: {skipped['rows']} rows; {escape(skipped['reason'])}.</p>")
    html.append("</section><section><h2>Input and transformation audit</h2>")
    html.append(_table(["Audit item", "Value"], list(audit.items())))
    html.append("<p>Adjusted-price input is validated as positive values, converted first with P(t)/P(t-1)-1 without filling, and then selected by return-end dates. A selected first return can use a baseline price before the requested start date, explicitly recorded above. Actual corporate-action adjustment cannot be inferred from a CSV.</p>")
    html.append("<details><summary>Reusable configuration</summary><pre>" + escape(json.dumps(config, indent=2)) + "</pre></details></section>")
    html.append("<small>Covariance Shrinkage Research v" + escape(__version__) + ". This self-contained report contains no remote image, script or API dependency.</small></main></body></html>")
    return "\n".join(html)


def build_exports(data: RiskInput, analysis: dict, config: dict, *, filename: str) -> dict[str, bytes]:
    assets = list(data.assets)
    selected = selected_returns_csv(data)
    figure = risk_figure(analysis)
    files = {"selected_returns.csv": selected, "analysis.json": json_bytes(analysis),
             "config.json": json_bytes(config), "input_audit.json": json_bytes(data.audit), "risk_diagnostics.png": figure}
    contribution_rows, spectrum_rows, summary_rows = [], [], []
    for method, values in analysis["methods"].items():
        files[f"covariance_{method}.csv"] = csv_bytes(["asset", *assets], [[asset, *values["covariance"][i]] for i, asset in enumerate(assets)])
        files[f"correlation_{method}.csv"] = csv_bytes(["asset", *assets], [[asset, *values["correlation"][i]] for i, asset in enumerate(assets)])
        for i, asset in enumerate(assets):
            contribution_rows.append([method, asset, analysis["weights"][i], values["volatility_components"][i] if values["volatility_components"] is not None else None,
                                      values["risk_contribution_shares"][i] if values["risk_contribution_shares"] is not None else None])
        spectrum_rows.extend([method, i + 1, value] for i, value in enumerate(values["eigenvalues"]))
        summary_rows.append([method, values["annualized_volatility"], values["period_variance"], values["numerical_rank"], values["condition_number"], values["singular"]])
    files["risk_contributions.csv"] = csv_bytes(["method", "asset", "weight", "annualized_volatility_component", "contribution_share"], contribution_rows)
    files["eigenspectrum.csv"] = csv_bytes(["method", "ascending_rank", "eigenvalue"], spectrum_rows)
    files["risk_summary.csv"] = csv_bytes(["method", "annualized_volatility", "period_variance", "rank", "condition_number", "singular"], summary_rows)
    if analysis["rolling_rows"]:
        headers = list(analysis["rolling_rows"][0])
        files["rolling_validation.csv"] = csv_bytes(headers, [[r[h] for h in headers] for r in analysis["rolling_rows"]])
        files["rolling_summary.json"] = json_bytes(analysis["rolling_summary"])
    files["report.html"] = render_html(analysis, config, filename, figure).encode("utf-8")
    manifest = {"version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_filename": filename, "source_sha256": data.audit["source_sha256"],
                "selected_returns_sha256": hashlib.sha256(selected).hexdigest(),
                "config_sha256": hashlib.sha256(files["config.json"]).hexdigest(),
                "python": platform.python_version(), "numpy": metadata.version("numpy"),
                "matplotlib": metadata.version("matplotlib"), "status": "success",
                "artifact_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}
    files["manifest.json"] = json_bytes(manifest)
    return files


def zip_exports(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return stream.getvalue()

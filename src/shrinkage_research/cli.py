"""Command-line research runner. Failures are retained rather than retried away."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

from . import __version__
from .experiment import config_digest, run_scenario, validate_config


def write_json(path: Path, content: object) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _environment() -> dict:
    versions = {}
    for package in ("numpy", "matplotlib", "scikit-learn", "pytest"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "project_version": __version__}


def _plot(results: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["id"].replace("_", "\n") for r in results]
    index = np.arange(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(index - 0.18, [r["sample_risk"] for r in results], 0.36,
                label="Sample covariance", color="#8995a5")
    axes[0].bar(index + 0.18, [r["shrinkage_risk"] for r in results], 0.36,
                label="Identity shrinkage", color="#176e83")
    axes[0].set_ylabel("Mean squared Frobenius error / p")
    axes[0].set_title("Covariance estimation loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].errorbar(index, [r["prial_percent"] for r in results],
                     yerr=[1.96 * r["prial_mc_se_pp"] for r in results],
                     fmt="o", color="#176e83", capsize=4,
                     label="Simulation +/- 1.96 paired MC SE")
    axes[1].scatter(index, [r["asymptotic_prial_reference_percent"] for r in results],
                    marker="x", color="#b87132", label="Asymptotic reference")
    axes[1].set_ylabel("Relative improvement in average loss (%)")
    axes[1].set_title("PRIAL is an estimation metric, not investment return")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(index, labels, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Ledoit-Wolf (2004): design-aligned simulation subset", fontsize=13)
    fig.savefig(output / "loss_comparison.png", dpi=170)
    plt.close(fig)


def _report(results: list[dict], digest: str) -> str:
    lines = ["# Reproducible simulation results", "",
             "Design-aligned subset; not an exact reconstruction of the authors' random draws.",
             "Known zero mean, covariance divided by n. No market returns or trading profits are measured.",
             "", f"Effective configuration SHA-256: `{digest}`", "",
             "| Scenario | p | n | alpha² | Sample risk (MC SE) | Shrinkage risk (MC SE) | PRIAL % (MC SE, pp) | Gates |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['p']} | {r['n']} | {r['alpha2']} | "
                     f"{r['sample_risk']:.6f} ({r['sample_risk_mc_se']:.6f}) | "
                     f"{r['shrinkage_risk']:.6f} ({r['shrinkage_risk_mc_se']:.6f}) | "
                     f"{r['prial_percent']:.3f} ({r['prial_mc_se_pp']:.3f}) | {r['status']} |")
    lines += ["", "The SEs describe simulation uncertainty conditional on each chosen population spectrum.",
              "They do not include uncertainty over alternative spectra, financial data, or model misspecification.",
              "", "## Published central reference", "",
              "Table 2 reports sample risk 0.5372 (SE 0.0033), shrinkage risk 0.2723 (SE 0.0013),",
              "and PRIAL 49.3% over 1,000 replications. Our configured experiment uses independent draws",
              "and an explicitly calibrated spectrum. Published values are descriptive references, not fitted targets.",
              "", "## Acceptance and retention", "",
              "`summary.json` contains every gate, realized spectrum, condition-number diagnostic and reference difference.",
              "`replications.csv` retains paired losses; `config.json` and `manifest.json` identify the run.",
              "A failed gate leaves all results in place and causes a nonzero exit status. Inspect, do not change seeds to pass.",
              "", "![Loss comparison](loss_comparison.png)", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="new or empty output directory; existing results are never overwritten")
    args = parser.parse_args(argv)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("--out must be a new or empty directory; choose a new run directory")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "environment": _environment(), "command": ["shrinkage-research", *([] if argv is None else argv)],
                "config_source_sha256": None}
    results = []
    try:
        source = args.config.read_bytes()
        manifest["config_source_sha256"] = hashlib.sha256(source).hexdigest()
        config = json.loads(source)
        validate_config(config)
        write_json(args.out / "config.json", config)
        digest = config_digest(config)
        manifest["effective_config_sha256"] = digest
        manifest["command"] = ["shrinkage-research", "--config", str(args.config), "--out", str(args.out)]
        rows = []
        for scenario in config["scenarios"]:
            result, new_rows = run_scenario(config, scenario)
            results.append(result)
            rows.extend(new_rows)
            # Checkpoint after each completed scenario, including gate failures.
            write_json(args.out / "summary.json", {"study_id": config["study_id"], "scenarios": results})
            with (args.out / "replications.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            print(f"{scenario['id']}: PRIAL {result['prial_percent']:.3f}% "
                  f"(+/- {result['prial_mc_se_pp']:.3f} MC SE pp); {result['status']}", flush=True)
        _plot(results, args.out)
        (args.out / "REPORT.md").write_text(_report(results, digest), encoding="utf-8")
        passed = all(r["status"] == "pass" for r in results)
        manifest["status"] = "pass" if passed else "gate_failure"
        if not passed:
            write_json(args.out / "failure.json", {
                "kind": "acceptance_gate_failure", "effective_config_sha256": digest,
                "failed_checks": {r["id"]: [k for k, v in r["checks"].items() if not v]
                                  for r in results if r["status"] != "pass"},
                "instruction": "Retain this run. Investigate causes without changing seeds to obtain a pass."})
        return_code = 0 if passed else 1
    except Exception as error:
        manifest["status"] = "error"
        write_json(args.out / "failure.json", {"kind": type(error).__name__, "message": str(error),
                                                "traceback": traceback.format_exc(),
                                                "completed_scenarios": [r["id"] for r in results]})
        print(f"Run failed; diagnostics saved in {args.out / 'failure.json'}: {error}", file=sys.stderr)
        return_code = 1
    manifest["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.out / "manifest.json", manifest)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

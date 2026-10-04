"""V2 real-input risk workflow, separate from the unchanged V1 experiment CLI."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from . import __version__
from .risk_analysis import analyze_risk, validate_risk_config
from .risk_data import read_risk_input
from .risk_export import build_exports, json_bytes


def analysis_fingerprint(content: bytes, config: dict, filename: str) -> str:
    """Bind browser results to source bytes, provenance label and settings."""
    metadata = json.dumps({"filename": filename, "config": config}, sort_keys=True).encode()
    return hashlib.sha256(content + metadata).hexdigest()


def run_analysis(content: bytes, config: dict, *, filename: str) -> tuple[dict, dict[str, bytes]]:
    validate_risk_config(config)
    data = read_risk_input(content, input_kind=config["input_kind"], start=config.get("start"),
                           end=config.get("end"), last_n=config.get("last_n"))
    analysis = analyze_risk(data, config)
    return analysis, build_exports(data, analysis, config, filename=Path(filename).name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory; existing directories are atomically refused")
    args = parser.parse_args(argv)
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--out must not already exist, including an empty directory")
    context = {"version": __version__, "created_utc": datetime.now(timezone.utc).isoformat(),
               "input_filename": args.input.name, "config_filename": args.config.name}
    try:
        content = args.input.read_bytes()
        context["source_sha256"] = hashlib.sha256(content).hexdigest()
        config_bytes = args.config.read_bytes()
        context["config_source_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        config = json.loads(config_bytes)
        analysis, files = run_analysis(content, config, filename=args.input.name)
        for name, value in files.items():
            (args.out / name).write_bytes(value)
        print(f"Analyzed {analysis['observations']} periods and {len(analysis['assets'])} assets; report: {args.out / 'report.html'}")
        return 0
    except Exception as error:
        context["status"] = "error"
        (args.out / "manifest.json").write_bytes(json_bytes(context))
        (args.out / "failure.json").write_bytes(json_bytes({"kind": type(error).__name__, "message": str(error),
                                                           "status": "error", "source_sha256": context.get("source_sha256")}))
        print(f"Risk analysis failed; diagnostics retained: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from shrinkage_research.risk_cli import analysis_fingerprint, main, run_analysis
from shrinkage_research.risk_demo import synthetic_csv
from shrinkage_research.risk_export import zip_exports


def config():
    return json.loads((Path(__file__).parents[1] / "configs" / "risk_demo.json").read_text())


def test_browser_fingerprint_tracks_filename_as_well_as_content_and_config():
    content, cfg = synthetic_csv(), config()
    original = analysis_fingerprint(content, cfg, "first.csv")
    assert original != analysis_fingerprint(content, cfg, "renamed.csv")
    assert original != analysis_fingerprint(content + b"\n", cfg, "first.csv")
    assert original != analysis_fingerprint(content, {**cfg, "periods_per_year": 12}, "first.csv")
    assert original == analysis_fingerprint(content, dict(reversed(list(cfg.items()))), "first.csv")


def test_html_is_self_contained_escapes_labels_and_bundle_preserves_audit():
    content = synthetic_csv().replace(b"ASSET_A", b"<script>alert(1)</script>")
    cfg = config()
    cfg["study_id"] = '<img src=x onerror="alert(2)">'
    analysis, files = run_analysis(content, cfg, filename="<script>name</script>.csv")
    html = files["report.html"].decode()
    assert "<script>" not in html
    assert '<img src=x' not in html
    assert "&lt;script&gt;" in html
    assert "data:image/png;base64," in html
    assert '<script src=' not in html
    assert 'src="http' not in html
    manifest = json.loads(files["manifest.json"])
    assert manifest["source_sha256"] == hashlib.sha256(content).hexdigest()
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256(files[name]).hexdigest() == digest
    with zipfile.ZipFile(io.BytesIO(zip_exports(files))) as archive:
        assert set(archive.namelist()) == set(files)
        assert archive.read("config.json") == files["config.json"]


def test_risk_cli_refuses_overwrite_and_keeps_failures(tmp_path):
    source, cfg, output = tmp_path / "input.csv", tmp_path / "config.json", tmp_path / "run"
    source.write_bytes(synthetic_csv())
    cfg.write_text(json.dumps(config()))
    args = ["--input", str(source), "--config", str(cfg), "--out", str(output)]
    assert main(args) == 0
    snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
    assert str(tmp_path) not in (output / "manifest.json").read_text()
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2
    assert snapshot == {p.name: p.read_bytes() for p in output.iterdir()}
    source.write_text("bad CSV")
    failure = tmp_path / "failure"
    assert main(args[:-1] + [str(failure)]) == 1
    assert json.loads((failure / "manifest.json").read_text())["status"] == "error"
    assert (failure / "failure.json").is_file()

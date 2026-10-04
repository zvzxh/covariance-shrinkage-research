# V2 validation record

Local verification on 2026-10-05, Python 3.12 on macOS. The final local suite
passed **72 tests**, including three Streamlit AppTest flows. The wheel and
source distribution built successfully. GitHub CI status is reported separately
by the remote workflow; these statements describe local checks.

## Executed checks

- **Strict ingestion:** invalid dates/order/duplicates, malformed or missing
  values, return/price bounds, validation before filtering, equivalent price and
  return inputs, baseline audit and selected-return roundtrip.
- **Independent statistics:** common centered 1/n sample covariance versus NumPy,
  independent Ledoit–Wolf versus scikit-learn, and analytical Euler contributions.
  For covariance `[[0.0004,-0.0003],[-0.0003,0.0009]]` and weights `(0.9,0.1)`,
  period variance is `0.000279` and contribution shares are `(33/31,-2/31)`.
- **Degeneracy:** singular but positive portfolio risk, exact offsetting zero
  portfolio variance, exactly constant assets, undefined correlations and
  correlation invariance under very small/large covariance scales.
- **Time integrity:** rolling test variance independently recomputed with ddof=1;
  perturbing future observations leaves previously fitted predictions unchanged;
  partial and singleton final blocks are explicitly handled.
- **Audit and failures:** malicious user labels escaped in HTML, embedded image,
  ZIP integrity and manifest hashes, no overwrite on repeated CLI calls, and
  failure diagnostics retained with nonzero status.
- **Browser:** synthetic results and four downloads; adjusted-price analysis;
  invalid-date rejection; numeric custom weights and wrong-sum rejection;
  setting changes clear old outputs. A fingerprint regression separately checks
  filename, content and configuration sensitivity.
- **Actual browser review:** the project coordinator checked the default analysis,
  ZIP/config download, custom-weight state clearing, and an uploaded adjusted-price
  CSV yielding 125 returns from 126 price rows. This complements AppTest rather
  than claiming every browser/version was exercised.

## Recorded synthetic analysis

Command from the repository root (choose a new output path):

```bash
python -m shrinkage_research.risk_cli --input data/risk_demo_returns.csv --config configs/risk_demo.json --out work/risk-verification
```

The final code was rerun against the committed source/config. All non-manifest
output bytes match `results/risk_demo`; the manifest records run-specific timing.
The PNG was visually inspected for legible labels and the synthetic-data banner.
This is generated data, not a market backtest.

| Method | Annualized volatility | Numerical condition number |
|---|---:|---:|
| Centered sample (1/n) | 7.696392% | 10.015513 |
| Ledoit–Wolf | 7.601023% | 8.088491 |
| Diagonal sample (1/n) | 5.952178% | 2.781021 |

The run has 125 observations and four assets. Rolling validation uses 60 training
rows, 20 test rows and four valid test blocks; the last has five rows. The sample,
Ledoit–Wolf and diagonal variance MSEs are respectively `2.54952964e-11`,
`2.40622649e-11` and `6.33864051e-11`. These are errors against noisy future sample
variances, averaged equally across blocks. Their ordering is not an investment
claim or evidence of universal estimator superiority.

## Compatibility and reproduction

The V1 reference directory, original paper configuration, estimator, design,
experiment and legacy CLI have no changes relative to V1. Version/package metadata
and new V2 files are additive to the old research workflow. The remote CI has a
specific Python 3.12 job installing `[app,test]`, so browser tests do not all skip
because Streamlit is absent.

```bash
python -m pip install -e '.[app,test]'
python -m pytest -q
python -m build
```

Sources/config/data/docs and recorded examples are included in the source
archive. The wheel contains the installed CLI, browser app and packaged synthetic
generator. Automated tests and agent review do not establish production readiness,
verify a data vendor's adjustments, or replace the owner's understanding of the
methods and limitations.

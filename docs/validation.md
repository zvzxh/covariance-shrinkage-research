# Recorded local validation

Reference run prepared on 2026-10-05 (Asia/Singapore). The machine-readable
manifest uses UTC explicitly. Validation was performed with Python 3.12.14,
NumPy 2.5.3 and scikit-learn 1.9.1 on macOS ARM64; see the complete recorded
package information in `results/reference/manifest.json`.

Commands, from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m shrinkage_research.cli --config configs/paper_subset.json --out results/reference
python -m build
```

The reference output directory is now committed and cannot be overwritten by
the runner. Use a new `--out work/my-run` when reproducing the experiment.

Observed results:

- **42 tests passed**, covering manual and independent numerical oracles,
  invariance, degenerate and invalid data, extreme representable magnitudes,
  exact population moments, uncertainty calculations, reproducibility, failure
  retention and overwrite refusal.
- **14,000 simulation replications** across seven scenarios; every configured
  numerical/empirical gate passed. All paired loss rows are retained.
- Central sample risk **0.538948** (MC SE **0.002590**), compared with analytic
  **0.537500**; shrinkage risk **0.272912** (MC SE **0.000996**).
- Central PRIAL **49.362%** (paired delta-method MC SE **0.242 percentage points**).
- The reference CLI manifest records roughly **1.12 seconds** elapsed on this
  particular local environment. This is a measured run, not a cross-machine
  performance guarantee.
- The plot was visually inspected for readable labels, ranges and clipping.
- Both source distribution and wheel built successfully. Source archives include
  configs, documentation and reference outputs; the wheel contains the library
  and explicit-config runner.

An independent agent also checked the Gaussian risk formula, reconstructed
Monte Carlo standard errors from the saved losses, compared additional cases
with scikit-learn, and exercised gate failures and overwrite protection.
This is AI-assisted review, not a claim of independent human certification.

GitHub Actions are configured for Python 3.11, 3.12 and 3.13 unit tests and a
separate fixed-protocol run. Local verification does not establish that remote
jobs have passed; consult their actual status after the repository is pushed.

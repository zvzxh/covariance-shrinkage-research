# Source and implementation map

## Primary paper

Ledoit, O. and Wolf, M. (2004), *A well-conditioned estimator for large-dimensional
covariance matrices*, Journal of Multivariate Analysis 88(2), 365–411.

- DOI: <https://doi.org/10.1016/S0047-259X(03)00096-4>
- UZH author-hosted full text:
  <https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffceb83f14/wellCond.pdf>
- Author publication/code index:
  <https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html>

| Topic | Published page / location | Repository use |
|---|---|---|
| Known-zero-mean sample covariance; normalized Frobenius loss | p. 367, Section 2 | `S=X.T@X/n`; loss divided by `p` |
| Plug-in target and shrinkage | pp. 379–380, Lemmas 3.2–3.5 and Eq. (14) | Independent estimator |
| Computational identity | p. 384, Section 4.1.4 | Efficient fourth-moment numerator |
| Gaussian/lognormal design and central parameters | p. 384, Section 4.2 | Seven selected design-aligned scenarios |
| Central numerical references | p. 385, Table 2 | Descriptive reference row, not fitted targets |
| Parameter sweeps and conditioning | pp. 385–388, Sections 4.3–4.4 | Restricted parameter comparisons and diagnostics |

The PDF uses 47 pages; printed page 365 is PDF page 1. No complete paper text,
paper figures or author code is redistributed in this repository.

## Independent code references

- scikit-learn public API:
  <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html>
  Used only as a test oracle. `assume_centered=True` is required for the known-
  zero-mean simulation; otherwise sklearn subtracts the sample mean.
- Python reference linked from Michael Wolf's publication page:
  <https://github.com/pald22/covShrinkage/blob/main/cov1Para.py>
  Reviewed for target and normalization conventions; **not copied**. Its optional
  demeaning/effective-sample-size conventions must not be silently mixed with
  this simulation. Its file-level licensing remains that project's concern.

The exact Gaussian sample-risk identity is derived from the Wishart second
moment and explicitly displayed in `methodology.md`. It provides a separate
analytic check; numerical closeness to a published Monte Carlo row alone would
not validate an implementation.

Source access was checked during repository preparation on 2026-10-05. Package
versions used for actual numerical verification appear in each run manifest.

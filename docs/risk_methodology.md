# V2 risk analysis: definitions and boundaries

## Input and time selection

The input is UTF-8 CSV with `date` first and at least two unique, nonempty asset
names. ISO dates must be unique and strictly increasing. All cells must be finite
numbers, with returns greater than -1 or prices greater than zero. The entire
source is validated before selection: date filters do not hide bad rows.

For `adjusted_prices`, `r[t] = P[t] / P[t-1] - 1` is computed before date selection,
without any fill. The first price is a baseline. Inclusive date filters refer to
**return-period-end dates**, followed by the last-N filter. This means the first
selected return can use a price before the selected start date. The audit records
both the source baseline and the first selected return's baseline date. Adjustment
quality is the data provider/user's responsibility; the program cannot verify it.
At least two returns must remain. Rows need not be consecutive trading dates;
frequency and annualization are user choices, not inferred from calendar gaps.

Source bytes are hashed with SHA-256. The selected returns export preserves the
shared `date,asset...` schema used by Portfolio Walk-Forward Lab. Configuration,
selection, conversion, asset names, dates and software versions accompany exports.

## Covariance estimates

For n observations x[t] of p assets, set z[t] = x[t] - mean(x) and
`S = sum(z[t] z[t]') / n`. All three estimates use this same centering and
**1/n normalization**, not an unbiased 1/(n-1) estimate:

- `sample_mle`: S.
- `diagonal_sample_mle`: diag(diag(S)), treating sample cross-covariances as zero.
- `ledoit_wolf`: the repository's independent scaled-identity shrinkage estimator
  called with `center=True`; the target is trace(S)/p times the identity. See
  [the original estimator formulas](methodology.md).

Exactly constant columns are represented as exactly zero after centering to
avoid spurious risk from floating-point mean rounding. Matrix CSVs are in
**period return-squared units**, not annualized units. The diagonal estimate is
a sensitivity comparator; omitting correlation is not automatically prudent.
The observed mean is estimated from the same data, unlike the known-zero-mean
V1 simulation. No claim of Gaussian maximum-likelihood adequacy for market
returns follows from the `sample_mle` identifier.

## Portfolio risk and contributions

Weights are equal by default. Explicit weights must name every input asset,
be finite and nonnegative, and sum to one within 1e-10. They are not silently
renormalized. For covariance C, weights w and annualization factor A:

- Period variance: `v = w' C w`.
- Annualized volatility: `sigma = sqrt(A v)`.
- Euler component: `RC[i] = A w[i] (Cw)[i] / sigma`.
- Share of volatility: `RC[i] / sigma`.

Components sum to sigma and shares sum to one up to floating-point precision.
Negative components are legitimate hedging contributions and are not made
absolute. If v is zero, components and shares are undefined (`null`), while
volatility is zero. For example, perfectly offsetting returns can yield zero
portfolio risk even when individual assets vary. A singular matrix does not
necessarily imply zero portfolio risk.

Scaling variance by A assumes zero serial covariance across periods. The tool
does not estimate autocorrelation corrections or a forecasting model. These
outputs describe risk under the fitted covariance and chosen exposures; they
are neither expected returns nor buy/sell signals.

## Spectrum, correlation and numerical warnings

Eigenvalues come from the symmetric eigensolver. Numerical rank uses tolerance
`100 × p × machine_epsilon × max(abs(eigenvalues))` (with a tiny positive fallback
at an all-zero spectrum). A materially negative eigenvalue or portfolio variance
is rejected; a negative portfolio variance within floating-point tolerance is
clipped to zero with a warning. Numerical condition number is undefined/infinite
for a rank-deficient matrix and exported as `null` with a warning. A finite
condition number above 1e8 also triggers a warning; this is a numerical diagnostic,
not a universal economic threshold.

Correlations are C[i,j]/sqrt(C[i,i] C[j,j]). If either marginal variance is zero,
the correlation is undefined, including that asset's diagonal. It is exported
as `null`/an empty CSV cell rather than a fabricated correlation of zero or one.

## Rolling out-of-sample variance check

Each fold fits a fresh covariance estimate using exactly `train_window` rows
strictly before its test block. Training windows roll forward; test blocks are
non-overlapping and normally `test_window` rows long. Weights are fixed across
folds. For each fitted method, prediction is `w' C_train w`. For test rows compute
`y[t] = w' r[t]` and the realized sample variance with **ddof=1**.

This realized variance is a noisy observable proxy, not true population variance.
The test's estimated mean and denominator differ intentionally from the 1/n
training covariance. Annualized volatility displays use the square root of each
variance times A. The calculation uses fixed exposures per observation and does
not simulate drifting holdings, transaction costs, rebalancing or investable NAV.
Use the companion backtest project for those accounting questions.

When enabled, a final short test block with at least two observations is included
and labeled partial. A singleton is always skipped because ddof=1 variance is
undefined. When partial blocks are disabled, the entire short final block is
skipped. Skips are recorded; no valid folds is an error. Summary MAE and MSE
average **equally over valid folds**, not over observations, so small partial
blocks can have disproportionate noise. No method is selected automatically
based on these results; holdout selection or model tuning would require an
additional honest validation design.

## Reports, browser state and failure handling

Reports escape all user-controlled text and embed the PNG chart as a data URI.
ZIP members use fixed internal relative paths. Manifests record basenames and
hashes, not local absolute input paths. A successful run exports all three matrix
pairs, contributions, spectra, summary tables, config, audit and selected returns.
The plot displays at most 20 asset labels for readability; exported tables retain
all assets.

Browser results are bound to the exact source bytes, filename and configuration.
Changing any of these clears old results; a failed analysis does not leave the
previous report presented as current. Config imports are optional and override
controls while present. Source provenance follows the chosen demo/upload mode.
To repeat a run, use the original input with its saved config; an exported returns
file has already undergone conversion and selection.

The V2 CLI atomically creates a new output directory and refuses any existing
directory, including an empty one. Invalid inputs return exit status 1 and retain
`failure.json` and an error manifest. Existing-output refusal returns 2 without
modifying that directory. This contract is stricter than the unchanged V1 runner.

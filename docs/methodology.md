# Methodology and implementation boundary

## Target and notation

Let the rows of `X` be `n` independent observations in `R^p`, with **known mean
zero**, and let `Sigma` be their population covariance. Define

```text
S       = X.T @ X / n
m       = trace(S) / p
d2      = ||S - m I||_F^2 / p
b2_raw  = sum_k ||x_k x_k.T - S||_F^2 / (p n^2)
delta   = min(1, max(0, b2_raw / d2))
Sigma_LW = delta m I + (1 - delta) S
```

This is the plug-in scaled-identity estimator from Equation (14), with the
normalized squared Frobenius norm used in that paper. The factor `1/p` cancels
in the shrinkage ratio but must be retained in reported losses.

The implementation computes the same numerator as

```text
b2_raw = [mean_k (x_k.T x_k)^2 - ||S||_F^2] / (p n)
```

because `sum_k x_k x_k.T = n S`. It avoids allocating `n` separate `p x p`
outer products. Internally scaling observations by their maximum absolute value
reduces overflow risk; the result is rescaled without modifying the input array.
When `d2=0`, target and sample coincide, so `delta=0` is a harmless convention.
All-zero input returns a zero covariance. Nonfinite input and unrepresentable
output scales raise errors.

`center=True` subtracts the empirical mean and still uses `1/n`, matching the
maximum-likelihood convention in the independent scikit-learn oracle. **The
paper experiment always uses `center=False`.** Replacing this with `numpy.cov`
defaults or blindly dividing by `n-1` changes the experiment.

## Disclosed population construction

For each dimension `p`, a fixed normal shape vector `z` is generated from a
recorded seed. Define

```text
lambda_i(t) = exp(t z_i) / mean_j exp(t z_j)
```

Bisection chooses nonnegative `t` so that the realized variance of `lambda`
equals the configured `alpha2`. The resulting positive spectrum has mean 1.
`Sigma = diag(lambda)` remains fixed over observation replications. Gaussian
samples are `Z * sqrt(lambda)`, with independent standard-normal entries in `Z`.

This is a lognormal-shaped, moment-matched spectrum. Its eigenvalues are not
independent unconditional lognormal draws after normalization and calibration.
That distinction is intentional and disclosed. The published realized spectrum
and normalization details are unavailable. Do not describe this construction as
recovering the authors' exact data or random-number stream.

An observation stream is derived from the configured seed and a SHA-256 digest
of the scenario ID. Reordering or adding other scenario IDs therefore does not
change existing scenario observations. Spectra and observation streams are
separate. The exact spectrum is saved with every scenario result.

## Loss, uncertainty and independent benchmark

For estimator `A`, each replication's loss is `||A-Sigma||_F^2 / p`. Risk is the
sample mean of those losses, and its Monte Carlo SE is their sample standard
deviation (`ddof=1`) divided by `sqrt(R)`.

For Gaussian observations with known zero mean:

```text
E[||S-Sigma||_F^2 / p] = [trace(Sigma^2) + trace(Sigma)^2] / (p n).
```

With the central exact spectrum moments, this equals `(20 + 1 + 0.5)/40 =
0.5375`. This Wishart-moment identity is an analytic correctness check independent
of the published simulation's particular random draws.

Let `A = mean(L_sample)`, `B = mean(L_LW)` and `q=B/A`. Then

```text
PRIAL = 100 (1 - B/A)
MC_SE(PRIAL) = 100 sd(L_LW - q L_sample) / [A sqrt(R)].
```

Both estimators use the same observations. The paired expression preserves
their loss covariance; treating the two risk estimates as independent would be
incorrect. Error bars show `1.96` times this delta-method SE. These are approximate
Monte Carlo intervals conditional on the selected population, not confidence
intervals for market performance. Paired risk-reduction SEs use
`sd(L_sample-L_LW)/sqrt(R)` directly.

Sample condition numbers are recorded as null when rank deficiency is known or
the smallest eigenvalue is below a machine-precision threshold. This is kept
separate from shrinkage condition numbers and is not silently replaced by a
pseudoinverse. The identity-shrinkage eigenvalue transformation is used for
efficient diagnostics.

## Limitations

- Two estimators and seven scenarios constitute a subset, not the full set of
  competitors and plotted experiments in Section 4.
- Our 2,000 replications differ from the paper's 1,000. Published reference
  values are not pass/fail thresholds and are never used to choose seeds.
- Fixed population spectra omit uncertainty across alternative eigenvalue
  shapes. Repeated observations estimate conditional performance only.
- IID Gaussian observations omit serial dependence, stochastic volatility,
  changing correlations, outliers and many properties of financial returns.
- A lower covariance Frobenius loss does not automatically imply lower portfolio
  risk, better forecasting or profitability. There is no trading backtest here.
- Statistical gates can fail by Monte Carlo chance. A failure must be retained
  and investigated; repeatedly drawing new seeds until a pass is not acceptable.
- Numerical agreement can differ at roundoff scale between BLAS libraries and
  platforms. The config, package versions and spectra are retained; byte-for-byte
  image or floating-point equality across all platforms is not promised.

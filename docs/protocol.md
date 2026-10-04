# Fixed experiment and acceptance protocol

The first reference run uses `configs/paper_subset.json` without retuning its
seeds, scenarios or gate thresholds after examining results. The config hash and
all completed simulation outcomes accompany each run. Tests of code correctness
and empirical outcome checks are distinct.

## Five numerical acceptance checks

1. **Independent estimator correctness.** Unit tests compare a hand-calculated
   nontrivial fixture, the literal outer-product formula, and scikit-learn's
   `ledoit_wolf` at matched centering/normalization. The oracle tolerance is
   `rtol=1e-11`, `atol=1e-12`; there are known-zero-mean and centered cases,
   including `p>n`, `p=n` and one variable.
2. **Matrix invariants.** The result is symmetric, trace preserving and has
   shrinkage in `[0,1]`. Nondegenerate Gaussian scenarios must have positive
   shrinkage eigenvalues and finite condition numbers. Scaled and orthogonally
   rotated inputs must transform the estimator accordingly. Degenerate zero
   inputs are tested separately and are not required to be positive definite.
3. **Population specification.** Realized spectrum mean and variance match `1`
   and configured `alpha2` within `1e-12`. This establishes which population was
   actually simulated rather than relying on finite lognormal draws to have
   exactly their distributional moments.
4. **Analytic sample-risk identity.** In each scenario,
   `abs(simulated sample risk - analytic risk) <= 4 * MC_SE`.
   The central benchmark is `0.5375`. This statistical check is predeclared and
   can occasionally fail by chance; it is not a deterministic mathematical proof.
5. **Paired empirical comparison and auditable uncertainty.** For these selected
   scenarios, require `mean(L_sample-L_LW) - 1.96*MC_SE > 0`. Unit tests independently
   reconstruct the reported PRIAL SE from the two-loss covariance matrix and its
   Jacobian. A negative or uncertain improvement is preserved as a failed
   hypothesis outcome, not hidden by the pipeline.

The five per-scenario check keys in `summary.json` implement checks 2–5 plus a
separate finite-condition-number diagnostic. Check 1 and uncertainty-formula
correctness run in the unit-test suite. Passing these checks supports this
implementation and these populations; it is not a universal superiority claim.

## Published Table 2 comparison

For `p=20`, `n=40`, `alpha2=0.5`, the report also shows the published central
sample/shrinkage risks and PRIAL. These are **descriptive reference values**.
They are not numerical acceptance thresholds because the realized population
spectrum and original random stream are not published. Different realized
spectra can affect finite-sample shrinkage performance even when first two
moments agree. Never choose a spectrum or seed to match the published row.

## Failure behavior

- Results are checkpointed after each scenario.
- Every replication in each completed scenario has its paired losses and
  shrinkage value saved. An exception during a scenario preserves earlier
  completed scenarios; partial work inside the interrupted scenario is not
  checkpointed.
- Statistical gate failures produce `failure.json`, retain tables and plots,
  and return exit code 1 after the full planned run.
- Exceptions produce diagnostic `failure.json` plus `manifest.json` and keep
  completed scenario checkpoints.
- An existing nonempty output directory is refused with exit code 2.
- CI uploads experiment artifacts even when its run fails.

## Changing the protocol

Save a new config and study ID, state the reason for the change, and preserve
the previous results. Increasing replication counts, adding seeds, changing
populations or adding estimators must be distinguishable from the reference
experiment. Do not turn published risks into optimization targets.

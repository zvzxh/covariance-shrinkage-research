# Reproducible simulation results

Design-aligned subset; not an exact reconstruction of the authors' random draws.
Known zero mean, covariance divided by n. No market returns or trading profits are measured.

Effective configuration SHA-256: `ddf876477646f0739e2623c85d04361f55c6fdebb5b8ffea35b8318441609b97`

| Scenario | p | n | alpha² | Sample risk (MC SE) | Shrinkage risk (MC SE) | PRIAL % (MC SE, pp) | Gates |
|---|---:|---:|---:|---:|---:|---:|---|
| central | 20 | 40 | 0.5 | 0.538948 (0.002590) | 0.272912 (0.000996) | 49.362 (0.242) | pass |
| ratio_low | 10 | 80 | 0.5 | 0.142655 (0.001071) | 0.118925 (0.000840) | 16.635 (0.505) | pass |
| ratio_high | 40 | 20 | 0.5 | 2.061436 (0.006841) | 0.420537 (0.000685) | 79.600 (0.062) | pass |
| dispersion_low | 20 | 40 | 0.1 | 0.526944 (0.001785) | 0.091661 (0.000217) | 82.605 (0.059) | pass |
| dispersion_high | 20 | 40 | 2.0 | 0.578128 (0.004817) | 0.482283 (0.003953) | 16.579 (0.633) | pass |
| size_small | 10 | 20 | 0.5 | 0.570461 (0.005184) | 0.310456 (0.002123) | 45.578 (0.464) | pass |
| size_large | 40 | 80 | 0.5 | 0.518383 (0.001145) | 0.258375 (0.000394) | 50.158 (0.119) | pass |

The SEs describe simulation uncertainty conditional on each chosen population spectrum.
They do not include uncertainty over alternative spectra, financial data, or model misspecification.

## Published central reference

Table 2 reports sample risk 0.5372 (SE 0.0033), shrinkage risk 0.2723 (SE 0.0013),
and PRIAL 49.3% over 1,000 replications. Our configured experiment uses independent draws
and an explicitly calibrated spectrum. Published values are descriptive references, not fitted targets.

## Acceptance and retention

`summary.json` contains every gate, realized spectrum, condition-number diagnostic and reference difference.
`replications.csv` retains paired losses; `config.json` and `manifest.json` identify the run.
A failed gate leaves all results in place and causes a nonzero exit status. Inspect, do not change seeds to pass.

![Loss comparison](loss_comparison.png)

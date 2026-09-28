# Exact computation, simulation study, and prior rule

R code accompanying *Bayesian Inference for Incomplete 2×2 Diagnostic Tables*
(Antonijevic, Sitalo, and Vidakovic). All posteriors are computed exactly, by
listing every possible value of the missing denominator; no MCMC is needed.

Keep all files in this one folder: the scripts load each other by file name
(for example, `source("core.R")`). Set the working directory to this folder
before running, e.g. `setwd("path/to/exact_computation")`.

## Files

| Script | Produces |
|---|---|
| `core.R`, `single_row_fn.R` | Shared functions (loaded by the other scripts, not run directly) |
| `single_row_exact.R` | Single-row posterior tables for the breast MRI example (main paper) and the Svirsky example (Supplement S1) |
| `validate.R` | Joint-model (Model 3) check on the breast MRI table |
| `sim.R`, then `summarize.R` | Simulation study with counts only (main paper figure; Supplement S3) |
| `sim_reported.R` | Simulation study with a reported, rounded sensitivity |
| `addon.R` | Reconstruction with reported summaries: Wismueller and breast MRI examples |
| `prior_rule.R`, `prior_rule_joint.R` | Priors chosen by rule and prior sensitivity analysis (main paper; Supplement S5) |
| `run_bugs_vs_exact.R`, `incomplete2x2_model.txt` | WinBUGS implementation of the reported-summary model (Supplement S4) |
| `incomplete2x2.R`, `demo.R` | General tool for completing any incomplete 2×2 table from its counts and reported summaries |

## Running times

Most scripts finish in seconds. `sim.R`, `sim_reported.R`, `single_row_exact.R`
and `prior_rule.R` take several minutes each.

## Requirements

R (version 4.0 or later). `run_bugs_vs_exact.R` additionally needs WinBUGS with
the R package `R2WinBUGS`, or JAGS with the R package `rjags`; set `ENGINE` and
`BUGS_DIR` at the top of that script.

# Micro-Publication Experiment: baseline_short

- Name: `baseline_short`
- Description: Short baseline run with privileged substrate cues and isotropic medium switching.
- Tags: baseline, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`True`, viscosity=`True`
- Anisotropy mode: `off`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `100.6187`
- Mixed-phase success rate: `0.67`
- Mixed-phase mean transitions: `4.00`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 127.12 | 0.066 |
| Single Land Zone | `pass` | 2 | 1400 | 600 | 0.30 | 55.02 | 0.000 |
| Two Land Zones | `fail` | 0 | 2500 | 0 | 0.00 | 34.01 | 0.000 |
| Full Complexity | `pass` | 10 | 2495 | 505 | 0.17 | 186.33 | 0.000 |

## Runtime Notes
- This package is a parity-focused local wrapper around vendored curriculum trainer utilities.
- The anisotropic mode is still a rigid-link directional drag formulation, not a soft-body mechanics model.

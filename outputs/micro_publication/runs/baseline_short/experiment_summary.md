# Micro-Publication Experiment: baseline_short

- Name: `baseline_short`
- Description: Short baseline run with privileged substrate cues and isotropic medium switching.
- Tags: baseline, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`True`, viscosity=`True`
- Anisotropy mode: `off`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `90.2325`
- Mixed-phase success rate: `1.00`
- Mixed-phase mean transitions: `3.33`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 127.71 | 0.066 |
| Single Land Zone | `pass` | 4 | 1423 | 577 | 0.29 | 89.88 | 0.000 |
| Two Land Zones | `pass` | 5 | 2135 | 365 | 0.15 | 34.09 | 0.000 |
| Full Complexity | `pass` | 1 | 2400 | 600 | 0.20 | 109.25 | 0.000 |

## Runtime Notes
- This package is a parity-focused local wrapper around vendored curriculum trainer utilities.
- The anisotropic mode is still a rigid-link directional drag formulation, not a soft-body mechanics model.

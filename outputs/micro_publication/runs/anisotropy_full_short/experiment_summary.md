# Micro-Publication Experiment: anisotropy_full_short

- Name: `anisotropy_full_short`
- Description: Short run with full per-segment anisotropic drag model enabled.
- Tags: anisotropy_full, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`True`, viscosity=`True`
- Anisotropy mode: `full`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `96.4337`
- Mixed-phase success rate: `1.00`
- Mixed-phase mean transitions: `2.67`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 139.72 | 0.066 |
| Single Land Zone | `pass` | 2 | 1400 | 600 | 0.30 | 78.02 | 0.000 |
| Two Land Zones | `pass` | 1 | 1900 | 600 | 0.24 | 96.27 | 0.000 |
| Full Complexity | `pass` | 5 | 1783 | 1217 | 0.41 | 71.72 | 0.000 |

## Runtime Notes
- This package is a parity-focused local wrapper around vendored curriculum trainer utilities.
- The anisotropic mode is still a rigid-link directional drag formulation, not a soft-body mechanics model.

# Micro-Publication Experiment: anisotropy_proxy_short

- Name: `anisotropy_proxy_short`
- Description: Short run with directional drag proxy on land.
- Tags: anisotropy_proxy, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`True`, viscosity=`True`
- Anisotropy mode: `proxy`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `108.0762`
- Mixed-phase success rate: `1.00`
- Mixed-phase mean transitions: `2.67`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 127.22 | 0.066 |
| Single Land Zone | `pass` | 5 | 1173 | 827 | 0.41 | 80.90 | 0.000 |
| Two Land Zones | `pass` | 1 | 1800 | 700 | 0.28 | 110.03 | 0.000 |
| Full Complexity | `pass` | 2 | 2400 | 600 | 0.20 | 114.16 | 0.000 |

## Runtime Notes
- This package is a parity-focused local wrapper around vendored curriculum trainer utilities.
- The anisotropic mode is still a rigid-link directional drag formulation, not a soft-body mechanics model.

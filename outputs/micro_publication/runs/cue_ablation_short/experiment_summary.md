# Micro-Publication Experiment: cue_ablation_short

- Name: `cue_ablation_short`
- Description: Short run with privileged environment and viscosity observations hidden.
- Tags: cue_ablation, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`False`, viscosity=`False`
- Anisotropy mode: `off`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `86.1993`
- Mixed-phase success rate: `1.00`
- Mixed-phase mean transitions: `2.33`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 133.62 | 0.066 |
| Single Land Zone | `pass` | 2 | 1662 | 338 | 0.17 | 55.51 | 0.000 |
| Two Land Zones | `pass` | 1 | 1900 | 600 | 0.24 | 117.60 | 0.000 |
| Full Complexity | `pass` | 4 | 2400 | 600 | 0.20 | 38.06 | 0.000 |

## Runtime Notes
- This package is a parity-focused local wrapper around vendored curriculum trainer utilities.
- The anisotropic mode is still a rigid-link directional drag formulation, not a soft-body mechanics model.

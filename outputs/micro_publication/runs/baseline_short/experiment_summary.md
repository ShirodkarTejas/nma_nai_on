# Micro-Publication Experiment: baseline_short

- Name: `baseline_short`
- Description: Short baseline run with privileged substrate cues and isotropic medium switching.
- Tags: baseline, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`True`, viscosity=`True`
- Anisotropy mode: `off`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `109.1289`
- Mixed-phase success rate: `0.67`
- Mixed-phase mean transitions: `3.67`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 127.30 | 0.066 |
| Single Land Zone | `fail` | 0 | 2000 | 0 | 0.00 | 22.61 | 0.000 |
| Two Land Zones | `pass` | 2 | 1900 | 600 | 0.24 | 100.27 | 0.000 |
| Full Complexity | `pass` | 9 | 1796 | 1204 | 0.40 | 186.33 | 0.000 |

## Runtime Notes
- This package is a clean micro-publication path separated from the legacy curriculum code.
- The full anisotropic mode is implemented as a per-segment directional drag model in the new environment package.

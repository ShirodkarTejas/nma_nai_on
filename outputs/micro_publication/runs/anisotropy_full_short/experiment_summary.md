# Micro-Publication Experiment: anisotropy_full_short

- Name: `anisotropy_full_short`
- Description: Short run with full per-segment anisotropic drag model enabled.
- Tags: anisotropy_full, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`True`, viscosity=`True`
- Anisotropy mode: `full`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `88.0587`
- Mixed-phase success rate: `1.00`
- Mixed-phase mean transitions: `3.33`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 139.63 | 0.066 |
| Single Land Zone | `pass` | 3 | 1111 | 889 | 0.44 | 60.97 | 0.000 |
| Two Land Zones | `pass` | 4 | 2357 | 143 | 0.06 | 37.62 | 0.000 |
| Full Complexity | `pass` | 3 | 1800 | 1200 | 0.40 | 114.01 | 0.000 |

## Runtime Notes
- This package is a clean micro-publication path separated from the legacy curriculum code.
- The full anisotropic mode is implemented as a per-segment directional drag model in the new environment package.

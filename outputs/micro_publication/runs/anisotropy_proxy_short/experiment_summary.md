# Micro-Publication Experiment: anisotropy_proxy_short

- Name: `anisotropy_proxy_short`
- Description: Short run with directional drag proxy on land.
- Tags: anisotropy_proxy, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`True`, viscosity=`True`
- Anisotropy mode: `proxy`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `91.8399`
- Mixed-phase success rate: `0.67`
- Mixed-phase mean transitions: `1.00`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 127.95 | 0.066 |
| Single Land Zone | `fail` | 0 | 2000 | 0 | 0.00 | 22.58 | 0.000 |
| Two Land Zones | `pass` | 2 | 1900 | 600 | 0.24 | 137.39 | 0.000 |
| Full Complexity | `pass` | 1 | 2125 | 875 | 0.29 | 79.44 | 0.000 |

## Runtime Notes
- This package is a clean micro-publication path separated from the legacy curriculum code.
- The full anisotropic mode is implemented as a per-segment directional drag model in the new environment package.

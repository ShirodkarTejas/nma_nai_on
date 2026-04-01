# Micro-Publication Experiment: cue_ablation_short

- Name: `cue_ablation_short`
- Description: Short run with privileged environment and viscosity observations hidden.
- Tags: cue_ablation, short_run, micro_publication
- Training steps: `20000`
- Observation cues: environment=`False`, viscosity=`False`
- Anisotropy mode: `off`

## Summary
- Mean phase distance: `0.0164`
- Mean phase reward: `84.8012`
- Mixed-phase success rate: `1.00`
- Mixed-phase mean transitions: `7.00`

## Phase Metrics
| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |
|---|---|---:|---:|---:|---:|---:|---:|
| Pure Swimming | `pass` | 0 | 1500 | 0 | 0.00 | 134.08 | 0.066 |
| Single Land Zone | `pass` | 8 | 1041 | 959 | 0.48 | 86.52 | 0.000 |
| Two Land Zones | `pass` | 8 | 1588 | 912 | 0.36 | 38.86 | 0.000 |
| Full Complexity | `pass` | 5 | 2819 | 181 | 0.06 | 79.75 | 0.000 |

## Runtime Notes
- This package is a clean micro-publication path separated from the legacy curriculum code.
- The full anisotropic mode is implemented as a per-segment directional drag model in the new environment package.

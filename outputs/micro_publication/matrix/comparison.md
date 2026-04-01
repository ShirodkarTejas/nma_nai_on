# Micro-Publication Matrix Comparison

| Experiment | Mean Phase Distance | Mean Phase Reward | Mixed Success | Mean Transitions | Notes |
|---|---:|---:|---:|---:|---|
| `baseline_short` | 0.0164 | 109.1289 | 0.67 | 3.67 | Privileged substrate cues on, isotropic medium switching. |
| `cue_ablation_short` | 0.0164 | 84.8012 | 1.00 | 7.00 | Privileged environment and viscosity cues hidden. |
| `anisotropy_proxy_short` | 0.0164 | 91.8399 | 0.67 | 1.00 | Directional drag proxy on land. |
| `anisotropy_full_short` | 0.0164 | 88.0587 | 1.00 | 3.33 | Full per-segment directional drag in both media. |

## Phase Comparison
| Experiment | P1 | P2 | P3 |
|---|---|---|---|
| `baseline_short` | fail / t=0 / land=0.00 | pass / t=2 / land=0.24 | pass / t=9 / land=0.40 |
| `cue_ablation_short` | pass / t=8 / land=0.48 | pass / t=8 / land=0.36 | pass / t=5 / land=0.06 |
| `anisotropy_proxy_short` | fail / t=0 / land=0.00 | pass / t=2 / land=0.24 | pass / t=1 / land=0.29 |
| `anisotropy_full_short` | pass / t=3 / land=0.44 | pass / t=4 / land=0.06 | pass / t=3 / land=0.40 |

## Run Folders
- `baseline_short`: `outputs/micro_publication/runs/baseline_short`
- `cue_ablation_short`: `outputs/micro_publication/runs/cue_ablation_short`
- `anisotropy_proxy_short`: `outputs/micro_publication/runs/anisotropy_proxy_short`
- `anisotropy_full_short`: `outputs/micro_publication/runs/anisotropy_full_short`

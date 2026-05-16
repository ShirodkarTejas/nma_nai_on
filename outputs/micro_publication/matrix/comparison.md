# Micro-Publication Matrix Comparison

| Experiment | Mean Phase Distance | Mean Phase Reward | Mixed Success | Mean Transitions | Notes |
|---|---:|---:|---:|---:|---|
| `baseline_short` | 0.0164 | 100.6187 | 0.67 | 4.00 | Privileged substrate cues on, isotropic medium switching. |
| `cue_ablation_short` | 0.0164 | 86.1993 | 1.00 | 2.33 | Privileged environment and viscosity cues hidden. |
| `anisotropy_proxy_short` | 0.0164 | 108.0762 | 1.00 | 2.67 | Directional drag proxy on land. |
| `anisotropy_full_short` | 0.0164 | 96.4337 | 1.00 | 2.67 | Full per-segment directional drag in both media. |

## Phase Comparison
| Experiment | P1 | P2 | P3 |
|---|---|---|---|
| `baseline_short` | pass / t=2 / land=0.30 | fail / t=0 / land=0.00 | pass / t=10 / land=0.17 |
| `cue_ablation_short` | pass / t=2 / land=0.17 | pass / t=1 / land=0.24 | pass / t=4 / land=0.20 |
| `anisotropy_proxy_short` | pass / t=5 / land=0.41 | pass / t=1 / land=0.28 | pass / t=2 / land=0.20 |
| `anisotropy_full_short` | pass / t=2 / land=0.30 | pass / t=1 / land=0.24 | pass / t=5 / land=0.41 |

## Run Folders
- `baseline_short`: `outputs/micro_publication/runs/baseline_short`
- `cue_ablation_short`: `outputs/micro_publication/runs/cue_ablation_short`
- `anisotropy_proxy_short`: `outputs/micro_publication/runs/anisotropy_proxy_short`
- `anisotropy_full_short`: `outputs/micro_publication/runs/anisotropy_full_short`

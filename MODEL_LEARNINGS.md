# Model Learnings & Design Decisions

## Reward Shaping

### The Stuck-Worm Problem
The agent was "shivering" instead of swimming.  Root cause: the primary `tolerance`
reward with a `linear` sigmoid gives a non-zero reward for even tiny positive forward
velocities (e.g. v=0.01 → reward ≈ 0.2).  High-frequency micro-contractions ("twitching")
could produce this minimal forward velocity at very low energy cost, so PPO happily
converged on twitching as the optimal policy.

### Applied Fixes (mixed_environment.py → get_reward)

| Component | Before | After | Rationale |
|---|---|---|---|
| Forward momentum bonus | none | +0.8 * v when v > 0.05 | Rewards *sustained* motion only; single-step twitching never passes the 0.05 threshold |
| Activity penalty (joint vel^2) | 0.0005 | 0.002 | 4x harder to profit from rapid joint oscillation |
| Soft torque penalty | 0.0001 | 0.001 | 10x more expensive to apply large torques |
| Hard torque penalty | none | -0.05 * sum(max(0, abs(ctrl)-0.8)^2) | Cliff penalty for any joint exceeding 0.8; discourages extreme contractions |

### Colleague Code Analysis (nma_nai_on / logs/)
The colleague's harness (logs/baseline/script.py, logs/nmap/script.py) used the same
ImprovedMixedSwimmerEnv reward function as us — no additional anti-twitching shaping.
Their key contributions were:
- MetricSwimmerTonicEnv: logs episode_distance and forward_velocity as explicit
  training metrics (good for monitoring).
- LocomotionTrainer: a Tonic Trainer subclass that stores per-episode physical metrics.

## Physics Calibration

### Water Viscosity
- Previous: log-uniform random in [1e-5, 0.3] each episode.
- Problem: extremely thick episodes (visc ~0.3) make it physically impossible for the
  6-link worm to generate thrust; the agent learns nothing in those episodes.
- Fix: Fixed at 0.005 — thin enough that undulatory strokes produce measurable
  displacement, thick enough that the medium provides meaningful resistance feedback.

## Training Scale

- training_steps default in SwimmerTrainer updated from 500 000 -> 2 000 000.
- run_experiments.py passes --training_steps 2000000 for all four ablation conditions.

## Directory Structure

```
NCAP/ (Archive root)
├── external/          <- PDFs, docx, ConnectomeToolbox, CElegansNeuroML
└── NMAP/         <- GitHub repo root
    ├── connectome_priors/
    ├── swimmer/
    │   ├── environments/
    │   ├── models/
    │   ├── training/
    │   └── utils/
    ├── tonic/          <- local Tonic fork (moved from NCAP/tonic)
    ├── results/        <- experiment outputs (moved from NCAP/results)
    ├── gym_bridge.py   <- MUST be first import in every entry point
    ├── main.py
    ├── run_experiments.py
    ├── README.md
    └── MODEL_LEARNINGS.md
```

## Entry Point Import Order (CRITICAL)
Every entry point script must import gym_bridge BEFORE any other package:
```python
try:
    import NMAP.gym_bridge  # noqa: F401  (when run from NCAP/)
except ModuleNotFoundError:
    import gym_bridge             # noqa: F401  (when run from inside NMAP/)
```
This patches the gym namespace so tonic and our environments see a consistent API.

## NCAP Prior Integration (NMAP)
- sparse_init=True applies Cook2019 6-pathway sparse weight initialization.
- sparse_reg_lambda=0.05 adds a topological prior loss term to the actor update.
- force_oscillation=True penalises low action variance, preventing policy collapse.
- prior_modulation_scale=0.15 scales the connectome-derived weight modulation.

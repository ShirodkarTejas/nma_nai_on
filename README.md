# NMAP — Neural Motor Architectural Priors

**Biologically-grounded locomotion control via C. elegans connectomics**

The NMAP project extends the [NCAP architecture](https://arxiv.org/abs/2201.05242) with connectome-informed weight initialization and topological regularization derived from the Cook 2019 *C. elegans* connectome. A 4-step ablation study isolates the contribution of each biological prior.

---

## Setup

```bash
# One-shot (creates conda env `nmap`, installs all deps, patches Tonic)
bash setup_env.sh
conda activate nmap
```

External data files required (not committed):

| File | Location |
|------|----------|
| Cook 2019 connectome XLSX | `../external/ConnectomeToolbox/cect/data/SI 5 Connectome adjacency matrices.xlsx` |
| c302 NML geometry | `../external/ConnectomeToolbox/cect/data/c302_C2_FW.net.nml` |

---

## Running the ablation study

```bash
cd NMAP
python run_experiments.py
```

This runs four sequential training jobs (1 M steps each), writes results to `results/{01..04}_*/`, then automatically calls `results/plot_ablation.py` to produce six analysis plots and `results/RESULTS.md`.

### What each ablation step adds

| Step | Directory | Extra flags |
|------|-----------|-------------|
| 01 | `01_baseline` | — |
| 02 | `02_oscillation` | `--force_oscillation` |
| 03 | `03_initialization` | `--force_oscillation --sparse_init` |
| 04 | `04_full_nmap` | `--force_oscillation --sparse_init --sparse_reg_lambda 0.05` |

### Re-running analysis only

```bash
python results/plot_ablation.py   # regenerates plots + RESULTS.md from existing checkpoints
```

### Manual single-step invocation

```bash
python main.py --mode train_curriculum --training_steps 1000000 --algorithm ppo --n_links 5 \
    --use_locomotion_only_early_training \
    --force_oscillation --sparse_init --sparse_reg_lambda 0.05 \
    --log_dir results/04_full_nmap

# Resume from checkpoint
python main.py --mode train_curriculum --training_steps 1000000 \
    --resume_checkpoint results/04_full_nmap/checkpoints/checkpoint_step_500000.pt \
    --log_dir results/04_full_nmap

# Evaluation only (no training)
python main.py --mode evaluate_curriculum \
    --resume_checkpoint results/04_full_nmap/checkpoints/checkpoint_step_1000000.pt \
    --eval_episodes 20 --eval_video_steps 400
```

---

## Key CLI flags

| Flag | Effect |
|------|--------|
| `--force_oscillation` | Adds a variance penalty `scale × max(0, min_var − Var(actions))` to the actor loss; prevents policy collapse to near-zero ("curl-and-shiver") |
| `--sparse_init` | Scales initial NCAP weights by Cook 2019 synapse counts × `prior_modulation_scale` (0.15); also forces `use_weight_sharing=False` |
| `--sparse_reg_lambda λ` | Topological L2 regularization: `λ × Σ dist_p × ‖w_p‖²`; `dist_p` is normalized soma distance from c302 geometry |
| `--model_type` | `enhanced_ncap` (default) or `biological_ncap` |
| `--log_dir PATH` | Root for checkpoints, logs, videos, plots |
| `--n_links N` | Body segments (default 5) |
| `--algorithm` | `ppo` (default) or `a2c` |

---

## Architecture

### Model hierarchy

```
BiologicalNCAPSwimmer          (models/biological_ncap.py)          — 9 params, sign-constrained
EnhancedBiologicalNCAPSwimmer  (models/enhanced_biological_ncap.py) — adds relaxation oscillator + goal nav
NCAPSwimmer                    (models/ncap_swimmer.py)              — DEPRECATED (LSTM-based)
SimpleNCAPSwimmer              (models/simple_ncap.py)               — legacy 4-param version
```

Both active models expose:

```python
model.configure_sparse_priors(scalars: dict)        # store {dist_*, syn_*} from connectome
model.compute_topological_prior_loss(lambda_val)    # λ × Σ dist_p × ‖w_p‖² for all pathways
```

Sign constraints: `excitatory()` clamps ≥ 0; `inhibitory()` clamps ≤ 0; `graded()` clamps activations to [0, 1].

### Trainer hierarchy

```
SwimmerTrainer             (training/swimmer_trainer.py)     — base; holds sparse prior cache
  └─ NCAPTrainer           (training/ncap_trainer.py)        — --mode train
  └─ CurriculumNCAPTrainer (training/curriculum_trainer.py)  — --mode train_curriculum
```

`custom_tonic_agent.py` wraps Tonic PPO/A2C with:
- `prior_reg_lambda` — topological L2 added in `update()`
- `force_oscillation` — variance penalty added to actor loss

### Connectome pathway map

`generate_ncap_segment_priors()` in `connectome_priors/swimmer_priors.py` extracts six pathway averages:

| Key | Biology | Synapse type |
|-----|---------|--------------|
| `ipsi_db` | DB → dorsal muscle | chemical (excitatory) |
| `ipsi_vb` | VB → ventral muscle | chemical (excitatory) |
| `contra_db` | DB → DD → ventral muscle | series conductance `N_eff = N1·N2/(N1+N2)` |
| `contra_vb` | VB → VD → dorsal muscle | series conductance |
| `next_db` | DB_i ↔ DB_{i+1} | gap junction (electrical) |
| `next_vb` | VB_i ↔ VB_{i+1} | gap junction (electrical) |

Returns per pathway:
- `syn_*` — average synapse count (weight-initialization scale)
- `dist_*` — average soma distance normalized to [0, 1] (regularization weight; longer-range = stronger penalty)

### Environment

```
dm_control Swimmer (MuJoCo)
  └─ MixedEnvironmentSwim        (environments/mixed_environment.py)
       ├─ water viscosity: 0.005 (fixed)
       ├─ 2 land islands at ±1.5 x, radius shrinks over training
       └─ get_reward():
            primary  : 3.0 × tolerance(v_forward, bounds=(0.15,∞), margin=0.15)
            bonus    : +0.8 × v_forward  when v_forward > 0.05
            penalty  : −0.002 × Σ joint_vel²   (activity)
            penalty  : −0.001 × vis_scale × Σ ctrl²  (soft torque)
            penalty  : −0.05 × Σ max(0, |ctrl|−0.8)²  (hard torque)
            milestone: +0.1 × (dist_from_start // 0.5)
  └─ TonicSwimmerWrapper / VectorizedEnv
```

### Curriculum phases

| Phase | Steps (of 1 M) | Environment |
|-------|----------------|-------------|
| 1 | 0–250k | Pure swimming (locomotion only) |
| 2 | 250k–500k | Single land island |
| 3 | 500k–750k | Two land islands |
| 4 | 750k–1 M | Full complexity |

### Critical import rule

`gym_bridge.py` must be the first import in every entry-point script. It patches `sys.modules["gym"]` → `gymnasium` so Tonic and the environment wrappers coexist.

```python
try:
    import NMAP.gym_bridge
except ModuleNotFoundError:
    import gym_bridge
```

---

## Interpreting results

After `run_experiments.py` completes, open `results/RESULTS.md` for the quantitative summary.

The six plots in `results/analysis_plots/` show:

| Plot | What to look for |
|------|-----------------|
| `01_learning_curves.png` | Reward and distance over training. 03/04 should rise faster than 01/02 if sparse init helps. |
| `02_evaluation_performance.png` | Per-phase bar chart at end of training. Compare 04 vs 03 to see whether regularization helps or hurts in each phase. |
| `03_weight_evolution.png` | Parameter trajectories over checkpoints. 01/02 (shared weights) will plateau near ±1; 03/04 (per-joint weights) should spread and learn. |
| `04_phase_breakdown.png` | Radar chart of phase-wise performance per condition. Balanced coverage = robust generalization. |
| `05_prior_diagnostics.png` | Histogram of prior scalars and their variance. Flat histogram = priors are uninformative; spread = priors are guiding the search. |
| `06_summary_table.png` | Condensed comparison table suitable for presentations. |

Key expected finding: adding `--sparse_init` (step 03) breaks weight-sharing symmetry and enables gradient flow; `--sparse_reg_lambda` (step 04) adds anatomical inductive bias that may help generalization but can over-constrain Phase 3.

---

## Tests

```bash
# Connectome priors smoke tests (fast, no training)
cd NMAP && python -m pytest connectome_priors/tests/ -v

# Curriculum rollout test
python tests/test_curriculum_setup.py
```

---

## Result layout

```
results/
├── 01_baseline/
│   ├── checkpoints/   — .pt snapshots every 50k steps
│   ├── logs/          — train.csv, log.csv
│   ├── tonic/         — Tonic logger output
│   └── videos/        — rendered MP4s
├── 02_oscillation/    (same structure)
├── 03_initialization/ (same structure)
├── 04_full_nmap/      (same structure)
├── analysis_plots/    — 01–06 PNG plots (auto-generated)
└── RESULTS.md         — quantitative summary (auto-generated)
```

---

## References

- **NCAP**: Bhatt & Bhattacharyya (2022). *Neural circuit architectural priors for embodied control.* [arXiv:2201.05242](https://arxiv.org/abs/2201.05242)
- **Cook et al. 2019**: Sub-millimeter resolution *C. elegans* connectome. Chemical synapse + gap-junction adjacency matrices.
- **OpenWorm c302**: 3D neuron geometry — `c302_C2_FW.net.nml`
- **Tonic RL**: [github.com/fabiopardo/tonic](https://github.com/fabiopardo/tonic)
- **DeepMind Control Suite**: [github.com/deepmind/dm_control](https://github.com/deepmind/dm_control)

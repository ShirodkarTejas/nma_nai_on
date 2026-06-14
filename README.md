# NMAP — Neural Motor Architectural Priors

**Biologically-grounded locomotion control via *C. elegans* connectomics**

The NMAP project extends the [NCAP architecture](https://arxiv.org/abs/2201.05242) with connectome-informed weight initialization and topological regularization derived from the Cook 2019 *C. elegans* connectome. A 5-step ablation study isolates the contribution of each biological prior, and a post-hoc trajectory-tangling analysis asks whether the neuromodulatory broadcast from the high-level controller physically separates locomotor attractors across viscosity regimes.

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

## Ablation Study — 5 steps

The full ablation progressively adds each biological prior to an NCAP baseline:

| Step | Directory | Flags added | What it tests |
|------|-----------|-------------|---------------|
| 01 | `results/01_ncap_baseline/` | — | Bare NCAP with relaxation oscillator; shared weights |
| 02 | `results/02_ncap_sparse_init/` | `--sparse_init` | Cook 2019 synapse-count initialization; forces per-joint weights |
| 03 | `results/03_ncap_reg_only/` | `--sparse_reg_lambda 0.005` | Topological L2 regularization only; shared weights |
| 04 | `results/04_ncap_full_priors/` | `--sparse_init --sparse_reg_lambda 0.005` | Both priors together |
| 05 | `results/05_ncap_full_nmap/` | `--sparse_init --sparse_reg_lambda 0.005` | Full NMAP; per-joint weights + both priors |

> **Note:** Steps 04 and 05 use the same flags but differ in training runs and random seeds; results are averaged across the lambda sweep to identify the best operating point.

### Running a single ablation step

```bash
cd NMAP

# Step 01 — baseline
python main.py --mode train_curriculum --training_steps 1000000 \
    --algorithm ppo --n_links 6 --model_type enhanced_ncap \
    --use_locomotion_only_early_training \
    --log_dir results/01_ncap_baseline

# Step 02 — sparse init
python main.py --mode train_curriculum --training_steps 1000000 \
    --algorithm ppo --n_links 6 --model_type enhanced_ncap \
    --use_locomotion_only_early_training --sparse_init \
    --log_dir results/02_ncap_sparse_init

# Step 03 — regularization only
python main.py --mode train_curriculum --training_steps 1000000 \
    --algorithm ppo --n_links 6 --model_type enhanced_ncap \
    --use_locomotion_only_early_training --sparse_reg_lambda 0.005 \
    --log_dir results/03_ncap_reg_only

# Step 04 — both priors
python main.py --mode train_curriculum --training_steps 1000000 \
    --algorithm ppo --n_links 6 --model_type enhanced_ncap \
    --use_locomotion_only_early_training \
    --sparse_init --sparse_reg_lambda 0.005 \
    --log_dir results/04_ncap_full_priors

# Step 05 — full NMAP
python main.py --mode train_curriculum --training_steps 1000000 \
    --algorithm ppo --n_links 6 --model_type enhanced_ncap \
    --use_locomotion_only_early_training \
    --sparse_init --sparse_reg_lambda 0.005 \
    --log_dir results/05_ncap_full_nmap

# Resume from checkpoint
python main.py --mode train_curriculum --training_steps 1000000 \
    --resume_checkpoint results/04_ncap_full_priors/curriculum_training/checkpoints/enhanced_ncap/<name>.pt \
    --sparse_init --sparse_reg_lambda 0.005 \
    --log_dir results/04_ncap_full_priors

# Evaluation only (no training)
python main.py --mode evaluate_curriculum \
    --resume_checkpoint results/05_ncap_full_nmap/curriculum_training/checkpoints/enhanced_ncap/<name>.pt \
    --eval_episodes 20 --eval_video_steps 400 \
    --log_dir results/05_ncap_full_nmap
```

---

## Key CLI Flags

| Flag | Effect |
|------|--------|
| `--sparse_init` | Scales initial NCAP weights by Cook 2019 synapse counts × `prior_modulation_scale` (0.15); forces `use_weight_sharing=False` so each body segment gets its own parameters |
| `--sparse_reg_lambda λ` | Topological L2 regularization: `λ × Σ dist_p × ‖w_p‖²`; `dist_p` is normalized soma distance from c302 geometry (longer-range connections penalized more) |
| `--force_oscillation` | Adds variance penalty `scale × max(0, min_var − Var(actions))` to actor loss; prevents policy collapse to near-zero ("curl-and-shiver") |
| `--model_type` | `enhanced_ncap` (default) or `biological_ncap` |
| `--use_locomotion_only_early_training` | Disables goal-directed navigation for the first 30% of training so the oscillator stabilizes before complex tasks are introduced |
| `--log_dir PATH` | Root for checkpoints, logs, videos, plots |
| `--n_links N` | Body segments (default 6) |
| `--algorithm` | `ppo` (default) or `a2c` |

---

## Architecture

### Model hierarchy

```
BiologicalNCAPSwimmer          (models/biological_ncap.py)          — 9 params, sign-constrained
EnhancedBiologicalNCAPSwimmer  (models/enhanced_biological_ncap.py) — adds relaxation oscillator + env adaptation
NCAPSwimmer                    (models/ncap_swimmer.py)              — DEPRECATED (LSTM-based)
SimpleNCAPSwimmer              (models/simple_ncap.py)               — legacy 4-param version
```

Both active models expose the connectome-prior API:

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
- `prior_reg_lambda` — topological L2 term added in `update()`
- `force_oscillation` — variance penalty added to actor loss

### Connectome priors — what the math does

`generate_ncap_segment_priors()` in `connectome_priors/swimmer_priors.py` extracts six pathway averages from Cook 2019:

| Key | Biology | Synapse type |
|-----|---------|--------------|
| `ipsi_db` | DB → dorsal muscle | chemical (excitatory) |
| `ipsi_vb` | VB → ventral muscle | chemical (excitatory) |
| `contra_db` | DB → DD → ventral muscle | series conductance `N_eff = N1·N2/(N1+N2)` |
| `contra_vb` | VB → VD → dorsal muscle | series conductance |
| `next_db` | DB_i ↔ DB_{i+1} | gap junction (electrical) |
| `next_vb` | VB_i ↔ VB_{i+1} | gap junction (electrical) |

Returns per pathway:
- `syn_*` — average synapse count → weight initialization scale
- `dist_*` — average soma distance normalized to [0, 1] → regularization weight (`--sparse_reg_lambda`)

**Sparse init** (`--sparse_init`): multiplies initial weights by `syn_* × 0.15`. Critically, this also forces `use_weight_sharing=False`, giving each body segment its own parameter set. Shared weights create a symmetric loss landscape where gradient flow is flat; per-joint weights break this symmetry.

**Sparse reg** (`--sparse_reg_lambda λ`): adds `λ × dist_* × w²` to the actor loss for each pathway parameter. Implemented via `model.compute_topological_prior_loss(lambda_val)`.

### EnhancedBiologicalNCAPSwimmer — neuromodulatory broadcast

`EnhancedBiologicalNCAPSwimmer` receives an `environment_type` vector `[water_flag, land_flag, viscosity_norm]` at every forward pass. This acts as a **neuromodulatory broadcast** to the entire locomotor circuit:

| Signal | Effect |
|--------|--------|
| `water_frequency_scale` (learned, init 2.5×) | Speeds up the relaxation oscillator in water |
| `land_frequency_scale` (learned, init 0.5×) | Slows the oscillator on land |
| `water_amplitude_scale` / `land_amplitude_scale` | Scales muscle activation amplitude per medium |
| `environment_modulation` (±0.1) | Globally biases all B-neuron and muscle weights |

These four parameters are learned jointly with the motor weights and constitute the model's environmental neuromodulation system — analogous to monoamine modulation in biological CPGs.

### Curriculum phases

| Phase | Steps (of 1 M) | Environment |
|-------|----------------|-------------|
| 1 | 0–300k | Pure swimming; locomotion-only mode |
| 2 | 300k–600k | Single land island introduced |
| 3 | 600k–900k | Two land islands |
| 4 | 900k–1 M | Full complexity; goal-directed navigation enabled |

### Micro-publication environment

`swimmer/environments/micro_publication_env.py` provides a cleaner, publication-oriented mixed-media task:

- **Reward**: `compute_navigation_reward()` in `micro_publication_rewards.py` — principled decomposition of forward progress, target proximity, and energy cost
- **Observation**: joints + body velocities + environment type (water/land flag + viscosity) + goal direction + time feature
- Land zones expand with training progress; land-start probability 35%
- Used for all 5 ablation runs in this study

### Critical import rule

`gym_bridge.py` must be the **first import** in every entry-point. It patches `sys.modules["gym"]` → `gymnasium` so Tonic and the environment wrappers coexist.

```python
try:
    import NMAP.gym_bridge
except ModuleNotFoundError:
    import gym_bridge
```

---

## Connectomic Priors — Quantitative Results

All models trained for 1 M steps with the curriculum trainer on the micro-publication environment (6-link swimmer, PPO, oscillator period 60). Reported metrics are from the final checkpoint (step 1 M).

### Ablation performance summary

*(Full plots in `results/analysis_plots/` after running `python results/plot_ablation.py`)*

Key trend: sparse initialization (step 02) breaks weight-sharing symmetry and enables gradient flow; regularization (step 03) adds anatomical inductive bias. Together (steps 04–05) they improve generalization but can over-constrain Phase 3 (two land islands) depending on λ.

### Lambda sweep

A sweep over `sparse_reg_lambda` in `[0.0001, 0.5]` was run with the full prior combination. Results in `results/sweep_nmap_lambda_*/`. Optimal λ ≈ 0.005 balances regularization strength against policy flexibility.

---

## Trajectory Tangling Analysis

**Script:** `tangling_analysis.py`  
**Output:** `results/tangling_analysis/` (7 figures)

### Scientific question

Does the neuromodulatory broadcast (the `environment_type` signal that scales oscillator frequency and muscle amplitude) physically **separate the locomotor attractors** for water, transition, and land — reducing trajectory tangling — or does it only create temporal (speed) separation while the underlying state-space geometry remains shared?

### Method

The model is run in closed-loop (self-driving via a spring-damper integrator) for 600 steps (10 oscillator cycles) under each of three viscosity conditions:

| Condition | `env_type` vector | Effective freq. scale |
|-----------|-------------------|-----------------------|
| Water | `[1.0, 0.0, 0.10]` | `water_frequency_scale` (≈ 2.5×) |
| Transition | `[0.5, 0.5, 0.40]` | interpolated |
| Land | `[0.0, 1.0, 0.80]` | `land_frequency_scale` (≈ 0.5×) |

The **hidden state** per timestep is a 25-dimensional vector:

```
h(t) ∈ ℝ²⁵ = [osc_d, osc_v,                   # 2 — relaxation oscillator output
               bneuron_d[0..4], bneuron_v[0..4], # 10 — B-neuron activations per joint
               muscle_d[0..4], muscle_v[0..4],   # 10 — muscle activations per joint
               env_mod, freq_scale, amp_scale]    # 3 — neuromodulatory state
```

**Trajectory tangling Q(τ)** (Russo et al., *Nat. Neurosci.* 2018):

```
Q(τ) = max_{τ'≠τ, |τ-τ'|>Δ}  ‖ẋ(τ) − ẋ(τ')‖² / (‖x(τ) − x(τ')‖² + ε)
```

where `ẋ(τ) = x(τ+1) − x(τ)`. Low Q within a condition = clean cyclical attractor. Low Q across conditions (merged trajectory) = well-separated attractors (large denominator from distant states).

**Attractor centroid distance** is also computed in standardized hidden-state space (Euclidean distance between per-condition trajectory means).

### Quantitative results

| Ablation | Q water | Q trans. | Q land | Q cross-cond. | Sep(W–L) |
|----------|---------|----------|--------|---------------|----------|
| 01 — Baseline (shared, no priors) | 76.1 | 84.4 | 127.0 | 86.3 | 3.311 |
| 02 — Sparse Init (per-joint) | 57.0 | 58.4 | 81.4 | 59.8 | 3.620 |
| 03 — Reg Only (shared+reg) | **30.9** | **15.7** | 79.3 | **16.7** | **4.077** |
| 04 — Full Priors (init+reg) | **21.5** | 49.2 | **56.7** | 51.7 | 3.560 |
| 05 — Full NMAP | 53.3 | 53.8 | 53.2 | 55.8 | 3.569 |

All Q values smoothed over 25-step window; Sep(W–L) = Euclidean centroid distance in standardized units.

### Interpretation

**1. Neuromodulation creates spatial attractor separation, but not maximally.**  
All five ablations achieve Water–Land centroid distances of 3.3–4.1 std units, confirming that the environmental broadcast signal does push the locomotor circuit into meaningfully different regions of hidden-state space per condition. However, this separation is not growing monotonically from baseline to full NMAP.

**2. Connectome regularization (step 03) achieves the best attractor separation.**  
The topological L2 penalty (anatomical distance weighting) is the single strongest driver of inter-attractor spacing, reaching Sep(W–L) = 4.08 vs 3.31 for the unregularized baseline. The anatomical distance weights in `dist_*` directly sculpt the loss landscape so that pathway weights serving long-range connections are preferentially compressed — this creates a more "compact" per-environment motor program that is easier to separate from other environments.

**3. Full NMAP (step 05) converges to balanced, environment-agnostic tangling.**  
The defining signature of step 05 is that Q values become nearly identical across all three conditions (Q ≈ 53 for water, transition, and land). No single environment has a dramatically cleaner or noisier attractor. This reflects a well-adapted model that does not over-specialize: its oscillator and motor pathways generalize across the viscosity continuum rather than carving out a dedicated attractor for each medium.

**4. Land attractors are inherently harder to stabilize (higher self-Q) across all ablations.**  
Q land > Q water in every model except step 05, consistent with the greater mechanical complexity of terrestrial locomotion relative to swimming. The full NMAP equalizes this gap.

**5. The neuromodulatory broadcast is primarily temporal, not structural.**  
Water and land trajectories share the same underlying state-space geometry (same weights → same manifold topology) but are traversed at different speeds (2.5× faster in water). The neuromodulatory parameters (`water_frequency_scale`, `land_frequency_scale`) create velocity-level separation rather than the kind of structural attractor bifurcation seen in, e.g., multi-gait biological CPGs. True structural separation would require the two conditions to visit entirely non-overlapping regions — the moderate centroid distances (3–4 std units) and non-trivial cross-condition Q values indicate partial rather than complete separation.

### Generated figures

| File | Contents |
|------|----------|
| `01_pca_3d_state_space.png` | 3D PCA of hidden-state trajectories per ablation; 2×3 subplot grid |
| `02_pca_2d_attractor_ellipses.png` | PC1–PC2 trajectories with 1.5σ covariance ellipses and centroid markers |
| `03_attractor_centroid_distances.png` | Pairwise centroid-distance heatmaps per ablation |
| `04_trajectory_tangling_Q.png` | Self-Q timeseries (rows = ablations, cols = conditions) + cross-condition Q overlaid |
| `05_oscillator_limit_cycles.png` | Phase portraits of the relaxation oscillator (dorsal vs. ventral activity); gradient-colored by time |
| `06_bneuron_population_activity.png` | Joint-wise B-neuron population activity heatmaps (time × joint) for the best-separating ablation |
| `07_ablation_summary_dashboard.png` | 5-panel dashboard: self-Q bars (A), W–L separation (B), cross-Q boundaries (C), neuromod parameter traces (D), normalized multi-metric radar chart (E) |

---

## Running the Analysis

```bash
# Re-run tangling analysis from existing 1M-step checkpoints
python tangling_analysis.py
# Output → results/tangling_analysis/

# Re-run ablation training plots
python results/plot_ablation.py

# Lambda sweep plots
python plot_lambda_sweep.py
```

---

## Tests

```bash
# Connectome priors smoke tests (fast, no training)
cd NMAP && python -m pytest connectome_priors/tests/ -v

# Single test
python -m pytest connectome_priors/tests/test_priors_smoke.py::test_generate_ncap_segment_priors_sparse_scalars -v

# Curriculum rollout test
python tests/test_curriculum_setup.py
```

---

## Result Layout

```
results/
├── 01_ncap_baseline/
│   └── curriculum_training/
│       ├── checkpoints/enhanced_ncap/   — .pt snapshots every 50k and at 1M steps
│       ├── logs/                        — training CSVs
│       ├── models/enhanced_ncap/        — final_model.pt
│       └── videos/                      — rendered MP4s per curriculum phase
├── 02_ncap_sparse_init/    (same structure)
├── 03_ncap_reg_only/       (same structure)
├── 04_ncap_full_priors/    (same structure)
├── 05_ncap_full_nmap/      (same structure)
├── sweep_nmap_lambda_*/    — lambda sweep runs (many subdirectories)
├── analysis_plots/         — ablation learning-curve plots (auto-generated)
└── tangling_analysis/      — 7 trajectory tangling figures (tangling_analysis.py)
```

---

## References

- **NCAP**: Bhatt & Bhattacharyya (2022). *Neural circuit architectural priors for embodied control.* [arXiv:2201.05242](https://arxiv.org/abs/2201.05242)
- **Trajectory tangling**: Russo et al. (2018). *Motor cortex embeds muscle-like commands in an untangled population response.* *Nature Neuroscience*, 21(11), 1349–1356. [doi:10.1038/s41593-018-0207-6](https://doi.org/10.1038/s41593-018-0207-6)
- **Relaxation oscillator model**: Xu et al. (2021). *Phase response analyses support a relaxation oscillator model of locomotor rhythm generation in C. elegans.* *eLife*, 10, e69905. [doi:10.7554/eLife.69905](https://doi.org/10.7554/eLife.69905)
- **Cook et al. 2019**: Sub-millimeter resolution *C. elegans* connectome. Chemical synapse + gap-junction adjacency matrices.
- **OpenWorm c302**: 3D neuron geometry — `c302_C2_FW.net.nml`
- **Tonic RL**: [github.com/fabiopardo/tonic](https://github.com/fabiopardo/tonic)
- **DeepMind Control Suite**: [github.com/deepmind/dm_control](https://github.com/deepmind/dm_control)

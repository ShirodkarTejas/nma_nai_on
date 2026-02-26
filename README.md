# NCAP Swimmer: Biologically-Inspired Neural Control for Adaptive Locomotion

## 🚀 **Current Status - Enhanced Training & Reward System Analysis** 
*Last Updated: 24-07-2025*

### **⚠️ Critical Reward System Discovery - Training Stagnation Resolved** 
Recent curriculum training revealed fundamental issues with the incentive structure that were preventing effective learning:

#### **🔍 Root Cause Analysis**
- **Training Stagnation**: Only 1-5 episodes per phase completion despite 1M target episodes
- **Reward Exploitation**: Original timeout system (40s) allowed agents to avoid targets and collect timeout rewards
- **Target Avoidance**: 1.5m target radius + generous timeouts encouraged "reward hacking" over actual navigation
- **Distance Performance**: Extremely low distances (0.4-0.5m) indicating no real locomotion learning

#### **🛠️ Major Fixes Implemented** [[memory:4186768]]
1. **Eliminated Timeout Exploitation**: 
   - Removed automatic target switching after timeouts
   - Reduced timeout rewards from 0.5 → 0.05 (nearly eliminated)
   - Agents must actually reach targets to advance
   
2. **Stricter Target Requirements**:
   - Target radius reduced from 1.5m → 0.8m (stricter completion)
   - Timeout duration reduced from 40s → 20s (urgency pressure)
   - Enhanced target type bonuses: Land targets worth 1.5-2.0x more reward
   
3. **Anti-Avoidance Mechanisms**:
   - Removed action clamping that limited speed to 30% (prevented fast swimming)
   - Fixed starting orientation to face +X direction (toward targets)
   - Land zone penalties eliminated - equal treatment for water/land locomotion
   
4. **Biological Constraint Relaxation**:
   - Oscillator sensitivity: 1.2 → 0.8 (more adaptive frequency changes)
   - Coupling strength: 0.8 → 0.5 (flexible joint coordination)  
   - Muscle parameters: 0.8 → 0.5 (stronger actuation capability)
   - Constraint application frequency: every 5k → 25k steps (less restrictive)

5. **Forward Momentum Reward Shaping** *(feats/connectomic_priors)*:
   - Added an explicit forward-momentum bonus: velocity projected onto the agent's heading vector is rewarded at every step, directly penalising lateral drift and backward motion.
   - Oscillation variance penalty (`--force_oscillation`) works in concert with the momentum reward to prevent the "curl-and-shiver" local optimum in viscous environments.
   - Land-zone curriculum radius extended to 0.2 m (`shrink_steps=[0,10,20,40]`, `radii=[1.0,0.6,0.4,0.2]`) so agents must navigate tight targets rather than drifting into large reward zones.

   

### **✅ All Systems Ready + Enhanced Incentive Design**
- **🎯 Target Cycling Fixed**: All navigation targets now cycle properly (not stuck on target 1)
- **🔄 Resume Training**: Seamless continuation from checkpoints (100k → 200k → 1M)
- **📊 Evaluation-Only Mode**: Test visualization changes without retraining
- **🏊 Enhanced Visibility**: Improved swimmer tracking and dynamic training status
- **💰 Reward System Overhaul**: Navigation-focused rewards with land target bonuses
- **Performance Target**: 5-15m distance (2-5 body lengths) for expert swim+crawl
- **Training Command**: `python main.py --mode train_curriculum --training_steps 1000000 --model_type enhanced_ncap`
- **Resume Command**: `python main.py --mode train_curriculum --training_steps 200000 --resume_checkpoint outputs/enhanced_ncap/checkpoints/enhanced_ncap_*_checkpoint_step_100000.pt --model_type enhanced_ncap`
- **Expected Duration**: 3-4 hours on RTX 3090 with checkpoints every 50k steps (biological model is 3x faster!)

### **⚡ Performance Benchmarks (RTX 3090)**
| **Training Steps** | **Duration** | **Speed** | **Use Case** |
|-------------------|-------------|-----------|--------------|
| 20k | 4 minutes | ~105 steps/s | Quick validation |
| 100k | 20 minutes | ~83 steps/s | Feature testing |
| 1M | 3.5 hours | ~77 steps/s | **Full training** |

*Biological NCAP is 3x faster than LSTM versions due to 99.4% fewer parameters (9 vs 1,418)*

### **🔬 Key Breakthroughs Achieved**
- **Environment Issues Solved**: Physics bottleneck identified and fixed with progressive curriculum
- **Body Scale Analysis**: Realistic targets set (5-15m vs previous 0.3m performance)  
- **Biological Validation**: NCAP architecture confirmed with traveling waves and proper oscillation
- **Comprehensive Testing**: All components verified working together
- **LSTM Removal**: Biological mechanisms outperform artificial memory (+59% adaptation, -99.4% parameters)

### **📊 Training Progression Plan**
| Phase | Steps | Environment | Target | Learning Focus |
|-------|-------|-------------|--------|----------------|
| 1 | 0-300k | Pure Swimming | 2-5m | Locomotion mastery |
| 2 | 300k-600k | Single Land Zone | 1-3m | Environmental adaptation |
| 3 | 600k-800k | Two Land Zones | 2-4m | Complex navigation |
| 4 | 800k-1M | Full Complexity | **5-15m** | **Expert swim+crawl** |

---

## 🎯 Project Overview

This project implements and extends the **Neural Central Pattern Generator (NCAP)** model for adaptive locomotion in mixed environments. Inspired by the C. elegans motor circuit, we aim to build biologically plausible neural models that can generalize well and adapt to different environmental conditions through **curriculum learning**.

### Key Goals
- **Build biologically plausible models** based on neural circuit architecture
- **Achieve effective swimming and crawling** through curriculum learning (target: 5-15m distance)
- **Enable adaptive locomotion** across different environments (water/land transitions)
- **Study progressive learning** from simple to complex environmental conditions

## 🧠 Biological Inspiration & Plausibility

The project is based on the **NCAP (Neural Central Pattern Generator)** architecture from the paper:
> ["Neural circuit architectural priors for embodied control"](https://arxiv.org/abs/2201.05242)

This biologically-inspired approach leverages:
- **Modular neural circuits** derived from C. elegans motor control
- **Oscillatory patterns** for rhythmic locomotion (60-step period)
- **Traveling wave coordination** with consistent phase delays between joints
- **Biological constraints** preserving muscle antagonism and coupling strength

### 🔬 **Biological Plausibility Analysis**

Our NCAP model achieves **high biological authenticity** by directly implementing real neural circuit principles:

#### **✅ Highly Biologically Plausible Components:**

**1. Central Pattern Generator (CPG)**
- **Real biology**: C. elegans has CPG neurons that generate rhythmic motor patterns
- **Our model**: Head oscillators create alternating dorsal/ventral activation
- **Plausibility**: ⭐⭐⭐⭐⭐ **Excellent** - directly mimics real neural circuits

**2. Proprioceptive Feedback**
- **Real biology**: Mechanoreceptors in body segments provide position feedback
- **Our model**: B-neurons receive input from previous joint positions  
- **Plausibility**: ⭐⭐⭐⭐⭐ **Excellent** - matches real sensory-motor integration

**3. Antagonistic Muscle Pairs**
- **Real biology**: Dorsal and ventral muscles work against each other
- **Our model**: `muscle_d - muscle_v` creates antagonistic torque output
- **Plausibility**: ⭐⭐⭐⭐⭐ **Excellent** - fundamental biological principle

**4. Graded Neural Activation** 
- **Real biology**: Neurons have saturation limits, can't fire infinitely
- **Our model**: `graded()` function clamps activation to [0,1]
- **Plausibility**: ⭐⭐⭐⭐⭐ **Excellent** - prevents unrealistic neural activity

**5. Excitatory/Inhibitory Constraints**
- **Real biology**: Synapses are either excitatory (≥0) or inhibitory (≤0)
- **Our model**: Parameter constraints enforce biological sign restrictions
- **Plausibility**: ⭐⭐⭐⭐⭐ **Excellent** - fundamental neuroscience principle

**6. Connectomic Initialization & Regularization** *(feats/connectomic_priors)*
We integrated the **Cook 2019 sub-millimeter *C. elegans* connectome** directly into the policy:
1. **Synaptic Weight Initialization:** Extracted precise chemical synapse counts for B-neurons to muscles, B-neurons to D-neurons, D-neurons to contralateral muscles, and DVA to B-neurons across both ventral and dorsal pathways to biologically seed the network.
2. **Spatial Regularization:** Extracted the 3D geometric distances between the aforementioned neuron somas, utilizing this spatial data as a topological L2 regularization hyperparameter (`--sparse_reg_lambda`).
3. **D-Neuron Series Conductance Abstraction:** For crossed inhibitory pathways, we calculate effective throughput using `N_eff = (N1 * N2) / (N1 + N2)` for DB→DD→ventral-muscle and VB→VD→dorsal-muscle chains.
4. **Exact Gap-Junction Extraction:** Sourced from Cook 2019 gap matrices for adjacent B-neuron electrical coupling (`DB_i↔DB_{i+1}`, `VB_i↔VB_{i+1}`).

#### **📊 Comparison to Real C. elegans:**

| **Aspect** | **Real C. elegans** | **Our NCAP Model** | **Match** |
|------------|-------------------|-------------------|-----------|
| **CPG oscillation** | AVB/AVA command neurons | Head oscillators | ⭐⭐⭐⭐⭐ |
| **Proprioception** | PLM/ALM mechanoreceptors | Joint position feedback | ⭐⭐⭐⭐⭐ |
| **Muscle antagonism** | Dorsal/ventral muscle classes | muscle_d vs muscle_v | ⭐⭐⭐⭐⭐ |
| **Segmentation** | Repeated VNC motifs | Per-joint B-neurons | ⭐⭐⭐⭐ |
| **Neural dynamics** | Continuous membrane potential | Discrete graded activation | ⭐⭐⭐ |
| **Adaptation** | Neuromodulation (dopamine, etc.) | LSTM + environment layers | ⭐⭐ |

#### **🏆 Overall Biological Plausibility Rating: ⭐⭐⭐⭐ (Very Good)**

**Biological Authenticity Strengths:**
- Core architecture follows established neuroscience principles
- Weight constraints prevent biologically impossible connections  
- Segmental organization matches real nervous system structure
- Sensory-motor integration replicates proprioceptive feedback loops
- Oscillator-driven locomotion mimics real CPG function

**Summary**: Our NCAP model is **highly biologically plausible** in its core architecture and constraints, with the main departures being computational conveniences. The fundamental swimming circuit closely matches real C. elegans neural organization!

### 🧠 **Model Evolution & Learnings**

Through empirical testing, we discovered that **biological mechanisms significantly outperform artificial ones**:

- **Biological NCAP**: 9 parameters, 59% stronger adaptation, ⭐⭐⭐⭐⭐ biological plausibility
- **LSTM NCAP**: 1,418 parameters, weaker adaptation, ⭐⭐ biological plausibility  
- **Key insight**: Neuromodulation-like adaptation > artificial memory for environment switching

**See `MODEL_LEARNINGS.md` for comprehensive analysis, empirical results, and biological insights.**

## 🏗️ Architecture

### Core Components
- **Biological NCAP Model**: Ultra-lightweight neural circuit with 9 biologically-constrained parameters
- **Progressive Mixed Environment**: Curriculum from pure swimming to complex swim+crawl
- **Curriculum Training Framework**: Phase-based learning with automatic progression
- **Comprehensive Evaluation**: Performance tracking across all training phases

### 🏆 **Model Evolution & Performance**

**Latest Discovery**: Biological adaptation mechanisms significantly outperform artificial memory systems!

| **Model Version** | **Parameters** | **Adaptation Strength** | **Biological Plausibility** | **Status** |
|------------------|----------------|-------------------------|----------------------------|-----------|
| Simple NCAP | 4 | None | ⭐⭐⭐⭐ | Legacy |
| Complex NCAP (LSTM) | 1,418 | 0.0473 | ⭐⭐ | **DEPRECATED** |
| **Biological NCAP** | **9** | **0.0753 (+59%)** | **⭐⭐⭐⭐⭐** | **ACTIVE** |

**Key Finding**: The biological model achieves **59% stronger environment adaptation** with **99.4% fewer parameters** than LSTM-based approaches. See `MODEL_LEARNINGS.md` for detailed analysis.

### Project Structure
```
nma_neuroai/
├── swimmer/
│   ├── models/
│   │   ├── biological_ncap.py       # ✅ ACTIVE: Biological NCAP (9 parameters, no LSTM)
│   │   ├── simple_ncap.py           # 📚 Legacy: Simple NCAP (4 parameters)
│   │   ├── ncap_swimmer.py          # ❌ DEPRECATED: Complex NCAP with LSTM (1,418 parameters)
│   │   └── tonic_ncap.py            # 📚 Tonic wrapper (legacy)
│   ├── training/
│   │   ├── curriculum_trainer.py    # ✅ Progressive curriculum training
│   │   ├── simple_biological_trainer.py # ✅ Biological constraint preservation
│   │   └── improved_ncap_trainer.py # 📚 Legacy improved trainer
│   ├── environments/
│   │   ├── progressive_mixed_env.py # ✅ Curriculum environment (pure swimming → mixed)
│   │   ├── simple_swimmer.py        # ✅ Optimized simple swimming environment
│   │   ├── mixed_environment.py     # 📚 Complex mixed environment (baseline comparison)
│   │   ├── physics_fix.py           # ✅ Gear ratio fixes for effective movement
│   │   └── tonic_wrapper.py         # ✅ Tonic compatibility layer
│   └── utils/
│       ├── visualization.py         # ✅ Evaluation and plotting utilities
│       ├── training_logger.py       # ✅ Comprehensive training logging
│       └── helpers.py               # ✅ Utility functions
├── connectome_priors/               # ✅ [feats/connectomic_priors] Cook 2019 connectome integration
│   ├── swimmer_priors.py            #    Cook 2019 → sparse pathway priors
│   ├── c_elegans_connectome.py      #    Connectome edge loading
│   └── c_elegans_geometry.py        #    c302 NML geometry parsing
├── tests/                           # ✅ All testing components
├── MODEL_LEARNINGS.md               # 🧠 Comprehensive model analysis & empirical findings
├── run_experiments.py               # ✅ [feats/connectomic_priors] 4-step ablation runner
├── outputs/
│   ├── curriculum_training/         # ✅ Complete curriculum training outputs
│   │   ├── checkpoints/             # ✅ Training checkpoints (resume from here)
│   │   ├── logs/                    # ✅ Detailed training metrics and logs
│   │   ├── videos/                  # ✅ Training and evaluation videos
│   │   ├── plots/                   # ✅ Performance plots and trajectory analysis
│   │   ├── summaries/               # ✅ Training summaries and reports
│   │   └── models/                  # ✅ Final trained models
│   ├── improved_mixed_env/          # ✅ Legacy training results
│   └── training_logs/               # ✅ Legacy logs (other training modes)
└── main.py                          # ✅ Multi-mode execution with curriculum option
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (highly recommended for 1M episode training)
- 16+ GB RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd nma_neuroai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv nma_neuroai_env
   source nma_neuroai_env/bin/activate  # On Windows: nma_neuroai_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tonic RL library**
   ```bash
   git clone https://github.com/fabiopardo/tonic.git
   cd tonic
   pip install -e .
   cd ..
   ```

### GPU Setup (Highly Recommended)
For 1M episode training, GPU acceleration is essential:
- CUDA toolkit installed
- Compatible NVIDIA drivers  
- PyTorch with CUDA support

The environment will automatically detect and use GPU if available.

## 🎮 Usage

### 🎓 Curriculum Training (Recommended for 1M Episodes)

#### **🚀 Enhanced NCAP (Default - Recommended)**
```bash
# Full curriculum training with Enhanced NCAP (relaxation oscillator + goal-directed navigation)
python main.py --mode train_curriculum --training_steps 1000000 --model_type enhanced_ncap --save_steps 50000

# Shorter validation runs
python main.py --mode train_curriculum --training_steps 100000 --model_type enhanced_ncap --n_links 5
```

#### **🧬 Biological NCAP (Baseline)**
```bash
# Training with original Biological NCAP (for comparison)
python main.py --mode train_curriculum --training_steps 1000000 --model_type biological_ncap --save_steps 50000

# Quick biological baseline validation
python main.py --mode train_curriculum --training_steps 100000 --model_type biological_ncap --n_links 5
```

### 🔄 Resume Training from Checkpoint

#### **🚀 Enhanced NCAP Checkpoints**
```bash
# Continue Enhanced NCAP training from checkpoint
python main.py --mode train_curriculum --training_steps 200000 --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_100000.pt

# Scale up Enhanced NCAP after validation
python main.py --mode train_curriculum --training_steps 1000000 --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_200000.pt
```

#### **🧬 Biological NCAP Checkpoints**
```bash
# Continue Biological NCAP training from checkpoint
python main.py --mode train_curriculum --training_steps 200000 --model_type biological_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/biological_ncap/biological_ncap_ppo_5links_checkpoint_step_100000.pt

# Scale up Biological NCAP after validation
python main.py --mode train_curriculum --training_steps 1000000 --model_type biological_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/biological_ncap/biological_ncap_ppo_5links_checkpoint_step_200000.pt
```

### 📊 Evaluation-Only Mode (No Training)

#### **🚀 Enhanced NCAP Evaluation**
```bash
# Comprehensive Enhanced NCAP evaluation from checkpoint
python main.py --mode evaluate_curriculum --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_100000.pt

# Custom Enhanced NCAP evaluation parameters
python main.py --mode evaluate_curriculum --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_100000.pt --eval_episodes 30 --eval_video_steps 600
```

#### **🧬 Biological NCAP Evaluation**
```bash
# Comprehensive Biological NCAP evaluation from checkpoint
python main.py --mode evaluate_curriculum --model_type biological_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/biological_ncap/biological_ncap_ppo_5links_checkpoint_step_100000.pt

# Custom Biological NCAP evaluation parameters  
python main.py --mode evaluate_curriculum --model_type biological_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/biological_ncap/biological_ncap_ppo_5links_checkpoint_step_100000.pt --eval_episodes 30 --eval_video_steps 600
```

**What this does:**
- **Phase 1 (0-30%)**: Pure swimming mastery with optimized physics
- **Phase 2 (30-60%)**: Introduction of single land zone for adaptation
- **Phase 3 (60-80%)**: Two land zones for complex navigation
- **Phase 4 (80-100%)**: Full mixed environment complexity
- **Automatic phase transitions** with performance monitoring
- **Regular checkpointing** every 50k steps with comprehensive visualizations
- **Enhanced trajectory analysis** similar to research publications
- **Expected duration**: 3-4 hours on RTX 3090 (biological model is 3x faster!)

### 🧠 Model Types Explained

#### **🚀 Enhanced NCAP (Default)**
- **Features**: Relaxation oscillator + goal-directed navigation + anti-tail-chasing
- **Based on**: [C. elegans research](https://elifesciences.org/articles/69905) (eLife, 2021)
- **Advantages**: Target-seeking behavior, dramatic frequency adaptation (3-5x)
- **Best for**: Navigation tasks, goal-directed locomotion

#### **🧬 Biological NCAP (Baseline)**  
- **Features**: Pure biological adaptation without artificial memory (LSTM)
- **Advantages**: 99.4% fewer parameters, 59% stronger environment adaptation
- **Best for**: Baseline comparison, minimal complexity requirements

### 🗂️ Organized Output Structure
Artifacts are automatically organized by model type to prevent overwriting:
```
outputs/
├── enhanced_ncap/              # Enhanced NCAP artifacts
│   ├── checkpoints/
│   ├── models/
│   ├── videos/
│   └── plots/
├── biological_ncap/            # Biological NCAP artifacts  
│   ├── checkpoints/
│   ├── models/
│   ├── videos/
│   └── plots/
└── comparisons/                # Cross-model comparisons
```

**Generated outputs for each model:**
- 📊 **Training plots**: Progress charts and performance analysis
- 🎬 **Videos with zone indicators**: Visual training progress with overlaid environment zones
- 📈 **Trajectory analysis**: Detailed swimmer path analysis with environment transitions (similar to research publications)
- 📄 **Comprehensive summaries**: Performance metrics and training statistics
- 🔄 **Model-specific checkpoints**: Organized by model type for easy comparison

**Enhanced visualization features:**
- 🎯 **Zone overlays**: Semi-transparent land zones with labels in videos
- 🗺️ **Interactive minimap**: Top-right inset showing environment zones, swimmer position, and movement trail
- 🔍 **Debug information**: Progress percentage and zone count displayed
- 🏊 **Enhanced swimmer visibility**: Bright cyan circle, yellow arrow, and "SWIMMER" label for clear position tracking
- 🎯 **Fixed target cycling**: All navigation targets now cycle properly (not stuck on target 1)
- 🕒 **Auto-advance targets**: Targets advance automatically after 300 steps if not reached (helps untrained models)
- 🏷️ **Dynamic training status**: Label changes from "UNTRAINED MODEL" → "TRAINING IN PROGRESS" → "TRAINED MODEL"
- 🎮 **Reduced motion artifacts**: Action clamping for smoother untrained model visualization
- 📈 **Research-quality plots**: Trajectory analysis similar to scientific publications with zone circles

**UI Layout Structure:**
```
Top-right corner:
┌─────────────────┐
│ Environment Map │ ← Minimap title
├─────────────────┤
│ 🗺️ Minimap      │ ← 120×120 minimap  
│   🟤🔴🟢        │ ← Zones, position, trail
└─────────────────┘
    Step: 1234      ← Step counter (below minimap)
```

### 🧬 Biological Training (Constraint Preservation)
```bash
# Train with strict biological constraints
python main.py --mode train_biological --training_steps 100000 --save_steps 20000 --log_episodes 10
```

### 🏊 Legacy Improved Training
```bash
# Original improved training approach (for comparison)
python main.py --mode train_improved --training_steps 30000 --save_steps 10000 --log_episodes 5
```

### 📊 Model Evaluation
```bash
# Evaluate trained curriculum model
python main.py --mode evaluate --load_model outputs/curriculum_final_model_5links.pt
```

---

## 🔬 Connectomic Priors & Ablation Study *(feats/connectomic_priors)*

This branch integrates Cook 2019 connectome data directly into the NCAP policy initialization and regularization, then isolates each contribution through a structured ablation study.

### 🧬 Connectome-Informed Policy

The `connectome_priors/` module (see `swimmer_priors.py`, `c_elegans_connectome.py`, `c_elegans_geometry.py`) extracts three types of biological signal from the Cook 2019 dataset and injects them into the optimizer loop:

| Signal | Source | Effect |
|--------|--------|--------|
| Synaptic weight initialization | Chemical synapse counts (B-neurons, D-neurons, DVA) | Biologically seeds excitatory/inhibitory weights at network init |
| Topological L2 regularization | 3D soma distances from c302 NML geometry | Penalizes long-range connections proportional to anatomical distance |
| Gap-junction coupling | Cook 2019 gap matrices | Initializes adjacent B-neuron electrical coupling (`DB_i↔DB_{i+1}`, `VB_i↔VB_{i+1}`) |

Connectome artifact paths are now fully configurable via CLI (no code edits required):

```bash
--xlsx_path PATH        # Path to Cook 2019 .xlsx connectome file
--chem_sheet NAME       # Sheet name for chemical synapses (default: "hermaphrodite chemical")
--gap_sheet NAME        # Sheet name for gap junctions (default: "hermaphrodite gap jn")
--nml_path PATH         # Path to c302_C2_FW.net.nml geometry file
```

### 🚀 Running the Ablation Study

The full 4-step ablation is launched from the repository root:

```bash
python run_experiments.py
```

This sequentially runs and archives four experiments to `results/`, each for 500,000 training steps:

| Step | Name | Flags |
|------|------|-------|
| 01 | Baseline (canonical NCAP) | — |
| 02 | Oscillation preservation | `--force_oscillation` |
| 03 | Sparse initialization | `--force_oscillation --sparse_init` |
| 04 | Full connectomic priors | `--force_oscillation --sparse_init --sparse_reg_lambda 0.05` |

Results are archived to `results/<name>/` on completion.

#### Manual invocation

```bash
python main.py --mode train --training_steps 500000
python main.py --mode train --training_steps 500000 --force_oscillation
python main.py --mode train --training_steps 500000 --force_oscillation --sparse_init
python main.py --mode train --training_steps 500000 --force_oscillation --sparse_init --sparse_reg_lambda 0.05
```

### ⚙️ Ablation Flags

| Flag | Effect |
|------|--------|
| `--sparse_init` | Connectome-informed weight initialization (excitatory ipsi, inhibitory contra, gap-junction next-neighbor) |
| `--sparse_reg_lambda <λ>` | Topological L2 regularization weighted by normalized Cook-distance per pathway |
| `--force_oscillation` | Variance penalty on actor output to prevent policy collapse to near-zero actions ("curl-and-shiver" fix) |
| `--xlsx_path PATH` | Override default Cook 2019 XLSX path |
| `--chem_sheet NAME` | Override chemical synapse sheet name |
| `--gap_sheet NAME` | Override gap-junction sheet name |
| `--nml_path PATH` | Override c302 NML geometry path |

All flags are fully orthogonal and can be combined independently to isolate each contribution.

---

## 🆕 Recent Improvements & Features

### 🎯 **Target Cycling Fixes**
- **Problem Solved**: Previously only the first target would flash/be active
- **Root Cause**: Untrained swimmers couldn't reach targets within 1.0m radius
- **Solution**: Larger target radius (2.0m) + auto-advance timer (300 steps)
- **Result**: All targets now cycle properly in all training phases

### 🏊 **Enhanced Swimmer Visibility**
- **Larger cyan circle** (25px radius) around swimmer position
- **Yellow arrow** pointing directly to swimmer for easy tracking
- **"SWIMMER" label** with enhanced visibility
- **Black borders** for better contrast against backgrounds
- **Error reporting** for debugging position issues

### 🔄 **Checkpoint & Resume System**
- **Auto-saves** every 50k steps to `outputs/curriculum_training/checkpoints/`
- **Seamless resuming** from any checkpoint without data loss
- **Comprehensive state** preservation (model, phase rewards, distances, training progress)
- **Flexible continuation** (100k → 200k → 1M as needed)

### 📊 **Evaluation-Only Mode**
- **No training required** - just analysis and visualization
- **Perfect for testing** visualization changes without retraining
- **Comprehensive outputs**: videos, plots, trajectory analysis, summaries
- **Customizable parameters**: episode count, video length
- **Multiple video formats**: phase comparison + individual phase videos

### 🏷️ **Dynamic Training Status**
- **Color-coded status indicator** in videos:
  - 🔴 **"UNTRAINED MODEL"** (0-10% progress)
  - 🟡 **"TRAINING IN PROGRESS"** (10-50% progress)  
  - 🟢 **"TRAINED MODEL"** (50%+ progress)
- **Real-time updates** based on actual training progress

### 🎬 **Video Library Outputs**
**Training Mode:**
- `curriculum_video_step_X.mp4` - Intermediate progress videos
- `curriculum_final_video.mp4` - Complete phase comparison

**Evaluation Mode:**
- `evaluation_phase_comparison.mp4` - All phases in one video
- `evaluation_phase_0_pure_swimming.mp4` - Phase 1 detailed analysis
- `evaluation_phase_1_single_land_zone.mp4` - Phase 2 detailed analysis
- `evaluation_phase_2_two_land_zones.mp4` - Phase 3 detailed analysis  
- `evaluation_phase_3_full_complexity.mp4` - Phase 4 detailed analysis

### ⚙️ **Command Line Options**
```bash
# Model selection
--model_type enhanced_ncap     # Enhanced NCAP with relaxation oscillator (default)
--model_type biological_ncap   # Original biological NCAP (baseline)

# Basic training options
--mode train_curriculum        # Curriculum training mode
--mode evaluate_curriculum     # Evaluation-only mode  
--training_steps N            # Total training steps target
--n_links N                   # Number of swimmer links (default: 5)
--algorithm ppo               # RL algorithm (default: ppo)

# Resume training options
--resume_checkpoint PATH      # Path to model-specific checkpoint file
--save_steps N               # Steps between saves (default: 50000)
--log_episodes N             # Episodes between logs (default: 50)

# Evaluation options  
--eval_episodes N            # Episodes per phase (default: 20)
--eval_video_steps N         # Video length in steps (default: 400)

# Connectomic priors options (feats/connectomic_priors)
--sparse_init                # Connectome-informed weight initialization
--sparse_reg_lambda <λ>      # Topological L2 regularization (e.g. 0.05)
--force_oscillation          # Action variance penalty to prevent policy collapse
--xlsx_path PATH             # Cook 2019 connectome XLSX path
--chem_sheet NAME            # Chemical synapse sheet name
--gap_sheet NAME             # Gap-junction sheet name
--nml_path PATH              # c302 NML geometry path
```

### 🔧 **Phase Duration Configuration**

Phase durations are now **data-driven and easily configurable**! Edit the `PHASE_DURATION_CONFIG` in `CurriculumNCAPTrainer`:

```python
PHASE_DURATION_CONFIG = {
    'evaluation_steps': [200, 200, 200, 400],     # Steps per episode for each phase
    'video_steps': [500, 500, 500, 1000],         # Steps per video for each phase  
    'trajectory_multiplier': [1.0, 1.0, 1.0, 2.0] # Multiplier for trajectory analysis
}
```

**Current Configuration (Phase 0-3):**
- **Evaluation**: 200, 200, 200, **400** steps per episode
- **Videos**: 500, 500, 500, **1000** steps per phase
- **Trajectory**: 1x, 1x, 1x, **2x** multiplier

**Why Full Complexity Gets More Time:**
- Complex environment needs time for **zone transitions**
- Swimmer must demonstrate **swimming → crawling → swimming**
- **2x duration** allows complete exploration of all zones

## 🎯 **Example Workflow**

### **Typical Training & Analysis Pipeline (Enhanced NCAP)**
```bash
# 1. Start with Enhanced NCAP validation training
python main.py --mode train_curriculum --training_steps 100000 --model_type enhanced_ncap --n_links 5

# 2. Evaluate the 100k Enhanced NCAP results  
python main.py --mode evaluate_curriculum --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_100000.pt

# 3. If satisfied, continue Enhanced NCAP to 200k
python main.py --mode train_curriculum --training_steps 200000 --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_100000.pt

# 4. Scale up Enhanced NCAP to full 1M training
python main.py --mode train_curriculum --training_steps 1000000 --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_200000.pt

# 5. Final comprehensive Enhanced NCAP analysis
python main.py --mode evaluate_curriculum --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_1000000.pt --eval_episodes 50 --eval_video_steps 800
```

### **Running the Connectomic Priors Ablation** *(feats/connectomic_priors)*
```bash
# Full automated ablation (4 experiments × 500k steps → results/)
python run_experiments.py

# Or run individual ablation steps manually
python main.py --mode train --training_steps 500000                                                              # Step 01: Baseline
python main.py --mode train --training_steps 500000 --force_oscillation                                          # Step 02: + oscillation
python main.py --mode train --training_steps 500000 --force_oscillation --sparse_init                            # Step 03: + sparse init
python main.py --mode train --training_steps 500000 --force_oscillation --sparse_init --sparse_reg_lambda 0.05   # Step 04: Full NMAP

# Use custom connectome artifact paths if needed
python main.py --mode train --training_steps 500000 --force_oscillation --sparse_init --sparse_reg_lambda 0.05 \
  --xlsx_path /path/to/Cook2019.xlsx \
  --nml_path /path/to/c302_C2_FW.net.nml
```

### **Testing Visualization Changes**
```bash
# Make code changes to visualization...

# Test immediately without retraining (Enhanced NCAP)
python main.py --mode evaluate_curriculum --model_type enhanced_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_100000.pt --eval_episodes 10 --eval_video_steps 300

# Test with Biological NCAP for comparison
python main.py --mode evaluate_curriculum --model_type biological_ncap --resume_checkpoint outputs/curriculum_training/checkpoints/biological_ncap/biological_ncap_ppo_5links_checkpoint_step_100000.pt --eval_episodes 10 --eval_video_steps 300

# Quick turnaround for iterative improvements
```

### **Organized Output Locations**

#### **🚀 Enhanced NCAP Artifacts**
- **Checkpoints**: `outputs/curriculum_training/checkpoints/enhanced_ncap/enhanced_ncap_ppo_5links_checkpoint_step_X.pt`
- **Training Logs**: `outputs/curriculum_training/logs/enhanced_ncap/`
- **Videos**: `outputs/curriculum_training/videos/enhanced_ncap/`
- **Plots**: `outputs/curriculum_training/plots/enhanced_ncap/`  
- **Summaries**: `outputs/curriculum_training/summaries/enhanced_ncap/`
- **Models**: `outputs/curriculum_training/models/enhanced_ncap/`

#### **🧬 Biological NCAP Artifacts**
- **Checkpoints**: `outputs/curriculum_training/checkpoints/biological_ncap/biological_ncap_ppo_5links_checkpoint_step_X.pt`
- **Training Logs**: `outputs/curriculum_training/logs/biological_ncap/`
- **Videos**: `outputs/curriculum_training/videos/biological_ncap/`
- **Plots**: `outputs/curriculum_training/plots/biological_ncap/`  
- **Summaries**: `outputs/curriculum_training/summaries/biological_ncap/`
- **Models**: `outputs/curriculum_training/models/biological_ncap/`

#### **🔬 Ablation Study Artifacts** *(feats/connectomic_priors)*
- **Results**: `results/01_baseline/`, `results/02_force_oscillation/`, `results/03_sparse_init/`, `results/04_full_nmap/`

## 🔬 Key Research Discoveries

### **🚨 Reward System Architecture - Critical Learning Challenge**
Our latest training revealed that **incentive structure, not just environment complexity, was the primary learning bottleneck**:

#### **The Reward Hacking Problem**
- **Timeout Exploitation**: Agents learned to avoid targets and collect guaranteed timeout rewards instead of navigating
- **Distance Stagnation**: 0.4-0.5m performance indicated complete avoidance of locomotion training
- **Target Radius Exploitation**: 1.5m radius + 40s timeouts made "doing nothing" more rewarding than learning
- **Action Limiting**: 30% action clamping prevented fast swimming, encouraging passive behavior

#### **The Navigation vs Survival Dilemma**
The original reward system inadvertently taught agents that **survival (avoiding failure) was more profitable than achievement (reaching targets)**:

```
Original System:
✗ Timeout after 40s → Small positive reward (0.5)
✗ Large target radius (1.5m) → Easy to accidentally complete
✗ Movement penalties → Encouraged minimal motion
✗ Equal treatment → No incentive for challenging land targets

Fixed System:
✓ No timeout rewards → Must reach targets for any navigation reward
✓ Strict target radius (0.8m) → Requires precise navigation  
✓ Land target bonuses (1.5-2.0x) → Encourages environment diversity
✓ Movement rewards → Active locomotion encouraged
✓ Forward momentum bonus → Penalises lateral drift and backward motion
```

#### **Biological Constraint vs Learning Trade-off**
We discovered a fundamental tension between biological authenticity and learning capability:
- **Too Strict**: Biological constraints prevented effective actuation and adaptation
- **Too Loose**: Lost biological plausibility and circuit authenticity
- **Sweet Spot**: 50% relaxation of constraints every 25k steps (vs every 5k steps) maintains biology while enabling learning

### **🚨 Environment Physics Issues Solved**
Our comprehensive analysis revealed that **environment complexity, not model architecture, was the primary bottleneck**:

- **Simple Environment Performance**: 0.30m (excellent zero-shot for NCAP)
- **Complex Environment Performance**: 0.06m (5x worse due to physics issues)
- **Solution**: Progressive curriculum starting simple, adding complexity gradually

### **📏 Body Scale Analysis**
Swimmer analysis revealed performance must be measured relative to body size:
- **Swimmer body length**: ~3.0m (6 links × 0.5m each)
- **Previous "good" 0.3m**: Only **0.1 body lengths** (poor performance)
- **Target performance**: **5-15m** (2-5 body lengths for meaningful swimming)

### **🧬 Biological Circuit Validation**  
NCAP architecture confirmed working perfectly:
- ✅ **Traveling wave patterns** with consistent 19-step phase delays
- ✅ **Oscillatory behavior** with 60-step periods (1 second at 60Hz)
- ✅ **Muscle antagonism** preserved with biological constraints
- ✅ **Zero-shot capability** demonstrates biological circuit effectiveness

## 🔧 Component Testing

Before running long training, verify setup:
```bash
# Test all curriculum components
python tests/test_curriculum_setup.py
```

This validates:
- Progressive environment phase transitions
- NCAP model creation and forward pass
- Agent-environment interaction  
- Biological constraint preservation
- Performance evaluation across phases

## 📚 Performance Benchmarks

### **Reward System Learning Journey**
- **Original Broken System**: 0.4-0.5m (reward hacking, timeout exploitation)
- **Fixed Incentive Structure**: Target 5-15m (proper navigation learning)
- **Key Insight**: **Incentive design matters more than model architecture** for learning success

### **Zero-Shot NCAP Performance**
- **Simple Environment**: 0.30m (good biological circuit function)
- **Complex Environment**: 0.06m (physics issues, now solved)
- **With Gear Fix**: 0.21m (physics optimization applied)

### **Training Expectations (Post-Reward Fix)**
- **Untrained/Random**: 0.0-0.5m (mostly random motion)
- **Learning Phase**: 0.5-2.0m (developing coordination)  
- **Good Performance**: 2.0-10.0m (effective swimming)
- **Excellent Performance**: 10.0-50.0m (optimized locomotion)
- **Expert Level**: 50.0+m (highly efficient, stretch goal)

### **Reward System Lessons Learned**
1. **Never reward inaction** - timeout rewards taught avoidance behavior
2. **Strict completion criteria** - loose target radius encouraged accidental success
3. **Environment neutrality** - don't penalize challenging environments
4. **Progress over proximity** - reward movement toward goals, not just closeness
5. **Balanced constraints** - biological authenticity vs learning capability trade-off

## 🎯 Success Criteria

### **Primary Goals (1M Episode Training)**
- [ ] **Swimming mastery**: >5m distance in pure water environment
- [ ] **Crawling capability**: >2m distance on land zones  
- [ ] **Adaptive behavior**: Successful environment transitions
- [ ] **Biological realism**: Preserved muscle antagonism and coupling

### **Technical Achievements**
- [x] **Environment issues diagnosed**: Complex physics problems identified
- [x] **Progressive curriculum**: Phase-based learning implemented
- [x] **Biological constraints**: Automatic parameter preservation
- [x] **Comprehensive testing**: All components verified working
- [x] **Ready for long training**: 1M episode setup tested and confirmed
- [x] **Connectomic priors integrated**: Cook 2019 sparse initialization and topological regularization *(feats/connectomic_priors)*
- [x] **Ablation study structure**: 4-step experiment runner with automated archiving *(feats/connectomic_priors)*
- [x] **Forward momentum reward**: Explicit heading-aligned velocity bonus added *(feats/connectomic_priors)*

## 📚 References

- **NCAP Paper**: [Neural circuit architectural priors for embodied control](https://arxiv.org/abs/2201.05242)
- **Original NCAP Implementation**: [ncap repository](https://github.com/nikhilxb/ncap)
- **Tonic RL Library**: [Tonic framework](https://github.com/fabiopardo/tonic)
- **DeepMind Control Suite**: [dm_control](https://github.com/deepmind/dm_control)
- **Cook et al. 2019**: *C. elegans* connectome adjacency matrices (chemical synapses and gap junctions)
- **eLife (2021)**: Phase-response evidence supporting relaxation-oscillator locomotor rhythm generation in *C. elegans*
- **OpenWorm c302**: `c302_C2_FW.net.nml` 3D neuron geometry

## 🤝 Contributing

This is a research project focused on biologically-inspired neural control through curriculum learning. Contributions welcome in:
- Training algorithm improvements
- NCAP architecture enhancements  
- Progressive environment design
- Performance optimization
- Documentation and analysis

## 📄 License

This project is for research purposes. Please respect the licenses of the underlying libraries (Tonic, dm_control, etc.).

## 🔗 Related Work

- **Neuromatch Academy**: [NeuroAI Course](https://github.com/neuromatch/NeuroAI_Course)
- **Hierarchical Motor Control**: [Nature paper](https://www.nature.com/articles/s41467-019-13239-6)
- **Curriculum Learning in RL**: [Various approaches to progressive difficulty](https://arxiv.org/abs/1707.02286)

---

*This project explores curriculum learning for biologically-inspired neural control, demonstrating how progressive complexity enables robust adaptive locomotion through neural central pattern generators.* 🧬🏊‍♂️🦎

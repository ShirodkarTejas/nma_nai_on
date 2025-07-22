# NCAP Swimmer: Biologically-Inspired Neural Control for Adaptive Locomotion

## 🎯 Project Overview

This project implements and extends the **Neural Central Pattern Generator (NCAP)** model for adaptive locomotion in mixed environments. Inspired by the C. elegans motor circuit, we aim to build biologically plausible neural models that can generalize well and adapt to different environmental conditions.

### Key Goals
- **Build biologically plausible models** based on neural circuit architecture
- **Improve NCAP model training** for better generalization and adaptation
- **Enable adaptive locomotion** across different environments (water/land transitions)
- **Study transfer learning** and zero-shot performance in novel environments

## 🧠 Biological Inspiration

The project is based on the **NCAP (Neural Central Pattern Generator)** architecture from the paper:
> ["Neural circuit architectural priors for embodied control"](https://arxiv.org/abs/2201.05242)

This biologically-inspired approach leverages:
- **Modular neural circuits** derived from C. elegans motor control
- **Oscillatory patterns** for rhythmic locomotion
- **Phase offsets** for coordinated movement
- **Weight constraints** that reflect biological connectivity patterns

## 🏗️ Architecture

### Core Components
- **NCAP Model**: Biologically-inspired neural circuit with oscillatory dynamics
- **Mixed Environment**: Swimmer that adapts between water (low viscosity) and land (high viscosity)
- **Training Framework**: Reinforcement learning with Tonic library
- **Evaluation System**: Comprehensive metrics and visualization

### Project Structure
```
nma_neuroai/
├── swimmer/
│   ├── models/
│   │   ├── ncap_swimmer.py          # NCAP neural circuit implementation
│   │   └── tonic_ncap.py            # Tonic-compatible wrapper
│   ├── training/
│   │   ├── swimmer_trainer.py       # Training orchestration
│   │   └── custom_tonic_agent.py    # Custom PPO agent
│   ├── environments/
│   │   ├── mixed_environment.py     # Mixed water/land environment
│   │   └── tonic_wrapper.py         # Tonic compatibility layer
│   └── utils/
│       └── visualization.py         # Evaluation and plotting
├── outputs/                         # Training results and models
├── test_logs/                       # Experiment logs
└── main.py                          # Main execution script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker (optional, for containerized setup)

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

### GPU Setup (Optional)
For GPU acceleration, ensure you have:
- CUDA toolkit installed
- Compatible NVIDIA drivers
- PyTorch with CUDA support

The environment will automatically detect and use GPU if available.

## 🎮 Usage

### Training (recommended pipeline)
```bash
# Train NCAP with all stability fixes and automatic evaluation
python main.py --mode train_improved --training_steps 30000 --save_steps 10000 --log_episodes 5
```

The script will:
1. Train for the specified steps using A2C + Tonic
2. Save checkpoints to `outputs/training_logs/.../checkpoints` every `--save_steps`
3. Evaluate the trained model in the mixed water→land environment, generate a video and plots in `outputs/improved_mixed_env/`

### Quick Evaluation of an Existing Model
```bash
# Run just the evaluation / video generation step
python main.py --mode evaluate --load_model outputs/training/improved_ncap_6links.pt  # 4× realtime evaluation
```

All other legacy modes (`train`, `train_simple`, `test_mixed`, etc.) are still available, but `train_improved` is the maintained entry-point.

## 🔬 Research Directions

### Roadmap to Improved Locomotion (July 2025)
1. **Curriculum**
   • Phase A – train 50 k steps in water‐only (no land islands) to master swimming.
   • Phase B – introduce two land islands at (±3 m, 0 m) with radius 1 m, shrink to 0.6 m over episodes.

2. **Reward Shaping**
   • Velocity term tripled; target swim speed lowered to 0.15 m s⁻¹; activity & torque penalties ×0.5.
   • Water viscosity set to 1 × 10⁻⁴ and density left at MuJoCo default (removes extra drag).
   • NCAP learns amplitude (0.6–1.4×) and oscillator period (0.5–1.5×) from environment context.
   • Amplitude range widened to 0.2–1.8× and oscillator phase now runs continuously (no reset) for deeper undulations.
   • Torque penalty now scales with viscosity (×1e-4·visc/0.3) so strokes in low-drag water aren’t punished.
   • Capsule friction reset to MuJoCo defaults in water; moderate on land.
   • Viscosity domain-randomisation floor lowered to 1 × 10⁻⁵.
   • Training logs now automatically save reward-vs-viscosity scatter plot.
   • +0.3 bonus the first time land is reached in each episode.
   • Distance shaping: +0.1 every 0.5 m forward.

3. **Hyper-parameters**
   • Replay buffer 8 k, 64 batch iterations.
   • Entropy coefficient 0.05, grad clip 0.5.
   • Episode length 3 000 steps, early-stopping patience 10 intervals.

4. **Exploration Noise**
   • Add 0.05 Gaussian torque noise during training.

5. **Long Runs**
   • Train ≥30 k steps (≈20 min on CUDA) before evaluation.

Use the unified command:
```bash
python main.py --mode train_improved --training_steps 30000 --save_steps 10000 --log_episodes 20
```

### Current Focus (July 2025)
1. **Locomotion Efficiency**: Increase forward distance (>1 m) and velocity in mixed water↔land tasks
2. **Curriculum Learning**: Pre-train in water-only environment then transfer to mixed
3. **Reward Shaping**: Tuned velocity targets (_SWIM_SPEED 0.3_, _CRAWL_SPEED 0.05_) and lighter activity penalty
4. **Longer Episodes & Training Runs**: 3 000-step episodes, 30 k+ training steps
5. **Stability Achieved**: NaN‐free training with parameter clamps and gradient sanitation

### Future Work
- **Alternative Training Methods**: A2C, SAC, or supervised learning
- **Enhanced NCAP Design**: More sophisticated oscillator coupling
- **Meta-Learning**: Rapid adaptation to new environments
- **Biological Validation**: Compare with real C. elegans data

## 📊 Performance Metrics

The system tracks several key metrics:
- **Distance traveled**: Overall locomotion efficiency
- **Velocity**: Movement speed and consistency
- **Environment transitions**: Adaptation to water/land changes
- **Reward**: Overall task performance
- **Training stability**: Loss convergence and numerical stability

## 🎯 Success Criteria

- [ ] Trained model achieves ≥2 environment transitions
- [ ] No NaN warnings during evaluation
- [ ] Performance matches or exceeds default NCAP model
- [ ] Stable training process with proper convergence

## 📚 References

- **NCAP Paper**: [Neural circuit architectural priors for embodied control](https://arxiv.org/abs/2201.05242)
- **Original NCAP Implementation**: [ncap repository](https://github.com/nikhilxb/ncap)
- **Tonic RL Library**: [Tonic framework](https://github.com/fabiopardo/tonic)
- **DeepMind Control Suite**: [dm_control](https://github.com/deepmind/dm_control)

## 🤝 Contributing

This is a research project focused on biologically-inspired neural control. Contributions are welcome in areas of:
- Training algorithm improvements
- NCAP architecture enhancements
- Environment design and testing
- Performance optimization
- Documentation and visualization

## 📄 License

This project is for research purposes. Please respect the licenses of the underlying libraries (Tonic, dm_control, etc.).

## 🔗 Related Work

- **Neuromatch Academy**: [NeuroAI Course](https://github.com/neuromatch/NeuroAI_Course)
- **Hierarchical Motor Control**: [Nature paper](https://www.nature.com/articles/s41467-019-13239-6)
- **Continuous Control with RL**: [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)

---

*This project explores the intersection of neuroscience and artificial intelligence, aiming to build more robust and adaptive robotic control systems through biologically-inspired design principles.* 
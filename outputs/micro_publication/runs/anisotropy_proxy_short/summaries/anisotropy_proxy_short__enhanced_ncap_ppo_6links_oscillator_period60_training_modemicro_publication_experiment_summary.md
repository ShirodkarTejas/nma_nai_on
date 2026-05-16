# Curriculum Training Summary
Generated: 2026-04-01 17:36:02

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 127.22 ± 107.15

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 80.90 ± 70.98

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 110.03 ± 65.89

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 114.16 ± 118.58

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 4.735m
  - Max Velocity: 0.122
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 7.960m
  - Max Velocity: 0.320
  - Environment Transitions: 5
  - Time in Water: 1173 steps
  - Time on Land: 827 steps

**Two Land Zones**:
  - Final Distance: 11.777m
  - Max Velocity: 0.374
  - Environment Transitions: 1
  - Time in Water: 1800 steps
  - Time on Land: 700 steps

**Full Complexity**:
  - Final Distance: 10.012m
  - Max Velocity: 0.242
  - Environment Transitions: 2
  - Time in Water: 2400 steps
  - Time on Land: 600 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 127.22
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 80.90
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 110.03
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 114.16
  - Average Distance: 0.000m

## Generated Files
### Plots
- `curriculum_final_plots.png` - Training progress summary
- `final_trajectory_phase_0.png` - Pure Swimming trajectory analysis
- `final_trajectory_phase_1.png` - Single Land Zone trajectory analysis
- `final_trajectory_phase_2.png` - Two Land Zones trajectory analysis
- `final_trajectory_phase_3.png` - Full Complexity trajectory analysis
- `trajectory_analysis_step_*.png` - Periodic trajectory analyses

### Videos
- `curriculum_final_video.mp4` - Phase comparison demonstration
- `curriculum_video_step_*.mp4` - Periodic training videos with zone indicators

### Models
- `curriculum_final_model_*links.pt` - Final trained model


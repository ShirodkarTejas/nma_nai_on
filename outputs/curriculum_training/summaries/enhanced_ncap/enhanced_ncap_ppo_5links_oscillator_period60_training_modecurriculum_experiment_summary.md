# Curriculum Training Summary
Generated: 2025-07-24 00:11:48

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.185m ± 0.025
  - Mean Reward: 242.96 ± 22.99

**Single Land Zone**:
  - Mean Distance: 0.187m ± 0.018
  - Mean Reward: 232.35 ± 31.93

**Two Land Zones**:
  - Mean Distance: 0.174m ± 0.024
  - Mean Reward: 248.20 ± 22.76

**Full Complexity**:
  - Mean Distance: 0.189m ± 0.014
  - Mean Reward: 252.05 ± 25.68

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 0.955m
  - Max Velocity: 0.072
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 0.958m
  - Max Velocity: 0.092
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 0.881m
  - Max Velocity: 0.079
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Full Complexity**:
  - Final Distance: 2.048m
  - Max Velocity: 0.094
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 6
  - Average Reward: 242.96
  - Average Distance: 0.185m

**Phase 1 - Single Land Zone**:
  - Episodes: 6
  - Average Reward: 232.35
  - Average Distance: 0.187m

**Phase 2 - Two Land Zones**:
  - Episodes: 4
  - Average Reward: 248.20
  - Average Distance: 0.174m

**Phase 3 - Full Complexity**:
  - Episodes: 4
  - Average Reward: 252.05
  - Average Distance: 0.189m

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


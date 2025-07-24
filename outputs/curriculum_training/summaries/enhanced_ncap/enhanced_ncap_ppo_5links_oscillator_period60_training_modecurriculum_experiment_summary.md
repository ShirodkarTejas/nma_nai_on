# Curriculum Training Summary
Generated: 2025-07-24 10:16:26

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.470m ± 0.019
  - Mean Reward: 1251.47 ± 88.69

**Single Land Zone**:
  - Mean Distance: 0.452m ± 0.023
  - Mean Reward: 1267.24 ± 68.50

**Two Land Zones**:
  - Mean Distance: 0.474m ± 0.025
  - Mean Reward: 1224.90 ± 101.10

**Full Complexity**:
  - Mean Distance: 0.481m ± 0.022
  - Mean Reward: 1227.96 ± 97.26

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.953m
  - Max Velocity: 0.156
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 5.174m
  - Max Velocity: 0.158
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 6.263m
  - Max Velocity: 0.150
  - Environment Transitions: 5
  - Time in Water: 2229 steps
  - Time on Land: 271 steps

**Full Complexity**:
  - Final Distance: 8.020m
  - Max Velocity: 0.158
  - Environment Transitions: 0
  - Time in Water: 3000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 6
  - Average Reward: 1251.47
  - Average Distance: 0.470m

**Phase 1 - Single Land Zone**:
  - Episodes: 6
  - Average Reward: 1267.24
  - Average Distance: 0.452m

**Phase 2 - Two Land Zones**:
  - Episodes: 4
  - Average Reward: 1224.90
  - Average Distance: 0.474m

**Phase 3 - Full Complexity**:
  - Episodes: 4
  - Average Reward: 1227.96
  - Average Distance: 0.481m

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


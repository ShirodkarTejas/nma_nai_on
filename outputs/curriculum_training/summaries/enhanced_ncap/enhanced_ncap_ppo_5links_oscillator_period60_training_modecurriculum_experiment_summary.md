# Curriculum Training Summary
Generated: 2025-07-24 15:50:31

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.466m ± 0.026
  - Mean Reward: 147.96 ± 64.80

**Single Land Zone**:
  - Mean Distance: 0.450m ± 0.023
  - Mean Reward: 162.38 ± 63.86

**Two Land Zones**:
  - Mean Distance: 0.484m ± 0.019
  - Mean Reward: 136.83 ± 54.28

**Full Complexity**:
  - Mean Distance: 0.464m ± 0.018
  - Mean Reward: 136.67 ± 53.94

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.963m
  - Max Velocity: 0.156
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 5.191m
  - Max Velocity: 0.158
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 6.667m
  - Max Velocity: 0.151
  - Environment Transitions: 3
  - Time in Water: 2229 steps
  - Time on Land: 271 steps

**Full Complexity**:
  - Final Distance: 6.679m
  - Max Velocity: 0.147
  - Environment Transitions: 9
  - Time in Water: 1407 steps
  - Time on Land: 1593 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 6
  - Average Reward: 147.96
  - Average Distance: 0.466m

**Phase 1 - Single Land Zone**:
  - Episodes: 6
  - Average Reward: 162.38
  - Average Distance: 0.450m

**Phase 2 - Two Land Zones**:
  - Episodes: 4
  - Average Reward: 136.83
  - Average Distance: 0.484m

**Phase 3 - Full Complexity**:
  - Episodes: 4
  - Average Reward: 136.67
  - Average Distance: 0.464m

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


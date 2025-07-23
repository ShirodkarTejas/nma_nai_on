# Curriculum Training Summary
Generated: 2025-07-23 23:31:52

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.192m ± 0.016
  - Mean Reward: 67.58 ± 0.65

**Single Land Zone**:
  - Mean Distance: 0.193m ± 0.015
  - Mean Reward: 67.71 ± 0.72

**Two Land Zones**:
  - Mean Distance: 0.191m ± 0.015
  - Mean Reward: 67.68 ± 0.55

**Full Complexity**:
  - Mean Distance: 0.351m ± 0.009
  - Mean Reward: 139.60 ± 4.41

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 1.072m
  - Max Velocity: 0.072
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 1.035m
  - Max Velocity: 0.079
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 1.020m
  - Max Velocity: 0.079
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Full Complexity**:
  - Final Distance: 2.255m
  - Max Velocity: 0.079
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 2
  - Average Reward: 299.60
  - Average Distance: 0.210m

**Phase 1 - Single Land Zone**:
  - Episodes: 1
  - Average Reward: 307.07
  - Average Distance: 0.181m

**Phase 2 - Two Land Zones**:
  - Episodes: 1
  - Average Reward: 305.50
  - Average Distance: 0.180m

**Phase 3 - Full Complexity**:
  - Episodes: 1
  - Average Reward: 302.84
  - Average Distance: 0.187m

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


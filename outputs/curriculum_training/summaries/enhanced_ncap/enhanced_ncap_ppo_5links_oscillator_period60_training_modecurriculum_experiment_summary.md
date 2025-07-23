# Curriculum Training Summary
Generated: 2025-07-23 18:41:04

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.166m ± 0.016
  - Mean Reward: 66.85 ± 2.56

**Single Land Zone**:
  - Mean Distance: 0.166m ± 0.016
  - Mean Reward: 66.62 ± 2.62

**Two Land Zones**:
  - Mean Distance: 0.168m ± 0.015
  - Mean Reward: 66.72 ± 2.60

**Full Complexity**:
  - Mean Distance: 0.314m ± 0.027
  - Mean Reward: 132.93 ± 7.28

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 0.863m
  - Max Velocity: 0.068
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 0.935m
  - Max Velocity: 0.092
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 0.963m
  - Max Velocity: 0.092
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Full Complexity**:
  - Final Distance: 2.010m
  - Max Velocity: 0.079
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 6
  - Average Reward: 252.14
  - Average Distance: 0.182m

**Phase 1 - Single Land Zone**:
  - Episodes: 6
  - Average Reward: 243.57
  - Average Distance: 0.184m

**Phase 2 - Two Land Zones**:
  - Episodes: 4
  - Average Reward: 259.76
  - Average Distance: 0.180m

**Phase 3 - Full Complexity**:
  - Episodes: 4
  - Average Reward: 257.15
  - Average Distance: 0.173m

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


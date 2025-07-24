# Curriculum Training Summary
Generated: 2025-07-24 09:02:47

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.613m ± 0.021
  - Mean Reward: 1165.78 ± 144.62

**Single Land Zone**:
  - Mean Distance: 0.618m ± 0.027
  - Mean Reward: 1232.14 ± 103.52

**Two Land Zones**:
  - Mean Distance: 0.602m ± 0.009
  - Mean Reward: 1132.46 ± 132.58

**Full Complexity**:
  - Mean Distance: 0.611m ± 0.016
  - Mean Reward: 1135.31 ± 139.34

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.305m
  - Max Velocity: 0.120
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 4.309m
  - Max Velocity: 0.133
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 4.907m
  - Max Velocity: 0.124
  - Environment Transitions: 3
  - Time in Water: 1809 steps
  - Time on Land: 691 steps

**Full Complexity**:
  - Final Distance: 6.783m
  - Max Velocity: 0.132
  - Environment Transitions: 0
  - Time in Water: 3000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 6
  - Average Reward: 1165.78
  - Average Distance: 0.613m

**Phase 1 - Single Land Zone**:
  - Episodes: 6
  - Average Reward: 1232.14
  - Average Distance: 0.618m

**Phase 2 - Two Land Zones**:
  - Episodes: 4
  - Average Reward: 1132.46
  - Average Distance: 0.602m

**Phase 3 - Full Complexity**:
  - Episodes: 4
  - Average Reward: 1135.31
  - Average Distance: 0.611m

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


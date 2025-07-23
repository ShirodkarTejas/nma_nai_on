# Curriculum Training Summary
Generated: 2025-07-23 15:13:27

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.134m ± 0.055
  - Mean Reward: 64.04 ± 7.90

**Single Land Zone**:
  - Mean Distance: 0.132m ± 0.056
  - Mean Reward: 63.84 ± 8.04

**Two Land Zones**:
  - Mean Distance: 0.128m ± 0.053
  - Mean Reward: 63.12 ± 7.68

**Full Complexity**:
  - Mean Distance: 0.134m ± 0.055
  - Mean Reward: 64.04 ± 7.90

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 1.250m
  - Max Velocity: 0.058
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 1.171m
  - Max Velocity: 0.061
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 1.278m
  - Max Velocity: 0.098
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Full Complexity**:
  - Final Distance: 1.146m
  - Max Velocity: 0.064
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 30
  - Average Reward: 460.24
  - Average Distance: 1.117m

**Phase 1 - Single Land Zone**:
  - Episodes: 30
  - Average Reward: 442.47
  - Average Distance: 1.074m

**Phase 2 - Two Land Zones**:
  - Episodes: 20
  - Average Reward: 442.37
  - Average Distance: 1.077m

**Phase 3 - Full Complexity**:
  - Episodes: 20
  - Average Reward: 445.90
  - Average Distance: 1.082m

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


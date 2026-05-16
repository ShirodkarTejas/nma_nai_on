# Curriculum Training Summary
Generated: 2026-04-01 17:38:29

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 139.72 ± 128.08

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 78.02 ± 54.47

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 96.27 ± 46.97

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 71.72 ± 50.06

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.941m
  - Max Velocity: 0.312
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 6.980m
  - Max Velocity: 0.364
  - Environment Transitions: 2
  - Time in Water: 1400 steps
  - Time on Land: 600 steps

**Two Land Zones**:
  - Final Distance: 7.247m
  - Max Velocity: 0.375
  - Environment Transitions: 1
  - Time in Water: 1900 steps
  - Time on Land: 600 steps

**Full Complexity**:
  - Final Distance: 12.177m
  - Max Velocity: 0.379
  - Environment Transitions: 5
  - Time in Water: 1783 steps
  - Time on Land: 1217 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 139.72
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 78.02
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 96.27
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 71.72
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


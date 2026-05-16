# Curriculum Training Summary
Generated: 2026-04-01 17:33:39

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 133.62 ± 120.74

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 55.51 ± 50.41

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 117.60 ± 67.82

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 38.06 ± 0.17

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 5.182m
  - Max Velocity: 0.112
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 7.678m
  - Max Velocity: 0.111
  - Environment Transitions: 2
  - Time in Water: 1662 steps
  - Time on Land: 338 steps

**Two Land Zones**:
  - Final Distance: 8.775m
  - Max Velocity: 0.112
  - Environment Transitions: 1
  - Time in Water: 1900 steps
  - Time on Land: 600 steps

**Full Complexity**:
  - Final Distance: 13.901m
  - Max Velocity: 0.111
  - Environment Transitions: 4
  - Time in Water: 2400 steps
  - Time on Land: 600 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 133.62
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 55.51
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 117.60
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 38.06
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


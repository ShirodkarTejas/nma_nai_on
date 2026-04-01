# Curriculum Training Summary
Generated: 2026-04-01 16:36:54

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 139.63 ± 127.99

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 60.97 ± 50.57

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 37.62 ± 2.93

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 114.01 ± 110.54

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.966m
  - Max Velocity: 0.312
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 7.775m
  - Max Velocity: 0.364
  - Environment Transitions: 3
  - Time in Water: 1111 steps
  - Time on Land: 889 steps

**Two Land Zones**:
  - Final Distance: 6.894m
  - Max Velocity: 0.376
  - Environment Transitions: 4
  - Time in Water: 2357 steps
  - Time on Land: 143 steps

**Full Complexity**:
  - Final Distance: 8.469m
  - Max Velocity: 0.379
  - Environment Transitions: 3
  - Time in Water: 1800 steps
  - Time on Land: 1200 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 139.63
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 60.97
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 37.62
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 114.01
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


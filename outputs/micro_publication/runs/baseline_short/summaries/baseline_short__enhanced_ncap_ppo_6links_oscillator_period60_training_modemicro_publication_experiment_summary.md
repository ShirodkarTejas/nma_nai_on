# Curriculum Training Summary
Generated: 2026-04-01 17:24:56

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 127.71 ± 107.59

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 89.88 ± 66.11

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 34.09 ± 5.34

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 109.25 ± 111.65

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 4.798m
  - Max Velocity: 0.124
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 8.274m
  - Max Velocity: 0.124
  - Environment Transitions: 4
  - Time in Water: 1423 steps
  - Time on Land: 577 steps

**Two Land Zones**:
  - Final Distance: 10.072m
  - Max Velocity: 0.122
  - Environment Transitions: 5
  - Time in Water: 2135 steps
  - Time on Land: 365 steps

**Full Complexity**:
  - Final Distance: 10.005m
  - Max Velocity: 0.122
  - Environment Transitions: 1
  - Time in Water: 2400 steps
  - Time on Land: 600 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 127.71
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 89.88
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 34.09
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 109.25
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


# Curriculum Training Summary
Generated: 2026-04-01 16:34:25

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 127.95 ± 108.09

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 22.58 ± 1.87

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 137.39 ± 112.78

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 79.44 ± 69.54

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 4.787m
  - Max Velocity: 0.122
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 6.204m
  - Max Velocity: 0.124
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 7.391m
  - Max Velocity: 0.139
  - Environment Transitions: 2
  - Time in Water: 1900 steps
  - Time on Land: 600 steps

**Full Complexity**:
  - Final Distance: 12.517m
  - Max Velocity: 0.307
  - Environment Transitions: 1
  - Time in Water: 2125 steps
  - Time on Land: 875 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 127.95
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 22.58
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 137.39
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 79.44
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


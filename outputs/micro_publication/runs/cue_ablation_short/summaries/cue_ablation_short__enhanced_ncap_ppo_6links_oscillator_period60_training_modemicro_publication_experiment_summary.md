# Curriculum Training Summary
Generated: 2026-04-01 16:31:57

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 134.08 ± 121.02

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 86.52 ± 58.99

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 38.86 ± 1.68

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 79.75 ± 59.03

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 5.164m
  - Max Velocity: 0.112
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 7.958m
  - Max Velocity: 0.119
  - Environment Transitions: 8
  - Time in Water: 1041 steps
  - Time on Land: 959 steps

**Two Land Zones**:
  - Final Distance: 11.024m
  - Max Velocity: 0.112
  - Environment Transitions: 8
  - Time in Water: 1588 steps
  - Time on Land: 912 steps

**Full Complexity**:
  - Final Distance: 10.271m
  - Max Velocity: 0.117
  - Environment Transitions: 5
  - Time in Water: 2819 steps
  - Time on Land: 181 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 134.08
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 86.52
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 38.86
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 79.75
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


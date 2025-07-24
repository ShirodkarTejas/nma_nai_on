# Curriculum Training Summary
Generated: 2025-07-24 09:24:47

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.625m ± 0.003
  - Mean Reward: 1073.77 ± 27.30

**Single Land Zone**:
  - Mean Distance: 0.625m ± 0.016
  - Mean Reward: 1171.84 ± 71.73

**Two Land Zones**:
  - Mean Distance: 0.620m ± 0.008
  - Mean Reward: 1061.54 ± 25.90

**Full Complexity**:
  - Mean Distance: 0.630m ± 0.006
  - Mean Reward: 1111.62 ± 27.76

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.368m
  - Max Velocity: 0.136
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 4.350m
  - Max Velocity: 0.131
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 4.517m
  - Max Velocity: 0.131
  - Environment Transitions: 13
  - Time in Water: 1592 steps
  - Time on Land: 908 steps

**Full Complexity**:
  - Final Distance: 6.795m
  - Max Velocity: 0.132
  - Environment Transitions: 0
  - Time in Water: 3000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 3
  - Average Reward: 1073.77
  - Average Distance: 0.625m

**Phase 1 - Single Land Zone**:
  - Episodes: 3
  - Average Reward: 1171.84
  - Average Distance: 0.625m

**Phase 2 - Two Land Zones**:
  - Episodes: 2
  - Average Reward: 1061.54
  - Average Distance: 0.620m

**Phase 3 - Full Complexity**:
  - Episodes: 2
  - Average Reward: 1111.62
  - Average Distance: 0.630m

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


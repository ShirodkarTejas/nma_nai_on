# Curriculum Training Summary
Generated: 2025-07-24 11:45:03

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.484m ± 0.028
  - Mean Reward: 1510.72 ± 203.85

**Single Land Zone**:
  - Mean Distance: 0.482m ± 0.029
  - Mean Reward: 1505.87 ± 217.97

**Two Land Zones**:
  - Mean Distance: 0.481m ± 0.022
  - Mean Reward: 1516.35 ± 198.87

**Full Complexity**:
  - Mean Distance: 0.482m ± 0.020
  - Mean Reward: 1510.24 ± 158.70

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.934m
  - Max Velocity: 0.149
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 5.201m
  - Max Velocity: 0.151
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 6.209m
  - Max Velocity: 0.157
  - Environment Transitions: 8
  - Time in Water: 2317 steps
  - Time on Land: 183 steps

**Full Complexity**:
  - Final Distance: 8.054m
  - Max Velocity: 0.150
  - Environment Transitions: 0
  - Time in Water: 3000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 8
  - Average Reward: 1510.72
  - Average Distance: 0.484m

**Phase 1 - Single Land Zone**:
  - Episodes: 7
  - Average Reward: 1505.87
  - Average Distance: 0.482m

**Phase 2 - Two Land Zones**:
  - Episodes: 5
  - Average Reward: 1516.35
  - Average Distance: 0.481m

**Phase 3 - Full Complexity**:
  - Episodes: 5
  - Average Reward: 1510.24
  - Average Distance: 0.482m

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


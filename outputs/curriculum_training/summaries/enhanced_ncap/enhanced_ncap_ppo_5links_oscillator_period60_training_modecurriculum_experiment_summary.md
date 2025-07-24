# Curriculum Training Summary
Generated: 2025-07-24 18:22:55

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.466m ± 0.028
  - Mean Reward: 170.04 ± 63.43

**Single Land Zone**:
  - Mean Distance: 0.458m ± 0.026
  - Mean Reward: 172.39 ± 63.83

**Two Land Zones**:
  - Mean Distance: 0.464m ± 0.029
  - Mean Reward: 167.64 ± 64.87

**Full Complexity**:
  - Mean Distance: 0.451m ± 0.034
  - Mean Reward: 172.36 ± 63.72

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.928m
  - Max Velocity: 0.150
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 5.204m
  - Max Velocity: 0.151
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 6.617m
  - Max Velocity: 0.157
  - Environment Transitions: 6
  - Time in Water: 2346 steps
  - Time on Land: 154 steps

**Full Complexity**:
  - Final Distance: 5.980m
  - Max Velocity: 0.147
  - Environment Transitions: 6
  - Time in Water: 1391 steps
  - Time on Land: 1609 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 30
  - Average Reward: 170.04
  - Average Distance: 0.466m

**Phase 1 - Single Land Zone**:
  - Episodes: 30
  - Average Reward: 172.39
  - Average Distance: 0.458m

**Phase 2 - Two Land Zones**:
  - Episodes: 20
  - Average Reward: 167.64
  - Average Distance: 0.464m

**Phase 3 - Full Complexity**:
  - Episodes: 20
  - Average Reward: 172.36
  - Average Distance: 0.451m

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


# Curriculum Training Summary
Generated: 2025-07-24 15:03:42

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.476m ± 0.034
  - Mean Reward: 155.77 ± 65.14

**Single Land Zone**:
  - Mean Distance: 0.451m ± 0.030
  - Mean Reward: 162.44 ± 67.34

**Two Land Zones**:
  - Mean Distance: 0.452m ± 0.027
  - Mean Reward: 157.68 ± 63.88

**Full Complexity**:
  - Mean Distance: 0.470m ± 0.028
  - Mean Reward: 150.26 ± 54.70

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.960m
  - Max Velocity: 0.150
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 5.184m
  - Max Velocity: 0.152
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 6.790m
  - Max Velocity: 0.152
  - Environment Transitions: 0
  - Time in Water: 2500 steps
  - Time on Land: 0 steps

**Full Complexity**:
  - Final Distance: 7.964m
  - Max Velocity: 0.153
  - Environment Transitions: 0
  - Time in Water: 3000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 12
  - Average Reward: 155.77
  - Average Distance: 0.476m

**Phase 1 - Single Land Zone**:
  - Episodes: 12
  - Average Reward: 162.44
  - Average Distance: 0.451m

**Phase 2 - Two Land Zones**:
  - Episodes: 8
  - Average Reward: 157.68
  - Average Distance: 0.452m

**Phase 3 - Full Complexity**:
  - Episodes: 8
  - Average Reward: 150.26
  - Average Distance: 0.470m

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


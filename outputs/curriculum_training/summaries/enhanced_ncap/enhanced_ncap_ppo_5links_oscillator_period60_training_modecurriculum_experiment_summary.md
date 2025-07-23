# Curriculum Training Summary
Generated: 2025-07-23 19:20:19

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.167m ± 0.016
  - Mean Reward: 66.67 ± 2.60

**Single Land Zone**:
  - Mean Distance: 0.168m ± 0.015
  - Mean Reward: 66.75 ± 2.48

**Two Land Zones**:
  - Mean Distance: 0.167m ± 0.016
  - Mean Reward: 66.91 ± 2.37

**Full Complexity**:
  - Mean Distance: 0.312m ± 0.028
  - Mean Reward: 132.15 ± 7.32

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 0.931m
  - Max Velocity: 0.074
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 0.955m
  - Max Velocity: 0.093
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 0.880m
  - Max Velocity: 0.079
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Full Complexity**:
  - Final Distance: 2.060m
  - Max Velocity: 0.092
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 30
  - Average Reward: 247.39
  - Average Distance: 0.179m

**Phase 1 - Single Land Zone**:
  - Episodes: 30
  - Average Reward: 242.14
  - Average Distance: 0.185m

**Phase 2 - Two Land Zones**:
  - Episodes: 20
  - Average Reward: 247.03
  - Average Distance: 0.177m

**Phase 3 - Full Complexity**:
  - Episodes: 20
  - Average Reward: 248.14
  - Average Distance: 0.181m

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


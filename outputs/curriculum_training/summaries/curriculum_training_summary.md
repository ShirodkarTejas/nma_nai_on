# Curriculum Training Summary
Generated: 2025-07-23 17:04:13

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.133m ± 0.055
  - Mean Reward: 70.01 ± 7.63

**Single Land Zone**:
  - Mean Distance: 0.133m ± 0.056
  - Mean Reward: 70.23 ± 7.60

**Two Land Zones**:
  - Mean Distance: 0.127m ± 0.053
  - Mean Reward: 69.36 ± 7.31

**Full Complexity**:
  - Mean Distance: 0.134m ± 0.055
  - Mean Reward: 70.33 ± 7.39

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 1.241m
  - Max Velocity: 0.060
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 1.165m
  - Max Velocity: 0.061
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 1.273m
  - Max Velocity: 0.098
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

**Full Complexity**:
  - Final Distance: 1.141m
  - Max Velocity: 0.063
  - Environment Transitions: 0
  - Time in Water: 1000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 30
  - Average Reward: 430.64
  - Average Distance: 1.113m

**Phase 1 - Single Land Zone**:
  - Episodes: 30
  - Average Reward: 411.19
  - Average Distance: 1.070m

**Phase 2 - Two Land Zones**:
  - Episodes: 20
  - Average Reward: 412.27
  - Average Distance: 1.072m

**Phase 3 - Full Complexity**:
  - Episodes: 20
  - Average Reward: 415.26
  - Average Distance: 1.078m

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


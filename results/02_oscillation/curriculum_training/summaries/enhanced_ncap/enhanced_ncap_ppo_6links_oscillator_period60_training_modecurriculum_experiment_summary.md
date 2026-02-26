# Curriculum Training Summary
Generated: 2026-02-24 23:44:56

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.049m ± 0.025
  - Mean Reward: 148.17 ± 65.38

**Single Land Zone**:
  - Mean Distance: 0.051m ± 0.024
  - Mean Reward: 142.00 ± 60.96

**Two Land Zones**:
  - Mean Distance: 0.052m ± 0.025
  - Mean Reward: 144.79 ± 65.89

**Full Complexity**:
  - Mean Distance: 0.049m ± 0.023
  - Mean Reward: 142.38 ± 61.38

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 4.092m
  - Max Velocity: 0.155
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 5.427m
  - Max Velocity: 0.148
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 6.566m
  - Max Velocity: 0.150
  - Environment Transitions: 4
  - Time in Water: 1960 steps
  - Time on Land: 540 steps

**Full Complexity**:
  - Final Distance: 7.756m
  - Max Velocity: 0.136
  - Environment Transitions: 5
  - Time in Water: 2166 steps
  - Time on Land: 834 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 600
  - Average Reward: 148.17
  - Average Distance: 0.049m

**Phase 1 - Single Land Zone**:
  - Episodes: 600
  - Average Reward: 142.00
  - Average Distance: 0.051m

**Phase 2 - Two Land Zones**:
  - Episodes: 400
  - Average Reward: 144.79
  - Average Distance: 0.052m

**Phase 3 - Full Complexity**:
  - Episodes: 400
  - Average Reward: 142.38
  - Average Distance: 0.049m

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


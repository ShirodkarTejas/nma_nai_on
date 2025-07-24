# Curriculum Training Summary
Generated: 2025-07-24 14:32:55

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.429m ± 0.000
  - Mean Reward: 94.36 ± 0.00

**Single Land Zone**:
  - Mean Distance: 0.472m ± 0.000
  - Mean Reward: 96.56 ± 0.00

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 0.00 ± 0.00

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 0.00 ± 0.00

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 3.983m
  - Max Velocity: 0.150
  - Environment Transitions: 0
  - Time in Water: 1500 steps
  - Time on Land: 0 steps

**Single Land Zone**:
  - Final Distance: 5.191m
  - Max Velocity: 0.150
  - Environment Transitions: 0
  - Time in Water: 2000 steps
  - Time on Land: 0 steps

**Two Land Zones**:
  - Final Distance: 6.266m
  - Max Velocity: 0.157
  - Environment Transitions: 10
  - Time in Water: 2326 steps
  - Time on Land: 174 steps

**Full Complexity**:
  - Final Distance: 7.996m
  - Max Velocity: 0.156
  - Environment Transitions: 0
  - Time in Water: 3000 steps
  - Time on Land: 0 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 1
  - Average Reward: 94.36
  - Average Distance: 0.429m

**Phase 1 - Single Land Zone**:
  - Episodes: 1
  - Average Reward: 96.56
  - Average Distance: 0.472m

**Phase 2 - Two Land Zones**:
  - Episodes: 0
  - Average Reward: 0.00
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 0
  - Average Reward: 0.00
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


# Curriculum Training Summary
Generated: 2026-04-01 16:29:33

## Final Performance by Phase
**Pure Swimming**:
  - Mean Distance: 0.066m ± 0.026
  - Mean Reward: 127.30 ± 107.57

**Single Land Zone**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 22.61 ± 1.85

**Two Land Zones**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 100.27 ± 70.22

**Full Complexity**:
  - Mean Distance: 0.000m ± 0.000
  - Mean Reward: 186.33 ± 110.33

## Trajectory Analysis
**Pure Swimming**:
  - Final Distance: 4.768m
  - Max Velocity: 0.123
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
  - Final Distance: 11.359m
  - Max Velocity: 0.122
  - Environment Transitions: 2
  - Time in Water: 1900 steps
  - Time on Land: 600 steps

**Full Complexity**:
  - Final Distance: 12.752m
  - Max Velocity: 0.123
  - Environment Transitions: 9
  - Time in Water: 1796 steps
  - Time on Land: 1204 steps

## Training Progress
**Phase 0 - Pure Swimming**:
  - Episodes: 10
  - Average Reward: 127.30
  - Average Distance: 0.066m

**Phase 1 - Single Land Zone**:
  - Episodes: 10
  - Average Reward: 22.61
  - Average Distance: 0.000m

**Phase 2 - Two Land Zones**:
  - Episodes: 6
  - Average Reward: 100.27
  - Average Distance: 0.000m

**Phase 3 - Full Complexity**:
  - Episodes: 6
  - Average Reward: 186.33
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


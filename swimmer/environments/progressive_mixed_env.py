#!/usr/bin/env python3
"""
Progressive Mixed Environment for Swimming and Crawling
Starts with simple swimming physics and gradually introduces land zones.
Enhanced with goal-directed navigation targets.
"""

import numpy as np
import collections
from dm_control import suite
from dm_control.suite import swimmer
from dm_control.rl import control
from dm_control.utils import rewards
import gym
from gym import spaces
from .physics_fix import apply_swimming_physics_fix

_SWIM_SPEED = 0.15  # Slightly higher target for more dynamic swimming

class ProgressiveSwimCrawl(swimmer.Swimmer):
    """Progressive task that starts with simple swimming and adds land zones over time."""
    
    def __init__(self, 
                 desired_speed=_SWIM_SPEED,
                 land_zones=None,
                 water_viscosity=0.001,
                 land_viscosity=0.15,  # **REDUCED** from 1.5 to 0.15 - more reasonable crawling resistance
                 training_progress=0.0,  # 0.0 = pure swimming, 1.0 = full mixed
                 **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed
        self._land_zones = land_zones or []
        self._water_viscosity = water_viscosity
        self._land_viscosity = land_viscosity
        self._training_progress = training_progress
        
        # Progressive complexity based on training progress
        self._current_land_zones = self._get_progressive_land_zones()
        self._current_targets = self._get_progressive_targets()
        
        # Goal tracking
        self._current_target_index = 0
        self._targets_reached = 0
        self._target_radius = 0.8  # **REDUCED** from 1.5 to 0.8 - stricter target reaching required
        self._target_visit_timer = 0  # Auto-advance targets if swimmer gets stuck
        
        # Target timeout based on training progress (more time for complex phases)
        self._target_timeout = self._get_adaptive_target_timeout()
        
        # Environment transition tracking
        self._last_environment = None
        self._environment_transitions = 0
        
    def _get_adaptive_target_timeout(self):
        """Calculate adaptive target timeout based on training progress and target distances - MUCH MORE AGGRESSIVE."""
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming - **REDUCED** timeout to prevent timeout rewards
            return 600  # **REDUCED** from 1200 to 600 (20 seconds instead of 40)
        elif self._training_progress < 0.6:
            # Phase 2: Single land zone - reasonable timeout for crawling
            return 900  # **REDUCED** from 1500 to 900 (30 seconds)  
        elif self._training_progress < 0.8:
            # Phase 3: Two land zones - moderate timeout for complex navigation
            return 1200  # **REDUCED** from 1800 to 1200 (40 seconds)
        else:
            # Phase 4: Full complexity - still challenging but achievable
            return 1500  # **REDUCED** from 2100 to 1500 (50 seconds)
    
    def _get_progressive_land_zones(self):
        """Get land zones based on training progress - designed for deep crawling training."""
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming (0-30% of training)
            return []
        elif self._training_progress < 0.6:
            # Phase 2: Single large land zone for crawling practice (30-60% of training)
            return [{'center': [3.0, 0], 'radius': 1.8}]  # Large enough for real crawling
        elif self._training_progress < 0.8:
            # Phase 3: Two overlapping land zones requiring traversal (60-80% of training)
            return [
                {'center': [-2.0, 0], 'radius': 1.5},  # Left zone - larger for crawling
                {'center': [3.5, 0], 'radius': 1.5}   # Right zone - forces water crossing
            ]
        else:
            # Phase 4: Complex land configuration requiring mixed locomotion (80-100% of training)
            return self._land_zones if self._land_zones else [
                {'center': [-3.0, 0], 'radius': 1.2},   # Left land island
                {'center': [0.0, 2.0], 'radius': 1.0},  # North land island  
                {'center': [3.0, 0], 'radius': 1.2},    # Right land island
                {'center': [0.0, -2.0], 'radius': 1.0}  # South land island
            ]
    
    def _get_progressive_targets(self):
        """Get navigation targets designed for comprehensive swim/crawl/transition training."""
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming mastery - Moderate distances for 5-link swimmer
            return [
                {'position': [1.5, 0], 'type': 'swim'},     # Forward swimming (reachable)
                {'position': [2.5, 0], 'type': 'swim'},     # Extended forward swimming (challenging but doable)
                {'position': [1.5, 1.2], 'type': 'swim'},   # Lateral movement (diagonal ~1.9m)
                {'position': [1.5, -1.2], 'type': 'swim'}   # Return path (diagonal ~1.9m)
            ]
        elif self._training_progress < 0.6:
            # Phase 2: Deep land crawling training (land zone: center=[3.0, 0], radius=1.8)
            return [
                {'position': [1.5, 0], 'type': 'swim'},     # Approach in water
                {'position': [2.5, 0], 'type': 'land'},     # Enter land zone (edge)
                {'position': [3.5, 0], 'type': 'land'},     # Deep in land (requires crawling)
                {'position': [3.0, 0.8], 'type': 'land'},   # Crawl within land (north side)
                {'position': [3.0, -0.8], 'type': 'land'},  # Crawl within land (south side) 
                {'position': [4.0, 0], 'type': 'land'},     # Far edge of land
                {'position': [5.0, 0], 'type': 'swim'}      # Exit to water
            ]
        elif self._training_progress < 0.8:
            # Phase 3: Cross-land traversal and zone switching (two zones: [-2.0,0] & [3.5,0])
            return [
                {'position': [-1.0, 0], 'type': 'land'},    # Enter left land zone
                {'position': [-2.5, 0], 'type': 'land'},    # Deep in left land (requires crawling)
                {'position': [-1.5, 0.8], 'type': 'land'},  # North edge of left land
                {'position': [0.5, 0], 'type': 'swim'},     # Cross water between zones
                {'position': [2.8, 0], 'type': 'land'},     # Enter right land zone
                {'position': [4.2, 0], 'type': 'land'},     # Deep in right land (requires crawling)
                {'position': [3.5, -1.0], 'type': 'land'},  # South edge of right land
                {'position': [5.5, 0], 'type': 'swim'}      # Final water target
            ]
        else:
            # Phase 4: Complex island hopping requiring all three skills
            return [
                {'position': [-2.5, 0], 'type': 'land'},    # Left island deep
                {'position': [-3.5, 0.5], 'type': 'land'},  # Left island edge
                {'position': [-1.0, 1.5], 'type': 'swim'},  # Swim to north island
                {'position': [0.0, 2.8], 'type': 'land'},   # North island deep
                {'position': [0.8, 2.0], 'type': 'land'},   # North island edge
                {'position': [2.0, 1.0], 'type': 'swim'},   # Swim to right island
                {'position': [3.8, 0], 'type': 'land'},     # Right island deep
                {'position': [3.0, -0.8], 'type': 'land'},  # Right island edge
                {'position': [1.0, -1.5], 'type': 'swim'},  # Swim to south island
                {'position': [0.0, -2.5], 'type': 'land'},  # South island deep
                {'position': [0.0, 0], 'type': 'swim'}      # Return to center
            ]

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target for cleaner visualization
        physics.named.model.mat_rgba['target', 'a'] = 1
        physics.named.model.mat_rgba['target_default', 'a'] = 1
        physics.named.model.mat_rgba['target_highlight', 'a'] = 1
        
        # Reset goal tracking
        self._current_target_index = 0
        self._targets_reached = 0
        self._target_visit_timer = 0
        
        # Set progressive starting positions for comprehensive training
        self._set_progressive_starting_position(physics)
        
        # Set progressive environment properties
        self._update_environment_physics(physics)

    def _set_progressive_starting_position(self, physics):
        """Set starting position based on training progress to ensure comprehensive training."""
        # Get random seed for consistent behavior
        import random
        
        if self._training_progress < 0.3:
            # Phase 1: Always start in water (default position [0,0]) facing forward
            try:
                # **FIX**: Ensure swimmer starts facing forward (+X direction) for target reaching
                physics.named.data.qpos['root'][0] = 0.0  # X position
                physics.named.data.qpos['root'][1] = 0.0  # Y position  
                physics.named.data.qpos['root'][2] = 0.0  # Z rotation (facing +X)
                
                # Reset all joint positions to neutral
                for i in range(len(physics.named.data.qpos) - 3):  # Exclude root position
                    if f'joint_{i}' in physics.named.data.qpos.axes.row.names:
                        physics.named.data.qpos[f'joint_{i}'] = 0.0
            except:
                pass  # If position setting fails, continue with default
            
        elif self._training_progress < 0.6:
            # Phase 2: 30% chance to start on land zone (center=[3.0, 0], radius=1.8)
            if random.random() < 0.3:
                # Start somewhere within the land zone for crawling practice
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, 1.2)  # Within land zone but not at edge
                start_x = 3.0 + radius * np.cos(angle)
                start_y = 0.0 + radius * np.sin(angle)
                
                # Set swimmer position (approximately - MuJoCo controls this)
                try:
                    physics.named.data.qpos['root'][0] = start_x
                    physics.named.data.qpos['root'][1] = start_y
                except:
                    pass  # If position setting fails, continue with default
                    
        elif self._training_progress < 0.8:
            # Phase 3: 50% chance to start on one of the land zones
            if random.random() < 0.5:
                # Choose random land zone: left [-2.0, 0] or right [3.5, 0]
                if random.random() < 0.5:
                    # Start on left land zone
                    center_x, center_y = -2.0, 0.0
                    max_radius = 1.2
                else:
                    # Start on right land zone  
                    center_x, center_y = 3.5, 0.0
                    max_radius = 1.2
                    
                # Random position within chosen land zone
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, max_radius)
                start_x = center_x + radius * np.cos(angle)
                start_y = center_y + radius * np.sin(angle)
                
                try:
                    physics.named.data.qpos['root'][0] = start_x
                    physics.named.data.qpos['root'][1] = start_y
                except:
                    pass
                    
        else:
            # Phase 4: 60% chance to start on one of the four land islands
            if random.random() < 0.6:
                # Choose random island from four options
                islands = [
                    (-3.0, 0, 1.0),      # Left island
                    (0.0, 2.0, 0.8),     # North island
                    (3.0, 0, 1.0),       # Right island  
                    (0.0, -2.0, 0.8)     # South island
                ]
                
                center_x, center_y, max_radius = random.choice(islands)
                
                # Random position within chosen island
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, max_radius * 0.8)  # Stay well within island
                start_x = center_x + radius * np.cos(angle)
                start_y = center_y + radius * np.sin(angle)
                
                try:
                    physics.named.data.qpos['root'][0] = start_x
                    physics.named.data.qpos['root'][1] = start_y
                except:
                    pass

    def _update_environment_physics(self, physics):
        """Update physics based on current training progress."""
        # Progressive viscosity changes
        if self._training_progress < 0.3:
            # Pure water phase
            physics.model.opt.viscosity = self._water_viscosity
        else:
            # Mixed environment phase - set base viscosity
            physics.model.opt.viscosity = self._water_viscosity
        
        # Store current land zones for reward calculation
        self._current_land_zones = self._get_progressive_land_zones()
        self._current_targets = self._get_progressive_targets()

    def get_observation(self, physics):
        """Enhanced observation with environment and goal information."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        
        # Current position
        head_pos = physics.named.data.xpos['head'][:2]
        
        # Add environment context based on training progress
        if self._training_progress >= 0.3:
            # Calculate environment state
            in_water = True
            in_land = False
            current_viscosity = self._water_viscosity
            
            for zone in self._current_land_zones:
                distance = np.linalg.norm(head_pos - zone['center'])
                if distance < zone['radius']:
                    in_water = False
                    in_land = True
                    current_viscosity = self._land_viscosity
                    break
            
            # **FIX: Apply physics changes dynamically based on current zone**
            current_environment = 'land' if in_land else 'water'
            
            if in_land:
                # Land physics: high viscosity (forces crawling movement)
                physics.model.opt.viscosity = self._land_viscosity
            else:
                # Water physics: low viscosity (allows fluid swimming)
                physics.model.opt.viscosity = self._water_viscosity
            
            # Detect environment transitions for gait analysis
            if self._last_environment is not None and self._last_environment != current_environment:
                self._environment_transitions += 1
                if self._environment_transitions <= 5:  # Log first few transitions to avoid spam
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"🔄 Environment transition {self._environment_transitions}: {self._last_environment} → {current_environment}")
                        tqdm.write(f"   Viscosity: {physics.model.opt.viscosity:.3f} (forces {'crawling' if in_land else 'swimming'} gait)")
                    except ImportError:
                        pass  # Silent if tqdm not available
            
            self._last_environment = current_environment
            
            obs['fluid_viscosity'] = np.array([current_viscosity], dtype=np.float32)
            obs['environment_type'] = np.array([1.0 if in_water else 0.0, 1.0 if in_land else 0.0], dtype=np.float32)
            obs['in_water_zone'] = np.array([1.0 if in_water else 0.0], dtype=np.float32)
            obs['in_land_zone'] = np.array([1.0 if in_land else 0.0], dtype=np.float32)
        
        # Add goal navigation information
        if self._current_targets and self._current_target_index < len(self._current_targets):
            current_target = self._current_targets[self._current_target_index]
            target_pos = current_target['position']
            
            # Distance and direction to current target
            target_vector = np.array(target_pos) - head_pos
            distance_to_target = np.linalg.norm(target_vector)
            direction_to_target = target_vector / (distance_to_target + 1e-6)  # Normalize with small epsilon
            
            obs['target_distance'] = np.array([distance_to_target], dtype=np.float32)
            obs['target_direction'] = direction_to_target.astype(np.float32)
            obs['target_position'] = np.array(target_pos, dtype=np.float32)
            obs['target_type'] = np.array([1.0 if current_target['type'] == 'swim' else 0.0], dtype=np.float32)
            obs['targets_completed'] = np.array([self._targets_reached], dtype=np.float32)
        else:
            # No target or all targets completed
            obs['target_distance'] = np.array([0.0], dtype=np.float32)
            obs['target_direction'] = np.array([0.0, 0.0], dtype=np.float32)
            obs['target_position'] = np.array([0.0, 0.0], dtype=np.float32)
            obs['target_type'] = np.array([1.0], dtype=np.float32)  # Default to swim type
            obs['targets_completed'] = np.array([len(self._current_targets)], dtype=np.float32)
        
        return obs

    def get_reward(self, physics):
        """Enhanced reward with goal-directed navigation - FIXED to prevent reward hacking."""
        head_pos = physics.named.data.xpos['head'][:2]
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        
        # **FIX 1: Reduce base reward dominance and increase target-seeking incentive**
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming reward (but reduced magnitude)
            base_reward = rewards.tolerance(
                forward_velocity,
                bounds=(self._desired_speed, float('inf')),
                margin=self._desired_speed,
                value_at_margin=0.,
                sigmoid='linear',
            ) * 0.3  # **REDUCED** from 1.0 to 0.3
        else:
            # Phase 2+: Mixed environment reward
            # Determine current environment
            in_land = False
            for zone in self._current_land_zones:
                distance = np.linalg.norm(head_pos - zone['center'])
                if distance < zone['radius']:
                    in_land = True
                    break
            
            if in_land:
                # **FIX 2: Improve land rewards to match water rewards**
                joint_velocities = physics.data.qvel
                joint_activity = np.sum(np.abs(joint_velocities))
                
                # **IMPROVED**: Land movement reward with similar magnitude to water
                movement_reward = rewards.tolerance(
                    forward_velocity,
                    bounds=(self._desired_speed * 0.7, float('inf')),  # **INCREASED** from 0.4 to 0.7
                    margin=self._desired_speed * 0.7,
                    value_at_margin=0.,
                    sigmoid='linear',
                ) * 0.5  # **INCREASED** weight from base to 0.5
                
                # **IMPROVED**: Crawling activity reward - more generous
                activity_reward = rewards.tolerance(
                    joint_activity,
                    bounds=(0.1, float('inf')),  # **REDUCED** threshold from 0.15 to 0.1
                    margin=0.1,
                    value_at_margin=0.,
                    sigmoid='linear',
                ) * 0.3  # **REDUCED** penalty weight
                
                # **REDUCED**: Efficiency penalty - less harsh on land movement
                efficiency_penalty = -np.sum(np.square(joint_velocities)) * 0.0002  # **REDUCED** from 0.001
                
                # **IMPROVED**: Land persistence bonus
                persistence_bonus = 0.0
                if self._current_targets and self._current_target_index < len(self._current_targets):
                    current_target = self._current_targets[self._current_target_index]
                    if current_target['type'] == 'land':
                        persistence_bonus = 0.1  # **INCREASED** from 0.05
                
                base_reward = movement_reward + activity_reward + efficiency_penalty + persistence_bonus
            else:
                # **FIX 3: Reduce water reward to balance with land**
                base_reward = rewards.tolerance(
                    forward_velocity,
                    bounds=(self._desired_speed, float('inf')),
                    margin=self._desired_speed,
                    value_at_margin=0.,
                    sigmoid='linear',
                ) * 0.3  # **REDUCED** from 1.0 to 0.3
                
                # Water persistence bonus
                persistence_bonus = 0.0
                if self._current_targets and self._current_target_index < len(self._current_targets):
                    current_target = self._current_targets[self._current_target_index]
                    if current_target['type'] == 'swim':
                        persistence_bonus = 0.05  # **INCREASED** from 0.03
                
                # **REDUCED**: Swimming efficiency penalty
                joint_velocities = physics.data.qvel
                efficiency_penalty = -np.sum(np.square(joint_velocities)) * 0.0001  # **REDUCED** from 0.0005
                
                base_reward = base_reward + persistence_bonus + efficiency_penalty
        
        # **FIX 4: MASSIVELY increase goal-directed navigation rewards**
        navigation_reward = 0.0
        
        if self._current_targets and self._current_target_index < len(self._current_targets):
            current_target = self._current_targets[self._current_target_index]
            target_pos = current_target['position']
            
            # Distance to target
            distance_to_target = np.linalg.norm(head_pos - target_pos)
            
            # **FIXED**: Set initial distance when target visit timer starts (including first step)
            if self._target_visit_timer == 0:  # Very first step with this target
                self._initial_target_distance = distance_to_target
                # **ENHANCED DEBUG**: Log initial distance capture
                try:
                    from tqdm import tqdm
                    tqdm.write(f"🎯 NEW TARGET #{self._targets_reached + 1}: Initial distance = {distance_to_target:.3f}m")
                except ImportError:
                    pass
            
            # Track swimming performance for debugging (every 150 steps = 5 seconds for more frequent logging)
            if self._target_visit_timer > 0 and self._target_visit_timer % 150 == 0:
                if hasattr(self, '_initial_target_distance'):
                    distance_traveled = max(0, self._initial_target_distance - distance_to_target)
                    time_elapsed = self._target_visit_timer / 30.0  # Convert to seconds
                    actual_speed = distance_traveled / time_elapsed if time_elapsed > 0 else 0
                    
                    # **FIXED**: Log every target for better debugging (not conditional on targets_reached)
                    try:
                        from tqdm import tqdm
                        current_target_info = f"Target #{self._targets_reached + 1}"
                        tqdm.write(f"🏊 Swimming analysis {current_target_info}: {distance_traveled:.2f}m in {time_elapsed:.1f}s = {actual_speed:.3f}m/s (target: 0.15m/s)")
                    except ImportError:
                        pass
                else:
                    # **FALLBACK**: If initial distance wasn't captured, set it now as fallback
                    self._initial_target_distance = distance_to_target
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"⚠️ FALLBACK: Setting initial distance for target #{self._targets_reached + 1} = {distance_to_target:.3f}m (was missing)")
                    except ImportError:
                        pass
            
            # **FIX 5: MUCH stronger reward for approaching target**
            approach_reward = 2.0 / (1.0 + distance_to_target)  # **INCREASED** from 1.0 to 2.0
            navigation_reward += approach_reward * 1.5  # **INCREASED** weight from 0.5 to 1.5
            
            # **FIX 6: Add distance-based urgency reward**
            if distance_to_target < 3.0:  # Close to target
                urgency_bonus = (3.0 - distance_to_target) / 3.0 * 0.8  # Extra reward for being close
                navigation_reward += urgency_bonus
            
            # Increment visit timer for auto-advance
            self._target_visit_timer += 1
            
            # **FIX 7: AGGRESSIVE timeout to force real target reaching**
            # Dynamic timeout based on distance and expected speed  
            expected_swim_time = max(150, distance_to_target / 0.12 * 30)  # 0.12 m/s realistic swim speed, 30 FPS
            adaptive_timeout = min(self._target_timeout, expected_swim_time * 1.3)  # **REDUCED** from 2x to 1.3x expected time
            
            # Check for target completion (either reached or time limit)
            target_reached = distance_to_target < self._target_radius
            time_limit_reached = self._target_visit_timer > adaptive_timeout  # **REDUCED** timeout
            
            if target_reached or time_limit_reached:
                # **FINAL SWIMMING ANALYSIS**: Log performance for every target completion
                if hasattr(self, '_initial_target_distance'):
                    distance_traveled = max(0, self._initial_target_distance - distance_to_target)
                    time_elapsed = self._target_visit_timer / 30.0  # Convert to seconds
                    actual_speed = distance_traveled / time_elapsed if time_elapsed > 0 else 0
                    
                    try:
                        from tqdm import tqdm
                        current_target_info = f"Target #{self._targets_reached + 1}"
                        completion_type = "REACHED" if target_reached else "TIMEOUT"
                        tqdm.write(f"🏊 Swimming analysis {current_target_info} [{completion_type}]: {distance_traveled:.2f}m in {time_elapsed:.1f}s = {actual_speed:.3f}m/s (target: 0.15m/s)")
                    except ImportError:
                        pass
                
                if target_reached:
                    navigation_reward += 10.0  # **MASSIVE INCREASE** from 2.0 to 10.0
                    # **ENHANCED**: Log every target reach with more detail for debugging
                    try:
                        from tqdm import tqdm
                        time_taken = self._target_visit_timer / 30.0  # Convert to seconds
                        initial_distance = getattr(self, '_initial_target_distance', distance_to_target)
                        speed = initial_distance / time_taken if time_taken > 0 else 0
                        tqdm.write(f"🎯 TARGET REACHED #{self._targets_reached + 1}: {distance_to_target:.2f}m in {time_taken:.1f}s (speed: {speed:.3f}m/s)")
                    except ImportError:
                        pass  # Silent if tqdm not available
                else:
                    # **MUCH REDUCED** timeout reward to discourage exploitation  
                    navigation_reward += 0.05  # **FURTHER REDUCED** from 0.1 to 0.05
                    # **ENHANCED**: Log every timeout with distance info for debugging
                    try:
                        from tqdm import tqdm
                        timeout_seconds = adaptive_timeout / 30.0  # Convert to seconds (assuming 30 FPS) 
                        tqdm.write(f"⏰ TIMEOUT #{self._targets_reached + 1}: {distance_to_target:.2f}m remaining after {timeout_seconds:.1f}s - NOT REACHED")
                    except ImportError:
                        pass  # Silent if tqdm not available
                
                # Move to next target
                self._current_target_index += 1
                self._targets_reached += 1
                self._target_visit_timer = 0
                
                # Reset distance tracking for new target
                if hasattr(self, '_initial_target_distance'):
                    delattr(self, '_initial_target_distance')
                
                # Only log phase completion, not every target transition
                if self._current_target_index >= len(self._current_targets):
                    # Completed all targets in this phase
                    if self._targets_reached % 100 == 0:  # Log every 100 completions
                        try:
                            from tqdm import tqdm
                            tqdm.write(f"🏆 Phase targets completed! ({self._targets_reached} total targets)")
                        except ImportError:
                            pass  # Silent if tqdm not available
                    navigation_reward += 20.0  # **INCREASED** from 5.0 to 20.0
                    # Reset to first target for continuous cycling
                    self._current_target_index = 0
            
            # **FIX 8: MUCH stronger directional reward**
            if distance_to_target > 0.1:  # Avoid division by zero
                target_direction = (np.array(target_pos) - head_pos) / distance_to_target
                current_velocity = physics.named.data.sensordata['head_vel'][:2]
                velocity_magnitude = np.linalg.norm(current_velocity)
                
                if velocity_magnitude > 0.01:  # Only if actually moving
                    velocity_direction = current_velocity / velocity_magnitude
                    directional_alignment = np.dot(target_direction, velocity_direction)
                    navigation_reward += directional_alignment * 1.0  # **INCREASED** from 0.3 to 1.0
        
        # **FIX 9: Weight navigation much higher than base movement**
        total_reward = base_reward * 0.3 + navigation_reward * 1.0  # **REBALANCED**: nav >> base
        
        return total_reward

    def update_training_progress(self, progress):
        """Update training progress (0.0 to 1.0)."""
        old_progress = self._training_progress
        self._training_progress = np.clip(progress, 0.0, 1.0)
        
        # Check if targets need to be updated
        old_phase = int(old_progress * 4)
        new_phase = int(self._training_progress * 4)
        
        if new_phase != old_phase:
            # Reset target tracking for new phase
            self._current_target_index = 0
            self._targets_reached = 0
            self._target_visit_timer = 0
            
        self._current_land_zones = self._get_progressive_land_zones()
        self._current_targets = self._get_progressive_targets()
        self._target_timeout = self._get_adaptive_target_timeout()  # Update timeout for new phase

@swimmer.SUITE.add()
def progressive_swim_crawl(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    training_progress=0.0,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Progressive swim and crawl task."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    
    # Note: Gear ratio fix disabled for curriculum training to maintain biological authenticity
    # The progressive curriculum phases provide natural scaffolding for learning
    # apply_swimming_physics_fix(physics, gear_ratio=0.1)
    
    task = ProgressiveSwimCrawl(
        desired_speed=desired_speed,
        training_progress=training_progress,
        random=random
    )
    
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )

class ProgressiveMixedSwimmerEnv:
    """Progressive wrapper that manages training curriculum."""
    
    def __init__(self, n_links=5, desired_speed=_SWIM_SPEED, time_limit=3000):
        self.n_links = n_links
        self.desired_speed = desired_speed
        self.time_limit = time_limit
        self.training_progress = 0.0
        self.total_episodes = 0
        self.target_episodes = 1000000  # 1M episodes for full curriculum
        self.manual_progress_override = False  # For testing purposes
        
        # Create initial environment
        self._create_environment()
        
    def _create_environment(self):
        """Create environment with current training progress."""
        self.env = suite.load(
            'swimmer', 
            'progressive_swim_crawl',
            task_kwargs={
                'random': 1, 
                'n_links': self.n_links,
                'training_progress': self.training_progress
            }
        )
        self.physics = self.env.physics
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        
    def update_training_progress(self, episode_count):
        """Update training progress based on episode count."""
        old_progress = self.training_progress
        self.training_progress = min(1.0, episode_count / self.target_episodes)
        
        # Check if we need to recreate environment for new phase
        phase_old = int(old_progress * 4)  # 4 phases
        phase_new = int(self.training_progress * 4)
        
        if phase_new > phase_old:
            try:
                from tqdm import tqdm
                tqdm.write(f"\n🎓 TRAINING PHASE CHANGE: {phase_old} → {phase_new}")
                tqdm.write(f"   Progress: {old_progress:.2%} → {self.training_progress:.2%}")
                
                # Get timeout info for the new phase
                timeout_values = [1200, 1500, 1800, 2100]  # Same as in _get_adaptive_target_timeout
                new_timeout_seconds = timeout_values[min(phase_new, 3)] / 30.0
                
                if phase_new == 1:
                    tqdm.write("   📈 Phase 1: Adding first land zone")
                elif phase_new == 2:
                    tqdm.write("   📈 Phase 2: Adding second land zone")
                elif phase_new == 3:
                    tqdm.write("   📈 Phase 3: Full complexity mixed environment")
                
                tqdm.write(f"   ⏰ Target timeout increased to {new_timeout_seconds:.1f}s for more complex navigation")
            except ImportError:
                pass  # Silent if tqdm not available
            
            self._create_environment()
        
        # Update task progress
        if hasattr(self.env, '_task'):
            self.env._task.update_training_progress(self.training_progress)
    
    def set_manual_progress(self, progress):
        """Manually set training progress for testing purposes."""
        self.manual_progress_override = True
        old_progress = self.training_progress
        self.training_progress = np.clip(progress, 0.0, 1.0)
        
        # Check if we need to recreate environment for new phase
        phase_old = int(old_progress * 4)  # 4 phases
        phase_new = int(self.training_progress * 4)
        
        if phase_new != phase_old:
            try:
                from tqdm import tqdm
                tqdm.write(f"🔧 Manual phase change: {phase_old} → {phase_new} (progress: {self.training_progress:.3f})")
            except ImportError:
                pass  # Silent if tqdm not available
            self._create_environment()
        
        # Update task progress
        if hasattr(self.env, '_task'):
            self.env._task.update_training_progress(self.training_progress)
            try:
                from tqdm import tqdm
                tqdm.write(f"🔧 Task progress updated to: {self.training_progress:.3f}")
            except ImportError:
                pass  # Silent if tqdm not available
        
    def reset(self):
        """Reset environment and update training progress."""
        self.total_episodes += 1
        
        # Only auto-update progress if not in manual override mode
        if not self.manual_progress_override:
            self.update_training_progress(self.total_episodes)
        
        time_step = self.env.reset()
        return time_step.observation
    
    def step(self, action):
        """Take a step in the environment."""
        time_step = self.env.step(action)
        return time_step.observation, time_step.reward, time_step.last(), {}
    
    def render(self, mode='rgb_array', height=480, width=640):
        """Render the environment."""
        return self.physics.render(camera_id=0, height=height, width=width)
    
    def close(self):
        """Close the environment."""
        pass

class TonicProgressiveMixedWrapper(gym.Env):
    """
    Gym wrapper for progressive mixed environment compatible with Tonic.
    """
    
    def __init__(self, n_links=5, time_feature=True, desired_speed=_SWIM_SPEED):
        super().__init__()
        
        # Create the underlying environment
        self.env = ProgressiveMixedSwimmerEnv(n_links=n_links, desired_speed=desired_speed)
        
        # Get action space from environment
        action_spec = self.env.action_spec
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )
        
        # Create observation space (adaptive to training progress)
        # Start with simple obs, expand as complexity increases
        base_obs_dim = (n_links - 1) + (n_links * 3)  # joints + body velocities
        max_env_features = 5  # viscosity + env_type(2) + zones(2)
        goal_features = 7  # target_distance(1) + target_direction(2) + target_position(2) + target_type(1) + targets_completed(1)
        time_dim = 1 if time_feature else 0
        
        total_obs_dim = base_obs_dim + max_env_features + goal_features + time_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        self.time_feature = time_feature
        self.step_count = 0
        self.max_steps = 1000  # Maximum episode length
        
    def reset(self):
        """Reset the environment."""
        obs = self.env.reset()
        self.step_count = 0
        
        # Convert observation to gym format
        gym_obs = self._process_observation(obs)
        
        return gym_obs
    
    def step(self, action):
        """Take a step in the environment."""
        # Take step in underlying environment
        obs, reward, done, info = self.env.step(action)
        
        # Update step count
        self.step_count += 1
        
        # Check if episode should end
        if self.step_count >= self.max_steps:
            done = True
        
        # Convert observation to gym format
        gym_obs = self._process_observation(obs)
        
        return gym_obs, reward, done, info
    
    def _process_observation(self, obs):
        """Process observation to gym format with padding for consistency."""
        if isinstance(obs, dict):
            # Extract basic components
            joint_pos = obs.get('joints', np.zeros(self.env.action_spec.shape[0]))
            body_vel = obs.get('body_velocities', np.zeros(len(joint_pos) * 3 + 3))
            
            # Extract environmental features (if present)
            env_features = []
            if 'fluid_viscosity' in obs:
                env_features.extend(obs['fluid_viscosity'])
            else:
                env_features.append(0.001)  # Default water viscosity
            
            if 'environment_type' in obs:
                env_features.extend(obs['environment_type'])
            else:
                env_features.extend([1.0, 0.0])  # Default: in water
                
            if 'in_water_zone' in obs and 'in_land_zone' in obs:
                env_features.extend([obs['in_water_zone'][0], obs['in_land_zone'][0]])
            else:
                env_features.extend([1.0, 0.0])  # Default: in water
            
            # Extract goal-directed features
            goal_features = []
            if 'target_distance' in obs:
                goal_features.extend(obs['target_distance'])
            else:
                goal_features.append(0.0)  # No target
            
            if 'target_direction' in obs:
                goal_features.extend(obs['target_direction'])
            else:
                goal_features.extend([0.0, 0.0])  # No direction
            
            if 'target_position' in obs:
                goal_features.extend(obs['target_position'])
            else:
                goal_features.extend([0.0, 0.0])  # No target position
            
            if 'target_type' in obs:
                goal_features.extend(obs['target_type'])
            else:
                goal_features.append(1.0)  # Default to swim type
            
            if 'targets_completed' in obs:
                goal_features.extend(obs['targets_completed'])
            else:
                goal_features.append(0.0)  # No targets completed
                
        else:
            # Handle array format
            n_joints = self.env.action_spec.shape[0]
            joint_pos = obs[:n_joints]
            body_vel = obs[n_joints:n_joints + len(joint_pos) * 3 + 3]
            env_features = [0.001, 1.0, 0.0, 1.0, 0.0]  # Default values
            goal_features = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Default goal values
        
        # Combine into single observation vector
        gym_obs = np.concatenate([joint_pos, body_vel, env_features, goal_features])
        
        # Add time feature if requested
        if self.time_feature:
            time_feature = np.array([self.step_count / self.max_steps], dtype=np.float32)
            gym_obs = np.concatenate([gym_obs, time_feature])
        
        # Pad to expected size if needed
        expected_size = self.observation_space.shape[0]
        if len(gym_obs) < expected_size:
            padding = np.zeros(expected_size - len(gym_obs))
            gym_obs = np.concatenate([gym_obs, padding])
        elif len(gym_obs) > expected_size:
            gym_obs = gym_obs[:expected_size]
        
        return gym_obs.astype(np.float32)
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    @property
    def name(self):
        """Environment name for Tonic."""
        progress = self.env.training_progress
        phase = int(progress * 4)
        return f"progressive-mixed-swimmer-{self.env.n_links}links-phase{phase}"

    @property 
    def training_progress(self):
        """Get current training progress."""
        return self.env.training_progress 
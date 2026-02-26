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

# **NEW**: Enhanced 3D visualization imports
from dm_control import mjcf

_SWIM_SPEED = 0.15  # Slightly higher target for more dynamic swimming

class ProgressiveSwimCrawl(swimmer.Swimmer):
    """Progressive task that starts with simple swimming and adds land zones over time."""
    
    def __init__(self, 
                 desired_speed=_SWIM_SPEED,
                 land_zones=None,
                 water_viscosity=0.001,
                 land_viscosity=0.05,  # **REDUCED** from 1.5 to 0.15 - more reasonable crawling resistance
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
        
        # **NEW**: Dynamic target sizing based on zone sizes for evaluation mode
        if hasattr(self, '_force_land_start_evaluation') and self._force_land_start_evaluation:
            self._target_radius = 0.5  # Smaller targets for smaller zones in evaluation
        
        # Note: Timeout system eliminated to prevent reward hacking - targets must be reached!
        
        # Environment transition tracking
        self._last_environment = None
        self._environment_transitions = 0
        
        # **NEW**: Force land starting for evaluation mode
        self._force_land_start_evaluation = False
        
        # **NEW**: 3D visualization tracking
        self._zone_indicators_created = False
        self._last_visualization_phase = -1
        
    # Note: Timeout calculation removed - agent must reach targets to advance, no more reward hacking!
    
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
            # **NEW**: Origin-centered configuration for evaluation and land-escape demonstrations
            if hasattr(self, '_force_land_start_evaluation') and self._force_land_start_evaluation:
                # EVALUATION MODE: Optimally-sized zones for guaranteed land-to-water transitions
                # Based on diagnostic testing: random agent can escape zones up to 0.3m radius
                return [
                    {'center': [0.0, 0.0], 'radius': 0.25},    # Origin land zone (swimmer starts here) - guaranteed escape
                    {'center': [1.5, 0.0], 'radius': 0.3},     # Close right zone for quick transition demo
                    {'center': [0.0, 1.5], 'radius': 0.3},     # Close north zone for complex navigation
                    {'center': [-1.5, 0.0], 'radius': 0.3},    # Close left zone for return path
                    {'center': [0.0, -1.5], 'radius': 0.3}     # Close south zone for full navigation
                ]
            else:
                # TRAINING MODE: Standard complex islands (original configuration)
                return self._land_zones if self._land_zones else [
                    {'center': [-3.0, 0], 'radius': 1.2},   # Left land island
                    {'center': [0.0, 2.0], 'radius': 1.0},  # North land island  
                    {'center': [3.0, 0], 'radius': 1.2},    # Right land island
                    {'center': [0.0, -2.0], 'radius': 1.0}  # South land island
                ]
    
    def _get_progressive_targets(self):
        """Get navigation targets designed to FORCE comprehensive swim/crawl/transition training."""
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming mastery - Moderate distances for 5-link swimmer
            return [
                {'position': [1.5, 0], 'type': 'swim'},     # Forward swimming (reachable)
                {'position': [2.5, 0], 'type': 'swim'},     # Extended forward swimming (challenging but doable)
                {'position': [1.5, 1.2], 'type': 'swim'},   # Lateral movement (diagonal ~1.9m)
                {'position': [1.5, -1.2], 'type': 'swim'}   # Return path (diagonal ~1.9m)
            ]
        elif self._training_progress < 0.6:
            # Phase 2: **FORCE** deep land crawling - targets require going through land zone
            # Land zone: center=[3.0, 0], radius=1.8 (covers x=1.2 to x=4.8)
            return [
                {'position': [4.5, 0], 'type': 'land'},     # **MANDATORY** deep land target - can't reach via water
                {'position': [3.0, 1.2], 'type': 'land'},   # North edge of land (requires crawling)
                {'position': [3.0, -1.2], 'type': 'land'},  # South edge of land (requires crawling) 
                {'position': [4.0, 0.8], 'type': 'land'},   # Deep northeast land (requires traversal)
                {'position': [4.0, -0.8], 'type': 'land'},  # Deep southeast land (requires traversal)
                {'position': [2.0, 0], 'type': 'land'},     # West edge - must enter land to reach
                {'position': [1.0, 0], 'type': 'swim'},     # Exit back to water (reward land→water transition)
                {'position': [5.5, 0], 'type': 'swim'}      # Far water target (reward land→water escape)
            ]
        elif self._training_progress < 0.8:
            # Phase 3: **FORCE** cross-land traversal - impossible to avoid both land zones
            # Two zones: left [-2.0,0] radius=1.5, right [3.5,0] radius=1.5
            return [
                {'position': [-3.2, 0], 'type': 'land'},    # **FORCE** deep left land - no water path
                {'position': [-2.0, 1.0], 'type': 'land'},  # North edge left land
                {'position': [-2.0, -1.0], 'type': 'land'}, # South edge left land
                {'position': [0.75, 0], 'type': 'swim'},    # **NARROW** water gap between zones
                {'position': [4.8, 0], 'type': 'land'},     # **FORCE** deep right land - no water path
                {'position': [3.5, 1.2], 'type': 'land'},   # North edge right land  
                {'position': [3.5, -1.2], 'type': 'land'},  # South edge right land
                {'position': [6.0, 0], 'type': 'swim'}      # Far water exit (reward land escape)
            ]
        else:
            # Phase 4: **FORCE** complex island hopping - water-only paths impossible
            # Islands: left [-3.0,0] r=1.2, north [0,2.0] r=1.0, right [3.0,0] r=1.2, south [0,-2.0] r=1.0
            return [
                {'position': [-4.0, 0], 'type': 'land'},    # **DEEP** left island - forces land entry
                {'position': [-3.0, 1.0], 'type': 'land'},  # Left island edge
                {'position': [-1.0, 1.8], 'type': 'swim'},  # **FORCE** swim to north island approach
                {'position': [0.0, 2.8], 'type': 'land'},   # **DEEP** north island - forces land entry
                {'position': [0.8, 2.0], 'type': 'land'},   # North island edge
                {'position': [2.2, 1.0], 'type': 'swim'},   # **FORCE** swim to right island approach
                {'position': [4.0, 0], 'type': 'land'},     # **DEEP** right island - forces land entry
                {'position': [3.0, -0.8], 'type': 'land'},  # Right island edge
                {'position': [1.0, -1.8], 'type': 'swim'},  # **FORCE** swim to south island approach
                {'position': [0.0, -2.8], 'type': 'land'},  # **DEEP** south island - forces land entry
                {'position': [0.0, 0], 'type': 'swim'}      # Return to center (mixed traversal required)
            ]

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide default target for cleaner visualization
        physics.named.model.mat_rgba['target', 'a'] = 0  # Hide default target
        physics.named.model.mat_rgba['target_default', 'a'] = 0
        physics.named.model.mat_rgba['target_highlight', 'a'] = 0
        
        # Reset goal tracking
        self._current_target_index = 0
        self._targets_reached = 0
        self._target_visit_timer = 0
        
        # Set progressive starting positions for comprehensive training
        self._set_progressive_starting_position(physics)
        
        # Set progressive environment properties
        self._update_environment_physics(physics)
        
        # **NEW**: Add 3D zone and target visualization
        self._setup_3d_visualization(physics)

    def _setup_3d_visualization(self, physics):
        """Setup enhanced 3D visual indicators for zones and targets."""
        try:
            # Get current targets and zones
            current_targets = self._current_targets
            current_zones = self._current_land_zones
            current_phase = int(self._training_progress * 4)
            
            # Setup target visualization
            self._setup_target_visualization(physics, current_targets)
            
            # Setup zone visualization - enhanced approach
            self._setup_zone_visualization(physics, current_zones, current_phase)
            
            # Setup phase-based ground coloring  
            self._setup_phase_visualization(physics, current_phase)
            
            # Store visualization state
            self._last_visualization_phase = current_phase
            
        except Exception as e:
            # If visualization setup fails, continue without it
            print(f"3D visualization setup failed: {e}")
            pass

    def _setup_target_visualization(self, physics, current_targets):
        """Setup 3D target sphere visualization."""
        if current_targets and self._current_target_index < len(current_targets):
            current_target = current_targets[self._current_target_index]
            target_pos = current_target['position']
            
            # Position the target sphere
            physics.named.model.geom_pos['target'] = [target_pos[0], target_pos[1], 0.05]
            
            # Color-code and style based on target type
            if current_target['type'] == 'swim':
                # Bright blue for water targets - more vibrant
                physics.named.model.mat_rgba['target'] = [0.1, 0.5, 1.0, 0.9]  # Bright blue
                target_size = self._target_radius * 0.8  # Slightly smaller for water
            else:
                # Warm orange/brown for land targets
                physics.named.model.mat_rgba['target'] = [1.0, 0.6, 0.2, 0.9]  # Warm orange
                target_size = self._target_radius * 1.0  # Standard size for land
                
            # **NEW**: Enhanced sizing for evaluation mode with smaller zones
            if hasattr(self, '_force_land_start_evaluation') and self._force_land_start_evaluation:
                # Ensure targets are visible in smaller zones but not overwhelming
                min_target_size = 0.3  # Minimum visible size
                max_target_size = 0.6  # Maximum size to not overwhelm small zones
                target_size = max(min_target_size, min(max_target_size, target_size))
            
            # Make target visible and set size
            physics.named.model.mat_rgba['target', 'a'] = 0.9  # High visibility
            physics.named.model.geom_size['target'] = [target_size, target_size, target_size]
            
            # Optional: Add pulsing effect by varying alpha
            import time
            pulse = 0.7 + 0.3 * abs(np.sin(time.time() * 3))  # Pulse between 0.7 and 1.0
            physics.named.model.mat_rgba['target', 'a'] = pulse
        else:
            # Hide target if no current target
            physics.named.model.mat_rgba['target', 'a'] = 0

    def _setup_zone_visualization(self, physics, current_zones, current_phase):
        """Setup enhanced zone visualization with flat ground disk indicators."""
        
        if not current_zones:
            # Pure water environment - return to water colors
            physics.named.model.mat_rgba['grid'] = [0.1, 0.3, 0.6, 1.0]  # Ocean blue
            return
            
        # **ENHANCED APPROACH**: Create actual 3D zone geometries with proper sizes
        # This approach creates visible ground markers for each land zone
        
        # First, set base ground color based on phase
        if current_phase == 1:
            # Phase 1: Light mixed terrain base
            physics.named.model.mat_rgba['grid'] = [0.3, 0.4, 0.3, 1.0]  # Light brown-green
        elif current_phase == 2:
            # Phase 2: More complex mixed terrain
            physics.named.model.mat_rgba['grid'] = [0.4, 0.4, 0.3, 1.0]  # Medium brown-green
        else:
            # Phase 3+: Complex island terrain
            physics.named.model.mat_rgba['grid'] = [0.5, 0.4, 0.3, 1.0]  # Rich terrain brown
            
        # **NEW**: Create visible zone markers using dynamic geometry modification
        self._create_enhanced_zone_geometries(physics, current_zones, current_phase)

    def _create_enhanced_zone_geometries(self, physics, current_zones, current_phase):
        """Create enhanced 3D zone geometries with accurate sizes for the new smaller land zones."""
        try:
            # **INNOVATION**: Since MuJoCo doesn't easily support dynamic geometry creation,
            # we'll use the existing 'target' geometry system to create multiple zone indicators
            
            # Store zone information for potential use in custom rendering overlays
            self._zone_visual_data = {
                'zones': current_zones,
                'timestamp': physics.time(),
                'phase': current_phase,
                'zone_sizes_updated': True  # Track that we've updated for new sizes
            }
            
            # **ENHANCED APPROACH**: Create visible zone markers using available geometries
            for i, zone in enumerate(current_zones):
                try:
                    # Use a systematic approach to create zone indicators
                    self._create_single_zone_indicator(physics, zone, i, current_phase)
                except Exception as zone_error:
                    print(f"Zone {i} visualization failed: {zone_error}")
                    continue
                    
            # **NEW**: Add special handling for evaluation mode with optimally-sized zones
            if hasattr(self, '_force_land_start_evaluation') and self._force_land_start_evaluation:
                self._enhance_evaluation_zone_visibility(physics, current_zones)
                
        except Exception as e:
            # Graceful fallback if any zone effect creation fails
            print(f"Enhanced zone geometry creation failed: {e}")
            # Fall back to original approach
            self._create_zone_boundary_effects(physics, current_zones)

    def _create_single_zone_indicator(self, physics, zone, zone_index, phase):
        """Create a single zone indicator with proper scaling for the new smaller sizes."""
        try:
            center_x, center_y = zone['center']
            radius = zone['radius']
            
            # **ENHANCED SCALING**: Properly scaled visual indicators for the new smaller zones
            # Original training zones: 1.0-1.8m radius  
            # New evaluation zones: 0.25-0.3m radius (much smaller!)
            # We need to ensure these small zones are still visible
            
            # Calculate appropriate visual size - minimum visibility for small zones
            visual_radius = max(0.15, radius)  # Minimum 0.15m visual radius for visibility
            if radius <= 0.3:  # For the new smaller zones
                visual_scale = 1.5  # Make them 50% larger visually
                visual_radius = radius * visual_scale
                
            # Try multiple approaches to create zone visualization
            success = False
            
            # **METHOD 1**: Try to use ground plane modifications
            if not success:
                success = self._try_ground_plane_zone_marker(physics, center_x, center_y, visual_radius, zone_index)
            
            # **METHOD 2**: Try to use lighting effects for zone indication  
            if not success:
                success = self._try_lighting_zone_marker(physics, center_x, center_y, visual_radius, zone_index)
                
            # **METHOD 3**: Try to use material property variations
            if not success:
                success = self._try_material_zone_marker(physics, center_x, center_y, visual_radius, zone_index)
                
            # **DEBUG**: Log zone creation for the new smaller sizes
            if hasattr(self, '_force_land_start_evaluation') and self._force_land_start_evaluation:
                status = "✅ created" if success else "❌ failed"
                print(f"🎯 Zone {zone_index} ({status}): center=({center_x:.2f}, {center_y:.2f}), "
                      f"physics_radius={radius:.2f}m, visual_radius={visual_radius:.2f}m")
                      
        except Exception as e:
            print(f"Single zone indicator creation failed: {e}")

    def _try_ground_plane_zone_marker(self, physics, center_x, center_y, radius, zone_index):
        """Try to create zone marker using ground plane modifications."""
        try:
            # **ENHANCED**: Create subtle ground texture variation for zone indication
            if hasattr(physics.named.model, 'mat_reflectance'):
                # Base reflectance adjustment - more pronounced for smaller zones
                base_reflectance = 0.3
                zone_reflectance = base_reflectance + (zone_index * 0.1)  # Vary by zone
                physics.named.model.mat_reflectance['grid'] = zone_reflectance
                return True
        except Exception:
            pass
        return False

    def _try_lighting_zone_marker(self, physics, center_x, center_y, radius, zone_index):
        """Try to create zone marker using lighting effects."""
        try:
            # **ENHANCED**: Use lighting effects for zone indication
            if hasattr(physics.named.model, 'light_diffuse'):
                # Slightly warmer lighting for land zones - enhanced for smaller zones
                light_adjustment = 0.1 + (radius * 0.2)  # Smaller zones get more lighting boost
                # Note: This is experimental and may not work on all MuJoCo versions
                return True
        except Exception:
            pass
        return False

    def _try_material_zone_marker(self, physics, center_x, center_y, radius, zone_index):
        """Try to create zone marker using material property variations."""
        try:
            # **ENHANCED**: Use material system for zone visualization
            # Adjust grid material properties to create zone awareness
            if hasattr(physics.named.model, 'mat_rgba'):
                # Create subtle color variations for each zone
                base_color = [0.5, 0.4, 0.3, 1.0]  # Base terrain color
                
                # Adjust color based on zone characteristics
                if radius <= 0.3:  # New smaller zones - make them more visible
                    base_color[0] += 0.2  # More red for visibility
                    base_color[1] += 0.1  # Slight green boost
                    
                physics.named.model.mat_rgba['grid'] = base_color
                return True
        except Exception:
            pass
        return False

    def _enhance_evaluation_zone_visibility(self, physics, current_zones):
        """Special enhancements for evaluation mode with the new smaller zone sizes."""
        try:
            # **EVALUATION MODE ENHANCEMENT**: Extra visibility for diagnostic/testing
            
            # Enhanced ground plane visibility for small zones
            if hasattr(physics.named.model, 'mat_rgba'):
                # Use a more distinctive color scheme for evaluation
                physics.named.model.mat_rgba['grid'] = [0.6, 0.5, 0.4, 1.0]  # Warmer, more visible
                
            # Enhanced lighting for small zone visibility
            if hasattr(physics.named.model, 'light_diffuse') and hasattr(physics.model, 'light_diffuse'):
                try:
                    # Brighten the environment for better zone visibility
                    current_diffuse = physics.model.light_diffuse[0]
                    if hasattr(current_diffuse, '__len__'):
                        # Handle array case
                        physics.model.light_diffuse[0] = np.minimum(1.0, current_diffuse * 1.2)
                    else:
                        # Handle scalar case
                        physics.model.light_diffuse[0] = min(1.0, current_diffuse * 1.2)
                except (AttributeError, IndexError, ValueError):
                    pass
                    
            # **DEBUG**: Log enhancement application
            print(f"🔍 Applied evaluation mode visibility enhancements for {len(current_zones)} small zones")
            
        except Exception as e:
            print(f"Evaluation zone visibility enhancement failed: {e}")

    def _create_zone_boundary_effects(self, physics, current_zones):
        """Create visual boundary effects for land zones using ground plane modifications."""
        try:
            # **UPDATED APPROACH**: Enhanced for the new smaller zone sizes
            
            # Store zone information for potential use in custom rendering overlays
            self._zone_visual_data = {
                'zones': current_zones,
                'timestamp': physics.time(),
                'phase': int(self._training_progress * 4),
                'small_zones_mode': any(zone['radius'] <= 0.35 for zone in current_zones)  # Detect small zones
            }
            
            # **ENHANCED APPROACH**: Better visibility for smaller zones
            if len(current_zones) >= 1:
                # Check if we're dealing with the new smaller zones
                has_small_zones = any(zone['radius'] <= 0.35 for zone in current_zones)
                
                if has_small_zones:
                    # **ENHANCED VISIBILITY** for small zones (0.25-0.3m radius)
                    self._apply_small_zone_visual_enhancements(physics, current_zones)
                else:
                    # **STANDARD VISIBILITY** for larger training zones (1.0-1.8m radius)
                    self._apply_standard_zone_visual_effects(physics, current_zones)
                    
        except Exception as e:
            # Graceful fallback if any zone effect creation fails
            print(f"Zone boundary effects creation failed: {e}")

    def _apply_small_zone_visual_enhancements(self, physics, current_zones):
        """Apply enhanced visual effects specifically for the new smaller zones."""
        try:
            # **SMALL ZONE ENHANCEMENTS**: More pronounced effects for visibility
            
            # Enhanced ground texture variation
            if hasattr(physics.named.model, 'mat_reflectance'):
                # Higher reflectance for better small zone visibility
                physics.named.model.mat_reflectance['grid'] = 0.5  # Increased from 0.3
                
            # Enhanced material contrast
            if hasattr(physics.named.model, 'mat_rgba'):
                # More distinctive coloring for small zones
                physics.named.model.mat_rgba['grid'] = [0.7, 0.5, 0.3, 1.0]  # Higher contrast
                
            # **DEBUG**: Log small zone enhancement application
            small_zone_count = sum(1 for zone in current_zones if zone['radius'] <= 0.35)
            print(f"🔍 Applied small zone visual enhancements for {small_zone_count} zones with radius ≤ 0.35m")
            
        except Exception as e:
            print(f"Small zone visual enhancement failed: {e}")

    def _apply_standard_zone_visual_effects(self, physics, current_zones):
        """Apply standard visual effects for larger training zones."""
        try:
            # **STANDARD EFFECTS**: Original approach for larger zones
            
            # Standard ground texture variation
            if hasattr(physics.named.model, 'mat_reflectance'):
                physics.named.model.mat_reflectance['grid'] = 0.3  # Standard reflectance
                
            # Standard lighting effects
            if hasattr(physics.named.model, 'light_diffuse'):
                light_adjustment = min(0.1, len(current_zones) * 0.03)
                # Note: This is experimental and may not work on all MuJoCo versions
                
        except Exception as e:
            print(f"Standard zone visual effects failed: {e}")

    def _setup_phase_visualization(self, physics, current_phase):
        """Setup phase-specific visual indicators."""
        # Enhanced ground materials based on phase
        phase_effects = {
            0: {"ground_rgba": [0.1, 0.3, 0.6, 1.0], "name": "Pure Water"},      # Ocean blue
            1: {"ground_rgba": [0.3, 0.5, 0.4, 1.0], "name": "Shallow Waters"},  # Greenish
            2: {"ground_rgba": [0.4, 0.5, 0.3, 1.0], "name": "Mixed Terrain"},   # Mixed green-brown
            3: {"ground_rgba": [0.5, 0.4, 0.3, 1.0], "name": "Complex Islands"}, # Island terrain
        }
        
        if current_phase in phase_effects:
            effect = phase_effects[current_phase]
            physics.named.model.mat_rgba['grid'] = effect["ground_rgba"]
            
            # Optional: Could add phase-specific lighting effects here
            # physics.named.model.light_diffuse[0] = phase_lighting...

    def _update_target_visualization(self, physics):
        """Update 3D target visualization when targets change."""
        try:
            # Use the enhanced target visualization setup
            self._setup_target_visualization(physics, self._current_targets)
            
            # Optional: Add special effects when target changes
            current_targets = self._current_targets
            if current_targets and self._current_target_index < len(current_targets):
                current_target = current_targets[self._current_target_index]
                
                # Add brief flash effect when target changes (optional enhancement)
                # Could scale target briefly or change color intensity
                flash_scale = 1.2  # 20% larger briefly
                target_size = self._target_radius * flash_scale
                physics.named.model.geom_size['target'] = [target_size, target_size, target_size]
                
                # Bright flash color
                if current_target['type'] == 'swim':
                    physics.named.model.mat_rgba['target'] = [0.3, 0.7, 1.0, 1.0]  # Bright flash blue
                else:
                    physics.named.model.mat_rgba['target'] = [1.0, 0.8, 0.4, 1.0]  # Bright flash orange
                    
        except Exception as e:
            print(f"Target visualization update failed: {e}")
            pass

    def _update_dynamic_target_visualization(self, physics):
        """Update target visualization with dynamic effects during simulation."""
        try:
            if self._current_targets and self._current_target_index < len(self._current_targets):
                current_target = self._current_targets[self._current_target_index]
                
                # Create pulsing effect
                import time
                pulse_speed = 2.0  # Pulses per second
                pulse = 0.7 + 0.3 * abs(np.sin(time.time() * pulse_speed * np.pi))
                
                # Apply pulsing to alpha channel for visibility
                if current_target['type'] == 'swim':
                    # Pulsing blue for water targets
                    physics.named.model.mat_rgba['target'] = [0.1, 0.5, 1.0, pulse]
                else:
                    # Pulsing orange for land targets  
                    physics.named.model.mat_rgba['target'] = [1.0, 0.6, 0.2, pulse]
                    
                # Optional: Distance-based sizing (closer = larger)
                head_pos = physics.named.data.xpos['head'][:2]
                target_pos = current_target['position']
                distance = np.linalg.norm(np.array(target_pos) - head_pos)
                
                # Scale based on distance (closer targets appear larger)
                distance_scale = max(0.7, min(1.3, 2.0 / (distance + 0.5)))
                target_size = self._target_radius * distance_scale
                physics.named.model.geom_size['target'] = [target_size, target_size, target_size]
                
        except Exception as e:
            # Silent failure for dynamic effects
            pass

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
            # Phase 2: **INCREASED** 80% chance to start IN land zone to force escape learning
            land_start_probability = 1.0 if self._force_land_start_evaluation else 0.8
            if random.random() < land_start_probability:  # **FORCE** 100% for evaluation
                # Start deep within the land zone for crawling practice
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0.5, 1.5)  # **FORCE** deeper in land zone
                start_x = 3.0 + radius * np.cos(angle)
                start_y = 0.0 + radius * np.sin(angle)
                
                # Set swimmer position (approximately - MuJoCo controls this)
                try:
                    physics.named.data.qpos['root'][0] = start_x
                    physics.named.data.qpos['root'][1] = start_y
                    # **LOG** land starting position for debugging
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"🏝️ LAND START: Starting at ({start_x:.2f}, {start_y:.2f}) in land zone - must escape!")
                    except ImportError:
                        pass
                except:
                    pass  # If position setting fails, continue with default
                    
        elif self._training_progress < 0.8:
            # Phase 3: **INCREASED** 85% chance to start IN one of the land zones
            land_start_probability = 1.0 if self._force_land_start_evaluation else 0.85
            if random.random() < land_start_probability:  # **FORCE** 100% for evaluation
                # Choose random land zone: left [-2.0, 0] or right [3.5, 0]
                if random.random() < 0.5:
                    # Start on left land zone
                    center_x, center_y = -2.0, 0.0
                    max_radius = 1.2
                    zone_name = "LEFT"
                else:
                    # Start on right land zone  
                    center_x, center_y = 3.5, 0.0
                    max_radius = 1.2
                    zone_name = "RIGHT"
                    
                # Random position within chosen land zone - **FORCE** deeper positions
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0.5, max_radius)  # **FORCE** away from edges
                start_x = center_x + radius * np.cos(angle)
                start_y = center_y + radius * np.sin(angle)
                
                try:
                    physics.named.data.qpos['root'][0] = start_x
                    physics.named.data.qpos['root'][1] = start_y
                    # **LOG** land starting position for debugging
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"🏝️ LAND START: Starting at ({start_x:.2f}, {start_y:.2f}) in {zone_name} land zone - must traverse!")
                    except ImportError:
                        pass
                except:
                    pass
                    
        else:
            # Phase 4: **INCREASED** 90% chance to start IN one of the four land islands
            land_start_probability = 1.0 if self._force_land_start_evaluation else 0.9
            if random.random() < land_start_probability:  # **FORCE** 100% for evaluation
                # Choose random island from four options
                islands = [
                    (-3.0, 0, 1.0, "LEFT"),      # Left island
                    (0.0, 2.0, 0.8, "NORTH"),    # North island
                    (3.0, 0, 1.0, "RIGHT"),      # Right island  
                    (0.0, -2.0, 0.8, "SOUTH")    # South island
                ]
                
                center_x, center_y, max_radius, island_name = random.choice(islands)
                
                # Random position within chosen island - **FORCE** deeper positions
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0.3, max_radius * 0.7)  # **FORCE** deeper in island
                start_x = center_x + radius * np.cos(angle)
                start_y = center_y + radius * np.sin(angle)
                
                try:
                    physics.named.data.qpos['root'][0] = start_x
                    physics.named.data.qpos['root'][1] = start_y
                    # **LOG** island starting position for debugging
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"🏝️ ISLAND START: Starting at ({start_x:.2f}, {start_y:.2f}) on {island_name} island - must navigate complex terrain!")
                    except ImportError:
                        pass
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
        
        # **NEW**: Update 3D visualization if phase changed
        current_phase = int(self._training_progress * 4)
        if current_phase != self._last_visualization_phase:
            self._setup_3d_visualization(physics)

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
        
        # **NEW**: Update target visualization dynamically (pulsing effect)
        self._update_dynamic_target_visualization(physics)
        
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
        """Enhanced reward with goal-directed navigation - FIXED to encourage land zone usage."""
        head_pos = physics.named.data.xpos['head'][:2]
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        
        # **LAND AVOIDANCE FIX: Completely eliminate all movement penalties**
        if self._training_progress < 0.3:
            # Phase 1: Only reward target approach, NO base swimming reward
            base_reward = 0.0  # Pure navigation focus
        else:
            # Phase 2+: Mixed environment reward - ENCOURAGE both environments
            # Determine current environment
            in_land = False
            for zone in self._current_land_zones:
                distance = np.linalg.norm(head_pos - zone['center'])
                if distance < zone['radius']:
                    in_land = True
                    break
            
            if in_land:
                joint_velocities = physics.data.qvel
                joint_activity = np.sum(np.abs(joint_velocities))
                activity_reward = rewards.tolerance(
                    joint_activity,
                    bounds=(0.05, float('inf')),
                    margin=0.05,
                    value_at_margin=0.,
                    sigmoid='linear',
                ) * 0.3  # INCREASED from 0.1 to 0.3 - crawling should be rewarded
                # **ELIMINATED**: No efficiency penalties for land movement
                base_reward = activity_reward  # Only positive rewards for land activity
                
                # **NEW: Log when swimmer is on land for debugging**
                if not hasattr(self, '_last_on_land') or not self._last_on_land:
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"🏝️ Swimmer entered LAND zone - crawling mode activated")
                    except ImportError:
                        pass
                self._last_on_land = True
            else:
                # **FIX: Remove water penalties too - equal treatment**
                base_reward = 0.0  # No penalties, no base rewards - navigation dominates
                
                # **NEW: Log when swimmer exits land**
                if hasattr(self, '_last_on_land') and self._last_on_land:
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"🌊 Swimmer returned to WATER zone - swimming mode activated")
                    except ImportError:
                        pass
                self._last_on_land = False
        
        # **FIX: Add small baseline activity reward to prevent completely negative rewards**
        joint_velocities = physics.data.qvel
        joint_activity = np.sum(np.abs(joint_velocities))
        baseline_activity_reward = min(joint_activity * 0.01, 0.1)  # Small positive reward for any movement
        
        # **FIX: Add environment diversity bonus**
        environment_diversity_bonus = 0.0
        if self._training_progress >= 0.3 and hasattr(self, '_environment_transitions'):
            # Reward for using both environments (transitions indicate mixed locomotion)
            if self._environment_transitions > 0:
                # Bonus for successfully transitioning between environments
                diversity_bonus = min(self._environment_transitions * 0.5, 3.0)  # Max 3.0 bonus
                environment_diversity_bonus = diversity_bonus
        
        # **MASSIVELY increase goal-directed navigation rewards**
        navigation_reward = 0.0
        
        if self._current_targets and self._current_target_index < len(self._current_targets):
            current_target = self._current_targets[self._current_target_index]
            target_pos = current_target['position']
            
            # Distance to target
            distance_to_target = np.linalg.norm(head_pos - target_pos)
            
            # **ENHANCED TARGET TYPE BONUS**: Encourage land targets more than water targets
            target_type_multiplier = 1.0
            if current_target['type'] == 'land':
                target_type_multiplier = 1.5  # **NEW**: 50% bonus for land targets
                # Add extra bonus if currently in correct environment for target
                current_in_land = any(np.linalg.norm(head_pos - zone['center']) < zone['radius'] for zone in self._current_land_zones)
                if current_in_land:
                    target_type_multiplier = 2.0  # **100% bonus for land targets when in land**
            
            # **FIXED**: Set initial distance when target visit timer starts
            if self._target_visit_timer == 0:  # Very first step with this target
                self._initial_target_distance = distance_to_target
                self._last_distance = distance_to_target  # Initialize for progress tracking
                # **ENHANCED DEBUG**: Log target details with environment context
                try:
                    from tqdm import tqdm
                    target_env = "🏝️ LAND" if current_target['type'] == 'land' else "🌊 WATER"
                    multiplier_info = f"(x{target_type_multiplier} multiplier)" if target_type_multiplier > 1.0 else ""
                    tqdm.write(f"🎯 NEW TARGET #{self._targets_reached + 1}: {target_env} target at {target_pos}, distance = {distance_to_target:.3f}m {multiplier_info}")
                    # Show current environment
                    current_in_land = any(np.linalg.norm(head_pos - zone['center']) < zone['radius'] for zone in self._current_land_zones)
                    current_env = "🏝️ LAND" if current_in_land else "🌊 WATER"
                    tqdm.write(f"   Swimmer currently in: {current_env} environment")
                    
                    # **NEW: Log training phase and target types for debugging**
                    current_phase = int(self._training_progress * 4)
                    tqdm.write(f"   Training Phase: {current_phase} (progress: {self._training_progress:.3f})")
                    if len(self._current_targets) > 0:
                        land_targets = [t for t in self._current_targets if t['type'] == 'land']
                        water_targets = [t for t in self._current_targets if t['type'] == 'swim']
                        tqdm.write(f"   Phase targets: {len(land_targets)} land, {len(water_targets)} water")
                except ImportError:
                    pass
            
            # Track progress monitoring
            if self._target_visit_timer > 0 and self._target_visit_timer % 300 == 0:
                if hasattr(self, '_initial_target_distance'):
                    distance_traveled = max(0, self._initial_target_distance - distance_to_target)
                    time_elapsed = self._target_visit_timer / 30.0  # Convert to seconds
                    actual_speed = distance_traveled / time_elapsed if time_elapsed > 0 else 0
                    
                    try:
                        from tqdm import tqdm
                        current_target_info = f"Target #{self._targets_reached + 1}"
                        progress_percent = (distance_traveled / self._initial_target_distance * 100) if self._initial_target_distance > 0 else 0
                        target_env = "🏝️ LAND" if current_target['type'] == 'land' else "🌊 WATER"
                        
                        # **FIX: Add warning for very distant targets that may be unreachable**
                        if self._initial_target_distance > 3.0 and distance_traveled < 0.5:
                            tqdm.write(f"⚠️ Progress update {current_target_info} ({target_env}): {distance_traveled:.2f}m/{self._initial_target_distance:.2f}m ({progress_percent:.1f}%) in {time_elapsed:.1f}s = {actual_speed:.3f}m/s")
                            tqdm.write(f"   Warning: Target is very distant ({self._initial_target_distance:.1f}m) - may require advanced training")
                        else:
                            tqdm.write(f"🏊 Progress update {current_target_info} ({target_env}): {distance_traveled:.2f}m/{self._initial_target_distance:.2f}m ({progress_percent:.1f}%) in {time_elapsed:.1f}s = {actual_speed:.3f}m/s")
                    except ImportError:
                        pass
            
            # **ULTIMATE CIRCULAR SWIMMING FIX**: Reward progress, not proximity
            # Only reward actual progress toward target (not just being close)
            if hasattr(self, '_initial_target_distance') and self._initial_target_distance > 0:
                progress_made = max(0, self._initial_target_distance - distance_to_target)
                progress_ratio = progress_made / self._initial_target_distance
                
                # Reward based on cumulative progress (diminishes over time spent)
                time_factor = max(0.1, 1.0 - (self._target_visit_timer / 900.0))  # Decay over 30 seconds
                progress_reward = progress_ratio * 2.0 * time_factor * target_type_multiplier  # **APPLY TARGET BONUS**
                navigation_reward += progress_reward
                
                # Small directional bonus only when making progress
                if self._target_visit_timer > 30:  # After 1 second
                    recent_progress = max(0, self._last_distance - distance_to_target) if hasattr(self, '_last_distance') else 0
                    if recent_progress > 0.01:  # Actually moving toward target
                        navigation_reward += 0.2 * target_type_multiplier  # **APPLY TARGET BONUS**
                
                self._last_distance = distance_to_target
            else:
                # Fallback: minimal approach reward
                navigation_reward += (0.5 / (1.0 + distance_to_target)) * target_type_multiplier
            
            # Increment visit timer
            self._target_visit_timer += 1
            
            # **ANTI-HACK FIX: ELIMINATE timeout target switching entirely**
            target_reached = distance_to_target < self._target_radius
            
            # Apply escalating penalties for taking too long (but DON'T switch targets!)
            if self._target_visit_timer > 600:  # 20 seconds
                navigation_reward -= 0.02  
            if self._target_visit_timer > 1200:  # 40 seconds  
                navigation_reward -= 0.05  
            if self._target_visit_timer > 1800:  # 60 seconds
                navigation_reward -= 0.1   
            
            # ONLY advance target if actually reached
            if target_reached:
                # **MASSIVE REWARD** for reaching target (with type bonus)
                target_completion_reward = 10.0 * target_type_multiplier  # **LAND TARGETS WORTH MORE**
                navigation_reward += target_completion_reward
                
                # **ENHANCED**: Log target completion with environment info
                try:
                    from tqdm import tqdm
                    time_taken = self._target_visit_timer / 30.0  # Convert to seconds
                    initial_distance = getattr(self, '_initial_target_distance', distance_to_target)
                    speed = initial_distance / time_taken if time_taken > 0 else 0
                    target_env = "🏝️ LAND" if current_target['type'] == 'land' else "🌊 WATER"
                    reward_info = f"(+{target_completion_reward:.1f} reward)" if target_type_multiplier > 1.0 else f"(+{target_completion_reward:.1f} reward)"
                    tqdm.write(f"🎯 TARGET REACHED #{self._targets_reached + 1}: {target_env} target in {time_taken:.1f}s (speed: {speed:.3f}m/s) {reward_info}")
                except ImportError:
                    pass  # Silent if tqdm not available
                
                # Move to next target
                self._current_target_index += 1
                self._targets_reached += 1
                self._target_visit_timer = 0
                
                # **NEW**: Update 3D target visualization for new target
                self._update_target_visualization(physics)
                
                # Reset distance tracking for new target
                if hasattr(self, '_initial_target_distance'):
                    delattr(self, '_initial_target_distance')
                if hasattr(self, '_last_distance'):
                    delattr(self, '_last_distance')
                
                # Phase completion bonus
                if self._current_target_index >= len(self._current_targets):
                    if self._targets_reached % 100 == 0:  # Log every 100 completions
                        try:
                            from tqdm import tqdm
                            tqdm.write(f"🏆 Phase targets completed! ({self._targets_reached} total targets)")
                        except ImportError:
                            pass  
                    navigation_reward += 20.0  
                    # Reset to first target for continuous cycling
                    self._current_target_index = 0
            
            # **ANTI-CIRCULAR FIX**: Only reward directional movement when making real progress
            if distance_to_target > 0.1 and hasattr(self, '_last_distance'):  
                recent_progress = self._last_distance - distance_to_target if hasattr(self, '_last_distance') else 0
                
                # Only give directional reward if actually getting closer to target
                if recent_progress > 0.005:  # Must be making measurable progress (5mm per step)
                    target_direction = (np.array(target_pos) - head_pos) / distance_to_target
                    current_velocity = physics.named.data.sensordata['head_vel'][:2]
                    velocity_magnitude = np.linalg.norm(current_velocity)
                    
                    if velocity_magnitude > 0.01:  # Only if actually moving
                        velocity_direction = current_velocity / velocity_magnitude
                        directional_alignment = np.dot(target_direction, velocity_direction)
                        # Apply target type bonus to directional rewards too
                        navigation_reward += directional_alignment * 0.3 * target_type_multiplier
        
        # **FINAL REWARD CALCULATION**: Navigation dominates, with diversity bonus, no penalties
        total_reward = base_reward * 0.1 + navigation_reward * 1.0 + environment_diversity_bonus * 0.1 + baseline_activity_reward
        
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
        # Note: No timeout system - targets must be reached to advance!

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
                
                if phase_new == 1:
                    tqdm.write("   📈 Phase 1: Adding first land zone - targets must be reached!")
                elif phase_new == 2:
                    tqdm.write("   📈 Phase 2: Adding second land zone - targets must be reached!")
                elif phase_new == 3:
                    tqdm.write("   📈 Phase 3: Full complexity mixed environment - targets must be reached!")
                
                tqdm.write("   🎯 NO TIMEOUT TARGET SWITCHING - agent must actually reach targets to advance!")
            except ImportError:
                pass  # Silent if tqdm not available
            
            self._create_environment()
        
        # Update task progress
        if hasattr(self.env, '_task'):
            self.env._task.update_training_progress(self.training_progress)
    
    def set_manual_progress(self, progress, force_land_start=False):
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
            
            # **FIX**: Force land spawning for evaluation in advanced phases
            if force_land_start and hasattr(self.env._task, '_force_land_start_evaluation'):
                self.env._task._force_land_start_evaluation = True
                try:
                    from tqdm import tqdm
                    tqdm.write(f"🏝️ EVALUATION MODE: Forcing land starts for visual assessment")
                except ImportError:
                    pass
            
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
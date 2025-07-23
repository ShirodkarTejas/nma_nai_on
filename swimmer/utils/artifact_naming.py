#!/usr/bin/env python3
"""
Artifact Naming Utility for Swimmer Training
Provides consistent naming for models, videos, plots, and other artifacts 
to prevent overwriting and enable easy comparison between different versions.
"""

import os
from datetime import datetime
from typing import Dict, Optional, Any

class ArtifactNamer:
    """
    Centralized naming utility for all training artifacts.
    
    Generates consistent, descriptive names that include:
    - Model type (ncap, biological_ncap, enhanced_ncap, mlp)
    - Configuration details (n_links, algorithm, etc.)
    - Training state (step number, phase, etc.)
    - Timestamps when needed
    """
    
    def __init__(self, model_type: str, n_links: int, algorithm: str = "ppo", 
                 additional_config: Optional[Dict[str, Any]] = None):
        """
        Initialize artifact namer with base configuration.
        
        Args:
            model_type: Type of model (ncap, biological_ncap, enhanced_ncap, mlp)
            n_links: Number of links in swimmer
            algorithm: RL algorithm used
            additional_config: Additional configuration parameters to include in names
        """
        self.model_type = model_type
        self.n_links = n_links
        self.algorithm = algorithm
        self.additional_config = additional_config or {}
        
        # Generate base identifier
        self.base_id = self._generate_base_id()
        
    def _generate_base_id(self) -> str:
        """Generate base identifier from core configuration."""
        base = f"{self.model_type}_{self.algorithm}_{self.n_links}links"
        
        # Add important additional config items
        if self.additional_config:
            for key, value in self.additional_config.items():
                if key in ['oscillator_period', 'training_mode', 'curriculum_type']:
                    base += f"_{key}{value}"
        
        return base
    
    def checkpoint_name(self, step: int, base_dir: str = "outputs/training/checkpoints", 
                       use_model_subfolder: bool = True) -> str:
        """Generate checkpoint filename with optional model-specific subfolder."""
        if use_model_subfolder:
            model_dir = os.path.join(base_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            return os.path.join(model_dir, f"{self.base_id}_checkpoint_step_{step}.pt")
        else:
            os.makedirs(base_dir, exist_ok=True)
            return os.path.join(base_dir, f"{self.base_id}_checkpoint_step_{step}.pt")
    
    def final_model_name(self, base_dir: str = "outputs/training/models", 
                        use_model_subfolder: bool = True) -> str:
        """Generate final model filename with optional model-specific subfolder."""
        if use_model_subfolder:
            model_dir = os.path.join(base_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            return os.path.join(model_dir, f"{self.base_id}_final_model.pt")
        else:
            os.makedirs(base_dir, exist_ok=True)
            return os.path.join(base_dir, f"{self.base_id}_final_model.pt")
    
    def evaluation_video_name(self, step: Optional[int] = None, 
                            evaluation_type: str = "mixed_env",
                            base_dir: str = "outputs/evaluation/videos",
                            use_model_subfolder: bool = True) -> str:
        """Generate evaluation video filename with optional model-specific subfolder."""
        if use_model_subfolder:
            model_dir = os.path.join(base_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            target_dir = model_dir
        else:
            os.makedirs(base_dir, exist_ok=True)
            target_dir = base_dir
        
        if step is not None:
            filename = f"{self.base_id}_eval_{evaluation_type}_step_{step}.mp4"
        else:
            filename = f"{self.base_id}_eval_{evaluation_type}_final.mp4"
        
        return os.path.join(target_dir, filename)
    
    def training_video_name(self, step: int, phase: Optional[str] = None,
                          base_dir: str = "outputs/training/videos",
                          use_model_subfolder: bool = True) -> str:
        """Generate training progress video filename with optional model-specific subfolder."""
        if use_model_subfolder:
            model_dir = os.path.join(base_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            target_dir = model_dir
        else:
            os.makedirs(base_dir, exist_ok=True)
            target_dir = base_dir
        
        if phase:
            filename = f"{self.base_id}_training_{phase}_step_{step}.mp4"
        else:
            filename = f"{self.base_id}_training_step_{step}.mp4"
        
        return os.path.join(target_dir, filename)
    
    def analysis_plot_name(self, analysis_type: str, step: Optional[int] = None,
                         phase: Optional[str] = None,
                         base_dir: str = "outputs/analysis/plots",
                         use_model_subfolder: bool = True) -> str:
        """Generate analysis plot filename with optional model-specific subfolder."""
        if use_model_subfolder:
            model_dir = os.path.join(base_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            target_dir = model_dir
        else:
            os.makedirs(base_dir, exist_ok=True)
            target_dir = base_dir
        
        filename_parts = [self.base_id, analysis_type]
        
        if phase:
            filename_parts.append(f"phase_{phase}")
        if step is not None:
            filename_parts.append(f"step_{step}")
        
        filename = "_".join(filename_parts) + ".png"
        return os.path.join(target_dir, filename)
    
    def training_log_dir(self, base_dir: str = "outputs/training/logs",
                        use_model_subfolder: bool = True) -> str:
        """Generate training log directory with optional model-specific subfolder."""
        if use_model_subfolder:
            model_dir = os.path.join(base_dir, self.model_type)
            log_dir = os.path.join(model_dir, self.base_id)
        else:
            log_dir = os.path.join(base_dir, self.base_id)
        
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def comparison_name(self, other_model_type: str, comparison_type: str = "performance",
                       base_dir: str = "outputs/comparisons") -> str:
        """Generate comparison artifact name between models."""
        os.makedirs(base_dir, exist_ok=True)
        
        # Sort model types for consistent naming
        models = sorted([self.model_type, other_model_type])
        filename = f"comparison_{models[0]}_vs_{models[1]}_{comparison_type}_{self.n_links}links.png"
        
        return os.path.join(base_dir, filename)
    
    def curriculum_name(self, curriculum_phase: int, artifact_type: str = "checkpoint",
                       step: Optional[int] = None,
                       base_dir: str = "outputs/curriculum") -> str:
        """Generate curriculum training artifact name."""
        phase_names = ["swimming", "single_land", "two_lands", "full_complexity"]
        phase_name = phase_names[min(curriculum_phase, len(phase_names) - 1)]
        
        subdir = os.path.join(base_dir, artifact_type + "s")
        os.makedirs(subdir, exist_ok=True)
        
        if artifact_type == "checkpoint" and step is not None:
            filename = f"{self.base_id}_curriculum_{phase_name}_step_{step}.pt"
        elif artifact_type == "video":
            filename = f"{self.base_id}_curriculum_{phase_name}.mp4"
        elif artifact_type == "plot":
            filename = f"{self.base_id}_curriculum_{phase_name}.png"
        else:
            filename = f"{self.base_id}_curriculum_{phase_name}_{artifact_type}"
        
        return os.path.join(subdir, filename)
    
    def timestamped_name(self, base_name: str, extension: str = "",
                        base_dir: str = "outputs/timestamped") -> str:
        """Generate timestamped filename for unique artifacts."""
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.base_id}_{base_name}_{timestamp}"
        
        if extension:
            filename += f".{extension}"
        
        return os.path.join(base_dir, filename)
    
    def experiment_summary_name(self, base_dir: str = "outputs/summaries",
                               use_model_subfolder: bool = True) -> str:
        """Generate experiment summary filename with optional model-specific subfolder."""
        if use_model_subfolder:
            model_dir = os.path.join(base_dir, self.model_type)
            os.makedirs(model_dir, exist_ok=True)
            return os.path.join(model_dir, f"{self.base_id}_experiment_summary.md")
        else:
            os.makedirs(base_dir, exist_ok=True)
            return os.path.join(base_dir, f"{self.base_id}_experiment_summary.md")
    
    def create_organized_folder_structure(self, base_output_dir: str = "outputs") -> Dict[str, str]:
        """Create organized folder structure for model artifacts and return paths."""
        model_base = os.path.join(base_output_dir, self.model_type)
        
        folders = {
            'checkpoints': os.path.join(model_base, 'checkpoints'),
            'models': os.path.join(model_base, 'models'),
            'videos': os.path.join(model_base, 'videos'),
            'plots': os.path.join(model_base, 'plots'),
            'logs': os.path.join(model_base, 'logs'),
            'summaries': os.path.join(model_base, 'summaries'),
            'evaluations': os.path.join(model_base, 'evaluations'),
            'comparisons': os.path.join(base_output_dir, 'comparisons'),  # Shared for cross-model comparisons
        }
        
        # Create all directories
        for folder_path in folders.values():
            os.makedirs(folder_path, exist_ok=True)
        
        return folders
    
    def get_organized_path(self, artifact_type: str, **kwargs) -> str:
        """Get organized path for artifact with model-specific subfolders enabled by default."""
        kwargs.setdefault('use_model_subfolder', True)
        
        if artifact_type == 'checkpoint':
            return self.checkpoint_name(**kwargs)
        elif artifact_type == 'final_model':
            return self.final_model_name(**kwargs)
        elif artifact_type == 'evaluation_video':
            return self.evaluation_video_name(**kwargs)
        elif artifact_type == 'training_video':
            return self.training_video_name(**kwargs)
        elif artifact_type == 'analysis_plot':
            return self.analysis_plot_name(**kwargs)
        elif artifact_type == 'training_log_dir':
            return self.training_log_dir(**kwargs)
        elif artifact_type == 'experiment_summary':
            return self.experiment_summary_name(**kwargs)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

# Convenience factory functions for common model types
def create_ncap_namer(n_links: int, algorithm: str = "ppo", **kwargs) -> ArtifactNamer:
    """Create namer for standard NCAP models."""
    return ArtifactNamer("ncap", n_links, algorithm, kwargs)

def create_biological_ncap_namer(n_links: int, algorithm: str = "ppo", **kwargs) -> ArtifactNamer:
    """Create namer for biological NCAP models."""
    return ArtifactNamer("biological_ncap", n_links, algorithm, kwargs)

def create_enhanced_ncap_namer(n_links: int, algorithm: str = "ppo", **kwargs) -> ArtifactNamer:
    """Create namer for enhanced biological NCAP models."""
    return ArtifactNamer("enhanced_ncap", n_links, algorithm, kwargs)

def create_curriculum_namer(n_links: int, algorithm: str = "ppo", **kwargs) -> ArtifactNamer:
    """Create namer for curriculum training."""
    kwargs['training_mode'] = 'curriculum'
    return ArtifactNamer("curriculum_ncap", n_links, algorithm, kwargs)

def create_mlp_namer(n_links: int, algorithm: str = "ppo", **kwargs) -> ArtifactNamer:
    """Create namer for MLP baseline models."""
    return ArtifactNamer("mlp", n_links, algorithm, kwargs)

# Model type detection utility
def detect_model_type(model_or_class) -> str:
    """Detect model type from model instance or class."""
    if hasattr(model_or_class, '__class__'):
        class_name = model_or_class.__class__.__name__
    else:
        class_name = str(model_or_class)
    
    if "Enhanced" in class_name and "NCAP" in class_name:
        return "enhanced_ncap"
    elif "Biological" in class_name and "NCAP" in class_name:
        return "biological_ncap"
    elif "NCAP" in class_name:
        return "ncap"
    elif "MLP" in class_name:
        return "mlp"
    else:
        return "unknown"

# Example usage and testing
if __name__ == "__main__":
    print("🏷️ Artifact Naming Utility Test")
    print("=" * 50)
    
    # Test different model types
    test_configs = [
        ("ncap", 5, "ppo", {}),
        ("biological_ncap", 6, "a2c", {"oscillator_period": 60}),
        ("enhanced_ncap", 5, "ppo", {"training_mode": "goal_directed"}),
        ("curriculum_ncap", 6, "ppo", {"curriculum_type": "progressive"}),
    ]
    
    for model_type, n_links, algorithm, config in test_configs:
        print(f"\n📋 {model_type} with {n_links} links:")
        namer = ArtifactNamer(model_type, n_links, algorithm, config)
        
        print(f"   Checkpoint: {os.path.basename(namer.checkpoint_name(50000))}")
        print(f"   Final model: {os.path.basename(namer.final_model_name())}")
        print(f"   Eval video: {os.path.basename(namer.evaluation_video_name(step=100000))}")
        print(f"   Training plot: {os.path.basename(namer.analysis_plot_name('trajectory', step=75000))}")
        print(f"   Log directory: {os.path.basename(namer.training_log_dir())}")
    
    print(f"\n🗂️ Testing organized folder structure:")
    enhanced_namer = ArtifactNamer("enhanced_ncap", 5, "ppo")
    folders = enhanced_namer.create_organized_folder_structure()
    print(f"   Model folders created under: outputs/{enhanced_namer.model_type}/")
    for folder_type, path in folders.items():
        print(f"     {folder_type}: {os.path.basename(path)}")
    
    print(f"\n📁 Folder organization examples:")
    bio_namer = ArtifactNamer("biological_ncap", 5, "ppo")
    enh_namer = ArtifactNamer("enhanced_ncap", 5, "ppo")
    
    print(f"   Biological NCAP checkpoint: {bio_namer.get_organized_path('checkpoint', step=50000)}")
    print(f"   Enhanced NCAP checkpoint:   {enh_namer.get_organized_path('checkpoint', step=50000)}")
    print(f"   → Models are organized in separate subfolders!")
    
    print(f"\n✅ Enhanced Artifact naming utility ready!")
    print(f"   • Prevents overwriting between model types")
    print(f"   • Enables easy comparison of different versions")
    print(f"   • Provides organized folder structure with model subfolders")
    print(f"   • Supports curriculum and comparison workflows")
    print(f"   • Customizable folder organization (use_model_subfolder=True/False)") 
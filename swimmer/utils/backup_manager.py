#!/usr/bin/env python3
"""
Backup Manager for Long Training Runs
Provides automatic backup and recovery capabilities.
"""

import os
import shutil
import json
from datetime import datetime
import torch


class BackupManager:
    """
    Manages automatic backups and recovery for long training sessions.
    """
    
    def __init__(self, experiment_name, backup_dir='outputs/backups', max_backups=5):
        self.experiment_name = experiment_name
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        
        # Create backup directory
        self.experiment_backup_dir = os.path.join(backup_dir, experiment_name)
        os.makedirs(self.experiment_backup_dir, exist_ok=True)
        
        # Backup state tracking
        self.backup_history = []
        self.recovery_log_path = os.path.join(self.experiment_backup_dir, 'recovery_log.json')
        
        print(f"💾 Backup manager initialized for {experiment_name}")
        print(f"   Backup directory: {self.experiment_backup_dir}")
        print(f"   Max backups: {max_backups}")
    
    def create_backup(self, model, training_state, step):
        """Create a backup of the current training state."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_step_{step}_{timestamp}"
        backup_path = os.path.join(self.experiment_backup_dir, backup_name)
        
        try:
            os.makedirs(backup_path, exist_ok=True)
            
            # Save model state
            model_path = os.path.join(backup_path, 'model.pt')
            torch.save(model.state_dict(), model_path)
            
            # Save training state
            state_path = os.path.join(backup_path, 'training_state.json')
            with open(state_path, 'w') as f:
                json.dump(training_state, f, indent=2, default=str)
            
            # Save backup metadata
            metadata = {
                'backup_name': backup_name,
                'step': step,
                'timestamp': timestamp,
                'model_path': model_path,
                'state_path': state_path,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(backup_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update backup history
            self.backup_history.append(metadata)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            print(f"💾 Backup created: {backup_name} (step {step})")
            return backup_path
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """Remove old backups to maintain max_backups limit."""
        if len(self.backup_history) > self.max_backups:
            # Sort by step number
            sorted_backups = sorted(self.backup_history, key=lambda x: x['step'])
            
            # Remove oldest backups
            while len(sorted_backups) > self.max_backups:
                old_backup = sorted_backups.pop(0)
                old_backup_path = os.path.dirname(old_backup['model_path'])
                
                try:
                    if os.path.exists(old_backup_path):
                        shutil.rmtree(old_backup_path)
                        print(f"🗑️ Removed old backup: {old_backup['backup_name']}")
                except Exception as e:
                    print(f"⚠️ Failed to remove old backup: {e}")
            
            self.backup_history = sorted_backups
    
    def list_backups(self):
        """List available backups."""
        if not self.backup_history:
            print("No backups available")
            return []
        
        print("Available backups:")
        for backup in sorted(self.backup_history, key=lambda x: x['step'], reverse=True):
            print(f"  {backup['backup_name']} - Step {backup['step']} - {backup['timestamp']}")
        
        return self.backup_history
    
    def load_latest_backup(self):
        """Load the most recent backup."""
        if not self.backup_history:
            print("No backups available for recovery")
            return None, None
        
        latest_backup = max(self.backup_history, key=lambda x: x['step'])
        return self.load_backup(latest_backup['backup_name'])
    
    def load_backup(self, backup_name):
        """Load a specific backup."""
        backup_path = os.path.join(self.experiment_backup_dir, backup_name)
        
        if not os.path.exists(backup_path):
            print(f"❌ Backup not found: {backup_name}")
            return None, None
        
        try:
            # Load metadata
            metadata_path = os.path.join(backup_path, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load model state
            model_path = os.path.join(backup_path, 'model.pt')
            model_state = torch.load(model_path, map_location='cpu')
            
            # Load training state
            state_path = os.path.join(backup_path, 'training_state.json')
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            
            print(f"✅ Loaded backup: {backup_name} (step {metadata['step']})")
            
            # Log recovery
            self._log_recovery(backup_name, metadata['step'])
            
            return model_state, training_state
            
        except Exception as e:
            print(f"❌ Failed to load backup {backup_name}: {e}")
            return None, None
    
    def _log_recovery(self, backup_name, step):
        """Log recovery information."""
        recovery_entry = {
            'backup_name': backup_name,
            'step': step,
            'recovery_time': datetime.now().isoformat(),
            'action': 'recovery'
        }
        
        # Load existing recovery log
        recovery_log = []
        if os.path.exists(self.recovery_log_path):
            try:
                with open(self.recovery_log_path, 'r') as f:
                    recovery_log = json.load(f)
            except:
                pass
        
        recovery_log.append(recovery_entry)
        
        # Save updated log
        with open(self.recovery_log_path, 'w') as f:
            json.dump(recovery_log, f, indent=2)
    
    def save_error_log(self, error_info):
        """Save error information for debugging."""
        error_log_path = os.path.join(self.experiment_backup_dir, 'error_log.json')
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_info': str(error_info),
            'type': type(error_info).__name__
        }
        
        # Load existing errors
        error_log = []
        if os.path.exists(error_log_path):
            try:
                with open(error_log_path, 'r') as f:
                    error_log = json.load(f)
            except:
                pass
        
        error_log.append(error_entry)
        
        # Save updated log
        with open(error_log_path, 'w') as f:
            json.dump(error_log, f, indent=2)
        
        print(f"❌ Error logged: {error_entry['type']}")
    
    def get_recovery_recommendations(self):
        """Provide recovery recommendations based on backup history."""
        if not self.backup_history:
            return "No backups available. Consider starting training from scratch."
        
        latest_backup = max(self.backup_history, key=lambda x: x['step'])
        recommendations = f"""
💾 RECOVERY RECOMMENDATIONS

Latest Backup Available:
  - Name: {latest_backup['backup_name']}
  - Step: {latest_backup['step']:,}
  - Created: {latest_backup['timestamp']}

Recovery Options:
  1. Load latest backup and resume training
  2. Load specific backup if latest is corrupted
  3. Check error logs for failure patterns

Available backups: {len(self.backup_history)}
Total training progress preserved: {(latest_backup['step'] / 1000000) * 100:.1f}%
"""
        return recommendations 
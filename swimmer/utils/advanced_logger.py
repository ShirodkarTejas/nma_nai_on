#!/usr/bin/env python3
"""
Advanced Training Logger for Long Training Runs
Includes hardware monitoring, ETA estimation, and comprehensive health tracking.
"""

import os
import time
import json
import psutil
import platform
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import torch
import threading
from .training_logger import TrainingLogger


class AdvancedTrainingLogger(TrainingLogger):
    """
    Enhanced logger for comprehensive monitoring during long training runs.
    Includes hardware tracking, ETA estimation, and training health analysis.
    """
    
    def __init__(self, log_dir='outputs/training_logs', experiment_name=None):
        super().__init__(log_dir, experiment_name)
        
        # Hardware and system monitoring
        self.hardware_metrics = []
        self.start_time = None
        self.last_checkpoint_time = None
        
        # Performance tracking
        self.steps_per_sec_history = deque(maxlen=100)  # Keep last 100 measurements
        self.training_health = {
            'gradient_norms': [],
            'loss_stability': [],
            'parameter_changes': [],
            'learning_progress': []
        }
        
        # ETA and progress tracking
        self.eta_history = deque(maxlen=10)
        self.progress_checkpoints = []
        
        # System info logging
        self.log_system_info()
        
        # Background monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        print(f"🔬 Advanced training logger initialized")
        print(f"   System: {platform.system()} {platform.release()}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    def log_system_info(self):
        """Log detailed system information for reproducibility."""
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            system_info['gpu_name'] = torch.cuda.get_device_name()
            system_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        system_info_path = os.path.join(self.log_path, 'system_info.json')
        with open(system_info_path, 'w') as f:
            json.dump(system_info, f, indent=2)
    
    def start_monitoring(self):
        """Start background hardware monitoring."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_hardware, daemon=True)
        self.monitoring_thread.start()
        print("🖥️ Hardware monitoring started")
    
    def stop_monitoring(self):
        """Stop background hardware monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("🖥️ Hardware monitoring stopped")
    
    def _monitor_hardware(self):
        """Background thread for hardware monitoring."""
        while self.monitoring_active:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU metrics if available
                gpu_metrics = {}
                if torch.cuda.is_available():
                    gpu_metrics = {
                        'gpu_memory_used_gb': torch.cuda.memory_allocated() / (1024**3),
                        'gpu_memory_cached_gb': torch.cuda.memory_reserved() / (1024**3),
                        'gpu_utilization': self._get_gpu_utilization()
                    }
                
                hardware_data = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'disk_usage_gb': psutil.disk_usage('.').used / (1024**3),
                    **gpu_metrics
                }
                
                self.hardware_metrics.append(hardware_data)
                
                # Keep only last 1000 measurements to prevent memory bloat
                if len(self.hardware_metrics) > 1000:
                    self.hardware_metrics = self.hardware_metrics[-1000:]
                    
            except Exception as e:
                print(f"⚠️ Hardware monitoring error: {e}")
            
            time.sleep(60)  # Monitor every minute
    
    def _get_gpu_utilization(self):
        """Get GPU utilization percentage (simplified)."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    def log_training_health(self, model=None, loss=None, gradients=None):
        """Log training health metrics."""
        health_data = {
            'timestamp': time.time(),
            'step': self.current_step
        }
        
        # Loss stability
        if loss is not None:
            self.training_health['loss_stability'].append(loss)
            if len(self.training_health['loss_stability']) > 100:
                recent_losses = self.training_health['loss_stability'][-100:]
                loss_variance = np.var(recent_losses)
                health_data['loss_variance'] = loss_variance
        
        # Gradient norms
        if gradients is not None or model is not None:
            if model is not None:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                health_data['gradient_norm'] = total_norm
                self.training_health['gradient_norms'].append(total_norm)
        
        # Parameter change tracking
        if model is not None and hasattr(self, 'previous_params'):
            param_change = 0.0
            for (name, param), (prev_name, prev_param) in zip(model.named_parameters(), self.previous_params):
                if name == prev_name:
                    param_change += torch.norm(param - prev_param).item()
            health_data['parameter_change'] = param_change
            self.training_health['parameter_changes'].append(param_change)
        
        # Store current parameters for next comparison
        if model is not None:
            self.previous_params = [(name, param.clone()) for name, param in model.named_parameters()]
        
        return health_data
    
    def calculate_eta(self, current_step, total_steps):
        """Calculate estimated time of arrival."""
        if self.start_time is None:
            return "N/A"
        
        elapsed_time = time.time() - self.start_time
        if current_step == 0:
            return "N/A"
        
        steps_per_second = current_step / elapsed_time
        remaining_steps = total_steps - current_step
        
        if steps_per_second > 0:
            remaining_seconds = remaining_steps / steps_per_second
            eta = datetime.now() + timedelta(seconds=remaining_seconds)
            
            # Store ETA for trend analysis
            self.eta_history.append({
                'timestamp': time.time(),
                'eta': eta.isoformat(),
                'steps_per_sec': steps_per_second,
                'progress_percent': (current_step / total_steps) * 100
            })
            
            return eta.strftime("%Y-%m-%d %H:%M:%S")
        
        return "N/A"
    
    def log_checkpoint(self, step, model=None, performance_metrics=None):
        """Enhanced checkpoint logging with health analysis."""
        checkpoint_time = time.time()
        
        # Calculate performance since last checkpoint
        if self.last_checkpoint_time:
            time_since_last = checkpoint_time - self.last_checkpoint_time
            steps_since_last = step - getattr(self, 'last_checkpoint_step', 0)
            steps_per_sec = steps_since_last / time_since_last if time_since_last > 0 else 0
            self.steps_per_sec_history.append(steps_per_sec)
        
        checkpoint_data = {
            'step': step,
            'timestamp': checkpoint_time,
            'elapsed_hours': (checkpoint_time - self.start_time) / 3600 if self.start_time else 0,
            'steps_per_sec': np.mean(self.steps_per_sec_history) if self.steps_per_sec_history else 0,
            'performance_metrics': performance_metrics or {}
        }
        
        # Hardware snapshot
        if self.hardware_metrics:
            recent_hw = self.hardware_metrics[-10:]  # Last 10 minutes
            checkpoint_data['hardware_snapshot'] = {
                'avg_cpu_percent': np.mean([h['cpu_percent'] for h in recent_hw]),
                'avg_memory_percent': np.mean([h['memory_percent'] for h in recent_hw]),
                'current_disk_usage_gb': recent_hw[-1]['disk_usage_gb']
            }
            
            if torch.cuda.is_available() and 'gpu_memory_used_gb' in recent_hw[-1]:
                checkpoint_data['hardware_snapshot']['gpu_memory_used_gb'] = recent_hw[-1]['gpu_memory_used_gb']
        
        # Training health analysis
        if model is not None:
            health_data = self.log_training_health(model)
            checkpoint_data['training_health'] = health_data
        
        self.progress_checkpoints.append(checkpoint_data)
        self.last_checkpoint_time = checkpoint_time
        self.last_checkpoint_step = step
        
        # Save checkpoint data
        checkpoint_path = os.path.join(self.log_path, 'checkpoints.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(self.progress_checkpoints, f, indent=2)
        
        return checkpoint_data
    
    def create_training_dashboard(self):
        """Create a comprehensive training dashboard summary."""
        if not self.progress_checkpoints:
            return "No checkpoints available for dashboard."
        
        latest = self.progress_checkpoints[-1]
        
        dashboard = f"""
=== TRAINING DASHBOARD ===
Experiment: {self.experiment_name}
Current Step: {latest['step']:,}
Elapsed Time: {latest['elapsed_hours']:.2f} hours
Performance: {latest['steps_per_sec']:.1f} steps/sec

=== HARDWARE STATUS ===
"""
        
        if 'hardware_snapshot' in latest:
            hw = latest['hardware_snapshot']
            dashboard += f"CPU Usage: {hw.get('avg_cpu_percent', 0):.1f}%\n"
            dashboard += f"Memory Usage: {hw.get('avg_memory_percent', 0):.1f}%\n"
            dashboard += f"Disk Usage: {hw.get('current_disk_usage_gb', 0):.1f} GB\n"
            if 'gpu_memory_used_gb' in hw:
                dashboard += f"GPU Memory: {hw['gpu_memory_used_gb']:.1f} GB\n"
        
        dashboard += f"""
=== TRAINING HEALTH ===
"""
        
        if 'training_health' in latest:
            health = latest['training_health']
            if 'gradient_norm' in health:
                dashboard += f"Gradient Norm: {health['gradient_norm']:.4f}\n"
            if 'loss_variance' in health:
                dashboard += f"Loss Variance: {health['loss_variance']:.4f}\n"
            if 'parameter_change' in health:
                dashboard += f"Parameter Change: {health['parameter_change']:.4f}\n"
        
        dashboard += f"""
=== PERFORMANCE TRENDS ===
"""
        
        if len(self.progress_checkpoints) >= 2:
            recent_perf = [cp['steps_per_sec'] for cp in self.progress_checkpoints[-5:]]
            dashboard += f"Recent Performance: {np.mean(recent_perf):.1f} ± {np.std(recent_perf):.1f} steps/sec\n"
            
            if len(self.progress_checkpoints) >= 10:
                early_perf = np.mean([cp['steps_per_sec'] for cp in self.progress_checkpoints[:5]])
                trend = "improving" if np.mean(recent_perf) > early_perf else "declining"
                dashboard += f"Performance Trend: {trend}\n"
        
        return dashboard
    
    def save_comprehensive_report(self):
        """Save a comprehensive training report with all metrics."""
        # Base training summary
        base_summary = super().create_summary_report()
        
        # Additional advanced metrics
        advanced_report = f"""

=== ADVANCED METRICS ===

Hardware Utilization:
  - Peak CPU: {max([h['cpu_percent'] for h in self.hardware_metrics]) if self.hardware_metrics else 0:.1f}%
  - Peak Memory: {max([h['memory_percent'] for h in self.hardware_metrics]) if self.hardware_metrics else 0:.1f}%
  - Total Disk Used: {self.hardware_metrics[-1]['disk_usage_gb'] if self.hardware_metrics else 0:.1f} GB

Training Health:
  - Average Gradient Norm: {np.mean(self.training_health['gradient_norms']) if self.training_health['gradient_norms'] else 0:.4f}
  - Loss Stability (variance): {np.var(self.training_health['loss_stability']) if self.training_health['loss_stability'] else 0:.4f}
  - Parameter Change Rate: {np.mean(self.training_health['parameter_changes']) if self.training_health['parameter_changes'] else 0:.4f}

Performance Analysis:
  - Average Steps/Sec: {np.mean(self.steps_per_sec_history) if self.steps_per_sec_history else 0:.1f}
  - Performance Stability: {np.std(self.steps_per_sec_history) if self.steps_per_sec_history else 0:.1f}

ETA Accuracy:
  - ETA Predictions Made: {len(self.eta_history)}
  - Final ETA: {self.eta_history[-1]['eta'] if self.eta_history else 'N/A'}

=== RECOMMENDATIONS ===
"""
        
        # Analysis and recommendations
        if self.hardware_metrics:
            avg_cpu = np.mean([h['cpu_percent'] for h in self.hardware_metrics])
            avg_memory = np.mean([h['memory_percent'] for h in self.hardware_metrics])
            
            if avg_cpu > 90:
                advanced_report += "- Consider reducing batch size or complexity (high CPU usage)\n"
            if avg_memory > 85:
                advanced_report += "- Monitor memory usage - consider gradient checkpointing\n"
        
        if self.training_health['gradient_norms']:
            avg_grad_norm = np.mean(self.training_health['gradient_norms'])
            if avg_grad_norm > 1.0:
                advanced_report += "- Consider reducing learning rate (high gradient norms)\n"
            elif avg_grad_norm < 0.001:
                advanced_report += "- Consider increasing learning rate (very small gradients)\n"
        
        full_report = base_summary + advanced_report
        
        # Save advanced report
        advanced_path = os.path.join(self.log_path, 'advanced_report.txt')
        with open(advanced_path, 'w') as f:
            f.write(full_report)
        
        # Save hardware metrics
        hardware_path = os.path.join(self.log_path, 'hardware_metrics.json')
        with open(hardware_path, 'w') as f:
            json.dump(self.hardware_metrics, f, indent=2)
        
        print(f"📊 Comprehensive report saved to: {advanced_path}")
        return full_report 
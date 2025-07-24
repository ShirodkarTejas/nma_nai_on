# Comprehensive Training Monitoring Guide

## 🔬 **Advanced Logging & Monitoring for 1M Episode Training**

Our curriculum training system includes comprehensive monitoring and logging capabilities designed for long-duration training runs.

## 📊 **Monitoring Capabilities**

### **Real-Time Metrics**
- ✅ **Episode rewards, distances, lengths** by training phase
- ✅ **Training losses** (policy, value, entropy)
- ✅ **Learning rate** and optimizer state
- ✅ **Steps/second performance** tracking
- ✅ **ETA estimation** with trend analysis
- ✅ **Environment transition counts**
- ✅ **Biological parameter stability**

### **Hardware Monitoring** (Background Thread)
- 🖥️ **CPU utilization** (real-time)
- 💾 **Memory usage** (RAM & VRAM)
- 💽 **Disk space consumption**
- 🎮 **GPU utilization** (via nvidia-smi)
- 📈 **Performance trends** over time
- ⚡ **Training speed stability**

### **Training Health Analysis**
- 🧬 **Gradient norm tracking**
- 📉 **Loss variance/stability**
- 🔄 **Parameter change rates**
- 🧪 **NCAP biological parameter monitoring**
- ⚠️ **NaN detection & recovery**
- 🔧 **Constraint enforcement tracking**

## 📁 **Output Structure**

```
outputs/
├── curriculum_training/
│   ├── videos/                          # 🎬 Training videos
│   │   ├── curriculum_video_step_50000.mp4   # With minimap & zones
│   │   ├── curriculum_video_step_100000.mp4  # Every checkpoint
│   │   └── curriculum_final_video.mp4        # Phase comparison
│   ├── plots/                           # 📊 Analysis plots
│   │   ├── trajectory_analysis_step_50000.png # Research-quality
│   │   ├── final_trajectory_phase_0-3.png    # Per-phase analysis
│   │   └── curriculum_final_plots.png        # Training summary
│   ├── summaries/                       # 📄 Text reports
│   │   └── curriculum_training_summary.md    # Markdown summary
│   └── models/                          # 💾 Trained models
│       └── curriculum_final_model_5links.pt  # Final model
├── curriculum_checkpoints/              # 🏁 Model checkpoints
│   ├── step_50000.pt                    # Every 50k steps
│   └── step_100000.pt                   # With full state
├── training_logs/curriculum_ncap_5links/ # 📈 Detailed logs
│   ├── system_info.json                 # Reproducibility info
│   ├── hardware_metrics.json            # Hardware usage over time
│   ├── checkpoints.json                 # Performance snapshots
│   ├── advanced_report.txt              # Comprehensive analysis
│   ├── metrics.json                     # All training metrics
│   └── episodes.json                    # Episode-by-episode data
└── backups/curriculum_ncap_5links/      # 💾 Automatic backups
    ├── backup_step_50000_20240123_143022/
    ├── recovery_log.json                # Recovery tracking
    └── error_log.json                   # Error logging
```

## 🎯 **Key Features for Long Training**

### **1. ETA & Progress Tracking**
```
[  250000/1000000] Phase 2 | Episode  12550 | Reward:  45.23 | Distance: 2.847m | Steps/s: 42.3 | ETA: 2024-01-25 18:45:22
```

### **2. Training Dashboard** (Every Checkpoint)
```
=== TRAINING DASHBOARD ===
Experiment: curriculum_ncap_5links
Current Step: 250,000
Elapsed Time: 14.2 hours
Performance: 42.3 steps/sec

=== HARDWARE STATUS ===
CPU Usage: 87.2%
Memory Usage: 72.1%
Disk Usage: 45.3 GB
GPU Memory: 8.2 GB

=== TRAINING HEALTH ===
Gradient Norm: 0.0832
Loss Variance: 0.0156
Parameter Change: 0.0023

=== PERFORMANCE TRENDS ===
Recent Performance: 42.1 ± 1.8 steps/sec
Performance Trend: improving
```

### **3. Automatic Backup System**
- 💾 **Every checkpoint** creates backup
- 🗂️ **Rolling backup** (keeps 5 most recent)
- 📝 **Recovery logs** for failure analysis
- ⚡ **Quick recovery** from any backup point

### **4. Phase-Specific Analysis**
- 📊 **Performance by phase** (0-30%, 30-60%, 60-80%, 80-100%)
- 🎬 **Videos with minimap** showing environment transitions
- 📈 **Trajectory analysis** with zone visualization
- 🔄 **Automatic phase transition** detection

### **5. Biological Parameter Monitoring**
- 🧬 **NCAP parameter stability** tracking
- ⚠️ **Constraint violation** detection
- 🔧 **Automatic correction** of parameter drift
- 📊 **Oscillator strength** maintenance

## 🚀 **Starting Your 1M Episode Training**

### **Command**
```bash
python main.py --mode train_curriculum --training_steps 1000000
```

### **What You'll Get**
1. **Real-time console output** with ETA and performance metrics
2. **Background hardware monitoring** (every minute)
3. **Checkpoint analysis** every 50k steps with dashboard
4. **Research-quality visualizations** automatically generated
5. **Comprehensive final report** with hardware utilization analysis

### **Monitoring During Training**
- 📺 **Console logs** show progress, ETA, and health warnings
- 📁 **Output folders** populate with videos and plots
- 💾 **Automatic backups** ensure no progress loss
- 🔍 **Error logging** captures any issues for debugging

## 📋 **Recommended Setup**

### **Before Starting**
1. **Verify CUDA** availability: `torch.cuda.is_available()`
2. **Check disk space**: ~50GB recommended for full run
3. **Monitor GPU memory**: 8GB+ VRAM recommended
4. **Set up monitoring**: Advanced logging will auto-start

### **During Training** 
- 👀 **Check console** for ETA and performance trends
- 📊 **Review dashboards** at each checkpoint
- 🔍 **Monitor hardware** usage through logs
- 💾 **Verify backups** are being created

### **After Training**
- 📈 **Analyze final plots** in `outputs/curriculum_training/plots/`
- 🎬 **Review videos** showing learned behavior
- 📄 **Read summary report** for performance insights
- 💾 **Save final model** for future evaluation

## ⚠️ **Troubleshooting & Recovery**

### **If Training Stops**
1. **Check error logs**: `outputs/backups/*/error_log.json`
2. **Review latest backup**: Automatic recovery recommendations
3. **Hardware analysis**: Check hardware utilization patterns
4. **Resume training**: Load from latest checkpoint

### **Performance Issues**
- 📉 **Slow training**: Check CPU/GPU utilization
- 💾 **Memory issues**: Monitor VRAM usage trends
- 🔄 **Parameter drift**: Biological constraint logs
- 📊 **Poor performance**: Phase-specific analysis

## 🎯 **Expected Outputs**

After successful 1M episode training, you'll have:

- ✅ **20+ checkpoint videos** with minimap and zones
- ✅ **20+ trajectory analysis plots** (research-quality)
- ✅ **4 final phase videos** showing learned behavior
- ✅ **Comprehensive training report** with hardware analysis
- ✅ **Complete training logs** for reproducibility
- ✅ **Final trained model** ready for evaluation

## 📞 **Support**

If issues arise during your 1M episode training:
1. **Console output** shows real-time status and warnings
2. **Advanced report** provides detailed diagnostics
3. **Backup system** ensures recovery options
4. **Error logs** capture failure information for debugging

**Your training session will be comprehensively monitored and logged! 🚀** 
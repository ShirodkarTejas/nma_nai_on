#!/bin/bash
# setup_env.sh - Clean environment setup for Swimmer project

set -e

echo "🚀 Starting clean environment setup..."

# 1. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment (.venv) already exists."
fi

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install core dependencies
echo "📥 Installing core dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install Tonic RL library from source (required for this project)
if [ ! -d "tonic_repo" ]; then
    echo "📥 Cloning Tonic RL library..."
    git clone https://github.com/fabiopardo/tonic.git tonic_repo
    cd tonic_repo
    echo "⚙️ Installing Tonic in editable mode..."
    pip install -e .
    cd ..
else
    echo "✅ Tonic directory already exists. Skipping clone."
    cd tonic_repo
    pip install -e .
    cd ..
fi

# 5. Verify CUDA and Multi-GPU setup
echo "🔍 Verifying CUDA and GPU setup..."
python3 <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs detected: {device_count}")
    for i in range(device_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    if device_count < 2:
        print("⚠️ Warning: Only one GPU detected. This project is being optimized for 2x3090.")
else:
    print("❌ Error: CUDA not available. Training will be slow on CPU.")

import tonic
print("✅ Tonic successfully imported")
EOF

echo "✨ Setup complete! To start training, run: source .venv/bin/activate"

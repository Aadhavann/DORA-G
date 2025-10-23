#!/bin/bash
# Quick Start Script for RunPod Deployment
# Run this after SSH'ing into your RunPod instance

set -e

echo "======================================"
echo "DoRA-G RunPod Quick Setup"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Are you in the DoRA-G directory?"
    exit 1
fi

# Step 1: Update system packages
print_step "Updating system packages..."
apt-get update -qq
print_success "System packages updated"

# Step 2: Install PyTorch with CUDA 11.8
print_step "Installing PyTorch with CUDA 11.8..."
pip install -q torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
print_success "PyTorch installed"

# Step 3: Install faiss-gpu
print_step "Installing faiss-gpu..."
pip install -q faiss-gpu
print_success "faiss-gpu installed"

# Step 4: Install remaining dependencies
print_step "Installing project dependencies..."
grep -v "faiss-cpu" requirements.txt > requirements_gpu.txt
pip install -q -r requirements_gpu.txt
rm requirements_gpu.txt
print_success "Dependencies installed"

# Step 5: Set up environment variables
print_step "Setting up environment variables..."
export PYTHONPATH=/workspace:$PYTHONPATH
export PYTHONHASHSEED=0
export HF_HOME=/workspace/cache/huggingface
export TRANSFORMERS_CACHE=/workspace/cache/transformers
export TORCH_HOME=/workspace/cache/torch
export WANDB_DIR=/workspace/logs/wandb
export DATA_DIR=/workspace/data
export OUTPUT_DIR=/workspace/outputs
export CACHE_DIR=/workspace/cache
export CHECKPOINTS_DIR=/workspace/checkpoints
export LOGS_DIR=/workspace/logs
export MODEL_CACHE_DIR=/workspace/models_cache

# Create directories
mkdir -p /workspace/{data,outputs,cache,checkpoints,logs,models_cache}
print_success "Environment configured"

# Step 6: Check for W&B API key
print_step "Checking W&B configuration..."
if [ -z "$WANDB_API_KEY" ]; then
    print_error "WANDB_API_KEY not set!"
    echo ""
    echo "Please set your Weights & Biases API key:"
    echo "  1. Get your key from: https://wandb.ai/settings"
    echo "  2. Run: export WANDB_API_KEY=your_key_here"
    echo "  3. Or set it in RunPod environment variables"
    echo ""
    read -p "Enter your W&B API key now (or press Enter to continue without W&B): " wandb_key
    if [ -n "$wandb_key" ]; then
        export WANDB_API_KEY=$wandb_key
        wandb login --relogin $WANDB_API_KEY
        print_success "W&B configured"
    else
        export WANDB_MODE=offline
        print_step "Continuing without W&B (offline mode)"
    fi
else
    wandb login --relogin $WANDB_API_KEY
    print_success "W&B configured"
fi

# Step 7: Verify GPU
print_step "Verifying GPU availability..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
print_success "GPU verified"

# Step 8: Run setup verification
print_step "Running setup verification..."
python scripts/00_verify_setup.py
print_success "Setup verified"

echo ""
echo "======================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "======================================"
echo ""
echo "You can now run:"
echo "  1. Full pipeline:     ./runpod_entrypoint.sh"
echo "  2. Individual steps:  python scripts/01_prepare_datasets.py"
echo "  3. Quick test:        python scripts/03_train_baseline.py --config-name experiments/dora_only"
echo ""
echo "Monitor progress:"
echo "  - W&B Dashboard: https://wandb.ai"
echo "  - GPU usage:     watch -n 1 nvidia-smi"
echo "  - Logs:          tail -f /workspace/logs/*.log"
echo ""
echo "Expected total runtime: 6-8 hours on A100 80GB"
echo "Estimated cost: ~$9.50 (@ $1.19/hr Community Cloud)"
echo ""

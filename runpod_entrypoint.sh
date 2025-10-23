#!/bin/bash
# RunPod Entrypoint Script for DoRA-G Project
# This script sets up the environment and runs the complete training pipeline

set -e  # Exit on error

echo "======================================"
echo "DoRA-G RunPod Training Pipeline"
echo "======================================"
echo ""

# Function to print colored messages
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Set up environment variables
print_info "Setting up environment variables..."
export PYTHONPATH=/workspace:$PYTHONPATH
export PYTHONHASHSEED=0
export HF_HOME=/workspace/cache/huggingface
export TRANSFORMERS_CACHE=/workspace/cache/transformers
export TORCH_HOME=/workspace/cache/torch
export WANDB_DIR=/workspace/logs/wandb

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    print_error "WANDB_API_KEY not set!"
    print_info "Please set it in RunPod environment variables or run:"
    print_info "export WANDB_API_KEY=your_key_here"
    print_info "Continuing without W&B logging..."
    export WANDB_MODE=offline
else
    print_success "W&B API key found"
    wandb login --relogin $WANDB_API_KEY 2>/dev/null || true
fi

# Verify GPU availability
print_info "Verifying GPU availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"; then
    print_error "CUDA/GPU not detected! This project requires a GPU."
    exit 1
fi

print_success "GPU detected and ready"
echo ""

# Run setup verification
print_info "Running setup verification..."
cd /workspace
python scripts/00_verify_setup.py || {
    print_error "Setup verification failed. Please check dependencies."
    exit 1
}
print_success "Setup verification passed"
echo ""

# Prepare datasets
print_info "Preparing datasets..."
python scripts/01_prepare_datasets.py || {
    print_error "Dataset preparation failed."
    exit 1
}
print_success "Datasets prepared"
echo ""

# Build RAG index (only if not already exists)
if [ ! -d "/workspace/data/faiss_index" ] || [ -z "$(ls -A /workspace/data/faiss_index)" ]; then
    print_info "Building RAG index (this may take a while on first run)..."
    python scripts/02_build_rag_index.py || {
        print_error "RAG index building failed."
        exit 1
    }
    print_success "RAG index built successfully"
else
    print_info "RAG index already exists, skipping build..."
fi
echo ""

# Print GPU memory before training
print_info "GPU Memory Status:"
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'GPU {i}: {mem_alloc:.2f}GB / {mem_total:.2f}GB allocated')
"
echo ""

# Run baseline training
print_info "Starting baseline model training..."
python scripts/03_train_baseline.py || {
    print_error "Baseline training failed."
    exit 1
}
print_success "Baseline training completed"
echo ""

# Run LoRA training
print_info "Starting LoRA training..."
python scripts/03_train_baseline.py --config-name experiments/lora_only || {
    print_error "LoRA training failed."
    exit 1
}
print_success "LoRA training completed"
echo ""

# Run DoRA training
print_info "Starting DoRA training..."
python scripts/03_train_baseline.py --config-name experiments/dora_only || {
    print_error "DoRA training failed."
    exit 1
}
print_success "DoRA training completed"
echo ""

# Run DoRA+RAG training
print_info "Starting DoRA+RAG training..."
python scripts/03_train_baseline.py --config-name experiments/dora_rag || {
    print_error "DoRA+RAG training failed."
    exit 1
}
print_success "DoRA+RAG training completed"
echo ""

# Run all evaluations
print_info "Running evaluations on all benchmarks..."
python scripts/04_evaluate.py || {
    print_error "Evaluation failed."
    exit 1
}
print_success "Evaluations completed"
echo ""

# Generate analysis and plots
print_info "Generating analysis and visualizations..."
python scripts/05_analyze_results.py || {
    print_error "Analysis generation failed."
    exit 1
}
print_success "Analysis completed"
echo ""

# Final summary
echo "======================================"
print_success "PIPELINE COMPLETED SUCCESSFULLY!"
echo "======================================"
echo ""
print_info "Results saved to:"
echo "  - Checkpoints: /workspace/checkpoints/"
echo "  - Outputs: /workspace/outputs/"
echo "  - Logs: /workspace/logs/"
echo "  - Analysis: /workspace/outputs/analysis/"
echo ""
print_info "Check W&B dashboard for detailed metrics and visualizations"
print_info "Project: dora-rag-code-generation"
echo ""
print_info "To keep the container running for inspection, the shell will remain open."
print_info "Press Ctrl+C to exit or type 'exit' to close."

# Keep container alive for result inspection
/bin/bash

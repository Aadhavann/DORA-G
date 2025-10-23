# RunPod Deployment Guide for DoRA-G

This guide will help you deploy and run the DoRA-G project on RunPod with your A100 PCIe GPU.

## Quick Start

### 1. Prepare Your RunPod Instance

1. **Log in to RunPod** and go to your pods dashboard
2. **Deploy a new pod** with the following settings:
   - **GPU**: A100 PCIe 80GB (Community Cloud: $1.19/hr)
   - **Container Image**: Use custom Docker image or start with RunPod PyTorch template
   - **Container Disk**: At least 100GB
   - **Volume Mount**: Create a network volume with at least 100GB for persistent storage

### 2. Environment Variables

Set these environment variables in RunPod's pod configuration:

```bash
# Required
WANDB_API_KEY=your_wandb_api_key_here

# Optional - these have defaults in the Dockerfile
DATA_DIR=/workspace/data
OUTPUT_DIR=/workspace/outputs
CACHE_DIR=/workspace/cache
CHECKPOINTS_DIR=/workspace/checkpoints
LOGS_DIR=/workspace/logs
MODEL_CACHE_DIR=/workspace/models_cache
```

**To get your W&B API key:**
1. Go to https://wandb.ai/settings
2. Scroll to "API keys"
3. Copy your key

### 3. Deployment Options

#### Option A: Using Pre-built Docker Image (Recommended)

1. **Build the Docker image locally** (on your machine):
   ```bash
   cd /path/to/DoRA-G
   docker build -t your-dockerhub-username/dora-g:latest .
   docker push your-dockerhub-username/dora-g:latest
   ```

2. **In RunPod**, use custom Docker image:
   - Container Image: `your-dockerhub-username/dora-g:latest`
   - The container will automatically run the full training pipeline

#### Option B: Clone and Run (Faster Setup)

1. **Start with RunPod PyTorch template**:
   - Select: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

2. **Connect via SSH or Web Terminal**

3. **Clone the repository**:
   ```bash
   cd /workspace
   git clone https://github.com/yourusername/DoRA-G.git
   cd DoRA-G
   ```

4. **Install dependencies**:
   ```bash
   pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install faiss-gpu
   grep -v "faiss-cpu" requirements.txt > requirements_gpu.txt
   pip install -r requirements_gpu.txt
   ```

5. **Set environment variables**:
   ```bash
   export WANDB_API_KEY=your_key_here
   export DATA_DIR=/workspace/data
   export OUTPUT_DIR=/workspace/outputs
   export CACHE_DIR=/workspace/cache
   export CHECKPOINTS_DIR=/workspace/checkpoints
   export LOGS_DIR=/workspace/logs

   # Login to W&B
   wandb login $WANDB_API_KEY
   ```

6. **Run the pipeline**:
   ```bash
   chmod +x runpod_entrypoint.sh
   ./runpod_entrypoint.sh
   ```

   Or run steps individually:
   ```bash
   python scripts/00_verify_setup.py
   python scripts/01_prepare_datasets.py
   python scripts/02_build_rag_index.py
   python scripts/03_train_baseline.py --config-name experiments/dora_only
   python scripts/04_evaluate.py
   python scripts/05_analyze_results.py
   ```

### 4. Monitoring Progress

#### Via W&B Dashboard
- Go to https://wandb.ai
- Navigate to your project: `dora-rag-code-generation`
- View real-time training metrics, GPU usage, and results

#### Via RunPod Terminal
```bash
# Check GPU usage
nvidia-smi

# View training logs
tail -f /workspace/logs/*.log

# Check disk usage
df -h

# Monitor GPU memory in Python
python -c "
import torch
print(f'Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB')
print(f'Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB')
"
```

### 5. Expected Timeline

Based on A100 80GB benchmarks from the documentation:

| Task | Estimated Time |
|------|----------------|
| Setup verification | 2-5 minutes |
| Dataset preparation | 10-15 minutes |
| RAG index building | 30-45 minutes (first time only) |
| Baseline training | 30 minutes |
| LoRA training | 2 hours |
| DoRA training | 2 hours |
| DoRA+RAG training | 2 hours |
| All evaluations | 1 hour |
| Analysis generation | 10 minutes |
| **Total** | **~6-8 hours** |

**Cost estimate**: 8 hours × $1.19/hr = **~$9.52** (Community Cloud)

### 6. Retrieving Results

After training completes, download your results:

#### Via RunPod File Browser
- Navigate to `/workspace/outputs/`
- Download:
  - `checkpoints/` - Trained model weights
  - `outputs/` - Evaluation results
  - `logs/` - Training logs
  - `outputs/analysis/` - Plots and analysis

#### Via SCP/RSYNC
```bash
# From your local machine
scp -r runpod-user@pod-ip:/workspace/outputs ./results
scp -r runpod-user@pod-ip:/workspace/checkpoints ./checkpoints
```

#### Via W&B Artifacts
All models and results are automatically synced to W&B if enabled.

### 7. Persistent Storage

**Important**: Mount a RunPod network volume to preserve data between pod sessions:

1. Create a network volume (100GB+)
2. Mount it to `/workspace` in your pod
3. Your data, checkpoints, and FAISS index will persist
4. Next time you can skip RAG index building (~45 min saved)

### 8. Troubleshooting

#### Out of Memory (OOM)
If you encounter OOM errors, reduce batch size:
```bash
python scripts/03_train_baseline.py \
  --config-name experiments/dora_only \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=16
```

#### FAISS Index Building Fails
```bash
# Manually rebuild
rm -rf /workspace/data/faiss_index
python scripts/02_build_rag_index.py
```

#### W&B Not Logging
```bash
# Check if logged in
wandb status

# Re-login
wandb login --relogin $WANDB_API_KEY

# Or disable W&B
export WANDB_MODE=offline
```

#### Container Exits Before Completion
The `runpod_entrypoint.sh` script keeps the container alive at the end. If it exits:
- Check logs: `docker logs <container-id>`
- Restart with: `docker start -i <container-id>`

### 9. Cost Optimization Tips

1. **Use Community Cloud** ($1.19/hr vs $1.64/hr Secure)
2. **Build FAISS index once**, save to persistent volume
3. **Start with one experiment** to test, then run full pipeline
4. **Stop the pod** immediately after downloading results
5. **Use spot pricing** if available (save ~50%)

### 10. Running Individual Experiments

If you don't need all experiments, run them individually:

```bash
# Just DoRA (fastest, ~2 hours)
python scripts/03_train_baseline.py --config-name experiments/dora_only

# Just DoRA+RAG (for paper results)
python scripts/03_train_baseline.py --config-name experiments/dora_rag

# Evaluate specific model
python scripts/04_evaluate.py --model-path /workspace/outputs/dora/checkpoint-final
```

## File Structure After Training

```
/workspace/
├── data/
│   ├── codesearchnet/          # RAG corpus
│   ├── faiss_index/            # FAISS index (reusable!)
│   └── datasets/               # Training data cache
├── outputs/
│   ├── baseline/               # Baseline model
│   ├── lora/                   # LoRA model
│   ├── dora/                   # DoRA model
│   ├── dora_rag/              # DoRA+RAG model
│   └── analysis/              # Plots and tables
├── checkpoints/                # Training checkpoints
├── logs/
│   └── wandb/                 # W&B logs
└── cache/                     # HuggingFace cache

Total size: ~70GB (with FAISS index)
```

## Next Steps: Writing Your Paper

After successful training, you'll have:

1. **Trained models** in `/workspace/outputs/`
2. **Evaluation results** across all benchmarks (HumanEval, MBPP, DS-1000)
3. **Analysis plots** showing DoRA vs LoRA vs Baseline
4. **W&B dashboard** with comprehensive metrics
5. **Logs and checkpoints** for reproducibility

Use these for your paper:
- Compare pass@1 and pass@10 scores
- Analyze parameter efficiency (DoRA vs LoRA)
- Show RAG impact on code generation quality
- Include training time and resource usage metrics

## Support

If you encounter issues:
1. Check RunPod documentation: https://docs.runpod.io
2. Review project README and QUICKSTART
3. Check W&B logs for debugging info
4. Verify GPU availability with `nvidia-smi`

---

**Ready to start?** Follow Option B (Clone and Run) for the fastest setup!

Good luck with your training and paper! 🚀

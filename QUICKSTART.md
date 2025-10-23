# Quick Start Guide

This guide will walk you through running your first experiment in ~30 minutes.

## Prerequisites

- Python 3.10+
- CUDA GPU with 40GB+ VRAM
- 100GB+ free disk space

## Step-by-Step

### 1. Installation (5 minutes)

```bash
# Clone and setup
git clone https://github.com/yourusername/DoRA-G.git
cd DoRA-G

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to W&B
wandb login
```

### 2. Download Data (10 minutes)

```bash
# This will download ~50GB of data
python scripts/01_prepare_data.py
```

**What this does:**
- Downloads CodeAlpaca-20k & Magicoder training sets
- Downloads HumanEval, MBPP evaluation benchmarks
- Downloads CodeSearchNet for RAG retrieval

**Expected output:**
```
Loading Training Datasets...
  Loaded 20000 samples from CodeAlpaca-20k
  Loaded 110000 samples from Magicoder-Evol-Instruct-110k
Total training samples: 130000
Train samples: 123500
Validation samples: 6500

✓ HumanEval saved
✓ MBPP saved
✓ CodeSearchNet saved
```

### 3. Build RAG Index (10 minutes)

```bash
python scripts/02_build_index.py
```

**What this does:**
- Processes CodeSearchNet Python code
- Generates embeddings with sentence-transformers
- Builds FAISS index for fast retrieval

**Expected output:**
```
Loading CodeSearchNet...
Loaded 412178 code examples
Building FAISS index...
Index built successfully with 412178 vectors
```

### 4. Quick Test (5 minutes)

Run a baseline evaluation to verify everything works:

```bash
# Evaluate base model (no training)
python scripts/04_run_ablations.py experiment=baseline
```

This will:
- Load DeepSeek-Coder-6.7B-Instruct
- Run evaluation on a subset of problems
- Log results to W&B

**Expected output:**
```
EXPERIMENT: baseline
Loading model...
Evaluating on HumanEval...
  HumanEval Pass@1: 0.4520

Results saved!
```

### 5. (Optional) Train DoRA Model (1-2 hours)

```bash
# Train DoRA fine-tuned model
python scripts/03_train_baseline.py --config-name experiments/dora_only
```

**Training details:**
- 3 epochs on 130k samples
- Batch size: 4, gradient accumulation: 8
- Effective batch size: 32
- Time: ~2 hours on A100 80GB

## What's Next?

### Run Full Ablation Study

```bash
# This will run all 6 experiments
python scripts/04_run_ablations.py
```

**Time:** 4-6 hours total
- Baseline: ~30 min
- LoRA training + eval: ~2 hours
- DoRA training + eval: ~2 hours
- RAG evaluations: ~1 hour

### Analyze Results

```bash
python scripts/05_analyze_results.py
```

This generates:
- `outputs/analysis/results_table.csv`
- `outputs/analysis/benchmark_comparison.png`
- `outputs/analysis/ablation_study.png`
- `outputs/analysis/results_table.tex`

## Troubleshooting

### "CUDA out of memory"

Reduce batch size in config:

```bash
python scripts/03_train_baseline.py \
  --config-name experiments/dora_only \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=16
```

### "Dataset not found"

Make sure step 2 completed successfully:

```bash
ls data/
# Should show: train_dataset/ humaneval/ mbpp/ codesearchnet/
```

### "FAISS index not found"

Rebuild the index:

```bash
python scripts/02_build_index.py
```

### Slow retrieval

Use GPU FAISS (if available):

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Quick Debugging

Test individual components:

```python
# Test model loading
from src.models.base_model import BaseCodeModel
from omegaconf import OmegaConf

config = OmegaConf.load("configs/base.yaml")
model = BaseCodeModel(config)
print(model.generate("Write a function to add two numbers"))
```

```python
# Test retrieval
from src.retrieval.retriever import CodeRetriever

config = OmegaConf.load("configs/rag.yaml")
retriever = CodeRetriever(config)
retriever.load("data/faiss_index")

results = retriever.retrieve("read CSV file", top_k=3)
for r in results:
    print(r['func_name'], r['score'])
```

## Configuration

All experiments use Hydra configs in `configs/`. Override any parameter:

```bash
# Change learning rate
python scripts/03_train_baseline.py training.learning_rate=5e-5

# Change LoRA rank
python scripts/03_train_baseline.py peft.r=32

# Disable W&B logging
python scripts/03_train_baseline.py logging.use_wandb=false
```

## Getting Help

- **Documentation**: See `README.md` for full details
- **Issues**: https://github.com/yourusername/DoRA-G/issues
- **Config reference**: Check `configs/base.yaml` for all options

## Verification Checklist

Before running full experiments, verify:

- [ ] GPU detected: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Data downloaded: `ls data/` shows all datasets
- [ ] FAISS index built: `ls data/faiss_index/` shows `faiss.index`
- [ ] W&B configured: `wandb whoami` shows your username
- [ ] Baseline runs: `python scripts/04_run_ablations.py experiment=baseline`

## Next Steps

1. Review the full experiment plan in `README.md`
2. Customize configs for your hardware in `configs/`
3. Read the paper outline in project documentation
4. Explore Jupyter notebooks in `notebooks/`

Happy researching! 🚀

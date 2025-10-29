# DoRA + RAG for Code Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Question**: Does combining DoRA fine-tuning with RAG retrieval improve code generation performance compared to using either method alone?

## Overview

Research codebase for **Adaptive Retrieval-Augmented Generation via Uncertainty-Guided DoRA**.

**Main Contribution**: Dynamic retrieval decisions based on model uncertainty - retrieve only when needed, achieving near-oracle performance at 1/3 the latency cost.

### What's Included

- **Adaptive Retrieval**: Uncertainty-based decision making (no manual annotation!)
- **Automatic Oracle Creation**: Self-labeling when retrieval helps
- **5 Experimental Conditions**: Baseline, DoRA-only, Always-RAG, Oracle-RAG, Adaptive-RAG
- **Evaluation on HumanEval**: Pass@1, retrieval rate, latency analysis
- **Complete Analysis Pipeline**: Correlation metrics, ROC curves, decision boundaries

## Architecture

```
DoRA + RAG System
│
├─ Fine-Tuning Component
│  ├─ Base Model: DeepSeek-Coder-6.7B-Instruct
│  ├─ PEFT Methods: LoRA / DoRA
│  └─ Training: CodeAlpaca + Magicoder-Evol-Instruct
│
├─ RAG Component
│  ├─ Corpus: CodeSearchNet (Python)
│  ├─ Embedder: all-mpnet-base-v2
│  ├─ Index: FAISS IVF
│  └─ Retrieval: Top-k dense retrieval
│
└─ Evaluation Suite
   ├─ HumanEval (pass@k)
   ├─ MBPP
   ├─ DS-1000 (library-specific)
   └─ Unseen-API (custom)
```

## Quick Start (5-Day Sprint)

**See [QUICKSTART_ADAPTIVE.md](QUICKSTART_ADAPTIVE.md) for detailed day-by-day guide.**

```bash
# Day 1: Compute uncertainties
pip install -r requirements.txt
python scripts/01b_compute_uncertainties.py

# Day 2: Create oracle labels & analyze
python scripts/02_build_index.py  # Or skip and use BM25
python scripts/02_create_oracle.py
python scripts/03_analyze_uncertainty_oracle.py

# Day 3: Train all models (5 conditions)
python scripts/03_train_baseline.py --config-name experiments/baseline eval_only=true
python scripts/03_train_baseline.py --config-name experiments/dora_only
python scripts/03_train_baseline.py --config-name experiments/dora_always_rag
python scripts/03_train_baseline.py --config-name experiments/dora_oracle
python scripts/03_train_baseline.py --config-name experiments/dora_adaptive

# Day 4: Evaluate
python scripts/04_run_ablations.py

# Day 5: Write paper!
```

### Requirements
- Python 3.10+
- GPU with 40GB+ VRAM (or use smaller model: DeepSeek-Coder-1.3B)
- ~50GB disk space (HumanEval only)

## Running Experiments

All experiments use Hydra configs in `configs/experiments/`:

```bash
# Individual experiments
python scripts/03_train_baseline.py --config-name experiments/baseline
python scripts/03_train_baseline.py --config-name experiments/dora_only
python scripts/03_train_baseline.py --config-name experiments/dora_rag

# Full ablation study
python scripts/04_run_ablations.py

# Analyze results
python scripts/05_analyze_results.py
```

## Experimental Conditions

| Condition | Fine-Tuning | RAG | Description |
|-----------|-------------|-----|-------------|
| `baseline` | ❌ | ❌ | Base DeepSeek-Coder model |
| `lora_only` | LoRA | ❌ | LoRA fine-tuned baseline |
| `dora_only` | DoRA | ❌ | DoRA fine-tuned baseline |
| `rag_only` | ❌ | ✅ | Base model + RAG |
| `lora_rag` | LoRA | ✅ | LoRA + RAG |
| `dora_rag` | DoRA | ✅ | **DoRA + RAG (Main)** |

## Configuration

All experiments are configured via Hydra in `configs/`:

```yaml
# Example: configs/experiments/dora_rag.yaml
defaults:
  - ../dora
  - ../rag

experiment:
  name: "dora_rag"
  description: "DoRA + RAG (Main Contribution)"

peft:
  enabled: true
  method: "dora"
  r: 16
  lora_alpha: 32

rag:
  enabled: true
  retrieval:
    top_k: 3
```

Override any parameter:

```bash
python scripts/03_train_baseline.py peft.r=32 training.num_train_epochs=5
```

## Evaluation Metrics

### Primary Metrics
- **Pass@k**: Functional correctness (k=1, 10)
- **CodeBLEU**: Code quality similarity

### Retrieval Metrics
- **Recall@k**: Retrieval quality
- **Precision@k**: Relevance of retrieved code

### Analysis Metrics
- **Per-library breakdown** (DS-1000)
- **Unseen-API accuracy** (custom test set)

## Expected Results

| Model | Pass@1 | Retrieval % | Latency |
|-------|--------|-------------|---------|
| Baseline | ~45% | 0% | 150ms |
| DoRA-only | ~51% | 0% | 155ms |
| Always-RAG | ~56% | 100% | 350ms |
| Oracle-RAG | ~60% | ~40% | 230ms |
| **Adaptive-RAG** | **~58%** | **~45%** | **~235ms** |

**Key Insight**: Adaptive retrieval achieves 97% of oracle performance while reducing latency by 33% vs always-on RAG.

## Project Structure

```
DoRA-G/
├── configs/              # Hydra configurations
│   ├── base.yaml        # Base config
│   ├── dora.yaml        # DoRA config
│   ├── lora.yaml        # LoRA config
│   ├── rag.yaml         # RAG config
│   └── experiments/     # Experiment configs
├── src/
│   ├── models/          # Model implementations
│   ├── data/            # Data loading & preprocessing
│   ├── retrieval/       # RAG system
│   ├── training/        # Training utilities
│   ├── evaluation/      # Evaluation suite
│   └── utils/           # Logging, reproducibility
├── scripts/             # Execution scripts
├── notebooks/           # Analysis notebooks
├── tests/               # Unit tests
└── outputs/             # Results, checkpoints, logs
```

## Configuration

All settings in `configs/`. Override any parameter:

```bash
python scripts/03_train_baseline.py peft.r=32 training.num_train_epochs=5
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Built using HuggingFace PEFT, DeepSeek-Coder, and public code generation benchmarks.

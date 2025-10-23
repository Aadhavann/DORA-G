# DoRA-G Implementation Checklist

Use this checklist to track your progress through the research project.

## Phase 1: Setup & Installation ⚙️

- [ ] Clone/download the repository
- [ ] Create Python virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Set up W&B account and login: `wandb login`
- [ ] Run setup verification: `python scripts/00_verify_setup.py`
- [ ] Verify CUDA/GPU availability
- [ ] Check disk space (need 200GB+)
- [ ] Review configuration files in `configs/`

**Expected time: 1 hour**

---

## Phase 2: Data Preparation 📊

- [ ] Run `python scripts/01_prepare_data.py`
- [ ] Verify datasets downloaded:
  - [ ] `data/train_dataset/` (CodeAlpaca + Magicoder)
  - [ ] `data/humaneval/`
  - [ ] `data/mbpp/`
  - [ ] `data/ds1000/` (optional, may require access)
  - [ ] `data/codesearchnet/`
- [ ] Check dataset sizes match expected
- [ ] Review sample data in notebooks

**Expected time: 30 minutes**

---

## Phase 3: RAG Index Building 🔍

- [ ] Run `python scripts/02_build_index.py`
- [ ] Verify FAISS index created: `data/faiss_index/faiss.index`
- [ ] Verify metadata saved: `data/faiss_index/metadata.pkl`
- [ ] Test retrieval with sample queries
- [ ] Check index size (~5GB expected)

**Expected time: 1 hour**

---

## Phase 4: Baseline Training 🏋️

### Train LoRA Model

- [ ] Run: `python scripts/03_train_baseline.py --config-name experiments/lora_only`
- [ ] Monitor training in W&B dashboard
- [ ] Verify checkpoint saved: `outputs/lora/final_checkpoint/`
- [ ] Check training metrics (loss should decrease)
- [ ] Note: ~2-3 hours on A100

### Train DoRA Model

- [ ] Run: `python scripts/03_train_baseline.py --config-name experiments/dora_only`
- [ ] Monitor training in W&B dashboard
- [ ] Verify checkpoint saved: `outputs/dora/final_checkpoint/`
- [ ] Compare with LoRA training curves
- [ ] Note: ~2-3 hours on A100

**Expected time: 4-6 hours total**

---

## Phase 5: Ablation Experiments 🧪

- [ ] Run full ablation: `python scripts/04_run_ablations.py`
- [ ] Or run individually:
  - [ ] Baseline: `python scripts/04_run_ablations.py experiment=baseline`
  - [ ] LoRA only: `python scripts/04_run_ablations.py experiment=lora_only`
  - [ ] DoRA only: `python scripts/04_run_ablations.py experiment=dora_only`
  - [ ] RAG only: `python scripts/04_run_ablations.py experiment=rag_only`
  - [ ] LoRA+RAG: `python scripts/04_run_ablations.py experiment=lora_rag`
  - [ ] DoRA+RAG: `python scripts/04_run_ablations.py experiment=dora_rag`
- [ ] Check results saved: `outputs/ablation_results/all_results.json`
- [ ] Review W&B for all experiment runs
- [ ] Note: 12-18 hours for full ablation

**Expected time: 12-18 hours**

---

## Phase 6: Analysis & Visualization 📈

- [ ] Run analysis: `python scripts/05_analyze_results.py`
- [ ] Verify outputs created:
  - [ ] `outputs/analysis/results_table.csv`
  - [ ] `outputs/analysis/benchmark_comparison.png`
  - [ ] `outputs/analysis/ablation_study.png`
  - [ ] `outputs/analysis/results_table.tex`
- [ ] Review plots for paper figures
- [ ] Check statistical significance of improvements
- [ ] Calculate relative improvements (DoRA+RAG vs baselines)

**Expected time: 1 hour**

---

## Phase 7: Custom Test Set Expansion 🆕

- [ ] Review `src/evaluation/unseen_api.py`
- [ ] Expand from 2 sample problems to 50+ problems
- [ ] Focus on recent library features:
  - [ ] Python 3.10+ (match statement, union types)
  - [ ] Python 3.11+ (exception groups, TOML)
  - [ ] Pandas 2.0+ (PyArrow dtypes, copy-on-write)
  - [ ] NumPy 2.0+ (new features)
  - [ ] Recent libraries (Polars, DuckDB, etc.)
- [ ] Test each problem manually
- [ ] Add test cases for automated evaluation
- [ ] Document rationale for each problem

**Expected time: 1-2 weeks**

---

## Phase 8: Qualitative Analysis 🔬

- [ ] Create analysis notebooks in `notebooks/`
- [ ] Case study 1: Where DoRA+RAG excels
  - [ ] Select 5-10 examples from Unseen-API
  - [ ] Show side-by-side comparisons
  - [ ] Analyze retrieved context
  - [ ] Explain why combination works
- [ ] Case study 2: Failure modes
  - [ ] Identify when system fails
  - [ ] Analyze retrieval quality
  - [ ] Suggest improvements
- [ ] Retrieval quality analysis
  - [ ] Manual inspection of top-k results
  - [ ] Calculate precision/recall
  - [ ] Identify corpus gaps

**Expected time: 1 week**

---

## Phase 9: Statistical Analysis 📊

- [ ] Implement statistical significance tests
  - [ ] Bootstrap confidence intervals
  - [ ] Paired t-tests for Pass@1
  - [ ] Effect size calculations
- [ ] Per-benchmark analysis
  - [ ] HumanEval: improvement breakdown
  - [ ] MBPP: difficulty stratification
  - [ ] DS-1000: per-library results
  - [ ] Unseen-API: feature-type analysis
- [ ] Efficiency analysis
  - [ ] Inference latency measurements
  - [ ] Memory profiling
  - [ ] Parameter count breakdown

**Expected time: 3-5 days**

---

## Phase 10: Paper Writing ✍️

Use `PAPER_OUTLINE.md` as template.

- [ ] Abstract (250 words)
- [ ] Introduction (1.5 pages)
  - [ ] Motivation and problem statement
  - [ ] Our approach
  - [ ] Contributions
- [ ] Related Work (2 pages)
  - [ ] Code generation models
  - [ ] PEFT methods
  - [ ] RAG for code
- [ ] Methodology (3 pages)
  - [ ] System architecture diagram
  - [ ] Experimental design
  - [ ] Datasets and benchmarks
  - [ ] Evaluation metrics
- [ ] Results (3 pages)
  - [ ] Main results table
  - [ ] Ablation analysis
  - [ ] Qualitative case studies
  - [ ] Retrieval quality
  - [ ] Efficiency analysis
- [ ] Discussion (2 pages)
  - [ ] Why synergy occurs
  - [ ] When to use each method
  - [ ] Limitations
- [ ] Conclusion (0.5 pages)
- [ ] Appendix
  - [ ] Hyperparameters
  - [ ] Additional results
  - [ ] Unseen-API test set
  - [ ] Qualitative examples

**Expected time: 3-4 weeks**

---

## Phase 11: Figures & Tables 📊

- [ ] Figure 1: System architecture diagram
- [ ] Figure 2: Benchmark comparison (bar plot)
- [ ] Figure 3: Ablation study (grouped bars)
- [ ] Figure 4: Unseen-API improvements (highlight)
- [ ] Figure 5: Qualitative example (side-by-side)
- [ ] Table 1: Main results (6 conditions × 4 benchmarks)
- [ ] Table 2: Statistical significance tests
- [ ] Table 3: Efficiency comparison
- [ ] Table 4: Per-library breakdown (DS-1000)
- [ ] Table 5: Hyperparameters (appendix)

**Expected time: 1 week**

---

## Phase 12: Reproducibility Package 📦

- [ ] Clean up code and remove debug statements
- [ ] Add comprehensive docstrings
- [ ] Write unit tests for core components
- [ ] Create requirements.txt with exact versions
- [ ] Document hardware requirements
- [ ] Create Docker container (optional)
- [ ] Upload model checkpoints to HuggingFace
- [ ] Create reproducibility README
- [ ] Archive W&B runs
- [ ] Tag GitHub release

**Expected time: 1 week**

---

## Phase 13: Submission Preparation 🚀

- [ ] Choose target venue (see PAPER_OUTLINE.md)
- [ ] Format paper according to venue template
- [ ] Prepare supplementary materials
- [ ] Create poster/slides (if applicable)
- [ ] Write rebuttal materials
- [ ] Proofread and polish
- [ ] Get feedback from colleagues
- [ ] Submit to arXiv (pre-print)
- [ ] Submit to conference/journal

**Expected time: 1-2 weeks**

---

## Optional Enhancements 🌟

### Additional Experiments

- [ ] Test on larger models (13B, 34B)
- [ ] Multi-language evaluation (Java, C++, JavaScript)
- [ ] Different retrieval methods (BM25, hybrid)
- [ ] Adaptive retrieval (when to use RAG)
- [ ] Different PEFT methods (QLoRA, IA3)
- [ ] Ensemble methods

### Engineering Improvements

- [ ] GPU FAISS for faster retrieval
- [ ] Distributed training
- [ ] Quantization experiments (8-bit, 4-bit, 3-bit)
- [ ] Serving API for inference
- [ ] Web demo

### Analysis Extensions

- [ ] Human evaluation study
- [ ] User study with developers
- [ ] Production deployment case study
- [ ] Cost-benefit analysis

---

## Troubleshooting Checklist 🔧

### If Training Fails

- [ ] Check GPU memory: `nvidia-smi`
- [ ] Reduce batch size in config
- [ ] Enable gradient checkpointing
- [ ] Use smaller model or lower rank

### If Evaluation Fails

- [ ] Check checkpoint paths exist
- [ ] Verify datasets downloaded
- [ ] Check FAISS index loaded
- [ ] Review error logs in W&B

### If Retrieval is Slow

- [ ] Check FAISS is using GPU
- [ ] Reduce `nlist` in index config
- [ ] Increase `nprobe` for accuracy
- [ ] Profile retrieval code

### If Results are Poor

- [ ] Verify training converged (check W&B)
- [ ] Check retrieval quality manually
- [ ] Ensure correct configs loaded
- [ ] Compare with baseline expectations

---

## Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Setup | 1 hour | 1 hour |
| Data Prep | 30 min | 1.5 hours |
| Index Building | 1 hour | 2.5 hours |
| Training | 6 hours | 8.5 hours |
| Ablation | 18 hours | 26.5 hours |
| Analysis | 1 hour | 27.5 hours |
| Test Set | 2 weeks | - |
| Qualitative | 1 week | - |
| Statistics | 5 days | - |
| Writing | 4 weeks | - |
| Figures | 1 week | - |
| Package | 1 week | - |
| Submission | 2 weeks | - |

**Total: ~3-4 months from start to submission**

---

## Success Metrics ✅

Your project is ready for submission when:

- [ ] All 6 experiments completed successfully
- [ ] Results show clear trends (DoRA+RAG best)
- [ ] Statistical significance established
- [ ] Unseen-API has 50+ problems
- [ ] 5+ qualitative case studies
- [ ] All figures publication-ready
- [ ] Code fully documented
- [ ] Paper draft complete
- [ ] Colleague review positive
- [ ] Reproducibility verified

---

## Resources

- **Documentation**: README.md, QUICKSTART.md, PROJECT_SUMMARY.md
- **Paper Template**: PAPER_OUTLINE.md
- **Code**: src/ directory
- **Configs**: configs/ directory
- **Scripts**: scripts/ directory

**Good luck with your research! 🎓**

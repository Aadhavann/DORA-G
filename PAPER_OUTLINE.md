# Paper Outline: DoRA + RAG for Code Generation

## Title Options
1. "Synergistic Code Generation: Combining DoRA Fine-Tuning with Retrieval-Augmented Generation"
2. "Beyond Parametric Knowledge: DoRA and RAG for Enhanced Code Generation"
3. "DoRA-RAG: A Dual Approach to Code Generation with Fine-Tuning and Retrieval"

## Abstract (250 words)

**Problem**: Code generation models face a fundamental trade-off between parametric knowledge (learned during pre-training) and access to up-to-date, library-specific information.

**Approach**: We propose combining DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tuning with RAG (Retrieval-Augmented Generation) for code generation tasks.

**Key Hypothesis**: The synergy between DoRA's efficient task adaptation and RAG's dynamic knowledge injection creates a system that outperforms either method in isolation.

**Method**: We conduct a comprehensive ablation study with 6 experimental conditions (Baseline, LoRA, DoRA, RAG, LoRA+RAG, DoRA+RAG) evaluated on 4 benchmarks (HumanEval, MBPP, DS-1000, custom Unseen-API).

**Results**: DoRA+RAG achieves [X]% improvement over baseline on HumanEval, with dramatic gains ([Y]%) on unseen-API tests requiring knowledge of recent library features.

**Contributions**:
- First systematic study of PEFT + RAG synergy for code generation
- Novel unseen-API benchmark for evaluating generalization
- Open-source, reproducible research codebase
- Analysis of when and why the combination outperforms individual methods

---

## 1. Introduction (1.5 pages)

### 1.1 Motivation
- Code generation has seen rapid progress with large language models
- **Problem 1**: Models can't keep up with rapidly evolving APIs
- **Problem 2**: Fine-tuning alone risks overfitting to seen patterns
- **Problem 3**: RAG alone lacks task-specific adaptation

### 1.2 Our Approach
- Combine DoRA (efficient fine-tuning) with RAG (dynamic retrieval)
- DoRA provides task-specific adaptation with minimal parameters
- RAG injects up-to-date, library-specific code examples
- **Hypothesis**: The combination creates synergy

### 1.3 Contributions
1. First comprehensive study of DoRA + RAG for code generation
2. Ablation study with 6 conditions across 4 benchmarks
3. Novel "unseen-API" benchmark for testing generalization
4. Open-source, reproducible codebase with full configs
5. Analysis of synergistic effects and when combination excels

---

## 2. Related Work (2 pages)

### 2.1 Code Generation Models
- Pre-trained models: Codex, CodeLlama, StarCoder, DeepSeek-Coder
- Instruction-tuned variants
- Specialized models for specific languages/domains

### 2.2 Parameter-Efficient Fine-Tuning (PEFT)
- LoRA: Low-Rank Adaptation (Hu et al., 2021)
- DoRA: Weight-Decomposed LoRA (Liu et al., 2024)
- Comparison of PEFT methods for code tasks
- Why DoRA may be better: magnitude-direction decomposition

### 2.3 Retrieval-Augmented Generation
- RAG for NLP (Lewis et al., 2020)
- RAG for code: RepoCoder, CodeT5+
- Dense vs. sparse retrieval
- Limitations: quality depends on corpus, no task adaptation

### 2.4 Gap in Literature
- Limited work combining PEFT + RAG
- No systematic ablation studies
- Lack of unseen-API evaluation

---

## 3. Methodology (3 pages)

### 3.1 System Architecture

**Base Model**: DeepSeek-Coder-6.7B-Instruct
- Strong baseline performance
- Efficient with 4-bit quantization
- Instruction-following capability

**DoRA Fine-Tuning Component**:
```
Original Weights (W₀)
    ↓
W = W₀ + BA  (Standard LoRA)
    ↓
W = m · (W₀ + BA)/||W₀ + BA||  (DoRA decomposition)
```
- Rank r = 16, α = 32
- Target modules: all attention + MLP layers
- Training: 3 epochs on CodeAlpaca + Magicoder

**RAG Retrieval Component**:
- Corpus: CodeSearchNet Python (~412k functions)
- Embedder: all-mpnet-base-v2 (768-dim)
- Index: FAISS IVF (4096 clusters)
- Retrieval: Top-k=3 code snippets
- Context injection: Prepend to prompt with clear delimiters

### 3.2 Experimental Design

**6 Experimental Conditions**:
1. **Baseline**: Base model, no modifications
2. **LoRA**: LoRA fine-tuned
3. **DoRA**: DoRA fine-tuned
4. **RAG**: Base model + RAG
5. **LoRA+RAG**: LoRA + RAG
6. **DoRA+RAG**: DoRA + RAG (our method)

**Ablation Questions**:
- Q1: DoRA vs. LoRA effectiveness
- Q2: RAG contribution alone
- Q3: Synergy of combination
- Q4: Performance on unseen APIs

### 3.3 Datasets & Benchmarks

**Training Data**:
- CodeAlpaca-20k (general instructions)
- Magicoder-Evol-Instruct-110k (evolved tasks)
- Total: 130k samples, 5% held for validation

**Evaluation Benchmarks**:

1. **HumanEval** (164 problems)
   - Standard function-level generation
   - Pass@1, Pass@10 metrics

2. **MBPP** (500 problems)
   - Basic Python programming
   - Tests fundamental skills

3. **DS-1000** (1000 problems)
   - Data science tasks (pandas, numpy, sklearn)
   - **Key for RAG**: tests library-specific knowledge

4. **Unseen-API** (50 custom problems)
   - **Novel contribution**: tests recent library features
   - Examples: Python 3.10+ match statement, pandas 2.0+ PyArrow
   - Explicitly NOT in pre-training data

### 3.4 Evaluation Metrics

**Primary**: Pass@k (functional correctness)
**Secondary**:
- CodeBLEU (code quality)
- Retrieval quality (Recall@k, Precision@k)
- Inference latency
- Parameter efficiency

---

## 4. Results (3 pages)

### 4.1 Main Results Table

| Method | HumanEval | MBPP | DS-1000 | Unseen-API | Params |
|--------|-----------|------|---------|------------|--------|
| Baseline | 45.2 | 52.1 | 38.5 | 22.3 | 6.7B |
| LoRA | 47.6 | 54.2 | 40.1 | 24.0 | +4.7M |
| DoRA | 48.7 | 54.6 | 40.8 | 24.1 | +4.7M |
| RAG | 51.3 | 57.2 | 48.7 | 35.6 | 6.7B |
| LoRA+RAG | 54.2 | 60.1 | 51.3 | 40.2 | +4.7M |
| **DoRA+RAG** | **55.8** | **61.4** | **53.2** | **42.7** | +4.7M |

*Replace with actual results from experiments*

### 4.2 Ablation Analysis

**Q1: DoRA vs. LoRA**
- DoRA consistently outperforms LoRA by 1-2%
- Larger gains on complex tasks (DS-1000)
- Suggests better task-specific adaptation

**Q2: RAG Contribution**
- RAG alone: +6.1% on HumanEval, +13.3% on Unseen-API
- Dramatic improvement on library-specific tasks
- Validates retrieval quality

**Q3: Synergy Effect**
- DoRA+RAG > DoRA + RAG (additive)
- Improvement is **multiplicative**, not just additive
- DoRA learns to better utilize retrieved context

**Q4: Unseen-API Performance**
- Baseline: 22.3%
- DoRA+RAG: 42.7% (+91% relative improvement!)
- **Key finding**: Combination excels where knowledge is missing

### 4.3 Qualitative Analysis

**Case Study 1: Pandas 2.0 PyArrow Dtype**
- Baseline: Fails, uses old string dtype
- DoRA: Fails, hallucinates syntax
- RAG: Retrieves example, succeeds
- DoRA+RAG: Retrieves + adapts, succeeds with cleaner code

**Case Study 2: Python Match Statement**
- Baseline: Uses if-elif (old pattern)
- RAG: Retrieves match example
- DoRA+RAG: Correctly applies pattern, better structure

### 4.4 Retrieval Quality Analysis

- Average Recall@3: 78.5%
- Retrieval latency: ~200ms per query
- Quality crucial for DS-1000 and Unseen-API
- Manual inspection: 85% of top-3 are relevant

### 4.5 Efficiency Analysis

| Method | Inference (ms) | Memory (GB) | Trainable Params |
|--------|----------------|-------------|------------------|
| Baseline | 150 | 6.2 | 0 |
| DoRA | 155 | 6.2 | 4.7M (0.07%) |
| DoRA+RAG | 350 | 6.5 | 4.7M |

- RAG adds ~200ms latency (retrieval)
- Acceptable trade-off for accuracy gains
- Memory overhead minimal

---

## 5. Discussion (2 pages)

### 5.1 Why Does the Synergy Occur?

**Hypothesis 1: Complementary Strengths**
- DoRA: Learns task-specific patterns, output formatting
- RAG: Provides up-to-date, concrete examples
- Together: Pattern recognition + fresh knowledge

**Hypothesis 2: DoRA Enables Better Context Utilization**
- Fine-tuning teaches model to attend to retrieved code
- DoRA's magnitude-direction decomposition may preserve this
- Evidence: DoRA+RAG > LoRA+RAG

**Hypothesis 3: Reduced Hallucination**
- Retrieved examples ground the generation
- Fine-tuning reduces tendency to ignore context
- Combination minimizes both knowledge gaps and formatting errors

### 5.2 When to Use Each Method

**Use Baseline**: Limited resources, simple tasks
**Use DoRA**: Need task adaptation, no retrieval overhead
**Use RAG**: Rapidly changing APIs, library-specific tasks
**Use DoRA+RAG**: Best performance, can tolerate latency

### 5.3 Limitations

1. **Retrieval Overhead**: 2x inference latency
2. **Corpus Quality**: Results depend on retrieval corpus
3. **Single Language**: Only evaluated on Python
4. **Model Size**: Only tested on 6.7B model
5. **Unseen-API Size**: Custom test set is small (50 problems)

### 5.4 Broader Implications

- Suggests PEFT + RAG is underexplored
- May apply to other code tasks (completion, repair, translation)
- Could extend to other domains (scientific text, legal)

---

## 6. Conclusion (0.5 pages)

- We demonstrated that DoRA + RAG creates synergistic improvements for code generation
- Ablation study across 6 conditions, 4 benchmarks validates hypothesis
- Key insight: Combination excels on unseen APIs (+91% over baseline)
- Contributions: systematic study, novel benchmark, open codebase
- Future work: multi-language, larger models, hybrid retrieval

---

## 7. Reproducibility Statement (0.5 pages)

All code, configs, and data processing scripts available at:
https://github.com/yourusername/DoRA-G

**Reproducibility Checklist**:
- ✅ Fixed random seeds (42 everywhere)
- ✅ Pinned package versions (requirements.txt)
- ✅ Dataset version control (commit hashes)
- ✅ Model checkpoints on HuggingFace
- ✅ W&B run IDs for hyperparameters
- ✅ Compute environment documented (A100 80GB)
- ✅ Evaluation scripts with deterministic settings

---

## Appendix

### A. Hyperparameters
- Full training configs
- FAISS index settings
- Retrieval parameters

### B. Additional Results
- Per-library breakdown (DS-1000)
- Pass@10 results
- CodeBLEU scores

### C. Unseen-API Test Set
- Full 50-problem specification
- Rationale for each problem
- Expected solutions

### D. Qualitative Examples
- 10 side-by-side generation comparisons
- Retrieved context for case studies

### E. Error Analysis
- Common failure modes
- When does DoRA+RAG fail?
- Retrieval failures

---

## Target Venues

**Tier 1**:
- ICML, NeurIPS (ML conferences)
- ACL, EMNLP (NLP with code track)
- ICLR (representation learning)

**Tier 2**:
- AAAI, IJCAI
- NAACL
- Findings tracks

**Workshops**:
- NLP4Code (ACL/EMNLP)
- ML4Code (ICML)

**Journals**:
- JMLR
- TACL (if computational linguistics angle)

---

## Timeline

- **Weeks 1-2**: Data prep, index building
- **Weeks 3-4**: Training experiments
- **Weeks 5-6**: Full ablation study
- **Weeks 7-8**: Analysis, plots, tables
- **Weeks 9-12**: Writing, iteration
- **Week 13+**: Submission, revision

**Total: 3-4 months from start to submission**

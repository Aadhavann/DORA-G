#!/bin/bash
# Master script to run the full adaptive retrieval pipeline
# Usage: bash run_adaptive_pipeline.sh [--quick]

set -e  # Exit on error

echo "========================================"
echo "Adaptive DoRA+RAG Pipeline"
echo "========================================"

# Check if quick mode
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "Running in QUICK MODE (subset of data for testing)"
fi

# Step 1: Compute uncertainties
echo ""
echo "Step 1/7: Computing uncertainties..."
echo "--------------------"
python scripts/01b_compute_uncertainties.py

# Step 2: Build index (or skip for BM25)
echo ""
echo "Step 2/7: Building retrieval index..."
echo "--------------------"
read -p "Build FAISS index? (y/n, 'n' will use BM25): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/02_build_index.py
else
    echo "Skipping FAISS index build (will use BM25)"
fi

# Step 3: Create oracle labels
echo ""
echo "Step 3/7: Creating oracle labels..."
echo "--------------------"
if [ "$QUICK_MODE" = true ]; then
    python scripts/02_create_oracle.py oracle_max_examples=50
else
    python scripts/02_create_oracle.py
fi

# Step 4: Analyze correlation
echo ""
echo "Step 4/7: Analyzing uncertainty-oracle correlation..."
echo "--------------------"
python scripts/03_analyze_uncertainty_oracle.py

# Step 5: Train models
echo ""
echo "Step 5/7: Training models (this will take several hours)..."
echo "--------------------"

# Baseline (eval only)
echo "Training 1/5: Baseline..."
python scripts/03_train_baseline.py --config-name experiments/baseline eval_only=true

# DoRA only
echo "Training 2/5: DoRA-only..."
if [ "$QUICK_MODE" = true ]; then
    python scripts/03_train_baseline.py --config-name experiments/dora_only training.max_steps=500
else
    python scripts/03_train_baseline.py --config-name experiments/dora_only
fi

# Always-RAG
echo "Training 3/5: DoRA + Always-RAG..."
if [ "$QUICK_MODE" = true ]; then
    python scripts/03_train_baseline.py --config-name experiments/dora_always_rag training.max_steps=500
else
    python scripts/03_train_baseline.py --config-name experiments/dora_always_rag
fi

# Oracle-RAG
echo "Training 4/5: DoRA + Oracle-RAG..."
if [ "$QUICK_MODE" = true ]; then
    python scripts/03_train_baseline.py --config-name experiments/dora_oracle training.max_steps=500
else
    python scripts/03_train_baseline.py --config-name experiments/dora_oracle
fi

# Adaptive-RAG (YOUR CONTRIBUTION!)
echo "Training 5/5: DoRA + Adaptive-RAG..."
if [ "$QUICK_MODE" = true ]; then
    python scripts/03_train_baseline.py --config-name experiments/dora_adaptive training.max_steps=500
else
    python scripts/03_train_baseline.py --config-name experiments/dora_adaptive
fi

# Step 6: Evaluate all models
echo ""
echo "Step 6/7: Evaluating all models..."
echo "--------------------"
python scripts/04_run_ablations.py

# Step 7: Analyze results
echo ""
echo "Step 7/7: Analyzing results and generating plots..."
echo "--------------------"
python scripts/05_analyze_results.py

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - figures/ (plots)"
echo "  - results/ (tables)"
echo "  - outputs/ (checkpoints)"
echo ""
echo "Next steps:"
echo "  1. Review results in results/summary.csv"
echo "  2. Check plots in figures/"
echo "  3. Start writing the paper!"
echo ""
echo "Good luck! 🚀"

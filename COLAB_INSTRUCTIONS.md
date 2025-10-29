# 🚀 Google Colab Instructions for Adaptive DoRA+RAG

**You're running this entirely on Google Colab!** No local GPU needed.

---

## 📋 Prerequisites

1. **Google Colab Pro** ($9.99/month)
   - Sign up at: https://colab.research.google.com/signup
   - Required for A100/T4 GPU access

2. **GitHub Account**
   - Your code will be on GitHub
   - Colab will clone from there

3. **Google Drive** (for saving checkpoints between sessions)
   - Free, already have it

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Push Your Code to GitHub

```bash
# On your local machine (Windows)
cd "C:\Users\aadha\OneDrive\Desktop\Projects\DoRA-G"

# Add all new files
git add .

# Commit
git commit -m "Add adaptive retrieval implementation"

# Push to GitHub
git push origin main
```

**OR** if you haven't set up remote yet:

```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/DoRA-G.git
git branch -M main
git push -u origin main
```

---

### Step 2: Upload Notebooks to Google Colab

I've created 5 notebooks in `notebooks/`:
- `day1_compute_uncertainties.ipynb`
- `day2_create_oracle.ipynb` (creating next)
- `day3_train_models.ipynb` (creating next)
- `day4_evaluate.ipynb` (creating next)
- `day5_analyze_results.ipynb` (creating next)

**To use them:**

1. Go to https://colab.research.google.com
2. **File → Upload notebook**
3. Upload `notebooks/day1_compute_uncertainties.ipynb`
4. **Runtime → Change runtime type → T4 GPU** (or A100 if available)
5. Run all cells!

---

## 📅 Day-by-Day Workflow

### **Day 1: Compute Uncertainties** (1-2 hours)

1. Upload `day1_compute_uncertainties.ipynb` to Colab
2. Set runtime to GPU (T4 or A100)
3. Run all cells
4. Download `humaneval_uncertainties.csv` at the end
5. Save to Google Drive for Day 2

**What it does:**
- Loads DeepSeek-Coder-6.7B
- Computes uncertainty for 164 HumanEval problems
- Saves CSV with uncertainty scores

---

### **Day 2: Create Oracle + Analyze** (3-4 hours)

1. Upload `day2_create_oracle.ipynb` to Colab
2. Upload `humaneval_uncertainties.csv` from Day 1
3. Run all cells
4. Download oracle CSV + correlation plots

**What it does:**
- Generates WITH and WITHOUT retrieval for each problem
- Labels when retrieval helps (automatic!)
- Analyzes uncertainty vs oracle correlation
- Creates 4 key plots for paper

---

### **Day 3: Train Models** (6-8 hours)

1. Upload `day3_train_models.ipynb` to Colab
2. Run 5 training runs (one per model)
3. Save checkpoints to Google Drive

**Models:**
- Baseline (eval only, fast)
- DoRA-only (~2 hours)
- Always-RAG (~2 hours)
- Oracle-RAG (~2 hours)
- **Adaptive-RAG** (~2 hours) ⭐

**Pro tip:** Use multiple Colab sessions in parallel if you have Colab Pro+

---

### **Day 4: Evaluate** (2-3 hours)

1. Upload `day4_evaluate.ipynb` to Colab
2. Load all 5 checkpoints from Google Drive
3. Run evaluation on HumanEval
4. Download results CSV

**What it measures:**
- Pass@1 accuracy
- Retrieval rate
- Latency
- Uncertainty statistics

---

### **Day 5: Analyze & Write** (3-4 hours code + 8 hours writing)

1. Upload `day5_analyze_results.ipynb` to Colab
2. Generate all plots and tables
3. Download everything
4. Write the paper!

---

## 🔄 Handling Session Timeouts

Colab sessions timeout after 12 hours (even Pro). Here's how to handle it:

### **Save Checkpoints to Google Drive**

Add this at the start of each notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

# Set output directory to Drive
output_dir = '/content/drive/MyDrive/DoRA-G-Checkpoints/'
```

### **Resume from Checkpoint**

If your session times out during training:

```python
# In next session, load from checkpoint
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/content/drive/MyDrive/DoRA-G-Checkpoints/dora_adaptive')
```

---

## ⚡ GPU Selection Strategy

### **For Day 1 (Uncertainties)**
- **T4**: 2 hours
- **A100**: 1 hour
- Recommendation: **T4 is fine**

### **For Day 2 (Oracle)**
- **T4**: 4 hours
- **A100**: 2 hours
- Recommendation: **A100 if available** (oracle creation is slow)

### **For Day 3 (Training)**
- **T4**: 2.5 hours per model × 5 = 12-13 hours total
- **A100**: 1.5 hours per model × 5 = 7-8 hours total
- Recommendation: **A100 strongly recommended**

### **For Day 4 (Eval)**
- **T4**: 3 hours
- **A100**: 1.5 hours
- Recommendation: **T4 is fine**

---

## 💰 Cost Estimate

### **Colab Pro ($9.99/month)**

**GPU Hours Needed:**
- Day 1: 1-2 hours (T4)
- Day 2: 3-4 hours (A100 if possible)
- Day 3: 8-12 hours (A100)
- Day 4: 2-3 hours (T4)
- Day 5: 1 hour (T4)

**Total**: ~15-23 GPU hours

**Cost**: Included in $9.99/month subscription! 🎉

**vs RunPod**: Would cost $18-28 for same GPUs

---

## 📦 File Management

### **Files to Keep Between Sessions**

Upload these to Google Drive and load in each session:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy files from Drive
!cp /content/drive/MyDrive/DoRA-G-Data/humaneval_uncertainties.csv ./data/
!cp /content/drive/MyDrive/DoRA-G-Data/humaneval_oracle.csv ./data/
```

### **Files to Download After Each Day**

**Day 1:**
- `humaneval_uncertainties.csv`
- `uncertainty_distribution.png`

**Day 2:**
- `humaneval_oracle.csv`
- `uncertainty_vs_oracle_violin.png`
- `roc_curve.png`
- `decision_boundary.png`
- `improvement_vs_uncertainty.png`

**Day 3:**
- Model checkpoints (5 folders, ~5GB each)

**Day 4:**
- `main_results.csv`
- `retrieval_stats.json`

**Day 5:**
- All figures for paper
- `main_results.tex` (LaTeX table)

---

## 🚨 Common Issues & Solutions

### Issue 1: "CUDA out of memory"

**Solution:**
```python
# Use smaller model
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Instead of 6.7B

# Or use 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
```

---

### Issue 2: "Session timed out during training"

**Solution:**
```python
# Save checkpoints every 100 steps
training_args = TrainingArguments(
    ...
    save_steps=100,
    save_total_limit=3
)

# Resume from last checkpoint
trainer.train(resume_from_checkpoint=True)
```

---

### Issue 3: "Can't access GPU / GPU not available"

**Solution:**
1. Runtime → Change runtime type → GPU (T4 or A100)
2. Verify: `!nvidia-smi`
3. If no GPU available, wait 30 min and try again (Colab Pro has limits)

---

### Issue 4: "Git clone fails"

**Solution:**
```python
# If repo is private, use personal access token
!git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/DoRA-G.git

# Or make repo public temporarily
```

---

## ✅ Pre-Flight Checklist

Before starting Day 1:

- [ ] **Colab Pro subscription** active ($9.99/month)
- [ ] **Code pushed to GitHub** (all new files)
- [ ] **Google Drive** has space (~20GB for checkpoints)
- [ ] **Notebook uploaded** to Colab
- [ ] **GPU runtime** selected (T4 or A100)
- [ ] **nvidia-smi** shows GPU

Then **Run all cells!**

---

## 🎯 Success Metrics

After each day, verify:

### Day 1 ✅
- [ ] CSV has 164 rows
- [ ] Mean uncertainty between 1.5-4.0
- [ ] Files downloaded safely

### Day 2 ✅
- [ ] Oracle CSV has 164 rows
- [ ] AUC > 0.65
- [ ] 4 plots generated

### Day 3 ✅
- [ ] 5 model checkpoints saved
- [ ] Training logs look good (loss decreasing)
- [ ] Each model took ~2 hours

### Day 4 ✅
- [ ] Pass@1 results for all 5 models
- [ ] Adaptive ≥ Always-RAG
- [ ] Adaptive ≥ 95% of Oracle

### Day 5 ✅
- [ ] All plots publication-quality
- [ ] Main results table complete
- [ ] Ready to write paper!

---

## 📞 Need Help?

If you get stuck:

1. **Check the cell output** - error messages are usually clear
2. **Restart runtime** - sometimes Colab gets stuck
3. **Check GPU availability** - run `!nvidia-smi`
4. **Verify files exist** - run `!ls data/`
5. **Ask me!** - I'm here to help

---

## 🚀 Ready to Start?

### Immediate Next Steps:

1. **Right now**: Push code to GitHub
   ```bash
   git add .
   git commit -m "Add adaptive retrieval"
   git push origin main
   ```

2. **Then**: Go to https://colab.research.google.com

3. **Upload**: `notebooks/day1_compute_uncertainties.ipynb`

4. **Set GPU**: Runtime → Change runtime type → T4 GPU

5. **Run!**: Runtime → Run all (or Ctrl+F9)

---

## 🎉 You've Got This!

- ✅ No local GPU needed
- ✅ Everything runs in Colab
- ✅ Cheaper than RunPod
- ✅ Notebook interface (you prefer this!)
- ✅ Save progress to Google Drive

**Now go to GitHub, push your code, and start Day 1!**

Good luck! 🚀

# 🚀 RunPod Quickstart for Adaptive DoRA+RAG

You're running this on **RunPod** - let's get you started in 10 minutes.

---

## 🎯 Quick Setup (10 Minutes)

### **Step 1: Launch a Pod** (3 minutes)

1. Go to https://www.runpod.io/console/pods
2. Click **+ Deploy**
3. Choose GPU:
   - **Recommended**: RTX 4090 ($0.44/hr) or A40 ($0.54/hr)
   - **Best**: A100 80GB ($1.29/hr) - if you want speed
   - **Budget**: RTX 3090 ($0.34/hr) - works but slower

4. Choose template:
   - **Select**: `RunPod Pytorch 2.1` or `RunPod PyTorch`
   - Has CUDA, PyTorch pre-installed

5. Storage:
   - **Container Disk**: 50 GB (enough)
   - **Volume**: Optional (if you want persistent storage between pods)

6. **Deploy On-Demand Pod**

**Wait ~2 minutes for pod to start**

---

### **Step 2: Connect to Jupyter** (2 minutes)

Once pod is running:

1. Click **Connect** button on your pod
2. Click **Connect to Jupyter Lab** (port 8888)
3. Password is shown in pod details (or `runpod` by default)

You're now in Jupyter!

---

### **Step 3: Open Terminal in Jupyter** (1 minute)

In Jupyter Lab:
1. Click **+** (new launcher)
2. Click **Terminal** (under "Other")

You now have a terminal inside your RunPod!

---

### **Step 4: Clone Your Repo** (2 minutes)

In the terminal:

```bash
cd /workspace

# Clone your repo
git clone https://github.com/YOUR_USERNAME/DoRA-G.git

cd DoRA-G

# Verify files
ls -la
```

---

### **Step 5: Install Dependencies** (2 minutes)

```bash
pip install -q transformers accelerate bitsandbytes peft sentence-transformers faiss-cpu omegaconf hydra-core wandb scikit-learn matplotlib seaborn datasets
```

---

### **Step 6: Verify Setup** (1 minute)

```bash
python scripts/00_verify_setup.py
```

Should show:
```
✓ Python version
✓ All packages installed
✓ CUDA available
✓ GPU: [your GPU]
```

---

## 🎯 You're Ready! Start Day 1

Now run:

```bash
python scripts/01b_compute_uncertainties.py
```

**Time**: 1-2 hours (progress bar will show)

**Output**: `data/humaneval_uncertainties.csv`

---

## 📱 Alternative: Use RunPod Jupyter Notebooks

If you prefer notebooks (like you mentioned):

### **Option A: Create Notebook in Jupyter**

1. In Jupyter Lab, click **+** (new launcher)
2. Click **Python 3 (ipykernel)**
3. Copy the code from my Colab notebook
4. Run cells one by one

### **Option B: Upload My Colab Notebook**

1. Download `notebooks/day1_compute_uncertainties.ipynb` from your local machine
2. In Jupyter Lab: **File → Upload files**
3. Upload the `.ipynb` file
4. Open it and run!

**Note**: Change the first cell from:
```python
!git clone https://github.com/YOUR_USERNAME/DoRA-G.git
```
To:
```python
# Already in /workspace, just cd
import os
os.chdir('/workspace/DoRA-G')
```

---

## 🔄 File Management

### **Download Files from RunPod**

After each day, download your results:

**Method 1: Via Jupyter**
- Right-click file in Jupyter file browser
- Click **Download**

**Method 2: Via Terminal**
```bash
# In your local machine, use SCP
scp -P [SSH_PORT] root@[POD_IP]:/workspace/DoRA-G/data/humaneval_uncertainties.csv ./
```

**Method 3: Via RunPod Volume**
If you created a volume, files persist automatically!

---

## 💾 Persistent Storage Strategy

### **Use RunPod Network Volume** (Recommended)

When deploying pod:
1. **Create network volume** (costs ~$0.10/GB/month)
2. Mount to `/workspace`
3. All your data persists between pod sessions!

**Benefits:**
- Don't lose data if pod crashes
- Can stop pod overnight, restart tomorrow
- Cheaper than keeping pod running 24/7

---

## 💰 Cost Estimates

### **GPU Options:**

| GPU | $/hour | Day 1 | Day 2 | Day 3 | Day 4 | Total |
|-----|--------|-------|-------|-------|-------|-------|
| **RTX 3090** | $0.34 | $0.51 | $1.36 | $4.08 | $0.68 | **$6.63** |
| **RTX 4090** | $0.44 | $0.66 | $1.76 | $5.28 | $0.88 | **$8.58** |
| **A40** | $0.54 | $0.81 | $2.16 | $6.48 | $1.08 | **$10.53** |
| **A100 80GB** | $1.29 | $1.29 | $3.87 | $11.61 | $1.94 | **$18.71** |

**Recommendation**: RTX 4090 ($8.58 total) - best price/performance

**Storage**: Add ~$1-2 for 20GB network volume

**Total**: ~$10-20 for complete project

---

## ⏱️ Timeline on RunPod

### **With RTX 4090:**

- **Day 1**: 1.5 hours compute
- **Day 2**: 4 hours compute (oracle creation)
- **Day 3**: 12 hours compute (5 models × 2.5 hours each)
- **Day 4**: 2 hours compute (evaluation)

**Total runtime**: ~20 hours
**Total cost**: ~$8.58

### **With A100:**

- **Day 1**: 1 hour
- **Day 2**: 2 hours
- **Day 3**: 8 hours
- **Day 4**: 1.5 hours

**Total runtime**: ~12.5 hours
**Total cost**: ~$18.71

---

## 🎯 Day-by-Day RunPod Workflow

### **Day 1: Compute Uncertainties** (1.5 hours)

```bash
cd /workspace/DoRA-G
python scripts/01b_compute_uncertainties.py
```

**Output**: `data/humaneval_uncertainties.csv`

**Cost**: $0.66 (RTX 4090)

---

### **Day 2: Oracle + Analysis** (4 hours)

```bash
# Build FAISS index (optional, can use BM25)
python scripts/02_build_index.py

# Create oracle labels
python scripts/02_create_oracle.py

# Analyze correlation
python scripts/03_analyze_uncertainty_oracle.py
```

**Output**:
- `data/humaneval_oracle.csv`
- `figures/uncertainty_vs_oracle_violin.png`
- `figures/roc_curve.png`

**Cost**: $1.76 (RTX 4090)

---

### **Day 3: Train Models** (12 hours)

You can either:

**Option A: Run sequentially** (keep 1 pod running)
```bash
python scripts/03_train_baseline.py --config-name experiments/baseline eval_only=true
python scripts/03_train_baseline.py --config-name experiments/dora_only
python scripts/03_train_baseline.py --config-name experiments/dora_always_rag
python scripts/03_train_baseline.py --config-name experiments/dora_oracle
python scripts/03_train_baseline.py --config-name experiments/dora_adaptive
```

**Cost**: $5.28 (12 hours × $0.44)

**Option B: Run in parallel** (5 pods at once)
- Launch 5 pods
- Run 1 training config per pod
- All finish in ~2.5 hours
- **Cost**: Same! ($0.44 × 2.5 × 5 = $5.50)

---

### **Day 4: Evaluate** (2 hours)

```bash
python scripts/04_run_ablations.py
python scripts/05_analyze_results.py
```

**Output**:
- `results/main_results.csv`
- All figures for paper

**Cost**: $0.88 (RTX 4090)

---

## 🚨 RunPod-Specific Tips

### **1. Keep Your Pod Running**

If you need to leave but want to keep training:
- Just close your browser
- Pod keeps running
- Come back later and reconnect

### **2. Monitor Training**

Use W&B:
```bash
wandb login
# Paste your API key
```

Then monitor from anywhere: https://wandb.ai

### **3. Stop Pod Overnight**

If using network volume:
```bash
# Save everything
git add .
git commit -m "checkpoint"
git push

# Then stop pod in RunPod console
# Restart tomorrow, your volume persists!
```

### **4. SSH Access**

For terminal access without Jupyter:
```bash
# Get SSH command from pod details
ssh root@[POD_IP] -p [SSH_PORT]
```

---

## 📥 Downloading Results

### **Method 1: Jupyter File Browser**
1. Navigate to file in Jupyter
2. Right-click → Download

### **Method 2: RunPod File Manager**
1. Click **Connect** on pod
2. Click **HTTP File Server**
3. Browse and download files

### **Method 3: SCP (from your machine)**
```bash
# Get SSH details from pod
scp -P [PORT] root@[IP]:/workspace/DoRA-G/data/*.csv ./local_folder/
```

### **Method 4: Git Push**
```bash
# Commit results to git (if not too large)
git add data/ figures/ results/
git commit -m "Add results"
git push
```

---

## ✅ Pre-Flight Checklist

Before starting:

- [ ] **RunPod account** with credits
- [ ] **Pod launched** (RTX 4090 or better)
- [ ] **Jupyter Lab** accessible
- [ ] **Terminal** opened in Jupyter
- [ ] **Repo cloned** to `/workspace/DoRA-G`
- [ ] **Dependencies installed**
- [ ] **GPU verified** (`nvidia-smi` shows GPU)

Then **start Day 1!**

---

## 🎯 Quick Commands Reference

```bash
# Check GPU
nvidia-smi

# Navigate to project
cd /workspace/DoRA-G

# Pull latest code
git pull

# Run Day 1
python scripts/01b_compute_uncertainties.py

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check progress
tail -f logs/training.log  # If using logging

# Download a file (from local machine)
scp -P [PORT] root@[IP]:/workspace/DoRA-G/data/humaneval_uncertainties.csv ./
```

---

## 🚀 Ready? Start Now!

### **Right This Minute:**

1. **Go to RunPod**: https://www.runpod.io/console/pods
2. **Deploy Pod**: RTX 4090 or A40
3. **Connect to Jupyter**
4. **Open Terminal**
5. **Run:**

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/DoRA-G.git
cd DoRA-G
pip install -q transformers accelerate bitsandbytes peft sentence-transformers faiss-cpu omegaconf hydra-core wandb scikit-learn matplotlib seaborn datasets
python scripts/01b_compute_uncertainties.py
```

**That's it! Day 1 is running!**

---

## 📞 Need Help?

Common issues:

**"Git clone fails"**
```bash
# Make sure repo is public, or use token
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/DoRA-G.git
```

**"Out of memory"**
```bash
# Use smaller model
# Edit configs/base.yaml
model:
  name_or_path: "deepseek-ai/deepseek-coder-1.3b-instruct"
```

**"Pod stopped unexpectedly"**
- Check your RunPod credits/balance
- Restart pod and resume

---

**GO! Launch your RunPod now!** 🚀

Cost: ~$8-9 for entire project on RTX 4090

Much better than Colab for your use case!

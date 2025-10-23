# RunPod Deployment Checklist

Use this checklist to ensure smooth deployment and training on RunPod.

## Pre-Deployment Checklist

### ☐ 1. Get Your W&B API Key
- [ ] Go to https://wandb.ai/settings
- [ ] Copy your API key
- [ ] Keep it handy for deployment

### ☐ 2. Push Code to GitHub/GitLab
- [ ] Commit all changes
- [ ] Push to your repository
- [ ] Verify all files are uploaded (especially new RunPod files)

### ☐ 3. Create RunPod Account
- [ ] Sign up at https://runpod.io
- [ ] Add billing information
- [ ] Fund account with ~$15 (for ~8 hours + buffer)

### ☐ 4. Optional: Create Network Volume (Recommended)
- [ ] Create 100GB+ network volume
- [ ] Name it: `dora-g-storage`
- [ ] This preserves data between pod restarts

---

## Deployment Checklist

### ☐ 1. Deploy RunPod Pod
- [ ] Select GPU: **A100 PCIe 80GB**
- [ ] Choose: **Community Cloud** ($1.19/hr) or Secure Cloud ($1.64/hr)
- [ ] Template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- [ ] Container Disk: **100GB minimum**
- [ ] Mount network volume to `/workspace` (if created)
- [ ] Expose ports: 8888 (for Jupyter, optional)

### ☐ 2. Set Environment Variables
In RunPod pod settings, add:
- [ ] `WANDB_API_KEY=your_key_here` (REQUIRED)
- [ ] `WANDB_PROJECT=dora-rag-code-generation` (optional)

### ☐ 3. Start Pod
- [ ] Click "Deploy"
- [ ] Wait for pod to start (~1-2 minutes)
- [ ] Note the pod IP address

### ☐ 4. Connect to Pod
- [ ] Click "Connect" → "Start Web Terminal" or SSH
- [ ] Verify you see a shell prompt

---

## Setup Checklist

### ☐ 1. Clone Repository
```bash
cd /workspace
git clone <your-repo-url> DoRA-G
cd DoRA-G
```
- [ ] Repository cloned successfully
- [ ] Current directory is `/workspace/DoRA-G`

### ☐ 2. Run Quick Setup
```bash
chmod +x runpod_quickstart.sh
./runpod_quickstart.sh
```
- [ ] Script completes without errors
- [ ] W&B API key accepted
- [ ] GPU detected successfully

### ☐ 3. Verify Setup
Check script output:
- [ ] ✓ PyTorch installed
- [ ] ✓ faiss-gpu installed
- [ ] ✓ Dependencies installed
- [ ] ✓ Environment configured
- [ ] ✓ W&B configured
- [ ] ✓ GPU verified
- [ ] ✓ Setup verification passed

---

## Training Checklist

### ☐ 1. Start Training Pipeline
```bash
chmod +x runpod_entrypoint.sh
./runpod_entrypoint.sh
```
- [ ] Script starts without errors
- [ ] GPU memory check shows available VRAM

### ☐ 2. Monitor Initial Steps
- [ ] Dataset preparation completes (~15 min)
- [ ] RAG index building starts
- [ ] No out-of-memory errors

### ☐ 3. Set Up Monitoring
Open in separate terminals/tabs:

**Terminal 1: GPU Monitor**
```bash
watch -n 1 nvidia-smi
```
- [ ] GPU utilization shows ~90-100% during training
- [ ] Memory usage ~35-45GB during training
- [ ] Temperature reasonable (<85°C)

**Terminal 2: W&B Dashboard**
- [ ] Go to https://wandb.ai
- [ ] Navigate to project: `dora-rag-code-generation`
- [ ] See training runs appearing
- [ ] Metrics updating in real-time

**Terminal 3: Logs** (optional)
```bash
tail -f /workspace/logs/*.log
```

### ☐ 4. Verify Training Progress
After 1 hour, check:
- [ ] RAG index built successfully
- [ ] Baseline training started
- [ ] Loss decreasing
- [ ] No errors in logs
- [ ] Checkpoints being saved to `/workspace/outputs/`

---

## During Training Checklist

### ☐ Monitor Every 2 Hours
- [ ] Check W&B for training curves
- [ ] Verify GPU still running (nvidia-smi)
- [ ] Check disk space: `df -h` (should have 20GB+ free)
- [ ] Ensure pod is still running (RunPod dashboard)

### ☐ Troubleshooting (If Needed)
If you see errors:

**Out of Memory:**
- [ ] Stop training (Ctrl+C)
- [ ] Reduce batch size (see RUNPOD_SETUP.md)
- [ ] Restart training

**W&B Errors:**
- [ ] Check: `wandb status`
- [ ] Re-login: `wandb login --relogin $WANDB_API_KEY`
- [ ] Or continue offline: `export WANDB_MODE=offline`

**Disk Full:**
- [ ] Check space: `df -h`
- [ ] Clear cache if needed: `rm -rf /workspace/cache/`
- [ ] Ensure network volume is mounted

---

## Post-Training Checklist

### ☐ 1. Verify Completion
Check that pipeline completed:
- [ ] Baseline training finished
- [ ] LoRA training finished
- [ ] DoRA training finished
- [ ] DoRA+RAG training finished
- [ ] All evaluations completed
- [ ] Analysis generated

### ☐ 2. Verify Results Exist
```bash
ls -lh /workspace/outputs/
ls -lh /workspace/checkpoints/
ls -lh /workspace/outputs/analysis/
```
- [ ] All model directories exist
- [ ] Checkpoint files present (*.bin or *.safetensors)
- [ ] Analysis plots generated (*.png)
- [ ] Evaluation results saved (*.json)

### ☐ 3. Check Result Quality
- [ ] W&B shows reasonable metrics (pass@1 > 0, not NaN)
- [ ] Training loss decreased over time
- [ ] Evaluation completed on all benchmarks
- [ ] Model files are non-zero size

---

## Data Retrieval Checklist

### ☐ 1. Download via RunPod File Browser
- [ ] Go to RunPod dashboard
- [ ] Click "Connect" → "File Browser"
- [ ] Download `/workspace/outputs/` (models and results)
- [ ] Download `/workspace/checkpoints/` (training checkpoints)
- [ ] Download `/workspace/outputs/analysis/` (plots)

### ☐ 2. Verify Downloads
On your local machine:
- [ ] All folders downloaded successfully
- [ ] Files are not corrupted (can open them)
- [ ] Model weights are present
- [ ] Plots are viewable

### ☐ 3. W&B Backup (Automatic)
- [ ] Check W&B dashboard has all runs
- [ ] Verify artifacts are uploaded
- [ ] Download any additional logs if needed

---

## Cleanup Checklist

### ☐ 1. Stop the Pod
- [ ] Go to RunPod dashboard
- [ ] Click "Stop" on your pod
- [ ] Confirm billing has stopped

### ☐ 2. Preserve Important Data
If you didn't use a network volume:
- [ ] Ensure all results are downloaded
- [ ] Verify backups in W&B
- [ ] Save FAISS index if you plan to rerun (optional)

If you used a network volume:
- [ ] Data is automatically preserved
- [ ] You can reuse FAISS index next time
- [ ] Can restart pod later without rebuilding

### ☐ 3. Cost Verification
- [ ] Check RunPod billing dashboard
- [ ] Verify total cost (~$9-10 for full pipeline)
- [ ] Ensure no idle charges

---

## Paper Writing Checklist

### ☐ 1. Organize Results
- [ ] Create results folder locally
- [ ] Organize models by experiment
- [ ] Create tables from evaluation metrics
- [ ] Select best plots for paper

### ☐ 2. Extract Key Metrics
From W&B or evaluation results:
- [ ] HumanEval pass@1 and pass@10 for all models
- [ ] MBPP pass@1 and pass@10 for all models
- [ ] DS-1000 scores for all models
- [ ] Training time for each model
- [ ] Parameter counts (already documented)

### ☐ 3. Create Paper Figures
- [ ] Comparison table (Baseline vs LoRA vs DoRA vs DoRA+RAG)
- [ ] Training curve plot
- [ ] Performance by benchmark plot
- [ ] RAG impact ablation chart

### ☐ 4. Document Reproducibility
Include in paper appendix:
- [ ] Hardware: A100 80GB PCIe
- [ ] Training time: ~6-8 hours
- [ ] Hyperparameters: From configs/base.yaml
- [ ] Code repository: GitHub link
- [ ] Docker image: Available for reproduction

---

## Estimated Timeline

| Phase | Duration | Cost @ $1.19/hr |
|-------|----------|-----------------|
| Setup | 5 min | $0.10 |
| Dataset prep | 15 min | $0.30 |
| RAG index build | 45 min | $0.89 |
| Baseline training | 30 min | $0.60 |
| LoRA training | 2 hr | $2.38 |
| DoRA training | 2 hr | $2.38 |
| DoRA+RAG training | 2 hr | $2.38 |
| Evaluation | 1 hr | $1.19 |
| Analysis | 10 min | $0.20 |
| **Total** | **~8 hr** | **~$9.50** |

---

## Emergency Contacts & Resources

- **RunPod Support**: support@runpod.io
- **RunPod Discord**: https://discord.gg/runpod
- **RunPod Docs**: https://docs.runpod.io
- **W&B Support**: support@wandb.com
- **Project Issues**: GitHub repository issues page

---

## Quick Reference Commands

### Check GPU
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Check Disk Space
```bash
df -h /workspace
```

### Monitor Training
```bash
watch -n 1 nvidia-smi
tail -f /workspace/logs/training.log
```

### Check Processes
```bash
ps aux | grep python
```

### Kill Stuck Process
```bash
pkill -f python
```

### Restart Training
```bash
cd /workspace/DoRA-G
./runpod_entrypoint.sh
```

---

## Success Criteria

You're done when:
- ✅ All 4 models trained successfully
- ✅ All benchmarks evaluated
- ✅ Results downloaded to local machine
- ✅ W&B has all experiment logs
- ✅ Analysis plots generated
- ✅ Pod stopped (no idle charges)
- ✅ Ready to write paper!

---

**Print this checklist and check off items as you go!** ✓

Good luck with your training! 🚀

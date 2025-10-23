# DoRA-G RunPod Deployment Dockerfile
# Optimized for A100 PCIe with CUDA 11.8

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create necessary directories for persistent storage
RUN mkdir -p /workspace/data \
    /workspace/outputs \
    /workspace/cache \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/models_cache

# Copy requirements first for better layer caching
COPY requirements.txt /workspace/

# Install Python dependencies
# Note: We'll install faiss-gpu instead of faiss-cpu for better performance
RUN pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install faiss-gpu separately
RUN pip install faiss-gpu

# Install remaining dependencies (excluding faiss-cpu)
RUN grep -v "faiss-cpu" requirements.txt > requirements_gpu.txt && \
    pip install -r requirements_gpu.txt && \
    rm requirements_gpu.txt

# Install additional dependencies for code execution
RUN pip install docker-py

# Copy the entire project
COPY . /workspace/

# Set environment variables for reproducibility and paths
ENV PYTHONPATH=/workspace
ENV PYTHONHASHSEED=0
ENV HF_HOME=/workspace/cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers
ENV TORCH_HOME=/workspace/cache/torch
ENV WANDB_DIR=/workspace/logs/wandb

# Environment variables for GPU optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set proper permissions
RUN chmod +x /workspace/runpod_entrypoint.sh 2>/dev/null || true

# Expose port for Jupyter (optional)
EXPOSE 8888

# Default command - run the entrypoint script
CMD ["/bin/bash", "-c", "if [ -f /workspace/runpod_entrypoint.sh ]; then /workspace/runpod_entrypoint.sh; else /bin/bash; fi"]

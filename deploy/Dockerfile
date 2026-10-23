# ==============================================================================
# deploy/Dockerfile
# Production Container for uTTT on NVIDIA RTX 5090 (32GB GDDR7, Blackwell sm_120)
# Stack: CUDA 12.8+ | PyTorch 2.6+ | Triton | FastAPI + WebSockets
# ==============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

LABEL maintainer="uTTT Systems Architecture Team"
LABEL description="Universal Test-Time Training (uTTT) Training & Personalization Engine for RTX 5090 (sm_120)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Target NVIDIA Blackwell RTX 5090 compute capability 12.0 (sm_120)
    TORCH_CUDA_ARCH_LIST="12.0" \
    CUDA_HOME="/usr/local/cuda" \
    # Prevent VRAM fragmentation during combined inner+outer loop meta-training
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
    TORCHINDUCTOR_CACHE_DIR="/workspace/uTTT/.cache/inductor" \
    TRITON_CACHE_DIR="/workspace/uTTT/.cache/triton"

# 1. Install system & build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ninja-build \
    git \
    curl \
    wget \
    rsync \
    tmux \
    htop \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Create isolated virtual environment
RUN python3 -m venv /opt/uttt_venv
ENV PATH="/opt/uttt_venv/bin:${PATH}"

# 3. Install PyTorch 2.6+ built for CUDA 12.8 (Blackwell sm_120 native BF16/FP8 support)
RUN pip install --upgrade pip setuptools wheel packaging ninja && \
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.6.0" \
        "torchvision" \
        "torchaudio" || \
    pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
        torch torchvision torchaudio

# 4. Install Triton & optional FlashAttention (graceful fallback to PyTorch 2.6 cuDNN SDPA on sm_120)
RUN pip install "triton>=3.2.0" && \
    (MAX_JOBS=4 pip install flash-attn --no-build-isolation || \
     echo "[Dockerfile] Using PyTorch 2.6 native SDPA / cuDNN FlashAttention for sm_120")

# 5. Install uTTT application dependencies
WORKDIR /workspace/uTTT
COPY requirements.txt /workspace/uTTT/requirements.txt
RUN pip install -r /workspace/uTTT/requirements.txt

# 6. Copy source code & create runtime directories
COPY . /workspace/uTTT
RUN chmod +x /workspace/uTTT/deploy/*.sh && \
    mkdir -p /workspace/uTTT/checkpoints \
             /workspace/uTTT/personas/snapshots \
             /workspace/uTTT/logs \
             /workspace/uTTT/.cache

EXPOSE 8000

# Default entrypoint runs the FastAPI + WebSocket Personalization Server
CMD ["python", "-m", "server.app", "--host", "0.0.0.0", "--port", "8000", "--config", "configs/5090_config.yaml"]

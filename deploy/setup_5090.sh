#!/usr/bin/env bash
# ==============================================================================
# deploy/setup_5090.sh
# Automated Remote Environment Provisioning for NVIDIA RTX 5090 (Blackwell sm_120)
# Verifies GPU driver, CUDA 12.8+, installs PyTorch 2.6+ (cu128) & Triton, and
# runs an on-device analytical SwiGLU TTT kernel verification check.
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

echo "=============================================================================="
echo " [uTTT Setup] Initializing NVIDIA RTX 5090 (Blackwell sm_120) Environment"
echo "=============================================================================="

# 1. Verify NVIDIA GPU & Driver
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. Please install NVIDIA Driver >= 570.xx for RTX 5090." >&2
    exit 1
fi

echo "[1/5] Detecting GPU Hardware & Driver..."
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"

if [[ "${GPU_NAME}" == *"5090"* ]] || [[ "${COMPUTE_CAP}" == "12.0" ]]; then
    echo "[OK] Detected Blackwell GPU: ${GPU_NAME} (Compute Capability ${COMPUTE_CAP} / sm_120)"
else
    echo "[WARN] Detected GPU '${GPU_NAME}' (Compute Capability ${COMPUTE_CAP}). Proceeding with compatible CUDA settings."
fi

# 2. Export Blackwell sm_120 build & allocator flags
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 3. Create or activate Python Virtual Environment
echo "[2/5] Preparing Python virtual environment at ${VENV_DIR}..."
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel packaging ninja

# 4. Install PyTorch 2.6+ (CUDA 12.8) & Application Requirements
echo "[3/5] Installing PyTorch 2.6+ (cu128) and Triton for Blackwell sm_120..."
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.6.0" torchvision torchaudio || \
pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision torchaudio

pip install "triton>=3.2.0" || true
pip install -r "${PROJECT_ROOT}/requirements.txt"

# 5. Prepare runtime directories & run hardware smoke test
echo "[4/5] Creating checkpoint and persona directories..."
mkdir -p "${PROJECT_ROOT}/checkpoints" \
         "${PROJECT_ROOT}/personas/snapshots" \
         "${PROJECT_ROOT}/logs"

echo "[5/5] Running analytical SwiGLU Fast-Weight Kernel verification..."
cd "${PROJECT_ROOT}"
python3 - <<'PYEOF'
import torch
from model.uttt_layer import SharedFastWeightPool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
pool = SharedFastWeightPool(d_model=256, d_ff=512).to(device=device, dtype=dtype)
state = pool.init_state(batch_size=2, device=device, dtype=dtype, detach=False)

q = torch.randn(2, 64, 256, device=device, dtype=dtype)
k = torch.randn(2, 64, 256, device=device, dtype=dtype)
v = torch.randn(2, 64, 256, device=device, dtype=dtype)
eta = torch.full((2, 64, 1), 0.01, device=device, dtype=dtype)
alpha = torch.full((2, 64, 1), 0.005, device=device, dtype=dtype)

out, next_state, s_loss = pool.chunk_step(state, q, k, v, eta, alpha, update_fast_weights=True)
drift = pool.compute_drift(next_state)
print(
    f"[SmokeTest PASS] Device={device} ({dtype}) | "
    f"Output={tuple(out.shape)} | SurpriseLoss={s_loss.item():.4f} | "
    f"FastWeightDrift={drift['drift_pct']:.3f}%"
)
PYEOF

echo "=============================================================================="
echo " [uTTT Setup] Environment ready! Activate with: source ${VENV_DIR}/bin/activate"
echo "=============================================================================="

# ==============================================================================
# deploy/setup_5090.ps1
# Native Windows 10/11 Pro Environment Provisioning for NVIDIA RTX 5090 (sm_120)
# Verifies GPU driver, creates Python venv, installs PyTorch 2.6+ (cu128), and
# runs an on-GPU analytical SwiGLU Fast-Weight kernel smoke test.
# ==============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { Join-Path $ProjectRoot ".venv" }

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " [uTTT Setup - Windows] Initializing NVIDIA RTX 5090 (Blackwell sm_120)" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

# 1. Verify NVIDIA GPU & Driver
if (-not (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
    Write-Error "[ERROR] nvidia-smi not found. Please install NVIDIA Windows Driver >= 570.xx for RTX 5090."
    exit 1
}

Write-Host "[1/5] Detecting GPU Hardware & Driver..." -ForegroundColor Yellow
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader

# 2. Configure Blackwell sm_120 environment variables
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"

# 3. Create or activate Python Virtual Environment
Write-Host "[2/5] Preparing Python virtual environment at $VenvDir..." -ForegroundColor Yellow
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    python -m venv $VenvDir
}

$PyExe = Join-Path $VenvDir "Scripts\python.exe"
& $PyExe -m pip install --upgrade pip setuptools wheel packaging

# 4. Install PyTorch 2.6+ (CUDA 12.8) & Requirements
Write-Host "[3/5] Installing PyTorch 2.6+ (cu128) for Windows RTX 5090..." -ForegroundColor Yellow
& $PyExe -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.6.0" torchvision torchaudio
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Stable cu128 wheel not matched; trying PyTorch nightly cu128..." -ForegroundColor Yellow
    & $PyExe -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio
}

& $PyExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

# 5. Prepare runtime directories & run smoke test
Write-Host "[4/5] Creating checkpoint, persona, and log directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "checkpoints") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "personas\snapshots") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null

Write-Host "[5/5] Running analytical SwiGLU Fast-Weight Kernel verification..." -ForegroundColor Yellow
Push-Location $ProjectRoot
try {
    & $PyExe -c @"
import torch
from model.uttt_layer import SharedFastWeightPool
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
pool = SharedFastWeightPool(d_model=256, d_ff=512).to(device=device, dtype=dtype)
state = pool.init_state(batch_size=2, device=device, dtype=dtype, detach=False)
q = torch.randn(2, 64, 256, device=device, dtype=dtype)
k = torch.randn(2, 64, 256, device=device, dtype=dtype)
v = torch.randn(2, 64, 256, device=device, dtype=dtype)
eta = torch.full((2, 64, 1), 0.01, device=device, dtype=dtype)
alpha = torch.full((2, 64, 1), 0.005, device=device, dtype=dtype)
out, next_state, s_loss = pool.chunk_step(state, q, k, v, eta, alpha, update_fast_weights=True)
drift = pool.compute_drift(next_state)
print(f'[SmokeTest PASS] Device={device} ({dtype}) | Output={tuple(out.shape)} | SurpriseLoss={s_loss.item():.4f} | FastWeightDrift={drift[\"drift_pct\"]:.3f}%')
"@
} finally {
    Pop-Location
}

Write-Host "==============================================================================" -ForegroundColor Green
Write-Host " [uTTT Setup - Windows] Environment ready! Python: $PyExe" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green

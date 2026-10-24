#!/usr/bin/env bash
# ==============================================================================
# deploy/remote_ctl.sh
# One-Click Remote Deployment, Training & Personalization Service Controller
# Target: Remote NVIDIA RTX 5090 (32GB GDDR7, Blackwell sm_120) Server
#
# Supported Environment Variables:
#   REMOTE_HOST   - Remote server IP or hostname (e.g. "192.168.1.50" or "gpu5090.example.com")
#   REMOTE_PORT   - Remote SSH port (default: "22")
#   REMOTE_USER   - Remote SSH username (default: "root")
#   REMOTE_DIR    - Remote workspace path (default: "/workspace/uTTT")
#   SSH_KEY       - Path to SSH private key (default: "${HOME}/.ssh/id_ed25519")
#
# Core Subcommands:
#   ./deploy/remote_ctl.sh init-env   # Detect RTX 5090 driver & provision remote environment
#   ./deploy/remote_ctl.sh sync       # High-speed rsync excluding checkpoints/ and .git/
#   ./deploy/remote_ctl.sh train      # Launch tmux background training & stream live loss
#   ./deploy/remote_ctl.sh pull-ckpt  # Pull trained checkpoints/base_uttt.pt to local machine
#   ./deploy/remote_ctl.sh serve      # Launch remote FastAPI+WS server & tunnel to localhost:8000
#   ./deploy/remote_ctl.sh status     # Inspect remote GPU VRAM, tmux sessions, and checkpoints
#   ./deploy/remote_ctl.sh stop       # Stop remote training or serving tmux sessions
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load optional local .env file if present
if [[ -f "${LOCAL_ROOT}/.env" ]]; then
    # shellcheck disable=SC1090
    source "${LOCAL_ROOT}/.env"
fi

# ------------------------------------------------------------------------------
# 1. Environment Variable Configuration & Defaults
# ------------------------------------------------------------------------------
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/uTTT}"
# Expand leading ~/ to /root/ for remote root shell compatibility
REMOTE_DIR="${REMOTE_DIR/#\~\///root/}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"

LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-8000}"
REMOTE_SERVICE_PORT="${REMOTE_SERVICE_PORT:-8000}"
TMUX_TRAIN_SESSION="${TMUX_TRAIN_SESSION:-uttt_train}"
TMUX_SERVE_SESSION="${TMUX_SERVE_SESSION:-uttt_serve}"
USE_DOCKER="${USE_DOCKER:-0}"
DOCKER_IMAGE="${DOCKER_IMAGE:-uttt-rtx5090:latest}"

# ------------------------------------------------------------------------------
# 2. Logging & SSH Helper Functions
# ------------------------------------------------------------------------------
log_info()  { printf "\033[1;34m[uTTT-Remote]\033[0m %s\n" "$*"; }
log_ok()    { printf "\033[1;32m[uTTT-OK]\033[0m     %s\n" "$*"; }
log_warn()  { printf "\033[1;33m[uTTT-WARN]\033[0m   %s\n" "$*"; }
log_error() { printf "\033[1;31m[uTTT-ERROR]\033[0m  %s\n" "$*" >&2; }

usage() {
    cat <<EOF
Usage:
  REMOTE_HOST=<ip> [REMOTE_PORT=22] [REMOTE_USER=root] [REMOTE_DIR=/workspace/uTTT] [SSH_KEY=~/.ssh/id_rsa] \\
    ./deploy/remote_ctl.sh <command> [args...]

Core Commands:
  init-env              Auto-detect remote RTX 5090 driver/CUDA and build environment
  sync                  Incremental rsync of local workspace (excludes checkpoints/ & .git/)
  prepare-data [args]   Download & format public Persona-Chat + Belle multi-turn dataset on remote
  train [extra_args]    Start train_5090.py in remote tmux session and stream live loss
  pull-ckpt             Pull remote checkpoints/base_uttt.pt to local checkpoints/base_uttt.pt
  serve                 Launch remote FastAPI+WebSocket server & map SSH tunnel to localhost:${LOCAL_TUNNEL_PORT}
  status                Show remote RTX 5090 GPU utilization, tmux sessions, and checkpoint info
  logs [train|serve]    Attach to real-time remote log stream
  stop [train|serve|all] Terminate remote background training/serving sessions

Environment Variables (Current Effective Values):
  REMOTE_HOST = ${REMOTE_HOST:-<REQUIRED>}
  REMOTE_PORT = ${REMOTE_PORT}
  REMOTE_USER = ${REMOTE_USER}
  REMOTE_DIR  = ${REMOTE_DIR}
  SSH_KEY     = ${SSH_KEY}
  USE_DOCKER  = ${USE_DOCKER} (set USE_DOCKER=1 to build & run via deploy/Dockerfile)
EOF
}

validate_config() {
    if [[ -z "${REMOTE_HOST}" ]]; then
        log_error "REMOTE_HOST is not set. Please export REMOTE_HOST=<server_ip> before running."
        usage
        exit 1
    fi

    SSH_OPTS=(
        -p "${REMOTE_PORT}"
        -o StrictHostKeyChecking=accept-new
        -o ConnectTimeout=10
        -o ServerAliveInterval=30
        -o ServerAliveCountMax=3
    )
    if [[ -n "${SSH_KEY}" ]] && [[ -f "${SSH_KEY}" ]]; then
        # If SSH_KEY resides on a Windows /mnt/* mount with 0777 permissions, stage to ~/.ssh with 0600
        if [[ "${SSH_KEY}" == /mnt/* ]]; then
            mkdir -p "${HOME}/.ssh" && chmod 700 "${HOME}/.ssh"
            SAFE_KEY="${HOME}/.ssh/$(basename "${SSH_KEY}")"
            cp -f "${SSH_KEY}" "${SAFE_KEY}"
            chmod 600 "${SAFE_KEY}"
            SSH_KEY="${SAFE_KEY}"
        else
            chmod 600 "${SSH_KEY}" 2>/dev/null || true
        fi
        SSH_OPTS+=(-i "${SSH_KEY}")
    elif [[ -n "${SSH_KEY}" ]]; then
        log_warn "SSH_KEY '${SSH_KEY}' not found on disk; relying on ssh-agent / default keys."
    fi

    REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
}

remote_exec() {
    ssh "${SSH_OPTS[@]}" "${REMOTE_TARGET}" "$@"
}

remote_exec_tty() {
    ssh -t "${SSH_OPTS[@]}" "${REMOTE_TARGET}" "$@"
}

# ------------------------------------------------------------------------------
# 3. Subcommand: sync (High-Speed Incremental Rsync)
# ------------------------------------------------------------------------------
cmd_sync() {
    validate_config
    log_info "Syncing local workspace '${LOCAL_ROOT}' -> '${REMOTE_TARGET}:${REMOTE_DIR}'..."

    remote_exec "mkdir -p '${REMOTE_DIR}/checkpoints' '${REMOTE_DIR}/personas/snapshots' '${REMOTE_DIR}/logs' '${REMOTE_DIR}/data'"

    RSYNC_SSH_CMD="ssh ${SSH_OPTS[*]}"
    rsync -avz --progress --delete \
        -e "${RSYNC_SSH_CMD}" \
        --exclude=".git/" \
        --exclude=".env" \
        --exclude="checkpoints/" \
        --exclude="personas/" \
        --exclude="data/" \
        --exclude="logs/" \
        --exclude=".venv/" \
        --exclude="venv/" \
        --exclude="__pycache__/" \
        --exclude="*.pyc" \
        --exclude=".pytest_cache/" \
        --exclude=".cache/" \
        "${LOCAL_ROOT}/" "${REMOTE_TARGET}:${REMOTE_DIR}/"

    remote_exec "chmod +x '${REMOTE_DIR}/deploy/'*.sh"
    log_ok "Codebase synchronized successfully (excluded checkpoints/ and .git/)."
}

# ------------------------------------------------------------------------------
# 4. Subcommand: init-env (Remote RTX 5090 Detection & Environment Build)
# ------------------------------------------------------------------------------
cmd_init_env() {
    validate_config
    log_info "Probing remote hardware & driver on ${REMOTE_TARGET}..."

    remote_exec "bash -s" <<'REMOTEEOF'
set -euo pipefail
echo "------------------------------------------------------------------------------"
echo " Remote Hostname : $(hostname)"
echo " OS / Kernel     : $(uname -srm)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found on remote host! Install NVIDIA driver >= 570.xx." >&2
    exit 1
fi
echo " GPU Details     :"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
echo "------------------------------------------------------------------------------"
REMOTEEOF

    # Sync latest codebase before building environment
    cmd_sync

    if [[ "${USE_DOCKER}" == "1" ]]; then
        log_info "Building Docker image '${DOCKER_IMAGE}' on remote RTX 5090 host..."
        remote_exec "cd '${REMOTE_DIR}' && docker build -f deploy/Dockerfile -t '${DOCKER_IMAGE}' ."
        log_ok "Docker image '${DOCKER_IMAGE}' built successfully on remote host."
    else
        log_info "Running deploy/setup_5090.sh on remote host..."
        remote_exec "cd '${REMOTE_DIR}' && bash deploy/setup_5090.sh"
        log_ok "Remote bare-metal virtualenv provisioned and verified for RTX 5090."
    fi
}

# ------------------------------------------------------------------------------
# 4.5 Subcommand: prepare-data (Download Public Persona Dataset on Remote Host)
# ------------------------------------------------------------------------------
cmd_prepare_data() {
    validate_config
    cmd_sync
    EXTRA_DATA_ARGS="$*"
    log_info "Downloading & formatting public Persona + Multi-Turn dataset on ${REMOTE_TARGET}..."
    remote_exec "cd '${REMOTE_DIR}' && python3 deploy/prepare_dataset.py ${EXTRA_DATA_ARGS}"
    log_ok "Remote public dataset prepared at ${REMOTE_DIR}/data/uttt_persona_train.jsonl"
}

# ------------------------------------------------------------------------------
# 5. Subcommand: train (Remote tmux Background Meta-Training + Live Loss Stream)
# ------------------------------------------------------------------------------
cmd_train() {
    validate_config
    cmd_sync

    EXTRA_TRAIN_ARGS="$*"
    LOG_FILE="${REMOTE_DIR}/logs/train_5090.log"

    log_info "Launching uTTT Meta-Training on ${REMOTE_TARGET} (session: ${TMUX_TRAIN_SESSION})..."

    if [[ "${USE_DOCKER}" == "1" ]]; then
        TRAIN_CMD="docker run --rm --gpus all --ipc=host -v '${REMOTE_DIR}/checkpoints:/workspace/uTTT/checkpoints' -v '${REMOTE_DIR}/logs:/workspace/uTTT/logs' ${DOCKER_IMAGE} python -u train_5090.py --config configs/5090_config.yaml ${EXTRA_TRAIN_ARGS} 2>&1 | tee '${LOG_FILE}'"
    else
        TRAIN_CMD="cd '${REMOTE_DIR}' && source .venv/bin/activate && export TORCH_CUDA_ARCH_LIST='12.0' && export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512' && '${REMOTE_DIR}/.venv/bin/python3' -u train_5090.py --config configs/5090_config.yaml ${EXTRA_TRAIN_ARGS} 2>&1 | tee '${LOG_FILE}'"
    fi

    remote_exec "bash -c \"
        mkdir -p '${REMOTE_DIR}/logs' '${REMOTE_DIR}/checkpoints';
        : > '${LOG_FILE}';
        if command -v tmux >/dev/null 2>&1; then
            tmux has-session -t '${TMUX_TRAIN_SESSION}' 2>/dev/null && tmux kill-session -t '${TMUX_TRAIN_SESSION}';
            tmux new-session -d -s '${TMUX_TRAIN_SESSION}' \\\"${TRAIN_CMD}\\\";
            echo '[Remote] Started tmux session: ${TMUX_TRAIN_SESSION}';
        else
            nohup bash -c \\\"${TRAIN_CMD}\\\" >/dev/null 2>&1 &
            echo '[Remote] tmux not found; started background training via nohup (PID: '\$!')';
        fi
    \""

    log_ok "Training started in background on remote RTX 5090."
    log_info "Streaming real-time training loss from ${LOG_FILE} (Press Ctrl+C to detach local log view; remote training will continue)..."
    remote_exec "bash -c \"
        sleep 1;
        tail -n +1 -f '${LOG_FILE}' &
        TAIL_PID=\\\$!;
        while tmux has-session -t '${TMUX_TRAIN_SESSION}' 2>/dev/null; do
            sleep 1;
        done;
        sleep 1;
        kill \\\${TAIL_PID} 2>/dev/null || true;
        echo '[Remote] Training session (${TMUX_TRAIN_SESSION}) finished.';
    \"" || true
}

# ------------------------------------------------------------------------------
# 6. Subcommand: pull-ckpt (Pull Trained base_uttt.pt Back to Local Workspace)
# ------------------------------------------------------------------------------
cmd_pull_ckpt() {
    validate_config
    REMOTE_CKPT="${REMOTE_DIR}/checkpoints/base_uttt.pt"
    LOCAL_CKPT_DIR="${LOCAL_ROOT}/checkpoints"
    LOCAL_CKPT="${LOCAL_CKPT_DIR}/base_uttt.pt"

    mkdir -p "${LOCAL_CKPT_DIR}"
    log_info "Checking remote checkpoint at ${REMOTE_TARGET}:${REMOTE_CKPT}..."

    if ! remote_exec "test -f '${REMOTE_CKPT}'"; then
        log_error "Remote checkpoint '${REMOTE_CKPT}' does not exist yet. Run './deploy/remote_ctl.sh train' first."
        exit 1
    fi

    RSYNC_SSH_CMD="ssh ${SSH_OPTS[*]}"
    rsync -avz --progress -e "${RSYNC_SSH_CMD}" \
        "${REMOTE_TARGET}:${REMOTE_CKPT}" "${LOCAL_CKPT}"

    REMOTE_HASH="$(remote_exec "sha256sum '${REMOTE_CKPT}' | awk '{print \$1}'")"
    if command -v sha256sum >/dev/null 2>&1; then
        LOCAL_HASH="$(sha256sum "${LOCAL_CKPT}" | awk '{print $1}')"
        if [[ "${REMOTE_HASH}" != "${LOCAL_HASH}" ]]; then
            log_error "SHA256 mismatch! Remote=${REMOTE_HASH} vs Local=${LOCAL_HASH}"
            exit 1
        fi
        log_ok "SHA256 verified: ${LOCAL_HASH}"
    fi

    log_ok "Pulled Base Checkpoint to '${LOCAL_CKPT}'."
}

# ------------------------------------------------------------------------------
# 7. Subcommand: serve (Start Remote FastAPI+WS Service & SSH Tunnel to :8000)
# ------------------------------------------------------------------------------
cmd_serve() {
    validate_config
    cmd_sync

    EXTRA_SERVE_ARGS="$*"
    SERVE_LOG="${REMOTE_DIR}/logs/serve_5090.log"

    log_info "Starting remote FastAPI + WebSocket Personalization Service (session: ${TMUX_SERVE_SESSION})..."

    if [[ "${USE_DOCKER}" == "1" ]]; then
        SERVE_CMD="docker run --rm --gpus all --ipc=host -p ${REMOTE_SERVICE_PORT}:${REMOTE_SERVICE_PORT} -v '${REMOTE_DIR}/checkpoints:/workspace/uTTT/checkpoints' -v '${REMOTE_DIR}/personas:/workspace/uTTT/personas' ${DOCKER_IMAGE} python -u -m server.app --host 0.0.0.0 --port ${REMOTE_SERVICE_PORT} --config configs/5090_config.yaml ${EXTRA_SERVE_ARGS} 2>&1 | tee '${SERVE_LOG}'"
    else
        SERVE_CMD="cd '${REMOTE_DIR}' && source .venv/bin/activate && '${REMOTE_DIR}/.venv/bin/python3' -u -m server.app --host 0.0.0.0 --port ${REMOTE_SERVICE_PORT} --config configs/5090_config.yaml ${EXTRA_SERVE_ARGS} 2>&1 | tee '${SERVE_LOG}'"
    fi

    remote_exec "bash -c \"
        mkdir -p '${REMOTE_DIR}/logs' '${REMOTE_DIR}/personas/snapshots';
        : > '${SERVE_LOG}';
        if command -v tmux >/dev/null 2>&1; then
            tmux has-session -t '${TMUX_SERVE_SESSION}' 2>/dev/null && tmux kill-session -t '${TMUX_SERVE_SESSION}';
            tmux new-session -d -s '${TMUX_SERVE_SESSION}' \\\"${SERVE_CMD}\\\";
            echo '[Remote] Started tmux session: ${TMUX_SERVE_SESSION}';
        else
            nohup bash -c \\\"${SERVE_CMD}\\\" >/dev/null 2>&1 &
            echo '[Remote] Started background server via nohup (PID: '\$!')';
        fi
    \""

    log_info "Waiting for remote FastAPI server on port ${REMOTE_SERVICE_PORT} to become ready..."
    for _ in {1..30}; do
        if remote_exec "curl -fsS http://localhost:${REMOTE_SERVICE_PORT}/health >/dev/null 2>&1"; then
            log_ok "Remote uTTT Personalization Server is healthy!"
            break
        fi
        sleep 1
    done

    log_ok "Establishing SSH Tunnel: localhost:${LOCAL_TUNNEL_PORT} -> ${REMOTE_HOST}:${REMOTE_SERVICE_PORT}"
    log_info "In another terminal, start chatting with: python client_chat.py --user-id alex --ws-url ws://localhost:${LOCAL_TUNNEL_PORT}"
    log_info "Press Ctrl+C to close the local SSH tunnel (remote server stays running in tmux '${TMUX_SERVE_SESSION}')."

    ssh "${SSH_OPTS[@]}" \
        -N -L "${LOCAL_TUNNEL_PORT}:localhost:${REMOTE_SERVICE_PORT}" \
        "${REMOTE_TARGET}"
}

# ------------------------------------------------------------------------------
# 8. Subcommands: status, logs, stop
# ------------------------------------------------------------------------------
cmd_status() {
    validate_config
    local MODE="${1:-stream}"
    if [[ "${MODE}" == "--once" || "${MODE}" == "once" ]]; then
        log_info "Querying remote RTX 5090 status snapshot on ${REMOTE_TARGET}..."
        remote_exec "
            echo '=== [GPU Status] ===';
            nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu --format=csv;
            echo '';
            echo '=== [Active tmux Sessions] ===';
            tmux ls 2>/dev/null || echo 'No active tmux sessions.';
            echo '';
            echo '=== [Checkpoints & Personas] ===';
            ls -lh '${REMOTE_DIR}/checkpoints' 2>/dev/null || true;
            ls -lh '${REMOTE_DIR}/personas' 2>/dev/null || true;
            echo '';
            echo '=== [Latest Training Progress] ===';
            tail -n 6 '${REMOTE_DIR}/logs/train_5090.log' 2>/dev/null || true;
        "
    else
        log_info "Streaming real-time RTX 5090 status from ${REMOTE_TARGET} (Refresh: 1s | Press Ctrl+C to exit)..."
        remote_exec "
            while true; do
                printf '\033[2J\033[H';
                echo '================================================================================';
                echo -n '[uTTT-Remote Live Monitor] '; date '+%Y-%m-%d %H:%M:%S';
                echo '================================================================================';
                echo '=== [GPU Real-Time Telemetry - RTX 5090] ===';
                nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu --format=csv;
                echo '';
                echo '=== [Active tmux Sessions] ===';
                tmux ls 2>/dev/null || echo 'No active tmux sessions.';
                echo '';
                echo '=== [Checkpoints & Personas] ===';
                ls -lh '${REMOTE_DIR}/checkpoints' 2>/dev/null || true;
                echo '';
                echo '=== [Latest Training Log] ===';
                tail -n 6 '${REMOTE_DIR}/logs/train_5090.log' 2>/dev/null || echo 'No training log yet.';
                echo '--------------------------------------------------------------------------------';
                echo 'Press Ctrl+C to stop live status stream.';
                sleep 1;
            done
        "
    fi
}

cmd_logs() {
    validate_config
    TARGET_MODE="${1:-train}"
    if [[ "${TARGET_MODE}" == "serve" ]]; then
        LOG_PATH="${REMOTE_DIR}/logs/serve_5090.log"
    else
        LOG_PATH="${REMOTE_DIR}/logs/train_5090.log"
    fi
    log_info "Tailing remote log: ${LOG_PATH}..."
    remote_exec "tail -n 100 -f '${LOG_PATH}'"
}

cmd_stop() {
    validate_config
    TARGET_MODE="${1:-all}"
    remote_exec "bash -c \"
        if [[ '${TARGET_MODE}' == 'train' || '${TARGET_MODE}' == 'all' ]]; then
            tmux kill-session -t '${TMUX_TRAIN_SESSION}' 2>/dev/null && echo 'Stopped ${TMUX_TRAIN_SESSION}' || true;
        fi
        if [[ '${TARGET_MODE}' == 'serve' || '${TARGET_MODE}' == 'all' ]]; then
            tmux kill-session -t '${TMUX_SERVE_SESSION}' 2>/dev/null && echo 'Stopped ${TMUX_SERVE_SESSION}' || true;
        fi
    \""
    log_ok "Requested stop (${TARGET_MODE}) completed."
}

# ------------------------------------------------------------------------------
# 9. Entrypoint Dispatcher
# ------------------------------------------------------------------------------
COMMAND="${1:-help}"
shift || true

case "${COMMAND}" in
    init-env)
        cmd_init_env "$@"
        ;;
    sync)
        cmd_sync "$@"
        ;;
    prepare-data)
        cmd_prepare_data "$@"
        ;;
    train)
        cmd_train "$@"
        ;;
    pull-ckpt)
        cmd_pull_ckpt "$@"
        ;;
    serve)
        cmd_serve "$@"
        ;;
    status)
        cmd_status "$@"
        ;;
    logs)
        cmd_logs "$@"
        ;;
    stop)
        cmd_stop "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Unknown subcommand: '${COMMAND}'"
        usage
        exit 1
        ;;
esac

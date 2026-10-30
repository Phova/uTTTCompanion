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
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
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
  prepare-tokenizer     Train a local BPE tokenizer on remote (--data-path required)
  evaluate [args]       Evaluate frozen-memory replies (--checkpoint required)
  profile-capacity      Measure full-AdamW capacity in isolated remote subprocesses
  train [extra_args]    Start train_5090.py in remote tmux session and stream live loss
  pull-ckpt             Pull remote checkpoints/base_uttt.pt to local checkpoints/base_uttt.pt
  serve [extra_args]    Launch remote FastAPI+WebSocket server & map SSH tunnel to localhost:${LOCAL_TUNNEL_PORT}
                        Supported [extra_args]:
                          --user-id <id>             Pre-warm user persona FastWeightState in GPU VRAM on startup
                          --config <path>            YAML config path (default: configs/5090_config.yaml)
                          --base-ckpt <path>         Base checkpoint path (default: checkpoints/base_uttt.pt)
                          --personas-dir <dir>       Persona state directory (default: personas)
                          --max-gpu-users <int>      Tier-1 GPU VRAM LRU pool size (default: 16)
                          --max-new-tokens <int>     Default max generated tokens per turn (default: 96)
                          --temperature <float>      Default sampling temperature (default: 0.7)
                          --surprise-threshold <f>   Override Surprise Gate threshold tau (default: 0.015)
                          --no-compile               Disable torch.compile kernel fusion on startup
                          --lightweight              Use lightweight fallback model when cold-starting
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

# Upload one small helper without putting its body into the SSH exec request.
# This also avoids Windows DefaultShell command-length limits.
remote_upload_helper() {
    local HELPER_LOCAL="$1" HELPER_HASH HELPER_REMOTE HELPER_BOOTSTRAP HELPER_CODE HELPER_COMMAND
    HELPER_HASH=$(sha256sum < "$HELPER_LOCAL")
    HELPER_HASH=${HELPER_HASH%% *}
    HELPER_REMOTE="${REMOTE_DIR}/.cache/remote-view/${HELPER_HASH}-$(basename "$HELPER_LOCAL")"
    HELPER_BOOTSTRAP=$(cat <<'HELPER_UPLOAD'
import hashlib,os,pathlib,sys,tempfile
p=pathlib.Path(sys.argv[1]);d=sys.stdin.buffer.read()
if hashlib.sha256(d).hexdigest()!=sys.argv[2]:raise SystemExit('Viewer upload checksum mismatch')
p.parent.mkdir(parents=True,exist_ok=True)
fd,t=tempfile.mkstemp(dir=p.parent)
try:
 with os.fdopen(fd,'wb') as f:f.write(d)
 os.replace(t,p)
finally:
 if os.path.exists(t):os.unlink(t)
HELPER_UPLOAD
)
    HELPER_CODE=$(printf '%s' "$HELPER_BOOTSTRAP" | base64 | tr -d '\r\n')
    printf -v HELPER_COMMAND 'python3 -c "import base64;exec(base64.b64decode('\''%s'\''))" %q %q' \
        "$HELPER_CODE" "$HELPER_REMOTE" "$HELPER_HASH"
    if ! remote_exec "$HELPER_COMMAND" < "$HELPER_LOCAL"; then
        log_error "Viewer helper upload failed; no viewer was started."
        return 1
    fi
    printf -v "$2" '%s' "$HELPER_REMOTE"
}

# Windows ConPTY/default-shell combinations may drop remote command arguments.
# Use non-PTY SSH with an anonymous stdin pipe; only this shell owns its write
# end. Parent exit (even SIGKILL) therefore delivers EOF to the remote supervisor.
# No heartbeat, sleep, FIFO or additional local pump process is needed.
remote_view_stream() {
    local VIEW_SSH_PID="" VIEW_STATUS=0 VIEW_STDIN_FD="" VIEW_UNUSED_FD="" VIEW_OUTPUT_FD=""
    local VIEW_HELPER VIEW_COMMAND
    remote_upload_helper "${SCRIPT_DIR}/remote_view.py" VIEW_HELPER || return 1
    printf -v VIEW_COMMAND '%q ' python3 -u "$VIEW_HELPER" "$@"
    cleanup_view_connection() {
        trap - EXIT INT TERM HUP
        if [[ -n "$VIEW_STDIN_FD" ]]; then
            exec {VIEW_STDIN_FD}>&-
            VIEW_STDIN_FD=""
        fi
        if [[ -n "$VIEW_SSH_PID" ]]; then
            kill -TERM "$VIEW_SSH_PID" 2>/dev/null || true
            wait "$VIEW_SSH_PID" 2>/dev/null || true
        fi
    }
    trap cleanup_view_connection EXIT
    trap 'cleanup_view_connection; exit 0' INT TERM HUP
    # Preserve panel output before coproc redirects its default stdout pipe.
    exec {VIEW_OUTPUT_FD}>&1
    coproc UTTT_VIEW_CONNECTION {
        exec 1>&"$VIEW_OUTPUT_FD"
        exec {VIEW_OUTPUT_FD}>&-
        exec ssh -T "${SSH_OPTS[@]}" "${REMOTE_TARGET}" "$VIEW_COMMAND"
    }
    VIEW_SSH_PID=$!
    VIEW_STDIN_FD=${UTTT_VIEW_CONNECTION[1]:-}
    VIEW_UNUSED_FD=${UTTT_VIEW_CONNECTION[0]:-}
    if [[ -n "$VIEW_UNUSED_FD" ]]; then
        exec {VIEW_UNUSED_FD}<&-
    fi
    exec {VIEW_OUTPUT_FD}>&-
    wait "$VIEW_SSH_PID" || VIEW_STATUS=$?
    VIEW_SSH_PID=""
    cleanup_view_connection
    return "$VIEW_STATUS"
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
        --exclude="runs/" \
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
    local EXTRA_DATA_ARGS
    printf -v EXTRA_DATA_ARGS '%q ' "$@"
    log_info "Downloading & formatting public Persona + Multi-Turn dataset on ${REMOTE_TARGET}..."
    remote_exec "cd '${REMOTE_DIR}' && python3 deploy/prepare_dataset.py ${EXTRA_DATA_ARGS}"
    log_ok "Remote public dataset prepared at ${REMOTE_DIR}/data/uttt_persona_train.jsonl"
}

cmd_python_tool() {
    local TOOL_SCRIPT="$1"
    shift
    validate_config
    cmd_sync
    local QUOTED_ARGS
    printf -v QUOTED_ARGS '%q ' "$@"
    remote_exec "cd '${REMOTE_DIR}' && .venv/bin/python -u 'deploy/${TOOL_SCRIPT}' ${QUOTED_ARGS}"
}

# ------------------------------------------------------------------------------
# 5. Subcommand: train (Remote tmux Background Meta-Training + Live Loss Stream)
# ------------------------------------------------------------------------------
cmd_train() {
    validate_config
    cmd_sync
    local QUOTED_ARGS
    printf -v QUOTED_ARGS '%q ' "${REMOTE_DIR}" "${TMUX_TRAIN_SESSION}" "${TMUX_SERVE_SESSION}" "${USE_DOCKER}" "${DOCKER_IMAGE}" "$@"
    remote_exec "bash -s -- ${QUOTED_ARGS}" <<'REMOTE_TRAIN'
set -euo pipefail
ROOT="$1"; SESSION="$2"; SERVE_SESSION="$3"; DOCKER="$4"; IMAGE="$5"
shift 5
cd "$ROOT"
mkdir -p logs checkpoints
if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        echo "[ERROR] Training session $SESSION is already active. Stop it explicitly before starting another run." >&2
        exit 1
    fi
    if tmux has-session -t "=$SERVE_SESSION" 2>/dev/null; then
        echo "[ERROR] Serving session $SERVE_SESSION is using the GPU. Stop it before starting training." >&2
        exit 1
    fi
elif [[ -f "logs/$SESSION.pid" ]] && kill -0 "$(cat "logs/$SESSION.pid")" 2>/dev/null; then
    echo "[ERROR] Background training is already active; stop it explicitly first." >&2
    exit 1
fi
if [[ "$DOCKER" == 1 ]]; then
    COMMAND=(docker run --rm --name "$SESSION" --gpus all --ipc=host -v "$ROOT/checkpoints:/workspace/uTTT/checkpoints" -v "$ROOT/logs:/workspace/uTTT/logs" -v "$ROOT/data:/workspace/uTTT/data:ro" "$IMAGE" python -u train_5090.py --config configs/5090_config.yaml "$@")
else
    export PATH="$ROOT/.venv/bin:$PATH" TORCH_CUDA_ARCH_LIST='12.0'
    COMMAND=("$ROOT/.venv/bin/python3" -u train_5090.py --config configs/5090_config.yaml "$@")
fi
printf -v RUN_COMMAND '%q ' "${COMMAND[@]}"
printf -v LOG_PATH '%q' "$ROOT/logs/train_5090.log"
RUN_COMMAND="set -o pipefail; $RUN_COMMAND 2>&1 | tee $LOG_PATH"
: > logs/train_5090.log
if command -v tmux >/dev/null 2>&1; then
    printf -v RUN_SHELL '%q' "$RUN_COMMAND"
    tmux new-session -d -s "$SESSION" "bash -c $RUN_SHELL"
    echo "[Remote] Started tmux session: $SESSION"
else
    PID_FILE="logs/$SESSION.pid"
    nohup bash -c "$RUN_COMMAND" >/dev/null 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "[Remote] Started background training (PID: $!)"
fi
REMOTE_TRAIN
    log_ok "Training launched on the remote host."
    log_info "Streaming training loss (Ctrl+C detaches; remote training continues)..."
    remote_view_stream --train-root "$REMOTE_DIR" --train-session "$TMUX_TRAIN_SESSION"
    log_info "Detached local training log view."
}

# ------------------------------------------------------------------------------
# 6. Subcommand: pull-ckpt (Pull Trained base_uttt.pt Back to Local Workspace)
# ------------------------------------------------------------------------------
cmd_pull_ckpt() {
    validate_config
    local TARGET_ARG="${1:-}"
    local CKPT_NAME=""

    if [[ -n "${TARGET_ARG}" ]]; then
        CKPT_NAME="$(basename "${TARGET_ARG}")"
    else
        CKPT_NAME="$(remote_exec "ls -t '${REMOTE_DIR}/checkpoints/'*.pt 2>/dev/null | head -n 1 | xargs -r basename" | tr -d '\r\n ')"
        if [[ -z "${CKPT_NAME}" ]]; then
            CKPT_NAME="base_uttt.pt"
        fi
    fi

    REMOTE_CKPT="${REMOTE_DIR}/checkpoints/${CKPT_NAME}"
    LOCAL_CKPT_DIR="${LOCAL_ROOT}/checkpoints"
    LOCAL_CKPT="${LOCAL_CKPT_DIR}/${CKPT_NAME}"

    mkdir -p "${LOCAL_CKPT_DIR}"
    log_info "Checking remote checkpoint at ${REMOTE_TARGET}:${REMOTE_CKPT}..."

    if ! remote_exec "test -f '${REMOTE_CKPT}'"; then
        log_error "Remote checkpoint '${REMOTE_CKPT}' does not exist yet. Run './deploy/remote_ctl.sh train' first."
        exit 1
    fi

    RSYNC_SSH_CMD="ssh ${SSH_OPTS[*]}"
    rsync -avz --progress -e "${RSYNC_SSH_CMD}" \
        "${REMOTE_TARGET}:${REMOTE_CKPT}" "${LOCAL_CKPT}"

    REMOTE_HASH="$(remote_exec "sha256sum '${REMOTE_CKPT}' | cut -d' ' -f1" | tr -d '\r\n ')"
    if command -v sha256sum >/dev/null 2>&1; then
        LOCAL_HASH="$(sha256sum "${LOCAL_CKPT}" | cut -d' ' -f1 | tr -d '\r\n ')"
        if [[ "${REMOTE_HASH}" != "${LOCAL_HASH}" ]]; then
            log_error "SHA256 mismatch! Remote=${REMOTE_HASH} vs Local=${LOCAL_HASH}"
            exit 1
        fi
        log_ok "SHA256 verified: ${LOCAL_HASH}"
    fi

    log_ok "Pulled Base Checkpoint to '${LOCAL_CKPT}'."
}

# ------------------------------------------------------------------------------
# 7. Subcommand: serve (Start Remote FastAPI+WS Service & Map Local SSH Tunnel)
# ------------------------------------------------------------------------------
cmd_serve() {
    validate_config
    cmd_sync

    local SERVE_LOG="${REMOTE_DIR}/logs/serve_5090.log" QUOTED_ARGS
    if [[ ! "${STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
        log_error "STARTUP_TIMEOUT must be a positive number of seconds."
        return 1
    fi
    printf -v QUOTED_ARGS '%q ' "${REMOTE_DIR}" "${TMUX_SERVE_SESSION}" "${USE_DOCKER}" "${DOCKER_IMAGE}" "${REMOTE_SERVICE_PORT}" "$@"
    remote_exec "bash -s -- ${QUOTED_ARGS}" <<'REMOTE_SERVE'
set -euo pipefail
ROOT="$1"; SESSION="$2"; DOCKER="$3"; IMAGE="$4"; PORT="$5"
shift 5
cd "$ROOT"
mkdir -p logs personas/snapshots
PYTHON=python3
[[ ! -x "$ROOT/.venv/bin/python3" ]] || PYTHON="$ROOT/.venv/bin/python3"
STOP_ARGS=(--workspace "$ROOT" --session "$SESSION" --pid-file "$ROOT/logs/$SESSION.pid" --kind serve)
[[ "$DOCKER" != 1 ]] || STOP_ARGS+=(--container "$SESSION")
"$PYTHON" deploy/stop_training.py "${STOP_ARGS[@]}"
tmux kill-session -t "=$SESSION" 2>/dev/null || true
if [[ "$DOCKER" == 1 ]]; then
    COMMAND=(docker run --rm --name "$SESSION" --gpus all --ipc=host -p "$PORT:$PORT" -v "$ROOT/checkpoints:/workspace/uTTT/checkpoints" -v "$ROOT/personas:/workspace/uTTT/personas" "$IMAGE" python -u -m server.app --host 0.0.0.0 --port "$PORT" --config configs/5090_config.yaml "$@")
else
    export PATH="$ROOT/.venv/bin:$PATH"
    COMMAND=("$ROOT/.venv/bin/python3" -u -m server.app --host 0.0.0.0 --port "$PORT" --config configs/5090_config.yaml "$@")
fi
printf -v RUN_COMMAND '%q ' "${COMMAND[@]}"
printf -v LOG_PATH '%q' "$ROOT/logs/serve_5090.log"
RUN_COMMAND="set -o pipefail; $RUN_COMMAND 2>&1 | tee $LOG_PATH"
: > logs/serve_5090.log
if command -v tmux >/dev/null 2>&1; then
    printf -v RUN_SHELL '%q' "$RUN_COMMAND"
    tmux new-session -d -s "$SESSION" "bash -c $RUN_SHELL"
    echo "[Remote] Started tmux session: $SESSION"
else
    nohup bash -c "$RUN_COMMAND" >/dev/null 2>&1 &
    echo "$!" > "logs/$SESSION.pid"
    echo "[Remote] Started background server (PID: $!)"
fi
REMOTE_SERVE

    log_info "Waiting for remote FastAPI server on port ${REMOTE_SERVICE_PORT} to become ready (includes kernel warmup)..."
    local READY=0
    if [[ ! "${STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
        log_error "STARTUP_TIMEOUT must be a positive number of seconds."
        return 1
    fi
    local DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
    while (( SECONDS < DEADLINE )); do
        if remote_exec "curl --connect-timeout 2 --max-time 2 -fsS http://localhost:${REMOTE_SERVICE_PORT}/health >/dev/null 2>&1"; then
            log_ok "Remote uTTT Personalization Server is healthy!"
            READY=1
            break
        fi
        if remote_exec "grep -q 'Application startup failed' '${SERVE_LOG}' 2>/dev/null"; then
            log_error "Server failed during application startup."
            remote_exec "tail -n 80 '${SERVE_LOG}'"
            return 1
        fi
        sleep 1
    done

    if [[ "${READY}" -eq 0 ]]; then
        log_error "Server did not become ready within ${STARTUP_TIMEOUT}s. Check logs:"
        remote_exec "cat '${SERVE_LOG}' 2>/dev/null || true"
        exit 1
    fi

    # Resolve remote internal IP (essential when remote sshd runs on Windows 10 host forwarding into WSL2 VM)
    local REMOTE_INTERNAL_IP
    REMOTE_INTERNAL_IP="$(remote_exec "hostname -I 2>/dev/null | cut -d' ' -f1" | tr -d '\r\n ')"
    if [[ -z "${REMOTE_INTERNAL_IP}" ]]; then
        REMOTE_INTERNAL_IP="127.0.0.1"
    fi

    log_ok "Establishing SSH Tunnel: localhost:${LOCAL_TUNNEL_PORT} -> ${REMOTE_HOST} (${REMOTE_INTERNAL_IP}:${REMOTE_SERVICE_PORT})"
    log_info "In another terminal, connect the interactive CLI client with:"
    log_info "  python3 client_chat.py --user-id Phova --ws-url ws://localhost:${LOCAL_TUNNEL_PORT}   (WSL / Linux)"
    log_info "  python  client_chat.py --user-id Phova --ws-url ws://localhost:${LOCAL_TUNNEL_PORT}   (Windows PowerShell)"
    log_info "Press Ctrl+C to close the local SSH tunnel (remote server stays running in tmux '${TMUX_SERVE_SESSION}')."

    trap 'printf "\n"; log_info "Closed local SSH tunnel (remote server remains active in tmux ${TMUX_SERVE_SESSION})."; exit 0' INT TERM
    ssh "${SSH_OPTS[@]}" \
        -o ExitOnForwardFailure=yes -N -L "${LOCAL_TUNNEL_PORT}:${REMOTE_INTERNAL_IP}:${REMOTE_SERVICE_PORT}" \
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
        local INTERVAL_MS="${1:-200}" TARGET_HZ MONITOR_REMOTE MONITOR_COLUMNS="${COLUMNS:-80}"
        if [[ ! "$INTERVAL_MS" =~ ^[0-9]{1,5}$ ]] || (( 10#$INTERVAL_MS < 1 || 10#$INTERVAL_MS > 60000 )); then
            log_error "Refresh interval must be an integer between 1 and 60000 milliseconds."
            return 1
        fi
        INTERVAL_MS=$((10#$INTERVAL_MS))
        TARGET_HZ=$(awk -v ms="$INTERVAL_MS" 'BEGIN {printf "%g", 1000/ms}')
        log_info "Live RTX 5090 Monitor on ${REMOTE_TARGET} (Refresh: ${INTERVAL_MS}ms / target ${TARGET_HZ}Hz | Ctrl+C to exit)"
        if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
            MONITOR_COLUMNS=$(tput cols 2>/dev/null) || MONITOR_COLUMNS=80
        fi
        if [[ ! "$MONITOR_COLUMNS" =~ ^[0-9]{2,5}$ ]] || (( 10#$MONITOR_COLUMNS < 20 || 10#$MONITOR_COLUMNS > 10000 )); then
            MONITOR_COLUMNS=80
        fi
        remote_upload_helper "${SCRIPT_DIR}/monitor_status.py" MONITOR_REMOTE || return 1
        remote_view_stream -- python3 -u "$MONITOR_REMOTE" --root "$REMOTE_DIR" --interval-ms "$INTERVAL_MS" --columns "$MONITOR_COLUMNS"
    fi
}

cmd_logs() {
    validate_config
    local TARGET_MODE="${1:-train}" LOG_PATH
    if [[ "$TARGET_MODE" != train && "$TARGET_MODE" != serve ]]; then
        log_error "Log target must be train or serve."
        return 1
    fi
    LOG_PATH="${REMOTE_DIR}/logs/${TARGET_MODE}_5090.log"
    log_info "Tailing remote log: ${LOG_PATH} (Press Ctrl+C to exit)..."
    remote_view_stream -- tail -n 100 -F -- "$LOG_PATH"
}

cmd_stop() {
    validate_config
    local TARGET_MODE="${1:-all}" QUOTED_ARGS
    if [[ "$TARGET_MODE" != train && "$TARGET_MODE" != serve && "$TARGET_MODE" != all ]]; then
        log_error "Stop target must be train, serve or all."
        return 1
    fi
    printf -v QUOTED_ARGS '%q ' "${REMOTE_DIR}" "$TARGET_MODE" "${TMUX_TRAIN_SESSION}" "${TMUX_SERVE_SESSION}" "${USE_DOCKER}"
    remote_exec "bash -s -- ${QUOTED_ARGS}" <<'REMOTE_STOP'
set -euo pipefail
ROOT="$1"; MODE="$2"; TRAIN_SESSION="$3"; SERVE_SESSION="$4"; DOCKER="$5"
cd "$ROOT"
if [[ "$MODE" == train || "$MODE" == all ]]; then
    PYTHON=python3
    [[ ! -x "$ROOT/.venv/bin/python3" ]] || PYTHON="$ROOT/.venv/bin/python3"
    STOP_ARGS=(--workspace "$ROOT" --session "$TRAIN_SESSION" --pid-file "$ROOT/logs/$TRAIN_SESSION.pid")
    [[ "$DOCKER" != 1 ]] || STOP_ARGS+=(--container "$TRAIN_SESSION")
    "$PYTHON" deploy/stop_training.py "${STOP_ARGS[@]}"
    # The helper waits for the selected trainer/pipeline to exit after saving.
    # A timeout exits above and leaves the session available for inspection.
    tmux kill-session -t "=$TRAIN_SESSION" 2>/dev/null || true
    rm -f -- "$ROOT/logs/$TRAIN_SESSION.pid"
fi
if [[ "$MODE" == serve || "$MODE" == all ]]; then
    PYTHON=python3
    [[ ! -x "$ROOT/.venv/bin/python3" ]] || PYTHON="$ROOT/.venv/bin/python3"
    STOP_ARGS=(--workspace "$ROOT" --session "$SERVE_SESSION" --pid-file "$ROOT/logs/$SERVE_SESSION.pid" --kind serve)
    [[ "$DOCKER" != 1 ]] || STOP_ARGS+=(--container "$SERVE_SESSION")
    "$PYTHON" deploy/stop_training.py "${STOP_ARGS[@]}"
    tmux kill-session -t "=$SERVE_SESSION" 2>/dev/null || true
    rm -f -- "$ROOT/logs/$SERVE_SESSION.pid"
fi
REMOTE_STOP
    log_ok "Requested stop ($TARGET_MODE) completed."
}

# ------------------------------------------------------------------------------
# 9. Entrypoint Dispatcher (Wrapped in main() to prevent on-disk edit offset shifts)
# ------------------------------------------------------------------------------
main() {
    local COMMAND="${1:-help}"
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
        prepare-tokenizer)
            cmd_python_tool train_tokenizer.py "$@"
            ;;
        evaluate)
            cmd_python_tool evaluate_model.py "$@"
            ;;
        profile-capacity)
            cmd_python_tool profile_capacity.py "$@"
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
}

main "$@"
exit 0

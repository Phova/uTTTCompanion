现在请完整实现第三部分：`deploy/Dockerfile`、`deploy/remote_ctl.sh` 和 `configs/5090_config.yaml`。
技术细节要求：
1. `remote_ctl.sh` 需要支持通过环境变量配置 `REMOTE_HOST`、`REMOTE_PORT`、`REMOTE_USER`、`REMOTE_DIR` 和 `SSH_KEY`。
2. 包含 `init-env`（远端自动检测 RTX 5090 驱动与构建环境）、`sync`（排除 `checkpoints/` 和 `.git/` 的高速 rsync）、`train`（远端 tmux 后台启动训练并回传实时 loss）、`serve`（启动远端个性化推理服务并通过 SSH 隧道映射到本地 `localhost:8000`）四个核心命令。
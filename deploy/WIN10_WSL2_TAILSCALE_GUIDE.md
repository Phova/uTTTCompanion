# 远端 Windows 10 专业版 (32GB 内存 + RTX 5090) + WSL2 + Tailscale 完整部署指南

本指南针对你的实际硬件与网络拓扑量身定制：
- **远端服务器**：Windows 10 专业版 | **32GB 系统内存** | **NVIDIA RTX 5090 (32GB GDDR7)**
- **组网方式**：**Tailscale** 点对点加密组网（虚拟内网 IP `100.x.y.z`）
- **部署架构**：**Win10 宿主机 OpenSSH + WSL2 (Ubuntu 24.04) 算子加速环境**（兼顾 Win10 防火墙/Tailscale 的极简网络管理与 Linux 下 `torch.compile` / Triton 的满血性能）

---

## 一、防火墙与网络架构说明：配置还是一样吗？

**完全一样，且无需做任何复杂的 WSL2 端口转发！**

原因如下：
1. **Tailscale 与 OpenSSH 均跑在远端 Win10 宿主机上**：连接直接走 Win10 宿主机的 Tailscale 虚拟网卡（`100.x.y.z:22`），防火墙规则依然只需在 Win10 PowerShell 中执行一条命令（限制 22 端口仅允许 Tailscale 网段 `100.64.0.0/10` 访问）。
2. **SSH 登录自动直入 WSL2**：我们将 Win10 OpenSSH 的 `DefaultShell` 设置为 `C:\Windows\System32\bash.exe`（WSL 桥接入口，原生支持 `-c` 参数，完美兼容 `ssh`、`rsync` 与 `scp`）。
3. **自动打通 8000 推理端口**：配合 `.wslconfig` 的 `localhostForwarding=true`，WSL2 内部启动的 `localhost:8000` 服务会自动映射到宿主机的 `localhost:8000`，再经由 SSH 隧道直接映射回你本地电脑的 `localhost:8000`。

---

## 二、第一阶段：远端 Windows 10 专业版一次性初始化（在远端操作）

请通过远程桌面或显示器，在**远端 Windows 10 服务器**上以**管理员身份**打开 PowerShell，按顺序完成以下 4 步：

### 步骤 1：针对 32GB 物理内存配置 `.wslconfig`
远端主机总内存为 **32GB**，需为 Win10 宿主机系统、NVIDIA 驱动及桌面预留约 **8GB**，将 **24GB 物理内存 + 16GB 虚拟交换空间**分配给 WSL2（防止保存/加载数 GB 的 `base_uttt.pt` 时主机内存不足）：

```powershell
# 在远端 Win10 管理员 PowerShell 中运行：写入 ~/.wslconfig
@"
[wsl2]
memory=24GB
swap=16GB
localhostForwarding=true
"@ | Set-Content -Path "$HOME\.wslconfig" -Encoding UTF8
```

### 步骤 2：安装并更新 WSL2 与 Ubuntu 24.04
> **前提**：确认远端 Win10 已安装最新 NVIDIA RTX 5090 Windows 驱动（版本 `>= 570.xx`）。**切勿**在 WSL2 Ubuntu 内部再次安装 Linux 显卡驱动（WSL2 会自动直通宿主机驱动）。

```powershell
# 1. 将 WSL 内核更新到最新版（确保完整支持 RTX 5090 sm_120 GPU 直通）
wsl --update

# 2. 安装 Ubuntu-24.04 并设为默认子系统
wsl --install -d Ubuntu-24.04
wsl --set-default Ubuntu-24.04
```
安装完成后会弹出 Ubuntu 终端提示创建 Linux 用户名和密码（例如设为 `uttt`），创建后输入 `exit` 回到 PowerShell。

### 步骤 3：在 WSL2 内配置 `sudo` 免密与必要工具
在远端 PowerShell 中运行以下命令，在 Ubuntu 内装好 `rsync` 和 `tmux`：
```powershell
wsl -u root bash -c "apt-get update && apt-get install -y rsync tmux curl git python3 python3-venv python3-pip build-essential"
```

### 步骤 4：开启 Win10 OpenSSH Server、绑定 `bash.exe` 并配置 Tailscale 防火墙
```powershell
# 1. 安装并启动 OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# 2. 防火墙：仅允许 Tailscale 虚拟内网 (100.64.0.0/10) 访问 22 端口（阻止公网/局域网扫描）
if (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -Enabled True -Action Allow -RemoteAddress "100.64.0.0/10"
} else {
    New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (Tailscale Only)" `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -RemoteAddress "100.64.0.0/10"
}

# 3. 将 SSH 默认 Shell 设为 C:\Windows\System32\bash.exe（原生支持 ssh -c 与 rsync）
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "C:\Windows\System32\bash.exe" -PropertyType String -Force

# 4. 解除 Windows OpenSSH 对管理员组公钥路径的特殊限制
$cfg = "C:\ProgramData\ssh\sshd_config"
(Get-Content $cfg) `
    -replace '^Match Group administrators', '#Match Group administrators' `
    -replace '^\s*AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys', '#AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys' |
    Set-Content $cfg

Restart-Service sshd

# 5. 查看当前 Win10 管理员真实账户名与 Tailscale IP（记下这两项填到本地 .env）
Write-Host "REMOTE_USER 应填: $env:USERNAME" -ForegroundColor Green
Write-Host "REMOTE_HOST 应填: $(tailscale ip -4)" -ForegroundColor Green
```

---

## 三、第二阶段：本地电脑配置 SSH 免密与 `.env`（在本地操作）

回到你**本地开发电脑**（`d:\VibeCoding\uTTT`）：

### 步骤 1：配置本地到远端 Win10 的 SSH 免密登录
在本地打开 PowerShell：
```powershell
# 1. 如果本地还没有 SSH 密钥，生成一对 ed25519 密钥（一路回车即可）
if (-not (Test-Path "$HOME\.ssh\id_ed25519.pub")) {
    ssh-keygen -t ed25519 -N '""' -f "$HOME\.ssh\id_ed25519"
}

# 2. 将公钥写入远端 Win10 管理员账户的 C:\Users\<管理员名>\.ssh\authorized_keys
# （注意：将下面的 YourAdminName 和 100.x.y.z 替换为远端输出的实际值）
$RemoteUser = "YourAdminName"
$RemoteIP   = "100.x.y.z"
$PubKey     = Get-Content "$HOME\.ssh\id_ed25519.pub"

ssh "${RemoteUser}@${RemoteIP}" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$PubKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```
> **补充说明**：因为 Windows OpenSSH 认证发生在 Windows 层，如果发现仍提示输密码，只需在远端 Win10 PowerShell 执行一句：
> `New-Item -ItemType Directory -Force "$HOME\.ssh"; Add-Content "$HOME\.ssh\authorized_keys" "<你的id_ed25519.pub内容>"` 即可。

### 步骤 2：创建本地 `d:\VibeCoding\uTTT\.env` 配置文件
在项目根目录新建 `.env` 文件：
```ini
# 远端 Win10 的 Tailscale IP
REMOTE_HOST=100.x.y.z
REMOTE_PORT=22

# 远端 Win10 的管理员账户名（即远端 $env:USERNAME 的输出）
REMOTE_USER=YourAdminName

# 重要：REMOTE_DIR 必须填 WSL2 Linux 内部路径（切勿填 /mnt/c 或 /mnt/d，保证 ext4 满速读写）
REMOTE_DIR=/home/uttt/uTTT

# 本地私钥路径
SSH_KEY=~/.ssh/id_ed25519
USE_DOCKER=0
```
*(并在远端 WSL2 中确保 `/home/uttt/uTTT` 对当前 WSL 默认用户可写：`ssh YourAdminName@100.x.y.z "sudo mkdir -p /home/uttt/uTTT && sudo chmod -R 777 /home/uttt/uTTT"`)*

---

## 四、第三阶段：一键部署、元训练、权重拉取与交互推理

在本地电脑使用 **Git Bash** 或 **WSL 终端**进入 `d:\VibeCoding\uTTT`：

### 1. 一键初始化远端 5090 环境 (`init-env`)
```bash
./deploy/remote_ctl.sh init-env
```
- 自动检测远端 RTX 5090 (`sm_120`) 与 32GB 显存；
- 使用 `rsync` 高速同步本地代码到 WSL2 `/home/uttt/uTTT`；
- 自动创建 `.venv`、安装 CUDA 12.8 版 `torch>=2.6.0` + `triton` 并完成 `SharedFastWeightPool` 算子校验。

### 2. 启动远端后台训练并实时查看 Loss (`train`)
```bash
# 先跑 50 步轻量级验证整条链路
./deploy/remote_ctl.sh train --lightweight --steps 50

# 验证通过后，启动正式 5090 元训练
./deploy/remote_ctl.sh train
```
- 远端在 `tmux` 会话 `uttt_train` 中后台运行，本地实时滚动打印训练 Loss、Surprise Loss、吞吐量（`tok/s`）与峰值显存（严格受控在 `< 28GB`）；
- 随时可按 `Ctrl + C` 断开本地终端，远端训练继续进行；
- 随时可通过 `./deploy/remote_ctl.sh status` 或 `./deploy/remote_ctl.sh logs train` 重新查看进度。

### 3. 导出并拉取 Base Checkpoint (`pull-ckpt`)
训练完成（或每满 200 步自动保存节点）后，在本地执行：
```bash
./deploy/remote_ctl.sh pull-ckpt
```
- 将远端 `/home/uttt/uTTT/checkpoints/base_uttt.pt` 拉取回本地 `d:\VibeCoding\uTTT\checkpoints\base_uttt.pt` 并自动完成 SHA256 完整性校验。

### 4. 启动远端实时个性化推理服务 (`serve`) 与本地对话
1. **终端 1（保持运行，维持 SSH 隧道）**：
   ```bash
   ./deploy/remote_ctl.sh serve
   ```
   远端在 `tmux` 会话 `uttt_serve` 中启动 FastAPI + WebSocket 服务，并将远端 `8000` 端口加密映射到本地 `localhost:8000`。

2. **终端 2（启动本地交互客户端）**：
   ```bash
   python client_chat.py --user-id alex --ws-url ws://localhost:8000
   ```
   每次对话后，客户端将实时输出 TTT 快权重演化状态栏：
   ```text
   [TTT Status] Surprise Loss: 0.42 | Weight Drift from Base: 3.18% | Persona Saved
   ```
   并支持随时输入 `/diff`、`/snapshot v1`、`/restore v1`、`/reset`、`/save` 管理用户专属快权重状态。

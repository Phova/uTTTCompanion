# ==============================================================================
# deploy/remote_ctl.ps1
# One-Click Remote Deployment & Operations Controller for Windows 10 Pro Server
# Target: Remote Windows 10/11 Pro Host with NVIDIA RTX 5090 (OpenSSH Server)
#
# Supported Environment Variables (or .env file in project root):
#   REMOTE_HOST   - Remote Windows 10 Pro IP or hostname
#   REMOTE_PORT   - Remote OpenSSH port (default: "22")
#   REMOTE_USER   - Remote Windows username (default: "Administrator")
#   REMOTE_DIR    - Remote workspace path (default: "C:/uTTT")
#   SSH_KEY       - Path to local SSH private key (default: "$HOME/.ssh/id_ed25519")
#
# Usage in PowerShell:
#   .\deploy\remote_ctl.ps1 init-env
#   .\deploy\remote_ctl.ps1 sync
#   .\deploy\remote_ctl.ps1 train [--lightweight --steps 100]
#   .\deploy\remote_ctl.ps1 pull-ckpt
#   .\deploy\remote_ctl.ps1 serve
#   .\deploy\remote_ctl.ps1 status
#   .\deploy\remote_ctl.ps1 logs [train|serve]
#   .\deploy\remote_ctl.ps1 stop [train|serve|all]
# ==============================================================================

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$LocalRoot = Split-Path -Parent $PSScriptRoot

# 1. Load optional .env file if present
$EnvFile = Join-Path $LocalRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $key = $parts[0].Trim()
            $val = $parts[1].Trim().Trim('"').Trim("'")
            if (-not [Environment]::GetEnvironmentVariable($key)) {
                [Environment]::SetEnvironmentVariable($key, $val, "Process")
            }
        }
    }
}

$RemoteHost = if ($env:REMOTE_HOST) { $env:REMOTE_HOST } else { "" }
$RemotePort = if ($env:REMOTE_PORT) { $env:REMOTE_PORT } else { "22" }
$RemoteUser = if ($env:REMOTE_USER) { $env:REMOTE_USER } else { "Administrator" }
$RemoteDir  = if ($env:REMOTE_DIR)  { $env:REMOTE_DIR.Replace("\", "/") } else { "C:/uTTT" }
$SshKey     = if ($env:SSH_KEY)     { $env:SSH_KEY.Replace("~", $HOME) } else { "$HOME\.ssh\id_ed25519" }
$LocalPort  = if ($env:LOCAL_TUNNEL_PORT) { $env:LOCAL_TUNNEL_PORT } else { "8000" }
$RemoteServicePort = if ($env:REMOTE_SERVICE_PORT) { $env:REMOTE_SERVICE_PORT } else { "8000" }

function Show-Usage {
    Write-Host @"
Usage (Windows 10 Pro Remote Host):
  `$env:REMOTE_HOST="192.168.1.50"; `$env:REMOTE_USER="Administrator"; `$env:REMOTE_DIR="C:/uTTT"
  .\deploy\remote_ctl.ps1 <command> [args...]

Core Commands:
  init-env              Detect RTX 5090 on remote Win10 Pro, sync code, and build .venv
  sync                  Fast tar-over-SSH sync excluding checkpoints/, personas/, .git/
  train [extra_args]    Start detached background training on remote Win10 & stream logs
  pull-ckpt             Pull remote checkpoints/base_uttt.pt to local checkpoints\base_uttt.pt
  serve                 Start remote FastAPI+WS server & open SSH tunnel to localhost:$LocalPort
  status                Check remote GPU status, running PIDs, and checkpoints
  logs [train|serve]    Stream remote log file in real time
  stop [train|serve|all] Stop background training/serving processes on remote Win10
"@ -ForegroundColor Cyan
}

function Assert-Config {
    if ([string]::IsNullOrWhiteSpace($RemoteHost)) {
        Write-Error "REMOTE_HOST is not set. Set `$env:REMOTE_HOST='<ip>' or configure .env first."
        exit 1
    }
}

function Get-SshArgs([bool]$ForScp = $false) {
    $argsList = @()
    if ($ForScp) {
        $argsList += @("-P", $RemotePort)
    } else {
        $argsList += @("-p", $RemotePort)
    }
    $argsList += @("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10")
    if ($SshKey -and (Test-Path $SshKey)) {
        $argsList += @("-i", $SshKey)
    }
    return $argsList
}

function Invoke-RemotePS([string]$PsScript) {
    Assert-Config
    $sshArgs = Get-SshArgs
    $bytes = [System.Text.Encoding]::Unicode.GetBytes($PsScript)
    $encoded = [Convert]::ToBase64String($bytes)
    & ssh @sshArgs "${RemoteUser}@${RemoteHost}" "powershell -NoProfile -NonInteractive -EncodedCommand $encoded"
}

# ------------------------------------------------------------------------------
# Subcommand: sync (Built-in tar over SSH — no rsync required on Windows 10)
# ------------------------------------------------------------------------------
function Invoke-Sync {
    Assert-Config
    Write-Host "[uTTT-Win10] Syncing local workspace -> ${RemoteUser}@${RemoteHost}:${RemoteDir}..." -ForegroundColor Cyan

    Invoke-RemotePS @"
New-Item -ItemType Directory -Force -Path '$RemoteDir/checkpoints' | Out-Null
New-Item -ItemType Directory -Force -Path '$RemoteDir/personas/snapshots' | Out-Null
New-Item -ItemType Directory -Force -Path '$RemoteDir/logs' | Out-Null
"@

    $TempTar = Join-Path ([System.IO.Path]::GetTempPath()) "uttt_sync_$([Guid]::NewGuid().ToString('N')).tar.gz"
    Push-Location $LocalRoot
    try {
        & tar.exe -czf $TempTar `
            --exclude=".git" `
            --exclude="checkpoints" `
            --exclude="personas" `
            --exclude="logs" `
            --exclude=".venv" `
            --exclude="venv" `
            --exclude="__pycache__" `
            --exclude=".pytest_cache" `
            --exclude=".cache" `
            .

        $scpArgs = Get-SshArgs -ForScp $true
        & scp.exe @scpArgs $TempTar "${RemoteUser}@${RemoteHost}:${RemoteDir}/uttt_sync.tar.gz"

        Invoke-RemotePS @"
Set-Location '$RemoteDir'
tar.exe -xzf uttt_sync.tar.gz
Remove-Item -Force uttt_sync.tar.gz
"@
    } finally {
        Pop-Location
        if (Test-Path $TempTar) { Remove-Item -Force $TempTar }
    }
    Write-Host "[uTTT-OK] Codebase synchronized to remote Windows 10 host." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# Subcommand: init-env
# ------------------------------------------------------------------------------
function Invoke-InitEnv {
    Assert-Config
    Write-Host "[uTTT-Win10] Checking remote Windows 10 Pro hardware & NVIDIA driver..." -ForegroundColor Cyan
    Invoke-RemotePS @"
Write-Host "Remote Host: `$env:COMPUTERNAME | OS: `$((Get-CimInstance Win32_OperatingSystem).Caption)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
"@
    Invoke-Sync
    Write-Host "[uTTT-Win10] Running deploy/setup_5090.ps1 on remote Windows 10 Pro host..." -ForegroundColor Cyan
    Invoke-RemotePS "Set-Location '$RemoteDir'; powershell -ExecutionPolicy Bypass -File deploy/setup_5090.ps1"
}

# ------------------------------------------------------------------------------
# Subcommand: train (Detached Start-Process on Remote Windows 10 + Live Log Tail)
# ------------------------------------------------------------------------------
function Invoke-Train {
    Assert-Config
    Invoke-Sync
    $ExtraJoined = ($ExtraArgs -join " ")
    Write-Host "[uTTT-Win10] Starting detached background training on remote RTX 5090..." -ForegroundColor Cyan

    Invoke-RemotePS @"
Set-Location '$RemoteDir'
`$PidFile = '$RemoteDir/logs/train.pid'
if (Test-Path `$PidFile) {
    `$OldPid = Get-Content `$PidFile -ErrorAction SilentlyContinue
    if (`$OldPid) { Stop-Process -Id `$OldPid -Force -ErrorAction SilentlyContinue }
}
`$PyExe = '$RemoteDir/.venv/Scripts/python.exe'
`$LogOut = '$RemoteDir/logs/train_5090.log'
`$LogErr = '$RemoteDir/logs/train_5090.err.log'
Set-Content -Path `$LogOut -Value "[Train] Starting uTTT Meta-Training on Windows 10 RTX 5090..."
`$Proc = Start-Process -FilePath `$PyExe `
    -ArgumentList "-u train_5090.py --config configs/5090_config.yaml $ExtraJoined" `
    -WorkingDirectory '$RemoteDir' `
    -RedirectStandardOutput `$LogOut `
    -RedirectStandardError `$LogErr `
    -WindowStyle Hidden `
    -PassThru
Set-Content -Path `$PidFile -Value `$Proc.Id
Write-Host "[Remote] Background training launched (PID: `$(`$Proc.Id)). Log: `$LogOut"
"@

    Write-Host "[uTTT-Win10] Streaming real-time training log (Press Ctrl+C to detach; remote training continues)..." -ForegroundColor Yellow
    Invoke-RemotePS "Get-Content -Path '$RemoteDir/logs/train_5090.log' -Wait -Tail 50"
}

# ------------------------------------------------------------------------------
# Subcommand: pull-ckpt
# ------------------------------------------------------------------------------
function Invoke-PullCkpt {
    Assert-Config
    $LocalCkptDir = Join-Path $LocalRoot "checkpoints"
    New-Item -ItemType Directory -Force -Path $LocalCkptDir | Out-Null
    $LocalCkpt = Join-Path $LocalCkptDir "base_uttt.pt"

    Write-Host "[uTTT-Win10] Pulling $RemoteDir/checkpoints/base_uttt.pt -> $LocalCkpt..." -ForegroundColor Cyan
    $scpArgs = Get-SshArgs -ForScp $true
    & scp.exe @scpArgs "${RemoteUser}@${RemoteHost}:${RemoteDir}/checkpoints/base_uttt.pt" $LocalCkpt

    $LocalHash = (Get-FileHash -Path $LocalCkpt -Algorithm SHA256).Hash
    Write-Host "[uTTT-OK] Pulled Base Checkpoint ($LocalCkpt) | SHA256: $LocalHash" -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# Subcommand: serve (Detached FastAPI Server + SSH Port Forwarding to :8000)
# ------------------------------------------------------------------------------
function Invoke-Serve {
    Assert-Config
    Invoke-Sync
    $ExtraJoined = ($ExtraArgs -join " ")
    Write-Host "[uTTT-Win10] Launching remote FastAPI + WebSocket Personalization Service..." -ForegroundColor Cyan

    Invoke-RemotePS @"
Set-Location '$RemoteDir'
`$PidFile = '$RemoteDir/logs/serve.pid'
if (Test-Path `$PidFile) {
    `$OldPid = Get-Content `$PidFile -ErrorAction SilentlyContinue
    if (`$OldPid) { Stop-Process -Id `$OldPid -Force -ErrorAction SilentlyContinue }
}
`$PyExe = '$RemoteDir/.venv/Scripts/python.exe'
`$LogOut = '$RemoteDir/logs/serve_5090.log'
`$LogErr = '$RemoteDir/logs/serve_5090.err.log'
`$Proc = Start-Process -FilePath `$PyExe `
    -ArgumentList "-u -m server.app --host 0.0.0.0 --port $RemoteServicePort --config configs/5090_config.yaml $ExtraJoined" `
    -WorkingDirectory '$RemoteDir' `
    -RedirectStandardOutput `$LogOut `
    -RedirectStandardError `$LogErr `
    -WindowStyle Hidden `
    -PassThru
Set-Content -Path `$PidFile -Value `$Proc.Id
Write-Host "[Remote] Personalization Server launched (PID: `$(`$Proc.Id)) on port $RemoteServicePort."
"@

    Start-Sleep -Seconds 3
    Write-Host "[uTTT-OK] Opening SSH Tunnel: localhost:$LocalPort -> ${RemoteHost}:$RemoteServicePort" -ForegroundColor Green
    Write-Host "[uTTT-Info] In another terminal, run: python client_chat.py --user-id alex --ws-url ws://localhost:$LocalPort" -ForegroundColor Cyan
    $sshArgs = Get-SshArgs
    & ssh.exe @sshArgs -N -L "${LocalPort}:localhost:${RemoteServicePort}" "${RemoteUser}@${RemoteHost}"
}

# ------------------------------------------------------------------------------
# Subcommands: status, logs, stop
# ------------------------------------------------------------------------------
function Invoke-Status {
    Assert-Config
    Invoke-RemotePS @"
Write-Host "=== [GPU Status] ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
Write-Host "`n=== [Background Processes] ==="
foreach (`$name in @('train', 'serve')) {
    `$pf = "$RemoteDir/logs/`$name.pid"
    if (Test-Path `$pf) {
        `$p = Get-Process -Id (Get-Content `$pf) -ErrorAction SilentlyContinue
        if (`$p) { Write-Host "[RUNNING] `$name (PID: `$(`$p.Id))" } else { Write-Host "[STOPPED] `$name" }
    } else { Write-Host "[NONE] `$name" }
}
Write-Host "`n=== [Checkpoints] ==="
Get-ChildItem "$RemoteDir/checkpoints" -ErrorAction SilentlyContinue | Select-Object Name, Length, LastWriteTime
"@
}

function Invoke-Logs {
    Assert-Config
    $Mode = if ($ExtraArgs.Count -gt 0) { $ExtraArgs[0] } else { "train" }
    $LogFile = if ($Mode -eq "serve") { "$RemoteDir/logs/serve_5090.log" } else { "$RemoteDir/logs/train_5090.log" }
    Invoke-RemotePS "Get-Content -Path '$LogFile' -Wait -Tail 100"
}

function Invoke-Stop {
    Assert-Config
    $Mode = if ($ExtraArgs.Count -gt 0) { $ExtraArgs[0] } else { "all" }
    Invoke-RemotePS @"
`$targets = if ('$Mode' -eq 'all') { @('train', 'serve') } else { @('$Mode') }
foreach (`$t in `$targets) {
    `$pf = "$RemoteDir/logs/`$t.pid"
    if (Test-Path `$pf) {
        `$id = Get-Content `$pf -ErrorAction SilentlyContinue
        if (`$id) {
            Stop-Process -Id `$id -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped `$t (PID: `$id)"
        }
        Remove-Item -Force `$pf -ErrorAction SilentlyContinue
    }
}
"@
}

switch ($Command.ToLower()) {
    "init-env"  { Invoke-InitEnv }
    "sync"      { Invoke-Sync }
    "train"     { Invoke-Train }
    "pull-ckpt" { Invoke-PullCkpt }
    "serve"     { Invoke-Serve }
    "status"    { Invoke-Status }
    "logs"      { Invoke-Logs }
    "stop"      { Invoke-Stop }
    default     { Show-Usage }
}

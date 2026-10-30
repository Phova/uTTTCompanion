# Windows SSH → WSL 的独立训练所有者

在部分 Windows/WSL 配置中，SSH 会话关闭后，关联的 `wsl.exe` 也会退出。Linux 中的 `tmux -d` 只分离终端，不能保证 Windows 侧仍持有该发行版的前台调用。这种退出可以没有 CUDA OOM、Python traceback 或训练退出日志。

此工具从 Windows 的 `Win32_Process.Create` 启动隐藏的独立 `wsl.exe`。该客户端前台运行 `wsl_training_owner.py`；Python 创建真实的 `uttt_train` tmux pane，再以前台 `tmux wait-for` 等待训练结束。空闲等待不启动 `sleep` 进程。不要对正在运行的训练重复执行下面的启动命令；工具会拒绝接管已有 session。

## 在 Windows 宿主机启动

先确保本次代码已经同步到 WSL 的 `/root/uTTT/deploy/`。在**拥有 Ubuntu 发行版的 Windows 用户**下运行 PowerShell；可以从该 Windows 用户的 SSH 会话调用此启动器。调整快照目录和训练参数后执行：

```powershell
$TrainingArgs = @(
    '/root/uTTT/.venv/bin/python', '-u', 'deploy/run_training_pipeline.py',
    '--run-id', 'YOUR_RUN_ID', '--pretrain-steps', '100000', '--sft-steps', '6000'
)
.\deploy\start_wsl_training.ps1 `
    -Distribution Ubuntu -LinuxUser root `
    -Workspace /root/uTTT -RunDirectory /root/uTTT/runs/YOUR_SNAPSHOT `
    -Session uttt_train -LogPath /root/uTTT/logs/train_5090.log `
    -TrainingCommand $TrainingArgs
```

`TrainingCommand` 是 Linux 命令参数数组。请沿用该次运行原有的配置、数据、检查点和环境设置；示例参数不是对已有训练的自动恢复配置。需要环境变量时，可令数组以 `/usr/bin/env`、`NAME=value` 和真实 Python 路径开始。

`-DryRun` 只输出将创建的命令，便于检查引用和参数，不创建进程。正常执行返回 Windows `wsl.exe` PID；WMI 创建成功仅表示进程已创建，仍需检查下列 Linux 记录：

实际主机还发现一项 WSL 入口兼容问题：首个选项写成带引号的 `"-d"` 时，会被当作 Linux 命令并报 `/bin/bash: line 1: -d: command not found`，退出码为 127。启动器现对 `-d`、`-e`、普通路径等无需引用的参数保留原样；空参数、空白和双引号继续按 Windows 原生参数规则转义。Windows 上可用 `python -m unittest tests.test_wsl_launcher_quoting -v` 验证实际生成的命令和参数往返，不启动 WSL。

```bash
cat /root/uTTT/logs/.wsl_owners/uttt_train.json
cat /root/uTTT/logs/.wsl_owners/uttt_train.owner.log
tmux list-panes -t '=uttt_train:' -F '#{pane_pid} #{pane_current_command}'
tail -n 5 /root/uTTT/logs/train_5090.log
```

`phase: running` 表示启动屏障已就绪；它本身不证明模型已完成初始化。启动异常的完整 traceback 写入 owner 日志。断开原 SSH，再重新连接，确认同一 owner/pane 身份和训练步数仍持续推进，才能确认宿主机上的独立持有生效。

如果直接在 Linux 中测试所有者，其接口是：

```bash
python3 /root/uTTT/deploy/wsl_training_owner.py run \
  --workspace /root/uTTT --session uttt_train \
  --cwd /root/uTTT/runs/YOUR_SNAPSHOT \
  --log /root/uTTT/logs/train_5090.log \
  -- /root/uTTT/.venv/bin/python -u train_5090.py YOUR_EXISTING_ARGUMENTS
```

这条 Linux 命令自身不能替代 Windows 侧独立 WSL 客户端。

## 安全停止

```bash
python3 /root/uTTT/deploy/wsl_training_owner.py stop \
  --workspace /root/uTTT --session uttt_train --timeout 300
```

停止前校验 workspace、Linux 用户、启动时刻、boot ID、owner/pane 命令行、tmux token 和锁。校验通过后委托既有 `stop_training.py` 通知 pipeline/trainer，等待保存结束；不杀 owner，也不直接杀 tmux session。原有 `remote_ctl.sh stop train` 的保存后停止行为仍可使用。

每次训练有独立 UUID 通道。pane 退出时写结果并通知 owner；owner 确认后再放行 pane 退出，避免短任务退出过快导致 tmux 销毁通知。训练完成或正常停止后，owner 自动退出、释放锁，Windows 前台 `wsl.exe` 随后结束。保留 JSON 状态和诊断日志供检查。

收到 owner 的 TERM/INT/HUP 时，所有者先请求同样的安全停止。若保存超时或身份校验失败，所有者继续存活并记录错误，避免撤掉 WSL 的最后一个持有者。不要用 `taskkill`、`wsl --terminate`、`wsl --shutdown` 或 `tmux kill-session` 代替正常停止；Windows 重启与这些强制退出不在保存保证范围内。未知、陈旧或已被替换的 session 会被拒绝，工具不会自动接管。

## 不使用 GPU 的验证

```bash
python3 -m unittest tests.test_wsl_training_owner -v
```

可在独立的临时 workspace 和 `uttt_owner_smoke` session 中，将训练命令替换为 `/bin/true` 或 `/bin/false`，验证快速退出的结果/确认协议。不要对真实的 `uttt_train` 做冒烟测试，也不要停止正在运行的训练来验证此工具。

2026-10-31 实际验证：远端 Linux 的 9 项所有者测试全部通过；独立临时 tmux session 中的 `/bin/true`、`/bin/false` 实测均通过，覆盖成功和失败的快速退出。验证期间真实训练持续运行，未用于测试或被停止。

修复引用后，远端原生 Windows WMI 启动器的独立短任务也通过了参数实测：包含空格、双引号、尾反斜线、空字符串、`$()` 和单引号的参数逐项保持一致，owner 正确记录退出码 0。

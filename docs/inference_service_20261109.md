# 在本地WSL连接远端推理服务

先等待当前训练和评测结束。该服务与训练共用远端一张RTX 5090；新启动检查会拒绝本工作区训练期间开启服务。以下显式选择已保存的校正模型，它仍是质量未达标的实验候选。

本地WSL终端一：

```bash
cd /mnt/d/VibeCoding/uTTT
./deploy/remote_ctl.sh serve \
  --config configs/5090_foundation_corrected_20261109.yaml \
  --base-ckpt /mnt/c/uTTT-artifacts/checkpoints/20261109_foundation_corrected/uttt_1.584b_20261109_foundation_corrected.best.pt \
  --personas-dir personas/foundation_20261109 \
  --user-id Phova --max-gpu-users 1 \
  --max-new-tokens 256 --no-compile
```

这个命令会同步代码、启动远端服务并建立到本机8000端口的隧道；保持窗口打开。在本地WSL终端二运行：

```bash
cd /mnt/d/VibeCoding/uTTT
python3 client_chat.py --user-id Phova \
  --ws-url ws://localhost:8000 \
  --max-new-tokens 256 --temperature 0.3
```

客户端会发送生成长度和温度，故需要显式设置；否则其默认64 token/0.7会覆盖服务默认值。连接失败会明确报错并退出，不会自动改用本地模型；只有显式指定`--local`才进入本地模式。服务健康检查为：

```bash
curl http://localhost:8000/health
```

`--base-ckpt`是远端WSL路径，不能填本地项目路径。显式路径可避免按目录内`.pt`修改时间自动选错文件；`.best.pt`只含推理所需权重，比含Adam的完整状态更适合启动服务。新模型应使用独立personas目录，避免混入其他基座的旧状态。路径不存在会报错。

本轮CLI保留最近8轮文字上下文，模型token预算不足时丢最旧完整轮次。`/clear`开始新的文字话题，成功`/reset`同时重置快权重和文字历史；重启客户端不会持久保存文字历史。SFT模型的在线TTT仍关闭，同一user-id不代表已经获得长期记忆。

关闭终端一的Ctrl+C只会断开隧道，远端服务会继续占用显存。训练前停止服务：

```bash
./deploy/remote_ctl.sh stop serve
```

查看服务日志：

```bash
./deploy/remote_ctl.sh logs serve
```

此连接方式使用WSL版`.sh`控制器；`.ps1`版本面向原生Windows远端。若本机8000端口已占用，可设置 `LOCAL_TUNNEL_PORT=8001`启动serve，并把客户端和health地址一起改成8001。

本轮3000步训练未形成整体能力提升，故上述示例继续使用父corrected实验权重；具体得失见 [本轮结果](foundation_balanced_results_20261109.md)。HTTP或自写WebSocket客户端需显式传入history成对消息，服务不会只凭user-id自动聚合文字历史；上述CLI会自动维护近期上下文。

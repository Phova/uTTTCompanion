# Windows SSH 监控兼容修复

真实主机核查发现，Windows OpenSSH、默认 WSL shell 与 ConPTY 的组合会错误传递远端命令参数：同一条 `python3 --version` 在普通连接中返回版本，强制伪终端时却启动交互解释器。较大的内联脚本还会触及 Windows 命令长度限制，表现为 `exec request failed on channel 0`。仅修改频率标签不能解决这条启动链路。

现在 status、logs 和训练日志查看统一使用 `ssh -T`。小型 Python 监督器通过 stdin 的 EOF、断线和信号判断客户端退出，并只停止、回收自己启动的查看进程组。Bash coproc 在客户端父进程中持有唯一的 stdin 写端，不需要心跳进程、每帧 sleep 或临时 FIFO。关闭面板、强杀客户端父进程、训练结束及输出堵塞都有对应回归。

监控和监督器先经短命令上传，校验 SHA-256 后原子安装到内容寻址的 `.cache/remote-view/`。脚本正文通过 stdin 传输，查看请求只包含短路径与参数；不会为查看状态同步整个项目或改写训练快照。SSH 错误会返回真实失败码。

输出转发采用有界、非阻塞缓冲。退出时丢弃并排空未消费的子进程输出，给监控正常清理 GPU 采集进程的机会，避免 `flush()` 堵塞后被强杀而留下孙进程。面板宽度从客户端显式传递，防止取消伪终端后五行重绘换行错位。

`status 10` 显示目标 **100Hz** 和实测循环频率；这是面板刷新率，不保证 GPU 硬件传感器以同一频率生成新数据。

## 验证

- 19 项 Linux 监控/流连接回归全部通过，覆盖进程回收、上传失败、错误码、参数含空格、父进程 SIGKILL 以及输出堵塞时的采集器和孙进程清理。
- 在实际 Windows OpenSSH → WSL 主机测试两种退出方式：stdin EOF 和强制结束 SSH 客户端。两次均观察到目标100Hz，监控和 GPU 采集进程全部退出。
- 实际远端命令最大691字节，未把整个监控程序放入执行请求。原始结果见 [真实连接测试记录](windows_monitor_actual_result_20261030.json)。
- 包含CUDA训练/推理在内的完整回归166项全部通过，0跳过；`bash -n` 和差异检查通过。

本地更新后运行：

```bash
git pull --ff-only
./deploy/remote_ctl.sh status 10
```

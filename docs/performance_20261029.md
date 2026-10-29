# 2026-10-29 RTX 5090 训练与推理优化

使用同一台 RTX 5090 32 GiB、PyTorch 2.11.0+cu128、WSL，模型为 1,583,879,748 参数。原始数据保存在 `performance_measurements_20261029.json`；大模型完整 AdamW 步骤与推理各在独立进程运行。

## 测量与选择

| 项目 | 原配置 | 采用方案 | 变化 |
| --- | ---: | ---: | ---: |
| 训练吞吐 | 5548 token/s | 7376 token/s | +33% |
| 完整训练步骤（32768 token） | 5.91 s | 4.44 s | -25% |
| 训练峰值 allocated / reserved | 22.17 / 23.56 GiB | 25.59 / 26.93 GiB | 使用显存余量减少重算 |
| 模型解码（512 token提示、64次单token forward） | 71.4 token/s | 123.1 token/s | +72% |
| 推理峰值 allocated | 3.31 GiB | 3.36 GiB | 固定窗口图缓存 |

训练基准为32k词表、序列1024、BF16激活、FP32参数/梯度/AdamW，1步预热后测量3个完整优化步骤，包含梯度裁剪和AdamW；合成输入隔离数据读取开销，正式训练日志另行核验。解码测量包含模型和GPU greedy，排除CPU复制与文本解码；预热后取多次中位数。普通解码五次复测为66.8 token/s，存在CPU提交波动；表中采用原始71.4基线。图的首次建立约0.28秒，重复会话可复用。512-token预填充约23ms，基本不变。实际流式生成（包含逐token CPU输出和增量BPE解码）从65.7提升至复用图后的115.9–116.2 token/s；首次建立图的整轮约80.4 token/s，首token仍约24ms，因为捕获延后到首token输出之后。

最终语言训练使用 micro-batch=1、accumulation=32，关闭激活检查点并跨微批保留AMP权重缓存；优化器更新之前退出缓存上下文。不改变32序列的有效批次、模型结构或损失函数。冻结语言训练通过只读快权重视图，不创建无用动量和输出状态；默认公共API仍返回独立可变状态。元训练保留原有检查点与OOM保护。

曾测试的替代方案：隔层重算6286 token/s；batch2加隔层重算7379 token/s；batch4/8全重算约7080 token/s；batch2不重算reserved达到31.57GiB并出现严重显存压力，仅566 token/s，排除。局部训练编译7639 token/s，存在BF16数值轨迹差异且收益小，保持关闭。诊断显示约0.31%的梯度元素变号贡献了约96%的首步Adam更新差异能量，梯度的小舍入差异被Adam放大；SGD梯度/更新一致性通过，未宣称编译Adam与原训练等价。投影融合单独使用约65.7 token/s；配合CUDA Graph为127.4 token/s，但额外占约0.85GiB，默认选择普通投影图解码。

## 推理实现与边界

CUDA Graph只用于已冻结、实际FP16/BF16参数、CUDA、batch1、非推测解码。图使用固定窗口环形KV、有效槽位mask、GPU位置索引RoPE，采样和文本仍沿原逻辑。每次会话重置KV和位置；模型/快权重变更时重建。到RoPE缓存边界时导出按时间排序的KV并回退普通解码。训练验证的FP32/可训练参数会被拒绝，避免捕获陈旧参数。

服务默认启用图解码；`UTTT_CUDA_GRAPH=0`可关闭。局部小内核编译默认关闭，`UTTT_NO_COMPILE=0`可显式启用。逐token文本解码使用原生增量BPE解码器并兼容UTF-8跨token字节；旧分词器保留兼容路径。

完整CPU/CUDA回归65项通过，另验证两项最终流式图回归。数值验证覆盖冻结快权重梯度、跨多次优化更新的AMP缓存、检查点路径别名恢复、滑动窗口多次回绕、会话重置、用户/模型权重变更及图回退。大模型图解码在该固定输入下64个greedy输出与原方案完全一致；不同attention归约顺序可能产生正常BF16舍入差异。

## 复现

```bash
OMP_NUM_THREADS=2 .venv/bin/python deploy/benchmark_training_speed.py --source-root runs/20261029 --output logs/train_before.json
OMP_NUM_THREADS=2 .venv/bin/python deploy/benchmark_training_speed.py --no-checkpoint --amp-cache --output logs/train_after.json
OMP_NUM_THREADS=2 .venv/bin/python deploy/benchmark_inference_speed.py --checkpoint checkpoints/uttt_1.584b_20261029_pretrain.best.pt --no-packed-projections --cuda-graph --prompt-length 512 --decode-tokens 64 --repeats 5 --output logs/infer_graph.json
```

实际训练从第500步完整检查点恢复，基础权重、Adam状态、token计数与分词器继续保留；源码固定在 `runs/20261029_perf/`。旧快照保留用于复现和回退。原检查点只保存至500步，暂停前尚未保存的步骤会重做。
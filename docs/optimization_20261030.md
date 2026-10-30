# 2026-10-30：后续性能优化与验证

本轮在[架构审计修复](architecture_audit_20261030.md)基础上，完成可独立验证的推理、元训练和检查点开销优化。审计基线在本地为 `1fe1f55`，GitHub 的同一文件树为 [`1b08175`](https://github.com/Phova/uTTTCompanion/commit/1b08175a8a334ab31830a8971b1d6f808e60a56b)。正在运行的 1.584B 预训练继续使用固定快照 `2680c84`；本轮没有停止训练、重置优化器或更换其样本顺序。

## 测量结果

| 项目 | 基线 | 优化后 | 测量范围 |
|---|---:|---:|---|
| CUDA Graph 构建额外保留显存 | 186,693,120 B | 8,697,344 B | 单层代用模型，实际 `2560×6912` 快权重尺寸、FP32 人格 |
| 图解码 50 步输出 | 基线输出 | 全部 logits 逐位一致 | 固定相同权重、输入和位置，覆盖窗口回绕 |
| 元训练前向+反向中位耗时 | 0.7282 s | 0.6882 s | D128、4层、T128、chunk16，5次交替测量，约快 5.8% |
| 350万索引的 Python 对象峰值 | 140,157,848 B | 1,584 B | 另保留 28,033,064 B 原生索引数组 |
| 索引首次取批 | 88.6 ms | 34.5 ms | CPU 基准，随机排列完全相同 |
| 同 epoch 回滚后首次取批 | 110.7 ms | 0.021 ms | 重用已有排列 |
| 小模型 full+best 总文件体积 | 103,633,072 B | 70,866,613 B | 442万参数 CPU 检查点测试，加载后权重一致 |

图构建实际少占 **169.75 MiB**；消除的逻辑重复权重为168.75 MiB，分配器统计另有取整差异。新图复用预填充的 BF16/FP16 人格读取缓存，并保留 FP32 主权重及身份/版本失效检查。

单层解码时延在这次共享 GPU 测量中为0.586→0.596 ms/token，不能据此宣称稳定解码吞吐提高。图构建单次耗时0.353→0.315秒也只作诊断。此项验收依据是显存减少和新旧输出逐位相同。graph/eager 之间原有的 BF16 内核/归约差异保持一致，未放宽生产回归测试容差。

元训练基准只测小模型的前向与反向，不包含 Adam 步骤，不代表1.584B完整元训练吞吐。当前语言预训练关闭了 TTT 写入，也不会因该项优化获得相同比例的加速。原始结果见[图缓存测量](read_cache_benchmark_20261030.json)、[元训练测量](meta_pruning_benchmark_20261030.json)及[主机开销测量](host_perf_20261030.json)。

## 具体改动

**去掉无效辅助反向图。** 当 `surprise_aux_weight=0` 时，总损失直接使用 CE，避免计算 `0×Surprise` 仍触发完整高阶反向。公开的 `surprise_loss` 保持可微，非零辅助权重行为不变。多块输入的梯度和两步 AdamW 轨迹通过对照测试。只有单块输入、写入不可能影响 CE 时，未使用 writer 的梯度由人工零值变成 `None`，AdamW 正确跳过这些未参与目标的参数。

**减少检查点复制。** full 与 best 共用一次不可变 CPU 快照，保留绑权的 embedding/lm_head 存储共享；完整文件写完后先释放优化器快照，再写权重文件。按最大模型参数推算，每次同时保存 full+best 的模型复制量从约13.326 GB降至6.336 GB，每个文件少存约327.68 MB。这个数字是累计模型复制量，不是主机峰值 RAM 节省；优化器快照仍占较大内存。

**减少索引对象并保持恢复。** 缓存紧凑的同序排列，跨 DataLoader 队列时只创建当前小批量的 Python 索引。OOM 改批量或恢复到非整批偏移时，尾批边界按剩余样本计算，避免漏掉完整批次后卡在 epoch 末尾。

**保留恢复文件。** 更新 `.prev` 时先建立临时硬链接，再原子替换；硬链接失败保留旧恢复文件。后台写盘失败在下一步及时报告，失败的 `.tmp` 会清理。磁盘失败回归同时检查旧文件内容及不可变快照。

## WSL 显存容量保护

现场测量中，CUDA runtime 曾报告约31.9 GB空闲，而 `nvidia-smi` 显示全卡只剩约3 GB。容量测试现在核对所选 GPU UUID，取两者较小值，并扣除余量。父进程和直接 `--worker` 调用都在创建模型前检查；无法确定全局显存或设备映射时拒绝猜测。

运行正式训练期间，对最大候选执行检查得到 `insufficient_global_memory`，没有分配第二个大模型。显存检查是即时快照，不能作为跨进程显存预留。新版1.584B完整元训练的28 GiB上限仍需在GPU空闲后重新测量。

## 复现与测试

```bash
python deploy/benchmark_host_training.py --output logs/host_perf.json
python deploy/benchmark_meta_pruning.py --device cuda
git show 1b08175:engine/inference.py > /tmp/uttt-inference-baseline.py
python deploy/benchmark_graph_cache.py --baseline /tmp/uttt-inference-baseline.py --output logs/read_cache_perf.json
OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 python -m unittest discover -s tests -q
```

测量不会加载或改写正式检查点；CUDA 基准也核查全局余量。RTX 5090主机的完整回归 **158项通过，0跳过**，覆盖原有训练、元梯度、CUDA Graph、监控断线，以及新缓存共享、索引恢复、快照去重和全局显存保护。正式训练的最近核查点为17500步，验证 CE=3.3362，生成仍有重复，语言质量需要继续验收。

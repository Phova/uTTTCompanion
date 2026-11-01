# 2026-11-01：在既有 DeepInfer 优化之上的增量审计

基线为 `fd78f37`。此前已实现 CUDA Graph、有界环形 KV、GQA/FlexAttention、可选投影打包、增量 BPE 解码和实验性自投机；这些已有收益不再计入本轮。正式 1.584B 训练继续使用冻结快照 `a0643fd`，本轮没有热替换其代码或改变训练目标。验证优化会在下一次用新代码启动时生效。

## DeepInfer 中哪些思路仍有价值

官方项目名称为 **DeepInfer**。截至本次核查，开源版本针对 DeepSeek-V4.1-Flash，列出的 NVIDIA 硬件是 SM100、要求 CUDA 13；本项目为 RTX 5090 SM120、PyTorch 2.11.0+cu128。相同 Blackwell 家族不构成内核兼容承诺。[官方 README](https://github.com/deepseek-ai/DeepInfer#readme)

| 路径 | 本轮判定与行动 |
|---|---|
| 固定缓冲与图复用 | 已有。本轮修复模型归一化常量/窗口变化时的图失效，并在切换人格重新捕获前释放旧图，避免同时持有两份私有池。没有把已有解码加速再次作为新增收益。 |
| 请求独立随机流 | 已落地可选 `seed`，HTTP/WS → personalizer → 普通采样/投机验证均透传独立 `torch.Generator`。默认不传种子时兼容原行为。这改善隔离与复现，不声称增加 token/s。 |
| 跨请求缓存 | 修复满 GPU/CPU 缓存轮换时先淘汰待晋升人格的问题；两用户反复切换的 CPU 命中现在不再反复 `torch.load`，且保留元数据。仍会在驱逐时按原策略持久化，未宣称消除了所有磁盘 I/O。 |
| 异步调度/采样后处理重叠 | 下一步应先测 CPU 提交、采样和文本处理占比。现有同线程生成器、完整轮次互斥和写入依赖必须保留；不能直接去掉操作锁。尚未实现或给出速度预测。 |
| DSpark | 继续默认关闭。我们的草稿来自 `fast_pool(embedding)`，没有专门训练的 MTP 置信头；其最大 softmax 不是实测接收率。兼容保留 `dspark_*` 配置名，注释已改为实验性自投机，明确不是官方实现。 |
| SWA bounded replay | 不直接将输入裁成最后一个窗口。当前多层局部注意力的深层 KV 仍可能依赖更早输入；元训练后的动态记忆也有版本依赖。未来前缀缓存至少绑定模型指纹、快权重版本和 token 前缀，写入后失效。 |

DeepInfer 的官方随机采样使用序列种子与采样位置；本轮借鉴随机流隔离，使用 PyTorch generator，并未复制其计数式内核或承诺跨后端逐位相同。复现仍要求模型、人格权重、输入、采样参数和算子设置相同。[随机采样源码](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py)

官方 DSpark 使用 MTP 置信信息及运行耗时估计来分配验证长度；SWA 有界重放依赖其 YoCo consumer 和窗口/回退覆盖条件。这些不能由同名开关自动迁移。[验证长度调度](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/mtp_varlen.hpp)、[重放前提](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/worker/model_runner.py)

FlashMLA 面向不同的注意力/KV 格式，DeepGEMM 官方列出 SM90/SM100 与更高 CUDA 工具链，均不能作为当前 GQA/BF16/SM120 的即插即用替代；多卡 MoE 通信也不适用于此单卡 Dense 模型。[FlashMLA 要求](https://github.com/deepseek-ai/FlashMLA#requirements)、[DeepGEMM 要求](https://github.com/deepseek-ai/DeepGEMM#requirements)

请求示例：`{"user_id":"alice","prompt":"你好","max_new_tokens":64,"temperature":0.7,"seed":42}`。`seed` 可省略；取值为 0 至 `2**63-1` 的整数。目标采样的随机流不受其他请求的全局 RNG 消费影响；此能力不保证动态人格在更新前后返回相同文本。

## 本轮修复的隐患

- **BF16/FP16 RoPE 扩容精度**：低精度位置超过可精确表示的整数范围时会合并位置；现在用原 theta 在 FP32 生成新增相位，保留已有缓存的位值和输出精度。覆盖 BF16 的 4096+ 和 FP16 的 65536+ 位置，以及缓存/完整前缀一致性。
- **采样硬掩码**：有限的 `-1e9` 可能胜过更低的合法 logit，改用负无穷；无效采样参数明确报错。零生成预算不再无谓预填充。
- **文本完整性和输入模板**：保留合法 U+FFFD；尾部补齐基于未 trim 的原始解码偏移，避免前导空白导致漏尾字。长用户输入保留 BOS/用户角色、正文尾部和助手角色，预算不足在加载人格前报错。
- **人格事务与设备失败**：晋升/更新失败时保留可重试状态；卸载完成前不移除活动人格。有效磁盘人格迁移 OOM 不再被误当成文件损坏并静默重置。
- **漂移缓存**：不再仅依赖 step。用户原地修改或替换权重、基础 W0 变化都会使缓存失效；没有版本计数的 inference tensor 直接重算，避免 `/diff` 返回陈旧结果。
- **Docker 正常停止**：容器进程不属于宿主 tmux 的训练子树；现在单独验证工作区 checkpoint 挂载并绑定不可变容器 ID，发送 TERM、等待保存，超时保留容器。
- **完整续训**：权重文件缺少 Adam 时不再接续旧步数。恢复前检查优化器分组、已有状态形状和标量步数；需要重置优化器时使用 `--init-from`。无梯度 writer 不需要伪造 Adam 状态。

## 验证开销与下一步训练优化

语言 CE 验证复用一份只读快权重视图，省掉逐样本的 W 克隆、动量分配和重复显式精度转换；整轮 AMP 读取缓存在固定权重期间复用，退出后才恢复训练。完整元训练验证仍为每个样本创建独立的可变状态。吞吐日志按实际非 padding token 计数，短 SFT 和不足整批的数据不再夸大显示值。

32 个样本、长度 64、D128/2 层、CPU 两线程的最终对照，7 次中位耗时 **25.52 → 23.33 ms（约减少 8.6%）**，CE 均为 `6.256748899817467`；可变状态初始化 **32 → 0 次**，整轮只创建一次只读视图。小模型 CUDA BF16 验证 CE 也与旧路径一致。这是验证/主机开销实验，不是正式 1.584B 的 GPU 稳态吞吐测量。[原始记录](audit_measurements_20261101.json)

同一 CPU 工作负载与内外循环梯度检查可这样复现，基线文件应只来自可信 Git 提交：

```bash
git show fd78f37:train_5090.py > /tmp/uttt-validation-fd78f37.py
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv/bin/python deploy/benchmark_validation_overhead.py \
  --baseline-trainer /tmp/uttt-validation-fd78f37.py --output logs/validation_overhead.json
```

后续较有价值但未默认启用的训练实验：

1. 语言阶段将不用的 writer 保留在 CPU，理论释放 **1.661 GiB**；但历史 batch2 reserved 减去该理论值仍约 **29.91 GiB**，不能保证满足 28 GiB 预算或更快。
2. 每次 forward 内合并共享 up/gate 投影；必须保留 autograd，不能跨微批缓存其图。GEMM 形状变化需检查梯度及多步 AdamW 轨迹。
3. **优先评估选择性元训练**：从 SFT 底座出发，冻结注意力/embedding 等语言骨干，训练全部记忆 Q/K/V/O 投影、门控及 W0。参数账本为总量1,583,879,748，记忆相关可训练944,727,108，冻结639,152,640；相对全参数FP32 AdamW，理论少保留约 **7.143 GiB 的梯度和两份动量**，但动态状态链、激活和重算仍需实测，不能保证28 GiB可运行。冻结权重时仍须让梯度穿过其计算，不能对整段骨干套 `no_grad`。这是备选训练配方，当前未启用。
4. 为本项目重新定义并测量 aggregated 写入、分头或低秩记忆，以降低元训练的状态链开销。这些会改变内循环协议和学习行为，需要新的元训练、质量验收，不能直接复用旧人格状态。

## Surprise loss 与当前训练是否符合 uTTT

必须区分**内循环写入目标**和**外循环优化目标**：

\[
W_t = \operatorname{Update}(W_{t-1},\nabla_W L_{\mathrm{recon}}),\qquad
L_{\mathrm{outer}} = L_{\mathrm{CE}}.
\]

本项目按 `.prompt` 的 Dense/MSE 变体，用重构损失驱动写入；外层 CE 经后续读出和可微写入链反传。因此 `surprise_aux_loss_weight=0` 可以正确训练 uTTT，前提是 **`update_fast_weights=True` 且写入后确有受监督的读取**。不必再把同一 MSE 叠进外层；单独压低一个由可学习 K/V 定义的重构目标，甚至可能得到没有记忆价值的平凡解。论文同样区分写入和任务目标，但采用的关联损失/聚合等实验细节并不与本项目 MSE 顺序变体完全相同。[uTTT 论文](https://arxiv.org/abs/2610.05484)

真实小模型的多块梯度检查中，辅助权重为 0、内循环开启时，12 份 K/V/学习率/遗忘门参数梯度均非零，重构指标约 1.3252；关闭内循环时这些梯度全部不存在，指标为未测量的 0。这个检查证明梯度路径存在，不证明模型已学会有用记忆。

当前自动运行是 **pretrain → SFT → evaluate**，两个语言阶段都关闭写入；因此这是 uTTT 模型的语言准备阶段，最终 SFT 产物仍缺少经过训练的记忆写入能力。代码会继续拒绝为这类检查点启用在线 TTT。完整设计还需要：

1. 用实际 BPE、FP32 动态状态、完整 AdamW 和至少两个优化器步骤重新测量元训练容量。
2. 从合适的语言检查点初始化，在写入之后仍有监督、并确实要求跨块记忆的数据上元训练。
3. 对照动态写入与冻结读取的 CE，以及超过 SWA 范围的检索、纠正和偏好记忆；据此选择阈值、漂移限制和训练预算。

**1.584B 是已经测过的语言训练配置，不是已经验证满足 28 GiB 的完整元训练上限。** 本轮没有把原训练静默切换为元训练。

容量工具现在拒绝不足 `chunk_size+2` 的序列（只有第二块的末位但没有 next-token 目标也不够），拒绝单步容量结论，记录实际 Adam 状态元素数，并支持真实 BPE：

```bash
# 仅在训练释放 GPU 后执行；工具仍先检查全局可用显存。
OMP_NUM_THREADS=2 .venv/bin/python deploy/profile_capacity.py \
  --seq-len 1024 --budget-gb 28 \
  --tokenizer-path data/pretrain_20261029/tokenized/tokenizer.json \
  --output logs/meta_capacity_bpe.json
```

## 验证范围与上线状态

远端隔离目录的 CPU 全量回归 **248 项，232 通过、16 项按平台/CUDA 条件跳过**；另有 **12 项 CUDA 图/缓存测试通过、0 跳过**，并验证最终 BF16 CE 一致。小 GPU 测试设独立分配上限，未加载第二个正式模型；其分配器峰值 reserved 为252 MiB。测试的 `/proc` 轮询同时处理进程在打开和读取之间退出时的 ENOENT/ESRCH，避免将正常退出误报为失败。未据此证明最大模型完整元训练容量，也没有重新宣称已有 CUDA Graph 的吞吐收益。

本轮代码在下一次新启动或服务启动时采用。正在运行的预训练保留原冻结快照、优化器和数据顺序；正式训练没有为本次审计停机。

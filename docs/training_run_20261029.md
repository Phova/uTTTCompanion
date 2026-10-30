# 2026-10-29：新语料与 1.584B 训练任务

本轮使用 `configs/5090_pretrain_20261029.yaml` 和 `configs/5090_sft_20261029.yaml`，模型为 1,583,879,748 参数，FP32 参数/梯度、BF16 激活、micro-batch=1、梯度累积=32、序列长度1024。速度优化后关闭语言训练的激活重算，启用跨累积组 AMP 权重缓存，保持训练小内核编译关闭。本轮流水线只包含预训练、SFT 和质量评估；2026-10-30 修复后的完整元训练需另行测量显存，不能沿用旧版容量结论。

## 新数据

- 预训练：从 [OpenCSG Chinese FineWeb Edu v2](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu-v2) 选择12个中文分片，从 [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 选择3个英文分片，固定种子42。共15个分片、18.10 GB压缩原文，通过 ModelScope 镜像下载；每个文件均与上游清单的大小、SHA-256核对。
- 清洗：Unicode规范化、长度/异常字符过滤、中文教育评分至少0.75、英文教育整数评分至少3、整篇内容精确去重。
- 词表与划分：从保留的训练文档训练32000词 BPE；以清洗后的整篇文档哈希固定划分2%验证集，词表训练也排除这些验证文档。训练与验证分别写入 uint16 token 文件，以BOS/EOS标记文档边界。
- 指令微调：[COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) 中文43785条，[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) 的 smol-magpie-ultra、smol-constraints、everyday-conversations 英文52239条，共96024条，保留回答结束符并去重。来源、版权字段与文件校验值保存在数据目录的 `sources.json`。

预训练数据目录为 `data/pretrain_20261029/`，指令数据目录为 `data/instruction_20261029/`。最终 token 数、语言分布和文件哈希以 `data/pretrain_20261029/tokenized/manifest.json` 为准。所有大文件被 Git 忽略，训练读取磁盘映射文件，不把整套语料装入内存。

本轮实际保留3099686篇文档，去除17746篇精确重复文本；训练3037521篇、3584728951 token，验证62165篇、72720638 token。中文1311487243 token，英文2345962346 token。该计数包含每篇文档的BOS/EOS，模型监督时忽略BOS。

## 训练计划与验证

1. 从随机权重进行100000步预训练，输出 `checkpoints/uttt_1.584b_20261029_pretrain.pt`。
2. 仅在预训练正常达到目标步数后，加载验证集最佳基础权重，进行6000步指令微调，输出 `checkpoints/uttt_1.584b_20261029_sft.pt`。
3. 验证集最优文件另存为同名 `.best.pt`，保留最近完整检查点的 `.prev`。最终对最佳SFT权重运行实际问答评测，输出 `logs/quality_20261029.json`。

预训练每500步验证，SFT每250步验证，同时输出实际续写/问答样例。小惊奇度不作为成功标准。当前SFT模型不自动启用未经记忆训练的在线TTT写入。

原始短程实测为每步6.04秒、约5428 token/秒。2026-10-29速度优化后的独立完整AdamW步骤实测7376 token/秒、每步4.44秒（正式日志另行验证），100000步纯计算约5.14天；包括验证、保存和运行波动，预训练先估计5.5–7天，SFT再估计半天到一天。实际速度以正式日志为准。该计算预算不保证充分预训练或指定任务能力，后续应依据独立验证与实际输出决定是否继续。

## 查看、停止与继续

持久会话为 `uttt_train`，断开SSH不会终止任务。训练启动前会停止 `uttt_serve`，释放GPU。

在远端 WSL 中查看：

```bash
tail -f /root/uTTT/logs/train_5090.log
cat /root/uTTT/logs/pipeline_20261029.json
```

本地项目目录可用原有命令：

```bash
./deploy/remote_ctl.sh logs train
./deploy/remote_ctl.sh status --once
./deploy/remote_ctl.sh stop train
```

停止信号会等待正在写入的检查点完成，流水线标记中断，并阻止自动进入下一阶段。恢复从最近完整检查点继续；未保存的步骤需要重做。

在远端 WSL 中重新启动相同任务：

```bash
cd /root/uTTT
tmux new-session -d -s uttt_train '.venv/bin/python -u runs/20261029_perf/deploy/run_training_pipeline.py --run-id 20261029 --pretrain-steps 100000 --sft-steps 6000 >> logs/train_5090.log 2>&1'
```

同一 run-id 采用独占锁，拒绝多个进程同时写同一组检查点。来源与分词器指纹不匹配时拒绝恢复；不要更改既有分词器或 token 文件后继续同一任务。

实际运行使用 `runs/20261029_perf/` 中固定的优化源码快照（原 `runs/20261029/` 保留），数据、日志、虚拟环境与检查点链接到项目目录；后续同步代码会保留该快照，避免一周后进入SFT时意外使用另一版实现。恢复时也应使用快照中的 `runs/20261029_perf/deploy/run_training_pipeline.py`。

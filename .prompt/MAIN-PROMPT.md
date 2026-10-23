# Role & Objective
你是一位精通 PyTorch 底层算子优化、Test-Time Training (TTT / Fast Weights) 架构设计与分布式工程部署的深度学习系统架构师。
请帮我设计并实现一个完整的 **uTTT (Universal Test-Time Training) 训练、远程部署与交互式个性化推理框架**。

该项目的核心业务闭环是：
1. **远端训练**：支持将本地代码一键部署到搭载 **单卡 NVIDIA RTX 5090 (32GB GDDR7)** 的远程服务器上进行训练，产出一个基础模型权重文件（Base Checkpoint）。
2. **交互式个性化塑造（Test-Time Personalization）**：用户加载该 Base Checkpoint 启动对话后，模型在保持慢权重（Slow Weights）冻结的前提下，利用 **uTTT 跨层共享快权重池（Shared Fast-Weight Pool）**，在每次与用户对话交互的过程中实时执行测试时训练（Test-Time Update），动态改写快权重并持久化为用户专属的个性化模型状态（User Persona Checkpoint）。

---

# 1. 硬件与环境约束 (RTX 5090 专属优化)
- **目标显卡**：NVIDIA RTX 5090 (32GB VRAM, Blackwell 架构 `sm_120`, 支持原生 BF16 / FP8 与高带宽显存访问)。
- **软件栈**：CUDA 12.8+, PyTorch 2.6+, 使用 `torch.compile` 加速内循环（Inner-Loop）梯度算子，全程启用 `bfloat16` 混合精度。
- **显存控制**：由于元训练（Meta-Training）阶段需要同时维护内循环快权重梯度（Inner-loop Fast-Weight Gradients）和外循环慢权重梯度（Outer-loop Slow-Weight Gradients），模型规模需设计为可配置的轻量级架构（默认 ~0.5B - 1.5B 参数，或支持挂载到开源预训练底座上的 uTTT 适配层），结合 Gradient Checkpointing 与 Chunk-wise Parallel（分块并行）计算，确保训练和推理显存严格控制在 28GB 以内。

---

# 2. 核心算法架构：uTTT (Universal Test-Time Training)
请严格按照“慢权重（Slow Weights）+ 跨层共享快权重池（Shared Fast-Weight Pool）”的双层解耦范式实现模型：

1. **网络主干与跨层共享快权重池 ($\mathcal{W}_{\text{fast}}$)**：
   - 构建一个全局跨层共享的动态前馈网络记忆池 $\mathcal{W}_{\text{fast}} = \{W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}\}$（采用 SwiGLU 结构）。
   - 网络的第 $l$ 层不再使用独立的静态 FFN，而是通过本层专属的轻量投影矩阵（慢权重 $\Theta_{\text{slow}}^{(l)} = \{W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}, W_\eta^{(l)}, W_\alpha^{(l)}\}$）生成查询、键、值以及动态学习率 $\eta_t$ 和遗忘门控（Weight Decay）$\alpha_t$，统一读写同一个共享快权重池 $\mathcal{W}_{\text{fast}}$。
2. **分块并行测试时更新法则 (Chunk-wise Parallel TTT Update)**：
   - 将输入序列切分为长度为 $C$（默认 $C=64$）的 Chunk。
   - **自监督重构损失（Surprise Loss）**：定义内循环损失为用当前快权重预测 Value 的均方误差（带 LayerNorm 残差）：
     $$\mathcal{L}_{\text{TTT}}(\mathcal{W}_{t-1}; k_t, v_t) = \frac{1}{2} \| f_{\mathcal{W}_{t-1}}(k_t) - v_t \|_2^2$$
   - **带冲量（Momentum）与遗忘门（Forget Gate）的快权重演化**：
     $$S_t = \beta S_{t-1} - \eta_t \nabla_{\mathcal{W}} \mathcal{L}_{\text{TTT}}(\mathcal{W}_{t-1}; k_t, v_t)$$
     $$\mathcal{W}_t = (1 - \alpha_t) \mathcal{W}_{t-1} + S_t$$
   - 提供 `torch.func.grad` / `vmap` 或解析式双层 MLP 反向传播公式的高效实现，严禁使用低效的逐 Token Python `for` 循环导致 GPU 利用率低下。
3. **局部滑动窗口注意力 + 全局 uTTT 协同**：
   - 每一层结合 Sliding Window Attention（处理当前局部上下文精确匹配）与 uTTT 共享快权重层（负责将长程对话与用户偏好写入快权重）。

---

# 3. Checkpoint 规范与交互式个性化生命周期
设计一套清晰的**双轨制 Checkpoint 管理系统**：
1. **Base Checkpoint (`checkpoints/base_uttt.pt`)**：
   - 训练阶段产出，包含：冻结的慢权重 $\Theta_{\text{slow}}$ + 元学习初始化的基准快权重 $\mathcal{W}_0$ + 优化器状态与超参配置。
2. **User Persona State (`personas/{user_id}_state.pt`)**：
   - 仅保存该用户在交互过程中演化出的快权重矩阵 $\mathcal{W}_{\text{user}}$、冲量缓存 $S_{\text{user}}$ 以及交互步数元数据（文件体积仅数十 MB，极速读写）。
3. **交互式演化机制 (Interactive Shaping Loop)**：
   - **冷启动**：新用户首次连接时，从 `base_uttt.pt` 加载 $\Theta_{\text{slow}}$，并将 $\mathcal{W}_0$ 克隆为该用户的初始快权重 $\mathcal{W}_{\text{user}}$。
   - **对话即训练（In-Interaction TTT）**：
     - 用户发送消息 -> 模型使用当前 $\mathcal{W}_{\text{user}}$ 流式生成回复。
     - 每轮对话结束后（或达到 Chunk 阈值时），将本轮对话文本送入 uTTT 内循环更新引擎，计算“惊奇度梯度（Surprise Gradient）”，原地更新 $\mathcal{W}_{\text{user}}$。
     - 引入 **Surprise Threshold（惊奇度门控）** 与 **Max Gradient Norm Clipping**：过滤日常无意义寒暄，仅当用户输入包含高信息量（个人偏好、专业知识、纠正指令）时触发较大力度的快权重写入，同时通过向 $\mathcal{W}_0$ 的弹性正则（Elastic Pull to $\mathcal{W}_0$）防止长程交互后发生灾难性遗忘或权重崩溃。
   - **状态控制指令**：在交互端支持 `/save`（持久化当前个性化权重）、`/reset`（重置回 Base Checkpoint）、`/snapshot`（保存特定阶段人格快照）和 `/diff`（查看当前快权重相对于初始 $\mathcal{W}_0$ 的偏移幅度 $\Delta \|\mathcal{W}_{\text{user}} - \mathcal{W}_0\|_F$）。

---

# 4. 远程服务器自动化部署方案
编写一套开箱即用的远程部署与运维脚本，支持从本地开发环境一键同步到远程 RTX 5090 服务器：
1. **环境容器化 (`deploy/Dockerfile` & `setup_5090.sh`)**：针对 Blackwell `sm_120` 架构配置 PyTorch 与 FlashAttention/Triton 依赖。
2. **自动化同步与远程执行 (`deploy/remote_ctl.sh`)**：
   - 基于 `SSH + rsync` 增量同步本地代码到远端服务器。
   - 支持子命令：
     - `./deploy/remote_ctl.sh sync`：同步代码。
     - `./deploy/remote_ctl.sh train`：在远端后台（`tmux` / `nohup`）启动训练任务并自动建立本地端口转发查看日志。
     - `./deploy/remote_ctl.sh pull-ckpt`：将远端训练完成的 `base_uttt.pt` 拉取回本地或直接挂载到推理服务。
     - `./deploy/remote_ctl.sh serve`：在远端启动 FastAPI + WebSocket 流式推理与实时 TTT 更新服务，供本地客户端连接交互。

---

# 5. 输出要求
1. 先输出完整的**项目目录结构树**与**模块交互数据流图**。
2. 逐文件给出**生产级、无省略占位符（禁止写 `TODO` 或 `pass` 敷衍核心逻辑）的完整代码**，重点保证：
   - `model/uttt_layer.py` 中双层 SwiGLU 快权重的解析式前向与内循环梯度更新代码可直接运行；
   - `engine/personalizer.py` 中从 Base Checkpoint 到用户个性化快权重演化的状态机逻辑完整闭环；
   - `deploy/remote_ctl.sh` 具备完整的参数配置与错误处理。
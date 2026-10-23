现在请完整实现第二部分：`engine/checkpoint_manager.py`、`engine/personalizer.py` 以及 `server/app.py` (FastAPI + WebSocket)。
技术细节要求：
1. 实现从 `base_uttt.pt` 初始化用户会话的逻辑：多个并发用户共享同一份驻留在 5090 显存中的 `slow_weights`，但每个 `user_id` 拥有独立分配的 `fast_weights` 状态张量（并支持 LRU 卸载到 CPU 内存或磁盘 `personas/{user_id}.pt`）。
2. 在 `personalizer.py` 中实现 `interact_and_evolve(user_id, prompt)`：
   - 第一步（Forward）：冻结快权重，流式生成回复 Token；
   - 第二步（Test-Time Update）：将本轮 `[User Prompt + Assistant Reply]` 拼接后执行一步 Chunk TTT 更新；
   - 第三步（Stability Guard）：加入惊奇度门控（只有当本轮重构 Loss 大于阈值 $\tau$ 时才更新快权重）以及慢锚点弹性约束 $\mathcal{W}_{\text{new}} \leftarrow (1 - \lambda)\mathcal{W}_{\text{updated}} + \lambda \mathcal{W}_0$，防止长对话后模型胡言乱语。
3. 提供 CLI 客户端 `client_chat.py`，在每次对话后实时打印一行状态栏：`[TTT Status] Surprise Loss: 0.42 | Weight Drift from Base: 3.18% | Persona Saved`。
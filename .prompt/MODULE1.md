现在请完整实现第一部分：`model/uttt_layer.py`、`model/uttt_model.py` 和 `train_5090.py`。
技术细节要求：
1. 在 `uttt_layer.py` 中，不要使用高开销的 `autograd.grad` 逐 token 求导，请直接用矩阵乘法推导出双层 MLP（或 SwiGLU）关于重构损失 $\mathcal{L} = \frac{1}{2}\|f_W(k) - v\|_2^2$ 的**解析式块状梯度（Analytical Chunk-wise Gradient）**，保证在 RTX 5090 上可以用 Tensor Core 满负荷并行计算。
2. 实现跨层共享机制：`SharedFastWeightPool` 作为顶层 `nn.Module` 实例化一次，并作为状态张量按层顺序穿过所有 `uTTTBlock`（第 $l$ 层更新后的快权重 $\mathcal{W}^{(l)}$ 直接传递给第 $l+1$ 层或下一时间块）。
3. 在 `train_5090.py` 中配置 `torch.amp.autocast('cuda', dtype=torch.bfloat16)`、梯度累积和 Base Checkpoint 导出逻辑（严格保存 `slow_weights` 和元初始化的 `init_fast_weights`）。
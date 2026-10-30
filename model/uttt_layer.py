"""
model/uttt_layer.py
===================
Core operator and layer implementation for Universal Test-Time Training (uTTT).

Implements:
1. `FastWeightState`: State container for dynamic SwiGLU weights W_fast = {W_up, W_gate, W_down}
   and inner-loop momentum buffers S = {S_up, S_gate, S_down}.
2. `SharedFastWeightPool`: Top-level nn.Module instantiated once across the entire network,
   holding meta-initialized fast weights W_0 and analytical chunk-wise SwiGLU update kernels.
3. `analytical_swiglu_chunk_grad_and_update`: Closed-form Tensor-Core-friendly chunk-wise
   backpropagation for L_TTT(W; k, v) = 1/2 || f_W(k) - v ||_2^2 where
   f_W(x) = x + LN( (SiLU(x W_gate) * (x W_up)) W_down ), avoiding slow per-token autograd.grad.
4. `SlidingWindowAttention`: Local sliding-window GQA with RoPE and KV-cache support.
5. `uTTTLayer` & `uTTTBlock`: Dual-decoupled Slow-Weight projection + Shared Fast-Weight Pool
   read/write block with sequential cross-layer state passing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _HAS_FLEX_ATTENTION = True
except Exception:
    create_block_mask = None
    flex_attention = None
    _HAS_FLEX_ATTENTION = False

_FLEX_BLOCK_MASK_CACHE: Dict[Tuple[int, int, int, torch.device], Any] = {}
_COMPILED_FLEX_ATTN: Optional[Any] = None


def compile_with_backward_amp_off(function, **compile_kwargs):
    """Compile lazily with the non-autocast backward used by the trainer.

    The invocation needs the patch because torch.compile traces on first use.
    An unsupported PyTorch build must use its caller's eager/SDPA fallback.
    """
    from torch._functorch import config as aot_config
    if not hasattr(aot_config, "backward_pass_autocast"):
        raise RuntimeError("This PyTorch build cannot configure compiled-backward AMP semantics")
    compiled = torch.compile(function, **compile_kwargs)
    return aot_config.patch(backward_pass_autocast="off")(compiled)


# ==============================================================================
# 1. Fast-Weight State Container
# ==============================================================================

@dataclass
class FastWeightState:
    """
    Holds a user's or batch's active Fast Weights W_fast and Momentum buffers S.

    Shapes (batched for Tensor Core bmm):
        w_up,   s_up:   (B, d_model, d_ff)
        w_gate, s_gate: (B, d_model, d_ff)
        w_down, s_down: (B, d_ff, d_model)
        ln_weight:      (d_model,)
        ln_bias:        (d_model,)
    """
    w_up: torch.Tensor
    w_gate: torch.Tensor
    w_down: torch.Tensor
    s_up: torch.Tensor
    s_gate: torch.Tensor
    s_down: torch.Tensor
    ln_weight: torch.Tensor
    ln_bias: torch.Tensor
    step: int = 0
    last_surprise_loss: float = 0.0
    surprise_history: List[float] = field(default_factory=list)
    _init_state: Optional["FastWeightState"] = field(default=None, repr=False)
    _chunk_states: Optional[List["FastWeightState"]] = field(default=None, repr=False)
    _packed_up_gate: Optional[torch.Tensor] = field(default=None, repr=False)
    _read_w_down: Optional[torch.Tensor] = field(default=None, repr=False)
    _packed_key: Optional[tuple] = field(default=None, repr=False)

    def clone(self, detach: bool = False) -> "FastWeightState":
        """Deep-copy all fast weight and momentum tensors."""
        def _cp(t: torch.Tensor) -> torch.Tensor:
            return t.detach().clone() if detach else t.clone()

        return FastWeightState(
            w_up=_cp(self.w_up),
            w_gate=_cp(self.w_gate),
            w_down=_cp(self.w_down),
            s_up=_cp(self.s_up),
            s_gate=_cp(self.s_gate),
            s_down=_cp(self.s_down),
            ln_weight=_cp(self.ln_weight),
            ln_bias=_cp(self.ln_bias),
            step=int(self.step),
            last_surprise_loss=float(self.last_surprise_loss),
            surprise_history=list(self.surprise_history[-100:]),
        )

    def detach(self) -> "FastWeightState":
        """Detach fast weights and momentum buffers from the autograd graph."""
        return self.clone(detach=True)

    def to(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> "FastWeightState":
        """Move state tensors to target device and/or dtype."""
        kwargs = {"non_blocking": non_blocking}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype

        return FastWeightState(
            w_up=self.w_up.to(**kwargs),
            w_gate=self.w_gate.to(**kwargs),
            w_down=self.w_down.to(**kwargs),
            s_up=self.s_up.to(**kwargs),
            s_gate=self.s_gate.to(**kwargs),
            s_down=self.s_down.to(**kwargs),
            ln_weight=self.ln_weight.to(**kwargs),
            ln_bias=self.ln_bias.to(**kwargs),
            step=self.step,
            last_surprise_loss=self.last_surprise_loss,
            surprise_history=list(self.surprise_history),
        )

    def ensure_batch_size(self, batch_size: int) -> "FastWeightState":
        """Expand or slice batch dimension to match current input batch size."""
        cur_b = self.w_up.shape[0]
        if cur_b == batch_size:
            return self
        if cur_b == 1 and batch_size > 1:
            return FastWeightState(
                w_up=self.w_up.expand(batch_size, -1, -1).contiguous(),
                w_gate=self.w_gate.expand(batch_size, -1, -1).contiguous(),
                w_down=self.w_down.expand(batch_size, -1, -1).contiguous(),
                s_up=self.s_up.expand(batch_size, -1, -1).contiguous(),
                s_gate=self.s_gate.expand(batch_size, -1, -1).contiguous(),
                s_down=self.s_down.expand(batch_size, -1, -1).contiguous(),
                ln_weight=self.ln_weight,
                ln_bias=self.ln_bias,
                step=self.step,
                last_surprise_loss=self.last_surprise_loss,
                surprise_history=list(self.surprise_history),
            )
        raise ValueError(
            f"Cannot broadcast FastWeightState from batch_size={cur_b} to {batch_size}"
        )

    def to_state_dict(self) -> Dict[str, object]:
        """Serialize for lightweight User Persona Checkpoint persistence."""
        return {
            "w_up": self.w_up.detach().cpu(),
            "w_gate": self.w_gate.detach().cpu(),
            "w_down": self.w_down.detach().cpu(),
            "s_up": self.s_up.detach().cpu(),
            "s_gate": self.s_gate.detach().cpu(),
            "s_down": self.s_down.detach().cpu(),
            "ln_weight": self.ln_weight.detach().cpu(),
            "ln_bias": self.ln_bias.detach().cpu(),
            "step": self.step,
            "last_surprise_loss": self.last_surprise_loss,
            "surprise_history": list(self.surprise_history[-100:]),
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, object],
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FastWeightState":
        """Restore FastWeightState from a saved persona dictionary."""
        state = cls(
            w_up=state_dict["w_up"],
            w_gate=state_dict["w_gate"],
            w_down=state_dict["w_down"],
            s_up=state_dict["s_up"],
            s_gate=state_dict["s_gate"],
            s_down=state_dict["s_down"],
            ln_weight=state_dict["ln_weight"],
            ln_bias=state_dict["ln_bias"],
            step=int(state_dict.get("step", 0)),
            last_surprise_loss=float(state_dict.get("last_surprise_loss", 0.0)),
            surprise_history=list(state_dict.get("surprise_history", [])),
        )
        if device is not None or dtype is not None:
            state = state.to(device=device, dtype=dtype)
        return state


# ==============================================================================
# 2. Analytical Chunk-wise SwiGLU Forward & Inner-Loop Gradient Kernel
# ==============================================================================

def swiglu_fast_forward(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Forward pass through the SwiGLU Fast-Weight memory network with LayerNorm residual:
        u = x @ W_up
        g = x @ W_gate
        h = SiLU(g) * u
        z = h @ W_down
        f_W(x) = x + LayerNorm(z; ln_weight, ln_bias)

    Args:
        x:         (B, T, D) input queries or keys
        w_up:      (B, D, d_ff)
        w_gate:    (B, D, d_ff)
        w_down:    (B, d_ff, D)
        ln_weight: (D,)
        ln_bias:   (D,)
    Returns:
        out:       (B, T, D) reconstructed representations
    """
    orig_dtype = x.dtype
    u = torch.bmm(x, w_up.to(dtype=orig_dtype))                  # (B, T, d_ff) contiguous
    g = torch.bmm(x, w_gate.to(dtype=orig_dtype))                # (B, T, d_ff) contiguous
    h = F.silu(g) * u                                           # (B, T, d_ff)
    z = torch.bmm(h, w_down.to(dtype=orig_dtype))                # (B, T, D)
    # Single fused CUDA LayerNorm kernel (accumulates FP32 stats in registers)
    ln_out = F.layer_norm(
        z, (z.shape[-1],), ln_weight.to(orig_dtype), ln_bias.to(orig_dtype), eps
    )
    return x + ln_out


def swiglu_packed_forward(
    x: torch.Tensor, packed_up_gate: torch.Tensor, w_down: torch.Tensor,
    ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float = 1e-5,
) -> torch.Tensor:
    """Inference SwiGLU with one shared up/gate projection and fusible pointwise ops."""
    up, gate = torch.bmm(x, packed_up_gate.to(dtype=x.dtype)).chunk(2, dim=-1)
    projected = torch.bmm(F.silu(gate) * up, w_down.to(dtype=x.dtype))
    return x + F.layer_norm(projected, (projected.shape[-1],), ln_weight.to(x.dtype), ln_bias.to(x.dtype), eps)


def analytical_swiglu_chunk_grad(
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    eta_chunk: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    normalize_by_dim: bool = False,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Analytical Chunk-wise Gradient for the SwiGLU Fast-Weight Pool.

    Derives the exact matrix-multiplication backward pass for the self-supervised
    Surprise Loss over a chunk of C tokens:
        f_W(k_t) = k_t + LN( (SiLU(k_t W_gate) * (k_t W_up)) W_down )
        L_TTT(W; k_t, v_t) = 1/(2 * scale) || f_W(k_t) - v_t ||_2^2

    where each token's gradient is weighted by its dynamic learning rate eta_t
    and averaged over valid tokens in the chunk:
        G_W = (1 / C_valid) * sum_{t=1}^C eta_t * d L_TTT(W; k_t, v_t) / d W

    Mathematical Derivation:
        1. Forward:
           U = K @ W_up                                 in R^{B x C x d_ff}
           G = K @ W_gate                               in R^{B x C x d_ff}
           sigma_G = sigmoid(G),  A = G * sigma_G       in R^{B x C x d_ff}
           H = A * U                                    in R^{B x C x d_ff}
           Z = H @ W_down                               in R^{B x C x D}
           Z_hat = (Z - mu) * rstd                      in R^{B x C x D}
           V_hat = K + (gamma * Z_hat + beta)           in R^{B x C x D}

        2. Upstream Gradient w.r.t. V_hat (incorporating per-token eta_t / C_valid):
           delta_V = (eta / C_valid) * (V_hat - V) / scale

        3. Analytical LayerNorm Backward w.r.t. Z:
           dZ_hat = delta_V * gamma
           delta_Z = rstd * (dZ_hat - mean(dZ_hat) - Z_hat * mean(dZ_hat * Z_hat))

        4. Analytical Down-Projection Backward:
           grad_W_down = H^T @ delta_Z                  in R^{B x d_ff x D}
           delta_H     = delta_Z @ W_down^T             in R^{B x C x d_ff}

        5. Analytical SwiGLU Backward (SiLU'(g) = sigma(g) * (1 + g * (1 - sigma(g)))):
           delta_U     = delta_H * A                    in R^{B x C x d_ff}
           dSiLU       = sigma_G * (1.0 + G * (1.0 - sigma_G))
           delta_G     = delta_H * U * dSiLU            in R^{B x C x d_ff}

        6. Analytical Up/Gate-Projection Backward:
           grad_W_up   = K^T @ delta_U                  in R^{B x D x d_ff}
           grad_W_gate = K^T @ delta_G                  in R^{B x D x d_ff}

    Returns:
        grad_w_up:     (B, D, d_ff)
        grad_w_gate:   (B, D, d_ff)
        grad_w_down:   (B, d_ff, D)
        surprise_loss: scalar tensor (mean unweighted L_TTT across valid tokens)
    """
    orig_dtype = k_chunk.dtype
    compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    bsz, chunk_len, dim = k_chunk.shape

    # --- 1. Batched Forward Pass through SwiGLU Fast Weights (Contiguous cuBLAS bmm) ---
    # Matmul/readout use the activation dtype while W/S may be FP32 master state.
    # Match swiglu_fast_forward's activation rounding and retain all cast gradients.
    w_up_read = w_up.to(dtype=orig_dtype)
    w_gate_read = w_gate.to(dtype=orig_dtype)
    w_down_read = w_down.to(dtype=orig_dtype)
    u = torch.bmm(k_chunk, w_up_read)                           # (B, C, d_ff)
    g = torch.bmm(k_chunk, w_gate_read)                         # (B, C, d_ff)
    h = F.silu(g) * u                                          # (B, C, d_ff)

    z = torch.bmm(h, w_down_read)                               # (B, C, D)

    # LayerNorm statistics in FP32
    z_fp32 = z.to(dtype=compute_dtype)
    mean = z_fp32.mean(dim=-1, keepdim=True)
    diff = z_fp32 - mean
    var = diff.square().mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    z_hat = diff * rstd                                         # (B, C, D)

    gamma = ln_weight.to(dtype=compute_dtype).view(1, 1, dim)
    beta = ln_bias.to(dtype=compute_dtype).view(1, 1, dim)
    v_hat = k_chunk.to(dtype=compute_dtype) + (z_hat * gamma + beta)

    # --- 2. Surprise Loss: L_t = 1/2 || f_W(k_t) - v_t ||_2^2 ---
    err = v_hat - v_chunk.to(dtype=compute_dtype)                # (B, C, D)
    inv_scale = (1.0 / float(dim)) if normalize_by_dim else 1.0
    token_loss = 0.5 * err.square().sum(dim=-1) * inv_scale     # (B, C)

    if token_mask is not None:
        mask_fp32 = token_mask.to(dtype=compute_dtype)          # (B, C)
        valid_counts = mask_fp32.sum(dim=-1, keepdim=True).clamp_min(1.0)
        surprise_loss = (token_loss * mask_fp32).sum() / mask_fp32.sum().clamp_min(1.0)
        weight_factor = (eta_chunk.to(dtype=compute_dtype) * mask_fp32.unsqueeze(-1)) / valid_counts.unsqueeze(-1)
    else:
        surprise_loss = token_loss.mean()
        weight_factor = eta_chunk.to(dtype=compute_dtype) * (1.0 / float(chunk_len))

    # --- 3. Analytical Backward through Residual & LayerNorm ---
    # Keep the complete derivative graph: the outer objective differentiates
    # through these analytical inner gradients, including LayerNorm statistics.
    dz_hat = (weight_factor * inv_scale) * err * gamma          # (B, C, D)

    mean_dz_hat = dz_hat.mean(dim=-1, keepdim=True)
    mean_dz_hat_z = (dz_hat * z_hat).mean(dim=-1, keepdim=True)
    delta_z = (rstd * (dz_hat - mean_dz_hat - z_hat * mean_dz_hat_z)).to(orig_dtype)

    # --- 4. Analytical Backward through W_down ---
    # grad_W_down = H^T @ delta_Z -> (B, d_ff, C) @ (B, C, D) = (B, d_ff, D)
    grad_w_down = torch.bmm(h.transpose(1, 2), delta_z)
    # delta_H = delta_Z @ W_down^T -> (B, C, D) @ (B, D, d_ff) = (B, C, d_ff)
    delta_h = torch.bmm(delta_z, w_down_read.transpose(1, 2))

    # --- 5. Analytical Backward through SwiGLU Activation (SRAM register recomputation) ---
    g_fp32 = g.to(dtype=compute_dtype)
    u_fp32 = u.to(dtype=compute_dtype)
    delta_h_fp32 = delta_h.to(dtype=compute_dtype)
    sigma_g = torch.sigmoid(g_fp32)
    delta_u = (delta_h_fp32 * (g_fp32 * sigma_g)).to(orig_dtype)
    dsilu_fp32 = sigma_g * (1.0 + g_fp32 * (1.0 - sigma_g))
    delta_g = (delta_h_fp32 * u_fp32 * dsilu_fp32).to(orig_dtype)

    # --- 6. Analytical Backward through W_up and W_gate (Contiguous cuBLAS bmm) ---
    k_t = k_chunk.transpose(1, 2)                               # (B, D, C)
    grad_w_up = torch.bmm(k_t, delta_u)                         # (B, D, d_ff)
    grad_w_gate = torch.bmm(k_t, delta_g)                       # (B, D, d_ff)

    return grad_w_up, grad_w_gate, grad_w_down, surprise_loss


def fused_chunk_ttt_step(
    q_chunk: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    eta_chunk: torch.Tensor,
    alpha_chunk: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    s_up: torch.Tensor,
    s_gate: torch.Tensor,
    s_down: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    momentum_beta: float,
    max_inner_grad_norm: float,
    normalize_by_dim: bool,
    eps: float,
    read_w_up: Optional[torch.Tensor] = None,
    read_w_gate: Optional[torch.Tensor] = None,
    read_w_down: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    End-to-end pure-tensor Chunk TTT Step designed for whole-step `torch.compile` fusion:
    Fuses differentiable Analytical Gradient + Frobenius Norm Clipping +
    Momentum S_t + Decay W_t +
    Strictly Causal SwiGLU Fast-Weight Readout into a single compiled graph.
    """
    bsz = q_chunk.shape[0]
    grad_up, grad_gate, grad_down, surprise_loss = analytical_swiglu_chunk_grad(
        k_chunk=k_chunk,
        v_chunk=v_chunk,
        eta_chunk=eta_chunk,
        w_up=w_up,
        w_gate=w_gate,
        w_down=w_down,
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        token_mask=token_mask,
        normalize_by_dim=normalize_by_dim,
        eps=eps,
    )

    master_dtype = torch.float64 if w_up.dtype == torch.float64 else torch.float32
    if max_inner_grad_norm > 0.0:
        # 2-stage shape-aligned reduction across all 170 SMs (no `with torch.no_grad():` graph split)
        gu_master = grad_up.to(dtype=master_dtype)
        gg_master = grad_gate.to(dtype=master_dtype)
        gd_master = grad_down.to(dtype=master_dtype).reshape(bsz, w_up.shape[1], w_up.shape[2])
        grad_norm_sq = (
            (gu_master.square() + gg_master.square() + gd_master.square())
            .sum(dim=-1)
            .sum(dim=-1)
        )
        clip_view = (
            (max_inner_grad_norm * torch.rsqrt(grad_norm_sq + 1e-8))
            .clamp(max=1.0)
            .view(bsz, 1, 1)
        )
        step_up = grad_up.to(dtype=master_dtype) * clip_view
        step_gate = grad_gate.to(dtype=master_dtype) * clip_view
        step_down = grad_down.to(dtype=master_dtype) * clip_view
    else:
        step_up = grad_up
        step_gate = grad_gate
        step_down = grad_down

    # W/S are the master state. FP32 accumulation keeps sub-BF16-ULP decay and
    # increments across chunks; readout casts remain differentiable above.
    s_up_base = momentum_beta * s_up.to(dtype=master_dtype)
    s_gate_base = momentum_beta * s_gate.to(dtype=master_dtype)
    s_down_base = momentum_beta * s_down.to(dtype=master_dtype)

    s_up_new = s_up_base - step_up.to(dtype=master_dtype)
    s_gate_new = s_gate_base - step_gate.to(dtype=master_dtype)
    s_down_new = s_down_base - step_down.to(dtype=master_dtype)

    if token_mask is None:
        alpha_bar = alpha_chunk.to(dtype=master_dtype).mean(dim=1, keepdim=True)
    else:
        mask = token_mask.unsqueeze(-1).to(dtype=master_dtype)
        alpha_bar = ((alpha_chunk.to(dtype=master_dtype) * mask).sum(dim=1, keepdim=True)
                     / mask.sum(dim=1, keepdim=True).clamp_min(1))
    decay = 1.0 - alpha_bar

    w_up_new = decay * w_up.to(dtype=master_dtype) + s_up_new
    w_gate_new = decay * w_gate.to(dtype=master_dtype) + s_gate_new
    w_down_new = decay * w_down.to(dtype=master_dtype) + s_down_new

    if token_mask is not None:
        active = token_mask.any(dim=1).view(bsz, 1, 1)
        w_up_new = torch.where(active, w_up_new, w_up)
        w_gate_new = torch.where(active, w_gate_new, w_gate)
        w_down_new = torch.where(active, w_down_new, w_down)
        s_up_new = torch.where(active, s_up_new, s_up)
        s_gate_new = torch.where(active, s_gate_new, s_gate)
        s_down_new = torch.where(active, s_down_new, s_down)

    # A shared layer-write cursor may already contain writes from this chunk.
    # The scheduler supplies the immutable chunk-start weights for causal reads;
    # a standalone step defaults to its own pre-update weights.
    out_chunk = swiglu_fast_forward(
        x=q_chunk,
        w_up=w_up if read_w_up is None else read_w_up,
        w_gate=w_gate if read_w_gate is None else read_w_gate,
        w_down=w_down if read_w_down is None else read_w_down,
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        eps=eps,
    )
    return out_chunk, w_up_new, w_gate_new, w_down_new, s_up_new, s_gate_new, s_down_new, surprise_loss


# ==============================================================================
# 3. Cross-Layer Shared Fast-Weight Pool (W_fast)
# ==============================================================================

class SharedFastWeightPool(nn.Module):
    """
    Global cross-layer shared SwiGLU Fast-Weight Pool W_fast = {W_up, W_gate, W_down}.

    Instantiated once in `uTTTModel`. Holds the meta-learned initial fast weights W_0
    and performs analytical chunk-wise test-time updates as the state flows sequentially
    through all uTTTBlocks.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        momentum_beta: float = 0.90,
        max_inner_grad_norm: float = 1.0,
        normalize_loss_by_dim: bool = False,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.momentum_beta = momentum_beta
        self.max_inner_grad_norm = max_inner_grad_norm
        self.normalize_loss_by_dim = normalize_loss_by_dim
        self.eps = rms_norm_eps

        # Meta-initialized base Fast Weights W_0 (optimized during outer-loop meta-training)
        std_in = 1.0 / math.sqrt(d_model)
        std_out = 1.0 / math.sqrt(d_ff)
        self.w_up_init = nn.Parameter(torch.randn(d_model, d_ff) * std_in)
        self.w_gate_init = nn.Parameter(torch.randn(d_model, d_ff) * std_in)
        # Initialize w_down small so initial residual f_{W_0}(x) = x + LN(z) starts near identity
        self.w_down_init = nn.Parameter(torch.randn(d_ff, d_model) * (0.1 * std_out))

        # Shared LayerNorm parameters inside the residual reconstruction f_W(k)
        self.ln_weight = nn.Parameter(torch.ones(d_model))
        self.ln_bias = nn.Parameter(torch.zeros(d_model))

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        detach: bool = False,
    ) -> FastWeightState:
        """
        Create initial FastWeightState cloned/expanded from meta-learned W_0.

        - During Meta-Training (`detach=False`), W_0 remains connected to the autograd graph
          so outer-loop gradients update `w_up_init`, `w_gate_init`, `w_down_init`.
        - During Test-Time Personalization (`detach=True`), creates an independent copy
          of W_0 for a user's persona session.
        """
        target_device = device if device is not None else self.w_up_init.device
        target_dtype = dtype if dtype is not None else self.w_up_init.dtype

        w_up = self.w_up_init.to(device=target_device, dtype=target_dtype)
        w_gate = self.w_gate_init.to(device=target_device, dtype=target_dtype)
        w_down = self.w_down_init.to(device=target_device, dtype=target_dtype)
        ln_w = self.ln_weight.to(device=target_device, dtype=target_dtype)
        ln_b = self.ln_bias.to(device=target_device, dtype=target_dtype)

        if detach:
            w_up = w_up.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
            w_gate = w_gate.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
            w_down = w_down.detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
            ln_w = ln_w.detach().clone()
            ln_b = ln_b.detach().clone()
        else:
            w_up = w_up.unsqueeze(0).expand(batch_size, -1, -1).clone()
            w_gate = w_gate.unsqueeze(0).expand(batch_size, -1, -1).clone()
            w_down = w_down.unsqueeze(0).expand(batch_size, -1, -1).clone()

        s_up = torch.zeros_like(w_up)
        s_gate = torch.zeros_like(w_gate)
        s_down = torch.zeros_like(w_down)

        return FastWeightState(
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            s_up=s_up,
            s_gate=s_gate,
            s_down=s_down,
            ln_weight=ln_w,
            ln_bias=ln_b,
            step=0,
            last_surprise_loss=0.0,
        )

    def readonly_state(self, batch_size=1, device=None, dtype=None, detach=False) -> FastWeightState:
        """Internal frozen readout views; never use as a mutable persona state."""
        def view(tensor):
            value = tensor.to(device=device, dtype=dtype)
            return value.detach() if detach else value
        up, gate, down = (view(tensor).unsqueeze(0).expand(batch_size, -1, -1)
                          for tensor in (self.w_up_init, self.w_gate_init, self.w_down_init))
        zero = up.new_zeros(())
        return FastWeightState(
            up, gate, down, zero.expand_as(up), zero.expand_as(gate), zero.expand_as(down),
            view(self.ln_weight), view(self.ln_bias),
        )

    def forward_readout(self, x: torch.Tensor, state: FastWeightState) -> torch.Tensor:
        """Read out from current FastWeightState without modifying W_fast."""
        state = state.ensure_batch_size(x.shape[0])
        weights = (state.w_up, state.w_gate, state.w_down)
        if (getattr(self, "fused_inference_readout", False) and not self.training
                and not torch.is_grad_enabled() and not any(tensor.is_inference() for tensor in weights)):
            # Persona writes accumulate in FP32. Reuse activation-precision read
            # copies across layers/tokens while keeping every master bit intact.
            key = (x.dtype, x.device, tuple(
                (id(tensor), tensor.data_ptr(), tensor._version, tensor.device, tensor.dtype, tuple(tensor.shape))
                for tensor in weights
            ))
            if state._packed_up_gate is None or state._read_w_down is None or state._packed_key != key:
                state._packed_up_gate = torch.cat(
                    (state.w_up.to(dtype=x.dtype), state.w_gate.to(dtype=x.dtype)), dim=-1
                )
                state._read_w_down = state.w_down.to(dtype=x.dtype)
                state._packed_key = key
            return swiglu_packed_forward(
                x, state._packed_up_gate, state._read_w_down, state.ln_weight, state.ln_bias, self.eps
            )
        return swiglu_fast_forward(
            x=x,
            w_up=state.w_up,
            w_gate=state.w_gate,
            w_down=state.w_down,
            ln_weight=state.ln_weight,
            ln_bias=state.ln_bias,
            eps=self.eps,
        )

    def chunk_step(
        self,
        state: FastWeightState,
        q_chunk: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        eta_chunk: torch.Tensor,
        alpha_chunk: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        update_fast_weights: bool = True,
        read_state: Optional[FastWeightState] = None,
    ) -> Tuple[torch.Tensor, FastWeightState, torch.Tensor]:
        """
        Execute one Chunk-wise Parallel TTT step:
        1. Compute analytical SwiGLU gradients on (k_chunk, v_chunk) using W_{t-1}.
        2. Clip inner-loop gradient Frobenius norm to `max_inner_grad_norm`.
        3. Update Momentum S_t and Fast Weights W_t:
               S_t = beta * S_{t-1} - eta_t * grad_W L_TTT(W_{t-1}; k_t, v_t)
               W_t = (1 - alpha_t) * W_{t-1} + S_t
        4. Read out using read_state (the causal chunk-start snapshot), or the
           incoming write state when no separate snapshot was provided:
               o_chunk = f_{W_{t-1}}(q_chunk)
        """
        bsz = q_chunk.shape[0]
        state = state.ensure_batch_size(bsz)
        read_state = state if read_state is None else read_state.ensure_batch_size(bsz)

        if not update_fast_weights:
            # Frozen fast-weight inference mode (used during token streaming generation):
            # Only read out f_{W_t}(q_chunk); skip redundant k_chunk readout & syncs
            out = self.forward_readout(q_chunk, read_state)
            surprise_loss = q_chunk.new_zeros((), dtype=torch.float32)
            return out, state, surprise_loss

        out_chunk, w_up_new, w_gate_new, w_down_new, s_up_new, s_gate_new, s_down_new, surprise_loss = (
            fused_chunk_ttt_step(
                q_chunk=q_chunk,
                k_chunk=k_chunk,
                v_chunk=v_chunk,
                eta_chunk=eta_chunk,
                alpha_chunk=alpha_chunk,
                w_up=state.w_up,
                w_gate=state.w_gate,
                w_down=state.w_down,
                s_up=state.s_up,
                s_gate=state.s_gate,
                s_down=state.s_down,
                ln_weight=state.ln_weight,
                ln_bias=state.ln_bias,
                token_mask=token_mask,
                momentum_beta=self.momentum_beta,
                max_inner_grad_norm=float(self.max_inner_grad_norm or 0.0),
                normalize_by_dim=self.normalize_loss_by_dim,
                eps=self.eps,
                read_w_up=read_state.w_up,
                read_w_gate=read_state.w_gate,
                read_w_down=read_state.w_down,
            )
        )

        # Defer scalar GPU->CPU transfer to uTTTModel.forward end-of-forward vector sync
        new_state = FastWeightState(
            w_up=w_up_new,
            w_gate=w_gate_new,
            w_down=w_down_new,
            s_up=s_up_new,
            s_gate=s_gate_new,
            s_down=s_down_new,
            ln_weight=state.ln_weight,
            ln_bias=state.ln_bias,
            step=state.step + 1,
            last_surprise_loss=state.last_surprise_loss,
            surprise_history=state.surprise_history,
        )
        return out_chunk, new_state, surprise_loss

    @torch.no_grad()
    def apply_elastic_pull(
        self,
        state: FastWeightState,
        anchor_lambda: float = 0.02,
        max_drift_ratio: Optional[float] = None,
    ) -> FastWeightState:
        """
        Slow-Anchor Elastic Constraint (Stability Guard):
            W_new <- (1 - lambda) * W_updated + lambda * W_0
        Optionally enforces a hard safety ceiling on ||W_new - W_0||_F / ||W_0||_F.
        """
        if anchor_lambda <= 0.0 and max_drift_ratio is None:
            return state

        dev = state.w_up.device
        dt = state.w_up.dtype
        w_up_0 = self.w_up_init.detach().to(device=dev, dtype=dt).unsqueeze(0)
        w_gate_0 = self.w_gate_init.detach().to(device=dev, dtype=dt).unsqueeze(0)
        w_down_0 = self.w_down_init.detach().to(device=dev, dtype=dt).unsqueeze(0)

        w_up = (1.0 - anchor_lambda) * state.w_up + anchor_lambda * w_up_0
        w_gate = (1.0 - anchor_lambda) * state.w_gate + anchor_lambda * w_gate_0
        w_down = (1.0 - anchor_lambda) * state.w_down + anchor_lambda * w_down_0

        if max_drift_ratio is not None and max_drift_ratio > 0.0:
            base_norm = torch.sqrt(
                w_up_0.float().pow(2).sum()
                + w_gate_0.float().pow(2).sum()
                + w_down_0.float().pow(2).sum()
                + 1e-8
            )
            diff_up = w_up - w_up_0
            diff_gate = w_gate - w_gate_0
            diff_down = w_down - w_down_0
            drift_norm = torch.sqrt(
                diff_up.float().pow(2).sum(dim=(1, 2), keepdim=True)
                + diff_gate.float().pow(2).sum(dim=(1, 2), keepdim=True)
                + diff_down.float().pow(2).sum(dim=(1, 2), keepdim=True)
                + 1e-8
            )
            max_allowed = max_drift_ratio * base_norm
            scale = torch.clamp(max_allowed / drift_norm, max=1.0).to(dt)
            w_up = w_up_0 + diff_up * scale
            w_gate = w_gate_0 + diff_gate * scale
            w_down = w_down_0 + diff_down * scale

        return FastWeightState(
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            s_up=state.s_up,
            s_gate=state.s_gate,
            s_down=state.s_down,
            ln_weight=state.ln_weight,
            ln_bias=state.ln_bias,
            step=state.step,
            last_surprise_loss=state.last_surprise_loss,
            surprise_history=list(state.surprise_history),
        )

    @torch.no_grad()
    def compute_drift(self, state: FastWeightState) -> Dict[str, float]:
        """
        Compute Frobenius norm drift between current user fast weights W_user and base W_0:
            Delta ||W_user - W_0||_F and percentage drift relative to ||W_0||_F.
        """
        dev = state.w_up.device
        w_up_0 = self.w_up_init.detach().to(device=dev).float().unsqueeze(0)
        w_gate_0 = self.w_gate_init.detach().to(device=dev).float().unsqueeze(0)
        w_down_0 = self.w_down_init.detach().to(device=dev).float().unsqueeze(0)

        # Single vectorized GPU->CPU sync instead of 6 separate .item() calls
        norms = torch.stack(
            [
                (state.w_up.float() - w_up_0).norm(p="fro"),
                (state.w_gate.float() - w_gate_0).norm(p="fro"),
                (state.w_down.float() - w_down_0).norm(p="fro"),
                w_up_0.norm(p="fro"),
                w_gate_0.norm(p="fro"),
                w_down_0.norm(p="fro"),
            ]
        ).tolist()
        d_up, d_gate, d_down, b_up, b_gate, b_down = [float(x) for x in norms]

        total_drift = math.sqrt(d_up ** 2 + d_gate ** 2 + d_down ** 2)
        total_base = math.sqrt(b_up ** 2 + b_gate ** 2 + b_down ** 2) + 1e-8
        drift_pct = (total_drift / total_base) * 100.0

        return {
            "drift_frobenius": total_drift,
            "base_frobenius": total_base,
            "drift_pct": drift_pct,
            "w_up_drift": d_up,
            "w_gate_drift": d_gate,
            "w_down_drift": d_down,
            "w_up_drift_pct": (d_up / (b_up + 1e-8)) * 100.0,
            "w_gate_drift_pct": (d_gate / (b_gate + 1e-8)) * 100.0,
            "w_down_drift_pct": (d_down / (b_down + 1e-8)) * 100.0,
            "step": state.step,
            "last_surprise_loss": state.last_surprise_loss,
        }


# ==============================================================================
# 4. Normalization, RoPE & Sliding Window Attention (Local Exact Context)
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.weight.shape[0],), self.weight.to(x.dtype), self.eps)
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_normed).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        if start_pos + seq_len > self.cos_cached.shape[0]:
            self._build_cache(start_pos + seq_len + 512)
        cos = self.cos_cached[start_pos : start_pos + seq_len].to(dtype=x.dtype)
        sin = self.sin_cached[start_pos : start_pos + seq_len].to(dtype=x.dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors of shape (B, H, T, D_h)."""
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d_half = x.shape[-1] // 2
        x1 = x[..., :d_half]
        x2 = x[..., d_half:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class SlidingWindowAttention(nn.Module):
    """
    Local Sliding Window Grouped-Query Attention (SWA) with DeepInfer Optimizations:
        1. Fused QKV Projection (`F.linear` over concatenated [wq, wk, wv]).
        2. Native GQA (`enable_gqa=True`) avoiding 4x KV `repeat_interleave` VRAM copies.
        3. Hardware FlashAttention-2 fast-path (`is_causal=True` when within window) +
           cached sliding-window mask buffer.
        4. DeepInfer `SWA Bounded Replay`: caps KV replay tail to `window_size` during
           long-context prefill/decoding while earlier chunks reside in `SharedFastWeightPool`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int = 512,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.dropout = dropout

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_mask_key: Optional[Tuple[int, int, torch.device]] = None
        self.register_buffer("_packed_qkv", None, persistent=False)
        self._packed_qkv_key = None

    def _projection_key(self):
        if any(weight.is_inference() for weight in (self.wq.weight, self.wk.weight, self.wv.weight)):
            return None
        return tuple((id(weight), weight._version, weight.device, weight.dtype)
                     for weight in (self.wq.weight, self.wk.weight, self.wv.weight))

    @torch.no_grad()
    def prepare_for_inference(self):
        key = self._projection_key()
        if key is None:
            self._packed_qkv = None
            self._packed_qkv_key = None
            return
        if self._packed_qkv is None or key != self._packed_qkv_key:
            self._packed_qkv = torch.cat((self.wq.weight, self.wk.weight, self.wv.weight), dim=0)
            self._packed_qkv_key = key

    def train(self, mode=True):
        if mode:
            self._packed_qkv = None
            self._packed_qkv_key = None
        return super().train(mode)

    def _get_sliding_causal_mask(
        self, seq_len: int, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        key = (seq_len, kv_len, device)
        if self._cached_mask is None or self._cached_mask_key != key:
            q_idx = torch.arange(kv_len - seq_len, kv_len, device=device).unsqueeze(1)
            k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
            self._cached_mask = (q_idx >= k_idx) & ((q_idx - k_idx) < self.window_size)
            self._cached_mask_key = key
        return self._cached_mask

    def _get_flex_block_mask(
        self, seq_len: int, kv_len: int, device: torch.device
    ) -> Any:
        key = (seq_len, kv_len, self.window_size, device)
        bm = _FLEX_BLOCK_MASK_CACHE.get(key)
        if bm is None and create_block_mask is not None:
            win = self.window_size
            offset = kv_len - seq_len

            def _swa_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
                q_pos = q_idx + offset
                return (q_pos >= kv_idx) & ((q_pos - kv_idx) < win)

            bm = create_block_mask(
                _swa_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=kv_len, device=device
            )
            _FLEX_BLOCK_MASK_CACHE[key] = bm
        return bm

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = x.shape

        if self._packed_qkv is not None and not self.training and not torch.is_grad_enabled():
            self.prepare_for_inference()
        if self._packed_qkv is not None and not self.training and not torch.is_grad_enabled():
            q_raw, k_raw, v_raw = F.linear(x, self._packed_qkv).split(
                (self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim, self.n_kv_heads * self.head_dim), dim=-1
            )
        else:
            q_raw, k_raw, v_raw = self.wq(x), self.wk(x), self.wv(x)
        q = q_raw.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, start_pos=start_pos)
        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            # Retain enough history for the FIRST query in a multi-token decode.
            # Only the returned cache is capped to the final window below.
            needed = self.window_size - 1 + seq_len
            if k.shape[2] > needed:
                k = k[:, :, -needed:, :]
                v = v[:, :, -needed:, :]

        # Bound stored KV cache to window_size even on initial long-context prefill
        if not self.training and k.shape[2] > self.window_size:
            new_kv_cache = (k[:, :, -self.window_size :, :].clone(memory_format=torch.contiguous_format),
                            v[:, :, -self.window_size :, :].clone(memory_format=torch.contiguous_format))
        else:
            new_kv_cache = (k, v)

        kv_len = k.shape[2]
        drop_p = self.dropout if self.training else 0.0

        # Native PyTorch 2.5+ GQA + Triton FlexAttention BlockMask (16.8x faster than bool mask SDPA)
        try:
            if seq_len == 1:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=(self.n_rep > 1)
                )
            elif kv_cache is None and kv_len <= self.window_size:
                # Native FlashAttention-2 causal fast-path when sequence fits within window
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=drop_p, is_causal=True, enable_gqa=(self.n_rep > 1)
                )
            elif (
                _HAS_FLEX_ATTENTION
                and x.is_cuda
                and drop_p == 0.0
                and seq_len >= 128
                and kv_len >= 128
            ):
                global _COMPILED_FLEX_ATTN
                if _COMPILED_FLEX_ATTN is None:
                    _COMPILED_FLEX_ATTN = compile_with_backward_amp_off(flex_attention)
                block_mask = self._get_flex_block_mask(seq_len, kv_len, x.device)
                attn_out = _COMPILED_FLEX_ATTN(
                    q, k, v, block_mask=block_mask, enable_gqa=(self.n_rep > 1)
                )
            else:
                attn_mask = self._get_sliding_causal_mask(seq_len, kv_len, x.device)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=drop_p, is_causal=False, enable_gqa=(self.n_rep > 1)
                )
        except Exception:
            # Fallback for older PyTorch or CPU
            k_attn = k.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else k
            v_attn = v.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else v
            if seq_len == 1:
                attn_out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn, attn_mask=None, dropout_p=0.0, is_causal=False
                )
            elif kv_cache is None and kv_len <= self.window_size:
                attn_out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn, attn_mask=None, dropout_p=drop_p, is_causal=True
                )
            else:
                attn_mask = self._get_sliding_causal_mask(seq_len, kv_len, x.device)
                attn_out = F.scaled_dot_product_attention(
                    q, k_attn, v_attn, attn_mask=attn_mask, dropout_p=drop_p, is_causal=False
                )

        out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.wo(out), new_kv_cache


# ==============================================================================
# 5. Layer-Specific Slow-Weight uTTT Projection Layer & uTTTBlock
# ==============================================================================

def fused_fast_q_proj(
    x: torch.Tensor,
    wq_w: torch.Tensor,
    q_norm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Fast query-only projection + RMSNorm for frozen fast-weight inference (`update_fast_weights=False`),
    skipping unused K, V, eta, and alpha projections during Step 1 token streaming.
    """
    dim = x.shape[-1]
    q_raw = F.linear(x, wq_w)
    if hasattr(F, "rms_norm"):
        return F.rms_norm(q_raw, (dim,), q_norm_w.to(q_raw.dtype), eps)
    q_fp32 = q_raw.float()
    return (q_norm_w.float() * q_fp32 * torch.rsqrt(q_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


def fused_slow_projections(
    x: torch.Tensor,
    wq_w: torch.Tensor,
    wk_w: torch.Tensor,
    wv_w: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    w_eta_w: torch.Tensor,
    w_eta_b: torch.Tensor,
    w_alpha_w: torch.Tensor,
    w_alpha_b: torch.Tensor,
    base_inner_lr: float,
    base_weight_decay: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure-tensor slow-weight projection + RMSNorm + gate activation kernel for `uTTTLayer`,
    compiled by `torch.compile` to eliminate eager launch gaps between SWA and TTT chunks.
    """
    dim = x.shape[-1]
    q_raw = F.linear(x, wq_w)
    k_raw = F.linear(x, wk_w)
    v_raw = F.linear(x, wv_w)

    if hasattr(F, "rms_norm"):
        q = F.rms_norm(q_raw, (dim,), q_norm_w.to(q_raw.dtype), eps)
        k = F.rms_norm(k_raw, (dim,), k_norm_w.to(k_raw.dtype), eps)
        v = F.rms_norm(v_raw, (dim,), None, eps)
    else:
        q_fp32 = q_raw.float()
        q = (q_norm_w.float() * q_fp32 * torch.rsqrt(q_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)
        k_fp32 = k_raw.float()
        k = (k_norm_w.float() * k_fp32 * torch.rsqrt(k_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)
        v_fp32 = v_raw.float()
        v = (v_fp32 * torch.rsqrt(v_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)

    eta = base_inner_lr * torch.sigmoid(F.linear(x, w_eta_w, w_eta_b))
    alpha = base_weight_decay * torch.sigmoid(F.linear(x, w_alpha_w, w_alpha_b))
    return q, k, v, eta, alpha


class uTTTLayer(nn.Module):
    """
    Layer-l Slow-Weight Projection & Chunk-wise Parallel TTT Controller.

    Slow Weights Theta_slow^(l) = {W_Q^(l), W_K^(l), W_V^(l), W_O^(l), W_eta^(l), W_alpha^(l)}
    generate layer-specific views (Q, K, V) and dynamic step-sizes (eta_t, alpha_t)
    that read from and write into the global `SharedFastWeightPool`.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 64,
        base_inner_lr: float = 0.015,
        base_weight_decay: float = 0.005,
        rms_norm_eps: float = 1e-5,
        chunk_checkpoint_segments: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.base_inner_lr = base_inner_lr
        self.base_weight_decay = base_weight_decay
        self.chunk_checkpoint_segments = chunk_checkpoint_segments

        # Slow Weights Theta_slow^(l)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Dynamic inner-loop learning rate (eta_t) and forget gate (alpha_t) routers
        self.w_eta = nn.Linear(d_model, 1, bias=True)
        self.w_alpha = nn.Linear(d_model, 1, bias=True)

        # Initialize bias so initial sigmoid(w_eta(x)) ~ 0.5 and sigmoid(w_alpha(x)) ~ 0.2
        nn.init.zeros_(self.w_eta.weight)
        nn.init.constant_(self.w_eta.bias, 0.0)
        nn.init.zeros_(self.w_alpha.weight)
        nn.init.constant_(self.w_alpha.bias, -1.386)

        self.q_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.k_norm = RMSNorm(d_model, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        fast_pool: SharedFastWeightPool,
        fast_state: FastWeightState,
        attention_mask: Optional[torch.Tensor] = None,
        update_fast_weights: bool = True,
        read_state: Optional[FastWeightState] = None,
    ) -> Tuple[torch.Tensor, FastWeightState, torch.Tensor]:
        """
        Process sequence `x` in parallel chunks of length `C = self.chunk_size`,
        updating `fast_state` sequentially across chunks and returning the evolved
        `fast_state` to be passed to layer `l + 1`.
        """
        bsz, seq_len, dim = x.shape

        # Fast path when fast weights are frozen (e.g. token generation Step 1):
        # Compute only Q projection + single Fast-Weight readout (2x faster per token)
        if not update_fast_weights:
            q = fused_fast_q_proj(
                x=x,
                wq_w=self.w_q.weight,
                q_norm_w=self.q_norm.weight,
                eps=self.q_norm.eps,
            )
            readout = fast_pool.forward_readout(q, fast_state.ensure_batch_size(bsz))
            surprise_loss = x.new_zeros((), dtype=torch.float32)
            return self.w_o(readout), fast_state, surprise_loss

        # Compiled Slow-Weight Projections + Fused RMSNorm + Gate Sigmoids
        q, k, v, eta, alpha = fused_slow_projections(
            x=x,
            wq_w=self.w_q.weight,
            wk_w=self.w_k.weight,
            wv_w=self.w_v.weight,
            q_norm_w=self.q_norm.weight,
            k_norm_w=self.k_norm.weight,
            w_eta_w=self.w_eta.weight,
            w_eta_b=self.w_eta.bias,
            w_alpha_w=self.w_alpha.weight,
            w_alpha_b=self.w_alpha.bias,
            base_inner_lr=self.base_inner_lr,
            base_weight_decay=self.base_weight_decay,
            eps=self.q_norm.eps,
        )

        # Chunk-wise Parallel TTT Update across the sequence
        chunk_outputs: List[torch.Tensor] = []
        chunk_losses: List[torch.Tensor] = []
        cur_state = fast_state

        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        use_seg_ckpt = (
            self.chunk_checkpoint_segments > 1
            and self.training
            and torch.is_grad_enabled()
            and num_chunks > 1
        )

        if use_seg_ckpt:
            seg_size = max(1, (num_chunks + self.chunk_checkpoint_segments - 1) // self.chunk_checkpoint_segments)
            for seg_start in range(0, num_chunks, seg_size):
                seg_end = min(seg_start + seg_size, num_chunks)
                tok_start = seg_start * self.chunk_size
                tok_end = min(seg_end * self.chunk_size, seq_len)

                q_seg = q[:, tok_start:tok_end, :]
                k_seg = k[:, tok_start:tok_end, :]
                v_seg = v[:, tok_start:tok_end, :]
                eta_seg = eta[:, tok_start:tok_end, :]
                alpha_seg = alpha[:, tok_start:tok_end, :]
                mask_seg = attention_mask[:, tok_start:tok_end] if attention_mask is not None else None
                step_before = cur_state.step

                def _run_segment(
                    q_s: torch.Tensor,
                    k_s: torch.Tensor,
                    v_s: torch.Tensor,
                    eta_s: torch.Tensor,
                    alpha_s: torch.Tensor,
                    w_up_s: torch.Tensor,
                    w_gate_s: torch.Tensor,
                    w_down_s: torch.Tensor,
                    s_up_s: torch.Tensor,
                    s_gate_s: torch.Tensor,
                    s_down_s: torch.Tensor,
                    _mask_s: Optional[torch.Tensor] = mask_seg,
                    _step_b: int = step_before,
                    _ln_w: torch.Tensor = cur_state.ln_weight,
                    _ln_b: torch.Tensor = cur_state.ln_bias,
                ) -> Tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ]:
                    st = FastWeightState(
                        w_up=w_up_s,
                        w_gate=w_gate_s,
                        w_down=w_down_s,
                        s_up=s_up_s,
                        s_gate=s_gate_s,
                        s_down=s_down_s,
                        ln_weight=_ln_w,
                        ln_bias=_ln_b,
                        step=_step_b,
                    )
                    seg_outs: List[torch.Tensor] = []
                    seg_losses: List[torch.Tensor] = []
                    s_len = q_s.shape[1]
                    n_c = (s_len + self.chunk_size - 1) // self.chunk_size
                    for ci in range(n_c):
                        cs = ci * self.chunk_size
                        ce = min(cs + self.chunk_size, s_len)
                        mc = _mask_s[:, cs:ce] if _mask_s is not None else None
                        oc, st, lc = fast_pool.chunk_step(
                            state=st,
                            q_chunk=q_s[:, cs:ce, :],
                            k_chunk=k_s[:, cs:ce, :],
                            v_chunk=v_s[:, cs:ce, :],
                            eta_chunk=eta_s[:, cs:ce, :],
                            alpha_chunk=alpha_s[:, cs:ce, :],
                            token_mask=mc,
                            update_fast_weights=True,
                            read_state=read_state,
                        )
                        seg_outs.append(oc)
                        seg_losses.append(lc)
                    return (
                        torch.cat(seg_outs, dim=1),
                        st.w_up,
                        st.w_gate,
                        st.w_down,
                        st.s_up,
                        st.s_gate,
                        st.s_down,
                        torch.stack(seg_losses).mean(),
                    )

                seg_out, wu, wg, wd, su, sg, sd, seg_loss = checkpoint(
                    _run_segment,
                    q_seg,
                    k_seg,
                    v_seg,
                    eta_seg,
                    alpha_seg,
                    cur_state.w_up,
                    cur_state.w_gate,
                    cur_state.w_down,
                    cur_state.s_up,
                    cur_state.s_gate,
                    cur_state.s_down,
                    use_reentrant=False,
                )
                cur_state = FastWeightState(
                    w_up=wu,
                    w_gate=wg,
                    w_down=wd,
                    s_up=su,
                    s_gate=sg,
                    s_down=sd,
                    ln_weight=cur_state.ln_weight,
                    ln_bias=cur_state.ln_bias,
                    step=step_before + (seg_end - seg_start),
                    last_surprise_loss=cur_state.last_surprise_loss,
                    surprise_history=cur_state.surprise_history,
                )
                chunk_outputs.append(seg_out)
                chunk_losses.append(seg_loss * (seg_end - seg_start))
        else:
            for c_idx in range(num_chunks):
                start = c_idx * self.chunk_size
                end = min(start + self.chunk_size, seq_len)

                q_c = q[:, start:end, :]
                k_c = k[:, start:end, :]
                v_c = v[:, start:end, :]
                eta_c = eta[:, start:end, :]
                alpha_c = alpha[:, start:end, :]
                mask_c = attention_mask[:, start:end] if attention_mask is not None else None

                out_c, cur_state, loss_c = fast_pool.chunk_step(
                    state=cur_state,
                    q_chunk=q_c,
                    k_chunk=k_c,
                    v_chunk=v_c,
                    eta_chunk=eta_c,
                    alpha_chunk=alpha_c,
                    token_mask=mask_c,
                    update_fast_weights=True,
                    read_state=read_state,
                )
                chunk_outputs.append(out_c)
                chunk_losses.append(loss_c)

        out = torch.cat(chunk_outputs, dim=1)                       # (B, T, D)
        mean_surprise_loss = torch.stack(chunk_losses).sum() / num_chunks
        return self.w_o(out), cur_state, mean_surprise_loss


class uTTTBlock(nn.Module):
    """
    Single Transformer + uTTT Block:
        1. Pre-RMSNorm + Local Sliding Window Attention + Residual
        2. Pre-RMSNorm + Global Shared-Pool uTTTLayer + Residual

    Receives `fast_state` W^(l-1) from the previous block and outputs the updated
    `fast_state` W^(l) for the next block.
    """

    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int = 512,
        chunk_size: int = 64,
        base_inner_lr: float = 0.015,
        base_weight_decay: float = 0.005,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attn = SlidingWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            window_size=window_size,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
        )
        self.uttt_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.uttt = uTTTLayer(
            d_model=d_model,
            chunk_size=chunk_size,
            base_inner_lr=base_inner_lr,
            base_weight_decay=base_weight_decay,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        fast_pool: SharedFastWeightPool,
        fast_state: FastWeightState,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        update_fast_weights: bool = True,
        read_state: Optional[FastWeightState] = None,
    ) -> Tuple[torch.Tensor, FastWeightState, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 1. Local Sliding Window Attention
        attn_out, new_kv_cache = self.attn(
            self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos
        )
        h = x + attn_out

        # 2. Cross-layer Shared Fast-Weight Pool Read/Write
        uttt_out, updated_fast_state, surprise_loss = self.uttt(
            self.uttt_norm(h),
            fast_pool=fast_pool,
            fast_state=fast_state,
            attention_mask=attention_mask,
            update_fast_weights=update_fast_weights,
            read_state=read_state,
        )
        out = h + uttt_out
        return out, updated_fast_state, surprise_loss, new_kv_cache

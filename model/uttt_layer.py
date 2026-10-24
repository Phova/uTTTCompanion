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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
    # DeepGEMM-style Gate+Up Fused Batched GEMM:
    # Combine w_up and w_gate along the last dimension so one Tensor Core bmm computes both.
    w_up_gate = torch.cat([w_up, w_gate], dim=-1)               # (B, D, 2 * d_ff)
    ug = torch.bmm(x, w_up_gate)                                # (B, T, 2 * d_ff)
    u, g = ug.chunk(2, dim=-1)                                  # each (B, T, d_ff)
    h = F.silu(g) * u                                           # (B, T, d_ff)
    z = torch.bmm(h, w_down)                                    # (B, T, D)

    # LayerNorm in FP32 for numerical stability on BF16 Tensor Cores
    z_fp32 = z.float()
    mean = z_fp32.mean(dim=-1, keepdim=True)
    var = (z_fp32 - mean).pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    z_norm = (z_fp32 - mean) * rstd
    ln_out = (z_norm * ln_weight.float() + ln_bias.float()).to(orig_dtype)
    return x + ln_out


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
    bsz, chunk_len, dim = k_chunk.shape

    # --- 1. Batched Forward Pass through SwiGLU Fast Weights (Fused Gate+Up GEMM) ---
    w_up_gate = torch.cat([w_up, w_gate], dim=-1)               # (B, D, 2 * d_ff)
    ug = torch.bmm(k_chunk, w_up_gate)                          # (B, C, 2 * d_ff)
    u, g = ug.chunk(2, dim=-1)                                  # each (B, C, d_ff)

    # Compute activation in FP32 to preserve precision during derivative calculation
    g_fp32 = g.float()
    u_fp32 = u.float()
    sigma_g = torch.sigmoid(g_fp32)
    a_fp32 = g_fp32 * sigma_g                                   # SiLU(G)
    h_fp32 = a_fp32 * u_fp32                                    # (B, C, d_ff)
    h = h_fp32.to(orig_dtype)

    z = torch.bmm(h, w_down)                                    # (B, C, D)

    # LayerNorm statistics in FP32
    z_fp32 = z.float()
    mean = z_fp32.mean(dim=-1, keepdim=True)
    diff = z_fp32 - mean
    var = diff.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    z_hat = diff * rstd                                         # (B, C, D)

    gamma = ln_weight.float().view(1, 1, dim)
    beta = ln_bias.float().view(1, 1, dim)
    ln_out = z_hat * gamma + beta
    v_hat = k_chunk.float() + ln_out                            # (B, C, D)

    # --- 2. Surprise Loss: L_t = 1/2 || f_W(k_t) - v_t ||_2^2 ---
    err = v_hat - v_chunk.float()                               # (B, C, D)
    scale = float(dim) if normalize_by_dim else 1.0
    token_loss = 0.5 * err.pow(2).sum(dim=-1) / scale           # (B, C)

    if token_mask is not None:
        mask_fp32 = token_mask.float()                          # (B, C)
        valid_counts = mask_fp32.sum(dim=-1, keepdim=True).clamp_min(1.0)
        surprise_loss = (token_loss * mask_fp32).sum() / mask_fp32.sum().clamp_min(1.0)
        weight_factor = (eta_chunk.float() * mask_fp32.unsqueeze(-1)) / valid_counts.unsqueeze(-1)
    else:
        surprise_loss = token_loss.mean()
        weight_factor = eta_chunk.float() / float(chunk_len)

    # --- 3. Analytical Backward through Residual & LayerNorm ---
    delta_v = weight_factor * (err / scale)                     # (B, C, D)
    dz_hat = delta_v * gamma                                    # (B, C, D)

    mean_dz_hat = dz_hat.mean(dim=-1, keepdim=True)
    mean_dz_hat_z = (dz_hat * z_hat).mean(dim=-1, keepdim=True)
    delta_z_fp32 = rstd * (dz_hat - mean_dz_hat - z_hat * mean_dz_hat_z)
    delta_z = delta_z_fp32.to(orig_dtype)                       # (B, C, D)

    # --- 4. Analytical Backward through W_down ---
    # grad_W_down = H^T @ delta_Z -> (B, d_ff, C) @ (B, C, D) = (B, d_ff, D)
    grad_w_down = torch.bmm(h.transpose(1, 2), delta_z)
    # delta_H = delta_Z @ W_down^T -> (B, C, D) @ (B, D, d_ff) = (B, C, d_ff)
    delta_h = torch.bmm(delta_z, w_down.transpose(1, 2))

    # --- 5. Analytical Backward through SwiGLU Activation ---
    delta_h_fp32 = delta_h.float()
    delta_u_fp32 = delta_h_fp32 * a_fp32
    dsilu_fp32 = sigma_g * (1.0 + g_fp32 * (1.0 - sigma_g))
    delta_g_fp32 = delta_h_fp32 * u_fp32 * dsilu_fp32

    # --- 6. Fused Analytical Backward through W_up and W_gate ---
    delta_ug = torch.cat([delta_u_fp32, delta_g_fp32], dim=-1).to(orig_dtype)  # (B, C, 2 * d_ff)
    grad_w_up_gate = torch.bmm(k_chunk.transpose(1, 2), delta_ug)              # (B, D, 2 * d_ff)
    grad_w_up, grad_w_gate = grad_w_up_gate.chunk(2, dim=-1)

    return grad_w_up, grad_w_gate, grad_w_down, surprise_loss


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
            w_up = w_up.detach().clone().unsqueeze(0).repeat(batch_size, 1, 1)
            w_gate = w_gate.detach().clone().unsqueeze(0).repeat(batch_size, 1, 1)
            w_down = w_down.detach().clone().unsqueeze(0).repeat(batch_size, 1, 1)
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

    def forward_readout(self, x: torch.Tensor, state: FastWeightState) -> torch.Tensor:
        """Read out from current FastWeightState without modifying W_fast."""
        state = state.ensure_batch_size(x.shape[0])
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
    ) -> Tuple[torch.Tensor, FastWeightState, torch.Tensor]:
        """
        Execute one Chunk-wise Parallel TTT step:
        1. Compute analytical SwiGLU gradients on (k_chunk, v_chunk) using W_{t-1}.
        2. Clip inner-loop gradient Frobenius norm to `max_inner_grad_norm`.
        3. Update Momentum S_t and Fast Weights W_t:
               S_t = beta * S_{t-1} - eta_t * grad_W L_TTT(W_{t-1}; k_t, v_t)
               W_t = (1 - alpha_t) * W_{t-1} + S_t
        4. Read out chunk representation using updated W_t:
               o_chunk = f_{W_t}(q_chunk)
        """
        bsz = q_chunk.shape[0]
        state = state.ensure_batch_size(bsz)

        if not update_fast_weights:
            # Frozen fast-weight inference mode (used during token streaming generation)
            with torch.no_grad():
                v_pred = self.forward_readout(k_chunk, state)
                scale = float(self.d_model) if self.normalize_loss_by_dim else 1.0
                surprise_loss = 0.5 * (v_pred.float() - v_chunk.float()).pow(2).sum(dim=-1).mean() / scale
            out = self.forward_readout(q_chunk, state)
            return out, state, surprise_loss

        # 1. Analytical Chunk-wise Gradient (already weighted by eta_chunk)
        # Use First-Order Meta-Approximation (detach W inside the inner derivative operator
        # while keeping W attached in the recurrence W_t = (1-alpha)*W_{t-1} + S_t and readout)
        # to prevent 2nd-order Hessian-vector explosion across 16 layers x 16 chunks (256 steps).
        grad_up, grad_gate, grad_down, surprise_loss = analytical_swiglu_chunk_grad(
            k_chunk=k_chunk,
            v_chunk=v_chunk,
            eta_chunk=eta_chunk,
            w_up=state.w_up.detach(),
            w_gate=state.w_gate.detach(),
            w_down=state.w_down.detach(),
            ln_weight=state.ln_weight.detach(),
            ln_bias=state.ln_bias.detach(),
            token_mask=token_mask,
            normalize_by_dim=self.normalize_loss_by_dim,
            eps=self.eps,
        )

        # 2. Max Inner-Loop Gradient Frobenius Norm Clipping per batch element
        # Compute clipping coefficient under no_grad using fused vector_norm to avoid
        # allocating 3x 88MB FP32 tensors per chunk and storing them in the autograd graph.
        if self.max_inner_grad_norm is not None and self.max_inner_grad_norm > 0.0:
            with torch.no_grad():
                norm_up = torch.linalg.vector_norm(grad_up, dim=(1, 2), dtype=torch.float32)
                norm_gate = torch.linalg.vector_norm(grad_gate, dim=(1, 2), dtype=torch.float32)
                norm_down = torch.linalg.vector_norm(grad_down, dim=(1, 2), dtype=torch.float32)
                grad_norm = torch.sqrt(
                    norm_up.square() + norm_gate.square() + norm_down.square() + 1e-8
                )
                clip_coef = torch.clamp(self.max_inner_grad_norm / grad_norm, max=1.0)
                clip_view = clip_coef.to(grad_up.dtype).view(bsz, 1, 1)
            grad_up = grad_up * clip_view
            grad_gate = grad_gate * clip_view
            grad_down = grad_down * clip_view

        # 3. Momentum & Forget Gate Update:
        #    S_t = beta * S_{t-1} - eta_t * grad_W
        #    W_t = (1 - alpha_t) * W_{t-1} + S_t
        beta = self.momentum_beta
        s_up_new = beta * state.s_up.detach() - grad_up
        s_gate_new = beta * state.s_gate.detach() - grad_gate
        s_down_new = beta * state.s_down.detach() - grad_down

        # Mean chunk forget gate alpha_t in (B, 1, 1)
        alpha_bar = alpha_chunk.mean(dim=1, keepdim=True).to(state.w_up.dtype)
        decay = 1.0 - alpha_bar

        w_up_new = decay * state.w_up + s_up_new
        w_gate_new = decay * state.w_gate + s_gate_new
        w_down_new = decay * state.w_down + s_down_new

        # Avoid synchronous GPU->CPU .item() stall inside the inner loop during training
        if not torch.is_grad_enabled():
            loss_val = float(surprise_loss.detach().item())
            new_history = list(state.surprise_history[-99:]) + [loss_val]
        else:
            loss_val = state.last_surprise_loss
            new_history = state.surprise_history

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
            last_surprise_loss=loss_val,
            surprise_history=new_history,
        )

        # 4. Read out using the newly updated fast weights W_t
        out_chunk = self.forward_readout(q_chunk, new_state)
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

        d_up = (state.w_up.float() - w_up_0).norm(p="fro").item()
        d_gate = (state.w_gate.float() - w_gate_0).norm(p="fro").item()
        d_down = (state.w_down.float() - w_down_0).norm(p="fro").item()

        b_up = w_up_0.norm(p="fro").item()
        b_gate = w_gate_0.norm(p="fro").item()
        b_down = w_down_0.norm(p="fro").item()

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

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = x.shape

        # Fused QKV GEMM launch (preserves individual wq/wk/wv state_dict keys)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        w_qkv = torch.cat([self.wq.weight, self.wk.weight, self.wv.weight], dim=0)
        qkv = F.linear(x, w_qkv)
        q_raw, k_raw, v_raw = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)

        q = q_raw.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, start_pos=start_pos)
        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            # DeepInfer SWA Bounded Replay: strictly cap cached KV state to window_size
            if k.shape[2] > self.window_size:
                k = k[:, :, -self.window_size :, :]
                v = v[:, :, -self.window_size :, :]

        # Bound stored KV cache to window_size even on initial long-context prefill
        if not self.training and k.shape[2] > self.window_size:
            new_kv_cache = (k[:, :, -self.window_size :, :], v[:, :, -self.window_size :, :])
        else:
            new_kv_cache = (k, v)

        kv_len = k.shape[2]
        drop_p = self.dropout if self.training else 0.0

        # Native PyTorch 2.5+ GQA (zero KV memory duplication) with fallback
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
            else:
                attn_mask = self._get_sliding_causal_mask(seq_len, kv_len, x.device)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=drop_p, is_causal=False, enable_gqa=(self.n_rep > 1)
                )
        except TypeError:
            # Fallback for older PyTorch without enable_gqa
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
    ) -> Tuple[torch.Tensor, FastWeightState, torch.Tensor]:
        """
        Process sequence `x` in parallel chunks of length `C = self.chunk_size`,
        updating `fast_state` sequentially across chunks and returning the evolved
        `fast_state` to be passed to layer `l + 1`.
        """
        bsz, seq_len, dim = x.shape

        # Fused Slow-Weight Projections: 1 GEMM for (Q, K, V) and 1 GEMM for (eta, alpha)
        w_qkv = torch.cat([self.w_q.weight, self.w_k.weight, self.w_v.weight], dim=0)
        q_raw, k_raw, v_raw = F.linear(x, w_qkv).chunk(3, dim=-1)
        q = self.q_norm(q_raw)                                      # (B, T, D)
        k = self.k_norm(k_raw)                                      # (B, T, D)
        # Unit-RMS normalize V targets so inner-loop reconstruction loss cannot scale-explode
        v = (v_raw.float() * torch.rsqrt(v_raw.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)).to(x.dtype)

        w_gates = torch.cat([self.w_eta.weight, self.w_alpha.weight], dim=0)
        b_gates = torch.cat([self.w_eta.bias, self.w_alpha.bias], dim=0)
        gates = torch.sigmoid(F.linear(x, w_gates, b_gates))        # (B, T, 2)
        eta_sig, alpha_sig = gates.chunk(2, dim=-1)
        eta = self.base_inner_lr * eta_sig                          # (B, T, 1)
        alpha = self.base_weight_decay * alpha_sig                  # (B, T, 1)

        # Fast path when fast weights are frozen (e.g. token generation Step 1)
        if not update_fast_weights:
            readout, fast_state, surprise_loss = fast_pool.chunk_step(
                state=fast_state,
                q_chunk=q,
                k_chunk=k,
                v_chunk=v,
                eta_chunk=eta,
                alpha_chunk=alpha,
                token_mask=attention_mask,
                update_fast_weights=False,
            )
            return self.w_o(readout), fast_state, surprise_loss

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
                        ln_weight=cur_state.ln_weight,
                        ln_bias=cur_state.ln_bias,
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
                        )
                        seg_outs.append(oc)
                        seg_losses.append(lc)
                    return (
                        torch.cat(seg_outs, dim=1),
                        st.w_up,
                        st.w_gate,
                        st.w_down,
                        st.s_up.detach(),
                        st.s_gate.detach(),
                        st.s_down.detach(),
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
                    cur_state.s_up.detach(),
                    cur_state.s_gate.detach(),
                    cur_state.s_down.detach(),
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
                chunk_losses.append(seg_loss)
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
                )
                chunk_outputs.append(out_c)
                chunk_losses.append(loss_c)

        out = torch.cat(chunk_outputs, dim=1)                       # (B, T, D)
        mean_surprise_loss = torch.stack(chunk_losses).mean()
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
        )
        out = h + uttt_out
        return out, updated_fast_state, surprise_loss, new_kv_cache

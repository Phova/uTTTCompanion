"""
model/uttt_nvs.py
=================
Multimodal Universal Test-Time Training for Novel View Synthesis (uTTT-NVS / LaCT-NVS).

Implements the paper's flagship 3D/Multimodal Fast-Weight Scene Reconstruction &
Novel View Synthesis paradigm on top of `SharedFastWeightPool` and `uTTTBlock`:

1. 6D Plucker Ray Coordinate Parameterization (`compute_plucker_rays`):
   Converts camera intrinsics K in R^{B x V x 3 x 3} and camera-to-world extrinsics
   T_c2w = [R | t] in R^{B x V x 4 x 4} into dense per-pixel Plucker ray coordinates:
       r = (d, o x d) in R^{B x V x 6 x H x W},   ||d||_2 = 1.
2. Dual-Stream Context/Target Patch Tokenization:
   - Context (Reference) Views: Concatenate RGB image I_ctx in R^{3 x H x W} and
     Plucker rays r_ctx in R^{6 x H x W} -> 9-channel input patchified into tokens.
   - Target (Novel Query) Views: Plucker rays r_tgt in R^{6 x H x W} ONLY (no RGB)
     patchified into query tokens.
3. Two-Phase Test-Time Scene Compression & Rendering (`uTTTNVSModel`):
   - Phase 1 (Scene TTT Compression, `update_fast_weights=True`):
     Streams all Context View tokens through `uTTTBlocks`, compressing 3D scene
     geometry & radiance into the shared SwiGLU Fast Weights W_scene = {W_up, W_gate, W_down}.
   - Phase 2 (Novel View Readout, `update_fast_weights=False`):
     Queries frozen W_scene using Target camera Plucker ray tokens Q_tgt to render
     novel-view RGB patches -> decoded to full-resolution RGB images via PixelShuffle/Linear head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.uttt_layer import FastWeightState, RMSNorm, SharedFastWeightPool, uTTTBlock
from model.uttt_model import uTTTConfig


@dataclass
class uTTTNVSOutput:
    """Structured output returned by `uTTTNVSModel.forward`."""
    rendered_rgb: torch.Tensor                  # (B, V_tgt, 3, H, W) in [0, 1]
    scene_fast_state: FastWeightState           # Scene-compressed SwiGLU fast weights W_scene
    loss: Optional[torch.Tensor] = None
    mse_loss: Optional[torch.Tensor] = None
    edge_loss: Optional[torch.Tensor] = None
    surprise_loss: Optional[torch.Tensor] = None
    psnr: Optional[float] = None


def compute_plucker_rays(
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Compute dense 6D Plucker ray coordinates `r = (d, o x d)` for each pixel.

    Args:
        intrinsics: (B, V, 3, 3) camera intrinsic matrices [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        c2w:        (B, V, 4, 4) camera-to-world transformation matrices [R | t]
        height:     Image height H
        width:      Image width W

    Returns:
        plucker:    (B, V, 6, H, W) tensor where channels 0:3 are unit direction `d`
                    and channels 3:6 are moment `o x d`.
    """
    bsz, n_views, _, _ = intrinsics.shape
    device = intrinsics.device
    dtype = intrinsics.dtype

    # Construct pixel grid (u, v, 1)
    y_range = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    x_range = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    ones = torch.ones_like(grid_x)
    # (3, H*W)
    pixel_coords = torch.stack([grid_x, grid_y, ones], dim=0).reshape(1, 1, 3, height * width)

    fx = intrinsics[:, :, 0:1, 0:1].float()
    fy = intrinsics[:, :, 1:2, 1:2].float()
    cx = intrinsics[:, :, 0:1, 2:3].float()
    cy = intrinsics[:, :, 1:2, 2:3].float()

    u = pixel_coords[:, :, 0:1, :]
    v = pixel_coords[:, :, 1:2, :]
    cam_dirs = torch.cat(
        [(u - cx) / fx.clamp_min(1e-6), (v - cy) / fy.clamp_min(1e-6), torch.ones_like(u).expand(bsz, n_views, -1, -1)],
        dim=2,
    )  # (B, V, 3, H*W)

    rot = c2w[:, :, :3, :3].float()                      # (B, V, 3, 3)
    trans = c2w[:, :, :3, 3:4].float()                   # (B, V, 3, 1)

    world_dirs = torch.matmul(rot, cam_dirs)             # (B, V, 3, H*W)
    dirs_norm = F.normalize(world_dirs, p=2, dim=2, eps=1e-6)
    origins = trans.expand_as(dirs_norm)                 # (B, V, 3, H*W)

    moment = torch.cross(origins, dirs_norm, dim=2)      # o x d: (B, V, 3, H*W)
    plucker = torch.cat([dirs_norm, moment], dim=2).reshape(bsz, n_views, 6, height, width)
    return plucker.to(dtype)


class uTTTNVSModel(nn.Module):
    """
    Universal Test-Time Training Model for Novel View Synthesis (NVS) and Multimodal Scene Compression.

    - Context Views `(I_ctx, K_ctx, c2w_ctx)` are tokenized into patches of size `P x P`
      and processed with `update_fast_weights=True` (chunk size = tokens per view or multi-view chunk),
      mapping 3D scene structure into the shared SwiGLU Fast Weight Pool `W_scene`.
    - Target Camera Rays `(K_tgt, c2w_tgt)` are tokenized into query patches and processed
      with `update_fast_weights=False`, reading out from `W_scene` to synthesize novel views.
    """

    def __init__(
        self,
        config: uTTTConfig,
        img_size: int = 256,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.config = config
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_view = (img_size // patch_size) ** 2

        # Context view patch embedding: (RGB 3ch + Plucker 6ch) = 9ch -> d_model
        self.ctx_patch_embed = nn.Conv2d(
            in_channels=9,
            out_channels=config.d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        # Target query ray patch embedding: (Plucker 6ch only) -> d_model
        self.tgt_ray_embed = nn.Conv2d(
            in_channels=6,
            out_channels=config.d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Shared Fast-Weight Pool (W_0 -> W_scene)
        self.fast_weight_pool = SharedFastWeightPool(
            d_model=config.d_model,
            d_ff=config.d_ff_fast,
            momentum_beta=config.momentum_beta,
            max_inner_grad_norm=config.max_inner_grad_norm,
            normalize_loss_by_dim=config.normalize_loss_by_dim,
            rms_norm_eps=config.rms_norm_eps,
        )

        # Stack of uTTTBlocks
        self.layers = nn.ModuleList(
            [
                uTTTBlock(
                    layer_idx=idx,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    window_size=config.sliding_window_size,
                    chunk_size=config.chunk_size,
                    base_inner_lr=config.base_inner_lr,
                    base_weight_decay=config.base_weight_decay,
                    max_seq_len=max(config.max_seq_len, self.patches_per_view * 16),
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    dropout=config.dropout,
                )
                for idx in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # Decode each target token back to (3 * P * P) RGB pixel patch
        self.rgb_patch_head = nn.Linear(config.d_model, 3 * patch_size * patch_size, bias=True)

    def encode_context_views(
        self,
        ctx_rgb: torch.Tensor,
        ctx_intrinsics: torch.Tensor,
        ctx_c2w: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize context RGB images + Plucker rays into (B, V_ctx * N_patches, D)."""
        bsz, v_ctx, _, h, w = ctx_rgb.shape
        plucker = compute_plucker_rays(ctx_intrinsics, ctx_c2w, h, w)
        ctx_in = torch.cat([ctx_rgb * 2.0 - 1.0, plucker], dim=2).reshape(bsz * v_ctx, 9, h, w)
        tokens = self.ctx_patch_embed(ctx_in).flatten(2).transpose(1, 2)  # (B*V_ctx, N_p, D)
        return tokens.reshape(bsz, v_ctx * tokens.shape[1], self.config.d_model)

    def encode_target_rays(
        self,
        tgt_intrinsics: torch.Tensor,
        tgt_c2w: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """Tokenize target camera Plucker rays into query tokens (B, V_tgt * N_patches, D)."""
        bsz, v_tgt, _, _ = tgt_intrinsics.shape
        plucker = compute_plucker_rays(tgt_intrinsics, tgt_c2w, height, width).reshape(
            bsz * v_tgt, 6, height, width
        )
        feat = self.tgt_ray_embed(plucker)
        gh, gw = feat.shape[2], feat.shape[3]
        tokens = feat.flatten(2).transpose(1, 2).reshape(bsz, v_tgt * gh * gw, self.config.d_model)
        return tokens, gh, gw

    def decode_rgb_patches(
        self, tokens: torch.Tensor, num_target_views: int, gh: int, gw: int
    ) -> torch.Tensor:
        """Unpatchify decoded target tokens into rendered RGB images (B, V_tgt, 3, H, W)."""
        bsz = tokens.shape[0]
        p = self.patch_size
        patch_rgb = torch.sigmoid(self.rgb_patch_head(self.norm(tokens)))  # (B, V_tgt*gh*gw, 3*p*p)
        patch_rgb = patch_rgb.view(bsz, num_target_views, gh, gw, 3, p, p)
        img = patch_rgb.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return img.view(bsz, num_target_views, 3, gh * p, gw * p)

    def forward(
        self,
        ctx_rgb: torch.Tensor,
        ctx_intrinsics: torch.Tensor,
        ctx_c2w: torch.Tensor,
        tgt_intrinsics: torch.Tensor,
        tgt_c2w: torch.Tensor,
        tgt_rgb: Optional[torch.Tensor] = None,
        surprise_aux_weight: float = 0.05,
    ) -> uTTTNVSOutput:
        """
        1. Compress Context Views into SwiGLU Fast Weights `W_scene` via inner-loop TTT.
        2. Query `W_scene` with Target Plucker Ray tokens (`update_fast_weights=False`)
           to synthesize novel views.
        """
        bsz, _, _, h, w = ctx_rgb.shape
        v_tgt = tgt_intrinsics.shape[1]

        ctx_tokens = self.encode_context_views(ctx_rgb, ctx_intrinsics, ctx_c2w)
        tgt_tokens, gh, gw = self.encode_target_rays(tgt_intrinsics, tgt_c2w, h, w)

        if torch.is_autocast_enabled():
            ac_dtype = torch.get_autocast_dtype("cuda" if ctx_tokens.is_cuda else "cpu")
            ctx_tokens = ctx_tokens.to(ac_dtype)
            tgt_tokens = tgt_tokens.to(ac_dtype)

        scene_state = self.fast_weight_pool.init_state(
            batch_size=bsz,
            device=ctx_tokens.device,
            dtype=ctx_tokens.dtype,
            detach=not self.training,
        )

        # Phase 1: Scene Compression into Fast Weights W_scene (update_fast_weights=True)
        h_ctx = ctx_tokens
        surprise_losses: List[torch.Tensor] = []
        for block in self.layers:
            h_ctx, scene_state, s_loss, _ = block(
                h_ctx,
                fast_pool=self.fast_weight_pool,
                fast_state=scene_state,
                update_fast_weights=True,
            )
            surprise_losses.append(s_loss)

        mean_surprise = torch.stack(surprise_losses).mean()

        # Phase 2: Novel View Synthesis Readout from W_scene (update_fast_weights=False)
        h_tgt = tgt_tokens
        for block in self.layers:
            h_tgt, scene_state, _, _ = block(
                h_tgt,
                fast_pool=self.fast_weight_pool,
                fast_state=scene_state,
                update_fast_weights=False,
            )

        rendered_rgb = self.decode_rgb_patches(h_tgt, v_tgt, gh, gw)

        total_loss: Optional[torch.Tensor] = None
        mse_loss: Optional[torch.Tensor] = None
        edge_loss: Optional[torch.Tensor] = None
        psnr_val: Optional[float] = None

        if tgt_rgb is not None:
            pred_f32 = rendered_rgb.float()
            gt_f32 = tgt_rgb.float()
            mse_loss = F.mse_loss(pred_f32, gt_f32)
            # Spatial gradient / edge consistency loss for sharp novel view synthesis
            dx_pred = pred_f32[:, :, :, :, 1:] - pred_f32[:, :, :, :, :-1]
            dx_gt = gt_f32[:, :, :, :, 1:] - gt_f32[:, :, :, :, :-1]
            dy_pred = pred_f32[:, :, :, 1:, :] - pred_f32[:, :, :, :-1, :]
            dy_gt = gt_f32[:, :, :, 1:, :] - gt_f32[:, :, :, :-1, :]
            edge_loss = F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)
            total_loss = mse_loss + 0.1 * edge_loss + surprise_aux_weight * mean_surprise
            with torch.no_grad():
                psnr_val = float((-10.0 * torch.log10(mse_loss.clamp_min(1e-8))).item())

        return uTTTNVSOutput(
            rendered_rgb=rendered_rgb,
            scene_fast_state=scene_state.detach() if self.training else scene_state,
            loss=total_loss,
            mse_loss=mse_loss,
            edge_loss=edge_loss,
            surprise_loss=mean_surprise,
            psnr=psnr_val,
        )

"""
model/uttt_model.py
===================
Complete Universal Test-Time Training (uTTT) Language Model Architecture.

Key Architectural Features:
1. Single top-level `SharedFastWeightPool` instance (`self.fast_weight_pool`) holding
   the meta-initialized SwiGLU fast weights W_0 = {W_up, W_gate, W_down}.
2. Sequential Cross-Layer Fast-Weight State Passing:
   Layer l receives W^(l-1) from Layer l-1, performs analytical chunk-wise TTT updates,
   and passes the updated W^(l) directly to Layer l+1 (and across time chunks).
3. Explicit separation between Slow Weights (Theta_slow) and Meta-Initialized Fast
   Weights (W_0) for dual-track checkpointing (`base_uttt.pt` vs `personas/{user_id}_state.pt`).
4. Native support for Gradient Checkpointing over tensorized FastWeightState to keep
   RTX 5090 meta-training VRAM strictly under 28 GB.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.uttt_layer import (
    FastWeightState,
    RMSNorm,
    SharedFastWeightPool,
    uTTTBlock,
)


# ==============================================================================
# 1. Model Configuration
# ==============================================================================

@dataclass
class uTTTConfig:
    """Configuration for the uTTT Model and Inner-Loop Fast-Weight Dynamics."""
    name: str = "uTTT-500M-Base"
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff_fast: int = 2816
    sliding_window_size: int = 512
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0

    # uTTT inner-loop hyperparameters
    chunk_size: int = 64
    base_inner_lr: float = 0.015
    base_weight_decay: float = 0.005
    momentum_beta: float = 0.90
    max_inner_grad_norm: float = 1.0
    normalize_loss_by_dim: bool = True
    surprise_threshold: float = 0.25
    elastic_anchor_lambda: float = 0.02
    max_weight_drift_ratio: float = 0.35

    # Training memory & schedule optimization
    gradient_checkpointing: bool = True
    tie_word_embeddings: bool = True
    write_schedule: str = "sequential"          # "sequential" (Eq. 14/17) or "aggregated" (Eq. 2)
    dspark_speculative_k: int = 0               # DeepInfer DSpark speculative draft length (0=off)
    dspark_conf_threshold: float = 0.82         # Confidence gate for DSpark draft token proposal

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "uTTTConfig":
        # Support nested YAML structure (model + uttt + training sections) or flat dict
        merged: Dict[str, Any] = {}
        if "model" in data or "uttt" in data:
            merged.update(data.get("model", {}))
            merged.update(data.get("uttt", {}))
            if "training" in data and "gradient_checkpointing" in data["training"]:
                merged["gradient_checkpointing"] = data["training"]["gradient_checkpointing"]
        else:
            merged.update(data)

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "uTTTConfig":
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)


@dataclass
class uTTTModelOutput:
    """Structured output returned by `uTTTModel.forward`."""
    logits: torch.Tensor
    fast_state: FastWeightState
    loss: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    surprise_loss: Optional[torch.Tensor] = None
    layer_surprise_losses: List[float] = field(default_factory=list)
    kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


# ==============================================================================
# 2. Built-in Self-Contained Tokenizer (Zero External Hub Dependency)
# ==============================================================================

class SimpleUTF8Tokenizer:
    """
    Deterministic hybrid UTF-8 byte + common-token vocabulary tokenizer.
    Works offline out-of-the-box and guarantees reversible text encoding/decoding
    within `vocab_size` without requiring external HuggingFace credentials.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    USER_ID = 4
    ASSISTANT_ID = 5
    BYTE_OFFSET = 16

    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = max(vocab_size, 512)
        self.special_tokens = {
            "<|pad|>": self.PAD_ID,
            "<|bos|>": self.BOS_ID,
            "<|eos|>": self.EOS_ID,
            "<|unk|>": self.UNK_ID,
            "<|user|>": self.USER_ID,
            "<|assistant|>": self.ASSISTANT_ID,
        }
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.BOS_ID)

        raw_bytes = text.encode("utf-8", errors="replace")
        for b in raw_bytes:
            ids.append(self.BYTE_OFFSET + int(b))

        if add_eos:
            ids.append(self.EOS_ID)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, token_ids: List[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        byte_buffer = bytearray()
        chunks: List[str] = []

        def _flush_bytes() -> None:
            if byte_buffer:
                chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for tid in token_ids:
            tid = int(tid)
            if tid in self.id_to_special:
                if not skip_special_tokens:
                    _flush_bytes()
                    chunks.append(self.id_to_special[tid])
                continue
            if self.BYTE_OFFSET <= tid < self.BYTE_OFFSET + 256:
                byte_buffer.append(tid - self.BYTE_OFFSET)
            else:
                # Hash higher synthetic vocabulary tokens deterministically to printable ASCII
                byte_buffer.append(32 + (tid % 95))

        _flush_bytes()
        return "".join(chunks)


# ==============================================================================
# 3. uTTT Backbone Model (Slow Weights + Shared Fast-Weight Pool)
# ==============================================================================

class uTTTModel(nn.Module):
    """
    Universal Test-Time Training Language Model.

    Architecture:
        - `tok_embeddings`, `layers[0..L-1]` (SWA + uTTT slow projections), `norm`, `lm_head`
          constitute the Slow Weights (Theta_slow), which are optimized during Meta-Training
          and frozen during Test-Time Personalization.
        - `fast_weight_pool` (`SharedFastWeightPool`) is instantiated once at the top level
          and holds the meta-learned initial Fast Weights W_0 = {W_up, W_gate, W_down}.
        - During a forward pass, `fast_state` starts at W_0 (or a user's persisted W_user)
          and flows sequentially through `layers[0] -> layers[1] -> ... -> layers[L-1]`.
    """

    def __init__(self, config: uTTTConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Slow Weights: Token Embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Top-Level Cross-Layer Shared Fast-Weight Pool (instantiated ONCE)
        self.fast_weight_pool = SharedFastWeightPool(
            d_model=config.d_model,
            d_ff=config.d_ff_fast,
            momentum_beta=config.momentum_beta,
            max_inner_grad_norm=config.max_inner_grad_norm,
            normalize_loss_by_dim=config.normalize_loss_by_dim,
            rms_norm_eps=config.rms_norm_eps,
        )

        # Stack of uTTTBlocks (each holding layer-specific Slow Weights)
        self.layers = nn.ModuleList(
            [
                uTTTBlock(
                    layer_idx=layer_idx,
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    window_size=config.sliding_window_size,
                    chunk_size=config.chunk_size,
                    base_inner_lr=config.base_inner_lr,
                    base_weight_decay=config.base_weight_decay,
                    max_seq_len=config.max_seq_len,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    dropout=config.dropout,
                )
                for layer_idx in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_slow_weights()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

    def _init_slow_weights(self) -> None:
        """Initialize slow weights with scaled normal distribution."""
        std = 0.02
        for name, module in self.named_modules():
            if "fast_weight_pool" in name:
                continue
            if isinstance(module, nn.Linear):
                if module.weight is not None and not name.endswith(("w_eta", "w_alpha")):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # --------------------------------------------------------------------------
    # Parameter Partitioning & Dual-Track Checkpoint Helpers
    # --------------------------------------------------------------------------

    def get_slow_parameters(self) -> List[nn.Parameter]:
        """Return all Slow Weight parameters Theta_slow (excluding SharedFastWeightPool)."""
        fast_param_ids = {id(p) for p in self.fast_weight_pool.parameters()}
        return [p for p in self.parameters() if id(p) not in fast_param_ids]

    def get_meta_fast_parameters(self) -> List[nn.Parameter]:
        """Return meta-initialized Fast Weight parameters W_0 from SharedFastWeightPool."""
        return list(self.fast_weight_pool.parameters())

    def slow_state_dict(self) -> Dict[str, torch.Tensor]:
        """Export only Slow Weights Theta_slow for `base_uttt.pt`."""
        full = self.state_dict()
        return {
            k: v.detach().cpu()
            for k, v in full.items()
            if not k.startswith("fast_weight_pool.")
        }

    def init_fast_state_dict(self) -> Dict[str, torch.Tensor]:
        """Export meta-initialized Fast Weights W_0 for `base_uttt.pt`."""
        return {
            k: v.detach().cpu()
            for k, v in self.fast_weight_pool.state_dict().items()
        }

    def load_dual_base_state(
        self,
        slow_weights: Dict[str, torch.Tensor],
        init_fast_weights: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Load Slow Weights and Meta-Init Fast Weights from `base_uttt.pt`."""
        merged = dict(slow_weights)
        for k, v in init_fast_weights.items():
            key = k if k.startswith("fast_weight_pool.") else f"fast_weight_pool.{k}"
            merged[key] = v
        self.load_state_dict(merged, strict=strict)

    def freeze_slow_weights(self) -> None:
        """Freeze all Slow Weights (and W_0 anchor) during Test-Time Personalization."""
        for p in self.parameters():
            p.requires_grad_(False)

    def init_fast_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        detach: bool = False,
    ) -> FastWeightState:
        """Initialize a fresh FastWeightState from the model's SharedFastWeightPool."""
        return self.fast_weight_pool.init_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            detach=detach,
        )

    def num_parameters(self) -> Dict[str, int]:
        """Report parameter counts broken down into Slow Weights vs Shared Fast Pool."""
        slow_cnt = sum(p.numel() for p in self.get_slow_parameters())
        fast_cnt = sum(p.numel() for p in self.get_meta_fast_parameters())
        return {
            "slow_weights": slow_cnt,
            "fast_weight_pool": fast_cnt,
            "total": slow_cnt + fast_cnt,
        }

    # --------------------------------------------------------------------------
    # Forward Pass with Cross-Layer Fast-Weight State Passing
    # --------------------------------------------------------------------------

    def _checkpointed_block_forward(
        self,
        block: uTTTBlock,
        x: torch.Tensor,
        fast_state: FastWeightState,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, FastWeightState, torch.Tensor]:
        """
        Wrap `uTTTBlock.forward` with `torch.utils.checkpoint.checkpoint` using pure
        tensor arguments so autograd can recompute inner-loop fast-weight states cleanly.
        """
        step_before = fast_state.step
        history_before = list(fast_state.surprise_history)

        def _pure_tensor_fn(
            x_in: torch.Tensor,
            w_up: torch.Tensor,
            w_gate: torch.Tensor,
            w_down: torch.Tensor,
            s_up: torch.Tensor,
            s_gate: torch.Tensor,
            s_down: torch.Tensor,
            ln_w: torch.Tensor,
            ln_b: torch.Tensor,
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
            st_in = FastWeightState(
                w_up=w_up,
                w_gate=w_gate,
                w_down=w_down,
                s_up=s_up,
                s_gate=s_gate,
                s_down=s_down,
                ln_weight=ln_w,
                ln_bias=ln_b,
                step=step_before,
            )
            out_h, st_out, s_loss, _ = block(
                x_in,
                fast_pool=self.fast_weight_pool,
                fast_state=st_in,
                attention_mask=attention_mask,
                kv_cache=None,
                start_pos=0,
                update_fast_weights=True,
            )
            return (
                out_h,
                st_out.w_up,
                st_out.w_gate,
                st_out.w_down,
                st_out.s_up.detach(),
                st_out.s_gate.detach(),
                st_out.s_down.detach(),
                s_loss,
            )

        out_h, w_up_o, w_gate_o, w_down_o, s_up_o, s_gate_o, s_down_o, s_loss = checkpoint(
            _pure_tensor_fn,
            x,
            fast_state.w_up,
            fast_state.w_gate,
            fast_state.w_down,
            fast_state.s_up.detach(),
            fast_state.s_gate.detach(),
            fast_state.s_down.detach(),
            fast_state.ln_weight,
            fast_state.ln_bias,
            use_reentrant=False,
        )

        num_chunks = (x.shape[1] + self.config.chunk_size - 1) // self.config.chunk_size
        loss_val = 0.0 if self.training else float(s_loss.detach().item())
        updated_state = FastWeightState(
            w_up=w_up_o,
            w_gate=w_gate_o,
            w_down=w_down_o,
            s_up=s_up_o,
            s_gate=s_gate_o,
            s_down=s_down_o,
            ln_weight=fast_state.ln_weight,
            ln_bias=fast_state.ln_bias,
            step=step_before + num_chunks,
            last_surprise_loss=loss_val,
            surprise_history=history_before if self.training else (history_before + [loss_val])[-100:],
        )
        return out_h, updated_state, s_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        fast_state: Optional[FastWeightState] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        update_fast_weights: bool = True,
        surprise_aux_weight: float = 0.10,
    ) -> uTTTModelOutput:
        """
        Forward pass through all uTTTBlocks with sequential cross-layer FastWeightState passing.

        Args:
            input_ids:           (B, T) token indices
            fast_state:          Initial FastWeightState W^(0). If None, initialized from W_0.
            labels:              Optional (B, T) next-token targets for outer-loop training.
            attention_mask:      Optional (B, T) mask (1 for valid tokens, 0 for pad).
            kv_caches:           Optional per-layer SWA KV caches for incremental decoding.
            start_pos:           Starting token position for RoPE during decoding.
            update_fast_weights: Whether to update W_fast in each block's inner loop.
            surprise_aux_weight: Weight of inner-loop surprise loss in total meta-training loss.
        """
        bsz, seq_len = input_ids.shape
        h = self.tok_embeddings(input_ids)

        if fast_state is None:
            fast_state = self.init_fast_state(
                batch_size=bsz,
                device=h.device,
                dtype=h.dtype,
                detach=not self.training,
            )
        else:
            fast_state = fast_state.ensure_batch_size(bsz)

        new_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        layer_surprise_tensors: List[torch.Tensor] = []
        layer_surprise_floats: List[float] = []

        cur_fast_state = fast_state

        # Sequentially thread `cur_fast_state` through Layer 0 -> Layer 1 -> ... -> Layer L-1
        for idx, block in enumerate(self.layers):
            layer_kv = kv_caches[idx] if kv_caches is not None else None
            use_ckpt = (
                self.gradient_checkpointing
                and self.training
                and update_fast_weights
                and kv_caches is None
            )

            if use_ckpt:
                h, cur_fast_state, s_loss = self._checkpointed_block_forward(
                    block, h, cur_fast_state, attention_mask
                )
            else:
                h, cur_fast_state, s_loss, updated_kv = block(
                    h,
                    fast_pool=self.fast_weight_pool,
                    fast_state=cur_fast_state,
                    attention_mask=attention_mask,
                    kv_cache=layer_kv,
                    start_pos=start_pos,
                    update_fast_weights=update_fast_weights,
                )
                new_kv_caches.append(updated_kv)

            layer_surprise_tensors.append(s_loss)
            if not self.training:
                layer_surprise_floats.append(float(s_loss.detach().item()))

        h = self.norm(h)
        logits = self.lm_head(h)

        mean_surprise_loss = torch.stack(layer_surprise_tensors).mean()
        if not self.training:
            cur_fast_state.last_surprise_loss = float(mean_surprise_loss.detach().item())

        ce_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = ce_loss + surprise_aux_weight * mean_surprise_loss

        return uTTTModelOutput(
            logits=logits,
            fast_state=cur_fast_state,
            loss=total_loss,
            ce_loss=ce_loss,
            surprise_loss=mean_surprise_loss,
            layer_surprise_losses=layer_surprise_floats,
            kv_caches=new_kv_caches if not (self.gradient_checkpointing and self.training) else None,
        )

    # --------------------------------------------------------------------------
    # Interactive Generation (Step 1) & Test-Time Update (Step 2)
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: torch.Tensor,
        fast_state: FastWeightState,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = SimpleUTF8Tokenizer.EOS_ID,
        speculative_draft_k: Optional[int] = None,
        dspark_conf_threshold: Optional[float] = None,
    ) -> Iterator[int]:
        """
        Step 1 of `interact_and_evolve`:
        Freeze `fast_state` (update_fast_weights=False) and stream-generate reply tokens
        autoregressively using Sliding-Window KV caching and optional DeepInfer DSpark
        Confidence-Scheduled Self-Speculative Decoding.
        """
        was_training = self.training
        self.eval()

        draft_k = (
            speculative_draft_k
            if speculative_draft_k is not None
            else int(getattr(self.config, "dspark_speculative_k", 0))
        )
        conf_thresh = (
            dspark_conf_threshold
            if dspark_conf_threshold is not None
            else float(getattr(self.config, "dspark_conf_threshold", 0.82))
        )

        try:
            # Prefill prompt with frozen fast weights
            prefill_out = self.forward(
                input_ids=input_ids,
                fast_state=fast_state,
                kv_caches=None,
                start_pos=0,
                update_fast_weights=False,
            )
            kv_caches = prefill_out.kv_caches
            next_token_logits = prefill_out.logits[:, -1, :]
            cur_pos = input_ids.shape[1]
            generated_count = 0

            while generated_count < max_new_tokens:
                next_token = self._sample_next_token(
                    next_token_logits, temperature=temperature, top_p=top_p
                )
                token_id = int(next_token.item())
                yield token_id
                generated_count += 1

                if eos_token_id is not None and token_id == eos_token_id:
                    break
                if generated_count >= max_new_tokens:
                    break

                # DeepInfer DSpark Confidence-Scheduled Self-Speculative Decoding:
                # When draft_k > 0 and top-1 confidence >= conf_thresh, use the universal
                # SharedFastWeightPool readout head to propose up to `draft_k` tokens and
                # verify them in a single batched forward pass.
                if draft_k > 0:
                    probs = F.softmax(next_token_logits.float(), dim=-1)
                    top_conf = float(probs.max().item())
                else:
                    top_conf = 0.0

                if draft_k > 0 and top_conf >= conf_thresh:
                    draft_tokens: List[int] = [token_id]
                    cur_tok_tensor = next_token.view(1, 1)
                    for _ in range(min(draft_k, max_new_tokens - generated_count)):
                        emb = self.tok_embeddings(cur_tok_tensor)
                        fast_feat = self.fast_weight_pool.forward_readout(emb, fast_state)
                        draft_logits = self.lm_head(self.norm(fast_feat))[:, -1, :]
                        draft_probs = F.softmax(draft_logits.float(), dim=-1)
                        if float(draft_probs.max().item()) < conf_thresh:
                            break
                        cand_tok = torch.argmax(draft_logits, dim=-1)
                        cand_id = int(cand_tok.item())
                        draft_tokens.append(cand_id)
                        if eos_token_id is not None and cand_id == eos_token_id:
                            break
                        cur_tok_tensor = cand_tok.view(1, 1)

                    if len(draft_tokens) > 1:
                        verify_input = torch.tensor(
                            [draft_tokens], dtype=torch.long, device=input_ids.device
                        )
                        verify_out = self.forward(
                            input_ids=verify_input,
                            fast_state=fast_state,
                            kv_caches=kv_caches,
                            start_pos=cur_pos,
                            update_fast_weights=False,
                        )
                        # Verify proposed draft tokens against full model predictions
                        accepted = 0
                        hit_eos = False
                        for idx in range(len(draft_tokens) - 1):
                            target_tok = self._sample_next_token(
                                verify_out.logits[:, idx, :],
                                temperature=temperature,
                                top_p=top_p,
                            )
                            target_id = int(target_tok.item())
                            if target_id == draft_tokens[idx + 1]:
                                yield target_id
                                generated_count += 1
                                accepted += 1
                                if eos_token_id is not None and target_id == eos_token_id:
                                    hit_eos = True
                                    break
                                if generated_count >= max_new_tokens:
                                    break
                            else:
                                # Accept the corrected token from the verifier and stop
                                yield target_id
                                generated_count += 1
                                accepted += 1
                                next_token_logits = verify_out.logits[:, idx, :]
                                if eos_token_id is not None and target_id == eos_token_id:
                                    hit_eos = True
                                break

                        if hit_eos or generated_count >= max_new_tokens:
                            break

                        # Trim KV caches to match the exact number of accepted tokens (1 + accepted)
                        valid_steps = 1 + accepted
                        trim_drop = len(draft_tokens) - valid_steps
                        if trim_drop > 0 and verify_out.kv_caches is not None:
                            kv_caches = [
                                (k_c[:, :, :-trim_drop, :], v_c[:, :, :-trim_drop, :])
                                for (k_c, v_c) in verify_out.kv_caches
                            ]
                        else:
                            kv_caches = verify_out.kv_caches
                        cur_pos += valid_steps
                        next_token_logits = verify_out.logits[:, valid_steps - 1, :]
                        continue

                # Standard 1-token autoregressive decoding step
                step_input = next_token.view(1, 1)
                step_out = self.forward(
                    input_ids=step_input,
                    fast_state=fast_state,
                    kv_caches=kv_caches,
                    start_pos=cur_pos,
                    update_fast_weights=False,
                )
                kv_caches = step_out.kv_caches
                next_token_logits = step_out.logits[:, -1, :]
                cur_pos += 1
        finally:
            if was_training:
                self.train()

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling with temperature scaling."""
        if temperature <= 1e-5:
            return torch.argmax(logits, dim=-1)

        scaled_logits = logits.float() / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled_in_sorted).squeeze(-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def test_time_update(
        self,
        interaction_ids: torch.Tensor,
        fast_state: FastWeightState,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[FastWeightState, float, List[float]]:
        """
        Step 2 of `interact_and_evolve`:
        Execute analytical chunk-wise TTT inner-loop update across all uTTTBlocks on the
        concatenated `[User Prompt + Assistant Reply]` token sequence while keeping all
        Slow Weights strictly frozen.

        Returns:
            candidate_state: Updated FastWeightState W_updated
            mean_surprise:   Mean Surprise Loss across all layers & chunks
            layer_surprises: Per-layer Surprise Losses
        """
        was_training = self.training
        self.eval()
        try:
            cloned_input_state = fast_state.clone(detach=True)
            out = self.forward(
                input_ids=interaction_ids,
                fast_state=cloned_input_state,
                attention_mask=attention_mask,
                update_fast_weights=True,
            )
            candidate_state = out.fast_state.detach()
            mean_surprise = float(out.surprise_loss.detach().item())
            return candidate_state, mean_surprise, out.layer_surprise_losses
        finally:
            if was_training:
                self.train()

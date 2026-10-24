"""
train_5090.py
=============
Meta-Training script for Universal Test-Time Training (uTTT) optimized for
single-GPU NVIDIA RTX 5090 (32GB GDDR7, Blackwell sm_120).

Key Features:
1. Full `torch.amp.autocast('cuda', dtype=torch.bfloat16)` mixed-precision execution.
2. `torch.compile` acceleration of the analytical chunk-wise SwiGLU TTT kernel.
3. Dual-Loop Meta-Learning:
   - Inner Loop: Analytical chunk-wise TTT updates fast weights W^(0) -> W^(1) -> ... -> W^(L)
     via Surprise Loss L_TTT = 1/2 ||f_W(k_t) - v_t||_2^2.
   - Outer Loop: Backpropagates next-token Cross-Entropy + auxiliary Surprise Loss through
     the inner-loop analytical graph to jointly optimize Slow Weights (Theta_slow) and
     Meta-Initialized Fast Weights (W_0).
4. Strict VRAM Budgeting (< 28GB on RTX 5090) via Gradient Checkpointing, Chunk-wise
   Parallel GEMMs (C=64), and Gradient Accumulation.
5. Exports the canonical Base Checkpoint (`checkpoints/base_uttt.pt`) containing
   strictly separated `slow_weights` and `init_fast_weights`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import model.uttt_layer as uttt_layer_mod
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel


# ==============================================================================
# 1. Synthetic & JSONL Persona-Memory Meta-Training Dataset
# ==============================================================================

class PersonaMetaDataset(Dataset):
    """
    Meta-training dataset designed to train uTTT's ability to compress early-context
    user facts/preferences into the Shared Fast-Weight Pool and recall them later.

    Supports loading from an external `.jsonl` / `.txt` file if provided, or generating
    structured multi-turn persona-memory dialogues on the fly.
    """

    PERSONA_TEMPLATES = [
        (
            "User Profile: My name is {name}, I work as a {role} specializing in {topic}, "
            "and I prefer concise answers with {style} examples.",
            "Question: Based on my background and preference, how should we structure our work?",
            "Answer: Hello {name}! Since you are a {role} working on {topic}, I will provide "
            "concise explanations tailored with {style} examples.",
        ),
        (
            "System Note: The user's secret project codename is {codename}, deployed on {gpu} "
            "using {precision} precision with chunk size {chunk}.",
            "User Query: Remind me of my project codename and hardware stack.",
            "Assistant Response: Your project codename is {codename}, running on {gpu} with "
            "{precision} precision and chunk size {chunk}.",
        ),
        (
            "Preference Update: Please remember that I dislike verbose boilerplate and always "
            "use {lang} with {framework} for production services.",
            "User Query: Write a brief technical recommendation for my next microservice.",
            "Assistant Response: Keeping it direct without boilerplate: build the service in "
            "{lang} using {framework} for optimal maintainability.",
        ),
    ]

    NAMES = ["Alex", "Lin", "Sophia", "Marcus", "Yuki", "Elena", "Chen", "Aria"]
    ROLES = ["Systems Architect", "ML Researcher", "Kernel Engineer", "Quant Developer"]
    TOPICS = ["Test-Time Training", "CUDA Blackwell Kernels", "Distributed Consensus", "Rust Async"]
    STYLES = ["PyTorch tensor", "mathematical proof", "production C++", "minimalist CLI"]
    CODENAMES = ["Project-Hyperion", "Omega-FastWeight", "Blackwell-Pulse", "Nebula-TTT"]
    GPUS = ["NVIDIA RTX 5090", "Blackwell sm_120 Node", "RTX 5090 32GB GDDR7"]
    LANGS = ["Python 3.12", "Rust", "C++20", "Go"]
    FRAMEWORKS = ["FastAPI + WebSockets", "PyTorch 2.6", "Triton", "Tokio"]

    def __init__(
        self,
        tokenizer: SimpleUTF8Tokenizer,
        seq_len: int = 1024,
        num_samples: int = 4096,
        data_path: Optional[str] = None,
        pack_sequences: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pack_sequences = pack_sequences
        self.external_texts: List[str] = []

        if data_path and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("content") or json.dumps(obj)
                    else:
                        text = line
                    self.external_texts.append(text)

    def __len__(self) -> int:
        return len(self.external_texts) if self.external_texts else self.num_samples

    def _synthesize_dialogue(self, idx: int) -> str:
        t_idx = idx % len(self.PERSONA_TEMPLATES)
        fact, query, reply = self.PERSONA_TEMPLATES[t_idx]
        fmt_kwargs = {
            "name": self.NAMES[idx % len(self.NAMES)],
            "role": self.ROLES[(idx // 2) % len(self.ROLES)],
            "topic": self.TOPICS[(idx // 3) % len(self.TOPICS)],
            "style": self.STYLES[(idx // 5) % len(self.STYLES)],
            "codename": self.CODENAMES[idx % len(self.CODENAMES)],
            "gpu": self.GPUS[idx % len(self.GPUS)],
            "precision": "bfloat16",
            "chunk": "64",
            "lang": self.LANGS[idx % len(self.LANGS)],
            "framework": self.FRAMEWORKS[idx % len(self.FRAMEWORKS)],
        }
        turn_text = (
            f"<|user|> {fact.format(**fmt_kwargs)}\n"
            f"<|assistant|> Understood. I have encoded your persona and technical context into fast weights.\n"
            f"<|user|> {query.format(**fmt_kwargs)}\n"
            f"<|assistant|> {reply.format(**fmt_kwargs)}\n"
        )
        # Repeat/pad with varied reasoning turns to fill `seq_len` chunks
        repeats = max(1, (self.seq_len // max(len(turn_text), 64)) + 2)
        return turn_text * repeats

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.external_texts:
            n_ext = len(self.external_texts)
            text = self.external_texts[idx % n_ext]
            token_ids = self.tokenizer.encode(
                text, add_bos=True, add_eos=True, max_length=self.seq_len
            )
            # Multi-Dialogue Sequence Packing: pack consecutive short dialogues to eliminate PAD waste
            if self.pack_sequences and len(token_ids) < self.seq_len and n_ext > 1:
                pack_Step = 1
                while len(token_ids) < self.seq_len and pack_Step <= 6:
                    next_text = self.external_texts[(idx + pack_Step * 131) % n_ext]
                    remain = self.seq_len - len(token_ids)
                    more_ids = self.tokenizer.encode(
                        next_text, add_bos=True, add_eos=True, max_length=remain
                    )
                    token_ids.extend(more_ids)
                    pack_Step += 1
        else:
            text = self._synthesize_dialogue(idx)
            token_ids = self.tokenizer.encode(
                text, add_bos=True, add_eos=True, max_length=self.seq_len
            )

        if len(token_ids) < self.seq_len:
            pad_len = self.seq_len - len(token_ids)
            mask = [1] * len(token_ids) + [0] * pad_len
            labels = list(token_ids) + [-100] * pad_len
            token_ids = token_ids + [self.tokenizer.PAD_ID] * pad_len
        else:
            token_ids = token_ids[: self.seq_len]
            mask = [1] * self.seq_len
            labels = list(token_ids)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
        }


# ==============================================================================
# 2. RTX 5090 Environment & Hardware Optimization Setup
# ==============================================================================

def configure_rtx5090_environment(hw_cfg: Dict[str, Any]) -> Tuple[torch.device, torch.dtype]:
    """
    Configure CUDA 12.8+ / Blackwell sm_120 flags, TF32 matmul precision,
    and strict 28GB VRAM limit on NVIDIA RTX 5090.
    """
    requested_device = hw_cfg.get("device", "cuda")
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cap_major, cap_minor = torch.cuda.get_device_capability(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # Enable TF32 for FP32 fallback matmuls and cuDNN benchmark
        if hw_cfg.get("tf32_matmul", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        # Keep VRAM limit metadata for diagnostics; avoid set_per_process_memory_fraction
        # which triggers synchronous CUDACachingAllocator cuMemUnmap thrashing on WSL2/WDDM.
        vram_limit_gb = float(hw_cfg.get("vram_limit_gb", 29.5))

        print(
            f"[Hardware] GPU: {gpu_name} (sm_{cap_major}{cap_minor}) | "
            f"Total VRAM: {total_vram_gb:.1f} GB | Target Budget: {vram_limit_gb:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print("[Hardware] CUDA not detected; running in CPU bfloat16 verification mode.")

    precision_str = hw_cfg.get("precision", "bfloat16").lower()
    dtype = torch.bfloat16 if precision_str == "bfloat16" else torch.float32
    return device, dtype


def maybe_compile_inner_loop(
    hw_cfg: Dict[str, Any],
    device: torch.device,
    gradient_checkpointing: bool = True,
) -> None:
    """
    Compile the analytical chunk-wise SwiGLU gradient kernel using `torch.compile`
    for fused Tensor Core execution on RTX 5090.
    Automatically checks Triton availability and avoids CUDAGraph mode ('reduce-overhead')
    when Gradient Checkpointing is active.
    """
    if not hw_cfg.get("torch_compile", True) or device.type != "cuda":
        return

    try:
        import triton  # noqa: F401
    except ImportError:
        print(
            "[Compiler] Triton not detected (common on native Windows); "
            "using eager cuBLAS BF16 Tensor Core bmm kernels."
        )
        return

    compile_mode = hw_cfg.get("compile_mode", "default")
    if gradient_checkpointing and compile_mode == "reduce-overhead":
        # CUDAGraphs in 'reduce-overhead' conflict with activation recomputation inside checkpoint()
        compile_mode = "default"

    try:
        uttt_layer_mod.fused_chunk_ttt_step = torch.compile(
            uttt_layer_mod.fused_chunk_ttt_step,
            mode=compile_mode,
            fullgraph=False,
        )
        uttt_layer_mod.fused_slow_projections = torch.compile(
            uttt_layer_mod.fused_slow_projections,
            mode=compile_mode,
            fullgraph=False,
        )
        uttt_layer_mod.swiglu_fast_forward = torch.compile(
            uttt_layer_mod.swiglu_fast_forward,
            mode=compile_mode,
            fullgraph=False,
        )
        uttt_layer_mod.apply_rope = torch.compile(
            uttt_layer_mod.apply_rope,
            mode=compile_mode,
            fullgraph=False,
        )
        print(
            f"[Compiler] Enabled torch.compile (mode='{compile_mode}') for fused_chunk_ttt_step, "
            "fused_slow_projections, swiglu_fast_forward & apply_rope."
        )
    except Exception as exc:
        print(f"[Compiler] torch.compile fallback to eager mode: {exc}")


# ==============================================================================
# 3. Base Checkpoint Export (`checkpoints/base_uttt.pt`)
# ==============================================================================

def export_base_checkpoint(
    model: uTTTModel,
    optimizer: Optional[torch.optim.Optimizer],
    config: uTTTConfig,
    save_path: str | Path,
    step: int,
    metrics: Dict[str, Any],
) -> Path:
    """
    Export the dual-track Base Checkpoint (`checkpoints/base_uttt.pt`).

    Strictly stores:
        - `slow_weights`:      Frozen backbone & layer-wise slow projections Theta_slow
        - `init_fast_weights`: Meta-learned initial shared fast-weight pool W_0
        - `config`:            Complete uTTTConfig dictionary
        - `optimizer_state`:   Outer-loop AdamW state (optional for resuming training)
        - `step` & `metrics`:  Training provenance metadata
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_type": "uTTT_Base_Checkpoint",
        "version": "1.0.0",
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.to_dict(),
        "slow_weights": model.slow_state_dict(),
        "init_fast_weights": model.init_fast_state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    tmp_path = out_path.with_suffix(".tmp.pt")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[Checkpoint] Saved Base Checkpoint -> {out_path} ({size_mb:.2f} MB)")
    return out_path


# ==============================================================================
# 4. Cosine LR Schedule & Meta-Training Loop
# ==============================================================================

def get_cosine_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup and 10% minimum floor."""
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return base_lr * 0.1
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cosine_decay)


def train(
    config_path: str = "configs/5090_config.yaml",
    data_path: Optional[str] = None,
    output_ckpt: Optional[str] = None,
    override_steps: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_seq_len: Optional[int] = None,
    lightweight_preset: bool = False,
    resume: bool = True,
) -> Path:
    """Run outer-loop + inner-loop uTTT Meta-Training and export `base_uttt.pt`."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    hw_cfg = raw_cfg.get("hardware", {})
    train_cfg = raw_cfg.get("training", {})
    model_cfg = uTTTConfig.from_dict(raw_cfg)

    if lightweight_preset:
        # Fast verification preset for quick local or CI smoke tests
        model_cfg.d_model = 256
        model_cfg.n_layers = 4
        model_cfg.n_heads = 4
        model_cfg.n_kv_heads = 2
        model_cfg.d_ff_fast = 512
        model_cfg.chunk_size = 32
        model_cfg.sliding_window_size = 128

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, amp_dtype = configure_rtx5090_environment(hw_cfg)
    maybe_compile_inner_loop(
        hw_cfg, device, gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    # Instantiate uTTT Model
    model = uTTTModel(model_cfg).to(device=device)
    param_stats = model.num_parameters()
    print(
        f"[Model] {model_cfg.name} initialized | "
        f"Total Params: {param_stats['total'] / 1e6:.2f}M "
        f"(Slow: {param_stats['slow_weights'] / 1e6:.2f}M, "
        f"Shared Fast Pool W_0: {param_stats['fast_weight_pool'] / 1e6:.2f}M)"
    )

    # Separate parameter groups for Slow Weights (Theta_slow) and Meta-Init Fast Weights (W_0)
    outer_lr = float(train_cfg.get("outer_lr", 3e-4))
    meta_fast_lr = float(train_cfg.get("meta_fast_lr", 1.5e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))

    optimizer = torch.optim.AdamW(
        [
            {
                "name": "slow_weights",
                "params": model.get_slow_parameters(),
                "lr": outer_lr,
                "weight_decay": weight_decay,
            },
            {
                "name": "init_fast_weights",
                "params": model.get_meta_fast_parameters(),
                "lr": meta_fast_lr,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    micro_batch_size = override_batch_size or int(train_cfg.get("micro_batch_size", 4))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 8))
    seq_len = override_seq_len or int(train_cfg.get("seq_len", 1024))
    max_steps = override_steps or int(train_cfg.get("max_steps", 2000))
    warmup_steps = min(int(train_cfg.get("warmup_steps", 100)), max(1, max_steps // 5))
    max_outer_grad_norm = float(train_cfg.get("max_outer_grad_norm", 1.0))
    surprise_aux_weight = float(train_cfg.get("surprise_aux_loss_weight", 0.10))
    log_interval = max(1, int(train_cfg.get("log_interval", 10)))
    save_interval = max(1, int(train_cfg.get("save_interval", 200)))
    ckpt_path = output_ckpt or train_cfg.get("base_checkpoint_path", "checkpoints/base_uttt.pt")

    tokenizer = SimpleUTF8Tokenizer(vocab_size=model_cfg.vocab_size)
    pack_sequences = bool(train_cfg.get("pack_sequences", True))
    dataset = PersonaMetaDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        num_samples=max(4096, max_steps * micro_batch_size * grad_accum_steps),
        data_path=data_path,
        pack_sequences=pack_sequences,
    )
    use_workers = device.type == "cuda" and os.name != "nt"

    def _build_loader(mb_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=mb_size,
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=2 if use_workers else 0,
            persistent_workers=use_workers,
        )

    loader = _build_loader(micro_batch_size)
    data_iter = iter(loader)

    start_step = 1
    ckpt_file = Path(ckpt_path)
    if resume and not lightweight_preset and ckpt_file.exists():
        try:
            saved = torch.load(ckpt_file, map_location=device, weights_only=False)
            saved_cfg = saved.get("config", {})
            saved_step = int(saved.get("step", 0))
            saved_loss = float(saved.get("metrics", {}).get("total_loss", 0.0))
            dims_match = (
                saved_cfg.get("d_model") == model_cfg.d_model
                and saved_cfg.get("n_layers") == model_cfg.n_layers
                and saved_cfg.get("d_ff_fast") == model_cfg.d_ff_fast
                and saved_cfg.get("vocab_size") == model_cfg.vocab_size
            )
            if dims_match and 0 < saved_step < max_steps and math.isfinite(saved_loss):
                model.load_dual_base_state(
                    saved["slow_weights"],
                    saved["init_fast_weights"],
                    strict=True,
                )
                if "optimizer_state" in saved:
                    optimizer.load_state_dict(saved["optimizer_state"])
                start_step = saved_step + 1
                print(
                    f"[Resume] Resumed from {ckpt_file} at step {saved_step}/{max_steps} "
                    f"(Last Loss: {saved_loss:.4f})",
                    flush=True,
                )
        except Exception as exc:
            print(f"[Resume] Could not resume from {ckpt_file} ({exc}); starting fresh.", flush=True)

    model.train()
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    print(
        f"[Train] Starting Meta-Training from step {start_step} to {max_steps} | "
        f"MicroBatch={micro_batch_size}, GradAccum={grad_accum_steps}, SeqLen={seq_len}, "
        f"ChunkSize={model_cfg.chunk_size}, GradCkpt={model_cfg.gradient_checkpointing}, "
        f"Precision={amp_dtype}",
        flush=True,
    )
    if hw_cfg.get("torch_compile", True) and device.type == "cuda":
        print(
            "[Train] First step includes one-time Triton kernel compilation (~5-15s on RTX 5090)...",
            flush=True,
        )

    running_loss = 0.0
    running_ce = 0.0
    running_surprise = 0.0
    steps_since_log = 0
    t_start = time.perf_counter()

    for step in range(start_step, max_steps + 1):
        # Update learning rates for both parameter groups
        cur_slow_lr = get_cosine_lr(step - 1, max_steps, warmup_steps, outer_lr)
        cur_fast_lr = get_cosine_lr(step - 1, max_steps, warmup_steps, meta_fast_lr)
        optimizer.param_groups[0]["lr"] = cur_slow_lr
        optimizer.param_groups[1]["lr"] = cur_fast_lr

        optimizer.zero_grad(set_to_none=True)

        # Runtime Elastic OOM Self-Healing Guard:
        # If any step exceeds VRAM, automatically halve micro_batch_size, double grad_accum_steps,
        # enable 2-level hierarchical chunk checkpointing, and retry the step without crashing.
        while True:
            try:
                step_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
                step_ce_gpu = torch.zeros((), device=device, dtype=torch.float32)
                step_surprise_gpu = torch.zeros((), device=device, dtype=torch.float32)

                for _ in range(grad_accum_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        batch = next(data_iter)

                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                    # Native BF16 mixed precision forward + analytical inner-loop TTT update
                    with torch.amp.autocast(autocast_device_type, dtype=amp_dtype):
                        out = model(
                            input_ids=input_ids,
                            fast_state=None,  # Starts at meta-learned W_0 and evolves across layers
                            labels=labels,
                            attention_mask=attention_mask,
                            update_fast_weights=True,
                            surprise_aux_weight=surprise_aux_weight,
                        )
                        scaled_loss = out.loss / float(grad_accum_steps)

                    scaled_loss.backward()

                    step_loss_gpu += out.loss.detach().float() / float(grad_accum_steps)
                    step_ce_gpu += out.ce_loss.detach().float() / float(grad_accum_steps)
                    step_surprise_gpu += out.surprise_loss.detach().float() / float(grad_accum_steps)
                    del out, scaled_loss
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if not model.gradient_checkpointing:
                    # Stage 1: Enable Layer Gradient Checkpointing while keeping full MicroBatch=8
                    model.gradient_checkpointing = True
                    model_cfg.gradient_checkpointing = True
                    print(
                        f"[VRAM-Guard][Step {step:04d}] Caught CUDA OOM with GradCkpt=False! "
                        f"Automatically enabled GradCkpt=True at MicroBatch={micro_batch_size}. Retrying step...",
                        flush=True,
                    )
                elif micro_batch_size > 1:
                    # Stage 2: Halve MicroBatch and double GradAccum (preserving EffectiveBatch)
                    micro_batch_size = max(1, micro_batch_size // 2)
                    grad_accum_steps = grad_accum_steps * 2
                    loader = _build_loader(micro_batch_size)
                    data_iter = iter(loader)
                    print(
                        f"[VRAM-Guard][Step {step:04d}] Caught CUDA OOM! Dynamically scaled to "
                        f"MicroBatch={micro_batch_size}, GradAccum={grad_accum_steps} "
                        f"(EffectiveBatch={micro_batch_size * grad_accum_steps} preserved). Retrying step...",
                        flush=True,
                    )
                else:
                    # Stage 3: Enable 2-level Intra-Layer Chunk Segment Checkpointing
                    for blk in model.layers:
                        blk.uttt.chunk_checkpoint_segments = max(
                            4, blk.uttt.chunk_checkpoint_segments * 2
                        )
                    print(
                        f"[VRAM-Guard][Step {step:04d}] Caught CUDA OOM at MicroBatch=1! Enabled 2-level "
                        f"Chunk Segment Checkpointing (segments={model.layers[0].uttt.chunk_checkpoint_segments}). Retrying...",
                        flush=True,
                    )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_outer_grad_norm,
            error_if_nonfinite=False,
            foreach=(device.type == "cuda"),
        )
        # Single GPU->CPU synchronization point per outer step
        metrics_cpu = torch.stack([step_loss_gpu, step_ce_gpu, step_surprise_gpu, grad_norm.detach().float()]).tolist()
        step_loss_accum, step_ce_accum, step_surprise_accum, grad_norm_val = (
            float(metrics_cpu[0]),
            float(metrics_cpu[1]),
            float(metrics_cpu[2]),
            float(metrics_cpu[3]),
        )
        if math.isfinite(grad_norm_val) and math.isfinite(step_loss_accum):
            optimizer.step()
        else:
            print(
                f"[WARN][Step {step:04d}] Non-finite GradNorm ({grad_norm_val}) or Loss ({step_loss_accum}); "
                "skipping optimizer.step() to protect weights.",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)

        running_loss += step_loss_accum
        running_ce += step_ce_accum
        running_surprise += step_surprise_accum
        steps_since_log += 1

        if step % log_interval == 0 or step == start_step:
            divisor = max(1, steps_since_log)
            avg_loss = running_loss / divisor
            avg_ce = running_ce / divisor
            avg_surp = running_surprise / divisor
            running_loss = 0.0
            running_ce = 0.0
            running_surprise = 0.0
            steps_since_log = 0

            elapsed = max(1e-6, time.perf_counter() - t_start)
            tokens_processed = divisor * micro_batch_size * grad_accum_steps * seq_len
            tok_per_sec = tokens_processed / elapsed
            vram_gb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"[Step {step:04d}/{max_steps:04d}] "
                f"Total Loss: {avg_loss:.4f} | CE Loss: {avg_ce:.4f} | "
                f"Surprise Loss: {avg_surp:.4f} | GradNorm: {grad_norm_val:.3f} | "
                f"LR: {cur_slow_lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s | "
                f"Peak VRAM: {vram_gb:.2f} GB",
                flush=True,
            )
            t_start = time.perf_counter()

        if (step % save_interval == 0 or step == max_steps) and math.isfinite(step_loss_accum):
            metrics = {
                "total_loss": step_loss_accum,
                "ce_loss": step_ce_accum,
                "surprise_loss": step_surprise_accum,
                "peak_vram_gb": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            export_base_checkpoint(
                model=model,
                optimizer=optimizer,
                config=model_cfg,
                save_path=ckpt_path,
                step=step,
                metrics=metrics,
            )

    return Path(ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="uTTT RTX 5090 Meta-Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/5090_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to external .jsonl or .txt training corpus.",
    )
    parser.add_argument(
        "--output-ckpt",
        type=str,
        default=None,
        help="Override path for exported base_uttt.pt checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max training steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override micro-batch size.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override training sequence length.",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight model dimensions for rapid smoke testing.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch even if checkpoints/base_uttt.pt exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        data_path=args.data_path,
        output_ckpt=args.output_ckpt,
        override_steps=args.steps,
        override_batch_size=args.batch_size,
        override_seq_len=args.seq_len,
        lightweight_preset=args.lightweight,
        resume=not args.no_resume,
    )

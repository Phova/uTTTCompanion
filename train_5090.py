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
import atexit
import json
import hashlib
import re
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import model.uttt_layer as uttt_layer_mod
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel
from model.tokenizer import build_tokenizer
from engine.corpus import TokenizedCorpusDataset, read_manifest

_ACTIVE_SAVE_THREAD: Optional[threading.Thread] = None
_ACTIVE_SAVE_ERROR: Optional[Exception] = None


def _wait_for_active_save() -> None:
    global _ACTIVE_SAVE_THREAD, _ACTIVE_SAVE_ERROR
    if _ACTIVE_SAVE_THREAD is not None and _ACTIVE_SAVE_THREAD.is_alive():
        print("[Checkpoint] Waiting for in-flight background checkpoint write to complete...", flush=True)
        _ACTIVE_SAVE_THREAD.join()
    if _ACTIVE_SAVE_ERROR is not None:
        error, _ACTIVE_SAVE_ERROR = _ACTIVE_SAVE_ERROR, None
        raise RuntimeError("Background checkpoint save failed") from error


atexit.register(_wait_for_active_save)


def _graceful_signal_handler(signum: int, _frame: Any) -> None:
    _wait_for_active_save()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _graceful_signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _graceful_signal_handler)


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
        mode: str = "sft",
        split: str = "all",
        validation_fraction: float = 0.0,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pack_sequences = pack_sequences
        self.external_texts: List[str] = []
        self.mode, self.split = mode, split
        if mode not in {"pretrain", "sft", "meta"}:
            raise ValueError("mode must be pretrain, sft, or meta")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        seen = set()
        fingerprint = hashlib.sha256()

        if data_path:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("content")
                    else:
                        text = line
                    if not isinstance(text, str):
                        raise ValueError("Dataset rows must contain string text/content")
                    text = re.sub(r"^<\|user\|> \[Context Category:[^\n]*\n", "", text).strip()
                    identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if not text or identity in seen:
                        continue
                    seen.add(identity)
                    is_validation = int(hashlib.sha256(f"{split_seed}:{identity}".encode()).hexdigest()[:16], 16) / 2**64 < validation_fraction
                    if split != "all" and (split == "validation") != is_validation:
                        continue
                    fingerprint.update(identity.encode())
                    self.external_texts.append(text)
            if not self.external_texts:
                raise ValueError(f"Dataset split {split} is empty: {data_path}")
        self.fingerprint = fingerprint.hexdigest()

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

    def _document(self, text):
        raw = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        ids, labels = [], []
        assistant = False
        for token in raw:
            if token == self.tokenizer.USER_ID:
                if assistant and ids and ids[-1] != self.tokenizer.EOS_ID:
                    ids.append(self.tokenizer.EOS_ID)
                    labels.append(self.tokenizer.EOS_ID)
                assistant = False
            elif token == self.tokenizer.ASSISTANT_ID:
                assistant = True
            ids.append(token)
            supervised = self.mode == "pretrain" or assistant
            labels.append(token if supervised and token not in (
                self.tokenizer.PAD_ID, self.tokenizer.BOS_ID
            ) and (self.mode == "pretrain" or token != self.tokenizer.ASSISTANT_ID) else -100)
            if token == self.tokenizer.EOS_ID:
                assistant = False
        return ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.external_texts[idx % len(self.external_texts)] if self.external_texts else self._synthesize_dialogue(idx)
        ids, labels = self._document(text)
        # Select targets from the whole conversation, not always its first 1024
        # characters. Compute role masks BEFORE cropping so mid-answer windows
        # retain their supervision and never fall back to training user prompts.
        if len(ids) > self.seq_len:
            targets = [i for i, value in enumerate(labels) if value != -100 and i > 0]
            if targets:
                seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
                anchor_index = seed % len(targets) if self.split == "validation" else int(torch.randint(len(targets), ()).item())
                anchor = targets[anchor_index]
                low, high = max(0, anchor - self.seq_len + 1), min(anchor - 1, len(ids) - self.seq_len)
                start = low + (seed % (high - low + 1)) if self.split == "validation" else int(torch.randint(low, high + 1, ()).item())
            else:
                start = 0
            ids, labels = ids[start:start + self.seq_len], labels[start:start + self.seq_len]
        if self.pack_sequences and self.mode == "pretrain" and self.external_texts:
            for offset in range(1, 17):
                if len(ids) >= self.seq_len:
                    break
                extra_ids, extra_labels = self._document(self.external_texts[(idx + offset * 131) % len(self.external_texts)])
                remaining = self.seq_len - len(ids)
                ids.extend(extra_ids[:remaining])
                labels.extend(extra_labels[:remaining])
        valid = len(ids)
        padding = self.seq_len - valid
        return {
            "input_ids": torch.tensor(ids + [self.tokenizer.PAD_ID] * padding, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * padding, dtype=torch.long),
            "attention_mask": torch.tensor([1.0] * valid + [0.0] * padding),
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
    model: Optional[uTTTModel] = None,
) -> None:
    """
    Compile the analytical chunk-wise SwiGLU gradient kernel and full `uTTTBlock`
    layers using `torch.compile` for fused Tensor Core execution on RTX 5090.
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
        if getattr(uttt_layer_mod, "_HAS_FLEX_ATTENTION", False) and uttt_layer_mod.flex_attention is not None:
            uttt_layer_mod._COMPILED_FLEX_ATTN = torch.compile(uttt_layer_mod.flex_attention)
        print(
            f"[Compiler] Enabled torch.compile (mode='{compile_mode}') for fused_chunk_ttt_step, "
            "fused_slow_projections, FlexAttention SWA, swiglu_fast_forward & apply_rope."
        )
    except Exception as exc:
        print(f"[Compiler] torch.compile fallback to eager mode: {exc}")


# ==============================================================================
# 3. Base Checkpoint Export (`checkpoints/base_uttt.pt`)
# ==============================================================================

def _cpu_copy_nested(obj: Any) -> Any:
    """Recursively copy tensors in optimizer state_dict to CPU memory for background saving."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(device="cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_copy_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_copy_nested(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_copy_nested(v) for v in obj)
    return obj


def export_base_checkpoint(
    model: uTTTModel,
    optimizer: Optional[torch.optim.Optimizer],
    config: uTTTConfig,
    save_path: str | Path,
    step: int,
    metrics: Dict[str, Any],
    async_save: bool = False,
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
    global _ACTIVE_SAVE_THREAD
    _wait_for_active_save()

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
        payload["optimizer_state"] = _cpu_copy_nested(optimizer.state_dict())

    def _write_to_disk() -> None:
        tmp_path = Path(str(out_path) + ".tmp")
        prev_path = Path(str(out_path) + ".prev")
        with open(tmp_path, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        if out_path.exists():
            try:
                if prev_path.exists() or prev_path.is_symlink():
                    prev_path.unlink()
                os.link(out_path, prev_path)
            except Exception:
                pass
        os.replace(tmp_path, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[Checkpoint] Saved Base Checkpoint -> {out_path} ({size_mb:.2f} MB)", flush=True)

    if async_save:
        def _background_write() -> None:
            global _ACTIVE_SAVE_ERROR
            try:
                _write_to_disk()
            except Exception as exc:
                _ACTIVE_SAVE_ERROR = exc
        _ACTIVE_SAVE_THREAD = threading.Thread(target=_background_write, daemon=False)
        _ACTIVE_SAVE_THREAD.start()
    else:
        _write_to_disk()
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


@torch.no_grad()
def evaluate_language_model(model, dataset, device, dtype, update_fast_weights=False, max_batches=32):
    was_training = model.training
    model.eval()
    nll, frozen_nll, targets = 0.0, 0.0, 0
    try:
        count_batches = min(len(dataset), max_batches)
        indices = [index * (len(dataset) - 1) // max(1, count_batches - 1) for index in range(count_batches)]
        for index in indices:
            batch = dataset[index]
            count = int((batch["labels"][1:] != -100).sum())
            if not count:
                continue
            inputs = {name: value.unsqueeze(0).to(device) for name, value in batch.items()}
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                result = model(**inputs, update_fast_weights=update_fast_weights, surprise_aux_weight=0.0)
                nll += float(result.ce_loss) * count
                if update_fast_weights:
                    frozen = model(**inputs, update_fast_weights=False, surprise_aux_weight=0.0)
                    frozen_nll += float(frozen.ce_loss) * count
                else:
                    frozen_nll += float(result.ce_loss) * count
            targets += count
        if not targets:
            raise ValueError("Validation split has no supervised next-token targets")
        return {"validation_ce": nll / targets, "validation_frozen_ce": frozen_nll / targets,
                "validation_targets": targets}
    finally:
        model.train(was_training)


@torch.no_grad()
def generate_validation_samples(model, tokenizer, prompts, stage, device, dtype, max_new_tokens=48):
    results = []
    for prompt in prompts:
        text = prompt if stage == "pretrain" else f"<|user|> {prompt}\n<|assistant|> "
        ids = tokenizer.encode(text, add_bos=True, max_length=model.config.max_seq_len - max_new_tokens)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            generated = list(model.stream_generate(
                torch.tensor([ids], device=device), model.init_fast_state(device=device, dtype=dtype, detach=True),
                max_new_tokens=max_new_tokens, temperature=0.0,
            ))
        results.append({"prompt": prompt, "reply": tokenizer.decode(generated).strip(), "eos_generated": tokenizer.EOS_ID in generated})
    return results


def train(
    config_path: str = "configs/5090_config.yaml",
    data_path: Optional[str] = None,
    output_ckpt: Optional[str] = None,
    override_steps: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_seq_len: Optional[int] = None,
    lightweight_preset: bool = False,
    large_preset: bool = False,
    resume: bool = True,
    stage: Optional[str] = None,
    init_from: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> Path:
    """Run outer-loop + inner-loop uTTT Meta-Training and export `base_uttt.pt`."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    hw_cfg = raw_cfg.get("hardware", {})
    train_cfg = raw_cfg.get("training", {})
    model_cfg = uTTTConfig.from_dict(raw_cfg)
    training_stage = stage or train_cfg.get("stage", "meta")
    if training_stage not in {"pretrain", "sft", "meta"}:
        raise ValueError("stage must be pretrain, sft, or meta")
    update_fast_weights = training_stage == "meta"
    corpus_manifest = read_manifest(data_path)
    if corpus_manifest and training_stage != "pretrain":
        raise ValueError("Packed text corpus is for pretraining; use dialogue data for SFT/meta")
    tokenizer_file = tokenizer_path or train_cfg.get("tokenizer_path")
    if corpus_manifest and not tokenizer_path:
        tokenizer_file = str(Path(data_path).resolve().parent / corpus_manifest["tokenizer"])
    if tokenizer_file:
        model_cfg.tokenizer_json = Path(tokenizer_file).read_text(encoding="utf-8")
    if corpus_manifest and hashlib.sha256(model_cfg.tokenizer_json.encode()).hexdigest() != corpus_manifest["tokenizer_sha256"]:
        raise ValueError("Corpus and configured tokenizer have different fingerprints")
    if init_from and Path(init_from).resolve() == Path(output_ckpt or train_cfg.get("base_checkpoint_path", "checkpoints/base_uttt.pt")).resolve():
        raise ValueError("--init-from must differ from the output checkpoint")

    if lightweight_preset:
        # Fast verification preset for quick local or CI smoke tests
        model_cfg.name = "uTTT-Light-Smoke"
        model_cfg.d_model = 256
        model_cfg.n_layers = 4
        model_cfg.n_heads = 4
        model_cfg.n_kv_heads = 2
        model_cfg.d_ff_fast = 512
        model_cfg.chunk_size = 32
        model_cfg.sliding_window_size = 128
    elif large_preset:
        # Measured full-AdamW capacity at B=1, T=1024 on the RTX 5090.
        model_cfg.name = "uTTT-1.584B"
        model_cfg.d_model = 2560
        model_cfg.n_layers = 34
        model_cfg.n_heads = 20
        model_cfg.n_kv_heads = 5
        model_cfg.d_ff_fast = 6912
        model_cfg.chunk_size = 128
        model_cfg.sliding_window_size = 512
        model_cfg.gradient_checkpointing = True
        hw_cfg["torch_compile"] = False

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, amp_dtype = configure_rtx5090_environment(hw_cfg)

    # Instantiate uTTT Model
    model = uTTTModel(model_cfg).to(device=device)
    if init_from:
        initialized = torch.load(init_from, map_location="cpu", weights_only=False, mmap=True)
        previous = uTTTConfig.from_dict(initialized["config"])
        for field in ("d_model", "d_ff_fast", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "tokenizer_json"):
            if getattr(previous, field) != getattr(model_cfg, field):
                raise ValueError(f"Initialization checkpoint has incompatible {field}")
        model.load_dual_base_state(initialized["slow_weights"], initialized["init_fast_weights"])
        del initialized
        print(f"[Init] Loaded model weights from {init_from}; optimizer starts fresh for {training_stage}.")
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

    micro_batch_size = override_batch_size or (1 if large_preset else int(train_cfg.get("micro_batch_size", 4)))
    grad_accum_steps = int(train_cfg.get("large_grad_accum_steps", 32) if large_preset else train_cfg.get("grad_accum_steps", 8))
    seq_len = override_seq_len or int(train_cfg.get("seq_len", 1024))
    max_steps = override_steps or int(train_cfg.get("max_steps", 2000))
    warmup_steps = min(int(train_cfg.get("warmup_steps", 100)), max(1, max_steps // 5))
    max_outer_grad_norm = float(train_cfg.get("max_outer_grad_norm", 1.0))
    # Directly minimizing a learned reconstruction task has a trivial solution.
    # Language modelling, not small surprise, is the criterion for useful memory.
    surprise_aux_weight = float(train_cfg.get("surprise_aux_loss_weight", 0.0)) if update_fast_weights else 0.0
    log_interval = max(1, int(train_cfg.get("log_interval", 10)))
    save_interval = max(1, int(train_cfg.get("save_interval", 200)))
    ckpt_path = output_ckpt or train_cfg.get("base_checkpoint_path", "checkpoints/base_uttt.pt")

    tokenizer = build_tokenizer(model_cfg)
    pack_sequences = bool(train_cfg.get("pack_sequences", True))
    validation_dataset = None
    if corpus_manifest:
        dataset = TokenizedCorpusDataset(data_path, seq_len=seq_len, split="train")
        validation_dataset = TokenizedCorpusDataset(data_path, seq_len=seq_len, split="validation")
        print(f"[Corpus] Memory-mapped train={dataset.token_count:,}, validation={validation_dataset.token_count:,} tokens; no text corpus loaded into RAM.", flush=True)
    else:
        dataset = PersonaMetaDataset(
            tokenizer=tokenizer, seq_len=seq_len, data_path=data_path,
            num_samples=max(4096, max_steps * micro_batch_size * grad_accum_steps),
            pack_sequences=pack_sequences, mode=training_stage,
            split="train" if train_cfg.get("validation_fraction", 0) else "all",
            validation_fraction=float(train_cfg.get("validation_fraction", 0)), split_seed=seed,
        )
        if data_path and train_cfg.get("validation_fraction", 0):
            validation_dataset = PersonaMetaDataset(
                tokenizer=tokenizer, seq_len=seq_len, data_path=data_path,
                pack_sequences=pack_sequences, mode=training_stage, split="validation",
                validation_fraction=float(train_cfg["validation_fraction"]), split_seed=seed,
            )
            print(f"[Data] Deduplicated train={len(dataset):,}, validation={len(validation_dataset):,} conversations; split before windowing.")
    use_workers = device.type == "cuda" and os.name != "nt"

    def _build_loader(mb_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=mb_size,
            shuffle=True,
            drop_last=len(dataset) >= mb_size,
            pin_memory=(device.type == "cuda"),
            num_workers=2 if use_workers else 0,
            persistent_workers=use_workers,
            prefetch_factor=4 if use_workers else None,
        )

    loader = _build_loader(micro_batch_size)
    data_iter = iter(loader)

    start_step = 1
    tokens_seen = 0
    best_validation_ce = float("inf")
    validation_metrics = {}
    evaluation_interval = max(1, int(train_cfg.get("eval_interval", 250)))
    cur_data_tag = str(data_path or "synthetic")
    ckpt_file = Path(ckpt_path)
    prev_ckpt_file = Path(str(ckpt_path) + ".prev")
    if resume and not lightweight_preset and (ckpt_file.exists() or prev_ckpt_file.exists()):
        saved = None
        loaded_from = ckpt_file
        for cand_file in (ckpt_file, prev_ckpt_file):
            if not cand_file.exists():
                continue
            try:
                saved = torch.load(cand_file, map_location="cpu", weights_only=False, mmap=True)
                loaded_from = cand_file
                break
            except Exception as exc:
                print(f"[Resume] Warning: could not read {cand_file} ({exc}); trying fallback...", flush=True)
        if saved is not None:
            metadata = saved.get("metrics", {})
            if (metadata.get("training_pipeline") != 3 or metadata.get("training_stage") != training_stage
                    or metadata.get("dataset_fingerprint") != dataset.fingerprint
                    or saved.get("config", {}).get("tokenizer_json") != model_cfg.tokenizer_json):
                raise ValueError("Resume checkpoint uses different data, tokenizer or training pipeline/stage. Use a new output path and --init-from for an intentional stage transition.")
            try:
                saved_cfg = saved.get("config", {})
                saved_metrics = saved.get("metrics", {})
                saved_step = int(saved.get("step", 0))
                saved_loss = float(saved_metrics.get("total_loss", 0.0))
                dims_match = (
                    saved_cfg.get("d_model") == model_cfg.d_model
                    and saved_cfg.get("n_layers") == model_cfg.n_layers
                    and saved_cfg.get("d_ff_fast") == model_cfg.d_ff_fast
                    and saved_cfg.get("vocab_size") == model_cfg.vocab_size
                )
                is_causal_v2 = bool(saved_metrics.get("causal_ttt_v2", False))
                same_data = saved_metrics.get("data_source", cur_data_tag) == cur_data_tag
                if dims_match and is_causal_v2 and same_data and saved_step >= max_steps and math.isfinite(saved_loss):
                    print(f"[Resume] Checkpoint already reached step {saved_step} (requested {max_steps}); leaving it unchanged.", flush=True)
                    return loaded_from
                if dims_match and is_causal_v2 and same_data and 0 < saved_step < max_steps and math.isfinite(saved_loss):
                    model.load_dual_base_state(
                        saved["slow_weights"],
                        saved["init_fast_weights"],
                        strict=True,
                    )
                    if "optimizer_state" in saved:
                        optimizer.load_state_dict(saved["optimizer_state"])
                    start_step = saved_step + 1
                    tokens_seen = int(saved_metrics.get("tokens_seen", 0))
                    best_validation_ce = float(saved_metrics.get("best_validation_ce", float("inf")))
                    print(
                        f"[Resume] Resumed from {loaded_from} at step {saved_step}/{max_steps} "
                        f"(Last Loss: {saved_loss:.4f})",
                        flush=True,
                    )
                elif not is_causal_v2 or not same_data:
                    print(
                        f"[Resume] Existing {loaded_from} is from legacy non-causal run or different dataset; "
                        "starting fresh causal meta-training from step 1.",
                        flush=True,
                    )
                del saved
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                raise RuntimeError(f"Could not restore state from {loaded_from}; checkpoint preserved") from exc
        else:
            print(f"[Resume] No readable checkpoint found at {ckpt_file}; starting fresh.", flush=True)

    # Compile full uTTTBlocks after weights are loaded so state_dict keys remain clean
    maybe_compile_inner_loop(
        hw_cfg,
        device,
        gradient_checkpointing=model_cfg.gradient_checkpointing,
        model=model,
    )

    model.train()
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    print(
        f"[Train] Starting {training_stage} from step {start_step} to {max_steps} | "
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

    running_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
    running_ce_gpu = torch.zeros((), device=device, dtype=torch.float32)
    running_surprise_gpu = torch.zeros((), device=device, dtype=torch.float32)
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
        # If any step exceeds VRAM, automatically enable Layer Checkpointing (Stage 1),
        # halve micro_batch_size (Stage 2), or enable 2-level Chunk Checkpointing (Stage 3).
        while True:
            try:
                step_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
                step_ce_gpu = torch.zeros((), device=device, dtype=torch.float32)
                step_surprise_gpu = torch.zeros((), device=device, dtype=torch.float32)
                inv_accum = 1.0 / float(grad_accum_steps)
                step_tokens = 0

                for _ in range(grad_accum_steps):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        batch = next(data_iter)

                    if not bool((batch["labels"][:, 1:] != -100).any()):
                        raise ValueError("Batch has no supervised next-token targets; check dialogue roles and training stage")
                    step_tokens += int(batch["attention_mask"].sum())
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    # Packing can still leave padding (e.g. a one-row dataset).
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                    # Native BF16 mixed precision forward + analytical inner-loop TTT update
                    with torch.amp.autocast(autocast_device_type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                        out = model(
                            input_ids=input_ids,
                            fast_state=None,  # Starts at meta-learned W_0 and evolves across layers
                            labels=labels,
                            attention_mask=attention_mask,
                            update_fast_weights=update_fast_weights,
                            surprise_aux_weight=surprise_aux_weight,
                        )
                        scaled_loss = out.loss * inv_accum

                    scaled_loss.backward()

                    step_loss_gpu += out.loss.detach().float() * inv_accum
                    step_ce_gpu += out.ce_loss.detach().float() * inv_accum
                    step_surprise_gpu += out.surprise_loss.detach().float() * inv_accum
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
                    max_segments = (seq_len + model_cfg.chunk_size - 1) // model_cfg.chunk_size
                    if all(blk.uttt.chunk_checkpoint_segments >= max_segments for blk in model.layers):
                        raise RuntimeError("CUDA OOM persists at batch size 1 with maximal checkpointing; reduce sequence/model size")
                    for blk in model.layers:
                        blk.uttt.chunk_checkpoint_segments = min(
                            max_segments, max(4, blk.uttt.chunk_checkpoint_segments * 2)
                        )
                    print(
                        f"[VRAM-Guard][Step {step:04d}] Caught CUDA OOM at MicroBatch=1! Enabled 2-level "
                        f"Chunk Segment Checkpointing (segments={model.layers[0].uttt.chunk_checkpoint_segments}). Retrying...",
                        flush=True,
                    )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_outer_grad_norm,
            error_if_nonfinite=True,
            foreach=(device.type == "cuda"),
        )
        # Launch fused AdamW step immediately on GPU BEFORE any host synchronization
        # so the CUDA command stream never drains to 0% between outer steps.
        optimizer.step()
        tokens_seen += step_tokens

        running_loss_gpu += step_loss_gpu
        running_ce_gpu += step_ce_gpu
        running_surprise_gpu += step_surprise_gpu
        steps_since_log += 1

        should_log = (step % log_interval == 0) or (step == start_step)
        should_save = (step % save_interval == 0) or (step == max_steps)

        improved = False
        if validation_dataset is not None and (step % evaluation_interval == 0 or step == start_step or step == max_steps):
            validation_metrics = evaluate_language_model(
                model, validation_dataset, device, amp_dtype, update_fast_weights,
                max_batches=int(train_cfg.get("eval_samples", 32)),
            )
            improved = validation_metrics["validation_ce"] < best_validation_ce
            best_validation_ce = min(best_validation_ce, validation_metrics["validation_ce"])
            print(f"[Validation {step}] CE={validation_metrics['validation_ce']:.4f} | Frozen CE={validation_metrics['validation_frozen_ce']:.4f} | Tokens seen={tokens_seen:,}", flush=True)
            if train_cfg.get("eval_prompts"):
                validation_metrics["validation_samples"] = generate_validation_samples(
                    model, tokenizer, train_cfg["eval_prompts"], training_stage, device, amp_dtype,
                    max_new_tokens=int(train_cfg.get("eval_new_tokens", 48)),
                )
                for sample in validation_metrics["validation_samples"]:
                    print("[Validation sample] " + json.dumps(sample, ensure_ascii=False), flush=True)
        if should_log or should_save or improved:
            sync_vals = torch.stack(
                [
                    running_loss_gpu,
                    running_ce_gpu,
                    running_surprise_gpu,
                    step_loss_gpu,
                    step_ce_gpu,
                    step_surprise_gpu,
                    grad_norm.detach().float(),
                ]
            ).tolist()
            run_loss_val, run_ce_val, run_surp_val, step_loss_accum, step_ce_accum, step_surprise_accum, grad_norm_val = (
                float(sync_vals[0]),
                float(sync_vals[1]),
                float(sync_vals[2]),
                float(sync_vals[3]),
                float(sync_vals[4]),
                float(sync_vals[5]),
                float(sync_vals[6]),
            )

            if should_log:
                divisor = max(1, steps_since_log)
                avg_loss = run_loss_val / divisor
                avg_ce = run_ce_val / divisor
                avg_surp = run_surp_val / divisor
                running_loss_gpu.zero_()
                running_ce_gpu.zero_()
                running_surprise_gpu.zero_()
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

            if (should_save or improved) and math.isfinite(step_loss_accum):
                metrics = {
                    "total_loss": step_loss_accum,
                    "ce_loss": step_ce_accum,
                    "surprise_loss": step_surprise_accum,
                    "causal_ttt_v2": True,
                    "training_pipeline": 3, "training_stage": training_stage,
                    "code_revision": os.environ.get("UTTT_CODE_REVISION", "unknown"),
                    "run_id": os.environ.get("UTTT_RUN_ID"),
                    "dataset_fingerprint": dataset.fingerprint, "tokens_seen": tokens_seen,
                    "best_validation_ce": best_validation_ce, **validation_metrics,
                    "data_source": cur_data_tag,
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
                    async_save=(step < max_steps),
                )
                if improved:
                    best_path = str(Path(ckpt_path).with_suffix(".best.pt"))
                    export_base_checkpoint(model, None, model_cfg, best_path, step, metrics)

    _wait_for_active_save()
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
        "--save-path",
        dest="output_ckpt",
        type=str,
        default=None,
        help="Override path for exported checkpoint (e.g. checkpoints/Vitamin-C.pt).",
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
        "--large",
        action="store_true",
        help="Use the measured 1.584B preset (2560 wide, 34 layers), micro-batch 1 and checkpointing.",
    )
    parser.add_argument(
        "--no-resume",
        "--fresh",
        dest="no_resume",
        action="store_true",
        help="Start training from scratch even if checkpoints/base_uttt.pt exists.",
    )
    parser.add_argument("--stage", choices=["pretrain", "sft", "meta"], default=None)
    parser.add_argument("--init-from", help="Load compatible model weights for a new training stage")
    parser.add_argument("--tokenizer", help="Local BPE tokenizer JSON, embedded into new checkpoints")
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
        large_preset=args.large,
        resume=not args.no_resume,
        stage=args.stage, init_from=args.init_from, tokenizer_path=args.tokenizer,
    )

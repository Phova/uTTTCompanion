"""Isolated short-answer SFT A/B; read a full checkpoint and never save weights.

Run each variant in a fresh process after the GPU owner has released the device.
Sixteen seeded B1 examples are materialized once on CPU and regrouped with the
normal trainer's execution policy. The identical plan is replayed for every step.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="Optional independent source checkout")
    parser.add_argument("--config", default="configs/5090_continue_20261107.yaml")
    parser.add_argument("--checkpoint", required=True, help="Read-only full model/Adam checkpoint")
    parser.add_argument("--data", required=True, help="Original instruction JSONL")
    parser.add_argument("--loss-only-head", action="store_true")
    parser.add_argument("--shared-readonly-gemm", action="store_true")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", help="Write measurements to this JSON file only")
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.steps < 1:
        parser.error("warmup steps must be nonnegative and measured steps must be positive")
    source = Path(args.source_root).resolve() if args.source_root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))

    def resolve(value):
        path = Path(value).expanduser()
        return (source / path).resolve() if not path.is_absolute() else path.resolve()

    config_path, checkpoint_path, data_path = map(resolve, (args.config, args.checkpoint, args.data))
    output = Path(args.output).expanduser().resolve() if args.output else None
    if output is not None and (output.suffix.lower() != ".json" or output in (config_path, checkpoint_path, data_path)):
        parser.error("--output must be a separate .json measurements file")
    import yaml
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hardware, training = dict(raw.get("hardware", {})), raw.get("training", {})
    if hardware.get("device") != "cuda" or hardware.get("torch_compile") is not False:
        parser.error("benchmark requires the CUDA config with torch_compile=false")
    if (training.get("stage") != "sft" or training.get("micro_batch_size") != 1
            or training.get("grad_accum_steps") != 16 or training.get("pack_sequences", True)
            or training.get("parameter_scope", "all") != "all"):
        parser.error("benchmark requires unpacked all-parameter SFT with semantic B1/A16")
    if hardware.get("cuda_allocator_conf"):
        # This must precede importing torch or initializing the CUDA allocator.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", hardware["cuda_allocator_conf"])

    import torch
    from torch.utils.data import DataLoader
    from engine.corpus import ResumableBatchSampler
    from engine.sft_execution import (materialize_sft_execution_step, supervised_token_indices,
                                      validate_sft_execution)
    from model.tokenizer import build_tokenizer
    from model.uttt_model import uTTTConfig, uTTTModel
    from train_5090 import (PersonaMetaDataset, accumulation_autocast, collate_trimmed_sft,
                           configure_rtx5090_environment, validate_resume_optimizer_state)

    if not torch.cuda.is_available():
        parser.error("CUDA is required; this script does not report CPU timings as GPU throughput")
    config = uTTTConfig.from_dict(raw)
    tokenizer_path = training.get("tokenizer_path")
    if not tokenizer_path:
        parser.error("the SFT config must specify tokenizer_path")
    config.tokenizer_json = resolve(tokenizer_path).read_text(encoding="utf-8")
    policy = validate_sft_execution(
        training.get("sft_execution_batch_size"), training.get("sft_padding_multiple", 128),
        training.get("sft_sort_by_length", False), stage="sft", dropout=config.dropout,
        torch_compile=False, micro_batch_size=1, trim_sft_padding=training.get("trim_sft_padding"),
        max_tokens_per_group=training.get("sft_max_group_tokens", 1536),
    )
    if policy is None:
        parser.error("benchmark requires the explicit SFT execution policy")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    tokenizer = build_tokenizer(config)
    dataset = PersonaMetaDataset(
        tokenizer, seq_len=int(training.get("seq_len", 512)), data_path=str(data_path),
        pack_sequences=False, mode="sft", split="train" if training.get("validation_fraction", 0) else "all",
        validation_fraction=float(training.get("validation_fraction", 0)),
        split_seed=int(training.get("seed", 42)), validation_group_key=training.get("validation_group_key"),
    )
    if len(dataset) < 16:
        parser.error("the training split must contain at least sixteen distinct examples")
    sampler = ResumableBatchSampler(len(dataset), 1, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed), collate_fn=collate_trimmed_sft)
    plan, _ = materialize_sft_execution_step(loader, iter(loader), sampler, 16, policy,
                                              pad_token_id=tokenizer.PAD_ID)
    plan_digest = hashlib.sha256()
    target_tokens = 0
    groups = []
    for group in plan.groups:
        plan_digest.update(json.dumps(group.source_indices).encode())
        for name in ("input_ids", "labels", "attention_mask"):
            tensor = group.batch[name]
            plan_digest.update(name.encode())
            plan_digest.update(str(tuple(tensor.shape)).encode())
            plan_digest.update(tensor.numpy().tobytes())
        targets = int((group.batch["labels"][:, 1:] != -100).sum())
        target_tokens += targets
        groups.append({"shape": list(group.batch["input_ids"].shape), "source_indices": group.source_indices,
                       "loss_weight": group.loss_weight, "target_tokens": targets})

    before = checkpoint_path.stat()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    previous = uTTTConfig.from_dict(saved["config"])
    for field in ("d_model", "d_ff_fast", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "tokenizer_json"):
        if getattr(config, field) != getattr(previous, field):
            raise ValueError(f"Checkpoint differs from the configured {field}")
    checkpoint_step = int(saved.get("step", 0))
    device, amp_dtype = configure_rtx5090_environment(hardware)
    model = uTTTModel(config).to(device).train()
    model.fast_weight_pool.shared_readonly_gemm = args.shared_readonly_gemm
    model.load_dual_base_state(saved["slow_weights"], saved["init_fast_weights"], strict=True)
    optimizer = torch.optim.AdamW([
        {"name": "slow_weights", "params": model.get_slow_parameters(),
         "lr": float(training.get("outer_lr", 5e-6)), "weight_decay": float(training.get("weight_decay", 0.01))},
        {"name": "init_fast_weights", "params": model.get_meta_fast_parameters(),
         "lr": float(training.get("meta_fast_lr", 2.5e-6)), "weight_decay": 0.0},
    ], betas=(0.9, 0.95), fused=True)
    validate_resume_optimizer_state(optimizer, saved.get("optimizer_state"), checkpoint_step)
    optimizer.load_state_dict(saved["optimizer_state"])
    checkpoint_revision = saved.get("metrics", {}).get("code_revision")
    del saved
    cache_amp = bool(training.get("autocast_cache_across_accumulation", False))
    max_grad_norm = float(training.get("max_outer_grad_norm", 1.0))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def step():
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device)
        with accumulation_autocast(device.type, amp_dtype, cache_amp):
            for group in plan.groups:
                # Match the trainer: CPU selection/pinning and H2D are included,
                # while initial dataset parsing and window materialization are not.
                batch = {name: tensor.pin_memory() for name, tensor in group.batch.items()}
                indices = supervised_token_indices(batch["labels"]) if args.loss_only_head else None
                inputs = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    result = model(**inputs, update_fast_weights=False, surprise_aux_weight=0.0,
                                   return_fast_state=False, ce_reduction="sequence_mean",
                                   **({"ce_token_indices": indices} if indices is not None else {}))
                    loss = result.loss * group.loss_weight
                with (torch.amp.autocast(device.type, enabled=False) if cache_amp else nullcontext()):
                    loss.backward()
                totals[0].add_(result.loss.detach().float() * group.loss_weight)
                totals[1].add_(result.ce_loss.detach().float() * group.loss_weight)
                totals[2].add_(result.surprise_loss.detach().float() * group.loss_weight)
                del result, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, foreach=True, error_if_nonfinite=True)
        optimizer.step()
        return totals, grad_norm.detach()

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    measured = []
    started = time.perf_counter()
    for _ in range(args.steps):
        measured.append(step())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = torch.stack([torch.cat((totals, norm.reshape(1))) for totals, norm in measured]).cpu()
    nonfinite = not bool(torch.isfinite(metrics).all())
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError("The read-only checkpoint changed during the benchmark")
    try:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    def finite_values(values):
        return [value if math.isfinite(value) else None for value in values.tolist()]

    report = {
        "status": "nonfinite" if nonfinite else "ok", "nonfinite": nonfinite,
        "source": str(source), "source_revision": revision, "checkpoint_revision": checkpoint_revision,
        "checkpoint": str(checkpoint_path), "checkpoint_step": checkpoint_step,
        "checkpoint_bytes": before.st_size, "config": str(config_path), "data": str(data_path),
        "dataset_fingerprint": dataset.fingerprint, "seed": args.seed, "plan_sha256": plan_digest.hexdigest(),
        "loss_only_head": args.loss_only_head, "shared_readonly_gemm": args.shared_readonly_gemm,
        "parameters": model.num_parameters()["total"], "precision": str(amp_dtype),
        "gradient_checkpointing": config.gradient_checkpointing, "compile": False, "amp_cache": cache_amp,
        "semantic_samples_per_step": plan.total_samples, "execution_groups": groups,
        "actual_tokens_per_step": plan.actual_tokens, "physical_tokens_per_step": plan.physical_tokens,
        "target_tokens_per_step": target_tokens, "warmup_steps": args.warmup_steps, "measured_steps": args.steps,
        "seconds_per_step": elapsed / args.steps, "elapsed_seconds": elapsed,
        "tokens_per_second": args.steps * plan.actual_tokens / elapsed,
        "physical_tokens_per_second": args.steps * plan.physical_tokens / elapsed,
        "target_tokens_per_second": args.steps * target_tokens / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "measured_loss": finite_values(metrics[:, 0]), "measured_ce": finite_values(metrics[:, 1]),
        "measured_surprise": finite_values(metrics[:, 2]), "measured_grad_norm": finite_values(metrics[:, 3]),
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device), "capability": list(torch.cuda.get_device_capability(device)),
    }
    print("PERFORMANCE_RESULT " + json.dumps(report, allow_nan=False), flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if nonfinite:
        raise RuntimeError("The benchmark produced nonfinite loss or gradient norm")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PERFORMANCE_ERROR " + json.dumps({"status": "error", "error": repr(error),
              "nonfinite": "nonfinite" in str(error).lower() or "non-finite" in str(error).lower()}), flush=True)
        raise

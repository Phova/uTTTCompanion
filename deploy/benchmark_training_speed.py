"""Isolated, steady-state full-optimizer training benchmark; never saves weights."""
import argparse
import contextlib
import inspect
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="Optional immutable source checkout for a baseline")
    parser.add_argument("--config", default="configs/5090_pretrain_20261029.yaml")
    parser.add_argument("--checkpoint", help="Optional compatible weights/optimizer checkpoint")
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--amp-cache", action="store_true", help="Retain BF16 weight casts across micro-batches")
    parser.add_argument("--width", type=int, help="Override width for a representative block test")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--stage", choices=["pretrain", "meta"], default="pretrain")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--profile-microbatch", help="Export one extra micro-batch CPU/CUDA trace after timing")
    args = parser.parse_args()
    if min(args.micro_batch, args.grad_accum, args.checkpoint_every, args.steps) < 1 or args.seq_len < 2:
        parser.error("batch, accumulation, checkpoint interval and steps must be positive; sequence length must be at least 2")
    if args.warmup_steps < 0:
        parser.error("warmup steps must be nonnegative")
    if ((args.layers is not None and args.layers < 1)
            or (args.d_ff is not None and args.d_ff < 1)):
        parser.error("layers and feed-forward width must be positive")
    if args.width is not None and (args.width < 128 or args.width % 128):
        parser.error("width override must be a positive multiple of the 128-dimensional attention head")
    source = Path(args.source_root).resolve() if args.source_root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    import torch
    import yaml
    from model.uttt_model import uTTTConfig, uTTTModel
    from train_5090 import configure_rtx5090_environment, maybe_compile_inner_loop

    if not torch.cuda.is_available():
        parser.error("CUDA is required for a throughput measurement")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = source / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = uTTTConfig.from_dict(raw)
    saved = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = source / checkpoint_path
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        # The checkpoint embeds the tokenizer used by the actual training run.
        # Omitting it silently benchmarks the smaller legacy output vocabulary.
        checkpoint_config = uTTTConfig.from_dict(saved["config"])
        config.tokenizer_json = checkpoint_config.tokenizer_json
    elif raw.get("training", {}).get("tokenizer_path"):
        tokenizer_path = Path(raw["training"]["tokenizer_path"])
        if not tokenizer_path.is_absolute():
            tokenizer_path = source / tokenizer_path
        config.tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
    for field, value in (("d_model", args.width), ("n_layers", args.layers), ("d_ff_fast", args.d_ff)):
        if value is not None:
            setattr(config, field, value)
    if args.width is not None:
        config.n_heads = args.width // 128
        config.n_kv_heads = max(1, config.n_heads // 4)
        if config.n_heads % config.n_kv_heads:
            parser.error("width override does not support the benchmark's approximately 4:1 grouped-query layout")
    config.gradient_checkpointing = not args.no_checkpoint
    if args.checkpoint_every != 1 and "gradient_checkpointing_every" not in config.__dataclass_fields__:
        parser.error("Selected source does not implement selective layer checkpointing")
    config.gradient_checkpointing_every = args.checkpoint_every
    if saved is not None:
        for field in ("d_model", "d_ff_fast", "n_layers", "n_heads", "n_kv_heads", "vocab_size"):
            if getattr(config, field) != getattr(checkpoint_config, field):
                parser.error(f"Checkpoint is incompatible with benchmark {field}")
    hardware = dict(raw.get("hardware", {}), device="cuda", precision="bfloat16")
    device, amp_dtype = configure_rtx5090_environment(hardware)
    torch.manual_seed(42)
    model = uTTTModel(config).to(device).train()
    optimizer = torch.optim.AdamW([
        {"name": "slow_weights", "params": model.get_slow_parameters(), "lr": 2e-4, "weight_decay": 0.1},
        {"name": "init_fast_weights", "params": model.get_meta_fast_parameters(), "lr": 1e-4, "weight_decay": 0.0},
    ], betas=(0.9, 0.95), fused=True)
    if saved is not None:
        model.load_dual_base_state(saved["slow_weights"], saved["init_fast_weights"])
        if "optimizer_state" in saved:
            optimizer.load_state_dict(saved["optimizer_state"])
        del saved
    maybe_compile_inner_loop(
        {"torch_compile": args.compile, "compile_mode": args.compile_mode, "compile_dynamic": False},
        device, model.gradient_checkpointing, model,
    )
    supports_return_state = "return_fast_state" in inspect.signature(model.forward).parameters
    inputs = torch.randint(6, model.active_vocab_limit, (args.micro_batch, args.seq_len), device=device)
    totals = torch.zeros(3, device=device)

    def microbatch():
        with torch.autocast(device.type, dtype=amp_dtype):
            kwargs = {"return_fast_state": False} if supports_return_state else {}
            result = model(inputs, labels=inputs, update_fast_weights=args.stage == "meta", surprise_aux_weight=0.0, **kwargs)
            loss = result.loss / args.grad_accum
        with (torch.autocast(device.type, enabled=False) if args.amp_cache else contextlib.nullcontext()):
            loss.backward()
        # Match the normal trainer's device-side running loss bookkeeping.
        totals[0].add_(result.loss.detach().float() / args.grad_accum)
        totals[1].add_(result.ce_loss.detach().float() / args.grad_accum)
        totals[2].add_(result.surprise_loss.detach().float() / args.grad_accum)

    def step():
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device.type, dtype=amp_dtype) if args.amp_cache else contextlib.nullcontext()
        with context:
            for _ in range(args.grad_accum):
                microbatch()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True, error_if_nonfinite=True)
        optimizer.step()

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "status": "ok", "source": str(source), "parameters": model.num_parameters()["total"],
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum, "seq_len": args.seq_len,
        "n_heads": config.n_heads, "n_kv_heads": config.n_kv_heads,
        "precision": str(amp_dtype), "active_vocab_limit": model.active_vocab_limit,
        "return_fast_state": not supports_return_state,
        "checkpoint_every": None if args.no_checkpoint else args.checkpoint_every,
        "compile": args.compile, "amp_cache": args.amp_cache, "stage": args.stage,
        "warmup_steps": args.warmup_steps, "measured_steps": args.steps,
        "seconds_per_step": elapsed / args.steps,
        "tokens_per_second": args.steps * args.micro_batch * args.grad_accum * args.seq_len / elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }
    print("PERFORMANCE_RESULT " + json.dumps(result), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.profile_microbatch:
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profile:
            microbatch()
            torch.cuda.synchronize()
        profile.export_chrome_trace(args.profile_microbatch)
        print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PERFORMANCE_ERROR " + repr(error), flush=True)
        raise

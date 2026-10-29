"""Isolated BF16 inference benchmark with separate prefill and fixed-token decode."""
import argparse
import hashlib
import inspect
import json
import statistics
import sys
import time
from pathlib import Path


READ_KERNELS = (
    "swiglu_fast_forward", "swiglu_packed_forward", "fused_fast_q_proj", "apply_rope",
)


def source_path(value, source):
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (source / path).resolve()


def cache_storage_report(caches):
    """Count retained storage, including oversized backing storage behind a view."""
    tensors = [tensor for pair in caches for tensor in pair]
    storage_bytes = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storage_bytes[key] = storage.nbytes()
    logical = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    retained = sum(storage_bytes.values())
    return {
        "layers": len(caches),
        "first_layer_shapes": [list(tensor.shape) for tensor in caches[0]] if caches else [],
        "logical_bytes": logical,
        "storage_bytes": retained,
        "storage_to_logical_ratio": retained / logical if logical else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help="Optional old source checkout for a baseline")
    parser.add_argument("--checkpoint", help="Compatible base or training checkpoint; never modified")
    parser.add_argument("--config", help="YAML config; defaults to checkpoint config when available")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured complete prefill/decode runs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile", action="store_true", help="Compile the same pure read kernels as serving")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--no-packed-projections", action="store_true")
    parser.add_argument("--output", help="Optional JSON results file")
    args = parser.parse_args()
    if min(args.prompt_length, args.decode_tokens, args.repeats) < 1 or args.warmup < 0:
        parser.error("prompt length, decode tokens and repeats must be positive; warmup may be zero")

    source = Path(args.source_root).resolve() if args.source_root else Path(__file__).resolve().parents[1]
    if not (source / "model" / "uttt_model.py").is_file():
        parser.error(f"No uTTT model source at {source}")
    sys.path.insert(0, str(source))
    import torch
    import model.uttt_layer as kernels
    from model.uttt_model import uTTTConfig, uTTTModel

    if not torch.cuda.is_available():
        parser.error("CUDA is required for an inference throughput measurement")
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    if not torch.cuda.is_bf16_supported():
        parser.error("The selected CUDA device must support BF16")
    torch.manual_seed(42)
    saved = None
    checkpoint_path = source_path(args.checkpoint, source) if args.checkpoint else None
    if checkpoint_path:
        try:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.config:
        config_path = source_path(args.config, source)
        config = uTTTConfig.from_yaml(config_path)
        config_source = str(config_path)
    elif saved is not None and saved.get("config"):
        config = uTTTConfig.from_dict(saved["config"])
        config_source = "checkpoint"
    else:
        config_path = source / "configs" / "5090_pretrain_20261029.yaml"
        config = uTTTConfig.from_yaml(config_path)
        config_source = str(config_path)

    model = uTTTModel(config)
    if saved is not None:
        if "slow_weights" in saved and "init_fast_weights" in saved:
            model.load_dual_base_state(saved["slow_weights"], saved["init_fast_weights"], strict=True)
        elif "model_state_dict" in saved:
            model.load_state_dict(saved["model_state_dict"], strict=True)
        else:
            raise ValueError("Checkpoint has neither dual-track weights nor model_state_dict")
        del saved
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    prepare = getattr(model, "prepare_for_inference", None)
    prepared = callable(prepare) and not args.no_packed_projections
    if prepared:
        prepare()
    compiled, unavailable = [], []
    if args.compile:
        for name in READ_KERNELS:
            function = getattr(kernels, name, None)
            if function is None:
                unavailable.append(name)
                continue
            setattr(kernels, name, torch.compile(function, dynamic=True))
            compiled.append(name)

    active_limit = min(config.vocab_size, int(getattr(model, "active_vocab_limit", config.vocab_size)))
    if active_limit <= 6:
        parser.error("The model needs at least one non-special vocabulary token")
    # Generate on CPU so source checkouts consume identical token IDs and RNG state.
    generator = torch.Generator(device="cpu").manual_seed(42)
    prompt_cpu = torch.randint(6, active_limit, (1, args.prompt_length), generator=generator)
    prompt = prompt_cpu.to(device)
    fast_state = model.init_fast_state(batch_size=1, device=device, dtype=dtype, detach=True)
    forward_kwargs = {"return_fast_state": False} if "return_fast_state" in inspect.signature(model.forward).parameters else {}

    graph_holder = [None]
    graph_build_seconds = []

    def run_once(record):
        # no_grad matches actual stream_generate serving dispatch; inference_mode
        # would create different compile guards and omit tensor version counters.
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(
                input_ids=prompt, fast_state=fast_state, kv_caches=None,
                start_pos=0, update_fast_weights=False, logits_to_keep=1, **forward_kwargs,
            )
            torch.cuda.synchronize(device)
            prefill_seconds = time.perf_counter() - started
            caches = output.kv_caches
            logits = output.logits[:, -1, :]
            prefill_logits = logits[:, :16].clone() if record else None
            prefill_cache = cache_storage_report(caches) if record else None
            next_token = logits.argmax(dim=-1)
            del output
            graph = None
            if args.cuda_graph:
                from engine.inference import CUDAGraphDecoder
                graph = graph_holder[0]
                if graph is None or not graph.matches(model, fast_state):
                    built_at = time.perf_counter()
                    graph = CUDAGraphDecoder(model, fast_state, caches, args.prompt_length)
                    torch.cuda.synchronize(device)
                    graph_build_seconds.append(time.perf_counter() - built_at)
                    graph_holder[0] = graph
                else:
                    graph.reset(caches, args.prompt_length)
            greedy_ids = []
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            # Exactly N one-token forwards: EOS and role tokens never stop timing.
            # Keep greedy sampling on device; CPU copies/text output are excluded.
            for index in range(args.decode_tokens):
                greedy_ids.append(next_token)
                if graph is not None:
                    logits = graph.step(next_token.view(1, 1))
                else:
                    output = model(
                        input_ids=next_token.view(1, 1), fast_state=fast_state,
                        kv_caches=caches, start_pos=args.prompt_length + index,
                        update_fast_weights=False, logits_to_keep=1, **forward_kwargs,
                    )
                    caches = output.kv_caches
                    logits = output.logits[:, -1, :]
                    del output
                next_token = logits.argmax(dim=-1)
            torch.cuda.synchronize(device)
            decode_seconds = time.perf_counter() - started
            if not record:
                return None
            if graph is not None:
                caches = graph.export_kv_caches()
            return {
                "prefill_seconds": prefill_seconds,
                "prefill_tokens_per_second": args.prompt_length / prefill_seconds,
                "decode_seconds": decode_seconds,
                "decode_forward_tokens": args.decode_tokens,
                "decode_tokens_per_second": args.decode_tokens / decode_seconds,
                "prefill_cache": prefill_cache,
                "decode_cache": cache_storage_report(caches),
                "greedy_ids": torch.cat(greedy_ids).cpu().tolist(),
                "next_greedy_id": int(next_token.item()),
                "prefill_logits_head": prefill_logits.float().cpu().flatten().tolist(),
                "decode_logits_head": logits[:, :16].float().cpu().flatten().tolist(),
            }

    for _ in range(args.warmup):
        run_once(record=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = [run_once(record=True) for _ in range(args.repeats)]
    prefill_times = [sample["prefill_seconds"] for sample in samples]
    decode_times = [sample["decode_seconds"] for sample in samples]
    result = {
        "status": "ok", "source": str(source), "model_source": inspect.getfile(uTTTModel),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "config_source": config_source, "parameters": model.num_parameters()["total"],
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "dtype": str(dtype), "seed": 42, "prompt_length": args.prompt_length,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_cpu.tolist()).encode("utf-8")).hexdigest(),
        "decode_tokens": args.decode_tokens, "warmup": args.warmup, "repeats": args.repeats,
        "cuda_graph": args.cuda_graph, "graph_build_seconds": graph_build_seconds,
        "compile": args.compile, "compiled_kernels": compiled, "unavailable_kernels": unavailable,
        "prepared_inference_projections": prepared,
        "sampling": "device_greedy_no_eos_stop",
        "timing_scope": "model forwards and device greedy; excludes CPU token copies and text decoding",
        "median_prefill_seconds": statistics.median(prefill_times),
        "median_prefill_tokens_per_second": args.prompt_length / statistics.median(prefill_times),
        "median_decode_seconds": statistics.median(decode_times),
        "median_decode_tokens_per_second": args.decode_tokens / statistics.median(decode_times),
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "repeat_greedy_ids_match": all(sample["greedy_ids"] == samples[0]["greedy_ids"] for sample in samples),
        "samples": samples,
    }
    print("PERFORMANCE_RESULT " + json.dumps(result), flush=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PERFORMANCE_ERROR " + repr(error), flush=True)
        raise

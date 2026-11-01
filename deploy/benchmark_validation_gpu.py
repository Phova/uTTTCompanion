"""Alternate full-checkpoint CUDA validation paths; never load Adam onto the GPU.

Run only while production training is stopped. Both functions share one FP32,
trainable model and the same packed validation windows. Timing covers the real
validation function, including CPU reads and transfers, with CUDA synchronization.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


SEQUENCE_LENGTH = 1024
MINIMUM_FREE_BYTES = 12 * 1024**3


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-trainer", type=Path, required=True,
                        help="train_5090.py extracted from baseline a0643fd")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3, help="Measured pairs; at least three")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path,
                        help="TF32 configuration; defaults to configs/5090_pretrain_20261029.yaml")
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 3:
        parser.error("samples must be positive and repeats must be at least three")
    for name in ("checkpoint", "manifest", "baseline_trainer"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f"{name.replace('_', '-')} is not a file: {path}")
        setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(f"output already exists; use a new report path: {args.output}")
    return args


def file_identity(path):
    info = path.stat()
    return {"device": info.st_dev, "inode": info.st_ino,
            "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("uttt_validation_trainer_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "evaluate_language_model", None)):
        raise ValueError("Baseline trainer has no evaluate_language_model function")
    return module


def available_memory(torch, device):
    from deploy.gpu_memory import query_available_gpu_memory
    runtime_free, _ = torch.cuda.mem_get_info(device)
    return query_available_gpu_memory(
        int(runtime_free), device_uuid=getattr(torch.cuda.get_device_properties(device), "uuid", None),
        device_index=device.index, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def main():
    args = arguments()
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    current_trainer_path = source / "train_5090.py"
    source_hashes = {"baseline": hashlib.sha256(args.baseline_trainer.read_bytes()).hexdigest(),
                     "optimized": hashlib.sha256(current_trainer_path.read_bytes()).hexdigest()}
    if source_hashes["baseline"] == source_hashes["optimized"]:
        raise ValueError("Baseline and optimized trainer are identical; refusing a meaningless comparison")

    import torch
    import yaml
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    device = torch.device("cuda", torch.cuda.current_device())
    memory = available_memory(torch, device)
    if memory["free_bytes"] < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"Only {memory['free_bytes'] / 1024**3:.2f} GiB globally free; "
                           "at least 12 GiB is required, refusing concurrent training benchmark")

    # Import the old trainer under its own name, but deliberately share the new
    # model implementation between both functions to isolate trainer changes.
    baseline = load_baseline(args.baseline_trainer)
    import train_5090 as optimized
    from engine.corpus import TokenizedCorpusDataset
    from model.tokenizer import build_tokenizer
    from model.uttt_model import uTTTConfig, uTTTModel

    hardware_path = (args.hardware_config.expanduser().resolve() if args.hardware_config
                     else source / "configs" / "5090_pretrain_20261029.yaml")
    raw_config = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware = dict(raw_config.get("hardware", {}), device="cuda", precision="bfloat16")
    # This applies the same TF32 policy as training. No compiler or inference
    # preparation is enabled: FP32 trainable parameters exercise the AMP cache.
    configured_device, dtype = optimized.configure_rtx5090_environment(hardware)
    if configured_device.type != "cuda" or dtype != torch.bfloat16:
        raise RuntimeError("Hardware configuration did not select CUDA BF16 activations")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    checkpoint_identity = file_identity(args.checkpoint)
    manifest_identity = file_identity(args.manifest)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    # mmap creates descriptors for full-checkpoint tensors; dropping these keys
    # before access avoids materializing Adam moment pages or copying them to GPU.
    saved.pop("optimizer_state", None)
    saved.pop("training_state", None)
    config = uTTTConfig.from_dict(saved["config"])
    if not config.tokenizer_json:
        raise ValueError("Benchmark requires the real BPE tokenizer embedded in the checkpoint")
    dataset = TokenizedCorpusDataset(args.manifest, seq_len=SEQUENCE_LENGTH, split="validation")
    tokenizer_hash = hashlib.sha256(config.tokenizer_json.encode("utf-8")).hexdigest()
    if dataset.manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Checkpoint BPE and validation corpus tokenizer fingerprints differ")
    tokenizer = build_tokenizer(config)
    if args.samples > len(dataset):
        raise ValueError("Requested more validation samples than full 1024-token windows")
    corpus_identity = file_identity(dataset.path)
    indices = [index * (len(dataset) - 1) // max(1, args.samples - 1) for index in range(args.samples)]

    class ObservedDataset:
        """Retain actual mmap/copy costs while verifying the windows read."""
        def __init__(self):
            self.observed = []

        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            self.observed.append(index)
            return dataset[index]

    observed_dataset = ObservedDataset()
    # Repeat the guard immediately before the first large CUDA allocation. There
    # is no cross-process reservation, so production must stay stopped throughout.
    memory_before_model = available_memory(torch, device)
    if memory_before_model["free_bytes"] < MINIMUM_FREE_BYTES:
        raise RuntimeError("Global free VRAM fell below 12 GiB while reading checkpoint metadata")
    with torch.device(device):
        model = uTTTModel(config)
    model.load_dual_base_state(saved["slow_weights"], saved["init_fast_weights"], strict=True)
    checkpoint_step = int(saved.get("step", 0))
    del saved
    gc.collect()
    model.train()
    if any(parameter.dtype != torch.float32 or not parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Validation must use FP32 trainable parameters, without freezing or inference packing")
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    functions = {"baseline": baseline.evaluate_language_model, "optimized": optimized.evaluate_language_model}
    runs, warmups = [], []

    def run(label, round_index, warmup=False):
        model.train()
        observed_dataset.observed.clear()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        started = time.perf_counter()
        metrics = functions[label](model, observed_dataset, device, dtype,
                                   update_fast_weights=False, max_batches=args.samples)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if observed_dataset.observed != indices:
            raise AssertionError(f"{label} did not read the required identical validation windows")
        if not model.training or any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("Validation changed training mode or created parameter gradients")
        if not all(math.isfinite(float(metrics[key])) for key in ("validation_ce", "validation_frozen_ce")):
            raise AssertionError("Nonfinite validation CE")
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        item = {"implementation": label, "round": round_index, "warmup": warmup,
                "seconds": elapsed, **metrics,
                "before_allocated_bytes": before_allocated, "before_reserved_bytes": before_reserved,
                "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                "peak_allocated_increment_bytes": peak_allocated - before_allocated,
                "peak_reserved_increment_bytes": peak_reserved - before_reserved,
                "after_allocated_bytes": torch.cuda.memory_allocated(device),
                "after_reserved_bytes": torch.cuda.memory_reserved(device)}
        print("VALIDATION_GPU_RUN " + json.dumps(item), flush=True)
        return item

    for label in functions:
        warmups.append(run(label, -1, warmup=True))
    for round_index in range(args.repeats):
        order = ("baseline", "optimized") if round_index % 2 == 0 else ("optimized", "baseline")
        for label in order:
            runs.append(run(label, round_index))

    reference = {key: warmups[0][key] for key in ("validation_ce", "validation_frozen_ce", "validation_targets")}
    parity = all(all(item[key] == value for key, value in reference.items()) for item in warmups + runs)
    if file_identity(args.checkpoint) != checkpoint_identity:
        raise RuntimeError("Checkpoint changed during benchmarking; refusing to attribute this measurement")
    if file_identity(args.manifest) != manifest_identity or file_identity(dataset.path) != corpus_identity:
        raise RuntimeError("Validation corpus changed during benchmarking")
    summary = {}
    for label in functions:
        items = [item for item in runs if item["implementation"] == label]
        summary[label] = {"median_seconds": statistics.median(item["seconds"] for item in items),
                          "min_seconds": min(item["seconds"] for item in items),
                          "max_seconds": max(item["seconds"] for item in items),
                          "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in items),
                          "peak_reserved_bytes": max(item["peak_reserved_bytes"] for item in items)}
    old_time, new_time = summary["baseline"]["median_seconds"], summary["optimized"]["median_seconds"]
    report = {
        "status": "ok" if parity else "failed_parity", "validation_metrics_exactly_equal": parity,
        "scope": "one full FP32 trainable model; BF16 no-grad CE validation only; no Adam or gradients allocated",
        "timing": "synchronized wall time including validation mmap reads, transfers and Python; allocator cache cleared before each run",
        "expected_baseline_commit": "a0643fd", "expected_optimized_commit": "33a0da6",
        "shared_model_source": str(source), "trainer_sha256": source_hashes,
        "checkpoint": str(args.checkpoint), "checkpoint_identity": checkpoint_identity, "checkpoint_step": checkpoint_step,
        "manifest": str(args.manifest), "validation_corpus_fingerprint": dataset.fingerprint,
        "tokenizer_sha256": tokenizer_hash, "tokenizer_vocab_size": tokenizer.vocab_size,
        "model_parameters": model.num_parameters()["total"], "model_parameter_bytes": parameter_bytes,
        "model_dtype": "torch.float32", "activation_dtype": str(dtype), "parameters_require_grad": True,
        "samples": args.samples, "sequence_length": SEQUENCE_LENGTH, "window_indices": indices,
        "repeats_per_implementation": args.repeats, "warmup_passes_per_implementation": 1,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "hardware_config": str(hardware_path), "available_memory_initial": memory,
        "available_memory_before_model": memory_before_model, "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "summary": summary, "median_validation_speedup": old_time / new_time,
        "median_validation_time_reduction_fraction": 1 - new_time / old_time,
        "caveat": "This is isolated validation, with no live Adam/gradients; it does not measure steady-state training throughput or its peak memory.",
        "warmups": warmups, "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print("VALIDATION_GPU_RESULT " + json.dumps({key: report[key] for key in
          ("status", "validation_metrics_exactly_equal", "summary", "median_validation_speedup")}), flush=True)
    if not parity:
        raise AssertionError(f"Baseline/optimized validation metrics differ; diagnostic report saved to {args.output}")


if __name__ == "__main__":
    main()

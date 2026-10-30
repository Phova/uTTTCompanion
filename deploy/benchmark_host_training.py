"""Compare deterministic host-side training overhead without using GPU memory.

The sampler workload matches the prepared corpus size. Checkpoint timings use a
small CPU model and are not a full-size training throughput measurement.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.corpus import ResumableBatchSampler
from model.uttt_model import uTTTConfig, uTTTModel
from train_5090 import _cpu_copy_nested, _wait_for_active_save, export_base_checkpoint


def legacy_iterator(size, batch_size, seed, offset=0):
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    limit = size - size % batch_size if size >= batch_size else size
    for start in range(offset, limit, batch_size):
        yield order[start:min(start + batch_size, limit)]


def sampler_measure(size, batches, batch_size):
    sampler = ResumableBatchSampler(size, batch_size, 42)
    result = {}
    for name, factory in (("legacy", lambda: legacy_iterator(size, batch_size, 42)),
                          ("optimized", lambda: iter(sampler))):
        gc.collect()
        start = time.perf_counter()
        iterator = factory()
        first = next(iterator)
        build = time.perf_counter() - start
        start = time.perf_counter()
        checksum = sum(first)
        consumed = 1
        for batch in iterator:
            checksum += sum(batch)
            consumed += 1
            if consumed >= batches:
                break
        consumption = time.perf_counter() - start
        start = time.perf_counter()
        retry = factory()
        retry_first = next(retry)
        retry_time = time.perf_counter() - start
        result[name] = {"build_to_first_batch_seconds": build,
                        "retry_to_first_batch_seconds": retry_time,
                        "consume_batches_per_second": max(0, consumed - 1) / max(consumption, 1e-9),
                        "consumed_batches": consumed, "checksum": checksum,
                        "retry_same_first_batch": first == retry_first}
        del iterator, retry
    assert result["legacy"]["checksum"] == result["optimized"]["checksum"]
    # Measure boxed-object memory in a separate run; tracing itself changes wall
    # time substantially and therefore must not influence the timings above.
    for name, factory in (("legacy", lambda: legacy_iterator(size, batch_size, 42)),
                          ("optimized", lambda: iter(ResumableBatchSampler(size, batch_size, 42)))):
        gc.collect()
        tracemalloc.start()
        try:
            iterator = factory()
            next(iterator)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        result[name]["peak_python_index_bytes"] = peak
        del iterator
    result["optimized"]["retained_index_buffer_bytes"] = sampler._permutation.nbytes
    return result


def write_payload(payload, path):
    with path.open("wb") as destination:
        torch.save(payload, destination)
        destination.flush()
        os.fsync(destination.fileno())


def checkpoint_measure(vocab_size, width):
    config = uTTTConfig(vocab_size=vocab_size, d_model=width, n_layers=2, n_heads=4,
                        n_kv_heads=2, d_ff_fast=width * 2, gradient_checkpointing=False)
    torch.manual_seed(24)
    model = uTTTModel(config).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = model.state_dict()
    legacy_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state.values()) * 2
    unique = {(tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype): tensor for tensor in state.values()}
    optimized_bytes = sum(tensor.numel() * tensor.element_size() for tensor in unique.values())
    with tempfile.TemporaryDirectory(prefix="uttt_host_benchmark_") as temporary:
        root = Path(temporary)
        start = time.perf_counter()
        full = {"slow_weights": model.slow_state_dict(), "init_fast_weights": model.init_fast_state_dict(),
                "optimizer_state": _cpu_copy_nested(optimizer.state_dict())}
        first_snapshot_time = time.perf_counter() - start
        write_payload(full, root / "legacy.pt")
        del full
        best = {"slow_weights": model.slow_state_dict(), "init_fast_weights": model.init_fast_state_dict()}
        write_payload(best, root / "legacy.best.pt")
        del best
        legacy_total = time.perf_counter() - start
        gc.collect()
        start = time.perf_counter()
        export_base_checkpoint(model, optimizer, config, root / "optimized.pt", 1, {},
                               async_save=True, best_save_path=root / "optimized.best.pt")
        optimized_return = time.perf_counter() - start
        _wait_for_active_save()
        optimized_total = time.perf_counter() - start
        legacy_size = sum((root / name).stat().st_size for name in ("legacy.pt", "legacy.best.pt"))
        optimized_size = sum((root / name).stat().st_size for name in ("optimized.pt", "optimized.best.pt"))
        old = torch.load(root / "legacy.pt", map_location="cpu", weights_only=False)
        new = torch.load(root / "optimized.pt", map_location="cpu", weights_only=False)
        for partition in ("slow_weights", "init_fast_weights"):
            for name in old[partition]:
                torch.testing.assert_close(old[partition][name], new[partition][name], atol=0, rtol=0)
        assert old["optimizer_state"].keys() == new["optimizer_state"].keys()
    return {"model_parameters": model.num_parameters()["total"], "device": "cpu",
            "legacy_model_cpu_copy_bytes": legacy_bytes, "optimized_model_cpu_copy_bytes": optimized_bytes,
            "legacy_primary_snapshot_seconds": first_snapshot_time,
            "legacy_total_seconds": legacy_total, "optimized_return_seconds": optimized_return,
            "optimized_total_seconds": optimized_total,
            "legacy_total_checkpoint_bytes": legacy_size, "optimized_total_checkpoint_bytes": optimized_size,
            "loaded_weights_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler-size", type=int, default=3_504_133)
    parser.add_argument("--batches", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-vocab", type=int, default=32_000)
    parser.add_argument("--checkpoint-width", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sampler_size, args.batches, args.batch_size, args.checkpoint_vocab, args.checkpoint_width) < 1:
        parser.error("All workload sizes must be positive")
    torch.set_num_threads(2)
    result = {"torch": torch.__version__, "inner_loop_protocol": uTTTModel.INNER_LOOP_PROTOCOL,
              "sampler": sampler_measure(args.sampler_size, args.batches, args.batch_size),
              "checkpoint": checkpoint_measure(args.checkpoint_vocab, args.checkpoint_width),
              "scope": "CPU host overhead only; no large-model GPU throughput claim"}
    content = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()

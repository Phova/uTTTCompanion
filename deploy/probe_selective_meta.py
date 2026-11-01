"""Isolated two-step exact-meta capacity probe; reads a checkpoint and saves no weights.

Run in its own process with an external timeout, only after stopping GPU training.
The synthetic BPE token IDs establish capacity and gradient flow, not model quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from deploy.gpu_memory import GpuMemoryError, query_available_gpu_memory
from model.tokenizer import build_tokenizer
from model.uttt_model import uTTTConfig, uTTTModel


GIB = 1024 ** 3
MIN_FREE_BYTES = 28 * GIB
MAX_ALLOCATOR_BYTES = 28 * GIB
DRIVER_RESERVE_BYTES = int(1.5 * GIB)
WRITER_MARKERS = (".w_k.", ".w_v.", ".w_eta.", ".w_alpha.")


def load_probe_checkpoint(path):
    """Expose only mmap-backed model data; never restore checkpoint Adam state."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or not all(
            isinstance(payload.get(key), dict) for key in ("config", "slow_weights", "init_fast_weights")):
        raise ValueError("Expected a dual-track uTTT checkpoint with config and both weight mappings")
    return {key: payload[key] for key in ("config", "slow_weights", "init_fast_weights")} | {
        "step": payload.get("step"),
    }


def scoped_parameters(model, scope):
    if scope not in ("memory", "all"):
        raise ValueError("scope must be memory or all")
    selected = []
    for name, parameter in model.named_parameters():
        enabled = scope == "all" or name.startswith("fast_weight_pool.") or ".uttt." in name
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append((name, parameter))
    if not selected:
        raise ValueError("Parameter scope is empty")
    return selected


def guard_cuda_allocator():
    """Check global and runtime free space before any model allocation on CUDA."""
    if not torch.cuda.is_available():
        raise GpuMemoryError("CUDA GPU required")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    runtime_free, _ = torch.cuda.mem_get_info(device)
    memory = query_available_gpu_memory(
        runtime_free, device_uuid=getattr(properties, "uuid", None), device_index=device,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if memory["global_free_bytes"] < MIN_FREE_BYTES or memory["free_bytes"] < MIN_FREE_BYTES:
        raise GpuMemoryError("Probe requires at least 28 GiB free in both global and CUDA runtime measurements; stop other GPU workloads")
    budget_bytes = min(MAX_ALLOCATOR_BYTES, memory["free_bytes"] - DRIVER_RESERVE_BYTES)
    # The allocator fraction is relative to CUDA's total device capacity, not
    # its current free estimate (which can be misleading under WSL/WDDM).
    total_bytes = int(properties.total_memory)
    if total_bytes <= 0 or budget_bytes >= total_bytes:
        raise GpuMemoryError("Cannot establish a bounded allocator budget for the selected GPU")
    torch.cuda.set_per_process_memory_fraction(budget_bytes / total_bytes, device=device)
    return device, memory, budget_bytes


def writer_gradients(named_parameters):
    writers = [(name, parameter.grad) for name, parameter in named_parameters
               if any(marker in name for marker in WRITER_MARKERS)]
    present = [(name, gradient) for name, gradient in writers if gradient is not None]
    norms = (torch.stack([gradient.detach().norm() for _, gradient in present]).cpu().tolist()
             if present else [])
    missing = [name for name, gradient in writers if gradient is None]
    zero = [name for (name, _), norm in zip(present, norms) if norm == 0.0]
    finite = all(float("-inf") < norm < float("inf") for norm in norms)
    return {
        "writer_parameter_count": len(writers), "writer_grad_count": len(present),
        "writer_nonzero_grad_count": sum(norm > 0.0 for norm in norms),
        "writer_missing_grad_names": missing, "writer_zero_grad_names": zero,
        "writer_grad_norm_min": min(norms) if norms else None,
        "writer_grad_norm_max": max(norms) if norms else None,
        "writer_gradients_finite": finite,
        "all_writers_have_nonzero_ce_grad": bool(writers) and not missing and not zero and finite,
    }


def probe(checkpoint_path, scope="memory", seq_len=1024, chunk_size=128, steps=2):
    if chunk_size < 1 or seq_len <= chunk_size + 1:
        raise ValueError("Meta probe requires seq-len >= chunk-size + 2")
    if steps < 2:
        raise ValueError("Meta probe requires at least two Adam steps")
    checkpoint = load_probe_checkpoint(checkpoint_path)
    config = uTTTConfig.from_dict(checkpoint["config"])
    if not config.tokenizer_json:
        raise ValueError("Probe requires the actual BPE tokenizer embedded in the checkpoint")
    if seq_len > config.max_seq_len:
        raise ValueError("seq-len exceeds the checkpoint context limit")
    config.chunk_size = chunk_size
    config.gradient_checkpointing = True
    config.gradient_checkpointing_every = 1
    tokenizer = build_tokenizer(config)
    if tokenizer.vocab_size <= 6:
        raise ValueError("BPE vocabulary has no ordinary tokens")
    # Count without allocating host or device model storage. All parameter
    # counts are deduplicated, including tied token embeddings / output weights.
    with torch.device("meta"):
        skeleton = uTTTModel(config)
    selected = scoped_parameters(skeleton, scope)
    total_parameters = sum(parameter.numel() for parameter in skeleton.parameters())
    trainable_parameters = sum(parameter.numel() for _, parameter in selected)
    del selected, skeleton
    device, memory, budget_bytes = guard_cuda_allocator()
    static_bytes = total_parameters * 4 + trainable_parameters * 12
    report = {
        "status": "running", "phase": "static_budget_check", "checkpoint": str(checkpoint_path), "checkpoint_step": checkpoint["step"],
        "scope": scope, "parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "static_fp32_weights_grads_adam_bytes": static_bytes,
        "available_memory": memory, "allocator_limit_bytes": budget_bytes,
        "driver_reserve_bytes": DRIVER_RESERVE_BYTES, "maximum_allocator_bytes": MAX_ALLOCATOR_BYTES,
        "seq_len": seq_len, "chunk_size": chunk_size, "batch_size": 1, "requested_steps": steps,
        "inner_loop_protocol": uTTTModel.INNER_LOOP_PROTOCOL,
        "gradient_checkpointing": True, "gradient_checkpointing_every": 1,
        "parameter_dtype": "float32", "gradient_dtype": "float32", "adam_moment_dtype": "float32",
        "activation_dtype": "bfloat16", "dynamic_state_precision": "float32",
        "update_fast_weights": True, "surprise_aux_weight": 0.0, "return_fast_state": False,
        "tokenizer_sha256": hashlib.sha256(config.tokenizer_json.encode()).hexdigest(),
        "active_vocab_limit": tokenizer.vocab_size, "synthetic_bpe_tokens": True, "seed": 42,
        "checkpoint_optimizer_restored": False, "weights_saved": False, "torch": torch.__version__,
        "step_timing_includes_writer_diagnostics": True,
        "finite": None, "steps": [],
    }
    if static_bytes > budget_bytes:
        return {**report, "status": "static_budget_exceeded"}
    optimizer = None
    torch.cuda.reset_peak_memory_stats(device)
    try:
        report["phase"] = "model_initialization"
        torch.manual_seed(42)
        # Construct real buffers on CPU before copying to CUDA. Meta.to_empty would
        # leave nonpersistent RoPE tables uninitialized even after strict weight load.
        model = uTTTModel(config).float()
        model.load_dual_base_state(checkpoint["slow_weights"], checkpoint["init_fast_weights"], strict=True)
        del checkpoint
        model = model.to(device=torch.device("cuda", device), dtype=torch.float32).train()
        selected = scoped_parameters(model, scope)
        parameters = [parameter for _, parameter in selected]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.1, fused=True)
        ids = torch.randint(6, tokenizer.vocab_size, (1, seq_len), device=torch.device("cuda", device))
        torch.cuda.synchronize(device)
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            report["phase"] = "forward"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(ids, labels=ids, update_fast_weights=True, surprise_aux_weight=0.0,
                               return_fast_state=False)
            report["phase"] = "backward"
            output.loss.backward()
            report["phase"] = "gradient_diagnostics"
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            diagnostics = writer_gradients(selected)
            ce, loss, norm = float(output.ce_loss.detach()), float(output.loss.detach()), float(grad_norm)
            if not all(float("-inf") < value < float("inf") for value in (ce, loss, norm)):
                raise ValueError("Non-finite CE loss or outer gradient")
            if any(parameter.grad is not None and parameter.grad.dtype != torch.float32 for parameter in parameters):
                raise ValueError("Probe gradient precision is not FP32")
            report["phase"] = "optimizer_step"
            optimizer.step()
            torch.cuda.synchronize(device)
            report["steps"].append({
                "step": step, "seconds": time.perf_counter() - start, "ce": ce, "loss": loss,
                "grad_norm": norm, "finite": True, **diagnostics,
            })
            print("SELECTIVE_META_STEP " + json.dumps(report["steps"][-1]), flush=True)
            del output
        if any(state[name].dtype != torch.float32 for state in optimizer.state.values()
               for name in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Fresh Adam moment precision is not FP32")
        report["finite"] = all(step["finite"] for step in report["steps"])
        report["all_writers_have_nonzero_ce_grad"] = all(
            step["all_writers_have_nonzero_ce_grad"] for step in report["steps"])
        report["status"] = "ok" if report["all_writers_have_nonzero_ce_grad"] else "writer_gradient_incomplete"
        report["phase"] = "complete"
    except torch.OutOfMemoryError as error:
        report.update(status="oom", error=str(error))
    except (RuntimeError, ValueError) as error:
        report.update(status="failed", error=str(error), error_type=type(error).__name__)
    finally:
        if optimizer is not None:
            report.update(
                optimizer_state_parameters=len(optimizer.state),
                optimizer_complete_state_parameters=sum(all(name in state for name in ("exp_avg", "exp_avg_sq"))
                                                        for state in optimizer.state.values()),
                optimizer_state_elements=sum(parameter.numel() for parameter in optimizer.state),
                optimizer_moment_elements=sum(value.numel() for state in optimizer.state.values()
                                              for name, value in state.items() if name in ("exp_avg", "exp_avg_sq")),
            )
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scope", choices=("memory", "all"), default="memory")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    try:
        result = probe(args.checkpoint.resolve(), args.scope, args.seq_len, args.chunk_size, args.steps)
    except (GpuMemoryError, ValueError) as error:
        result = {"status": "refused", "error": str(error), "weights_saved": False}
    print("SELECTIVE_META_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

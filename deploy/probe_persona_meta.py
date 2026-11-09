"""Bounded writer-only capacity/gradient probe on real support/query episodes.

No checkpoint or persona weights are saved. This establishes resource use and
the CE gradient path, not useful memory or answer quality. Run in an isolated
process with an external timeout and an idle GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROTOCOL = "persona-episode-capacity-v1"
WRITER_PARTS = ("w_k", "w_v", "w_eta", "w_alpha")
STATE_NAMES = ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down", "ln_weight", "ln_bias")


def is_writer(name):
    return any(f".uttt.{part}." in name for part in WRITER_PARTS)


def storage_ledger(d_model, d_ff, n_layers):
    """Exact logical tensor bytes; serialization/allocator overhead is additional."""
    if any(type(value) is not int or value < 1 for value in (d_model, d_ff, n_layers)):
        raise ValueError("Model dimensions must be positive integers")
    matrices = 3 * d_model * d_ff
    norm = 2 * d_model
    writers = n_layers * (2 * d_model * d_model + 2 * (d_model + 1))
    return {"writer_parameters": writers, "writer_parameter_tensors": n_layers * 6,
            "persona_W_fp32_bytes": matrices * 4, "persona_S_fp32_bytes": matrices * 4,
            "persona_LN_fp32_bytes": norm * 4, "persona_master_fp32_bytes": (2 * matrices + norm) * 4,
            "persona_bf16_read_cache_bytes": matrices * 2,
            "persona_hot_master_plus_read_cache_bytes": (2 * matrices + norm) * 4 + matrices * 2,
            "persona_hardlink_alias_logical_bytes": (2 * matrices + norm) * 8,
            "persona_copy_alias_physical_tensor_bytes": (2 * matrices + norm) * 8,
            "persona_atomic_hardlink_tensor_peak_bytes": (2 * matrices + norm) * 8,
            "scope": "One B1 persona. Alias hardlinks share one physical payload; snapshots add another payload per tag. KV/temporary tensors and serialization overhead excluded."}


def choose_episodes(dataset, *, max_total_tokens=384):
    """Select the worst actual write count/length, never manufacture random IDs."""
    if type(max_total_tokens) is not int or max_total_tokens < 10:
        raise ValueError("max_total_tokens must be an integer >= 10")
    if not len(dataset):
        raise ValueError("Probe needs nonempty real training episodes")
    shapes = []
    for index in range(len(dataset)):
        row = dataset[index]
        writes, query = row["write_input_ids"], row["input_ids"]
        if writes.ndim != 2 or query.ndim != 1 or writes.shape[0] < 1:
            raise ValueError("Unexpected unbatched episode dimensions")
        physical = writes.numel() + query.numel()
        if physical > max_total_tokens:
            raise ValueError("A real episode exceeds the padded total-token probe budget; do not crop or skip it")
        write_lengths = [int(value) for value in row["write_attention_mask"].sum(dim=-1).tolist()]
        query_length = int(row["attention_mask"].sum())
        shapes.append({"index": index, "write_rounds": writes.shape[0], "write_tokens": write_lengths,
                       "query_tokens": query_length, "padded_total_tokens": physical,
                       "actual_total_tokens": sum(write_lengths) + query_length})
    shapes.sort(key=lambda row: (-row["write_rounds"], -row["actual_total_tokens"],
                                -row["query_tokens"], row["index"]))
    return shapes[:min(2, len(shapes))]


def parameter_digest(named_parameters):
    """Hash one parameter at a time on CPU; never clone the frozen model on GPU."""
    digest = hashlib.sha256()
    for name, value in named_parameters:
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        cpu = value.detach().to(device="cpu", copy=True).contiguous()
        digest.update(memoryview(cpu.numpy()).cast("B"))
        del cpu
    return digest.hexdigest()


def _batched(dataset, index, device):
    from torch.utils.data import default_collate
    return {key: value.to(device) for key, value in default_collate([dataset[index]]).items()}


def run_episode_steps(model, dataset, *, device, amp_dtype, steps=2, learning_rate=1e-5,
                      weight_decay=0.0, max_total_tokens=384, memory_reporter=None, report=None):
    """The same Dataset/episode_forward/scope API as the real trainer; CPU-testable."""
    import torch
    from train_5090 import apply_training_parameter_scope, episode_forward

    if type(steps) is not int or steps < 2:
        raise ValueError("At least two complete Adam steps are required")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if model.config.meta_checkpoint_reentrant:
        raise ValueError("Real episode probing requires non-reentrant checkpointing")
    device = torch.device(device)
    scope = apply_training_parameter_scope(model, "meta", "writer")
    selected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if not selected or any(not is_writer(name) for name, _ in selected):
        raise ValueError("Probe optimizer must contain only writer parameters")
    if any(is_writer(name) for name, _ in frozen):
        raise ValueError("A writer parameter was unexpectedly frozen")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Probe model parameters must remain FP32")
    shapes = choose_episodes(dataset, max_total_tokens=max_total_tokens)
    result = report if report is not None else {}
    result.update(parameter_scope="writer", **scope, selected_episodes=shapes,
                  requested_steps=steps, steps=[], weights_saved=False,
                  checkpoint_optimizer_restored=False, step_timing_includes_writer_diagnostics=True)
    result["frozen_parameters_sha256_before"] = parameter_digest(frozen)
    optimizer = torch.optim.AdamW([parameter for _, parameter in selected], lr=learning_rate,
                                   weight_decay=weight_decay, betas=(0.9, 0.95), fused=device.type == "cuda")
    baseline = []
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        for shape in shapes:
            output = episode_forward(model, _batched(dataset, shape["index"], device), apply_writes=False)
            baseline.append(output.logits.detach().cpu())
            del output
    model.train()
    original_forward = model.forward
    checked_queries = 0
    expected_query = None

    def audited_forward(*args, **kwargs):
        nonlocal checked_queries
        if kwargs.get("kv_caches") is not None or kwargs.get("start_pos", 0) != 0:
            raise ValueError("Support/query calls must start with no KV cache and reset positions")
        if kwargs.get("labels") is not None:
            ids = kwargs.get("input_ids", args[0] if args else None)
            if kwargs.get("update_fast_weights") is not False or not torch.equal(ids, expected_query):
                raise ValueError("Query received support text or performed new writes")
            state = kwargs.get("fast_state")
            if state is None or state._chunk_states is not None or state._init_state is not None:
                raise ValueError("Query must receive only the newly written W/S/LN state")
            if any(getattr(state, name).dtype != torch.float32 for name in STATE_NAMES):
                raise ValueError("Written persona masters must remain FP32")
            if any(not getattr(state, name).requires_grad for name in STATE_NAMES[:6]):
                raise ValueError("Support-to-query gradient chain was detached")
            checked_queries += 1
        return original_forward(*args, **kwargs)

    try:
        model.forward = audited_forward
        for step in range(1, steps + 1):
            batch = _batched(dataset, shapes[(step - 1) % len(shapes)]["index"], device)
            expected_query = batch["input_ids"]
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                output = episode_forward(model, batch, apply_writes=True)
            if output.loss is None or not torch.isfinite(output.loss):
                raise ValueError("Non-finite or missing query CE")
            output.loss.backward()
            missing, zero, nonfinite, norms = [], [], [], []
            for name, parameter in selected:
                if parameter.grad is None:
                    missing.append(name)
                    continue
                if parameter.grad.dtype != torch.float32:
                    raise ValueError("Writer gradients must remain FP32")
                norm = float(parameter.grad.detach().norm())
                norms.append(norm)
                if not math.isfinite(norm):
                    nonfinite.append(name)
                elif norm == 0:
                    zero.append(name)
            if missing or zero or nonfinite:
                result["incomplete_writer_gradients"] = {"missing": missing, "zero": zero, "nonfinite": nonfinite}
                raise ValueError("A writer did not receive finite nonzero query CE gradients")
            if any(parameter.grad is not None for _, parameter in frozen):
                raise ValueError("Frozen language parameters unexpectedly received gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], 1., error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            entry = {"step": step, "seconds": time.perf_counter() - start,
                     "query_ce": float(output.ce_loss.detach()), "support_surprise_monitor_only": float(output.surprise_loss.detach()),
                     "grad_norm": float(grad_norm), "writer_parameter_tensors": len(selected),
                     "finite_nonzero_writer_gradients": len(norms),
                     "writer_grad_norm_min": min(norms), "writer_grad_norm_max": max(norms)}
            if memory_reporter is not None:
                entry["memory"] = memory_reporter()
            result["steps"].append(entry)
            print("PERSONA_META_STEP " + json.dumps(entry), flush=True)
            del output, batch
    finally:
        model.forward = original_forward
    result["query_calls_checked_no_KV_or_support_text"] = checked_queries
    result["frozen_parameters_sha256_after"] = parameter_digest(frozen)
    result["frozen_parameters_unchanged"] = (result["frozen_parameters_sha256_before"] == result["frozen_parameters_sha256_after"])
    if not result["frozen_parameters_unchanged"]:
        raise ValueError("Frozen language parameters changed")
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        comparisons = []
        for shape, expected in zip(shapes, baseline):
            output = episode_forward(model, _batched(dataset, shape["index"], device), apply_writes=False)
            comparisons.append(torch.equal(output.logits.detach().cpu(), expected))
            del output
    result["no_write_logits_bitwise_unchanged"] = all(comparisons)
    if not result["no_write_logits_bitwise_unchanged"]:
        raise ValueError("No-write language logits changed after writer-only updates")
    result["optimizer_state_parameters"] = len(optimizer.state)
    result["optimizer_state_elements"] = sum(parameter.numel() for parameter in optimizer.state)
    result["optimizer_steps"] = sorted({int(state["step"]) for state in optimizer.state.values()})
    result["optimizer_moments_fp32"] = all(state[name].dtype == torch.float32 for state in optimizer.state.values()
                                           for name in ("exp_avg", "exp_avg_sq"))
    if len(optimizer.state) != len(selected) or result["optimizer_steps"] != [steps] or not result["optimizer_moments_fp32"]:
        raise ValueError("Incomplete full FP32 Adam state")
    result.update(status="ok", finite=True, useful_memory_quality_established=False)
    return result


def probe(checkpoint_path, episodes_path, *, seq_len=128, steps=2, chunk_size=128,
          meta_checkpoint_layers=4, max_total_tokens=384, learning_rate=1e-5, weight_decay=0.0):
    import torch
    from deploy.probe_selective_meta import guard_cuda_allocator, load_probe_checkpoint
    from model.tokenizer import build_tokenizer
    from model.uttt_model import uTTTConfig, uTTTModel
    from train_5090 import EpisodeMetaDataset, apply_training_parameter_scope

    if (type(seq_len) is not int or seq_len < 5 or type(chunk_size) is not int or chunk_size < 1
            or type(meta_checkpoint_layers) is not int or meta_checkpoint_layers < 1 or type(steps) is not int or steps < 2):
        raise ValueError("Invalid sequence, chunk, depth checkpoint or step budget")
    checkpoint = load_probe_checkpoint(checkpoint_path)
    config = uTTTConfig.from_dict(checkpoint["config"])
    if not config.tokenizer_json or seq_len > config.max_seq_len:
        raise ValueError("A compatible BPE checkpoint and sequence budget are required")
    config.chunk_size = chunk_size
    config.gradient_checkpointing = True
    config.gradient_checkpointing_every = 1
    config.meta_checkpoint_layers = meta_checkpoint_layers
    config.meta_checkpoint_time_chunks = True
    config.meta_checkpoint_reentrant = False
    dataset = EpisodeMetaDataset(build_tokenizer(config), seq_len, episodes_path, split="train")
    selected_shapes = choose_episodes(dataset, max_total_tokens=max_total_tokens)
    with torch.device("meta"):
        skeleton = uTTTModel(config)
    counts = apply_training_parameter_scope(skeleton, "meta", "writer")
    del skeleton
    static = counts["total"] * 4 + counts["trainable"] * 12
    ledger = storage_ledger(config.d_model, config.d_ff_fast, config.n_layers)
    if counts["trainable"] != ledger["writer_parameters"]:
        raise ValueError("Writer implementation differs from the verified ledger")
    result = {"protocol": PROTOCOL, "checkpoint": str(checkpoint_path), "checkpoint_step": checkpoint["step"],
              "episodes": str(episodes_path), "episodes_sha256": hashlib.sha256(Path(episodes_path).read_bytes()).hexdigest(),
              "dataset_fingerprint": dataset.fingerprint, "split": "train", "selected_episodes": selected_shapes,
              "seq_len_per_call": seq_len, "chunk_size": chunk_size, "max_padded_episode_tokens": max_total_tokens,
              "meta_checkpoint_layers": meta_checkpoint_layers, "meta_checkpoint_time_chunks": True,
              "meta_checkpoint_reentrant": False, "inner_loop_protocol": uTTTModel.INNER_LOOP_PROTOCOL,
              "static_fp32_weights_grads_adam_bytes": static, "persona_storage_ledger": ledger,
              "parameter_dtype": "float32", "activation_dtype": "bfloat16", "persona_master_dtype": "float32",
              "status": "running", "weights_saved": False, "useful_memory_quality_established": False}
    device_index, memory, limit = guard_cuda_allocator()
    result.update(available_memory=memory, allocator_limit_bytes=limit)
    if static > limit:
        return {**result, "status": "static_budget_exceeded"}
    device = torch.device("cuda", device_index)
    torch.cuda.reset_peak_memory_stats(device)

    def report_memory():
        return {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}

    try:
        torch.manual_seed(42)
        model = uTTTModel(config).float()
        model.load_dual_base_state(checkpoint["slow_weights"], checkpoint["init_fast_weights"], strict=True)
        del checkpoint
        model.to(device=device, dtype=torch.float32)
        run_episode_steps(model, dataset, device=device, amp_dtype=torch.bfloat16, steps=steps,
                          learning_rate=learning_rate, weight_decay=weight_decay, max_total_tokens=max_total_tokens,
                          memory_reporter=report_memory, report=result)
        if torch.cuda.max_memory_allocated(device) > 28 * 1024**3:
            result.update(status="over_budget")
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except (RuntimeError, ValueError) as error:
        result.update(status="failed", error=str(error), error_type=type(error).__name__)
    finally:
        result.update(report_memory())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--meta-checkpoint-layers", type=int, default=4)
    parser.add_argument("--max-total-tokens", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Probe reports are immutable; choose a new output path")
    try:
        result = probe(args.checkpoint.resolve(), args.episodes.resolve(), seq_len=args.seq_len, steps=args.steps,
                       chunk_size=args.chunk_size, meta_checkpoint_layers=args.meta_checkpoint_layers,
                       max_total_tokens=args.max_total_tokens, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    except (RuntimeError, ValueError) as error:
        result = {"protocol": PROTOCOL, "status": "refused", "error": str(error), "weights_saved": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("PERSONA_META_RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

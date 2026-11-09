"""Bounded recall diagnostics, separate from memory quality/activation scores.

No checkpoint or persona is saved. Run on an idle GPU, after training exits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STATE_NAMES = ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down", "ln_weight", "ln_bias")
GROUPS = ("payload_first", "payload_rest", "eos", "ready", "format_whitespace", "other")
REQUEST_BUDGET = 128


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_layout(tokenizer, row):
    """Map actual joint BPE offsets to content, never label bare spaces payload."""
    from deploy.prepare_persona_memory import query_prefix
    query, answer, payload = row["query_text"], row["answer"], row["memory_task"]["payload"]
    if answer not in (payload, payload + " READY"):
        raise ValueError("Diagnostic answers must be payload or payload + READY")
    prefix = query_prefix(query)
    text = prefix + " " + answer + "<|eos|>"
    encoded = tokenizer.backend.encode(text, add_special_tokens=False)
    ids = [tokenizer.BOS_ID] + encoded.ids
    offsets = [(0, 0)] + list(encoded.offsets)
    if ids != tokenizer.encode(text, add_bos=True, add_eos=False):
        raise ValueError("Tokenizer backend and model wrapper disagree")
    prefix_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    if ids[:len(prefix_ids)] != prefix_ids or ids[-1] != tokenizer.EOS_ID:
        raise ValueError("Generation prefix is not the exact joint-training token prefix")
    payload_start, payload_end = len(prefix) + 1, len(prefix) + 1 + len(payload)
    ready_start = payload_end + 1 if answer.endswith(" READY") else None
    positions, mixed, seen_payload = [], [], False
    for index in range(len(prefix_ids), len(ids)):
        start, end = offsets[index]
        overlaps_payload = max(start, payload_start) < min(end, payload_end)
        overlaps_ready = ready_start is not None and max(start, ready_start) < min(end, ready_start + 5)
        if ids[index] == tokenizer.EOS_ID:
            group = "eos"
        elif overlaps_payload:
            group = "payload_rest" if seen_payload else "payload_first"
            seen_payload = True
            if overlaps_ready:
                mixed.append(index)
        elif overlaps_ready:
            group = "ready"
        elif text[start:end] and text[start:end].isspace():
            group = "format_whitespace"
        else:
            group = "other"
        positions.append({"index": index, "token_id": ids[index], "group": group,
                          "offset": [start, end], "piece": text[start:end]})
    if not seen_payload or len(ids) > REQUEST_BUDGET:
        raise ValueError("Missing payload tokens or query exceeds the 128-token diagnostic budget")
    return {"ids": ids, "prefix_ids": prefix_ids, "positions": positions,
            "first_payload_index": next(value["index"] for value in positions if value["group"] == "payload_first"),
            "mixed_payload_ready_tokens": mixed}


def summarize_nll(positions, token_nll):
    result = {}
    for group in GROUPS:
        values = [token_nll[value["index"]] for value in positions if value["group"] == group]
        result[group] = {"tokens": len(values), "total_nll": sum(values),
                         "mean_nll": sum(values) / len(values) if values else None}
    payload = [token_nll[value["index"]] for value in positions if value["group"].startswith("payload_")]
    result["payload_all"] = {"tokens": len(payload), "total_nll": sum(payload), "mean_nll": sum(payload) / len(payload)}
    return result


def guard_idle_gpu(*, runner=None, own_pid=None):
    """Fail closed on an occupied or unobservable GPU; never send a signal."""
    own_pid = os.getpid() if own_pid is None else own_pid
    result = (runner or subprocess.run)(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=True,
    )
    if result.returncode:
        raise RuntimeError("Cannot establish idle GPU status")
    pids = []
    for line in result.stdout.split("\n"):
        value = line.strip()
        if value:
            if not value.isdecimal():
                raise RuntimeError("Unrecognized GPU process inventory; refusing diagnostics")
            if int(value) != own_pid:
                pids.append(int(value))
    if pids:
        raise RuntimeError(f"GPU is in use by PIDs {pids}; run diagnostics after training exits")
    return {"other_compute_pids": [], "scope": "Idle snapshot, not a GPU reservation"}


def _device(model):
    return next(model.parameters()).device


def _context(model, dtype):
    import torch
    return torch.autocast(_device(model).type, dtype=dtype, enabled=dtype != torch.float32)


def _input(model, ids):
    import torch
    return torch.tensor([ids], device=_device(model), dtype=torch.long)


def write_supports(model, tokenizer, row, *, dtype, padded=False):
    import torch
    from deploy.prepare_persona_memory import support_texts
    state = model.init_fast_state(batch_size=1, device=_device(model), dtype=torch.float32, detach=True)
    lengths = []
    with torch.no_grad(), _context(model, dtype):
        for text in support_texts(row):
            ids = tokenizer.encode(text, add_bos=True, add_eos=False)
            if not 1 <= len(ids) <= REQUEST_BUDGET:
                raise ValueError("Support exceeds the 128-token diagnostic budget")
            valid = len(ids)
            lengths.append(valid)
            if padded:
                ids = ids + [tokenizer.PAD_ID] * (REQUEST_BUDGET - valid)
            mask = torch.tensor([[1] * valid + [0] * (len(ids) - valid)], device=_device(model))
            state, _, _ = model.test_time_update(_input(model, ids), state, mask)
    return state, lengths


def compare_states(left, right):
    import torch
    fields = {name: {"bitwise_equal": torch.equal(getattr(left, name), getattr(right, name)),
                     "max_abs_difference": float((getattr(left, name).double() - getattr(right, name).double()).abs().max())}
              for name in STATE_NAMES}
    return {"all_tensors_bitwise_equal": all(value["bitwise_equal"] for value in fields.values()),
            "step_equal": left.step == right.step, "left_step": left.step, "right_step": right.step, "fields": fields}


def teacher_logits(model, state, ids, *, dtype, pad_to=None):
    import torch
    valid = len(ids)
    if pad_to is not None:
        if valid > pad_to:
            raise ValueError("Teacher-forced query exceeds padding budget")
        ids = ids + [0] * (pad_to - valid)
    inputs = _input(model, ids)
    mask = torch.tensor([[1] * valid + [0] * (len(ids) - valid)], device=_device(model))
    with torch.no_grad(), _context(model, dtype):
        result = model(inputs, fast_state=state.clone(detach=True), attention_mask=mask,
                       update_fast_weights=False, return_fast_state=False)
    return result.logits[0, :valid].float()


def token_nll(logits, layout):
    import torch
    return {value["index"]: float(-torch.log_softmax(logits[value["index"] - 1], -1)[value["token_id"]])
            for value in layout["positions"]}


def logits_difference(left, right):
    return {"max_abs_difference": float((left.float() - right.float()).abs().max()),
            "top1_equal": int(left.argmax()) == int(right.argmax()),
            "left_top1_id": int(left.argmax()), "right_top1_id": int(right.argmax())}


def cached_gold_alignment(model, state, layout, full_logits, *, dtype):
    """Advance cached decoding with identical gold tokens, never sampled tokens."""
    import torch
    state = state.clone(detach=True)
    versions = [getattr(state, name)._version for name in STATE_NAMES]
    inputs = _input(model, layout["prefix_ids"])
    result_rows, nll = [], {}
    with torch.no_grad(), _context(model, dtype):
        output = model(inputs, fast_state=state, update_fast_weights=False, logits_to_keep=1,
                       return_fast_state=False, kv_caches=None, start_pos=0)
        cache, next_logits = output.kv_caches, output.logits[0, -1].float()
        for offset, position in enumerate(layout["positions"]):
            index, target = position["index"], position["token_id"]
            if index != len(layout["prefix_ids"]) + offset:
                raise ValueError("Gold positions must be contiguous after the generation prefix")
            nll[index] = float(-torch.log_softmax(next_logits, -1)[target])
            result_rows.append({"index": index, "group": position["group"], "gold_token_id": target,
                                "cached_nll": nll[index],
                                **logits_difference(full_logits[index - 1], next_logits)})
            if offset + 1 < len(layout["positions"]):
                output = model(_input(model, [target]), fast_state=state, update_fast_weights=False,
                               logits_to_keep=1, return_fast_state=False, kv_caches=cache, start_pos=index)
                cache, next_logits = output.kv_caches, output.logits[0, -1].float()
    if any(getattr(state, name)._version != version for name, version in zip(STATE_NAMES, versions)):
        raise RuntimeError("Cached read mutated fast state")
    return {"tokens": result_rows, "all_top1_equal": all(value["top1_equal"] for value in result_rows),
            "max_abs_difference": max(value["max_abs_difference"] for value in result_rows),
            "nll_groups": summarize_nll(layout["positions"], nll)}


def diagnose_episode(model, tokenizer, row, counterfactual, *, dtype):
    if (row["query_text"] != counterfactual["query_text"] or row["answer"] == counterfactual["answer"]
            or row["memory_task"]["entity"] != counterfactual["memory_task"]["entity"]):
        raise ValueError("NLL control requires the same query/entity and a distinct written answer")
    correct, lengths = write_supports(model, tokenizer, row, dtype=dtype)
    padded, _ = write_supports(model, tokenizer, row, dtype=dtype, padded=True)
    wrong, _ = write_supports(model, tokenizer, counterfactual, dtype=dtype)
    layout = answer_layout(tokenizer, row)
    full = teacher_logits(model, correct, layout["ids"], dtype=dtype)
    other = teacher_logits(model, wrong, layout["ids"], dtype=dtype)
    padded_read = teacher_logits(model, correct, layout["ids"], dtype=dtype, pad_to=REQUEST_BUDGET)
    first = layout["first_payload_index"]
    before_payload = layout["ids"][:first]
    ordinary = teacher_logits(model, correct, before_payload, dtype=dtype)[-1]
    padded_write_logits = teacher_logits(model, padded, before_payload, dtype=dtype)[-1]
    correct_nll, wrong_nll = token_nll(full, layout), token_nll(other, layout)
    correct_groups = summarize_nll(layout["positions"], correct_nll)
    wrong_groups = summarize_nll(layout["positions"], wrong_nll)
    answer_indices = [value["index"] - 1 for value in layout["positions"]]
    return {"episode_id": row["episode_id"], "family": row["family"],
            "task": row["memory_task"]["task"], "language": row["memory_task"]["language"],
            "counterfactual_episode_id": counterfactual["episode_id"], "support_tokens": lengths,
            "padding_write_state": compare_states(correct, padded),
            "padding_write_first_payload_logits": logits_difference(ordinary, padded_write_logits),
            "query_padding_max_abs_difference": float((full[answer_indices] - padded_read[answer_indices]).abs().max()),
            "query_padding_all_top1_equal": bool((full[answer_indices].argmax(-1) == padded_read[answer_indices].argmax(-1)).all()),
            "cached_gold_alignment": cached_gold_alignment(model, correct, layout, full, dtype=dtype),
            "correct_memory_nll": correct_groups, "same_query_counterfactual_memory_nll": wrong_groups,
            "payload_nll_gain_from_correct_memory": wrong_groups["payload_all"]["mean_nll"] - correct_groups["payload_all"]["mean_nll"],
            "first_payload_nll_gain_from_correct_memory": wrong_nll[first] - correct_nll[first],
            "token_positions": layout["positions"], "mixed_payload_ready_tokens": layout["mixed_payload_ready_tokens"]}


def oracle_examples(rows):
    if not rows:
        raise ValueError("Oracle diagnostics require source rows")
    templates = (("zh", "当前资料中的代码是{payload}。请仅复制这个代码。"),
                 ("en", "The code in the current note is {payload}. Copy only that exact code."),
                 ("zh", "请逐字复制方括号内的代码[{payload}]，只输出代码。"),
                 ("en", "Copy exactly the code inside brackets [{payload}]. Output the code alone."))
    return [{"language": language, "query": text.format(payload=rows[index % len(rows)]["memory_task"]["payload"]),
             "expected": rows[index % len(rows)]["memory_task"]["payload"],
             "source_episode_id": rows[index % len(rows)]["episode_id"]}
            for index, (language, text) in enumerate(templates)]


def select_diagnostic_rows(rows, maximum):
    """Cover recall/correction/READY within four cases; this is not a score sample."""
    from deploy.evaluate_persona_memory import balanced_indices
    ordered = balanced_indices(rows, len(rows))
    chosen = []
    for task, language in (("recall", "en"), ("correction", "zh"), ("reply_prefix", "en")):
        candidates = [index for index in ordered if rows[index]["memory_task"]["task"] == task]
        preferred = [index for index in candidates if rows[index]["memory_task"]["language"] == language]
        if candidates and len(chosen) < maximum:
            chosen.append((preferred or candidates)[0])
    for index in ordered:
        if index not in chosen and len(chosen) < maximum:
            chosen.append(index)
    return [rows[index] for index in chosen]


def diagnose(checkpoint, data_path, output, *, split="dev", max_samples=4, temp_dir, device="cuda"):
    import torch
    from deploy.evaluate_memory import guard_gpu
    from deploy.evaluate_persona_memory import ModelAdapter, load_episodes
    from deploy.prepare_memory_curriculum import load_tokenizer
    from deploy.prepare_persona_memory import query_prefix
    from deploy.serve_guard import assert_serving_available
    from engine.checkpoint_manager import DualTrackCheckpointManager
    from model.tokenizer import build_tokenizer
    from model.uttt_model import uTTTModel

    if split != "dev" or type(max_samples) is not int or not 1 <= max_samples <= 4:
        raise ValueError("Diagnostics use dev only, with 1 to 4 episodes")
    checkpoint, data_path, output, scratch = map(lambda value: Path(value).resolve(), (checkpoint, data_path, output, temp_dir))
    protected = {checkpoint, data_path, data_path.parent / "manifest.json", data_path.parent / "test.jsonl"}
    if output.exists() or output in protected:
        raise FileExistsError("Choose a fresh diagnostic JSON output")
    device = torch.device(device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Only CPU/CUDA devices are supported")
    if device.type == "cuda" and not (str(scratch).startswith("/mnt/c/") or scratch.drive.upper() == "C:"):
        raise ValueError("CUDA diagnostic temporary personas require an explicit C-volume scratch directory")
    idle = gpu = None
    if device.type == "cuda":
        assert_serving_available()
        idle = guard_idle_gpu()
        gpu = guard_gpu(torch, device, checkpoint.stat().st_size)
    checkpoint_sha, data_sha = file_hash(checkpoint), file_hash(data_path)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recall-diagnostic-", dir=scratch) as temporary:
        temporary = Path(temporary)
        header = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        metadata, config = header.get("metrics", {}), header["config"]
        checkpoint_step = header.get("step")
        if metadata.get("training_stage") != "meta" or metadata.get("inner_loop_protocol") != uTTTModel.INNER_LOOP_PROTOCOL:
            raise ValueError("A compatible actual meta checkpoint is required")
        tokenizer_path = temporary / "tokenizer.json"
        tokenizer_path.write_text(config["tokenizer_json"], encoding="utf-8")
        backend, tokenizer_sha = load_tokenizer(tokenizer_path)
        rows, manifest, provenance = load_episodes(data_path, backend, tokenizer_sha, split=split,
                                                  context_limit=config["max_seq_len"])
        chosen = select_diagnostic_rows(rows, max_samples)
        partners = []
        for row in chosen:
            matches = [other for other in rows if other["query_text"] == row["query_text"]
                       and other["memory_task"]["entity"] == row["memory_task"]["entity"] and other["answer"] != row["answer"]]
            if len(matches) != 1:
                raise ValueError("Use paired dev data with exactly one same-query counterfactual per selected row")
            partners.append(matches[0])
        del header, config
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        manager = DualTrackCheckpointManager(checkpoint, personas_dir=temporary / "personas", device=str(device),
                                             dtype=dtype, max_gpu_users=1, max_cpu_users=1, auto_create_base_if_missing=False)
        model = manager.model.eval().requires_grad_(False)
        if any(value.dtype != torch.float32 for value in model.fast_weight_pool.parameters()):
            raise ValueError("Diagnostic requires the original FP32 fast-weight and LN masters")
        tokenizer = build_tokenizer(manager.config)
        versions = {name: value._version for name, value in model.named_parameters()}
        with torch.no_grad():
            results = [diagnose_episode(model, tokenizer, row, partner, dtype=dtype) for row, partner in zip(chosen, partners)]
            adapter = ModelAdapter(manager, max_new_tokens=32, apply_guards=False)
            oracles = []
            for case in oracle_examples(chosen):
                if len(tokenizer.encode(query_prefix(case["query"]), add_bos=True, add_eos=False)) + 32 > REQUEST_BUDGET:
                    raise ValueError("Oracle copy exceeds the 128-token diagnostic request budget")
                reply = adapter.read(adapter.base_state(), case["query"])
                oracles.append({**case, "reply": reply, "exact_copy": reply == case["expected"],
                                "answer_in_current_query": True, "writes_enabled": False})
        if any(value._version != versions[name] for name, value in model.named_parameters()):
            raise RuntimeError("Diagnostic changed a model parameter")
        report = {"protocol": "persona-recall-diagnostic-v1", "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                  "checkpoint_step": checkpoint_step, "inner_loop_protocol": uTTTModel.INNER_LOOP_PROTOCOL,
                  "diagnostic_source_sha256": file_hash(__file__),
                  "data_path": str(data_path), "data_sha256": data_sha, "split": split, **provenance,
                  "request_budget": REQUEST_BUDGET, "max_samples": max_samples, "results": results,
                  "selection": "Deterministic task coverage including READY where available; not a memory-quality score sample",
                  "oracle_copy_diagnostic": {"separate_from_memory_score": True, "examples": oracles},
                  "cuda_graphs_used": False, "checkpoint_or_persona_saved": False, "stability_guards_applied": False,
                  "notes": ["Gold-token NLL is teacher forced; cached alignment feeds identical gold tokens, not sampled replies.",
                            "Whitespace-only tokens never count as payload. A token crossing payload/READY counts once as payload and is disclosed.",
                            "Oracle copy includes the answer in the current request and is not a memory-recall score.",
                            "Temporary personas use the configured scratch directory; no saved persona or checkpoint is created."]}
        if gpu is not None:
            report["gpu_guard"] = {**gpu, **idle, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if file_hash(checkpoint) != checkpoint_sha or file_hash(data_path) != data_sha:
        raise RuntimeError("Checkpoint or data changed during diagnostics")
    if file_hash(data_path.parent / "manifest.json") != provenance["manifest_sha256"]:
        raise RuntimeError("Data manifest changed during diagnostics")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--temp-dir", required=True, type=Path, help="Explicit C-volume scratch directory on CUDA/WSL")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    report = diagnose(**vars(args))
    print(json.dumps({"protocol": report["protocol"], "episodes": len(report["results"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

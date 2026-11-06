"""Paired teacher-forced delayed-memory evaluation; never enables online writes.

Only final assistant answer tokens and their first real EOS are CE targets.
Teacher-forced token accuracy is not free-generation exact-answer accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys


GIB = 1024 ** 3
MAX_ALLOCATOR_BYTES = 28 * GIB
DRIVER_RESERVE_BYTES = int(1.5 * GIB)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(not isinstance(row.get("text"), str) or not isinstance(row.get("memory_task"), dict) for row in rows):
        raise ValueError("Memory data needs nonempty text and memory_task metadata")
    return rows


def identifier_sets(rows):
    entities, payloads = set(), set()
    for row in rows:
        metadata = row["memory_task"]
        if metadata.get("task") not in ("recall", "correction") or metadata.get("language") not in ("en", "zh"):
            raise ValueError("Memory records require a supported task and language")
        entity, answer = metadata.get("entity"), metadata.get("answer")
        if not isinstance(entity, str) or not entity or not isinstance(answer, str) or not answer:
            raise ValueError("Memory records need entity and answer metadata")
        entities.add(entity)
        payloads.add(answer)
        old = metadata.get("obsolete_answer")
        if old is not None:
            if not isinstance(old, str) or not old or old == answer:
                raise ValueError("Correction metadata must contain a distinct obsolete answer")
            payloads.add(old)
    return entities, payloads


def load_heldout(path, tokenizer_sha256, *, chunk_size, sliding_window, context_limit):
    from deploy.prepare_memory_curriculum import PROTOCOL
    path = Path(path).resolve()
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or manifest.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("Memory manifest protocol/tokenizer does not match this checkpoint")
    if (manifest.get("chunk_size") != chunk_size or manifest.get("sliding_window") != sliding_window
            or not 1 <= manifest.get("seq_len", 0) <= context_limit):
        raise ValueError("Memory curriculum chunk/window/sequence geometry differs from the checkpoint")
    if manifest.get("identifier_intersections") != {"entities": 0, "payloads": 0}:
        raise ValueError("Memory manifest does not establish disjoint heldout identifiers")
    files = manifest.get("files", {})
    heldout, train = files.get("heldout", {}), files.get("train", {})
    if heldout.get("file") != path.name or path.name == train.get("file"):
        raise ValueError("Evaluation requires the manifest's heldout file, never the training file")
    for entry in (heldout, train):
        name = entry.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Manifest files must be siblings of the heldout file")
        if file_hash(path.parent / name) != entry.get("sha256"):
            raise ValueError("Curriculum file hash differs from its preparation manifest")
    if train.get("sha256") == heldout.get("sha256"):
        raise ValueError("Training and heldout file hashes must differ")
    rows, training_rows = read_rows(path), read_rows(path.parent / train["file"])
    if len(rows) != heldout.get("rows") or len(training_rows) != train.get("rows"):
        raise ValueError("Curriculum row counts differ from their manifest")
    heldout_ids, training_ids = identifier_sets(rows), identifier_sets(training_rows)
    if any(left & right for left, right in zip(heldout_ids, training_ids)):
        raise ValueError("Training and heldout identifiers overlap")
    return rows, manifest, {"data_sha256": heldout["sha256"], "train_sha256": train["sha256"],
                            "manifest_sha256": file_hash(manifest_path), "heldout_rows": len(rows),
                            "verified_identifier_intersections": {"entities": 0, "payloads": 0}}


def sample_indices(rows, maximum, seed=42):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max-samples must be a positive integer")
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row["memory_task"]
        buckets[(metadata.get("task"), metadata.get("language"))].append(index)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(maximum, len(rows)):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < maximum:
                selected.append(buckets[key].pop())
    return selected


def prepare_document(row, tokenizer, config):
    from deploy.prepare_memory_curriculum import measure
    from train_5090 import PersonaMetaDataset
    text, metadata = row["text"], row["memory_task"]
    if not text.startswith("<|user|> ") or "\n\n" not in text:
        raise ValueError("Memory row is missing its facts/filler boundary")
    observed = measure(tokenizer.backend, text, text.index("\n\n"), config.chunk_size)
    if any(metadata.get(key) != value for key, value in observed.items()):
        raise ValueError("Memory row's BPE distance metadata does not match its text")
    if (observed["tokens_with_dataset_bos_eos"] > config.max_seq_len
            or observed["fact_to_answer_gap_tokens"] <= config.sliding_window_size
            or observed["answer_chunk"] <= observed["fact_last_chunk"]):
        raise ValueError("Memory row does not place the completed fact beyond one attention window and in an earlier chunk")
    document = object.__new__(PersonaMetaDataset)
    document.tokenizer, document.mode = tokenizer, "meta"
    ids, labels = document._document(text)
    assistant = ids.index(tokenizer.ASSISTANT_ID)
    if (ids.count(tokenizer.USER_ID) != 1 or ids.count(tokenizer.ASSISTANT_ID) != 1
            or any(value != -100 for value in labels[:assistant + 1])
            or ids[-2:] != [tokenizer.EOS_ID, tokenizer.EOS_ID]
            or labels[-2:] != [tokenizer.EOS_ID, -100]
            or tokenizer.decode(ids[assistant + 1:-2]).strip() != metadata.get("answer")):
        raise ValueError("Only one final assistant answer with a supervised real EOS is allowed")
    answer = metadata["answer"]
    facts, remainder = text.split("\n\n", 1)
    before_answer = remainder.split("\n<|assistant|>", 1)[0]
    if answer not in facts or any(value and value.casefold() in before_answer.casefold()
                                 for value in (answer, metadata.get("obsolete_answer"))):
        raise ValueError("Answer payload must occur in the early facts, never the filler or retrieval question")
    return ids, labels, observed


def guard_gpu(torch, device, parameter_bytes):
    from deploy.gpu_memory import GpuMemoryError, query_available_gpu_memory
    if not torch.cuda.is_available():
        raise GpuMemoryError("CUDA evaluation requested but CUDA is unavailable")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    runtime_free, _ = torch.cuda.mem_get_info(index)
    memory = query_available_gpu_memory(runtime_free, device_uuid=getattr(properties, "uuid", None),
                                        device_index=index, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    minimum = max(2 * GIB, parameter_bytes + 2 * GIB)
    budget = min(MAX_ALLOCATOR_BYTES, memory["free_bytes"] - DRIVER_RESERVE_BYTES)
    if budget < minimum or budget <= 0 or int(properties.total_memory) <= budget:
        raise GpuMemoryError("Insufficient independently verified free GPU memory for isolated memory evaluation; stop other GPU workloads")
    torch.cuda.set_per_process_memory_fraction(budget / int(properties.total_memory), device=index)
    torch.cuda.reset_peak_memory_stats(index)
    return {key: value for key, value in memory.items() if key != "device_uuid"} | {"allocator_limit_bytes": budget}


def evaluate_pair(model, ids, labels, *, device, activation_dtype):
    import torch
    inputs = torch.tensor([ids], device=device, dtype=torch.long)
    targets = torch.tensor([labels], device=device, dtype=torch.long)
    shifted = targets[:, 1:]
    valid = shifted != -100
    count = int(valid.sum())
    if count < 2:
        raise ValueError("Answer supervision needs content and EOS targets")
    result = {}
    previous = model.training
    model.eval()
    versions = {name: value._version for name, value in model.named_parameters()}
    try:
        with torch.no_grad():
            for name, dynamic in (("frozen", False), ("dynamic", True)):
                # Both conditions start from their own FP32 W_0 copy and zero
                # momentum. No state is carried between arms or documents.
                state = model.init_fast_state(batch_size=1, device=device, dtype=torch.float32, detach=True)
                with torch.autocast(device.type, dtype=activation_dtype, enabled=activation_dtype != torch.float32):
                    output = model(inputs, labels=targets, attention_mask=torch.ones_like(inputs),
                                   fast_state=state, update_fast_weights=dynamic, surprise_aux_weight=0.0,
                                   return_fast_state=False)
                ce = float(output.ce_loss)
                correct = int(((output.logits[:, :-1].argmax(-1) == shifted) & valid).sum())
                if not math.isfinite(ce):
                    raise ValueError("Memory evaluation produced non-finite answer CE")
                result[name] = {"answer_ce": ce, "teacher_forced_token_accuracy": correct / count,
                                "correct_tokens": correct, "answer_tokens_including_eos": count}
                del output, state
        if any(parameter._version != versions[name] for name, parameter in model.named_parameters()):
            raise RuntimeError("Evaluation changed a checkpoint model parameter")
    finally:
        model.train(previous)
    result["paired_ce_gain"] = result["frozen"]["answer_ce"] - result["dynamic"]["answer_ce"]
    result["dynamic_improved"] = result["paired_ce_gain"] > 0
    return result


def aggregate(rows):
    count = len(rows)
    tokens = sum(row["frozen"]["answer_tokens_including_eos"] for row in rows)
    result = {"documents": count, "answer_tokens_including_eos": tokens}
    for arm in ("frozen", "dynamic"):
        result[arm] = {
            "answer_ce": sum(row[arm]["answer_ce"] * row[arm]["answer_tokens_including_eos"] for row in rows) / tokens,
            "teacher_forced_token_accuracy": sum(row[arm]["correct_tokens"] for row in rows) / tokens,
        }
    result["paired_ce_gain"] = result["frozen"]["answer_ce"] - result["dynamic"]["answer_ce"]
    result["mean_document_ce_gain"] = sum(row["paired_ce_gain"] for row in rows) / count
    result["improved_document_fraction"] = sum(row["dynamic_improved"] for row in rows) / count
    return result


def evaluate(checkpoint, data_path, output, max_samples=64, device=None, *, allow_untrained_baseline=False):
    import torch
    from model.tokenizer import build_tokenizer
    from model.uttt_model import uTTTConfig, uTTTModel
    checkpoint, data_path, output = (Path(value).resolve() for value in (checkpoint, data_path, output))
    if output.exists() or output in (checkpoint, data_path, data_path.parent / "manifest.json"):
        raise FileExistsError("Use a new evaluation output; checkpoint and curriculum files are read-only")
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError("max-samples must be a positive integer")
    checkpoint_stat = checkpoint.stat()
    checkpoint_hash = file_hash(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if not all(isinstance(payload.get(key), dict) for key in ("config", "slow_weights", "init_fast_weights")):
        raise ValueError("An existing dual-track checkpoint is required")
    metadata = payload.get("metrics", {})
    checkpoint_step = payload.get("step")
    trained = metadata.get("training_stage") == "meta"
    if trained and metadata.get("inner_loop_protocol") != uTTTModel.INNER_LOOP_PROTOCOL:
        raise ValueError("Meta checkpoint inner-loop protocol does not match this evaluator")
    if not trained and not allow_untrained_baseline:
        raise ValueError("Memory assessment requires a meta checkpoint; explicitly allow an untrained baseline for diagnostic comparison only")
    config = uTTTConfig.from_dict(payload["config"])
    if not config.tokenizer_json:
        raise ValueError("The actual BPE tokenizer must be embedded in the checkpoint")
    tokenizer_hash = hashlib.sha256(config.tokenizer_json.encode("utf-8")).hexdigest()
    rows, manifest, provenance = load_heldout(data_path, tokenizer_hash, chunk_size=config.chunk_size,
                                              sliding_window=config.sliding_window_size, context_limit=config.max_seq_len)
    tokenizer = build_tokenizer(config)
    indices = sample_indices(rows, max_samples)
    documents = [(index, prepare_document(rows[index], tokenizer, config)) for index in indices]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Memory evaluation supports cpu or cuda")
    with torch.device("meta"):
        skeleton = uTTTModel(config)
        parameter_bytes = sum(parameter.numel() * 4 for parameter in skeleton.parameters())
    del skeleton
    gpu = guard_gpu(torch, device, parameter_bytes) if device.type == "cuda" else None
    model = uTTTModel(config).float()
    model.load_dual_base_state(payload["slow_weights"], payload["init_fast_weights"], strict=True)
    del payload
    model = model.to(device=device).eval()
    model.requires_grad_(False)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    results = []
    for index, (ids, labels, geometry) in documents:
        task = rows[index]["memory_task"]
        result = {"document_index": index, "text_sha256": hashlib.sha256(rows[index]["text"].encode()).hexdigest(),
                  "task": task.get("task"), "language": task.get("language"), "sequence": geometry,
                  **evaluate_pair(model, ids, labels, device=device, activation_dtype=dtype)}
        results.append(result)
        print("MEMORY_EVALUATION_DOCUMENT " + json.dumps(result, sort_keys=True), flush=True)
    after_stat = checkpoint.stat()
    if (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
        raise RuntimeError("Checkpoint changed during evaluation; refusing an ambiguous result")
    if file_hash(data_path) != provenance["data_sha256"]:
        raise RuntimeError("Heldout data changed during evaluation")
    groups = {}
    for field in ("task", "language"):
        groups[field] = {value: aggregate([row for row in results if row[field] == value])
                         for value in sorted({row[field] for row in results})}
    report = {"protocol": "paired_teacher_forced_memory_v1", "checkpoint": str(checkpoint),
              "checkpoint_sha256": checkpoint_hash, "checkpoint_step": checkpoint_step,
              "training_stage": metadata.get("training_stage"),
              "training_dataset_fingerprint": metadata.get("dataset_fingerprint"),
              "inner_loop_protocol": uTTTModel.INNER_LOOP_PROTOCOL, "meta_trained": trained,
              "untrained_baseline": not trained, "online_writes_allowed": False,
              "free_generation_accuracy_measured": False, "batch_size": 1,
              "parameter_dtype": "float32", "dynamic_state_dtype": "float32", "activation_dtype": str(dtype),
              "data_path": str(data_path), "tokenizer_sha256": tokenizer_hash, **provenance,
              "curriculum_protocol": manifest["protocol"], "sample_seed": 42, "maximum_samples": max_samples,
              "summary": aggregate(results), "groups": groups, "results": results,
              "notes": ["CE and token accuracy use teacher forcing on final assistant answer tokens and their first EOS only.",
                        "Prior answer tokens are supplied as input; these are not free-generation exact-answer scores.",
                        "Each arm and each document starts from independent FP32 base weights and zero momentum.",
                        "A gap beyond one attention window does not rule out information propagation through multiple frozen attention layers.",
                        "A paired CE gain alone does not prove useful or safe online memory; this evaluator never authorizes online writes."]}
    if not trained:
        report["notes"].append("Untrained baseline: dynamic versus frozen differences cannot establish learned memory capability.")
    if gpu is not None:
        report["gpu_memory"] = gpu | {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--allow-untrained-baseline", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.data_path, args.output, args.max_samples, args.device,
                      allow_untrained_baseline=args.allow_untrained_baseline)
    print("MEMORY_EVALUATION_COMPLETE " + json.dumps(report["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())

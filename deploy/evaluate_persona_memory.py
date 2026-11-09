"""Query-only free-generation evaluation of split-request persona fast weights.

Support text/KV never enters a read request. This tests analytical writes and
isolated persona persistence, not concurrent HTTP routing or general chat.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .prepare_persona_memory import PROTOCOL, measure_episode, query_prefix, support_texts
except ImportError:
    from prepare_persona_memory import PROTOCOL, measure_episode, query_prefix, support_texts


ARMS = ("base_w0", "no_write", "mismatched_state", "correct_write", "save_reload")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episodes(path, tokenizer, tokenizer_sha256, *, split="test", context_limit=4096):
    """Verify bytes, actual split separation and BPE before any model allocation."""
    path = Path(path).resolve()
    if split not in ("dev", "test") or path.name != ("test.jsonl" if split == "test" else "train.jsonl"):
        raise ValueError("Evaluation requires explicit dev/train.jsonl or test/test.jsonl")
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or manifest.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("Episode protocol/tokenizer does not match checkpoint")
    budget = manifest.get("seq_len")
    if type(budget) is not int or not 16 <= budget <= context_limit:
        raise ValueError("Episode token budget exceeds model context")
    all_rows = []
    for name in ("train.jsonl", "test.jsonl"):
        entry = manifest.get("files", {}).get(name, {})
        sibling = path.parent / name
        if file_hash(sibling) != entry.get("sha256"):
            raise ValueError("Episode file differs from preparation manifest")
        with sibling.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != entry.get("rows"):
            raise ValueError("Episode row count differs from preparation manifest")
        for row in rows:
            metadata = row.get("memory_task", {})
            part = metadata.get("split")
            if (part not in ("train", "dev", "test") or (part == "test") != (name == "test.jsonl")
                    or metadata.get("task") not in ("recall", "correction", "reply_prefix")
                    or metadata.get("language") not in ("en", "zh")
                    or any(not isinstance(metadata.get(key), str) or not metadata[key]
                           for key in ("entity", "payload", "prompt_family"))
                    or not isinstance(row.get("family"), str) or not row["family"]
                    or row.get("validation_group_id") != row["family"]
                    or row.get("tokenizer_sha256") != tokenizer_sha256):
                raise ValueError("Episode is missing explicit split/family/fact metadata")
            if measure_episode(row, tokenizer, budget) != row.get("lengths"):
                raise ValueError("Episode actual BPE lengths differ from recorded lengths")
            payload, old = metadata["payload"], metadata.get("obsolete_payload")
            expected = payload + " READY" if metadata["task"] == "reply_prefix" else payload
            if (row.get("answer") != expected or payload.casefold() in row["query_text"].casefold()
                    or (old and old.casefold() in row["query_text"].casefold())):
                raise ValueError("Episode target is inconsistent or exposed in the query")
            if metadata["task"] == "correction":
                if not old or old == payload or old not in row["write_text"] or payload not in row.get("correction_text", ""):
                    raise ValueError("Correction must be a distinct independent replacement write")
            elif payload not in row["write_text"]:
                raise ValueError("Support text does not contain the expected fact")
        all_rows.extend(rows)
    identities = {part: {key: set() for key in ("entities", "payloads", "prompt_families", "families")}
                  for part in ("train", "dev", "test")}
    for row in all_rows:
        metadata = row["memory_task"]
        sets = identities[metadata["split"]]
        sets["entities"].add(metadata["entity"])
        sets["payloads"].update(value for value in (metadata["payload"], metadata.get("obsolete_payload")) if value)
        sets["prompt_families"].add(metadata["prompt_family"])
        sets["families"].add(row["family"])
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if any(identities[left][key] & identities[right][key] for key in identities[left]):
            raise ValueError("Actual episode split identities overlap")
    selected = [row for row in all_rows if row["memory_task"]["split"] == split]
    if len(selected) < 2:
        raise ValueError("At least two heldout users are required for mismatched-state controls")
    return selected, manifest, {"data_sha256": file_hash(path), "manifest_sha256": file_hash(manifest_path),
                                "split": split, "verified_split_intersections": 0}


def balanced_indices(rows, maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_samples must be a positive integer")
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row["memory_task"]
        buckets[metadata["task"], metadata["language"]].append(index)
    for values in buckets.values():
        values.sort(key=lambda index: hashlib.sha256(rows[index]["episode_id"].encode()).hexdigest())
    chosen = []
    while len(chosen) < min(maximum, len(rows)):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < min(maximum, len(rows)):
                chosen.append(buckets[key].pop())
    return chosen


def evaluate_episode(adapter, row, other):
    """Adapter reads receive exactly query_text and a cloned tensor-only state."""
    base = adapter.base_state()
    untouched_user = adapter.clone(base)
    other_before = adapter.read(adapter.clone(untouched_user), other["query_text"])
    no_write = adapter.clone(base)
    for text in support_texts(row):
        no_write = adapter.no_write(no_write, text)
    written = adapter.write(adapter.clone(base), row["write_text"])
    correction = None
    if row.get("correction_text"):
        before = adapter.read(adapter.clone(written), row["query_text"])
        written = adapter.write(written, row["correction_text"])
        correction = {"before_exact_old": before.strip() == row["memory_task"]["obsolete_payload"]}
    for text in row.get("interference_texts", []):
        written = adapter.write(written, text)
    wrong = adapter.clone(base)
    for text in support_texts(other):
        wrong = adapter.write(wrong, text)
    other_after = adapter.read(adapter.clone(untouched_user), other["query_text"])
    reloaded = adapter.save_reload(adapter.clone(written))
    states = {"base_w0": base, "no_write": no_write, "mismatched_state": wrong,
              "correct_write": written, "save_reload": reloaded}
    outcomes = {}
    replies = {}
    for name in ARMS:
        reply = adapter.read(adapter.clone(states[name]), row["query_text"]).strip()
        replies[name] = reply
        outcomes[name] = {"exact_match": reply == row["answer"],
                          "mismatched_payload_recalled": reply == other["answer"]}
    if correction is not None:
        correction.update(after_exact_new=outcomes["correct_write"]["exact_match"],
                          obsolete_recalled_after=replies["correct_write"] == row["memory_task"]["obsolete_payload"])
    return {"episode_id": row["episode_id"], "task": row["memory_task"]["task"],
            "language": row["memory_task"]["language"],
            "query_sha256": hashlib.sha256(row["query_text"].encode()).hexdigest(), "arms": outcomes,
            "expected": row["answer"], "replies": replies,
            "correct_state_changed": not adapter.equal(base, written),
            "no_write_state_unchanged": adapter.equal(base, no_write),
            "other_user_state_unchanged": adapter.equal(base, untouched_user),
            "other_user_reply_unchanged": other_before == other_after,
            "save_reload_state_exact": adapter.equal(written, reloaded),
            "save_reload_reply_exact": replies["correct_write"] == replies["save_reload"],
            "correction": correction}


def evaluate_with_adapter(rows, adapter, *, max_samples=64, progress=None):
    if len(rows) < 2:
        raise ValueError("At least two independent users are required")
    results = []
    for index in balanced_indices(rows, max_samples):
        row = rows[index]
        candidates = [other for other in rows if other["memory_task"]["entity"] != row["memory_task"]["entity"]]
        if not candidates:
            raise ValueError("Mismatched-state control needs another user")
        same_group = [other for other in candidates if (other["memory_task"]["task"], other["memory_task"]["language"])
                      == (row["memory_task"]["task"], row["memory_task"]["language"])]
        other = (same_group or candidates)[0]
        results.append(evaluate_episode(adapter, row, other))
        if progress is not None:
            progress({"episode": len(results), "episode_id": row["episode_id"],
                      "correct_write": results[-1]["arms"]["correct_write"]["exact_match"]})
    def aggregate(items):
        count = len(items)
        arms = {arm: {"passes": sum(row["arms"][arm]["exact_match"] for row in items), "total": count,
                      "accuracy": sum(row["arms"][arm]["exact_match"] for row in items) / count}
                for arm in ARMS}
        return {"episodes": count, "arms": arms,
                "write_minus_no_write_accuracy": arms["correct_write"]["accuracy"] - arms["no_write"]["accuracy"],
                "write_minus_mismatch_accuracy": arms["correct_write"]["accuracy"] - arms["mismatched_state"]["accuracy"],
                "state_changed_episodes": sum(row["correct_state_changed"] for row in items),
                "isolation_failures": sum(not row["other_user_state_unchanged"] or not row["other_user_reply_unchanged"] for row in items),
                "persistence_failures": sum(not row["save_reload_state_exact"] or not row["save_reload_reply_exact"] for row in items),
                "no_write_state_failures": sum(not row["no_write_state_unchanged"] for row in items)}
    groups = {field: {value: aggregate([row for row in results if row[field] == value])
                      for value in sorted({row[field] for row in results})} for field in ("task", "language")}
    return {"protocol": "persona_query_only_free_generation_v1", "free_generation_accuracy_measured": True,
            "teacher_forcing_used": False, "support_text_or_kv_carried_to_query": False,
            "summary": aggregate(results), "groups": groups, "results": results,
            "notes": ["Every read is query-only, starts at position zero and uses an independently cloned fast-weight/momentum state.",
                      "State changes or low surprise alone do not demonstrate useful memory; compare exact recall against both no-write and mismatch.",
                      "Isolation checks independent persona states and persistence, not concurrent HTTP session routing.",
                      "Reply-prefix exercises are narrow verifiable personalization, not a natural-conversation quality score."]}


class ModelAdapter:
    def __init__(self, manager, *, max_new_tokens=32, apply_guards=False):
        import torch
        from model.tokenizer import build_tokenizer
        self.torch = torch
        self.manager, self.model = manager, manager.model
        self.tokenizer = build_tokenizer(manager.config)
        self.device = manager.device
        self.dtype = torch.bfloat16 if manager.device.type == "cuda" else torch.float32
        self.max_new_tokens, self.apply_guards = max_new_tokens, apply_guards
        self.write_metrics = []

    def clone(self, state):
        # clone() copies W/S/LN + scalar metadata and drops all private caches.
        return state.clone(detach=True)

    def base_state(self):
        return self.model.init_fast_state(batch_size=1, device=self.device, dtype=self.torch.float32, detach=True)

    def equal(self, left, right):
        tensors = ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down", "ln_weight", "ln_bias")
        return left.step == right.step and all(self.torch.equal(getattr(left, name), getattr(right, name)) for name in tensors)

    def _input(self, text):
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)
        if len(ids) > self.model.config.max_seq_len:
            raise ValueError("A benchmark request exceeds model context; no cropping is permitted")
        return self.torch.tensor([ids], device=self.device, dtype=self.torch.long)

    def write(self, state, text):
        torch = self.torch
        inputs = self._input(text)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            candidate, surprise, _ = self.model.test_time_update(inputs, self.clone(state), self.torch.ones_like(inputs))
        if not math.isfinite(surprise) or surprise < 0:
            raise ValueError("Non-finite support surprise")
        committed = True
        if self.apply_guards:
            committed = surprise > self.model.config.surprise_threshold
            candidate = (self.model.fast_weight_pool.apply_elastic_pull(candidate,
                         anchor_lambda=self.model.config.elastic_anchor_lambda,
                         max_drift_ratio=self.model.config.max_weight_drift_ratio) if committed else self.clone(state))
        self.write_metrics.append({"surprise": surprise, "committed": committed})
        return self.clone(candidate)

    def no_write(self, state, text):
        torch = self.torch
        inputs = self._input(text)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            self.model(inputs, fast_state=self.clone(state), attention_mask=torch.ones_like(inputs),
                       update_fast_weights=False, return_fast_state=False, logits_to_keep=-1)
        return self.clone(state)

    def read(self, state, query):
        # No history parameter, support IDs, previous KV or previous positions.
        torch = self.torch
        inputs = self._input(query_prefix(query))
        if inputs.shape[1] + self.max_new_tokens > self.model.config.max_seq_len:
            raise ValueError("Query plus generation budget exceeds context")
        versions = [(value, value._version) for value in (state.w_up, state.w_gate, state.w_down,
                                                         state.s_up, state.s_gate, state.s_down)]
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            generated = list(self.model.stream_generate(inputs, fast_state=state, max_new_tokens=self.max_new_tokens,
                             temperature=0.0, top_p=1.0, eos_token_id=self.tokenizer.EOS_ID, use_cuda_graph=False))
        if any(value._version != version for value, version in versions):
            raise RuntimeError("A query-only read mutated persona tensor values")
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def save_reload(self, state):
        # Three fixed temporary identities force the real manager through disk
        # without loading a second model or retaining one persona per episode.
        manager = self.manager
        manager.update_user_state("memory-eval-a", state, auto_save=True)
        manager.get_user_state("memory-eval-b")
        manager.get_user_state("memory-eval-c")
        restored = manager.get_user_state("memory-eval-a")
        return self.clone(restored)


def evaluate_production_turns(manager, rows, *, maximum=2, max_new_tokens=32):
    """Real generated acknowledgments and guards, then an empty-history query.

    Unlike the controlled arms, support replies here come from the model. No
    gold acknowledgment or retrieved support text is supplied to the query.
    """
    from engine.personalizer import uTTTPersonalizer
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError("Production smoke test requires between 1 and 8 users")
    engine = uTTTPersonalizer(manager, enable_online_ttt=True, auto_save_every_turn=True,
                             default_max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    adapter = ModelAdapter(manager, max_new_tokens=max_new_tokens, apply_guards=True)
    results = []
    for index in balanced_indices(rows, maximum):
        row = rows[index]
        user = "production-eval-" + str(index)
        untouched = adapter.clone(manager.get_user_state("production-other"))
        writes = []
        for text in support_texts(row):
            prompt = text.split("\n<|assistant|>", 1)[0].removeprefix("<|user|> ")
            result = engine.interact_and_evolve(user, prompt, history=[], temperature=0.0,
                                               max_new_tokens=max_new_tokens)
            writes.append({"generated_reply": result.reply, "committed": result.gate_triggered,
                           "surprise": result.surprise_loss, "step": result.step})
        written = adapter.clone(manager.get_user_state(user))
        manager.get_user_state("production-evict-b")
        manager.get_user_state("production-evict-c")
        reloaded = adapter.clone(manager.get_user_state(user))
        isolated = adapter.equal(untouched, manager.get_user_state("production-other"))
        result = engine.interact_and_evolve(user, row["query_text"], history=[], temperature=0.0,
                                           max_new_tokens=max_new_tokens)
        results.append({"episode_id": row["episode_id"], "writes": writes,
                        "reply": result.reply, "expected": row["answer"],
                        "exact_match": result.reply.strip() == row["answer"],
                        "save_reload_state_exact": adapter.equal(written, reloaded),
                        "other_user_state_unchanged": isolated})
    return {"history_messages_supplied": 0, "model_generated_support_replies": True,
            "production_guards_applied": True, "passes": sum(row["exact_match"] for row in results),
            "total": len(results), "results": results}


def evaluate(checkpoint, data_path, output, *, split="test", max_samples=64, max_new_tokens=32,
             device=None, apply_guards=False, allow_untrained_baseline=False, production_turns=0):
    import torch
    from engine.checkpoint_manager import DualTrackCheckpointManager
    from model.uttt_model import uTTTModel
    from deploy.evaluate_memory import guard_gpu
    from deploy.prepare_memory_curriculum import load_tokenizer
    checkpoint, data_path, output = (Path(value).resolve() for value in (checkpoint, data_path, output))
    if output.exists() or output in (checkpoint, data_path, data_path.parent / "manifest.json"):
        raise FileExistsError("Choose a fresh report path outside model and curriculum artifacts")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if type(production_turns) is not int or not 0 <= production_turns <= 8:
        raise ValueError("production_turns must be between 0 and 8")
    identity = file_hash(checkpoint)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA evaluation is supported")
    gpu = guard_gpu(torch, device, checkpoint.stat().st_size) if device.type == "cuda" else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="persona-memory-eval-", dir=output.parent) as directory:
        # Both arms retain original FP32 parameters and use BF16 CUDA GEMMs.
        # Compatible meta serving preserves masters even with BF16 activation
        # dtype. A language-stage diagnostic must explicitly request FP32 load.
        header = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        metadata = header.get("metrics", {})
        compatible_meta = (metadata.get("training_stage") == "meta"
                           and metadata.get("inner_loop_protocol") == uTTTModel.INNER_LOOP_PROTOCOL)
        del header
        manager = DualTrackCheckpointManager(checkpoint, personas_dir=Path(directory) / "personas", device=str(device),
                                            dtype=torch.bfloat16 if compatible_meta and device.type == "cuda" else torch.float32,
                                            max_gpu_users=1, max_cpu_users=1, auto_create_base_if_missing=False)
        metadata = manager.training_metadata
        trained = metadata.get("training_stage") == "meta" and metadata.get("inner_loop_protocol") == uTTTModel.INNER_LOOP_PROTOCOL
        if not trained and not allow_untrained_baseline:
            raise ValueError("A compatible meta checkpoint is required, or explicit untrained-baseline diagnostic opt-in")
        # Use the checkpoint's actual BPE, not a guessed or fixture tokenizer.
        tokenizer_json = manager.config.tokenizer_json
        if not tokenizer_json:
            raise ValueError("Checkpoint does not contain its BPE tokenizer")
        tokenizer_path = Path(directory) / "tokenizer.json"
        tokenizer_path.write_text(tokenizer_json, encoding="utf-8")
        backend, tokenizer_hash = load_tokenizer(tokenizer_path)
        rows, manifest, provenance = load_episodes(data_path, backend, tokenizer_hash, split=split,
                                                 context_limit=manager.config.max_seq_len)
        manager.model.eval().requires_grad_(False)
        versions = {name: value._version for name, value in manager.model.named_parameters()}
        adapter = ModelAdapter(manager, max_new_tokens=max_new_tokens, apply_guards=apply_guards)
        report = evaluate_with_adapter(rows, adapter, max_samples=max_samples,
                                       progress=lambda row: print("PERSONA_EVAL " + json.dumps(row), flush=True))
        if production_turns:
            report["production_turns"] = evaluate_production_turns(
                manager, rows, maximum=production_turns, max_new_tokens=max_new_tokens)
        if any(value._version != versions[name] for name, value in manager.model.named_parameters()):
            raise RuntimeError("Evaluation mutated frozen checkpoint parameters")
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=identity, meta_trained=trained,
                      untrained_baseline=not trained, data_path=str(data_path), **provenance,
                      tokenizer_sha256=tokenizer_hash, max_new_tokens=max_new_tokens, decoding="raw_greedy",
                      stability_guards_applied=apply_guards,
                      write_mode="guarded_persona_update" if apply_guards else "raw_analytical_mechanism_probe",
                      support_writes=len(adapter.write_metrics),
                      committed_support_writes=sum(row["committed"] for row in adapter.write_metrics),
                      mean_support_surprise=sum(row["surprise"] for row in adapter.write_metrics) / max(1, len(adapter.write_metrics)))
        if gpu is not None:
            report["gpu_memory"] = gpu | {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if file_hash(checkpoint) != identity or file_hash(data_path) != provenance["data_sha256"]:
        raise RuntimeError("Checkpoint or evaluated data changed during the diagnostic")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--apply-guards", action="store_true")
    parser.add_argument("--allow-untrained-baseline", action="store_true")
    parser.add_argument("--production-turns", type=int, default=0)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.data_path, args.output, split=args.split, max_samples=args.max_samples,
                      max_new_tokens=args.max_new_tokens, device=args.device, apply_guards=args.apply_guards,
                      allow_untrained_baseline=args.allow_untrained_baseline, production_turns=args.production_turns)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

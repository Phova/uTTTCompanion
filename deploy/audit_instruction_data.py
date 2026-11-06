"""Read-only BPE audit of SFT conditioning, lengths and repeated answer templates.

Reports aggregate counts and hashes, never raw questions or answers. The seeded
reservoir covers the entire file without retaining every conversation in RAM.
Run on CPU; this tool does not construct a model or alter training/checkpoints.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.tokenizer import BPETokenizer
from train_5090 import PersonaMetaDataset


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    result = {f"p{p}": ordered[round((len(ordered) - 1) * p / 100)]
              for p in (0, 50, 90, 95, 99, 100)}
    result["mean"] = sum(ordered) / len(ordered)
    return result


def role_structure(text):
    roles = re.findall(r"<\|(user|assistant)\|>", text)
    return bool(roles) and roles[0] == "user" and roles[-1] == "assistant" and all(
        left != right for left, right in zip(roles, roles[1:]))


def legacy_window(ids, labels, seq_len, rng):
    """Mirror the previous uniform answer-token anchor/random start policy."""
    if len(ids) <= seq_len:
        return 0, ids, labels
    targets = [i for i, value in enumerate(labels) if value != -100 and i > 0]
    if targets:
        anchor = targets[rng.randrange(len(targets))]
        low = max(0, anchor - seq_len + 1)
        high = min(anchor - 1, len(ids) - seq_len)
        start = rng.randrange(low, high + 1)
    else:
        start = 0
    return start, ids[start:start + seq_len], labels[start:start + seq_len]


def first_target_has_question(start, labels, turns):
    target = next((start + i for i, value in enumerate(labels) if i > 0 and value != -100), None)
    if target is None:
        return None
    turn = next((turn for turn in turns if turn[1] < target < turn[2]), None)
    return turn is not None and start <= turn[0] < turn[1] < target


def audit(data_path, tokenizer_path, seq_len=1024, sample_size=10000, seed=42, trials=3):
    if seq_len < 5 or sample_size < 1 or trials < 1:
        raise ValueError("seq_len >= 5, sample_size >= 1 and trials >= 1 are required")
    data_path, tokenizer_path = Path(data_path), Path(tokenizer_path)
    tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
    tokenizer = BPETokenizer(tokenizer_json, 65536)
    dataset = object.__new__(PersonaMetaDataset)
    dataset.tokenizer, dataset.seq_len, dataset.mode, dataset.split = tokenizer, seq_len, "sft", "validation"
    seen, reservoir, sources, counts = set(), [], Counter(), Counter()
    data_hash, reservoir_rng = hashlib.sha256(), random.Random(seed)
    with data_path.open("rb") as stream:
        for raw in stream:
            data_hash.update(raw)
            if not raw.strip():
                continue
            counts["rows"] += 1
            try:
                row = json.loads(raw)
                text = row.get("text") or row.get("content")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("missing text")
                text = re.sub(r"^<\|user\|> \[Context Category:[^\n]*\n", "", text).strip()
            except (json.JSONDecodeError, ValueError, AttributeError, UnicodeDecodeError):
                counts["invalid_rows"] += 1
                continue
            identity = hashlib.sha256(text.encode()).digest()
            if identity in seen:
                counts["exact_duplicate_rows"] += 1
                continue
            seen.add(identity)
            counts["unique_rows"] += 1
            source = row.get("source_file") or row.get("source") or "unspecified"
            sources[str(source)] += 1
            counts["invalid_role_order_rows"] += not role_structure(text)
            item = (text, str(source))
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                index = reservoir_rng.randrange(counts["unique_rows"])
                if index < sample_size:
                    reservoir[index] = item
    lengths, question_lengths, answer_lengths, ratios = [], [], [], []
    prefixes, answers, prefix_sources = Counter(), Counter(), {}
    sampled, old, new, source_sampled = Counter(), Counter(), Counter(), Counter()
    crop_rng = random.Random(seed + 1)
    for text, source in reservoir:
        source_sampled[source] += 1
        ids, labels = dataset._document(text)
        turns = dataset._sft_turns(ids, labels)
        lengths.append(len(ids))
        sampled["long_rows"] += len(ids) > seq_len
        sampled["no_answered_user_turn"] += not turns
        sampled["no_supervised_targets"] += not any(value != -100 for value in labels[1:])
        for user, assistant, end in turns:
            question_lengths.append(assistant - user - 1)
            answer_lengths.append(end - assistant - 1)
            sampled["answered_turns"] += 1
            sampled["short_question_and_answer_turns"] += assistant - user - 1 <= 64 and end - assistant - 1 <= 64
            # Hash normalized decoded answer prefixes to expose concentration
            # without including potentially identifying content in the report.
            answer = tokenizer.decode(ids[assistant + 1:end])
            normalized = " ".join(answer.split())
            prefix_hash = hashlib.sha256(normalized[:64].encode()).hexdigest()
            prefixes[prefix_hash] += 1
            answers[hashlib.sha256(normalized.encode()).hexdigest()] += 1
            prefix_sources.setdefault(prefix_hash, Counter())[source] += 1
        for _ in range(trials):
            start, old_ids, old_labels = legacy_window(ids, labels, seq_len, crop_rng)
            old["windows"] += 1
            old["missing_user_role"] += tokenizer.USER_ID not in old_ids
            old["missing_assistant_role"] += tokenizer.ASSISTANT_ID not in old_ids
            conditioned = first_target_has_question(start, old_labels, turns)
            old["first_target_without_its_question"] += conditioned is False
            old["windows_without_targets"] += conditioned is None
            if len(ids) > seq_len:
                old["long_windows"] += 1
                old["long_first_target_without_its_question"] += conditioned is False
        new_ids, new_labels = (dataset._sft_window(ids, labels, text) if len(ids) > seq_len else (ids, labels))
        new_turns = dataset._sft_turns(new_ids, new_labels)
        conditioned = first_target_has_question(0, new_labels, new_turns)
        new["windows"] += 1
        new["first_target_without_its_question"] += conditioned is False
        new["windows_without_targets"] += conditioned is None
        new["long_answer_truncated_without_eos"] += len(ids) > seq_len and any(
            value != -100 for value in new_labels[1:]) and new_ids[-1] != tokenizer.EOS_ID
        ratios.append(sum(value != -100 for value in new_labels[1:]) / seq_len)
    return {
        "format_version": 1, "dataset_sha256": data_hash.hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_json.encode()).hexdigest(),
        "vocab_size": tokenizer.vocab_size, "seq_len": seq_len,
        "sample_size_requested": sample_size, "sampled_rows": len(reservoir), "seed": seed,
        "legacy_trials_per_row": trials, "sft_window_protocol": dataset.SFT_WINDOW_PROTOCOL,
        "rows": dict(counts), "sources": dict(sources), "sampled_sources": dict(source_sampled),
        "sampled": dict(sampled), "document_tokens": quantiles(lengths),
        "question_tokens": quantiles(question_lengths), "answer_tokens": quantiles(answer_lengths),
        "new_supervised_targets_per_padded_token": quantiles(ratios),
        "legacy_windows": dict(old), "conditioned_windows": dict(new),
        "top_answer_prefixes": [{"sha256": key, "turns": value, "sources": dict(prefix_sources[key])}
                                for key, value in prefixes.most_common(12)],
        "top_complete_answers": [{"sha256": key, "turns": value} for key, value in answers.most_common(12)],
        "notes": ["All rows contribute to source, role and duplicate counts; token statistics use a seeded reservoir.",
                  "Repeated prefixes are normalized first-64-character answer hashes; no raw content is emitted.",
                  "The former random crop is estimated by independent seeded trials, not replayed from a training cursor.",
                  "Conditioning checks preserve role boundaries; they do not establish question relevance or factual accuracy."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-trials", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.data_path, args.tokenizer, args.seq_len, args.sample_size, args.seed, args.legacy_trials)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        # Exclusive creation prevents an audit invocation from overwriting an
        # existing report or either of the supplied dataset/tokenizer files.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report + "\n")
    print(report, flush=True)

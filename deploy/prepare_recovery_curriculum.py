"""Build a reproducible short-instruction supplement, with a separate heldout set.

This curriculum supplies verifiable basic instruction practice. It does not
replace broad language pretraining or establish general assistant quality.
Only the optional token-length measurement requires the tokenizers package.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import re
import string
import unicodedata


PROTOCOL = "uttt-recovery-curriculum-v1"
SOURCE = "uttt:programmatic-short-instructions-v1"
SPECIAL = re.compile(r"<\|[^<>]*\|>")
ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
TASKS = ("add", "subtract", "multiply", "copy", "uppercase", "lowercase", "reverse", "length", "sort", "extract")
# These remain independent probes, rather than examples inserted to improve
# the existing four-task score. Equivalent generated 1+1 addition is excluded.
RESERVED_PROBES = (
    "Hello", "翻译成中文，只输出译文：hello", "1+1等于多少？只输出数字。",
    "请只回答：你好", "Translate to English, output only the translation: 你好",
)

# Each pair contains a training wording and an unseen heldout wording.
TEMPLATES = {
    "add": {
        "en": ("Calculate {a} + {b}. Return only the integer.", "Give only the sum of {a} and {b}; do not explain."),
        "zh": ("计算 {a} + {b}，仅输出整数。", "把 {a} 与 {b} 相加，回复结果数字即可。"),
    },
    "subtract": {
        "en": ("Calculate {a} - {b}. Return only the integer.", "Subtract {b} from {a}; answer with the number alone."),
        "zh": ("计算 {a} - {b}，仅输出整数。", "从 {a} 中减去 {b}，只回复所得数字。"),
    },
    "multiply": {
        "en": ("Calculate {a} * {b}. Return only the integer.", "What is the product of {a} and {b}? Give only the number."),
        "zh": ("计算 {a} × {b}，仅输出整数。", "求 {a} 与 {b} 的乘积，答案只需一个数字。"),
    },
    "copy": {
        "en": ("Copy the text inside brackets exactly, without brackets: [{text}]", "Repeat precisely the following string, adding nothing: {text}"),
        "zh": ("原样复制方括号内的字符串，不输出括号：[{text}]", "将以下字符串逐字照写，不添加说明：{text}"),
    },
    "uppercase": {
        "en": ("Convert to uppercase; output only the result: {text}", "Write the capital-letter version of this string and nothing else: {text}"),
        "zh": ("将字符串转成大写，只输出结果：{text}", "把下面英文字母全部改为大写，直接给出转换后的字符串：{text}"),
    },
    "lowercase": {
        "en": ("Convert to lowercase; output only the result: {text}", "Write the small-letter version of this string and nothing else: {text}"),
        "zh": ("将字符串转成小写，只输出结果：{text}", "把下面英文字母全部改为小写，直接给出转换后的字符串：{text}"),
    },
    "reverse": {
        "en": ("Reverse the character order; output only the string: {text}", "Read this string from its last character to its first, with no explanation: {text}"),
        "zh": ("将字符串的字符顺序反转，只输出结果：{text}", "从最后一个字符到第一个字符依次写出以下字符串，别加其他内容：{text}"),
    },
    "length": {
        "en": ("Count the characters in [{text}]. Output only the integer, excluding brackets.", "How many characters does this string contain? Answer only with its length: {text}"),
        "zh": ("统计方括号内的字符个数，仅输出整数，不计括号：[{text}]", "下面字符串有多少个字符？只回答长度数字：{text}"),
    },
    "sort": {
        "en": ("Sort these integers ascending. Return a JSON array: {values}", "Arrange the following numbers from smallest to largest; give only a JSON list: {values}"),
        "zh": ("将下列整数从小到大排序，只返回 JSON 数组：{values}", "把这些数字按升序排列，答案仅保留 JSON 列表：{values}"),
    },
    "extract": {
        "en": ("From this JSON object, return only the value of {field}, without quotes: {object}", "Read the {field} field of the following object; print its value alone: {object}"),
        "zh": ("读取 JSON 对象中 {field} 字段的值，只输出值，不带引号：{object}", "下面对象的 {field} 字段是什么？直接输出字段内容：{object}"),
    },
}


def stable_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_prompt(value):
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def parse_conversation(text):
    """Accept alternating nonempty user/assistant turns, each answer with EOS.

    Unexpected special tokens and role markers inside content are rejected;
    malformed conversations are never repaired by silently dropping turns.
    """
    if not isinstance(text, str) or not text.strip() or "\x00" in text or "\ufffd" in text:
        raise ValueError("invalid_text")
    markers = list(SPECIAL.finditer(text))
    if not markers or text[:markers[0].start()].strip():
        raise ValueError("missing_initial_role")
    expected, turns, pending = "<|user|>", [], None
    for index, marker in enumerate(markers):
        tag = marker.group()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        content = text[marker.end():end]
        if tag != expected:
            raise ValueError("role_or_boundary_order")
        if expected == "<|user|>":
            if not content.strip():
                raise ValueError("empty_user")
            pending = content.strip()
            expected = "<|assistant|>"
        elif expected == "<|assistant|>":
            if not content.strip():
                raise ValueError("empty_assistant")
            turns.append((pending, content.strip()))
            expected = "<|eos|>"
        else:
            if content.strip():
                raise ValueError("content_after_eos")
            expected = "<|user|>"
    if expected != "<|user|>" or not turns:
        raise ValueError("incomplete_conversation")
    return turns


def answer_for(task, arguments):
    """Reference execution also used to verify every generated record."""
    if task == "add":
        return str(arguments["a"] + arguments["b"])
    if task == "subtract":
        return str(arguments["a"] - arguments["b"])
    if task == "multiply":
        return str(arguments["a"] * arguments["b"])
    if task == "copy":
        return arguments["text"]
    if task == "uppercase":
        return arguments["text"].upper()
    if task == "lowercase":
        return arguments["text"].lower()
    if task == "reverse":
        return arguments["text"][::-1]
    if task == "length":
        return str(len(arguments["text"]))
    if task == "sort":
        return json.dumps(sorted(arguments["values"]), separators=(",", ":"))
    if task == "extract":
        return arguments["object"][arguments["field"]]
    raise ValueError("Unknown generated task")


def payload_partition(payload, seed):
    return "heldout" if int(stable_hash(f"{seed}:{payload.casefold()}")[:12], 16) % 5 == 0 else "train"


def generate_synthetic(count, heldout_count, seed):
    rng = random.Random(seed)
    seen_keys, seen_prompts, seen_payloads = set(), set(), set()
    rows = {"train": [], "heldout": []}
    maximum = max(99, math.ceil(math.sqrt(max(count, heldout_count) / len(TASKS) * 5)))

    def payload(split, length_range=(6, 16)):
        for _ in range(10000):
            value = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(*length_range)))
            if value.casefold() not in seen_payloads and payload_partition(value, seed) == split:
                seen_payloads.add(value.casefold())
                return value
        raise RuntimeError("Unable to allocate a disjoint payload")

    for split, requested in (("train", count), ("heldout", heldout_count)):
        pool = [number for number in range(maximum + 1) if (number % 5 == 4) == (split == "heldout")]
        for index in range(requested):
            task, language = TASKS[(index // 2) % len(TASKS)], ("en" if index % 2 == 0 else "zh")
            for _ in range(20000):
                operands, payloads = [], []
                if task in {"add", "subtract", "multiply"}:
                    a, b = rng.choice(pool), rng.choice(pool)
                    if task in {"add", "multiply"}:
                        a, b = sorted((a, b))
                    if task == "add" and a == b == 1:
                        continue
                    arguments, operands = {"a": a, "b": b}, [a, b]
                elif task == "sort":
                    values = rng.sample(pool, rng.randint(2, 6))
                    arguments, operands = {"values": values}, values
                elif task == "extract":
                    obj = {key: payload(split) for key in ("name", "color", "code")}
                    arguments = {"object": obj, "field": rng.choice(tuple(obj))}
                    payloads = list(obj.values())
                else:
                    value = payload(split, (3, 24) if task == "length" else (6, 16))
                    arguments, payloads = {"text": value}, [value]
                semantic = json.dumps({"task": task, "arguments": arguments}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                key = stable_hash(semantic)
                display = {**arguments}
                if task == "sort":
                    display["values"] = json.dumps(arguments["values"], separators=(",", ":"))
                if task == "extract":
                    display["object"] = json.dumps(arguments["object"], separators=(",", ":"))
                template_index = int(split == "heldout")
                prompt = TEMPLATES[task][language][template_index].format(**display)
                identity = normalized_prompt(prompt)
                if key in seen_keys or identity in seen_prompts:
                    continue
                seen_keys.add(key)
                seen_prompts.add(identity)
                answer = answer_for(task, arguments)
                row = {"text": f"<|user|> {prompt}\n<|assistant|> {answer}<|eos|>",
                       "source": SOURCE, "copyright": "Programmatically generated by this repository; no external text copied.",
                       "protocol": PROTOCOL, "task": task, "language": language, "split": split,
                       "semantic_key": key, "arguments": arguments, "operands": operands, "payloads": payloads,
                       "template_id": f"{task}.{language}.{split}", "prompt": prompt, "answer": answer}
                if parse_conversation(row["text"]) != [(prompt, answer)]:
                    raise AssertionError("Generated conversation does not match its reference")
                rows[split].append(row)
                break
            else:
                raise RuntimeError(f"Could not generate enough distinct {task} examples")
    return rows


def verify_generated_splits(train, heldout):
    report = {}
    for field in ("semantic_key", "prompt", "template_id", "operands", "payloads"):
        def values(rows):
            if field in {"operands", "payloads"}:
                return {str(value).casefold() for row in rows for value in row[field]}
            return {normalized_prompt(row[field]) if field == "prompt" else row[field] for row in rows}
        overlap = values(train) & values(heldout)
        report[field + "_overlap"] = len(overlap)
        if overlap:
            raise ValueError(f"Generated split leakage in {field}")
    for row in [*train, *heldout]:
        if row["answer"] != answer_for(row["task"], row["arguments"]):
            raise ValueError("Programmatic answer verification failed")
    return report


def allocate_quotas(capacities, requested, balanced=False):
    """Deterministic bounded apportionment, preserving small source buckets."""
    capacities = {key: value for key, value in capacities.items() if value > 0}
    target = min(requested, sum(capacities.values()))
    if not capacities or target <= 0:
        return {key: 0 for key in capacities}
    keys = sorted(capacities)
    minimum = 1 if target >= len(keys) else 0
    quotas = {key: minimum for key in keys}
    remaining = target - sum(quotas.values())
    weights = {key: (1 if balanced else capacities[key]) for key in keys}
    weight_sum = sum(weights.values())
    ideals = {key: remaining * weights[key] / weight_sum for key in keys}
    for key in keys:
        quotas[key] += min(capacities[key] - quotas[key], math.floor(ideals[key]))
    remaining = target - sum(quotas.values())
    order = sorted(keys, key=lambda key: (-(ideals[key] % 1), key))
    while remaining:
        for key in order:
            if quotas[key] < capacities[key]:
                quotas[key] += 1
                remaining -= 1
                if not remaining:
                    break
    return quotas


def public_bucket(row, text):
    source = str(row.get("source", "unknown"))
    source_file = str(row.get("source_file", ""))
    language = row.get("language")
    if language not in {"zh", "en"}:
        lowered = source.casefold()
        language = ("zh" if "coig" in lowered else "en" if "smoltalk" in lowered else
                    "zh" if any("\u4e00" <= character <= "\u9fff" for character in text[:500]) else "en")
    return language, source, source_file


def select_existing(path, limit, seed, heldout, tokenizer=None, max_tokens=1024):
    """Keep source records unchanged, with deterministic preference for full examples.

    Public data cannot satisfy the generated subset's individual-number split:
    ordinary numbers can occur in unrelated public text. Exact user prompts
    and exact ASCII payload words belonging to heldout are excluded here.
    """
    counts, rejected = collections.Counter(), collections.Counter()
    heaps, eligible = collections.defaultdict(list), collections.Counter()
    reserved = {normalized_prompt(value) for value in RESERVED_PROBES}
    reserved.update(normalized_prompt(row["prompt"]) for row in heldout)
    heldout_payloads = {value.casefold() for row in heldout for value in row["payloads"]}
    seen_texts, seen_user_sets, input_hash = set(), set(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            input_hash.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("record_not_object")
                turns = parse_conversation(row.get("text"))
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                rejected[str(error) if not isinstance(error, json.JSONDecodeError) else "invalid_json"] += 1
                continue
            text = row["text"]
            source = str(row.get("source", "unknown"))
            counts[source] += 1
            identities = [normalized_prompt(user) for user, _ in turns]
            if any(value in reserved for value in identities):
                rejected["heldout_or_reserved_prompt"] += 1
                continue
            if any(word.casefold() in heldout_payloads for word in ASCII_WORD.findall(text)):
                rejected["heldout_payload_word"] += 1
                continue
            identity = stable_hash(text)
            user_key = stable_hash(json.dumps(identities, ensure_ascii=False))
            if identity in seen_texts or user_key in seen_user_sets:
                rejected["duplicate_text_or_user_sequence"] += 1
                continue
            seen_texts.add(identity)
            seen_user_sets.add(user_key)
            bucket = public_bucket(row, text)
            eligible[bucket] += 1
            length = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2 if tokenizer else None
            # The text-length heuristic is labeled as such, never reported as
            # a guarantee that an example fits the training token budget.
            complete = length <= max_tokens if length is not None else len(text) <= max_tokens
            rank = int(stable_hash(f"{seed}:{identity}"), 16)
            priority = (int(not complete), rank)
            item = (-priority[0], -priority[1], line_number, row, length, len(turns))
            heap = heaps[bucket]
            if limit and len(heap) < limit:
                heapq.heappush(heap, item)
            elif limit and (item[0], item[1]) > (heap[0][0], heap[0][1]):
                heapq.heapreplace(heap, item)
    language_capacities = collections.Counter()
    for bucket, count in eligible.items():
        language_capacities[bucket[0]] += count
    language_quotas = allocate_quotas(language_capacities, limit, balanced=True)
    source_quotas = {}
    for language, quota in language_quotas.items():
        source_quotas.update(allocate_quotas({key: count for key, count in eligible.items() if key[0] == language}, quota))
    selected, bucket_stats = [], []
    for bucket, quota in sorted(source_quotas.items()):
        chosen = sorted(heaps[bucket], key=lambda item: (-item[0], -item[1]))[:quota]
        selected.extend(chosen)
        bucket_stats.append({"language": bucket[0], "source": bucket[1], "source_file": bucket[2],
                             "eligible": eligible[bucket], "quota": quota, "selected": len(chosen),
                             "complete_within_token_budget": sum(item[4] is not None and item[4] <= max_tokens for item in chosen)})
    selected.sort(key=lambda item: (-item[0], -item[1]))
    return [item[3] for item in selected], {
        "input_sha256": input_hash.hexdigest(), "valid_records_by_source": dict(sorted(counts.items())),
        "rejected": dict(sorted(rejected.items())), "selected": len(selected),
        "selected_by_source": dict(sorted(collections.Counter(str(item[3].get("source", "unknown")) for item in selected).items())),
        "selected_by_language": dict(sorted(collections.Counter(public_bucket(item[3], item[3]["text"])[0] for item in selected).items())),
        "source_quotas": bucket_stats,
        "selected_single_turn": sum(item[5] == 1 for item in selected),
        "selected_multi_turn": sum(item[5] > 1 for item in selected),
        "complete_within_token_budget": sum(item[4] is not None and item[4] <= max_tokens for item in selected),
        "over_token_budget": sum(item[4] is not None and item[4] > max_tokens for item in selected),
        "token_lengths_measured": tokenizer is not None,
        "selection_priority": "balanced_languages_then_proportional_source_files_then_" + (
            "complete_token_length_then_seeded_hash" if tokenizer else "character_length_heuristic_then_seeded_hash"),
    }


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "records": len(rows), "bytes": path.stat().st_size}


def prepare(existing, output, tokenizer=None, seed=42, synthetic_count=12000, real_count=36000,
            heldout_count=None, max_tokens=1024, eval_count=60):
    existing, output = Path(existing), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output directory: {output}")
    if not existing.is_file():
        raise FileNotFoundError(existing)
    if synthetic_count < 20 or real_count < 0 or max_tokens < 16 or eval_count < 20:
        raise ValueError("Require synthetic-count >= 20, real-count >= 0, max-tokens >= 16, eval-count >= 20")
    heldout_count = max(20, synthetic_count // 10) if heldout_count is None else heldout_count
    if heldout_count < 20:
        raise ValueError("Require heldout-count >= 20 for task/language coverage")
    backend, tokenizer_hash = None, None
    if tokenizer is not None:
        from tokenizers import Tokenizer
        tokenizer_path = Path(tokenizer)
        tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        backend = Tokenizer.from_file(str(tokenizer_path))
        backend.no_padding()
        backend.no_truncation()
    synthetic = generate_synthetic(synthetic_count, heldout_count, seed)
    isolation = verify_generated_splits(synthetic["train"], synthetic["heldout"])
    real, real_stats = select_existing(existing, real_count, seed, synthetic["heldout"], backend, max_tokens)
    train = [*synthetic["train"], *real]
    random.Random(seed + 1).shuffle(train)
    monitor_groups = collections.defaultdict(list)
    for row in synthetic["heldout"]:
        monitor_groups[(row["task"], row["language"])].append(row)
    monitor_quotas = allocate_quotas({key: len(rows) for key, rows in monitor_groups.items()}, eval_count, balanced=True)
    monitor = []
    for group, rows in sorted(monitor_groups.items()):
        rows = sorted(rows, key=lambda row: stable_hash(f"{seed}:monitor:{row['semantic_key']}"))
        monitor.extend(rows[:monitor_quotas[group]])
    eval_tasks = [{"prompt": row["prompt"], "accepted": [row["answer"]], "mode": "chat",
                   "task": row["task"], "language": row["language"], "semantic_key": row["semantic_key"],
                   "template_id": row["template_id"], "arguments": row["arguments"]} for row in monitor]
    # All validation finishes before publishing; the manifest is written last
    # as the completion marker. Existing datasets are never opened for writing.
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, rows in (("train.jsonl", train), ("heldout.jsonl", synthetic["heldout"]), ("eval_tasks.jsonl", eval_tasks)):
        files[name] = write_jsonl(output / name, rows)
    source_manifest = existing.parent / "sources.json"
    if source_manifest.is_file():
        content = source_manifest.read_bytes()
        with (output / "upstream_sources.json").open("xb") as destination:
            destination.write(content)
        files["upstream_sources.json"] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    report = {"protocol": PROTOCOL, "status": "complete", "seed": seed,
              "existing_file": str(existing.resolve()), "existing": real_stats,
              "tokenizer_sha256": tokenizer_hash, "max_tokens": max_tokens,
              "train_records": len(train), "synthetic_train_records": len(synthetic["train"]),
              "public_train_records": len(real), "heldout_records": len(synthetic["heldout"]),
              "monitor_eval_records": len(eval_tasks),
              "monitor_by_task_language": {task + ":" + language: count for (task, language), count in sorted(monitor_quotas.items())},
              "sources": dict(sorted(collections.Counter(str(row.get("source", "unknown")) for row in train).items())),
              "synthetic_train_by_task_language": dict(sorted(collections.Counter(row["task"] + ":" + row["language"] for row in synthetic["train"]).items())),
              "synthetic_heldout_by_task_language": dict(sorted(collections.Counter(row["task"] + ":" + row["language"] for row in synthetic["heldout"]).items())),
              "generated_split_checks": isolation, "files": files,
              "notes": ["Every synthetic answer is recomputed from its arguments before publication.",
                        "Generated train/heldout have disjoint semantic keys, prompts, individual operands, payloads and wording templates.",
                        "Public records preserve their original fields, including source and copyright.",
                        "Public exclusion checks cover exact normalized prompts and exact ASCII payload words; broader public semantic or numeric contamination is not claimed absent.",
                        "heldout.jsonl and eval_tasks.jsonl are evaluation-only; never concatenate them into training.",
                        "eval_tasks.jsonl is a task/language-stratified monitoring subset of heldout.jsonl, not the full heldout or a general-quality evaluation.",
                        "Public selection balances Chinese and English where available, then apportions each language across source_file buckets; length preference applies within each source quota.",
                        "This is a basic-instruction supplement, not a replacement for broad pretraining or a general-quality benchmark."]}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True, help="Prepared public instruction train.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--tokenizer", type=Path, help="Optional actual BPE JSON for complete-example preference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-count", type=int, default=12000)
    parser.add_argument("--real-count", type=int, default=36000)
    parser.add_argument("--heldout-count", type=int)
    parser.add_argument("--eval-count", type=int, default=60, help="Task/language-stratified monitor subset of heldout")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    result = prepare(args.existing, args.output, args.tokenizer, args.seed, args.synthetic_count,
                     args.real_count, args.heldout_count, args.max_tokens, args.eval_count)
    print("RECOVERY_CURRICULUM_COMPLETE " + json.dumps({key: result[key] for key in (
        "protocol", "train_records", "synthetic_train_records", "public_train_records", "heldout_records", "monitor_eval_records", "sources")}, ensure_ascii=False), flush=True)

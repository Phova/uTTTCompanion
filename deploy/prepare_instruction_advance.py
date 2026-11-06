"""Augment verified training instructions; preserve the existing heldout bytes.

This is a new, synthetic-heavy SFT stage, not a rewrite of the recovery run.
Public examples retain their original metadata and must fit the real BPE budget.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

try:
    from .prepare_recovery_curriculum import (
        TASKS, TEMPLATES, RESERVED_PROBES, SOURCE as PARENT_SYNTHETIC_SOURCE, allocate_quotas, answer_for,
        normalized_prompt, parse_conversation, public_bucket, stable_hash,
        verify_generated_splits, write_jsonl,
    )
except ImportError:
    from prepare_recovery_curriculum import (
        TASKS, TEMPLATES, RESERVED_PROBES, SOURCE as PARENT_SYNTHETIC_SOURCE, allocate_quotas, answer_for,
        normalized_prompt, parse_conversation, public_bucket, stable_hash,
        verify_generated_splits, write_jsonl,
    )


PROTOCOL = "uttt-instruction-advance-v1"
SOURCE = "uttt:programmatic-instruction-advance-v1"

# These are training wordings. The parent's heldout wording is never included.
PARAPHRASES = {
    "add": {
        "en": ("Add {a} and {b}. Answer with just the integer.",
               "Compute the addition {a}+{b}; give the result only.",
               "Return the numerical answer to {a} plus {b}.",
               "What does {a} + {b} evaluate to? Respond with an integer.",
               "Output only the result of adding {b} to {a}.",
               "Complete this expression with one number: {a} + {b} ="),
        "zh": ("只给出数值：{a} 加 {b}。", "{a} + {b} = ? 请用一个整数作答。",
               "做一道加法题，{a} 与 {b} 的和是多少？不要过程。", "输出 {a} 加上 {b} 的结果。",
               "用数字回答这道加法：{a} + {b}。", "求和并仅写答案：{a}、{b}。"),
    },
    "subtract": {
        "en": ("What is {a} minus {b}? Give just the integer.",
               "Compute {a}-{b}. No explanation, only its result.",
               "Take {b} away from {a}; return the integer answer.",
               "Evaluate this subtraction: {a} - {b} =", "Output the numerical value of {a} minus {b}.",
               "Find the difference {a} - {b}. Reply with one number."),
        "zh": ("{a} 减 {b} 是多少？只写数值。", "求减法结果：{a}-{b}，无需解释。",
               "从 {a} 中扣掉 {b}，输出答案。", "填入一个整数：{a} - {b} =",
               "请用数字给出 {a} 减去 {b} 的值。", "算出差值，仅写结果：{a}－{b}。"),
    },
    "multiply": {
        "en": ("Multiply {a} by {b}. Reply only with the integer.",
               "Evaluate {a} times {b}; omit the explanation.",
               "Compute the multiplication {a} * {b} =", "Output the result of {a} multiplied by {b}.",
               "Find {a} times {b}. Only a number is wanted.",
               "Return one integer: {a} x {b}."),
        "zh": ("{a} 乘以 {b} 得多少？只写整数。", "计算乘法 {a} * {b}，不要列过程。",
               "填入乘法的数值答案：{a} × {b} =", "输出 {a} 与 {b} 相乘后的值。",
               "请只给一个数字：{a} 乘 {b}。", "求乘法结果，不加说明：{a}×{b}。"),
    },
    "copy": {
        "en": ("Echo this string exactly: {text}", "Repeat this text unchanged: {text}",
               "Write this text unchanged, preserving letter case: {text}",
               "Do not transform the characters; output this string: {text}",
               "Reproduce the same characters in the same order: {text}",
               "Your entire response must be the following text: {text}"),
        "zh": ("照抄字符串：{text}", "返回与输入完全相同的文字：{text}",
               "不改变大小写和顺序，复制：{text}", "不要转换字符，直接输出：{text}",
               "按原顺序原样写出以下内容：{text}", "完整回复必须与这段文字一致：{text}"),
    },
    "uppercase": {
        "en": ("Make every letter uppercase: {text}", "Change all letters to capitals: {text}",
               "Return the all-uppercase text for: {text}",
               "Use only capital letters to rewrite this text: {text}",
               "Uppercase this string, keeping its character order: {text}",
               "Replace lowercase letters with uppercase equivalents: {text}"),
        "zh": ("把所有字母变为大写：{text}", "只返回大写字母形式：{text}",
               "转换为全大写，保持字符顺序：{text}", "将小写字母换成对应的大写：{text}",
               "用大写形式重写，不加说明：{text}", "请做大写转换并仅写结果：{text}"),
    },
    "lowercase": {
        "en": ("Make every letter lowercase: {text}", "Change all capitals to lowercase letters: {text}",
               "Rewrite using small letters only; preserve order: {text}",
               "Use only lowercase letters to rewrite this text: {text}",
               "Lowercase this string, keeping its character order: {text}",
               "Replace uppercase letters with lowercase equivalents: {text}"),
        "zh": ("把所有字母变为小写：{text}", "只返回小写字母形式：{text}",
               "转换为全小写，保持字符顺序：{text}", "将大写字母换成对应的小写：{text}",
               "用小写形式重写，不加说明：{text}", "请做小写转换并仅写结果：{text}"),
    },
    "reverse": {
        "en": ("Write these characters backwards: {text}", "Reverse this text character by character: {text}",
               "Return the characters in reverse order: {text}",
               "Put the final character first and the first character last: {text}",
               "Output a reversed copy of this string: {text}",
               "Keep letter case and invert the character sequence: {text}"),
        "zh": ("把这些字符倒着写：{text}", "逐字符反转文本：{text}",
               "以相反的字符顺序输出：{text}", "把末尾字符放到开头，依次倒序：{text}",
               "返回字符串的倒序副本：{text}", "保持大小写，将字符排列顺序颠倒：{text}"),
    },
    "length": {
        "en": ("Count the characters inside [{text}]. Give only the count.",
               "Return the character count as an integer: {text}",
               "Report the number of characters, not the text itself: {text}",
               "Measure this string's length in characters: {text}",
               "Give one integer equal to the length of: {text}",
               "Count each character once; output the total only: {text}"),
        "zh": ("数一数字符数量，只写个数：{text}", "返回字符串的字符长度整数：{text}",
               "请给字符总数，不要复述原文：{text}", "这段文字的字符数量是多少：{text}",
               "用一个整数表示以下字符串长度：{text}", "逐个计数字符，只输出总数：{text}"),
    },
    "sort": {
        "en": ("Sort in increasing order and output a JSON array: {values}",
               "Return these numbers in ascending numerical order: {values}. Use JSON.",
               "Order from least to greatest, as a JSON array: {values}",
               "Produce a JSON list of the integers sorted low to high: {values}",
               "Reorder into ascending order without adding numbers: {values}",
               "Give only the ascending JSON array for: {values}"),
        "zh": ("按数值递增排序，输出 JSON 数组：{values}",
               "返回这些整数的升序 JSON 表示：{values}",
               "从小到大重排，只写数组：{values}",
               "将整数按递增顺序排好，不增删数字：{values}",
               "请输出排序后的升序 JSON 数组：{values}",
               "给出由小到大的数字列表：{values}"),
    },
    "extract": {
        "en": ("Extract field {field} from {object}; return just its value.",
               "Look up key {field} and output its unquoted value: {object}",
               "Output the text stored under {field}: {object}",
               "Find {field} in this JSON and return only its contents: {object}",
               "Give the value corresponding to JSON key {field}: {object}",
               "Read JSON key {field}; no quotes or explanation: {object}"),
        "zh": ("从 {object} 提取 {field} 的值，仅写内容。",
               "查找键 {field} 并输出不带引号的值：{object}",
               "返回 {field} 键下存储的文字：{object}",
               "找到 JSON 中的 {field}，只写对应内容：{object}",
               "请给出键 {field} 对应的字段值：{object}",
               "读取键 {field}，不要引号或解释：{object}"),
    },
}


def read_jsonl(path):
    # Iterating the stream preserves valid U+2028/U+0085 inside JSON strings.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_arguments(row):
    values = dict(row["arguments"])
    if row["task"] == "sort":
        values["values"] = json.dumps(values["values"], separators=(",", ":"))
    elif row["task"] == "extract":
        values["object"] = json.dumps(values["object"], ensure_ascii=False, separators=(",", ":"))
    return values


def augment_synthetic(rows, seed):
    """Keep each original prompt and add two balanced, train-only wordings."""
    buckets = defaultdict(list)
    for row in rows:
        if row.get("task") not in TASKS or row.get("language") not in ("en", "zh") or row.get("split") != "train":
            raise ValueError("Expected verified synthetic training records")
        expected = answer_for(row["task"], row["arguments"])
        if row.get("answer") != expected or parse_conversation(row["text"]) != [(row["prompt"], expected)]:
            raise ValueError("Synthetic answer/reference mismatch")
        buckets[(row["task"], row["language"])].append(row)
    result = []
    for (task, language), items in sorted(buckets.items()):
        pool = PARAPHRASES[task][language]
        if len(pool) < 6 or len(set(pool)) != len(pool) or TEMPLATES[task][language][1] in pool:
            raise ValueError("Training wording pool must be diverse and exclude heldout templates")
        items = sorted(items, key=lambda row: stable_hash(f"augmentation:{seed}:{row['semantic_key']}"))
        for index, row in enumerate(items):
            variants = [("original", row["prompt"])]
            for number in (index % len(pool), (index + len(pool) // 2) % len(pool)):
                variants.append((f"paraphrase_{number}", pool[number].format(**display_arguments(row))))
            for variant, prompt in variants:
                answer = answer_for(task, row["arguments"])
                result.append({**row, "text": f"<|user|> {prompt}\n<|assistant|> {answer}<|eos|>",
                               "source": SOURCE, "parent_source": row["source"], "protocol": PROTOCOL,
                               "parent_template_id": row["template_id"],
                               "validation_group_id": "synthetic:" + row["semantic_key"],
                               "template_id": f"{task}.{language}.advance.{variant}",
                               "augmentation_variant": variant, "prompt": prompt, "answer": answer})
    return result


def public_validation_group(row):
    prompts = [normalized_prompt(user) for user, _ in parse_conversation(row["text"])]
    return "public:" + stable_hash(json.dumps(prompts, ensure_ascii=False, separators=(",", ":")))


def select_public(rows, count, seed, length_fn, max_tokens, heldout):
    reserved = {normalized_prompt(value) for value in RESERVED_PROBES}
    reserved.update(normalized_prompt(row["prompt"]) for row in heldout)
    heldout_payloads = {value.casefold() for row in heldout for value in row["payloads"]}
    groups, seen, rejected = defaultdict(list), set(), Counter()
    for row in rows:
        turns = parse_conversation(row["text"])
        prompts = {normalized_prompt(user) for user, _ in turns}
        if prompts & reserved:
            rejected["reserved_or_heldout_prompt"] += 1
            continue
        # Reuse the parent's exact-word payload exclusion, without pretending
        # unrelated public numeric text has disjoint arithmetic operands.
        if any(word.casefold() in heldout_payloads for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", row["text"])):
            rejected["heldout_payload_word"] += 1
            continue
        identity = stable_hash(row["text"].strip())
        if identity in seen:
            rejected["duplicate_text"] += 1
            continue
        seen.add(identity)
        length = length_fn(row["text"])
        if length > max_tokens:
            rejected["over_token_budget"] += 1
            continue
        bucket = public_bucket(row, row["text"])
        # Domain separation matters: the trainer uses seed:text_hash for its
        # validation split. Reusing it to prefer LOW hashes biases that split.
        rank = stable_hash(f"public-selection:{seed}:{identity}")
        groups[bucket].append((rank, row, length))
    language_capacities = Counter()
    for key, items in groups.items():
        language_capacities[key[0]] += len(items)
    language_quotas = allocate_quotas(language_capacities, min(count, sum(language_capacities.values())), balanced=True)
    quotas = {}
    for language, quota in language_quotas.items():
        source_capacities = Counter()
        for key, items in groups.items():
            if key[0] == language:
                source_capacities[key[1]] += len(items)
        source_quotas = allocate_quotas(source_capacities, quota, balanced=True)
        for source, source_quota in source_quotas.items():
            capacities = {key: len(items) for key, items in groups.items() if key[:2] == (language, source)}
            quotas.update(allocate_quotas(capacities, source_quota))
    selected, bucket_stats = [], []
    for key, quota in sorted(quotas.items()):
        chosen = sorted(groups[key], key=lambda item: item[0])[:quota]
        selected.extend(chosen)
        bucket_stats.append({"language": key[0], "source": key[1], "source_file": key[2],
                             "eligible": len(groups[key]), "selected": len(chosen)})
    selected.sort(key=lambda item: item[0])
    return [item[1] for item in selected], {
        "requested": count, "selected": len(selected), "available_complete": sum(map(len, groups.values())),
        "by_language": dict(Counter(public_bucket(item[1], item[1]["text"])[0] for item in selected)),
        "by_source": dict(Counter(str(item[1].get("source", "unknown")) for item in selected)),
        "buckets": bucket_stats, "rejected": dict(rejected),
        "max_tokens_selected": max((item[2] for item in selected), default=0),
        "ranking_domain": "public-selection:seed:text_sha256 (independent of validation split)",
    }


def prepare(parent, output, tokenizer=None, *, seed=20261106, public_count=12000,
            max_tokens=512, length_fn=None, validation_fraction=.02):
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    if public_count < 0 or max_tokens < 16 or not 0 <= validation_fraction < 1:
        raise ValueError("Require nonnegative public count, max_tokens >= 16 and validation fraction in [0,1)")
    tokenizer_hash = None
    if length_fn is None:
        if tokenizer is None:
            raise ValueError("An actual BPE tokenizer is required")
        from tokenizers import Tokenizer
        path = Path(tokenizer)
        tokenizer_hash = digest(path)
        backend = Tokenizer.from_file(str(path))
        backend.no_padding()
        backend.no_truncation()
        # Conservative BOS/EOS allowance, including documents with explicit EOS.
        length_fn = lambda text: len(backend.encode(text, add_special_tokens=False).ids) + 2
    input_rows = read_jsonl(parent / "train.jsonl")
    heldout = read_jsonl(parent / "heldout.jsonl")
    monitor = read_jsonl(parent / "eval_tasks.jsonl")
    synthetic = [row for row in input_rows if row.get("source") == PARENT_SYNTHETIC_SOURCE]
    public = [row for row in input_rows if row.get("source") != PARENT_SYNTHETIC_SOURCE]
    if not synthetic or not heldout or not monitor:
        raise ValueError("Require parent synthetic training and independent heldout/monitor records")
    verify_generated_splits(synthetic, heldout)
    augmented = augment_synthetic(synthetic, seed)
    split_checks = verify_generated_splits(augmented, heldout)
    heldout_prompts = {normalized_prompt(row["prompt"]) for row in heldout}
    reserved = {normalized_prompt(value) for value in RESERVED_PROBES}
    if any(normalized_prompt(row["prompt"]) in heldout_prompts | reserved for row in augmented):
        raise ValueError("Augmented prompts overlap heldout/reserved probes")
    texts = [row["text"] for row in augmented]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate augmented training text")
    if any(parse_conversation(row["text"]) != [(row["prompt"], answer_for(row["task"], row["arguments"]))]
           for row in augmented):
        raise ValueError("Augmented text/reference mismatch")
    lengths = [length_fn(row["text"]) for row in augmented]
    if max(lengths) > max_tokens:
        raise ValueError("A synthetic instruction exceeds the token budget")
    chosen_public, public_stats = select_public(public, public_count, seed, length_fn, max_tokens, heldout)
    chosen_public = [{**row, "validation_group_id": public_validation_group(row)} for row in chosen_public]
    train = [*augmented, *chosen_public]
    random.Random(seed).shuffle(train)
    all_lengths = [*lengths, *(length_fn(row["text"]) for row in chosen_public)]
    groups = {row["validation_group_id"] for row in train}
    validation_groups = {group for group in groups if
                         int(stable_hash(f"group-validation-v1:{seed}:{group}")[:16], 16) / 2**64 < validation_fraction}
    train_groups = groups - validation_groups
    if train_groups & validation_groups or any(not row["validation_group_id"] for row in train):
        raise AssertionError("Internal group split is invalid")
    validation_rows = sum(row["validation_group_id"] in validation_groups for row in train)
    parent_hashes = {name: digest(parent / name) for name in ("train.jsonl", "heldout.jsonl", "eval_tasks.jsonl", "manifest.json")}
    output.mkdir(parents=True, exist_ok=False)
    files = {"train.jsonl": write_jsonl(output / "train.jsonl", train)}
    for name in ("heldout.jsonl", "eval_tasks.jsonl"):
        data = (parent / name).read_bytes()
        with (output / name).open("xb") as stream:
            stream.write(data)
        files[name] = {"sha256": digest(output / name), "bytes": len(data)}
        if files[name]["sha256"] != parent_hashes[name]:
            raise AssertionError("Heldout byte copy changed")
    for name in ("upstream_sources.json",):
        if (parent / name).exists():
            with (output / name).open("xb") as stream:
                stream.write((parent / name).read_bytes())
            files[name] = {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
    report = {
        "protocol": PROTOCOL, "status": "complete", "seed": seed, "max_tokens": max_tokens,
        "builder_sha256": digest(__file__),
        "reference_builder_sha256": digest(answer_for.__code__.co_filename),
        "parent": str(parent.resolve()), "parent_sha256": parent_hashes,
        "tokenizer_sha256": tokenizer_hash,
        "token_measurement": "actual_BPE_plus_conservative_BOS_EOS" if tokenizer_hash else "injected_test_length",
        "internal_validation": {"group_key": "validation_group_id", "group_hash_protocol": "group-validation-v1",
                                "seed": seed, "validation_fraction": validation_fraction,
                                "groups": len(groups), "training_groups": len(train_groups),
                                "validation_groups": len(validation_groups), "group_overlap": 0,
                                "training_rows": len(train) - validation_rows, "validation_rows": validation_rows},
        "train_records": len(train), "synthetic_train_records": len(augmented),
        "synthetic_unique_semantic_cases": len({row["semantic_key"] for row in augmented}),
        "synthetic_variants_per_parent": 3,
        "synthetic_by_task_language": dict(sorted(Counter(row["task"] + ":" + row["language"] for row in augmented).items())),
        "synthetic_by_template": dict(sorted(Counter(row["template_id"] for row in augmented).items())),
        "synthetic_token_lengths": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "all_token_lengths": {"min": min(all_lengths), "max": max(all_lengths),
                              "mean": sum(all_lengths) / len(all_lengths),
                              "rounded_128_histogram": dict(sorted(Counter(
                                  str(((value + 127) // 128) * 128) for value in all_lengths).items()))},
        "public": public_stats, "generated_split_checks": split_checks,
        "heldout_records": len(heldout), "monitor_records": len(monitor), "files": files,
        "notes": [
            "Each original synthetic TRAIN argument set has its original wording and two new train-only paraphrases; answers are recomputed.",
            "Repeated training semantic cases are intentional instruction-wording augmentation, not new arithmetic or payload identities.",
            "Parent heldout and monitor files are copied byte-for-byte; no heldout answers or wordings are inserted into training.",
            "Synthetic isolation covers semantic keys, individual operands, payloads and prompts; public exclusions cover exact prompts and payload words only.",
            "Public records retain all original metadata and are selected only when their entire conversation fits the BPE budget.",
            "Public ranking is domain-separated from the trainer's seed:text-hash validation split.",
            "Enable training.validation_group_key=validation_group_id; all synthetic paraphrases share their semantic case group, and public groups hash the normalized user sequence.",
            "Sampling is 75% synthetic when 12000 complete public examples are available; B1 training averages per-example CE, not target-token-weighted CE.",
            "This stage practices basic instruction following; it does not replace broad pretraining or establish general assistant quality.",
        ],
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--public-count", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=.02)
    args = parser.parse_args()
    result = prepare(args.parent, args.output, args.tokenizer, seed=args.seed,
                     public_count=args.public_count, max_tokens=args.max_tokens,
                     validation_fraction=args.validation_fraction)
    print("INSTRUCTION_ADVANCE_COMPLETE " + json.dumps({key: result[key] for key in (
        "protocol", "train_records", "synthetic_train_records", "synthetic_unique_semantic_cases", "heldout_records", "monitor_records")}), flush=True)

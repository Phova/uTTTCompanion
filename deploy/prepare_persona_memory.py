"""Build short, independently tokenized support/query persona-memory episodes.

train.jsonl contains explicit train and dev rows; test.jsonl is separate. These
are split-request memory exercises, not evidence that a trained model recalls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

try:
    from .prepare_memory_curriculum import PAYLOAD_ALPHABET, load_tokenizer
except ImportError:
    from prepare_memory_curriculum import PAYLOAD_ALPHABET, load_tokenizer


PROTOCOL = "persona_split_request_memory_v1"
MAX_EPISODE_WRITES = 10
TASK_CYCLE = ("recall",) * 6 + ("correction", "reply_prefix")
TEMPLATES = {
    "train": {
        "en": ("Remember this profile fact: {entity}'s private code is {value}.",
               "Update {entity}'s code to {value}; the previous code is obsolete.",
               "What code did I ask you to remember for {entity}? Reply with the code alone.",
               "For {entity}, remember the reply prefix {value}.",
               "Use my saved reply prefix for {entity}, then a space and READY. Return only that response."),
        "zh": ("请记住这条资料：{entity}的私有代码是{value}。",
               "更新：{entity}的代码改为{value}，之前的代码已作废。",
               "我让你记住的{entity}代码是什么？只输出代码。",
               "请为{entity}记住固定回复前缀{value}。",
               "使用为{entity}保存的回复前缀，后接一个空格和READY，只输出这段回复。"),
    },
    "dev": {
        "en": ("Profile card for {entity}. Current lookup code: {value}.",
               "The profile card for {entity} has a replacement code: {value}. Discard the earlier value.",
               "Retrieve {entity}'s currently valid code. Output only its exact value.",
               "The preferred response prefix on {entity}'s profile card is {value}.",
               "Apply {entity}'s stored response prefix to READY, separated by one space. No other text."),
        "zh": ("资料卡：{entity}目前登记的检索代码为{value}。",
               "修订资料卡：{entity}现行代码为{value}，请弃用原值。",
               "取回{entity}目前有效的代码，仅给出准确值。",
               "资料卡记载{entity}偏好的回答前缀为{value}。",
               "按{entity}资料中的回答前缀回应READY，二者间留一个空格，不加其他文字。"),
    },
    "test": {
        "en": ("Keep this association for later: {entity} uses code {value}.",
               "A revision for {entity}: from now on use {value} instead of the previously stored code.",
               "Which code belongs to {entity} now? Give the code and nothing else.",
               "Keep {value} as the prefix that {entity} wants at the start of a reply.",
               "Reply to READY with the prefix previously chosen by {entity}, followed by one space and READY only."),
        "zh": ("保留一项稍后查询的对应关系：{entity}使用代码{value}。",
               "现有记录需要更正：{entity}今后应采用{value}，替代先前保存的代码。",
               "现在属于{entity}的代码是哪一个？仅回答代码本身。",
               "保留{entity}希望出现在回答开头的前缀：{value}。",
               "为READY作答，使用{entity}先前指定的前缀，再接一个空格和READY，仅此而已。"),
    },
}
NEUTRAL = {
    "en": ("The library opens near the small park.", "A notebook lies beside the window.",
           "The train crosses a stone bridge.", "A gardener checks the dry soil.",
           "The baker weighs flour before mixing it.", "A quiet path follows the river.",
           "The museum labels each display.", "An editor checks a paragraph."),
    "zh": ("图书馆在小公园附近开放。", "窗边放着一本笔记本。", "列车经过一座石桥。", "园丁查看干燥的土壤。",
           "烘焙者先称面粉再混合。", "安静的小路沿着河岸延伸。", "博物馆为展品添加说明。", "编辑核对了一段文字。"),
}


def support_text(message, language):
    acknowledgment = "已记录。" if language == "zh" else "Recorded."
    return f"<|user|> {message}\n<|assistant|> {acknowledgment}<|eos|>"


def query_prefix(query):
    return f"<|user|> {query}\n<|assistant|>"


def support_texts(row):
    texts = [row["write_text"]]
    if row.get("correction_text") is not None:
        texts.append(row["correction_text"])
    texts.extend(row.get("interference_texts", []))
    return texts


def measure_episode(row, tokenizer, seq_len):
    """Actual joint BPE, one BOS and one explicit real EOS, never cropping."""
    texts = support_texts(row)
    if not 1 <= len(texts) <= MAX_EPISODE_WRITES:
        raise ValueError("An episode requires between 1 and 10 independent support writes")
    if any(not isinstance(text, str) or not text.startswith("<|user|> ") or not text.endswith("<|eos|>")
           for text in texts):
        raise ValueError("Every support must be a complete role/EOS text")
    query, answer = row.get("query_text"), row.get("answer")
    if any(not isinstance(text, str) or not text.strip() or "<|" in text or "|>" in text for text in (query, answer)):
        raise ValueError("Query and answer must be nonempty plain text without role delimiters")
    encoded_supports = [[1] + tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    prefix = query_prefix(query)
    query_ids = [1] + tokenizer.encode(prefix + " " + answer + "<|eos|>", add_special_tokens=False).ids
    generation_ids = [1] + tokenizer.encode(prefix, add_special_tokens=False).ids
    for ids in encoded_supports + [query_ids]:
        if len(ids) > seq_len:
            raise ValueError(f"Episode would require cropping: {len(ids)} BPE tokens exceeds {seq_len}")
        if ids.count(4) != 1 or ids.count(5) != 1 or ids.count(2) != 1 or ids[-1] != 2:
            raise ValueError("Each independently encoded request requires one user, assistant and final EOS")
    assistant = query_ids.index(5)
    if tokenizer.decode(query_ids[assistant + 1:-1]).strip() != answer:
        raise ValueError("Joint BPE does not reconstruct the exact answer")
    return {"support_tokens": [len(ids) for ids in encoded_supports], "query_prefix_tokens": len(generation_ids),
            "query_with_answer_tokens": len(query_ids), "answer_tokens_including_eos": len(query_ids) - assistant - 1}


def prepare(output, tokenizer_path, *, train_episodes=256, dev_episodes=32, test_episodes=32,
            payload_length=3, seq_len=128, interference_writes=0, seed=20261109):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new persona-memory curriculum directory")
    counts = {"train": train_episodes, "dev": dev_episodes, "test": test_episodes}
    if any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError("Each explicit split requires a positive episode count")
    if type(payload_length) is not int or not 2 <= payload_length <= 8 or type(seq_len) is not int or seq_len < 16:
        raise ValueError("Use a payload length from 2 to 8 and a positive request token budget")
    if type(interference_writes) is not int or not 0 <= interference_writes <= 8:
        raise ValueError("interference_writes must be from 0 to 8")
    payloads_needed = sum(count + sum(TASK_CYCLE[(index // 2) % len(TASK_CYCLE)] == "correction"
                                      for index in range(count)) for count in counts.values())
    if payloads_needed > len(PAYLOAD_ALPHABET) ** payload_length:
        raise ValueError("Not enough distinct short payloads for all splits; increase payload_length")
    tokenizer, tokenizer_hash = load_tokenizer(tokenizer_path)
    used_entities, used_payloads, rows = set(), set(), []
    for split, count in counts.items():
        rng = random.Random(f"{PROTOCOL}:{seed}:{split}")
        def fresh(used, length, forbidden=""):
            for _ in range(100000):
                value = "".join(rng.choice(PAYLOAD_ALPHABET) for _ in range(length))
                if value not in used and value.casefold() not in forbidden.casefold():
                    used.add(value)
                    return value
            raise ValueError("Not enough distinct nonleaking payloads; increase payload_length")
        for index in range(count):
            language = "zh" if index % 2 else "en"
            task = TASK_CYCLE[(index // 2) % len(TASK_CYCLE)]
            entity = "profile-" + fresh(used_entities, 8)
            write, correction, query, preference, prefix_query = TEMPLATES[split][language]
            question = (prefix_query if task == "reply_prefix" else query).format(entity=entity)
            forbidden = question + " ".join(NEUTRAL[language])
            value = fresh(used_payloads, payload_length, forbidden)
            old = fresh(used_payloads, payload_length, forbidden) if task == "correction" else None
            initial_message = (preference if task == "reply_prefix" else write).format(entity=entity, value=old or value)
            answer = value + " READY" if task == "reply_prefix" else value
            family = "persona-memory:" + hashlib.sha256(entity.encode()).hexdigest()
            metadata = {"task": task, "language": language, "entity": entity, "payload": value,
                        "obsolete_payload": old, "split": split, "prompt_family": f"{PROTOCOL}:{split}:{language}:{task}"}
            row = {"episode_id": f"{split}-{index:06d}", "family": family, "validation_group_id": family,
                   "write_text": support_text(initial_message, language), "query_text": question, "answer": answer,
                   "memory_task": metadata, "tokenizer_sha256": tokenizer_hash,
                   "source": PROTOCOL, "copyright": "Repository-generated synthetic facts and instructions."}
            if old is not None:
                row["correction_text"] = support_text(correction.format(entity=entity, value=value), language)
            neutral = list(NEUTRAL[language])
            rng.shuffle(neutral)
            row["interference_texts"] = [support_text(message, language) for message in neutral[:interference_writes]]
            if any(payload.casefold() in question.casefold() for payload in (value, old) if payload):
                raise ValueError("A generated payload accidentally occurs in its query; choose another seed")
            row["lengths"] = measure_episode(row, tokenizer, seq_len)
            rows.append(row)
    sets = {split: {"entities": {row["memory_task"]["entity"] for row in rows if row["memory_task"]["split"] == split},
                    "payloads": {value for row in rows if row["memory_task"]["split"] == split
                                 for value in (row["memory_task"]["payload"], row["memory_task"]["obsolete_payload"]) if value},
                    "prompt_families": {row["memory_task"]["prompt_family"] for row in rows if row["memory_task"]["split"] == split}}
            for split in counts}
    intersections = {f"{left}:{right}": {key: len(sets[left][key] & sets[right][key]) for key in sets[left]}
                     for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    if any(value for pair in intersections.values() for value in pair.values()):
        raise RuntimeError("Episode split identities overlap")
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, splits in (("train.jsonl", {"train", "dev"}), ("test.jsonl", {"test"})):
        selected = [row for row in rows if row["memory_task"]["split"] in splits]
        path = output / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        files[name] = {"rows": len(selected), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"protocol": PROTOCOL, "seed": seed, "seq_len": seq_len, "payload_length": payload_length,
                "tokenizer_sha256": tokenizer_hash, "files": files, "split_episodes": counts,
                "split_intersections": intersections, "interference_writes": interference_writes,
                "split_tasks": {split: dict(Counter(row["memory_task"]["task"] for row in rows if row["memory_task"]["split"] == split)) for split in counts},
                "notes": ["Explicit train/dev splits share train.jsonl; test.jsonl must never be trained on.",
                          "Each support and the query are separate calls. Only fast weights/momentum may cross the boundary.",
                          "Exact free-generation recall, mismatch, isolation and save/reload are measured separately from teacher forcing.",
                          "Reply-prefix tasks check a minimal stored preference, not broad natural conversation quality."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--dev-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=32)
    parser.add_argument("--payload-length", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--interference-writes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.tokenizer, train_episodes=args.train_episodes,
                             dev_episodes=args.dev_episodes, test_episodes=args.test_episodes,
                             payload_length=args.payload_length, seq_len=args.seq_len,
                             interference_writes=args.interference_writes, seed=args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()

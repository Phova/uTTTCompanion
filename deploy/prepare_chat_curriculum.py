"""Build an explicitly classified chat/reasoning/code curriculum without upsampling.

train.jsonl contains both training and internal-validation rows. Their split is
the exact family hash used by PersonaMetaDataset. heldout.jsonl is independently
family-isolated and must never be passed back as training data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

try:
    from .prepare_recovery_curriculum import RESERVED_PROBES, normalized_prompt, parse_conversation
except ImportError:
    from prepare_recovery_curriculum import RESERVED_PROBES, normalized_prompt, parse_conversation


PROTOCOL = "uttt-chat-curriculum-v1"
CATEGORIES = ("chat", "reasoning", "coding")
INITIAL_GENERIC_GREETINGS = frozenset({
    "你好", "您好", "嗨", "哈喽", "嘿", "你好呀", "您好呀", "在吗", "早上好", "早安", "下午好", "晚上好",
    "你好，你能帮我解答一个问题吗", "你好，能帮我解答一个问题吗", "你好，我想问你一个问题",
    "你好，我有个问题想问你", "你好，我有一个问题想请教你", "你好，可以问你一个问题吗",
    "你好，你能回答我一个问题吗", "你好，我想向你请教一个问题",
    "hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening",
    "hi, can i ask you a question", "hello, can you help me with a question",
})


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render_turns(turns):
    return "\n".join(f"<|user|> {user}\n<|assistant|> {answer}<|eos|>" for user, answer in turns)


def contextual_prompts(turns):
    """Bind repeated questions in the same history, not unrelated generic followups."""
    return [normalized_prompt(render_turns(turns[:index]) + "\n<|user|> " + user)
            for index, (user, _) in enumerate(turns)]


def is_initial_generic_greeting(prompt):
    return normalized_prompt(prompt).strip("。，,.!！?？~～ ") in INITIAL_GENERIC_GREETINGS


def quotas(count, ratios):
    if len(ratios) != 3 or any(not math.isfinite(value) or value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must be three finite nonnegative values with a positive sum")
    values = [count * value / sum(ratios) for value in ratios]
    result = {name: math.floor(value) for name, value in zip(CATEGORIES, values)}
    remainder = count - sum(result.values())
    order = sorted(range(3), key=lambda index: (-(values[index] - math.floor(values[index])), index))
    for index in order[:remainder]:
        result[CATEGORIES[index]] += 1
    return result


def validation_group_split(group_id, seed, validation_fraction):
    # Keep this byte-for-byte compatible with PersonaMetaDataset.__init__.
    value = int(stable_hash(f"group-validation-v1:{seed}:{group_id}")[:16], 16) / 2**64
    return "validation" if value < validation_fraction else "train"


def load_exclusions(paths):
    prompts = {normalized_prompt("<|user|> " + value) for value in RESERVED_PROBES}
    identifiers = set()
    files = {}

    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if isinstance(value.get("prompt"), str):
                prompts.add(normalized_prompt("<|user|> " + value["prompt"]))
            if isinstance(value.get("text"), str):
                try:
                    prompts.update(contextual_prompts(parse_conversation(value["text"])))
                except ValueError:
                    pass
            for key in ("family", "validation_group_id", "semantic_key"):
                if isinstance(value.get(key), str):
                    identifiers.add(value[key])
            for key, child in value.items():
                if key not in ("text", "prompt"):
                    visit(child)

    for item in paths:
        path = Path(item)
        content = path.read_bytes()
        files[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        if path.suffix == ".jsonl":
            visit(read_jsonl(path))
        else:
            visit(json.loads(content.decode("utf-8")))
    return prompts, identifiers, files


def complete_prefix(text, turns, max_tokens, length_fn):
    if length_fn(text) <= max_tokens:
        return text, turns, False
    best = None
    for end in range(1, len(turns) + 1):
        prefix = render_turns(turns[:end])
        if length_fn(prefix) <= max_tokens:
            best = (prefix, turns[:end], True)
        else:
            break
    return best


def _rank_candidates(rows, *, seed, split, category, balance_key):
    ranked = sorted(rows, key=lambda row: (
        row.get("selection_priority", 0), stable_hash(f"selection:{seed}:{row['text_sha256']}")))
    if balance_key is None:
        return ranked
    layers = defaultdict(lambda: defaultdict(deque))
    for row in ranked:
        layers[row.get("selection_priority", 0)][row[balance_key]].append(row)
    balanced = []
    for priority in sorted(layers):
        strata = layers[priority]
        active = deque(sorted(strata, key=lambda name: (
            stable_hash(f"balance:{seed}:{split}:{category}:{priority}:{name}"), name)))
        while active:
            name = active.popleft()
            balanced.append(strata[name].popleft())
            if strata[name]:
                active.append(name)
    return balanced


def _select(pool, required, *, seed, split, min_multi_turn_fraction, balance_key=None):
    def rank(rows, category):
        return _rank_candidates(rows, seed=seed, split=split, category=category, balance_key=balance_key)
    ranked = {category: rank(pool[category], category) for category in CATEGORIES}
    available = {category: len(ranked[category]) for category in CATEGORIES}
    multi = {category: [row for row in ranked[category] if row["turns"] > 1] for category in CATEGORIES}
    if any(available[category] < required[category] for category in CATEGORIES):
        raise ValueError("Insufficient distinct candidates: " + json.dumps(
            {"split": split, "required": required, "available": available}, sort_keys=True))
    target = math.ceil(sum(required.values()) * min_multi_turn_fraction)
    assigned = {}
    for category in CATEGORIES:  # Prefer actual chat conversations for the multi-turn quota.
        assigned[category] = min(target, required[category], len(multi[category]))
        target -= assigned[category]
    if target:
        raise ValueError("Insufficient distinct multi-turn candidates: " + json.dumps(
            {"split": split, "missing": target, "available": {key: len(value) for key, value in multi.items()}}))
    selected = []
    for category in CATEGORIES:
        # Rank the multi-turn subset separately so single-turn rows cannot
        # consume a stratum's turn in the required conversation quota.
        chosen = rank(multi[category], category)[:assigned[category]]
        identities = {row["text_sha256"] for row in chosen}
        remaining = [row for row in ranked[category] if row["text_sha256"] not in identities]
        chosen += rank(remaining, category)[:required[category] - len(chosen)]
        selected.extend(chosen)
    return selected


def prepare(candidates, output, *, tokenizer=None, count=12000, ratios=(.6, .2, .2), seed=20261108,
            max_tokens=512, validation_fraction=.02, heldout_fraction=.1, heldout_count=None,
            min_multi_turn_fraction=.25, exclusions=(), length_fn=None, balance_key=None):
    """Select exactly count mixed train/validation examples plus a separate heldout.

    Category labels are supplied by the upstream curator, never inferred from
    keywords. Missing classification/provenance fails; malformed conversations,
    exclusions and explicit deterministic-answer conflicts are rejected with
    manifest counts. Open-ended questions may have multiple reasonable answers.
    Optional selection_priority is a nonnegative integer: smaller values rank
    first within the existing split/category and multi-turn quota, default 0.
    Optional balance_key names an explicit nonempty string stratum on every
    candidate. Strata rotate fairly within each priority layer; exhausted strata
    yield their remaining capacity to others without duplicating candidates.
    """
    if (type(count) is not int or count < 3 or max_tokens < 16 or not 0 < validation_fraction < .5
            or not 0 < heldout_fraction < .5 or not 0 <= min_multi_turn_fraction <= 1):
        raise ValueError("Invalid count, token budget, split fraction or multi-turn fraction")
    if balance_key is not None and (not isinstance(balance_key, str) or not balance_key.strip()):
        raise ValueError("balance_key must be a nonempty field name or None")
    if heldout_count is None:
        heldout_count = max(3, math.ceil(count * .05))
    if type(heldout_count) is not int or heldout_count < 1:
        raise ValueError("heldout_count must be a positive integer")
    total_quotas = quotas(count, ratios)
    val_quotas = quotas(max(1, math.ceil(count * validation_fraction)), ratios)
    train_quotas = {key: total_quotas[key] - val_quotas[key] for key in CATEGORIES}
    external_quotas = quotas(heldout_count, ratios)
    if any(value < 0 for value in train_quotas.values()):
        raise ValueError("count is too small for the requested split quotas")
    tokenizer_hash = None
    if length_fn is None:
        if tokenizer is None:
            raise ValueError("An actual tokenizer is required (or explicit length_fn for tests)")
        from tokenizers import Tokenizer
        path = Path(tokenizer)
        tokenizer_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        backend = Tokenizer.from_file(str(path))
        length_fn = lambda text: len(backend.encode(text, add_special_tokens=False).ids) + 2
    blocked_prompts, blocked_ids, exclusion_files = load_exclusions(exclusions)
    rejected = Counter()
    prepared = []
    originals = read_jsonl(candidates)
    for index, original in enumerate(originals):
        family = original.get("family", original.get("validation_group_id"))
        if (original.get("category") not in CATEGORIES or not isinstance(family, str) or not family.strip()
                or any(not isinstance(original.get(key), str) or not original[key].strip()
                       for key in ("language", "source", "text"))):
            raise ValueError(f"Candidate {index} requires explicit category/language/source/family/text")
        priority = original.get("selection_priority", 0)
        if type(priority) is not int or priority < 0:
            raise ValueError(f"Candidate {index} selection_priority must be a nonnegative integer")
        if balance_key is not None and (not isinstance(original.get(balance_key), str) or not original[balance_key].strip()):
            raise ValueError(f"Candidate {index} requires a nonempty string stratum for balance_key={balance_key!r}")
        if "deterministic_answer" in original and type(original["deterministic_answer"]) is not bool:
            raise ValueError(f"Candidate {index} deterministic_answer must be boolean")
        try:
            turns = parse_conversation(original["text"])
        except ValueError as error:
            rejected["invalid_conversation:" + str(error)] += 1
            continue
        if any(context in blocked_prompts for context in contextual_prompts(turns)) or any(
                isinstance(original.get(key), str) and original[key] in blocked_ids
                for key in ("family", "validation_group_id", "semantic_key")):
            rejected["excluded_probe_or_family"] += 1
            continue
        selected = complete_prefix(original["text"].strip(), turns, max_tokens, length_fn)
        if selected is None:
            rejected["no_complete_prefix_within_budget"] += 1
            continue
        text, selected_turns, shortened = selected
        if shortened and len(selected_turns) == 1 and is_initial_generic_greeting(selected_turns[0][0]):
            rejected["shortened_to_generic_opening_only"] += 1
            continue
        contexts = contextual_prompts(turns)
        prepared.append({**original, "family": family, "text": text, "turns": len(selected_turns),
                         "original_turns": len(turns), "shortened_to_complete_prefix": shortened,
                         "token_length": length_fn(text), "text_sha256": stable_hash(render_turns(selected_turns)),
                         "_contexts": contexts, "_answers": [answer for _, answer in turns],
                         "_group_contexts": contexts[1:] if is_initial_generic_greeting(turns[0][0]) else contexts,
                         "_conversation_sha256": stable_hash(render_turns(turns))})

    # Only a curator's explicit deterministic contract makes differing targets
    # contradictory. Natural conversation may have many valid completions.
    completions = defaultdict(set)
    for row in prepared:
        if row.get("deterministic_answer") is True:
            for context, answer in zip(row["_contexts"], row["_answers"]):
                completions[context].add(answer)
    conflicts = {key for key, answers in completions.items() if len(answers) > 1}
    clean = []
    for row in prepared:
        if row.get("deterministic_answer") is True and any(context in conflicts for context in row["_contexts"]):
            rejected["conflicting_answer_for_same_context"] += 1
        else:
            clean.append(row)

    # Connected components bind both explicit semantic families and repeated
    # contextual questions. This also joins aliases of duplicate conversations.
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    def join(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    for row in clean:
        family = "family:" + row["family"]
        find(family)
        # Original families and identical full/selected conversations always
        # bind. A generic initial greeting alone is not a semantic topic ID.
        join(family, "conversation:" + row["_conversation_sha256"])
        join(family, "selected-conversation:" + row["text_sha256"])
        for context in row["_group_contexts"]:
            join(family, "context:" + context)
    members = defaultdict(list)
    for name in parents:
        members[find(name)].append(name)
    groups = {key: "chat:" + stable_hash(json.dumps(sorted(values), ensure_ascii=False))
              for key, values in members.items()}
    duplicate_rows = defaultdict(list)
    for row in clean:
        row["validation_group_id"] = groups[find("family:" + row["family"])]
        duplicate_rows[row["text_sha256"]].append(row)
    distinct = []
    for duplicates in duplicate_rows.values():
        if len({row["category"] for row in duplicates}) != 1:
            rejected["duplicate_category_conflict"] += len(duplicates)
            continue
        duplicates.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
        row = duplicates[0]
        row["duplicate_sources"] = [{key: item[key] for key in ("source", "source_file", "copyright") if key in item}
                                    for item in duplicates]
        rejected["duplicate_text"] += len(duplicates) - 1
        distinct.append(row)
    pools = {split: {category: [] for category in CATEGORIES} for split in ("train", "validation", "heldout")}
    for row in distinct:
        group = row["validation_group_id"]
        external = int(stable_hash(f"external-heldout:{seed}:{group}")[:16], 16) / 2**64 < heldout_fraction
        split = "heldout" if external else validation_group_split(group, seed, validation_fraction)
        row["split"] = split
        pools[split][row["category"]].append(row)
    selected = []
    for split, required in (("train", train_quotas), ("validation", val_quotas), ("heldout", external_quotas)):
        selected += _select(pools[split], required, seed=seed, split=split,
                            min_multi_turn_fraction=min_multi_turn_fraction, balance_key=balance_key)
    grouping_stats = {"connected_components": len(members),
                      "largest_candidate_group": max(Counter(row["validation_group_id"] for row in clean).values(), default=0),
                      "initial_greeting_links_skipped": sum(len(row["_group_contexts"]) < len(row["_contexts"]) for row in clean)}
    for row in selected:
        for key in ("_contexts", "_answers", "_group_contexts", "_conversation_sha256"):
            del row[key]
        row["protocol"] = PROTOCOL
    combined = sorted((row for row in selected if row["split"] != "heldout"),
                      key=lambda row: stable_hash(f"shuffle:{seed}:{row['text_sha256']}"))
    heldout = sorted((row for row in selected if row["split"] == "heldout"), key=lambda row: row["text_sha256"])
    validation = [row for row in combined if row["split"] == "validation"]
    def stats(items):
        return {"rows": len(items), "category": dict(Counter(row["category"] for row in items)),
                "language": dict(Counter(row["language"] for row in items)),
                "multi_turn": sum(row["turns"] > 1 for row in items),
                "groups": len({row["validation_group_id"] for row in items}),
                "shortened_complete_prefixes": sum(row["shortened_to_complete_prefix"] for row in items)}
    split_groups = {split: {row["validation_group_id"] for row in selected if row["split"] == split} for split in pools}
    if any(split_groups[a] & split_groups[b] for a, b in (("train", "validation"), ("train", "heldout"), ("validation", "heldout"))):
        raise RuntimeError("Internal split overlap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, items in (("train.jsonl", combined), ("validation.jsonl", validation), ("heldout.jsonl", heldout)):
        path = output / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in items:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(items)}
    manifest = {"protocol": PROTOCOL, "seed": seed, "count_including_internal_validation": count,
                "ratios": dict(zip(CATEGORIES, ratios)), "validation_fraction": validation_fraction,
                "heldout_candidate_fraction": heldout_fraction, "max_tokens": max_tokens,
                "tokenizer_sha256": tokenizer_hash, "input_sha256": hashlib.sha256(Path(candidates).read_bytes()).hexdigest(),
                "exclusion_files": exclusion_files, "input_rows": len(originals), "rejected": dict(rejected),
                "grouping": grouping_stats,
                "available": {split: {key: len(value) for key, value in pool.items()} for split, pool in pools.items()},
                "training": stats([row for row in combined if row["split"] == "train"]),
                "validation": stats(validation), "heldout": stats(heldout), "all_mixture": stats(combined),
                "group_overlap": 0, "files": files,
                "notes": ["Category labels come from the input curator, never keyword inference.",
                          "Alternate open-ended answers are retained; only deterministic_answer=true conflicts are rejected.",
                          "Generic initial greetings do not join unrelated topics; original families/full conversations/later contexts still bind.",
                          "train.jsonl includes internal validation: use the SAME seed, fraction and validation_group_id in the trainer.",
                          "validation.jsonl is an inspection copy, not a second independent holdout.",
                          "heldout.jsonl is independently family isolated and excluded from train.jsonl.",
                          "No duplicate upsampling or fake EOS; long conversations retain only a complete prefix."]}
    if balance_key is not None:
        availability = {split: {category: dict(sorted(Counter(row[balance_key] for row in pool[category]).items()))
                                for category in CATEGORIES} for split, pool in pools.items()}
        distribution = {split: {category: dict(sorted(Counter(row[balance_key] for row in selected
                                                              if row["split"] == split and row["category"] == category).items()))
                                for category in CATEGORIES} for split in pools}
        manifest["balance"] = {"key": balance_key, "unit": "distinct_rows", "available": availability, "selected": distribution,
                               "saturated": {split: {category: [name for name, capacity in strata.items()
                                                                 if distribution[split][category].get(name, 0) == capacity]
                                                      for category, strata in categories.items()}
                                             for split, categories in availability.items()},
                               "scope": "Round-robin within priority layers and existing category/split/multi-turn quotas; finite capacity can prevent equal counts."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12000)
    parser.add_argument("--ratios", type=lambda value: tuple(float(item) for item in value.split(",")), default=(.6, .2, .2))
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=.02)
    parser.add_argument("--heldout-fraction", type=float, default=.1)
    parser.add_argument("--heldout-count", type=int)
    parser.add_argument("--min-multi-turn-fraction", type=float, default=.25)
    parser.add_argument("--balance-key", help="Candidate field containing an explicit string skill/stratum; optional")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = prepare(args.candidates, args.output, tokenizer=args.tokenizer, count=args.count,
                     ratios=args.ratios, seed=args.seed, max_tokens=args.max_tokens,
                     validation_fraction=args.validation_fraction, heldout_fraction=args.heldout_fraction,
                     heldout_count=args.heldout_count, min_multi_turn_fraction=args.min_multi_turn_fraction,
                     exclusions=args.exclude, balance_key=args.balance_key)
    print(json.dumps({"status": "complete", "output": str(args.output), "counts": result["all_mixture"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

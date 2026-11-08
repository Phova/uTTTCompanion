"""Generate reference-checked elementary reasoning, with no duplicate upsampling."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from .prepare_recovery_curriculum import RESERVED_PROBES, normalized_prompt
except ImportError:
    from prepare_recovery_curriculum import RESERVED_PROBES, normalized_prompt

SOURCE = "uttt:basic-reasoning-v1"
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ZH_DAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
OPERATIONS = ("add", "subtract", "multiply", "max", "min", "weekday")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reference(operation, arguments, language="en"):
    if operation == "weekday":
        index = (DAYS.index(arguments["weekday"]) + arguments["offset"]) % 7
        return (ZH_DAYS if language == "zh" else DAYS)[index]
    a, b = arguments["a"], arguments["b"]
    return str({"add": lambda: a + b, "subtract": lambda: a - b,
                "multiply": lambda: a * b, "max": lambda: max(a, b),
                "min": lambda: min(a, b)}[operation]())


def exclusions(path):
    blocked, prompts = set(), {normalized_prompt(prompt) for prompt in RESERVED_PROBES}
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                blocked.update(value for value in row.get("operands", []) if type(value) is int)
                arguments = row.get("arguments", {})
                blocked.update(value for key, value in arguments.items() if key in ("a", "b") and type(value) is int)
                blocked.update(value for value in arguments.get("values", []) if type(value) is int)
                if isinstance(row.get("prompt"), str):
                    prompts.add(normalized_prompt(row["prompt"]))
    return blocked, prompts


def family_id(operation, arguments):
    return "basic-reasoning:" + digest(json.dumps([operation, arguments], sort_keys=True, separators=(",", ":")))


def prompt_for(operation, arguments, language, variant):
    a, b = arguments.get("a"), arguments.get("b")
    if operation == "add":
        choices = {
            "en": [(f"What is {a} plus {b}? Give just the number.", "direct"),
                   (f"A basket has {a} apples. I add {b} apples. How many apples are there now? Give just the number.", "quantity_increase")],
            "zh": [(f"{a} 加 {b} 等于多少？只给出数字。", "direct"),
                   (f"篮子里有{a}个苹果，再放入{b}个，现在一共有多少个？只回答数字。", "quantity_increase")],
        }
    elif operation == "subtract":
        choices = {
            "en": [(f"What is {a} minus {b}? Give just the number.", "direct"),
                   (f"There are {a} books on a shelf. I remove {b}. How many remain? Give just the number.", "quantity_decrease")],
            "zh": [(f"{a} 减去 {b} 的结果是多少？只写数字。", "direct"),
                   (f"书架上有{a}本书，拿走{b}本后还剩多少本？只回答数字。", "quantity_decrease")],
        }
    elif operation == "multiply":
        choices = {
            "en": [(f"Multiply {a} by {b}. Give just the number.", "direct"),
                   (f"One pencil costs {a} yuan. What is the cost of {b} pencils? Give just the number.", "shopping")],
            "zh": [(f"{a} 乘以 {b} 得多少？只写数字。", "direct"),
                   (f"一支铅笔{a}元，买{b}支一共需要多少元？只回答数字。", "shopping")],
        }
    elif operation in ("max", "min"):
        # Change presentation order without creating another semantic family.
        if variant % 2:
            a, b = b, a
        word, zh_word = ("larger", "大") if operation == "max" else ("smaller", "小")
        choices = {"en": [(f"Which number is {word}: {a} or {b}? Return that number only.", "comparison"),
                          (f"Choose the {word} number from [{a}, {b}]. Give just the number.", "comparison")],
                   "zh": [(f"{a}和{b}哪个更{zh_word}？只回答选中的数字。", "comparison"),
                          (f"从[{a}, {b}]中选出较{zh_word}的数，只写数字。", "comparison")]}
    else:
        day = (ZH_DAYS if language == "zh" else DAYS)[DAYS.index(arguments["weekday"])]
        offset = arguments["offset"]
        choices = {"en": [(f"Starting on {day}, which day is {offset} days later? Give only the weekday name.", "calendar")],
                   "zh": [(f"从{day}开始往后数{offset}天，会到星期几？只回答星期几。", "calendar")]}
    return choices[language][variant % len(choices[language])]


def generate(count=6000, seed=20261108, exclude_heldout=None):
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError("count must be a positive even number; every family supplies both languages")
    blocked, blocked_prompts = exclusions(exclude_heldout)
    allowed = [value for value in range(51) if value not in blocked]
    small = [value for value in range(13) if value not in blocked]
    pools = {operation: [] for operation in OPERATIONS}
    for a in allowed:
        for b in allowed:
            if a <= b and a + b <= 50 and (a, b) != (1, 1):
                pools["add"].append({"a": a, "b": b})
            if a >= b:
                pools["subtract"].append({"a": a, "b": b})
            if a < b:
                for operation in ("max", "min"):
                    pools[operation].append({"a": a, "b": b})
    pools["multiply"] = [{"a": a, "b": b} for a in small for b in small if a <= b]
    pools["weekday"] = [{"weekday": day, "offset": offset} for day in DAYS for offset in range(1, 8)
                         if offset not in blocked and (day, offset) != ("Tuesday", 2)]
    for operation, pool in pools.items():
        pool.sort(key=lambda args: digest(f"{seed}:{family_id(operation, args)}"))
    capacity = {operation: len(pool) for operation, pool in pools.items()}
    required = count // 2
    if sum(capacity.values()) < required:
        raise ValueError("Insufficient distinct families: " + json.dumps({"required": required, "available": capacity}))
    selected, positions = [], Counter()
    while len(selected) < required:
        for operation in OPERATIONS:
            position = positions[operation]
            if position < len(pools[operation]) and len(selected) < required:
                selected.append((operation, pools[operation][position]))
                positions[operation] += 1
    rows, prompts = [], set()
    for operation, arguments in selected:
        family = family_id(operation, arguments)
        for language in ("en", "zh"):
            variant = int(digest(f"wording:{seed}:{family}:{language}")[:8], 16)
            prompt, scenario = prompt_for(operation, arguments, language, variant)
            normalized = normalized_prompt(prompt)
            if normalized in blocked_prompts or normalized in prompts:
                raise ValueError("Generated wording overlaps an excluded or duplicate prompt")
            prompts.add(normalized)
            answer = reference(operation, arguments, language)
            operands = [arguments["offset"]] if operation == "weekday" else [arguments["a"], arguments["b"]]
            if blocked.intersection(operands):
                raise RuntimeError("Blocked operand entered the curriculum")
            rows.append({"text": f"<|user|> {prompt}\n<|assistant|> {answer}<|eos|>",
                         "category": "reasoning", "language": language, "source": SOURCE,
                         "copyright": "Programmatically generated by this repository; no external text copied.",
                         "family": family, "validation_group_id": family, "deterministic_answer": True,
                         "task": operation, "scenario": scenario, "arguments": arguments, "operands": operands,
                         "prompt": prompt, "answer": answer, "accepted": [answer], "match_mode": "strict_text",
                         "reference_verified": True})
    rows.sort(key=lambda row: digest(f"shuffle:{seed}:{row['language']}:{row['family']}"))
    report = {"source": SOURCE, "seed": seed, "rows": len(rows), "distinct_families": required,
              "by_operation": dict(Counter(row["task"] for row in rows)),
              "by_scenario": dict(Counter(row["scenario"] for row in rows)),
              "by_language": dict(Counter(row["language"] for row in rows)), "available_families": capacity,
              "blocked_individual_operands": sorted(blocked),
              "notes": ["One English and one Chinese row per operation/parameter family; no duplicate upsampling.",
                        "Simple capabilities may replay prior training combinations; semantic novelty is not claimed.",
                        "Addition/subtraction operands and results are <=50; multiplication operands are <=12.",
                        "Shopping/quantity wording shares the underlying arithmetic family.",
                        "Reserved 1+1 addition and Tuesday+2-day diagnostic family are excluded."]}
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--exclude-heldout", type=Path)
    args = parser.parse_args()
    rows, report = generate(args.count, args.seed, args.exclude_heldout)
    manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError("Refusing to replace an existing generated curriculum")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

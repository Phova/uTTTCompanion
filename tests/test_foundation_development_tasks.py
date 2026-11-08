"""Independent parameter cohorts have reproducible, executable references."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import unittest

from deploy.evaluate_chat_curriculum import restricted_python_check, validate_tasks


ROOT = Path(__file__).resolve().parents[1]
FILES = ("foundation_development_tasks_20261108.jsonl", "foundation_holdout_tasks_20261108.jsonl")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ZH_DAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def load(name):
    with (ROOT / "deploy" / name).open(encoding="utf-8") as stream:
        return validate_tasks([json.loads(line) for line in stream if line.strip()])


def semantic_identity(row):
    arguments = dict(row["arguments"])
    operation = row.get("foundation_operation", row.get("foundation_behavior"))
    if row["category"] == "reasoning" and operation in {"add", "multiply", "max", "min"}:
        arguments["a"], arguments["b"] = sorted((arguments["a"], arguments["b"]))
    if row["category"] == "reasoning" and operation == "order":
        arguments["values"] = sorted(arguments["values"])
    return hashlib.sha256(json.dumps([row["category"], operation, arguments], sort_keys=True,
                                     ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def reference_reasoning_answers(row):
    args, operation = row["arguments"], row["foundation_operation"]
    if operation == "add":
        return {str(sum((args["a"], args["b"])))}
    if operation == "subtract":
        return {str(args["a"] - args["b"])}
    if operation == "multiply":
        return {str(sum([args["a"]] * args["b"]))}
    if operation == "divide":
        quotient, remainder = divmod(args["a"], args["b"])
        if remainder:
            raise AssertionError("An exact-division task has a remainder")
        return {str(quotient)}
    if operation in {"max", "min"}:
        ordered = sorted((args["a"], args["b"]))
        return {str(ordered[-1 if operation == "max" else 0])}
    if operation == "quantity":
        return {str(sum((args["initial"], args["gained"], -args["removed"])))}
    if operation == "clock":
        at = datetime.datetime(2020, 1, 1, args["hour"], args["minute"])
        return {(at + datetime.timedelta(minutes=args["delta"])).strftime("%H:%M")}
    if operation == "weekday":
        index = (DAYS.index(args["weekday"]) + args["offset"]) % len(DAYS)
        if row["language"] == "en":
            return {DAYS[index]}
        return {"星期日", "星期天"} if index == 6 else {ZH_DAYS[index]}
    if operation == "order":
        ordered = sorted(args["values"])
        return {str(ordered[-1 if args["query"] == "max" else 0])}
    raise AssertionError("Unknown reference operation")


def reference_function(row):
    """Known test-owned source, interpreted rather than executing model code."""
    args, behavior = row["arguments"], row["foundation_behavior"]
    function = row["code_check"]["function"]
    if behavior == "affine":
        name, expression = "value", f"({args['a']}) * value + ({args['b']})"
    elif behavior == "clamp":
        name = "value"
        expression = f"{args['low']} if value < {args['low']} else {args['high']} if value > {args['high']} else value"
    elif behavior == "interval":
        name = "value"
        lower, upper = ">=" if args["include_low"] else ">", "<=" if args["include_high"] else "<"
        expression = f"value {lower} {args['low']} and value {upper} {args['high']}"
    elif behavior == "mod_equals":
        name, expression = "number", f"number % {args['divisor']} == {args['remainder']}"
    elif behavior == "length_test":
        name = "items"
        expression = "len(items) " + {"eq": "==", "ge": ">=", "lt": "<"}[args["operator"]] + f" {args['limit']}"
    elif behavior == "nth_default":
        name = "items"
        expression = f"items[{args['index']}] if -len(items) <= {args['index']} < len(items) else {args['default']!r}"
    elif behavior == "slice_text":
        name = "text"
        expression = f"text[:{args['length']}]" if args["side"] == "prefix" else f"text[-{args['length']}:]"
    else:
        raise AssertionError("Unknown reference behavior")
    return f"def {function}({name}):\n    return {expression}"


class FoundationDevelopmentTasksTests(unittest.TestCase):
    def test_cohorts_have_declared_mix_seeds_metadata_and_unique_identities(self):
        groups = [load(name) for name in FILES]
        for index, rows in enumerate(groups):
            with self.subTest(file=FILES[index]):
                self.assertEqual(len(rows), 64)
                self.assertEqual(Counter(row["category"] for row in rows), {"reasoning": 40, "coding": 24})
                self.assertEqual(Counter(row["language"] for row in rows), {"zh": 48, "en": 16})
                self.assertEqual({row["parameter_seed"] for row in rows}, {2026110803 + index})
                self.assertEqual({row["evaluation_partition"] for row in rows},
                                 {"development" if index == 0 else "parameter_holdout"})
                self.assertEqual({row["used_for_model_selection"] for row in rows}, {index == 0})
                self.assertTrue(all(row["evaluation_only"] for row in rows))
                self.assertEqual(len({row["id"] for row in rows}), 64)
                self.assertEqual(len({row["family"] for row in rows}), 64)
                for row in rows:
                    self.assertEqual(row["semantic_key"], semantic_identity(row))
                    self.assertEqual(row["family"], "foundation-eval20261108:" + row["semantic_key"])
                    if row["category"] == "coding":
                        self.assertEqual(row["format_requirement"], "code_only")
                        self.assertIs(row["format_scored_by_functional_check"], False)
        for identity in ("semantic_key", "family", "id", "prompt"):
            self.assertFalse({row[identity] for row in groups[0]} & {row[identity] for row in groups[1]}, identity)
        self.assertEqual(Counter(row["task"] for row in groups[0]), Counter(row["task"] for row in groups[1]))

    def test_reasoning_references_and_in_domain_parameters_are_correct(self):
        for name in FILES:
            for row in load(name):
                if row["category"] != "reasoning":
                    continue
                with self.subTest(file=name, id=row["id"]):
                    self.assertEqual(set(row["accepted"]), reference_reasoning_answers(row))
                    args, op = row["arguments"], row["foundation_operation"]
                    if op in {"add", "subtract", "max", "min"}:
                        self.assertTrue(all(type(args[k]) is int and 0 <= args[k] <= 20 for k in ("a", "b")))
                    elif op == "multiply":
                        self.assertTrue(all(0 <= args[k] <= 10 for k in ("a", "b")))
                    elif op == "divide":
                        self.assertTrue(0 <= args["a"] <= 100 and 1 <= args["b"] <= 10)
                    elif op == "quantity":
                        self.assertTrue(0 <= args["initial"] <= 10 and 0 <= args["gained"] <= 5 and 0 <= args["removed"] <= 5)
                        self.assertGreaterEqual(args["initial"] + args["gained"], args["removed"])
                    elif op == "clock":
                        self.assertTrue(0 <= args["hour"] < 24 and 0 <= args["minute"] < 60 and args["minute"] % 5 == 0)
                        self.assertTrue(1 <= args["delta"] <= 30)
                    elif op == "weekday":
                        self.assertIn(args["weekday"], DAYS)
                        self.assertTrue(1 <= args["offset"] <= 7)
                    elif op == "order":
                        self.assertEqual(len(set(args["values"])), 3)
                        self.assertTrue(all(0 <= value <= 20 for value in args["values"]))

    def test_every_code_case_matches_an_independent_supported_reference(self):
        for name in FILES:
            for row in load(name):
                if row["category"] != "coding":
                    continue
                with self.subTest(file=name, id=row["id"]):
                    checks = row["code_check"]
                    self.assertGreaterEqual(len(checks["cases"]), 4)
                    result = restricted_python_check(reference_function(row), checks["function"], checks["cases"])
                    self.assertTrue(result["passed"], result)
                    self.assertEqual(result["cases_passed"], len(checks["cases"]))
                    self.assertFalse(result["generated_code_executed"])

    def test_new_prompts_do_not_reuse_the_previous_chat_development_or_recorded_probes(self):
        previous = load_previous = ROOT / "deploy/chat_development_tasks_20261108.jsonl"
        with previous.open(encoding="utf-8") as stream:
            blocked = {json.loads(line)["prompt"] for line in stream if line.strip()}
        diagnosis = ROOT / "docs/quality_diagnosis_20261108.json"
        if diagnosis.exists():
            document = json.loads(diagnosis.read_text(encoding="utf-8"))
            for section in ("results", "product_probes", "context_probes"):
                blocked.update(row["prompt"] for row in document[section])
        for name in FILES:
            self.assertFalse(blocked & {row["prompt"] for row in load(name)})


if __name__ == "__main__":
    unittest.main()

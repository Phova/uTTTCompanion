"""Bounded code functionality and honest independent chat reporting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy.evaluate_chat_curriculum import (
    evaluate, restricted_python_check, summarize_results, validate_tasks,
)


def task(category="chat", **changes):
    row = {"category": category, "task": "short_chat", "language": "zh",
           "family": "diagnostic:one", "prompt": "今天怎么样？"}
    row.update(changes)
    return row


class RestrictedPythonTests(unittest.TestCase):
    def check(self, code, function, arguments, expected):
        return restricted_python_check(code, function, [{"args": arguments, "expected": expected}])

    def test_correct_code_is_checked_on_inputs_not_syntax(self):
        cases = [{"args": [2, 3], "expected": 5}, {"args": [-2, 2], "expected": 0},
                 {"args": [0, 0], "expected": 0}]
        actual = restricted_python_check("```python\ndef add(a, b):\n    return a + b\n```", "add", cases)
        self.assertTrue(actual["passed"])
        self.assertFalse(actual["generated_code_executed"])
        self.assertEqual(actual["cases_passed"], 3)
        for invalid in ("add(a, b)", "def add(a, b):\n    return a - b", "def other(a, b):\n    return a + b"):
            with self.subTest(source=invalid):
                self.assertFalse(restricted_python_check(invalid, "add", cases)["passed"])

    def test_slices_conditionals_assignment_and_fixed_builtins(self):
        self.assertTrue(self.check("def reverse_text(text):\n    return text[::-1]", "reverse_text", ["AbC"], "CbA")["passed"])
        code = "def last_item(items):\n    if len(items) == 0:\n        return None\n    value = items[-1]\n    return value"
        self.assertTrue(self.check(code, "last_item", [[]], None)["passed"])
        self.assertTrue(self.check(code, "last_item", [[2, 9]], 9)["passed"])
        self.assertTrue(self.check("def ordered(items):\n    return sorted(items)", "ordered", [[2, -3, 1]], [-3, 1, 2])["passed"])
        self.assertTrue(self.check("def even(n):\n    return n % 2 == 0", "even", [6], True)["passed"])
        self.assertTrue(self.check("def choose(a, b):\n    return a if a < b else b", "choose", [3, 7], 3)["passed"])

    def test_comparisons_and_short_circuit_do_not_evaluate_invalid_tail(self):
        self.assertTrue(self.check("def between(n):\n    return 1 < n < 5", "between", [3], True)["passed"])
        self.assertTrue(self.check("def safe(n):\n    return n != 0 and 2 / n > 0", "safe", [0], False)["passed"])
        self.assertFalse(self.check("def one():\n    return True", "one", [], 1)["passed"])

    def test_imports_attributes_recursion_loops_and_side_effects_are_never_executed(self):
        samples = [
            "import os\ndef f(x):\n    return x",
            "def f(x):\n    return __import__('os').system('never run')",
            "def f(x):\n    return open('/tmp/never-create', 'w')",
            "def f(x):\n    return x.__class__",
            "def f(x):\n    return f(x)",
            "def f(x):\n    while True:\n        pass",
            "def f(x):\n    return [a for a in x]",
            "@dangerous()\ndef f(x):\n    return x",
            "def f(x=dangerous()):\n    return x",
        ]
        for code in samples:
            with self.subTest(code=code):
                result = self.check(code, "f", [1], 1)
                self.assertFalse(result["passed"])
                self.assertFalse(result["generated_code_executed"])

    def test_resource_bounds_reject_allocation_and_numeric_explosion(self):
        for code in ("def f(x):\n    return 'a' * 1000000000",
                     "def f(x):\n    return x * x * x",
                     "def f(x):\n    return 1e309",
                     "def f(x):\n    return [0] * 1000000000"):
            self.assertFalse(self.check(code, "f", [100000000], None)["passed"])
        self.assertFalse(self.check("def f(x):\n    return 1 / 0", "f", [1], 0)["passed"])

    def test_builtin_shadowing_is_rejected_instead_of_using_wrong_python_semantics(self):
        self.assertFalse(self.check("def f(len):\n    return len([1])", "f", [2], 1)["passed"])
        self.assertFalse(self.check("def f(x):\n    sum = 3\n    return sum([1, 2])", "f", [0], 3)["passed"])

    def test_known_builtin_signature_annotations_are_inspected_without_evaluation(self):
        cases = [
            ("def f(x: float) -> float:\n    return x + 1", [2.5], 3.5),
            ("def f(x: int) -> bool:\n    return x > 0", [4], True),
            ("def f(x: str) -> str:\n    return x[::-1]", ["abc"], "cba"),
            ("def f(x: list[int]) -> int:\n    return len(x)", [[1, 2]], 2),
            ("def f(x: dict[str, list[int]]) -> list[int]:\n    return x['a']", [{"a": [1, 2]}], [1, 2]),
            ("def f(x: tuple[int, ...]) -> int:\n    return x[0]", [[3, 4]], 3),
            ("def f(x: int) -> None:\n    return None", [0], None),
        ]
        for source, arguments, expected in cases:
            with self.subTest(source=source):
                result = self.check(source, "f", arguments, expected)
                self.assertTrue(result["passed"], result)
                self.assertFalse(result["generated_code_executed"])

    def test_annotation_calls_attributes_and_unknown_expressions_are_rejected(self):
        for source in (
            "def f(x: dangerous()) -> int:\n    return x",
            "def f(x: int) -> dangerous():\n    return x",
            "def f(x: __import__('os').system('never run')):\n    return x",
            "def f(x: typing.Any):\n    return x",
            "def f(x: _Private):\n    return x",
            "def f(x: int + int):\n    return x",
            "def f(x: list[int, str]):\n    return x",
        ):
            with self.subTest(source=source):
                result = self.check(source, "f", [1], 1)
                self.assertFalse(result["passed"])
                self.assertFalse(result["generated_code_executed"])


class ChatEvaluationTests(unittest.TestCase):
    def test_development_fixture_is_balanced_and_every_reference_function_passes(self):
        from collections import Counter
        path = Path(__file__).resolve().parents[1] / "deploy/chat_development_tasks_20261108.jsonl"
        with path.open(encoding="utf-8") as stream:
            rows = validate_tasks([json.loads(line) for line in stream if line.strip()])
        self.assertEqual(Counter(row["category"] for row in rows), {"chat": 12, "reasoning": 16, "coding": 12})
        self.assertEqual(Counter(row["language"] for row in rows), {"zh": 30, "en": 10})
        self.assertEqual(len({row["family"] for row in rows}), 40)
        self.assertEqual(sum(bool(row.get("history")) for row in rows), 4)
        self.assertTrue(all(row["development_only"] for row in rows))
        functions = {
            "increment_by_two": "def increment_by_two(value):\n    return value + 2",
            "is_odd_number": "def is_odd_number(number):\n    return number % 2 != 0",
            "larger_value": "def larger_value(left, right):\n    return left if left >= right else right",
            "first_or_none": "def first_or_none(values):\n    return values[0] if len(values) else None",
            "sum_three": "def sum_three(a, b, c):\n    return a + b + c",
            "clamp_to_zero": "def clamp_to_zero(value):\n    return value if value >= 0 else 0",
            "middle_character": "def middle_character(text):\n    return text[len(text) // 2] if len(text) else ''",
            "rotate_once": "def rotate_once(values):\n    return values[1:] + values[:1]",
            "count_items": "def count_items(values):\n    return len(values)",
            "is_between": "def is_between(value, lower, upper):\n    return lower <= value <= upper",
            "sum_values": "def sum_values(values):\n    return sum(values)",
            "duplicate_text": "def duplicate_text(text):\n    return text + text",
        }
        for row in rows:
            if row["category"] == "coding":
                check = row["code_check"]
                with self.subTest(function=check["function"]):
                    result = restricted_python_check(functions[check["function"]], check["function"], check["cases"])
                    self.assertTrue(result["passed"], result)

    def test_category_labels_and_case_definitions_are_required_before_gpu_work(self):
        self.assertTrue(validate_tasks([task()])[0]["id"])
        for invalid in (task(category="unknown"), task(family=None), task(prompt="<|assistant|> bad"),
                        task(category="reasoning"), task(category="coding"), task(history=[{"role": "user", "content": "hi"}])):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                validate_tasks([invalid])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_tasks([task(), task()])
        with self.assertRaisesRegex(ValueError, "History"):
            validate_tasks([task(history=[{"role": "user", "content": "hi"},
                                         {"role": "assistant", "content": "<|eos|>"}])])

    def test_open_chat_is_manual_review_not_an_automatic_success_score(self):
        rows = [{"category": "chat", "task": "greet", "language": "zh", "automatic_pass": None,
                 "eos_generated": True, "stop_reason": "eos", "text_4gram_repetition": 0.},
                {"category": "coding", "task": "add", "language": "en", "automatic_pass": False,
                 "eos_generated": True, "stop_reason": "eos", "text_4gram_repetition": 0.}]
        summary = summarize_results(rows)
        self.assertEqual(summary["category:chat"]["manual_review_required"], 1)
        self.assertEqual(summary["category:chat"]["automatic_checks"], 0)
        self.assertEqual(summary["all"]["automatic_passes"], 0)
        self.assertNotIn("quality_passed", summary["all"])

    def test_wrapper_preserves_metadata_and_structured_history_and_tests_real_code(self):
        tasks = [task(history=[{"role": "user", "content": "我喜欢蓝色。"},
                               {"role": "assistant", "content": "知道了。"}]),
                 task(category="coding", task="add", language="en", family="code:add", prompt="Define add.",
                      code_check={"protocol": "bounded_python_ast_v1", "function": "add",
                                  "cases": [{"args": [3, 4], "expected": 7}]})]
        prepared = []
        def fake(checkpoint, values, output, *args, **kwargs):
            prepared.extend(values)
            return {"training_metrics": {"training_stage": "sft"}, "results": [
                {"prompt": row["prompt"], "reply": reply, "eos_generated": True,
                 "stop_reason": "eos", "text_4gram_repetition": 0.}
                for row, reply in zip(values, ("你好", "def add(a, b):\n    return a - b"))]}
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory, \
                patch.dict("sys.modules", {"deploy.evaluate_model": SimpleNamespace(evaluate=fake)}):
            output = Path(directory) / "report.json"
            report = evaluate("read-only.pt", tasks, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["protocol"], report["protocol"])
        self.assertEqual(prepared[0]["mode"], "continuation")
        self.assertTrue(prepared[0]["prompt"].startswith("<|user|> 我喜欢蓝色。"))
        self.assertTrue(prepared[0]["prompt"].endswith("<|assistant|>"))
        self.assertEqual(report["results"][0]["prompt"], tasks[0]["prompt"])
        self.assertEqual(report["results"][0]["family"], "diagnostic:one")
        self.assertIsNone(report["results"][0]["automatic_pass"])
        self.assertFalse(report["results"][1]["automatic_pass"])
        self.assertEqual(report["results"][1]["code_check"]["cases_passed"], 0)


if __name__ == "__main__":
    unittest.main()

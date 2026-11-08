import json
from pathlib import Path
import tempfile
import unittest

from deploy.generate_basic_reasoning import DAYS, SOURCE, family_id, generate, reference
from deploy.prepare_recovery_curriculum import parse_conversation


class BasicReasoningTests(unittest.TestCase):
    def test_references_cover_arithmetic_comparison_and_calendar_wrap(self):
        for operation, args, answer in (("add", {"a": 17, "b": 8}, "25"),
                                        ("subtract", {"a": 20, "b": 8}, "12"),
                                        ("multiply", {"a": 6, "b": 7}, "42"),
                                        ("max", {"a": 3, "b": 10}, "10"),
                                        ("min", {"a": 10, "b": 3}, "3")):
            self.assertEqual(reference(operation, args), answer)
        self.assertEqual(reference("weekday", {"weekday": "Sunday", "offset": 2}), "Tuesday")
        self.assertEqual(reference("weekday", {"weekday": "Sunday", "offset": 2}, "zh"), "星期二")

    def test_generated_rows_have_unique_prompts_correct_answers_eos_and_bilingual_families(self):
        rows, report = generate(6000, 311)
        self.assertEqual(len(rows), 6000)
        self.assertEqual(len({row["prompt"] for row in rows}), 6000)
        self.assertEqual(report["distinct_families"], 3000)
        self.assertTrue({"shopping", "quantity_increase", "quantity_decrease"}.issubset(report["by_scenario"]))
        families = {}
        for row in rows:
            self.assertEqual(row["source"], SOURCE)
            self.assertEqual(row["category"], "reasoning")
            self.assertIs(row["deterministic_answer"], True)
            self.assertEqual(parse_conversation(row["text"]), [(row["prompt"], row["answer"])])
            self.assertEqual(row["answer"], reference(row["task"], row["arguments"], row["language"]))
            self.assertEqual(row["family"], family_id(row["task"], row["arguments"]))
            families.setdefault(row["family"], set()).add(row["language"])
            if row["task"] in ("add", "subtract"):
                self.assertTrue(all(0 <= value <= 50 for value in row["arguments"].values()))
                self.assertTrue(0 <= int(row["answer"]) <= 50)
            if row["task"] == "multiply":
                self.assertTrue(all(0 <= value <= 12 for value in row["arguments"].values()))
            self.assertFalse(row["task"] == "add" and row["arguments"] == {"a": 1, "b": 1})
        self.assertTrue(all(languages == {"en", "zh"} for languages in families.values()))

    def test_all_heldout_individual_operands_are_excluded_and_shortage_is_explicit(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            path = Path(temporary) / "heldout.jsonl"
            path.write_text(json.dumps({"operands": [number for number in range(100) if number % 5 == 4]}) + "\n")
            rows, report = generate(6000, 312, path)
            self.assertEqual(len(rows), 6000)
            self.assertEqual(report["distinct_families"], 3000)
            self.assertTrue(all(value % 5 != 4 for row in rows for value in row["operands"]))
            path.write_text(json.dumps({"operands": list(range(100))}) + "\n")
            with self.assertRaisesRegex(ValueError, "Insufficient distinct families"):
                generate(6000, 312, path)

    def test_reproducible_and_counts_are_not_filled_by_repetition(self):
        self.assertEqual(generate(120, 313), generate(120, 313))
        self.assertNotEqual(generate(120, 313)[0], generate(120, 314)[0])
        for count in (0, -2, 3, True):
            with self.assertRaisesRegex(ValueError, "positive even"):
                generate(count)
        with self.assertRaisesRegex(ValueError, "Insufficient distinct families"):
            generate(100000)


if __name__ == "__main__":
    unittest.main()

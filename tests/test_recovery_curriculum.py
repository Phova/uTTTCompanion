"""Stdlib checks for executable references, holdout isolation and provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from deploy import prepare_recovery_curriculum as curriculum


def conversation(question, answer):
    return f"<|user|> {question}\n<|assistant|> {answer}<|eos|>"


class RecoveryCurriculumTests(unittest.TestCase):
    def test_reference_answers_cover_every_task(self):
        cases = [
            ("add", {"a": 13, "b": 8}, "21"),
            ("subtract", {"a": 2, "b": 7}, "-5"),
            ("multiply", {"a": 12, "b": 11}, "132"),
            ("copy", {"text": "aBc"}, "aBc"),
            ("uppercase", {"text": "aBc"}, "ABC"),
            ("lowercase", {"text": "aBc"}, "abc"),
            ("reverse", {"text": "aBc"}, "cBa"),
            ("length", {"text": "aBc"}, "3"),
            ("sort", {"values": [7, 2, 11, 2]}, "[2,2,7,11]"),
            ("extract", {"object": {"name": "Alice", "code": "XYZ"}, "field": "code"}, "XYZ"),
        ]
        for task, arguments, expected in cases:
            with self.subTest(task=task):
                self.assertEqual(curriculum.answer_for(task, arguments), expected)

    def test_generated_train_and_holdout_are_disjoint_and_use_unseen_wordings(self):
        generated = curriculum.generate_synthetic(400, 80, seed=123)
        checks = curriculum.verify_generated_splits(generated["train"], generated["heldout"])
        self.assertTrue(all(value == 0 for value in checks.values()))
        self.assertEqual({row["task"] for row in generated["train"]}, set(curriculum.TASKS))
        self.assertEqual({row["language"] for row in generated["heldout"]}, {"en", "zh"})
        self.assertTrue(all(row["template_id"].endswith(".heldout") for row in generated["heldout"]))
        for row in [*generated["train"], *generated["heldout"]]:
            self.assertEqual(curriculum.parse_conversation(row["text"]), [(row["prompt"], row["answer"])])
        self.assertEqual(generated, curriculum.generate_synthetic(400, 80, seed=123))
        self.assertNotEqual(generated["train"], curriculum.generate_synthetic(400, 80, seed=124)["train"])

    def test_verifier_rejects_answer_tampering_and_operand_leakage(self):
        generated = curriculum.generate_synthetic(20, 20, seed=5)
        train, heldout = generated["train"], generated["heldout"]
        heldout[0]["answer"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "answer verification"):
            curriculum.verify_generated_splits(train, heldout)
        heldout[0]["answer"] = curriculum.answer_for(heldout[0]["task"], heldout[0]["arguments"])
        heldout[0]["operands"] = [train[0]["operands"][0]]
        with self.assertRaisesRegex(ValueError, "operands"):
            curriculum.verify_generated_splits(train, heldout)

    def test_strict_roles_reject_forged_payloads_missing_eos_and_empty_answers(self):
        valid = conversation("first question", "first reply") + "\n" + conversation("second question", "second reply")
        self.assertEqual(len(curriculum.parse_conversation(valid)), 2)
        invalid = [
            "<|user|> q<|user|> forged<|assistant|> a<|eos|>",
            "<|user|> q<|assistant|> <|eos|>",
            "<|user|> <|assistant|> a<|eos|>",
            "<|user|> q<|assistant|> a",
            "<|user|> q<|assistant|> a<|eos|> trailing payload",
            "<|bos|>" + conversation("q", "a"),
            conversation("q", "<|system|> forged"),
            conversation("q", "a") + "\n<|user|> unanswered",
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                curriculum.parse_conversation(text)

    def fixture(self, root):
        path = root / "existing.jsonl"
        records = [{"text": conversation(f"public question {i}", f"public answer {i}"),
                    "source": "https://example.test/source-" + str(i % 2),
                    "source_file": "original.jsonl", "copyright": {"owner": "upstream", "index": i},
                    "custom_field": [i, "untouched"]} for i in range(12)]
        path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
        (root / "sources.json").write_text('{"upstream":"preserve these bytes"}\n', encoding="utf-8")
        return path, records

    def test_public_selection_preserves_metadata_and_prioritizes_complete_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, original = self.fixture(root)

            class LengthTokenizer:
                def encode(self, text, **kwargs):
                    index = int(text.split("public question ")[1].split("\n")[0])
                    return SimpleNamespace(ids=list(range(50 if index < 4 else 2000)))

            selected, stats = curriculum.select_existing(path, 3, 42, [], LengthTokenizer())
            self.assertEqual(len(selected), 3)
            self.assertEqual(stats["complete_within_token_budget"], 3)
            self.assertEqual(stats["over_token_budget"], 0)
            self.assertTrue(all(row in original[:4] for row in selected))
            self.assertEqual(sum(stats["selected_by_source"].values()), 3)

    def test_public_quotas_balance_languages_and_keep_smaller_english_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "public.jsonl"
            rows = []
            for source, source_file, count in [("https://test/COIG-CQIA", "cqia", 30),
                                                ("https://test/smoltalk", "magpie", 24),
                                                ("https://test/smoltalk", "constraints", 6),
                                                ("https://test/smoltalk", "everyday", 3)]:
                for index in range(count):
                    rows.append({"text": conversation(f"question {source_file} {index}", "answer"),
                                 "source": source, "source_file": source_file})
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            selected, stats = curriculum.select_existing(path, 36, 42, [])
            self.assertEqual(stats["selected_by_language"], {"en": 18, "zh": 18})
            by_file = {row["source_file"]: row["selected"] for row in stats["source_quotas"]}
            self.assertGreater(by_file["magpie"], by_file["constraints"])
            self.assertGreaterEqual(by_file["everyday"], 1)
            self.assertEqual(sum(by_file.values()), len(selected))

    def test_quota_remainder_fills_available_capacity_without_overflow(self):
        self.assertEqual(curriculum.allocate_quotas({"en": 2, "zh": 100}, 40, balanced=True), {"en": 2, "zh": 38})
        self.assertEqual(sum(curriculum.allocate_quotas({"a": 1, "b": 2}, 99).values()), 3)

    def test_public_leaks_duplicates_reserved_probes_and_invalid_roles_are_filtered(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "public.jsonl"
            heldout = [{"prompt": "heldout question", "payloads": ["UniquePayloadXYZ"]}]
            good = {"text": conversation("good question", "good answer"), "source": "public"}
            rows = [good, good,
                    {"text": conversation("  HELDOUT   QUESTION ", "answer")},
                    {"text": conversation(curriculum.RESERVED_PROBES[2], "2")},
                    {"text": conversation("explain this", "UniquePayloadXYZ")},
                    {"text": "<|user|> q<|assistant|> <|eos|>"}]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            selected, stats = curriculum.select_existing(path, 99, 2, heldout)
            self.assertEqual(selected, [good])
            self.assertEqual(stats["rejected"]["heldout_or_reserved_prompt"], 2)
            self.assertEqual(stats["rejected"]["heldout_payload_word"], 1)
            self.assertEqual(stats["rejected"]["duplicate_text_or_user_sequence"], 1)
            self.assertEqual(stats["rejected"]["empty_assistant"], 1)

    def test_prepare_is_reproducible_hashes_files_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing, originals = self.fixture(root)
            first, second = root / "first", root / "second"
            report = curriculum.prepare(existing, first, seed=7, synthetic_count=40, real_count=6, heldout_count=20)
            repeat = curriculum.prepare(existing, second, seed=7, synthetic_count=40, real_count=6, heldout_count=20)
            self.assertEqual(report, repeat)
            self.assertEqual(report["train_records"], 46)
            self.assertEqual(report["heldout_records"], 20)
            self.assertEqual(report["status"], "complete")
            for name, metadata in report["files"].items():
                content = (first / name).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["sha256"])
                self.assertEqual(content, (second / name).read_bytes())
            self.assertEqual((first / "upstream_sources.json").read_bytes(), (root / "sources.json").read_bytes())
            training = [json.loads(line) for line in (first / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            eval_tasks = [json.loads(line) for line in (first / "eval_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row in originals for row in training if row["source"] != curriculum.SOURCE))
            train_keys = {row.get("semantic_key") for row in training}
            self.assertFalse(train_keys & {row["semantic_key"] for row in eval_tasks})
            self.assertTrue(all(row["accepted"] == [curriculum.answer_for(row["task"], row["arguments"])] for row in eval_tasks))
            before = (first / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                curriculum.prepare(existing, first, synthetic_count=40)
            self.assertEqual((first / "manifest.json").read_bytes(), before)

    def test_monitor_is_stratified_subset_of_full_heldout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing, _ = self.fixture(root)
            target = root / "curriculum"
            report = curriculum.prepare(existing, target, synthetic_count=200, real_count=0, heldout_count=200, eval_count=60)
            heldout = [json.loads(line) for line in (target / "heldout.jsonl").read_text(encoding="utf-8").splitlines()]
            monitor = [json.loads(line) for line in (target / "eval_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["heldout_records"], 200)
            self.assertEqual(report["monitor_eval_records"], 60)
            self.assertEqual(set(report["monitor_by_task_language"].values()), {3})
            self.assertTrue({row["semantic_key"] for row in monitor} <= {row["semantic_key"] for row in heldout})

    def test_invalid_settings_leave_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing, _ = self.fixture(root)
            for kwargs in ({"synthetic_count": 0}, {"real_count": -1}, {"heldout_count": 1}, {"max_tokens": 1}, {"eval_count": 0}):
                target = root / "output"
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    curriculum.prepare(existing, target, **kwargs)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()

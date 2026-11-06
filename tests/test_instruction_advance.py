"""Instruction augmentation preserves references and untouched heldout files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deploy.prepare_instruction_advance import (
    PARAPHRASES, augment_synthetic, prepare, public_validation_group, read_jsonl, select_public,
)
from deploy.prepare_recovery_curriculum import (
    TEMPLATES, answer_for, generate_synthetic, parse_conversation, verify_generated_splits,
)


class InstructionAdvanceTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.directory.name)
        self.parent = self.root / "parent"
        self.parent.mkdir()
        generated = generate_synthetic(200, 80, 91)
        self.synthetic, self.heldout = generated["train"], generated["heldout"]
        self.public = []
        for language, source in (("zh", "COIG-CQIA"), ("en", "smoltalk")):
            for index in range(12):
                self.public.append({"text": f"<|user|> {language} public question {index}\n<|assistant|> answer {index}<|eos|>",
                                    "source": source, "source_file": "public.jsonl", "copyright": "retain me"})
        long = {**self.public[0], "text": "<|user|> long\n<|assistant|> " + "x" * 700 + "<|eos|>"}
        self.public.append(long)
        self.write("train.jsonl", self.synthetic + self.public)
        self.write("heldout.jsonl", self.heldout)
        self.write("eval_tasks.jsonl", [{"prompt": row["prompt"], "accepted": [row["answer"]]}
                                       for row in self.heldout[:20]])
        (self.parent / "manifest.json").write_text('{"parent":"fixture"}\n', encoding="utf-8")
        (self.parent / "upstream_sources.json").write_text('{"copyright":"keep"}\n', encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, rows):
        with (self.parent / name).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_reference_correct_augmentation_has_seven_wordings_per_task_and_language(self):
        augmented = augment_synthetic(self.synthetic, 52)
        self.assertEqual(len(augmented), 3 * len(self.synthetic))
        groups = {}
        for row in augmented:
            self.assertEqual(parse_conversation(row["text"]), [(row["prompt"], answer_for(row["task"], row["arguments"]))])
            groups.setdefault((row["task"], row["language"]), set()).add(row["augmentation_variant"])
        self.assertTrue(all(len(variants) == 7 for variants in groups.values()))
        self.assertEqual(len({row["semantic_key"] for row in augmented}), len(self.synthetic))
        for semantic in {row["semantic_key"] for row in augmented}:
            self.assertEqual(len({row["validation_group_id"] for row in augmented if row["semantic_key"] == semantic}), 1)
        self.assertEqual(len({row["text"] for row in augmented}), len(augmented))
        self.assertTrue(all(value == 0 for value in verify_generated_splits(augmented, self.heldout).values()))
        for task, languages in PARAPHRASES.items():
            for language, pool in languages.items():
                self.assertNotIn(TEMPLATES[task][language][1], pool)

    def test_prepare_copies_holdout_bytes_and_keeps_only_complete_balanced_public_rows(self):
        before = {name: (self.parent / name).read_bytes() for name in ("heldout.jsonl", "eval_tasks.jsonl")}
        output = self.root / "new"
        report = prepare(self.parent, output, seed=53, public_count=10, length_fn=len)
        rows = read_jsonl(output / "train.jsonl")
        public = [row for row in rows if row.get("task") is None]
        self.assertEqual(len(rows), 610)
        self.assertEqual(len(public), 10)
        self.assertEqual(report["public"]["by_language"], {"en": 5, "zh": 5})
        self.assertEqual(report["public"]["rejected"]["over_token_budget"], 1)
        self.assertTrue(all({key: value for key, value in row.items() if key != "validation_group_id"} in self.public for row in public))
        self.assertTrue(all(row["copyright"] == "retain me" for row in public))
        self.assertTrue(all(row["validation_group_id"] for row in rows))
        self.assertEqual(report["internal_validation"]["group_overlap"], 0)
        self.assertEqual(report["internal_validation"]["groups"], len(self.synthetic) + 10)
        self.assertTrue(all(len(row["text"]) <= 512 for row in rows))
        for name, data in before.items():
            self.assertEqual((output / name).read_bytes(), data)
            self.assertEqual((self.parent / name).read_bytes(), data)
            self.assertEqual(report["files"][name]["sha256"], hashlib.sha256(data).hexdigest())
        self.assertIn("independent", report["public"]["ranking_domain"])

    def test_reproducible_contents_and_refuses_existing_output(self):
        first, second = self.root / "first", self.root / "second"
        prepare(self.parent, first, seed=54, public_count=12, length_fn=len)
        prepare(self.parent, second, seed=54, public_count=12, length_fn=len)
        for name in ("train.jsonl", "heldout.jsonl", "eval_tasks.jsonl", "manifest.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        before = (first / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(self.parent, first, length_fn=len)
        self.assertEqual((first / "manifest.json").read_bytes(), before)

    def test_wrong_reference_is_rejected_before_publication(self):
        rows = read_jsonl(self.parent / "train.jsonl")
        rows[0]["answer"] = "not the computed answer"
        self.write("train.jsonl", rows)
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "answer"):
            prepare(self.parent, output, length_fn=len)
        self.assertFalse(output.exists())

    def test_stream_read_preserves_unicode_line_separator_inside_valid_json(self):
        path = self.root / "unicode.jsonl"
        path.write_text(json.dumps({"text": "a\u2028b\u0085c"}, ensure_ascii=False) + "\n", encoding="utf-8")
        self.assertEqual(read_jsonl(path), [{"text": "a\u2028b\u0085c"}])

    def test_public_ranking_is_independent_of_the_trainer_split_hash(self):
        # Ranking the lowest trainer split hashes would force this selection
        # into validation. Verify the actual chosen rows differ from that rank.
        rows = []
        for index in range(100):
            rows.append({"text": f"<|user|> distinct {index}\n<|assistant|> reply {index}<|eos|>",
                         "source": "smoltalk", "source_file": "one"})
        selected, _ = select_public(rows, 10, 55, len, 512, [])
        old_rank = sorted(rows, key=lambda row: hashlib.sha256(
            f"55:{hashlib.sha256(row['text'].encode()).hexdigest()}".encode()).hexdigest())[:10]
        self.assertNotEqual({row["text"] for row in selected}, {row["text"] for row in old_rank})

    def test_public_group_uses_user_sequence_instead_of_answer(self):
        first = {"text": "<|user|> Same user\n<|assistant|> first answer<|eos|>"}
        second = {"text": "<|user|> Same user\n<|assistant|> another answer<|eos|>"}
        different = {"text": "<|user|> Different user\n<|assistant|> first answer<|eos|>"}
        self.assertEqual(public_validation_group(first), public_validation_group(second))
        self.assertNotEqual(public_validation_group(first), public_validation_group(different))


if __name__ == "__main__":
    unittest.main()

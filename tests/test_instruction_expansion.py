"""Fresh contrastive instruction families preserve heldout and source boundaries."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deploy.prepare_instruction_expansion import (
    STRING_TASKS, generate_families, longest_complete_prefix, prepare, read_jsonl,
)
from deploy.prepare_recovery_curriculum import (
    RESERVED_PROBES, answer_for, generate_synthetic, parse_conversation,
    stable_hash, verify_generated_splits,
)


class InstructionExpansionTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.directory.name)
        assert self.root.resolve().is_relative_to(cache.resolve())
        self.parent = self.root / "parent"
        self.parent.mkdir()
        self.synthetic = generate_synthetic(200, 80, 291)
        self.public = [{"text": f"<|user|> {lang} question {index}\n<|assistant|> answer {index}<|eos|>",
                        "source": source, "source_file": "original.jsonl", "copyright": "keep unchanged"}
                       for lang, source in (("zh", "COIG-CQIA"), ("en", "smoltalk")) for index in range(12)]
        self.public_path = self.root / "public.jsonl"
        self.write(self.public_path, self.public)
        self.write(self.parent / "train.jsonl", self.synthetic["train"] + self.public[:3])
        self.write(self.parent / "heldout.jsonl", self.synthetic["heldout"])
        self.write(self.parent / "eval_tasks.jsonl", [{"prompt": row["prompt"], "accepted": [row["answer"]]}
                                                   for row in self.synthetic["heldout"][:20]])
        (self.parent / "manifest.json").write_text('{"fixture":true}\n', encoding="utf-8")
        (self.parent / "upstream_sources.json").write_text('{"copyright":"retain"}\n', encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def write(path, rows):
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def build(self, name="new", **changes):
        options = dict(seed=292, string_families=8, arithmetic_families=12, other_rows=8,
                       public_count=12, max_tokens=512, length_fn=len, validation_fraction=.2)
        options.update(changes)
        output = self.root / name
        report = prepare(self.parent, self.public_path, output, **options)
        return output, report, read_jsonl(output / "train.jsonl")

    def test_string_families_contrast_five_tasks_in_two_languages_and_stay_grouped(self):
        _, report, rows = self.build()
        families = defaultdict(list)
        for row in rows:
            if row.get("task") in STRING_TASKS:
                families[row["validation_group_id"]].append(row)
        self.assertEqual(len(families), 8)
        for group in families.values():
            self.assertEqual(len(group), 10)
            self.assertEqual({(row["task"], row["language"]) for row in group},
                             {(task, language) for task in STRING_TASKS for language in ("en", "zh")})
            self.assertEqual(len({row["arguments"]["text"] for row in group}), 1)
        self.assertEqual(set(report["generation"]["string_families_by_length_band"].values()), {2})
        self.assertEqual(report["generation"]["parent_semantic_overlap"], 0)
        self.assertTrue(all(value == 0 for value in report["generated_split_checks"].values()))

    def test_new_arithmetic_pairs_share_all_operations_languages_and_do_not_repeat_parent_pairs(self):
        _, report, rows = self.build()
        arithmetic = [row for row in rows if row.get("task") in ("add", "subtract", "multiply")]
        groups = defaultdict(list)
        for row in arithmetic:
            groups[row["validation_group_id"]].append(row)
        self.assertEqual(len(groups), 12)
        old_pairs = {tuple(sorted((row["arguments"]["a"], row["arguments"]["b"])))
                     for row in self.synthetic["train"] if row["task"] in ("add", "subtract", "multiply")}
        for group in groups.values():
            self.assertEqual(len(group), 6)
            self.assertEqual({row["task"] for row in group}, {"add", "subtract", "multiply"})
            pairs = {(row["arguments"]["a"], row["arguments"]["b"]) for row in group}
            self.assertEqual(len(pairs), 1)
            self.assertNotIn(tuple(sorted(next(iter(pairs)))), old_pairs)
        self.assertEqual(report["generation"]["small_arithmetic_selected"], 3)
        self.assertEqual(report["generation"]["expanded_arithmetic_selected"], 9)

    def test_small_domain_exhaustion_redirects_with_an_explicit_report(self):
        # Forbid all small integers in heldout fixtures. Expanded numbers remain
        # available; an unavailable 25% quota must not loop or silently reuse.
        heldout = [dict(self.synthetic["heldout"][0], operands=list(range(100)))]
        rows, stats = generate_families(self.synthetic["train"], heldout, seed=294,
                                       string_families=0, arithmetic_families=8, other_rows=0)
        self.assertEqual(len(rows), 48)
        self.assertEqual(stats["small_arithmetic_selected"], 0)
        self.assertEqual(stats["small_arithmetic_redirected_to_new_domain"], 2)
        self.assertTrue(all(100 <= value <= 999 for row in rows for value in row["operands"]))

    def test_all_generated_references_semantics_payloads_and_split_metadata_are_verified(self):
        _, report, rows = self.build()
        generated = [row for row in rows if row.get("task")]
        self.assertEqual(len(generated), 8 * 10 + 12 * 6 + 8)
        parent_keys = {row["semantic_key"] for row in self.synthetic["train"]}
        old_payloads = {value.casefold() for row in [*self.synthetic["train"], *self.synthetic["heldout"]]
                        for value in row["payloads"]}
        for row in generated:
            self.assertNotIn(row["semantic_key"], parent_keys)
            self.assertFalse({value.casefold() for value in row["payloads"]} & old_payloads)
            self.assertEqual(row["answer"], answer_for(row["task"], row["arguments"]))
            self.assertEqual(parse_conversation(row["text"]), [(row["prompt"], row["answer"])])
        self.assertTrue(all(value == 0 for value in verify_generated_splits(generated, self.synthetic["heldout"]).values()))
        self.assertEqual(report["internal_validation"]["group_overlap"], 0)
        self.assertEqual(report["internal_validation"]["training_rows"] + report["internal_validation"]["validation_rows"], len(rows))
        self.assertEqual(Counter(row["language"] for row in generated), {"en": 80, "zh": 80})

    def test_public_longest_prefix_keeps_context_true_eos_and_first_user_group(self):
        first = "<|user|> shared first question\n<|assistant|> first complete answer<|eos|>"
        second = "<|user|> dependent followup\n<|assistant|> another complete answer<|eos|>"
        third = "<|user|> final followup\n<|assistant|> " + "x" * 700 + "<|eos|>"
        full = first + second + third
        prefix, kept, total = longest_complete_prefix(full, len, len(first + second))
        self.assertEqual((prefix, kept, total), (first + second, 2, 3))
        self.assertEqual(parse_conversation(prefix), parse_conversation(full)[:2])
        self.assertEqual(longest_complete_prefix(full, len, 16)[0], None)
        shared = {"text": full, "source": "smoltalk", "source_file": "multi", "copyright": "original license"}
        variant = {**shared, "text": first.replace("first complete answer", "a different first answer")}
        self.write(self.public_path, [shared, variant])
        _, report, rows = self.build(public_count=10, max_tokens=512)
        public = [row for row in rows if not row.get("task")]
        self.assertEqual(len(public), 2)
        self.assertEqual(len({row["validation_group_id"] for row in public}), 1)
        for row in public:
            self.assertEqual(row["copyright"], "original license")
            self.assertEqual(row["source_file"], "multi")
            self.assertLessEqual(len(row["text"]), 512)
            self.assertTrue(row["text"].endswith("<|eos|>"))
        derived = next(row for row in public if row["expansion_extraction"]["kept_turns"] == 2)
        self.assertEqual(derived["expansion_parent_text_sha256"], stable_hash(full))
        self.assertEqual(derived["text"], first + second)
        self.assertEqual(report["public"]["selected_shortened_prefixes"], 1)
        self.assertEqual(report["public"]["selected"], 2)

    def test_public_exclusions_metadata_overlap_and_bounded_capacity(self):
        forbidden = {"text": f"<|user|> {RESERVED_PROBES[0]}\n<|assistant|> hi<|eos|>", "source": "smoltalk"}
        heldout_prompt = {"text": f"<|user|> {self.synthetic['heldout'][0]['prompt']}\n<|assistant|> no<|eos|>", "source": "smoltalk"}
        payload = next(value for row in self.synthetic["heldout"] for value in row["payloads"])
        leaked_payload = {"text": f"<|user|> discuss {payload}\n<|assistant|> no<|eos|>", "source": "smoltalk"}
        malformed = {"text": "<|user|> missing answer", "source": "smoltalk"}
        self.write(self.public_path, [*self.public, forbidden, heldout_prompt, leaked_payload, malformed])
        previous = self.root / "previous.jsonl"
        self.write(previous, self.public[:4])
        _, report, rows = self.build(public_count=1000, previous_data_paths=[previous])
        public = [row for row in rows if not row.get("task")]
        self.assertEqual(len(public), 24)
        self.assertEqual(report["public"]["selected_text_previously_in_parent"], 3)
        self.assertEqual(report["public"]["selected_text_previously_in_additional_data"][str(previous.resolve())], 4)
        self.assertGreaterEqual(report["public"]["rejected"]["reserved_or_heldout_prompt"], 2)
        self.assertEqual(report["public"]["rejected"]["heldout_payload_word"], 1)
        self.assertTrue(all(row["copyright"] == "keep unchanged" for row in public))
        self.assertTrue(report["public"]["prefix_rejected"])
        for row in public:
            original = next(item for item in self.public if item["text"] == row["text"])
            self.assertTrue(all(row[key] == value for key, value in original.items()))

    def test_reproducible_hashes_byte_heldout_copy_and_existing_output_refusal(self):
        before = {name: (self.parent / name).read_bytes() for name in ("heldout.jsonl", "eval_tasks.jsonl", "upstream_sources.json")}
        output, report, _ = self.build("first")
        second, _, _ = self.build("second")
        for name in ("train.jsonl", "heldout.jsonl", "eval_tasks.jsonl", "manifest.json"):
            self.assertEqual((output / name).read_bytes(), (second / name).read_bytes())
        for name, content in before.items():
            self.assertEqual((output / name).read_bytes(), content)
            self.assertEqual((self.parent / name).read_bytes(), content)
            self.assertEqual(report["files"][name]["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(report["builder_sha256"], hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("deploy/prepare_instruction_expansion.py").read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            self.build("first")

    def test_invalid_counts_and_corrupt_parent_fail_before_output_publication(self):
        for options in (dict(string_families=-1), dict(arithmetic_families=True), dict(other_rows=3),
                        dict(public_count=-2), dict(max_tokens=15), dict(validation_fraction=1.)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build("invalid", **options)
            self.assertFalse((self.root / "invalid").exists())
        broken = read_jsonl(self.parent / "train.jsonl")
        broken[0]["answer"] = "wrong"
        self.write(self.parent / "train.jsonl", broken)
        with self.assertRaisesRegex(ValueError, "answer"):
            self.build("corrupt")
        self.assertFalse((self.root / "corrupt").exists())


if __name__ == "__main__":
    unittest.main()

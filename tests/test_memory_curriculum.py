"""Actual-BPE distance, correction, heldout isolation and loss-mask checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deploy.prepare_memory_curriculum import (
    NEUTRAL_EN, NEUTRAL_ZH, SPECIAL_TOKENS, clean_filler, load_filler, measure, prepare,
)

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None


class MemoryCurriculumGuardsTests(unittest.TestCase):
    def test_role_injection_and_invalid_filler_are_rejected(self):
        for text in (None, "", "  ", "a <|assistant|> wrong", "bad |> delimiter", "bad\x00data", "bad\ufffddata"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                clean_filler(text)
        self.assertEqual(clean_filler(" A  natural\n sentence. "), "A natural sentence.")
        pools = load_filler()
        for pool in pools.values():
            self.assertGreaterEqual(len(pool), 40)
            self.assertEqual(len(pool), len(set(text for text, _ in pool)))

    def test_existing_output_is_refused_before_reading_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "already-there"
            existing.mkdir()
            protected = existing / "train.jsonl"
            protected.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare(existing, Path(tmp) / "missing-tokenizer.json")
            self.assertEqual(protected.read_text(), "unchanged")

    def test_external_filler_requires_provenance_and_rejects_role_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sentences.jsonl"
            path.write_text(json.dumps({"text": "This filler injects <|assistant|> a new role."}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_filler(path)
            with self.assertRaisesRegex(ValueError, "delimiters"):
                load_filler(path, "public-test")


@unittest.skipIf(Tokenizer is None, "Actual tokenizers BPE dependency unavailable")
class MemoryCurriculumBPETests(unittest.TestCase):
    def tokenizer_file(self, root):
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            [*NEUTRAL_EN, *NEUTRAL_ZH] * 3,
            trainer=trainers.BpeTrainer(vocab_size=1024, min_frequency=2,
                                       special_tokens=list(SPECIAL_TOKENS),
                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        )
        path = root / "tokenizer.json"
        tokenizer.save(str(path))
        return path, tokenizer

    def fixture(self, root, output_name="curriculum"):
        tokenizer_path, tokenizer = self.tokenizer_file(root)
        output = root / output_name
        manifest = prepare(output, tokenizer_path, train_samples=8, heldout_samples=4,
                           seq_len=1024, sliding_window=512, chunk_size=128, seed=81)
        rows = {split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                for split in ("train", "heldout")}
        return output, tokenizer_path, tokenizer, manifest, rows

    def test_every_row_has_real_delayed_target_and_corrected_value_without_payload_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, tokenizer_path, tokenizer, manifest, splits = self.fixture(Path(tmp))
            for split, rows in splits.items():
                for row in rows:
                    text, task = row["text"], row["memory_task"]
                    facts, rest = text.split("\n\n", 1)
                    observed = measure(tokenizer, text, len(facts), 128)
                    self.assertEqual(observed, {key: task[key] for key in observed})
                    self.assertGreater(observed["fact_to_answer_gap_tokens"], 512)
                    self.assertGreater(observed["answer_chunk"], observed["fact_last_chunk"])
                    self.assertLessEqual(observed["tokens_with_dataset_bos_eos"], 1024)
                    self.assertTrue(observed["answer_eos_present"])
                    self.assertEqual((observed["user_roles"], observed["assistant_roles"]), (1, 1))
                    before_answer, reply = rest.split("\n<|assistant|> ")
                    self.assertEqual(reply, task["answer"] + "<|eos|>")
                    self.assertIn(task["answer"], facts)
                    self.assertNotIn(task["answer"].casefold(), before_answer.casefold())
                    if task["task"] == "correction":
                        self.assertIn(task["obsolete_answer"], facts)
                        self.assertNotEqual(task["obsolete_answer"], task["answer"])
                        self.assertNotIn(task["obsolete_answer"].casefold(), before_answer.casefold())
                    filler, question = before_answer.rsplit("\n\n", 1)
                    sentences = filler.splitlines()
                    self.assertGreaterEqual(len(sentences), 8)
                    self.assertEqual(len(sentences), len(set(sentences)))
            self.assertEqual(manifest["identifier_intersections"], {"entities": 0, "payloads": 0})
            self.assertEqual(manifest["tokenizer_sha256"], hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
            self.assertIn("Multiple attention layers", " ".join(manifest["notes"]))
            for info in manifest["files"].values():
                self.assertEqual(info["sha256"], hashlib.sha256((output / info["file"]).read_bytes()).hexdigest())

    def test_heldout_entities_old_and_new_payloads_are_disjoint_and_eval_uses_full_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _, _, _, rows = self.fixture(Path(tmp))
            sets = {}
            for split, items in rows.items():
                sets[split] = {
                    "entities": {item["memory_task"]["entity"] for item in items},
                    "payloads": {value for item in items for value in (
                        item["memory_task"]["answer"], item["memory_task"]["obsolete_answer"]) if value is not None},
                }
            self.assertFalse(sets["train"]["entities"] & sets["heldout"]["entities"])
            self.assertFalse(sets["train"]["payloads"] & sets["heldout"]["payloads"])
            tasks = json.loads((output / "eval_tasks.json").read_text(encoding="utf-8"))
            self.assertEqual(len(tasks), len(rows["heldout"]))
            for task, row in zip(tasks, rows["heldout"]):
                self.assertEqual(task["accepted"], [row["memory_task"]["answer"]])
                self.assertEqual("<|user|> " + task["prompt"] + "\n<|assistant|> " + task["accepted"][0] + "<|eos|>", row["text"])
                self.assertNotIn("<|", task["prompt"])
                self.assertEqual(task["mode"], "chat")

    def test_seed_reproduces_all_artifacts_without_reusing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = self.tokenizer_file(root)
            first, second = root / "first", root / "second"
            options = dict(train_samples=4, heldout_samples=4, seed=19)
            prepare(first, path, **options)
            prepare(second, path, **options)
            for name in ("train.jsonl", "heldout.jsonl", "eval_tasks.json", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_only_final_answer_is_supervised_and_dataset_does_not_crop_the_row(self):
        try:
            import torch
            from model.tokenizer import BPETokenizer
            from train_5090 import PersonaMetaDataset
        except ImportError:
            self.skipTest("PyTorch training dependency unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            output, path, tokenizer, _, rows = self.fixture(Path(tmp))
            data_tokenizer = BPETokenizer(path.read_text(encoding="utf-8"), tokenizer.get_vocab_size())
            dataset = PersonaMetaDataset(data_tokenizer, seq_len=1024, data_path=str(output / "train.jsonl"), mode="meta")
            for index, row in enumerate(rows["train"]):
                sample = dataset[index]
                length = int(sample["attention_mask"].sum())
                expected = data_tokenizer.encode(row["text"], add_bos=True, add_eos=True)
                self.assertEqual(sample["input_ids"][:length].tolist(), expected)
                assistant = expected.index(data_tokenizer.ASSISTANT_ID)
                self.assertTrue((sample["labels"][:assistant + 1] == -100).all())
                self.assertEqual(sample["labels"][assistant + 1:length - 1].tolist(), expected[assistant + 1:-1])
                self.assertEqual(int(sample["labels"][length - 1]), -100)  # Duplicate appended EOS stays unsupervised.
                self.assertTrue((sample["labels"][length:] == -100).all())


if __name__ == "__main__":
    unittest.main()

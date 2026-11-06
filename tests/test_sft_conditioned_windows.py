import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from deploy.audit_instruction_data import audit, first_target_has_question, legacy_window
from deploy.train_tokenizer import train_tokenizer
from model.uttt_model import SimpleUTF8Tokenizer
from train_5090 import PersonaMetaDataset


class ConditionedSFTWindowTests(unittest.TestCase):
    def dataset(self, text, seq_len=48, split="all", mode="sft"):
        dataset = PersonaMetaDataset(SimpleUTF8Tokenizer(), seq_len=seq_len, split=split, mode=mode)
        dataset.external_texts = [text]
        return dataset

    def assert_conditioned(self, dataset, batch):
        ids = batch["input_ids"].tolist()
        labels = batch["labels"].tolist()
        first = next(i for i, label in enumerate(labels) if label != -100)
        assistant = max(i for i, token in enumerate(ids[:first]) if token == dataset.tokenizer.ASSISTANT_ID)
        user = max(i for i, token in enumerate(ids[:assistant]) if token == dataset.tokenizer.USER_ID)
        self.assertLess(user, assistant)
        self.assertTrue(all(label == -100 for label in labels[:assistant + 1]))
        self.assertEqual(ids[0], dataset.tokenizer.BOS_ID)
        return ids, labels, user, assistant

    def test_long_answer_starts_at_answer_and_keeps_question_without_fake_eos(self):
        text = "<|user|> Which letter?\n<|assistant|> " + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4
        dataset = self.dataset(text)
        for _ in range(8):
            batch = dataset[0]
            ids, labels, user, assistant = self.assert_conditioned(dataset, batch)
            reply = dataset.tokenizer.decode([token for token in labels if token != -100])
            self.assertTrue(reply.startswith(" ABCDE"), reply)
            question = dataset.tokenizer.decode(ids[user + 1:assistant])
            self.assertIn("Which letter?", question)
            self.assertNotIn(dataset.tokenizer.EOS_ID, ids)

    def test_very_long_question_keeps_head_tail_and_complete_short_answer_eos(self):
        text = "<|user|> BEGIN_" + "x" * 200 + "_END\n<|assistant|> Yes.<|eos|>"
        dataset = self.dataset(text, seq_len=32)
        ids, labels, user, assistant = self.assert_conditioned(dataset, dataset[0])
        question = dataset.tokenizer.decode(ids[user + 1:assistant])
        self.assertIn("BEGIN_", question)
        self.assertIn("_END", question)
        self.assertEqual(dataset.tokenizer.decode([token for token in labels if token != -100]), " Yes.")
        self.assertIn(dataset.tokenizer.EOS_ID, labels)

    def test_validation_window_is_stable_and_does_not_consume_training_rng(self):
        text = "<|user|> one?\n<|assistant|> " + "A" * 100 + "<|eos|>\n<|user|> two?\n<|assistant|> " + "B" * 100
        dataset = self.dataset(text, split="validation")
        torch.manual_seed(15)
        before = torch.get_rng_state().clone()
        first, second = dataset[0], dataset[0]
        torch.testing.assert_close(first["input_ids"], second["input_ids"])
        torch.testing.assert_close(first["labels"], second["labels"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assert_conditioned(dataset, first)

    def test_training_turn_choice_is_replayable_and_covers_both_questions(self):
        text = "<|user|> first?\n<|assistant|> " + "A" * 100 + "<|eos|>\n<|user|> second?\n<|assistant|> " + "B" * 100
        dataset = self.dataset(text)
        torch.manual_seed(33)
        state = torch.get_rng_state().clone()
        batches = [dataset[0] for _ in range(20)]
        answers = set()
        for batch in batches:
            ids, labels, user, assistant = self.assert_conditioned(dataset, batch)
            question = dataset.tokenizer.decode(ids[user + 1:assistant])
            answer = dataset.tokenizer.decode([token for token in labels if token != -100]).strip()
            answers.add(answer[0])
            self.assertEqual(answer[0], "A" if "first?" in question else "B")
        self.assertEqual(answers, {"A", "B"})
        torch.set_rng_state(state)
        for expected in batches:
            actual = dataset[0]
            torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
            torch.testing.assert_close(actual["labels"], expected["labels"])

    def test_complete_recent_history_is_preserved_when_it_fits(self):
        text = ("<|user|> " + "old" * 50 + "<|assistant|> old answer<|eos|>\n"
                "<|user|> middle?<|assistant|> middle.<|eos|>\n"
                "<|user|> newest?<|assistant|> newest.<|eos|>")
        dataset = self.dataset(text, seq_len=96, split="validation")
        ids, labels = dataset._document(text)
        # Select the last answered turn with a deterministic hash, independently
        # of the actual question content used to construct the token document.
        choice_text = next(str(i) for i in range(100)
                           if int(hashlib.sha256(str(i).encode()).hexdigest()[:16], 16) % 3 == 2)
        window, targets = dataset._sft_window(ids, labels, choice_text)
        decoded = dataset.tokenizer.decode(window)
        self.assertIn("middle?", decoded)
        self.assertIn("newest?", decoded)
        self.assertNotIn("oldoldold", decoded)
        self.assertEqual(dataset.tokenizer.decode([token for token in targets if token != -100]), " newest.")

    def test_short_rows_and_non_sft_modes_retain_their_existing_semantics(self):
        short = "<|user|> Q?<|assistant|> A.<|eos|>"
        dataset = self.dataset(short, seq_len=64)
        ids, labels = dataset._document(short)
        actual = dataset[0]
        self.assertEqual(actual["input_ids"].tolist()[:len(ids)], ids)
        self.assertEqual(actual["labels"].tolist()[:len(labels)], labels)
        long = "<|user|> question?<|assistant|> " + "a" * 400
        for mode in ("pretrain", "meta"):
            dataset = self.dataset(long, seq_len=32, mode=mode)
            torch.manual_seed(5)
            windows = [dataset[0]["input_ids"].tolist() for _ in range(20)]
            self.assertTrue(any(dataset.tokenizer.USER_ID not in window for window in windows))

    def test_new_sft_protocol_changes_fingerprint_without_changing_content_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            text = "<|user|> Q?<|assistant|> A."
            path.write_text(json.dumps({"text": text}), encoding="utf-8")
            options = {"tokenizer": SimpleUTF8Tokenizer(), "data_path": str(path)}
            first = PersonaMetaDataset(**options, seq_len=32)
            second = PersonaMetaDataset(**options, seq_len=64)
            legacy = hashlib.sha256(hashlib.sha256(text.encode()).hexdigest().encode()).hexdigest()
            self.assertNotEqual(first.fingerprint, legacy)
            self.assertNotEqual(first.fingerprint, second.fingerprint)
            self.assertEqual(first.external_texts, second.external_texts)
            self.assertEqual(PersonaMetaDataset(**options, mode="meta").fingerprint, legacy)


class InstructionAuditTests(unittest.TestCase):
    def test_actual_bpe_audit_detects_unconditioned_old_windows_and_emits_no_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = "<|user|> PRIVATE_QUESTION\n<|assistant|> " + "PRIVATE_ANSWER " * 100 + "<|eos|>"
            path = root / "rows.jsonl"
            rows = [{"text": text, "source_file": "public-test"}, {"text": text},
                    {"text": "<|user|> Q?<|assistant|> A.<|eos|>", "source_file": "short-test"}]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            tokenizer_path = root / "tokenizer.json"
            train_tokenizer(path, tokenizer_path, 320, validation_fraction=0)
            report = audit(path, tokenizer_path, seq_len=32, sample_size=10, seed=4, trials=8)
            self.assertEqual(report["rows"]["exact_duplicate_rows"], 1)
            self.assertGreater(report["legacy_windows"]["first_target_without_its_question"], 0)
            self.assertEqual(report["conditioned_windows"]["first_target_without_its_question"], 0)
            serialized = json.dumps(report)
            self.assertNotIn("PRIVATE_QUESTION", serialized)
            self.assertNotIn("PRIVATE_ANSWER", serialized)
            self.assertEqual(report, audit(path, tokenizer_path, seq_len=32, sample_size=10, seed=4, trials=8))


if __name__ == "__main__":
    unittest.main()

"""Related SFT paraphrases must remain in a single CE-validation partition."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    import yaml
    from model.uttt_model import SimpleUTF8Tokenizer
    import train_5090 as training
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class ValidationGroupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def rows(self):
        return [{"text": f"<|user|> question {group}, wording {variant}?<|assistant|> answer {group}.<|eos|>",
                 "validation_group_id": f"g{group:02}"}
                for group in range(40) for variant in range(3)]

    def write(self, rows, name="rows.jsonl"):
        path = self.root / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    def dataset(self, path, **kwargs):
        return training.PersonaMetaDataset(SimpleUTF8Tokenizer(), seq_len=32, data_path=str(path), **kwargs)

    def test_all_paraphrases_stay_in_one_partition_with_fixed_namespace(self):
        rows = self.rows()
        path = self.write(rows)
        options = {"validation_fraction": .25, "split_seed": 23,
                   "validation_group_key": "validation_group_id"}
        train = self.dataset(path, split="train", **options)
        validation = self.dataset(path, split="validation", **options)
        text_to_group = {row["text"]: row["validation_group_id"] for row in rows}
        train_groups = {text_to_group[text] for text in train.external_texts}
        validation_groups = {text_to_group[text] for text in validation.external_texts}
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(train_groups | validation_groups, {row["validation_group_id"] for row in rows})
        self.assertEqual(len(train.external_texts), len(train_groups) * 3)
        self.assertEqual(len(validation.external_texts), len(validation_groups) * 3)
        # Frozen expected membership also guards the namespaced split hash.
        self.assertEqual(validation_groups, EXPECTED_VALIDATION_GROUPS)
        reordered = self.dataset(self.write(list(reversed(rows)), "reversed.jsonl"), split="validation", **options)
        self.assertEqual(set(reordered.external_texts), set(validation.external_texts))

    def test_default_none_preserves_previous_fingerprints_and_ignores_group_metadata(self):
        rows = [{"text": "<|user|> Q1?<|assistant|> A1.<|eos|>", "validation_group_id": "first"},
                {"content": "<|user|> Q2?<|assistant|> A2.<|eos|>", "validation_group_id": "second"},
                {"text": "<|user|> Q1?<|assistant|> A1.<|eos|>", "validation_group_id": "ignored-duplicate"}]
        path = self.write(rows)
        self.assertEqual(self.dataset(path).fingerprint, LEGACY_SFT_FINGERPRINT)
        self.assertEqual(self.dataset(path, validation_group_key=None).fingerprint, LEGACY_SFT_FINGERPRINT)
        self.assertEqual(self.dataset(path, mode="meta").fingerprint, LEGACY_TEXT_FINGERPRINT)
        self.assertEqual(self.dataset(path, mode="pretrain").fingerprint, LEGACY_TEXT_FINGERPRINT)

    def test_group_key_group_identity_and_text_changes_are_fingerprinted(self):
        rows = [{"text": "<|user|> Q?<|assistant|> A.<|eos|>", "group": "same", "other": "same"}]
        path = self.write(rows)
        original = self.dataset(path, validation_group_key="group")
        other_key = self.dataset(path, validation_group_key="other")
        self.assertEqual(original.external_texts, other_key.external_texts)
        self.assertNotEqual(original.fingerprint, other_key.fingerprint)
        self.assertNotEqual(original.fingerprint, self.dataset(path).fingerprint)
        changed_group = [{**rows[0], "group": "different"}]
        self.assertNotEqual(original.fingerprint, self.dataset(self.write(changed_group, "changed-group.jsonl"),
                                                               validation_group_key="group").fingerprint)
        changed_text = [{**rows[0], "text": rows[0]["text"].replace("A.", "B.")}]
        self.assertNotEqual(original.fingerprint, self.dataset(self.write(changed_text, "changed-text.jsonl"),
                                                               validation_group_key="group").fingerprint)
        duplicated = self.dataset(self.write(rows + rows, "duplicated.jsonl"), validation_group_key="group")
        self.assertEqual(original.fingerprint, duplicated.fingerprint)
        self.assertEqual(len(duplicated), 1)

    def test_contradictory_groups_for_identical_normalized_text_fail_before_split_filter(self):
        text = "<|user|> Q?<|assistant|> A.<|eos|>"
        rows = [{"text": text, "validation_group_id": "g00"},
                {"text": text, "validation_group_id": "g01"}]
        path = self.write(rows)
        for split in ("all", "train", "validation"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "contradictory validation group"):
                self.dataset(path, split=split, validation_fraction=.5,
                             validation_group_key="validation_group_id", split_seed=23)
        # The dataset's historical category-prefix normalization must not hide
        # contradictory group declarations behind otherwise different rows.
        rows[0]["text"] = "<|user|> [Context Category: one]\n" + text
        rows[1]["text"] = "<|user|> [Context Category: two]\n" + text
        with self.assertRaisesRegex(ValueError, "contradictory validation group"):
            self.dataset(self.write(rows, "normalized.jsonl"), validation_group_key="validation_group_id")

    def test_invalid_mode_key_missing_fields_and_nonjson_rows_are_rejected(self):
        path = self.write(self.rows())
        for mode in ("pretrain", "meta"):
            with self.assertRaisesRegex(ValueError, "validation_group_key"):
                self.dataset(path, mode=mode, validation_group_key="validation_group_id")
        for key in ("", "  ", 1, False):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "validation_group_key"):
                self.dataset(path, validation_group_key=key)
        for value in (None, "", " ", 1, False, [], {}):
            row = {"text": "<|user|> Q?<|assistant|> A.<|eos|>", "validation_group_id": value}
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonempty string validation_group_id"):
                self.dataset(self.write([row], "invalid.jsonl"), validation_group_key="validation_group_id")
        with self.assertRaisesRegex(ValueError, "nonempty string validation_group_id"):
            self.dataset(self.write([{"text": "Q"}], "missing.jsonl"), validation_group_key="validation_group_id")
        path = self.root / "plain.txt"
        path.write_text("<|user|> Q?<|assistant|> A.<|eos|>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON rows"):
            self.dataset(path, validation_group_key="validation_group_id")
        with self.assertRaisesRegex(ValueError, "external JSONL"):
            training.PersonaMetaDataset(SimpleUTF8Tokenizer(), validation_group_key="group")

    def test_training_and_validation_constructors_both_receive_the_group_policy(self):
        path = self.write(self.rows())
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump({
            "model": {"vocab_size": 512, "d_model": 8, "n_layers": 1, "n_heads": 2,
                      "n_kv_heads": 1, "d_ff_fast": 16, "chunk_size": 4},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": "sft", "micro_batch_size": 1, "grad_accum_steps": 1,
                         "max_steps": 1, "seq_len": 128, "pack_sequences": False,
                         "gradient_checkpointing": False, "validation_fraction": .25, "seed": 23,
                         "eval_samples": 2, "validation_group_key": "validation_group_id"},
        }), encoding="utf-8")
        original, calls = training.PersonaMetaDataset, []

        class RecordingDataset(original):
            def __init__(self, *args, **kwargs):
                calls.append((kwargs.get("split"), kwargs.get("validation_group_key")))
                super().__init__(*args, **kwargs)

        with redirect_stdout(io.StringIO()), mock.patch.object(training, "PersonaMetaDataset", RecordingDataset):
            saved = training.train(str(config), str(path), str(self.root / "full.pt"))
        self.assertEqual(calls, [("train", "validation_group_id"), ("validation", "validation_group_id")])
        checkpoint = torch.load(saved, map_location="cpu", weights_only=False)
        # The selected sequence length also belongs to the fingerprint.
        expected = original(SimpleUTF8Tokenizer(), seq_len=128, data_path=str(path), split="train",
                            validation_fraction=.25, split_seed=23, validation_group_key="validation_group_id")
        self.assertEqual(checkpoint["metrics"]["dataset_fingerprint"], expected.fingerprint)


LEGACY_SFT_FINGERPRINT = "4deb35ef5ee1308649d48b1874acaedf3d93183c831517e2732fcbf0ce7168aa"
LEGACY_TEXT_FINGERPRINT = "8c19655c20a2b0b1ec7b402164cd0ce5c307b6f37c600238eaa1bd6e7dfe98c7"
EXPECTED_VALIDATION_GROUPS = {"g02", "g09", "g21", "g27", "g29", "g34", "g39"}


if __name__ == "__main__":
    unittest.main()

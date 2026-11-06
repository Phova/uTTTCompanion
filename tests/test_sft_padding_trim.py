"""Shorter SFT execution must preserve valid targets, gradients and data replay."""
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
    from torch.utils.data import DataLoader, Dataset
    from engine.corpus import ResumableBatchSampler
    from model.uttt_model import uTTTConfig, uTTTModel
    import train_5090 as training
except ImportError:
    torch = None


def batch(lengths, width=1024):
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.float32)
    for row, length in enumerate(lengths):
        ids[row, :length] = torch.randint(6, 64, (length,))
        ids[row, length - 1] = 2  # Real EOS remains a supervised shifted target.
        labels[row, :length] = ids[row, :length]
        labels[row, :min(8, length - 1)] = -100
        mask[row, :length] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class SftPaddingTrimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_execution_policy_is_explicit_and_excludes_meta_dropout_and_compile(self):
        self.assertFalse(training.validate_sft_padding_trim(False, "meta", .1, True))
        self.assertTrue(training.validate_sft_padding_trim(True, "sft", 0., False))
        for settings in ((True, "pretrain", 0., False), (True, "meta", 0., False),
                         (True, "sft", .1, False), (True, "sft", 0., True),
                         ("false", "sft", 0., False)):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "trim_sft_padding"):
                training.validate_sft_padding_trim(*settings)
        training.validate_resume_sft_padding({}, False)
        training.validate_resume_sft_padding({"trim_sft_padding": True}, True)
        for metadata, enabled in (({}, True), ({"trim_sft_padding": True}, False),
                                  ({"trim_sft_padding": "false"}, False)):
            with self.assertRaisesRegex(ValueError, "trim_sft_padding"):
                training.validate_resume_sft_padding(metadata, enabled)

    def test_batch_max_rounding_preserves_eos_supervision_metadata_and_token_counts(self):
        original = batch([129, 257])
        original["sample_id"] = torch.tensor([17, 19])
        before = {name: value.clone() for name, value in original.items()}
        rng = torch.get_rng_state().clone()
        trimmed = training.trim_sft_batch_padding(original)
        self.assertEqual(trimmed["input_ids"].shape, (2, 384))
        for name in ("input_ids", "labels", "attention_mask"):
            torch.testing.assert_close(trimmed[name], before[name][:, :384])
            self.assertTrue(trimmed[name].is_contiguous())
        for row, length in enumerate((129, 257)):
            self.assertEqual(int(trimmed["labels"][row, length - 1]), 2)
        self.assertEqual(int(original["attention_mask"].sum()), int(trimmed["attention_mask"].sum()))
        self.assertEqual(int((original["labels"][:, 1:] != -100).sum()),
                         int((trimmed["labels"][:, 1:] != -100).sum()))
        for name in original:
            torch.testing.assert_close(original[name], before[name])
        torch.testing.assert_close(trimmed["sample_id"], before["sample_id"])
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for lengths, width, expected in (([123], 1024, 128), ([128], 1024, 128),
                                         ([511], 1024, 512), ([513], 1024, 640),
                                         ([1000], 1024, 1024), ([129], 130, 130)):
            with self.subTest(lengths=lengths, width=width):
                self.assertEqual(training.trim_sft_batch_padding(batch(lengths, width))["labels"].shape[1], expected)

    def test_unsafe_or_empty_masks_are_rejected_without_removing_targets(self):
        bad = batch([50], 256)
        bad["attention_mask"][0, 8] = 0
        with self.assertRaisesRegex(ValueError, "right padding"):
            training.trim_sft_batch_padding(bad)
        bad = batch([50], 256)
        bad["labels"][0, 200] = 7
        with self.assertRaisesRegex(ValueError, "supervised padding"):
            training.trim_sft_batch_padding(bad)
        for value in (.5, float("nan")):
            bad = batch([50], 256)
            bad["attention_mask"][0, 80] = value
            with self.assertRaisesRegex(ValueError, "binary"):
                training.trim_sft_batch_padding(bad)
        bad = batch([50, 50], 256)
        bad["attention_mask"][1].zero_()
        with self.assertRaisesRegex(ValueError, "at least two"):
            training.trim_sft_batch_padding(bad)
        bad = batch([50], 256)
        bad["labels"] = bad["labels"][:, :-1]
        with self.assertRaisesRegex(ValueError, "matching"):
            training.trim_sft_batch_padding(bad)

    def test_real_frozen_read_logits_ce_and_every_parameter_gradient_match(self):
        for lengths, expected_width in (([123], 128), ([350, 499], 512), ([610, 699], 768)):
            with self.subTest(lengths=lengths):
                torch.manual_seed(47)
                config = uTTTConfig(vocab_size=64, d_model=16, n_layers=2, n_heads=2,
                                    n_kv_heads=1, d_ff_fast=32, chunk_size=128,
                                    sliding_window_size=512, dropout=0., gradient_checkpointing=False)
                full, short = uTTTModel(config).train(), uTTTModel(config).train()
                short.load_state_dict(full.state_dict())
                original = batch(lengths)
                trimmed = training.trim_sft_batch_padding(original)
                self.assertEqual(trimmed["input_ids"].shape[1], expected_width)
                before = full(**original, update_fast_weights=False, return_fast_state=False)
                after = short(**trimmed, update_fast_weights=False, return_fast_state=False)
                valid = original["attention_mask"].bool()
                torch.testing.assert_close(after.logits[valid[:, :expected_width]], before.logits[valid],
                                           atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(after.ce_loss, before.ce_loss, atol=1e-6, rtol=1e-6)
                before.loss.backward()
                after.loss.backward()
                for (name, left), (_, right) in zip(full.named_parameters(), short.named_parameters()):
                    self.assertEqual(left.grad is None, right.grad is None, name)
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_collation_preserves_random_crops_sampler_cursor_and_rng(self):
        class Crops(Dataset):
            def __len__(self):
                return 7

            def __getitem__(self, index):
                length = int(torch.randint(20, 300, ()))
                item = batch([length], 512)
                item = {key: value[0] for key, value in item.items()}
                item["index"] = torch.tensor(index)
                return item

        def collect(collator):
            sampler = ResumableBatchSampler(7, 2, 17)
            loader = DataLoader(Crops(), batch_sampler=sampler, collate_fn=collator,
                                generator=torch.Generator().manual_seed(91))
            rows = []
            for item in loader:
                for row in range(item["input_ids"].shape[0]):
                    length = int(item["attention_mask"][row].sum())
                    rows.append((int(item["index"][row]),
                                 item["input_ids"][row, :length].clone(), item["labels"][row, :length].clone()))
                sampler.advance(item["input_ids"].shape[0])
            return rows, sampler.state_dict(), torch.get_rng_state().clone()

        torch.manual_seed(19)
        initial = torch.get_rng_state().clone()
        expected, cursor, rng = collect(None)
        torch.set_rng_state(initial)
        actual, actual_cursor, actual_rng = collect(training.collate_trimmed_sft)
        self.assertEqual(cursor, actual_cursor)
        torch.testing.assert_close(rng, actual_rng)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertEqual(left[0], right[0])
            torch.testing.assert_close(left[1], right[1])
            torch.testing.assert_close(left[2], right[2])

    def test_training_uses_cpu_trim_and_resume_cannot_silently_change_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train.jsonl"
            data.write_text("\n".join(json.dumps({"text": f"<|user|> {prompt}\n<|assistant|> {answer}<|eos|>"})
                                      for prompt, answer in (("question one", "answer one"),
                                                             ("question two", "answer two"))), encoding="utf-8")
            config = {"model": {"vocab_size": 512, "d_model": 8, "n_layers": 1, "n_heads": 2,
                                "n_kv_heads": 1, "d_ff_fast": 16, "chunk_size": 4, "dropout": 0.},
                      "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
                      "training": {"stage": "sft", "micro_batch_size": 1, "grad_accum_steps": 1,
                                   "max_steps": 1, "seq_len": 256, "pack_sequences": False,
                                   "gradient_checkpointing": False, "trim_sft_padding": True}}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            widths = []
            forward = training.uTTTModel.forward

            def observe(model, *args, **kwargs):
                widths.append(kwargs["input_ids"].shape[1])
                self.assertFalse(kwargs["update_fast_weights"])
                return forward(model, *args, **kwargs)

            with redirect_stdout(io.StringIO()), mock.patch.object(training.uTTTModel, "forward", new=observe):
                checkpoint = training.train(str(path), str(data), str(root / "full.pt"))
            self.assertEqual(widths, [128])
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertTrue(saved["metrics"]["trim_sft_padding"])
            before = checkpoint.read_bytes()
            config["training"]["trim_sft_padding"] = False
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "trim_sft_padding"):
                training.train(str(path), str(data), str(checkpoint), override_steps=2)
            self.assertEqual(checkpoint.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from deploy.prepare_dataset import format_sharegpt_conversation
from deploy.train_tokenizer import train_tokenizer
from engine.checkpoint_manager import DualTrackCheckpointManager
from engine.personalizer import uTTTPersonalizer
from model.tokenizer import build_tokenizer
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel
from train_5090 import PersonaMetaDataset, train


class LanguageDataTests(unittest.TestCase):
    def test_short_instructions_and_ending_are_preserved(self):
        text = format_sharegpt_conversation({"category": "irrelevant", "conversation": [{"human": "Hello", "assistant": "Hi!"}]})
        self.assertEqual(text, "<|user|> Hello\n<|assistant|> Hi!<|eos|>")

    def test_long_user_prompt_does_not_hide_answer_or_train_prompt_as_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.txt"
            path.write_text("<|user|> " + "a" * 200 + "<|assistant|> Yes.")
            dataset = PersonaMetaDataset(SimpleUTF8Tokenizer(), seq_len=16, data_path=str(path))
            torch.manual_seed(19)
            batches = [dataset[0] for _ in range(30)]
            self.assertTrue(all((batch["labels"][1:] != -100).any() for batch in batches))
            self.assertTrue(any((batch["labels"] == SimpleUTF8Tokenizer.EOS_ID).any() for batch in batches))
            for batch in batches:
                self.assertFalse((batch["labels"] == SimpleUTF8Tokenizer.BYTE_OFFSET + ord("a")).any())

    def test_deduplication_precedes_conversation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            texts = [f"<|user|> question {i}<|assistant|> answer {i}" for i in range(100)]
            path.write_text("\n".join(json.dumps({"text": text}) for text in texts * 2))
            options = dict(tokenizer=SimpleUTF8Tokenizer(), data_path=str(path), validation_fraction=0.2)
            train_set = PersonaMetaDataset(**options, split="train")
            validation = PersonaMetaDataset(**options, split="validation")
            self.assertFalse(set(train_set.external_texts) & set(validation.external_texts))
            self.assertEqual(len(train_set) + len(validation), len(texts))
            first, second = validation[0], validation[0]
            torch.testing.assert_close(first["input_ids"], second["input_ids"])

    def test_sft_never_falls_back_to_supervising_user_only_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.txt"
            path.write_text("<|user|> no assistant answer here")
            dataset = PersonaMetaDataset(SimpleUTF8Tokenizer(), data_path=str(path), seq_len=16)
            self.assertTrue((dataset[0]["labels"] == -100).all())
            pretrain = PersonaMetaDataset(SimpleUTF8Tokenizer(), data_path=str(path), seq_len=16, mode="pretrain")
            self.assertTrue((pretrain[0]["labels"][1:] != -100).any())


class LanguagePipelineTests(unittest.TestCase):
    def test_bpe_is_lossless_compresses_text_and_travels_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.txt"
            sample = "Hello world, hello world! 你好世界 🙂"
            data.write_text("\n".join([sample] * 50), encoding="utf-8")
            trained = train_tokenizer(data, root / "tokenizer.json", 512)
            config = uTTTConfig(vocab_size=512, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff_fast=32,
                                tokenizer_json=trained.to_str())
            tokenizer = build_tokenizer(config)
            self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)
            self.assertLess(len(tokenizer.encode(sample)), len(SimpleUTF8Tokenizer(512).encode(sample)))
            unseen = "éλ雪🦄"
            self.assertEqual(tokenizer.decode(tokenizer.encode(unseen)), unseen)
            model = uTTTModel(config)
            self.assertEqual(model.active_vocab_limit, tokenizer.vocab_size)
            checkpoint = root / "base.pt"
            torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict()}, checkpoint)
            manager = DualTrackCheckpointManager(checkpoint, root / "personas", device="cpu")
            self.assertEqual(build_tokenizer(manager.config).decode(tokenizer.encode(sample)), sample)

    def test_stages_validation_and_safe_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps({"text": f"<|user|> {i}?<|assistant|> {i}."}) for i in range(40)))
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "model": {"vocab_size": 512, "d_model": 16, "n_layers": 1, "n_heads": 2,
                          "n_kv_heads": 1, "d_ff_fast": 32, "chunk_size": 4},
                "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
                "training": {"micro_batch_size": 2, "grad_accum_steps": 1, "max_steps": 2,
                             "seq_len": 16, "gradient_checkpointing": True, "validation_fraction": 0.2,
                             "eval_interval": 1, "eval_samples": 2, "save_interval": 2},
            }))
            initial = None
            for stage in ("pretrain", "sft", "meta"):
                output = root / f"{stage}.pt"
                train(str(config), str(data), str(output), stage=stage, init_from=initial)
                saved = torch.load(output, weights_only=False)
                self.assertEqual(saved["metrics"]["training_stage"], stage)
                self.assertGreater(saved["metrics"]["tokens_seen"], 0)
                self.assertTrue(torch.isfinite(torch.tensor(saved["metrics"]["validation_ce"])))
                self.assertTrue(output.with_suffix(".best.pt").exists())
                manager = DualTrackCheckpointManager(output, root / f"personas_{stage}", device="cpu")
                self.assertEqual(uTTTPersonalizer(manager).online_updates_enabled, stage == "meta")
                if stage != "meta":
                    result = uTTTPersonalizer(manager).interact_and_evolve("test", "hello", max_new_tokens=1)
                    self.assertFalse(result.gate_triggered)
                    self.assertIn("Online TTT disabled", result.status_bar)
                initial = str(output)
            before = (root / "meta.pt").read_bytes()
            with self.assertRaisesRegex(ValueError, "Resume checkpoint"):
                train(str(config), str(data), str(root / "meta.pt"), stage="sft")
            self.assertEqual(before, (root / "meta.pt").read_bytes())


if __name__ == "__main__":
    unittest.main()

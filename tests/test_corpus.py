import hashlib
import json
import pickle
import os
import signal
import subprocess
import sys
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from deploy.prepare_pretrain_corpus import normalize, identity_and_split
from deploy.train_tokenizer import train_tokenizer
from deploy.download_pretrain import download
from deploy.prepare_instructions import format_messages
from engine.corpus import TokenizedCorpusDataset
from train_5090 import train


class CorpusTests(unittest.TestCase):
    def test_completed_partial_download_is_published_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"a previously completed shard"
            (root / "shard.part").write_bytes(content)
            entry = {"local_file": "shard.parquet", "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                     "download_url": "not-a-network-url"}
            download(entry, root)
            self.assertEqual((root / "shard.parquet").read_bytes(), content)

    def test_instruction_format_preserves_constraints_and_stop_tokens(self):
        text = format_messages([{"role": "system", "content": "Use one word."},
                                {"role": "user", "content": "Name a color."},
                                {"role": "assistant", "content": "Red"}])
        self.assertEqual(text, "<|user|> Use one word.\n\nName a color.\n<|assistant|> Red<|eos|>")

    def test_packed_windows_cover_boundaries_and_workers_reopen_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.asarray([1, 10, 11, 2, 1, 12, 13, 14, 15, 2], dtype=np.uint16)
            values.tofile(root / "train.bin")
            manifest = {"format": "uttt-token-corpus-v1", "dtype": "uint16", "fingerprint": "test",
                        "splits": {"train": {"file": "train.bin", "tokens": len(values)}}}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            dataset = TokenizedCorpusDataset(path, seq_len=4)
            self.assertIsNone(dataset._tokens)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0]["labels"].tolist(), [-100, 10, 11, 2])
            self.assertEqual(dataset[1]["input_ids"].tolist(), [2, 1, 12, 13])
            self.assertEqual(dataset[2]["input_ids"][-1], 2)
            copied = pickle.loads(pickle.dumps(dataset))
            self.assertIsNone(copied._tokens)
            torch.testing.assert_close(copied[1]["input_ids"], dataset[1]["input_ids"])

    def test_cleaning_and_split_are_content_stable(self):
        text = "科学与教育提供可靠知识。" * 20
        cleaned = normalize("  " + text + "\n\n")
        self.assertEqual(cleaned, text)
        self.assertEqual(identity_and_split(cleaned), identity_and_split(text))
        self.assertIsNone(normalize("x"))
        self.assertIsNone(normalize("\ufffd" * 200))

    def test_training_reads_token_manifest_and_exports_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = root / "text.txt"
            text.write_text("Hello world! " * 100)
            tokenizer = train_tokenizer(text, root / "tokenizer.json", 512, validation_fraction=0)
            values = np.asarray([1, *tokenizer.encode("Hello world! " * 100).ids, 2], dtype=np.uint16)
            manifest = {"format": "uttt-token-corpus-v1", "dtype": "uint16", "fingerprint": "fixture",
                        "tokenizer": "tokenizer.json", "tokenizer_sha256": hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest(),
                        "splits": {}}
            for split in ("train", "validation"):
                values.tofile(root / f"{split}.bin")
                manifest["splits"][split] = {"file": f"{split}.bin", "tokens": len(values)}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "model": {"vocab_size": 512, "d_model": 16, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "d_ff_fast": 32},
                "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
                "training": {"stage": "pretrain", "micro_batch_size": 1, "grad_accum_steps": 1, "max_steps": 1,
                             "seq_len": 16, "eval_samples": 2, "eval_prompts": ["Hello"], "eval_new_tokens": 2,
                             "tokenizer_path": "ignored_default.json"},
            }))
            checkpoint = train(str(config), str(path), str(root / "base.pt"))
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(saved["metrics"]["dataset_fingerprint"], "fixture:train")
            self.assertIn("validation_ce", saved["metrics"])
            self.assertEqual(len(saved["metrics"]["validation_samples"]), 1)
            self.assertIsNotNone(saved["config"]["tokenizer_json"])

    @unittest.skipUnless(os.name == "posix", "Linux/WSL pipeline")
    def test_pipeline_stop_forwards_to_child_and_does_not_start_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "deploy").mkdir()
            script = root / "deploy" / "run_training_pipeline.py"
            script.write_text((Path(__file__).resolve().parents[1] / "deploy" / script.name).read_text())
            (root / "data/pretrain_test/tokenized").mkdir(parents=True)
            (root / "data/pretrain_test/tokenized/manifest.json").write_text("{}")
            (root / "data/instruction_test").mkdir()
            (root / "data/instruction_test/train.jsonl").write_text("{}")
            (root / "train_5090.py").write_text(
                "import os,signal,sys,time\n"
                "signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))\n"
                "open('child_started','w').write(str(os.getpid()))\n"
                "while True: time.sleep(0.1)\n"
            )
            command = [sys.executable, str(script), "--run-id", "test", "--pretrain-steps", "10"]
            parent = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                while not (root / "child_started").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue((root / "child_started").exists())
                child_pid = int((root / "child_started").read_text())
                duplicate = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                self.assertNotEqual(duplicate.returncode, 0)
                parent.terminate()
                self.assertNotEqual(parent.wait(timeout=10), 0)
                state = json.loads((root / "logs/pipeline_test.json").read_text())
                self.assertEqual(state["status"], "interrupted")
                self.assertFalse((root / "logs/sft_test.log").exists())
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()


if __name__ == "__main__":
    unittest.main()

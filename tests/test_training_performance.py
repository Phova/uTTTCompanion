"""Numerical safeguards for opt-in training performance settings."""
import unittest
from contextlib import nullcontext
from pathlib import Path
import tempfile

import torch
import yaml

from model.uttt_model import uTTTConfig, uTTTModel
from train_5090 import accumulation_autocast, same_training_data, train


class TrainingPerformanceTests(unittest.TestCase):
    def test_dataset_fingerprint_survives_path_aliases_without_relaxing_legacy_checks(self):
        self.assertTrue(same_training_data(
            {"dataset_fingerprint": "verified", "data_source": "/old/snapshot/train.txt"},
            "verified", "/new/snapshot/train.txt",
        ))
        self.assertFalse(same_training_data(
            {"dataset_fingerprint": "other-content", "data_source": "/same/train.txt"},
            "verified", "/same/train.txt",
        ))
        self.assertTrue(same_training_data({"data_source": "/same/train.txt"}, "verified", "/same/train.txt"))
        self.assertFalse(same_training_data({"data_source": "/old/train.txt"}, "verified", "/new/train.txt"))
        self.assertFalse(same_training_data({}, "verified", "/new/train.txt"))

    def test_resume_restores_weights_after_snapshot_path_changes_and_rejects_unverified_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train.txt"
            data.write_text("<|user|> a <|assistant|> b", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "model": {"vocab_size": 512, "d_model": 16, "n_layers": 1, "n_heads": 2,
                          "n_kv_heads": 1, "d_ff_fast": 32, "chunk_size": 4},
                "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
                "training": {"stage": "pretrain", "micro_batch_size": 1, "grad_accum_steps": 1,
                             "max_steps": 1, "seq_len": 16, "outer_lr": 0.0, "meta_fast_lr": 0.0,
                             "gradient_checkpointing": False},
            }), encoding="utf-8")
            checkpoint = train(str(config), str(data), str(root / "base.pt"))
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved["metrics"]["data_source"] = "/retired/run-snapshot/data/train.txt"
            # A fresh run reaches the requested step too. A sentinel weight and
            # the Adam step count prove that both model and optimizer resumed.
            saved["slow_weights"]["tok_embeddings.weight"].fill_(0.125)
            saved["slow_weights"]["lm_head.weight"].fill_(0.125)
            torch.save(saved, checkpoint)
            train(str(config), str(data), str(checkpoint), override_steps=2)
            resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(resumed["step"], 2)
            self.assertEqual(resumed["metrics"]["data_source"], str(data.resolve()))
            self.assertTrue(torch.all(resumed["slow_weights"]["tok_embeddings.weight"] == 0.125))
            parameter_steps = [int(state["step"]) for state in resumed["optimizer_state"]["state"].values()
                               if "step" in state]
            self.assertTrue(parameter_steps)
            self.assertTrue(all(step == 2 for step in parameter_steps))

            del resumed["metrics"]["dataset_fingerprint"]
            torch.save(resumed, checkpoint)
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "Resume checkpoint"):
                train(str(config), str(data), str(checkpoint), override_steps=3)
            self.assertEqual(before, checkpoint.read_bytes())

    def test_accumulation_amp_cache_preserves_gradients_and_parameter_updates(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device_name in devices:
            for checkpointed in (False, True):
                with self.subTest(device=device_name, checkpointed=checkpointed):
                    device = torch.device(device_name)
                    config = uTTTConfig(
                        vocab_size=512, d_model=16, n_layers=2, n_heads=2,
                        n_kv_heads=1, d_ff_fast=32, chunk_size=4,
                        sliding_window_size=16, gradient_checkpointing=checkpointed,
                    )
                    torch.manual_seed(731)
                    reference = uTTTModel(config).to(device).train()
                    cached = uTTTModel(config).to(device).train()
                    cached.load_state_dict(reference.state_dict())
                    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3) for model in (reference, cached)]
                    batches = [torch.randint(6, 512, (1, 8), device=device) for _ in range(3)]
                    # More than one optimizer update catches a cache that survives
                    # parameter mutation; separate backwards catch reused graph state.
                    for _ in range(2):
                        gradients = []
                        for model, optimizer, use_cache in zip((reference, cached), optimizers, (False, True)):
                            optimizer.zero_grad(set_to_none=True)
                            with accumulation_autocast(device_name, torch.bfloat16, use_cache):
                                for ids in batches:
                                    with torch.autocast(device_name, dtype=torch.bfloat16):
                                        out = model(ids, labels=ids, update_fast_weights=False, return_fast_state=False)
                                        loss = out.loss / len(batches)
                                    with (torch.autocast(device_name, enabled=False) if use_cache else nullcontext()):
                                        loss.backward()
                            gradients.append({name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                                              if parameter.grad is not None})
                            optimizer.step()
                        self.assertEqual(gradients[0].keys(), gradients[1].keys())
                        for name in gradients[0]:
                            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-7, rtol=1e-6, msg=name)
                        for (name, expected), (_, actual) in zip(reference.named_parameters(), cached.named_parameters()):
                            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6, msg=name)


if __name__ == "__main__":
    unittest.main()

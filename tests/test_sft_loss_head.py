"""Sparse SFT vocabulary projection must preserve the complete CE objective."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from engine.sft_execution import supervised_token_indices, validate_sft_compute_option
from model.uttt_model import uTTTConfig, uTTTModel
from train_5090 import train


class SftLossHeadTests(unittest.TestCase):
    def test_indices_keep_shifted_answers_and_real_eos_only(self):
        labels = torch.tensor([[-100, -100, 11, 2, -100], [-100, 7, -100, 8, 2]])
        self.assertEqual(supervised_token_indices(labels).tolist(), [1, 2, 4, 6, 7])
        for invalid in (labels.float(), labels[0], torch.ones(2, 1, dtype=torch.long)):
            with self.assertRaisesRegex(ValueError, "CPU long labels"):
                supervised_token_indices(invalid)

    def test_feature_gate_rejects_meta_pretrain_and_compilation(self):
        for stage in ("sft", "meta", "pretrain"):
            self.assertFalse(validate_sft_compute_option("head", False, stage=stage, torch_compile=True))
        self.assertTrue(validate_sft_compute_option("head", True, stage="sft", torch_compile=False))
        for value in (0, 1, "false", None):
            with self.assertRaisesRegex(ValueError, "boolean"):
                validate_sft_compute_option("head", value, stage="sft", torch_compile=False)
        for stage, compile_value in (("meta", False), ("pretrain", False), ("sft", True), ("sft", 0)):
            with self.assertRaisesRegex(ValueError, "requires stage=sft"):
                validate_sft_compute_option("head", True, stage=stage, torch_compile=compile_value)

    def test_loss_all_parameter_gradients_and_prompt_gradients_match(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for bsz in (1, 2, 4):
                for reduction in ("token_mean", "sequence_mean"):
                    for checkpointed in (False, True):
                        with self.subTest(device=device, bsz=bsz, reduction=reduction, checkpointed=checkpointed):
                            torch.manual_seed(114)
                            config = uTTTConfig(vocab_size=512, d_model=16, n_layers=2, n_heads=2,
                                                n_kv_heads=1, d_ff_fast=32, chunk_size=4,
                                                sliding_window_size=16, gradient_checkpointing=checkpointed,
                                                tie_word_embeddings=False)
                            reference = uTTTModel(config).to(device=device, dtype=torch.float64).train()
                            selected = copy.deepcopy(reference)
                            ids = torch.arange(6, 6 + bsz * 12).reshape(bsz, 12).to(device)
                            labels = torch.full_like(ids, -100, device="cpu")
                            for row in range(bsz):
                                labels[row, 7:9 + row % 3] = ids[row, 7:9 + row % 3].cpu()
                                labels[row, 8 + row % 3] = 2
                            indices = supervised_token_indices(labels)
                            expected = reference(ids, labels=labels.to(device), update_fast_weights=False,
                                                 return_fast_state=False, ce_reduction=reduction)
                            actual = selected(ids, labels=labels.to(device), update_fast_weights=False,
                                              return_fast_state=False, ce_reduction=reduction, ce_token_indices=indices)
                            self.assertEqual(actual.logits.shape, (bsz, 0, 512))
                            torch.testing.assert_close(actual.loss, expected.loss, atol=1e-11, rtol=1e-10)
                            expected.loss.backward()
                            actual.loss.backward()
                            for (name, param), (_, other) in zip(reference.named_parameters(), selected.named_parameters()):
                                self.assertEqual(param.grad is None, other.grad is None, name)
                                if param.grad is not None:
                                    torch.testing.assert_close(other.grad, param.grad, atol=1e-10, rtol=1e-8, msg=name)
                            self.assertGreater(float(selected.tok_embeddings.weight.grad[6].abs().sum()), 0.)

    def test_bf16_unequal_targets_and_empty_sequences_match(self):
        torch.manual_seed(24)
        config = uTTTConfig(vocab_size=512, d_model=16, n_layers=1, n_heads=2,
                            n_kv_heads=1, d_ff_fast=32, chunk_size=4, gradient_checkpointing=False)
        reference = uTTTModel(config).train()
        selected = copy.deepcopy(reference)
        ids = torch.randint(6, 512, (4, 12))
        for all_ignored in (False, True):
            labels = torch.full_like(ids, -100)
            if not all_ignored:
                labels[0, 7:9] = ids[0, 7:9]
                labels[1, 6:] = ids[1, 6:]
                labels[3, 11] = 2
            for model, indices in ((reference, None), (selected, supervised_token_indices(labels))):
                model.zero_grad(set_to_none=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(ids, labels=labels, update_fast_weights=False, return_fast_state=False,
                                   ce_reduction="sequence_mean", ce_token_indices=indices)
                output.loss.backward()
                if model is reference:
                    loss = output.loss.detach()
                else:
                    torch.testing.assert_close(output.loss, loss, atol=3e-3, rtol=3e-3)
            for (name, param), (_, other) in zip(reference.named_parameters(), selected.named_parameters()):
                self.assertEqual(param.grad is None, other.grad is None, name)
                if param.grad is not None:
                    torch.testing.assert_close(other.grad, param.grad, atol=3e-3, rtol=.04, msg=name)

    def test_loss_only_contract_rejects_writes_and_invalid_indices(self):
        model = uTTTModel(uTTTConfig(vocab_size=512, d_model=16, n_layers=1, n_heads=2,
                                     n_kv_heads=1, d_ff_fast=32, chunk_size=4))
        ids = torch.ones(1, 8, dtype=torch.long) * 6
        for indices in (torch.tensor([1, 1]), torch.tensor([-1]), torch.tensor([7]),
                        torch.tensor([2, 1]), torch.tensor([1.]), torch.ones(1, 1, dtype=torch.long)):
            with self.assertRaisesRegex(ValueError, "ce_token_indices"):
                model(ids, labels=ids, update_fast_weights=False, ce_token_indices=indices)
        with self.assertRaisesRegex(ValueError, "update_fast_weights=False"):
            model(ids, labels=ids, ce_token_indices=torch.tensor([1]))

    def test_training_opt_in_is_recorded_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train.txt"
            data.write_text("<|user|> Say hi <|assistant|> Hello", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "model": {"vocab_size": 512, "d_model": 16, "n_layers": 1, "n_heads": 2,
                          "n_kv_heads": 1, "d_ff_fast": 32, "chunk_size": 4},
                "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
                "training": {"stage": "sft", "micro_batch_size": 1, "grad_accum_steps": 1,
                             "max_steps": 1, "seq_len": 32, "gradient_checkpointing": False,
                             "sft_loss_only_head": True, "sft_shared_readonly_gemm": True},
            }), encoding="utf-8")
            output = root / "test.pt"
            train(str(config), str(data), str(output))
            train(str(config), str(data), str(output), override_steps=2)
            saved = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(saved["step"], 2)
            for key in ("sft_loss_only_head", "sft_shared_readonly_gemm"):
                self.assertIs(saved["metrics"][key], True)


if __name__ == "__main__":
    unittest.main()

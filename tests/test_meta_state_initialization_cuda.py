"""Autocast may round read activations, never the mutable W/S/LN master state."""
import unittest
from unittest.mock import patch

import torch

from model.uttt_model import uTTTConfig, uTTTModel


STATE_NAMES = ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down", "ln_weight", "ln_bias")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast regression")
class MetaStateInitializationCudaTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(203)
        return uTTTModel(uTTTConfig(
            vocab_size=64, d_model=12, d_ff_fast=20, n_layers=3,
            n_heads=3, n_kv_heads=1, chunk_size=3, sliding_window_size=5,
            gradient_checkpointing=False,
        )).float().cuda().train()

    def test_implicit_meta_state_has_eight_fp32_masters_without_bf16_roundtrip(self):
        model = self.model()
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]], device="cuda")
        with torch.no_grad():
            model.fast_weight_pool.ln_weight.fill_(1.0001234)
            model.fast_weight_pool.ln_bias.fill_(0.00013731)
        reference = model.init_fast_state(batch_size=1, dtype=torch.float32, detach=False)
        expected = {name: getattr(reference, name).detach().clone() for name in STATE_NAMES}
        captured = []
        original = model.init_fast_state

        def track(*args, **kwargs):
            state = original(*args, **kwargs)
            captured.append(state)
            return state

        with patch.object(model, "init_fast_state", side_effect=track):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                implicit = model(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
        self.assertEqual(len(captured), 1)
        for name in STATE_NAMES:
            tensor = getattr(captured[0], name)
            self.assertEqual(tensor.dtype, torch.float32, name)
            torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0, msg=name)
        self.assertFalse(torch.equal(captured[0].w_up, captured[0].w_up.bfloat16().float()))
        self.assertFalse(torch.equal(captured[0].ln_weight, captured[0].ln_weight.bfloat16().float()))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            explicit = model(ids, labels=ids, fast_state=reference, surprise_aux_weight=0.0, return_fast_state=False)
        torch.testing.assert_close(implicit.logits, explicit.logits, atol=0, rtol=0)
        torch.testing.assert_close(implicit.loss, explicit.loss, atol=0, rtol=0)
        implicit.loss.backward()
        for parameter in model.get_meta_fast_parameters():
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_frozen_readonly_language_path_keeps_activation_precision(self):
        model = self.model().eval()
        ids = torch.tensor([[6, 7, 8, 9]], device="cuda")
        observed = []
        original = model.fast_weight_pool.readonly_state

        def track(*args, **kwargs):
            state = original(*args, **kwargs)
            observed.append(state)
            return state

        with torch.no_grad(), patch.object(model.fast_weight_pool, "readonly_state", side_effect=track), \
                torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(ids, update_fast_weights=False, return_fast_state=False)
        self.assertEqual(len(observed), 1)
        self.assertTrue(all(getattr(observed[0], name).dtype == torch.bfloat16 for name in STATE_NAMES))
        self.assertEqual(output.logits.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

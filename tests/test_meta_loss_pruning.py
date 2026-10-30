"""A disabled surprise objective must not run its higher-order backward graph."""
import copy
import unittest

import torch

from model.uttt_model import uTTTConfig, uTTTModel


class MetaLossPruningTests(unittest.TestCase):
    def make_model(self, checkpointed=False):
        torch.manual_seed(784)
        return uTTTModel(uTTTConfig(
            vocab_size=128, d_model=16, d_ff_fast=32, n_layers=2,
            n_heads=2, n_kv_heads=1, chunk_size=4, sliding_window_size=8,
            gradient_checkpointing=checkpointed, max_inner_grad_norm=1.0,
        )).train()

    def test_disabled_auxiliary_prunes_backward_and_preserves_multichunk_adam_updates(self):
        # Batch 3 exercises the sliced CE implementation as well as normal CE.
        for batch_size in (1, 3):
            for checkpointed in (False, True):
                with self.subTest(batch_size=batch_size, checkpointed=checkpointed):
                    reference = self.make_model(checkpointed)
                    optimized = copy.deepcopy(reference)
                    before = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=0.1)
                    after = torch.optim.AdamW(optimized.parameters(), lr=1e-4, weight_decay=0.1)
                    ids = torch.randint(6, 128, (batch_size, 12))
                    for _ in range(2):
                        before.zero_grad(set_to_none=True)
                        after.zero_grad(set_to_none=True)
                        expected = reference(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
                        actual = optimized(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
                        expected_calls, actual_calls = [], []
                        expected.surprise_loss.register_hook(lambda grad: expected_calls.append(grad))
                        actual.surprise_loss.register_hook(lambda grad: actual_calls.append(grad))
                        self.assertIs(actual.loss, actual.ce_loss)
                        # This reconstructs the old zero-weight connected graph.
                        (expected.ce_loss + 0.0 * expected.surprise_loss).backward()
                        actual.loss.backward()
                        self.assertEqual(len(expected_calls), 1)
                        self.assertEqual(actual_calls, [])
                        for (name, left), (_, right) in zip(reference.named_parameters(), optimized.named_parameters()):
                            self.assertEqual(left.grad is None, right.grad is None, name)
                            if left.grad is not None:
                                self.assertTrue(torch.isfinite(right.grad).all(), name)
                                torch.testing.assert_close(right.grad, left.grad, rtol=1e-5, atol=1e-7, msg=name)
                        before.step()
                        after.step()
                        for (name, left), (_, right) in zip(reference.named_parameters(), optimized.named_parameters()):
                            torch.testing.assert_close(right, left, rtol=1e-6, atol=1e-7, msg=name)

    def test_nonzero_auxiliary_still_contributes_its_requested_gradient(self):
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                model = self.make_model()
                ids = torch.randint(6, 128, (batch_size, 8))
                result = model(ids, labels=ids, surprise_aux_weight=0.125, return_fast_state=False)
                gradients = []
                result.surprise_loss.register_hook(lambda grad: gradients.append(grad))
                result.loss.backward()
                self.assertEqual(len(gradients), 1)
                torch.testing.assert_close(gradients[0], torch.tensor(0.125))

    def test_disabled_auxiliary_metric_remains_separately_differentiable(self):
        model = self.make_model()
        ids = torch.randint(6, 128, (1, 8))
        result = model(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
        self.assertTrue(result.surprise_loss.requires_grad)
        gradients = torch.autograd.grad(result.surprise_loss, model.get_meta_fast_parameters(), allow_unused=True)
        self.assertTrue(any(gradient is not None and gradient.abs().max() > 0 for gradient in gradients))
        self.assertTrue(all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients))

    def test_single_chunk_has_no_artificial_zero_gradient_for_unused_writers(self):
        model = self.make_model()
        ids = torch.randint(6, 128, (1, 4))
        result = model(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
        result.loss.backward()
        # No subsequent chunk can read these writes. Their CE gradient is absent;
        # AdamW therefore skips them instead of decaying an artificial zero grad.
        for block in model.layers:
            for module in (block.uttt.w_k, block.uttt.w_v, block.uttt.w_eta, block.uttt.w_alpha):
                for parameter in module.parameters():
                    self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()

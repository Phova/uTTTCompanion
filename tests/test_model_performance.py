import unittest
import torch

from model.uttt_layer import SharedFastWeightPool, SlidingWindowAttention
from model.uttt_model import uTTTConfig, uTTTModel


def config(**kwargs):
    return uTTTConfig(vocab_size=512, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2,
                      d_ff_fast=64, chunk_size=4, sliding_window_size=8, **kwargs)


class ModelPerformanceTests(unittest.TestCase):
    def test_readonly_training_has_same_logits_gradients_and_no_state_copy(self):
        torch.manual_seed(12)
        reference = uTTTModel(config()).train()
        optimized = uTTTModel(config(gradient_checkpointing=False)).train()
        optimized.load_state_dict(reference.state_dict())
        ids = torch.randint(6, 512, (2, 12))
        expected = reference(ids, labels=ids, update_fast_weights=False)
        result = optimized(ids, labels=ids, update_fast_weights=False, return_fast_state=False)
        self.assertIsNone(result.fast_state)
        torch.testing.assert_close(result.logits, expected.logits)
        expected.loss.backward()
        result.loss.backward()
        for (name, actual), (_, original) in zip(optimized.named_parameters(), reference.named_parameters()):
            if original.grad is not None:
                torch.testing.assert_close(actual.grad, original.grad, msg=name)
        state = optimized.fast_weight_pool.readonly_state(2)
        self.assertEqual(state.s_up.untyped_storage().nbytes(), 4)
        self.assertEqual(state.w_up.data_ptr(), optimized.fast_weight_pool.w_up_init.data_ptr())
        # Public default remains an independent state that callers may mutate.
        public = optimized(ids, update_fast_weights=False).fast_state
        before = optimized.fast_weight_pool.w_up_init.detach().clone()
        public.w_up.add_(1)
        torch.testing.assert_close(optimized.fast_weight_pool.w_up_init, before)

    def test_packed_inference_matches_and_invalidates_on_weight_updates(self):
        torch.manual_seed(2)
        model = uTTTModel(config()).eval()
        ids = torch.randint(6, 512, (1, 12))
        state = model.init_fast_state(detach=True)
        with torch.no_grad():
            reference = model(ids, fast_state=state, update_fast_weights=False).logits
            keys = set(model.state_dict())
            model.prepare_for_inference()
            actual = model(ids, fast_state=state, update_fast_weights=False).logits
            torch.testing.assert_close(actual, reference)
            self.assertEqual(set(model.state_dict()), keys)
            old_pack = model.layers[0].attn._packed_qkv
            model.layers[0].attn.wq.weight.add_(0.1)
            model(ids, fast_state=state, update_fast_weights=False)
            self.assertIsNot(old_pack, model.layers[0].attn._packed_qkv)
            state.w_up.add_(0.05)
            fused = model.fast_weight_pool.forward_readout(torch.randn(1, 3, 32), state)
            self.assertTrue(torch.isfinite(fused).all())
        model.train()
        self.assertIsNone(model.layers[0].attn._packed_qkv)

    def test_prefill_cache_owns_only_window_storage_even_for_single_head(self):
        attention = SlidingWindowAttention(32, 4, 1, window_size=4).eval()
        with torch.no_grad():
            _, cache = attention(torch.randn(1, 32, 32))
        for tensor in cache:
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())

    def test_packed_readout_updates_and_inference_mode_are_correct(self):
        pool = SharedFastWeightPool(32, 64).eval()
        x = torch.randn(1, 3, 32)
        state = pool.init_state(detach=True)
        pool.fused_inference_readout = True
        with torch.no_grad():
            pool.forward_readout(x, state)
            old_pack = state._packed_up_gate
            state.w_gate.add_(0.1)
            actual = pool.forward_readout(x, state)
            self.assertIsNot(old_pack, state._packed_up_gate)
            pool.fused_inference_readout = False
            reference = pool.forward_readout(x, state)
            torch.testing.assert_close(actual, reference)
        with torch.inference_mode():
            state = pool.init_state(detach=True)
            pool.fused_inference_readout = True
            self.assertTrue(torch.isfinite(pool.forward_readout(x, state)).all())


if __name__ == "__main__":
    unittest.main()

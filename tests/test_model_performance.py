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

    def test_short_packed_prefill_cache_does_not_retain_full_qkv_projection(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch in (1, 2):
                for kv_heads in (1, 2, 4):
                    for length in (1, 5, 8):
                        with self.subTest(device=device, batch=batch, kv_heads=kv_heads, length=length):
                            torch.manual_seed(527)
                            plain = SlidingWindowAttention(32, 4, kv_heads, window_size=8).to(device).eval()
                            packed = SlidingWindowAttention(32, 4, kv_heads, window_size=8).to(device).eval()
                            packed.load_state_dict(plain.state_dict())
                            packed.prepare_for_inference()
                            tokens = torch.randn(batch, length + 3, 32, device=device)
                            with torch.no_grad():
                                expected, expected_cache = plain(tokens[:, :length])
                                actual, actual_cache = packed(tokens[:, :length])
                                torch.testing.assert_close(actual, expected)
                                for position in range(length, length + 4):
                                    for cached, reference in zip(actual_cache, expected_cache):
                                        torch.testing.assert_close(cached, reference)
                                        self.assertEqual(cached.untyped_storage().nbytes(),
                                                         cached.numel() * cached.element_size())
                                    if position < length + 3:
                                        token = tokens[:, position:position + 1]
                                        expected, expected_cache = plain(token, expected_cache, start_pos=position)
                                        actual, actual_cache = packed(token, actual_cache, start_pos=position)
                                        torch.testing.assert_close(actual, expected)

    def test_window_one_packed_prefill_and_chunk_decode_keep_compact_current_kv(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch in (1, 2):
                for kv_heads in (1, 4):
                    with self.subTest(device=device, batch=batch, kv_heads=kv_heads):
                        torch.manual_seed(207)
                        plain = SlidingWindowAttention(32, 4, kv_heads, window_size=1).to(device).eval()
                        packed = SlidingWindowAttention(32, 4, kv_heads, window_size=1).to(device).eval()
                        packed.load_state_dict(plain.state_dict())
                        packed.prepare_for_inference()
                        tokens = torch.randn(batch, 6, 32, device=device)
                        position, cache = 0, None
                        with torch.no_grad():
                            for count in (1, 1, 3, 1):
                                actual, cache = packed(tokens[:, position:position + count], cache, start_pos=position)
                                # With window=1, each query can see only its own
                                # token, even when it arrives in a cached chunk.
                                pieces = []
                                for index in range(position, position + count):
                                    expected, reference_cache = plain(tokens[:, index:index + 1], start_pos=index)
                                    pieces.append(expected)
                                torch.testing.assert_close(actual, torch.cat(pieces, dim=1))
                                for tensor, reference in zip(cache, reference_cache):
                                    torch.testing.assert_close(tensor, reference)
                                    self.assertEqual(tensor.shape[2], 1)
                                    self.assertEqual(tensor.untyped_storage().nbytes(),
                                                     tensor.numel() * tensor.element_size())
                                position += count

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

"""Prefill and graph decode must share immutable persona read copies."""
import unittest

import torch

from engine.inference import CUDAGraphDecoder
from model.uttt_layer import SharedFastWeightPool
from model.uttt_model import uTTTConfig, uTTTModel


class SharedReadCacheTests(unittest.TestCase):
    def test_direct_cache_request_reuses_prefill_and_invalidates_after_master_write(self):
        pool = SharedFastWeightPool(8, 16).eval()
        pool.fused_inference_readout = True
        state = pool.init_state(detach=True)
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        with torch.no_grad():
            pool.forward_readout(query, state)
            packed, down = pool.inference_read_weights(state, query.dtype, query.device)
            self.assertIs(packed, state._packed_up_gate)
            self.assertIs(down, state._read_w_down)
            state.w_gate.add_(1e-5)
            updated, updated_down = pool.inference_read_weights(state, query.dtype, query.device)
            self.assertIsNot(updated, packed)
            self.assertIsNot(updated_down, down)
            self.assertEqual(state.w_gate.dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
class GraphSharedReadCacheTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(187)
        config = uTTTConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=2,
                            n_kv_heads=1, d_ff_fast=64, chunk_size=4,
                            sliding_window_size=8, max_seq_len=64,
                            gradient_checkpointing=False)
        model = uTTTModel(config).cuda().bfloat16().eval().requires_grad_(False)
        model.prepare_for_inference()
        state = model.init_fast_state(device="cuda", dtype=torch.float32, detach=True)
        return model, state

    def prefill(self, model, state):
        ids = torch.tensor([[6, 7, 8]], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(ids, fast_state=state, update_fast_weights=False,
                           logits_to_keep=1, return_fast_state=False)
        return result.kv_caches

    def test_graph_borrows_prefill_pack_and_down_without_extra_casts(self):
        model, state = self.make_model()
        caches = self.prefill(model, state)
        packed, down = state._packed_up_gate, state._read_w_down
        decoder = CUDAGraphDecoder(model, state, caches, 3)
        self.assertIs(decoder._packed_up_gate, packed)
        self.assertIs(decoder._read_state.w_down, down)
        self.assertIsNone(decoder._read_state.w_up)
        self.assertIsNone(decoder._read_state.w_gate)
        self.assertTrue(decoder.matches(model, state))
        self.assertEqual(state.w_up.dtype, torch.float32)
        token = torch.tensor([[9]], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model(token, fast_state=state, kv_caches=caches, start_pos=3,
                             update_fast_weights=False, logits_to_keep=1, return_fast_state=False)
            actual = decoder.step(token)
        torch.testing.assert_close(actual.float(), expected.logits[:, -1].float(), atol=.015, rtol=.025)

    def test_master_update_invalidates_graph_and_rebuild_borrows_new_prefill_copies(self):
        model, state = self.make_model()
        caches = self.prefill(model, state)
        old_decoder = CUDAGraphDecoder(model, state, caches, 3)
        old_pack = old_decoder._packed_up_gate
        state.w_up.add_(1e-5)
        self.assertFalse(old_decoder.matches(model, state))
        caches = self.prefill(model, state)
        decoder = CUDAGraphDecoder(model, state, caches, 3)
        self.assertIsNot(decoder._packed_up_gate, old_pack)
        self.assertIs(decoder._packed_up_gate, state._packed_up_gate)
        self.assertIs(decoder._read_state.w_down, state._read_w_down)
        self.assertTrue(decoder.matches(model, state))


if __name__ == "__main__":
    unittest.main()

"""Frozen read caches preserve FP32 persona masters and refresh after writes."""
import unittest

import torch

from model.uttt_layer import SharedFastWeightPool, swiglu_fast_forward


class ReadCacheAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(384)
        self.pool = SharedFastWeightPool(8, 16).eval()
        self.pool.fused_inference_readout = True
        self.state = self.pool.init_state(detach=True)
        self.query = torch.randn(1, 4, 8, dtype=torch.bfloat16)

    def reference(self, query=None):
        query = self.query if query is None else query
        return swiglu_fast_forward(
            query, self.state.w_up, self.state.w_gate, self.state.w_down,
            self.state.ln_weight, self.state.ln_bias, self.pool.eps,
        )

    def test_bf16_reads_reuse_casts_without_rounding_the_fp32_master(self):
        self.state.w_up.add_(1e-5)
        masters = tuple(tensor.clone() for tensor in
                        (self.state.w_up, self.state.w_gate, self.state.w_down))
        with torch.no_grad():
            actual = self.pool.forward_readout(self.query, self.state)
            packed, down = self.state._packed_up_gate, self.state._read_w_down
            repeated = self.pool.forward_readout(self.query, self.state)
            expected = self.reference()
        self.assertEqual(packed.dtype, torch.bfloat16)
        self.assertEqual(down.dtype, torch.bfloat16)
        self.assertIs(self.state._packed_up_gate, packed)
        self.assertIs(self.state._read_w_down, down)
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
        for master, original in zip((self.state.w_up, self.state.w_gate, self.state.w_down), masters):
            self.assertEqual(master.dtype, torch.float32)
            torch.testing.assert_close(master, original, atol=0, rtol=0)

    def test_each_master_write_and_tensor_replacement_refreshes_the_read_copies(self):
        with torch.no_grad():
            self.pool.forward_readout(self.query, self.state)
            for name in ("w_up", "w_gate", "w_down"):
                with self.subTest(name=name):
                    previous_pack, previous_down = self.state._packed_up_gate, self.state._read_w_down
                    getattr(self.state, name).add_(torch.randn_like(getattr(self.state, name)) * 0.1)
                    actual = self.pool.forward_readout(self.query, self.state)
                    self.assertIsNot(self.state._packed_up_gate, previous_pack)
                    self.assertIsNot(self.state._read_w_down, previous_down)
                    torch.testing.assert_close(actual, self.reference(), atol=0.016, rtol=0.016)
            previous = self.state._packed_up_gate
            self.state.w_up = self.state.w_up.clone()
            self.pool.forward_readout(self.query, self.state)
            self.assertIsNot(self.state._packed_up_gate, previous)

    def test_changing_activation_dtype_refreshes_both_read_copies(self):
        with torch.no_grad():
            self.pool.forward_readout(self.query, self.state)
            bf16_pack = self.state._packed_up_gate
            query_fp32 = self.query.float()
            actual = self.pool.forward_readout(query_fp32, self.state)
            self.assertIsNot(self.state._packed_up_gate, bf16_pack)
            self.assertEqual(self.state._packed_up_gate.dtype, torch.float32)
            self.assertEqual(self.state._read_w_down.dtype, torch.float32)
            torch.testing.assert_close(actual, self.reference(query_fp32), atol=1e-6, rtol=1e-5)
            self.pool.forward_readout(self.query, self.state)
            self.assertEqual(self.state._packed_up_gate.dtype, torch.bfloat16)
            self.assertEqual(self.state._read_w_down.dtype, torch.bfloat16)

    def test_gradient_and_inference_tensors_bypass_the_versioned_cache(self):
        differentiable = self.pool.init_state(detach=False)
        result = self.pool.forward_readout(self.query.float(), differentiable)
        result.square().mean().backward()
        self.assertIsNone(differentiable._packed_up_gate)
        self.assertIsNone(differentiable._read_w_down)
        for parameter in (self.pool.w_up_init, self.pool.w_gate_init, self.pool.w_down_init):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with torch.inference_mode():
            inference_state = self.pool.init_state(detach=True)
            self.pool.forward_readout(self.query, inference_state)
            self.assertIsNone(inference_state._packed_up_gate)
            self.assertIsNone(inference_state._read_w_down)


if __name__ == "__main__":
    unittest.main()

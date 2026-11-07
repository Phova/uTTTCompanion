"""Shared-W_0 GEMMs must preserve gradients and isolated persona semantics."""
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from model.uttt_layer import SharedFastWeightPool, swiglu_shared_readonly_forward
from model.uttt_model import uTTTConfig, uTTTModel


class SharedReadonlyGemmTests(unittest.TestCase):
    def test_pool_outputs_and_all_gradients_match_for_shared_weights(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float64, torch.float32, torch.bfloat16):
                for batch in (1, 2, 4):
                    with self.subTest(device=device, dtype=dtype, batch=batch):
                        torch.manual_seed(3801)
                        master_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                        reference = SharedFastWeightPool(16, 32).to(device=device, dtype=master_dtype)
                        optimized = SharedFastWeightPool(16, 32).to(device=device, dtype=master_dtype)
                        optimized.load_state_dict(reference.state_dict())
                        optimized.shared_readonly_gemm = True
                        # Transposed sequence/batch storage also exercises reshape's
                        # differentiable copy, without requiring contiguous inputs.
                        original = torch.randn(7, batch, 16, device=device, dtype=master_dtype).transpose(0, 1)
                        upstream = torch.randn_like(original)
                        results, gradients, inputs = [], [], []
                        for pool in (reference, optimized):
                            query = original.detach().clone().requires_grad_()
                            context = (torch.autocast(device, dtype=dtype) if dtype == torch.bfloat16
                                       else nullcontext())
                            with context:
                                state = pool.readonly_state(batch, dtype=dtype)
                                output = pool.forward_readout(query.to(dtype), state)
                                loss = (output.to(master_dtype) * upstream).sum()
                            loss.backward()
                            results.append(output)
                            gradients.append({name: parameter.grad for name, parameter in pool.named_parameters()})
                            inputs.append(query.grad)
                        tolerance = ({"atol": 1e-10, "rtol": 1e-10} if dtype == torch.float64 else
                                     {"atol": 2e-5, "rtol": 2e-5} if dtype == torch.float32 else
                                     {"atol": 0.04, "rtol": 0.025})
                        torch.testing.assert_close(results[0], results[1], **tolerance)
                        torch.testing.assert_close(inputs[0], inputs[1], **tolerance)
                        exact_gradients = None
                        if dtype == torch.bfloat16:
                            exact = SharedFastWeightPool(16, 32).to(device=device, dtype=torch.float64)
                            exact.load_state_dict(reference.state_dict())
                            exact_query = original.double().detach().requires_grad_()
                            exact_output = exact.forward_readout(exact_query, exact.readonly_state(batch))
                            (exact_output * upstream.double()).sum().backward()
                            exact_gradients = {name: parameter.grad for name, parameter in exact.named_parameters()}
                        for name in gradients[0]:
                            self.assertIsNotNone(gradients[1][name], name)
                            if exact_gradients is None:
                                torch.testing.assert_close(gradients[0][name], gradients[1][name], msg=name, **tolerance)
                                continue
                            # Expanded BMM rounds each sample's weight gradient
                            # to BF16 before summing the batch. A shared GEMM
                            # rounds one reduction over B*T. Cancellation can
                            # make elementwise relative error between these two
                            # rounded results misleading, so compare BOTH with
                            # the same unrounded FP64 mathematical reference.
                            exact_gradient = exact_gradients[name]
                            scale = exact_gradient.norm().clamp_min(1e-12)
                            errors = [((value[name].double() - exact_gradient).norm() / scale).item()
                                      for value in gradients]
                            epsilon = torch.finfo(torch.bfloat16).eps
                            for error in errors:
                                self.assertLessEqual(error, 2 * epsilon, (name, errors))
                            # Bound any new normalized error to one quarter of
                            # BF16's unit roundoff; keep FP32/FP64 checks strict.
                            self.assertLessEqual(errors[1], errors[0] + epsilon / 8, (name, errors))

    def test_full_readonly_language_loss_and_parameter_gradients_match(self):
        torch.manual_seed(290)
        config = uTTTConfig(vocab_size=64, d_model=16, n_layers=2, n_heads=2,
                            n_kv_heads=1, d_ff_fast=32, chunk_size=4,
                            sliding_window_size=8, gradient_checkpointing=False)
        reference, optimized = uTTTModel(config).train(), uTTTModel(config).train()
        optimized.load_state_dict(reference.state_dict())
        optimized.fast_weight_pool.shared_readonly_gemm = True
        ids = torch.randint(6, 64, (4, 12))
        outputs = [model(ids, labels=ids, update_fast_weights=False, return_fast_state=False)
                   for model in (reference, optimized)]
        torch.testing.assert_close(outputs[0].loss, outputs[1].loss, atol=1e-6, rtol=1e-6)
        for output in outputs:
            output.loss.backward()
        for (name, expected), (_, actual) in zip(reference.named_parameters(), optimized.named_parameters()):
            self.assertEqual(expected.grad is None, actual.grad is None, name)
            if expected.grad is not None:
                torch.testing.assert_close(expected.grad, actual.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_only_internal_readonly_views_use_shared_gemm(self):
        pool = SharedFastWeightPool(8, 16)
        pool.shared_readonly_gemm = True
        query = torch.randn(3, 5, 8)
        readonly = pool.readonly_state(3)
        with patch("model.uttt_layer.swiglu_shared_readonly_forward", wraps=swiglu_shared_readonly_forward) as shared:
            pool.forward_readout(query, readonly)
            shared.assert_called_once()
            shared.reset_mock()
            # Mutable user states can contain different memories per batch row.
            persona = pool.init_state(3)
            with torch.no_grad():
                persona.w_up[1].add_(0.2)
            optimized = pool.forward_readout(query, persona)
            pool.shared_readonly_gemm = False
            reference = pool.forward_readout(query, persona)
            torch.testing.assert_close(optimized, reference, atol=0, rtol=0)
            shared.assert_not_called()
            pool.shared_readonly_gemm = True
            for state in (readonly.clone(), readonly.detach(), readonly.to(dtype=torch.float64),
                          pool.init_state(1).ensure_batch_size(3)):
                self.assertFalse(state._shared_readonly)
                pool.forward_readout(query, state)
            # Replacement with distinct per-row tensors invalidates the shared
            # layout even if the internal provenance hint was left in place.
            readonly.w_up = readonly.w_up.clone()
            pool.forward_readout(query, readonly)
            shared.assert_not_called()
        self.assertNotIn("_shared_readonly", readonly.to_state_dict())


if __name__ == "__main__":
    unittest.main()

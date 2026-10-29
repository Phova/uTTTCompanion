"""Run separately on CUDA: compiled BF16 forwards must preserve eager training."""
from contextlib import nullcontext
import importlib.util
import json
import unittest
from unittest import mock

import torch

import model.uttt_layer as layer_ops
from model.uttt_model import uTTTConfig, uTTTModel
from train_5090 import accumulation_autocast, maybe_compile_inner_loop


class CompilationContextTests(unittest.TestCase):
    def test_backward_amp_setting_is_applied_at_lazy_invocation_and_restored(self):
        from torch._functorch import config as aot_config
        if not hasattr(aot_config, "backward_pass_autocast"):
            self.skipTest("PyTorch does not expose the compiled-backward AMP setting")
        observed = []

        def lazily_compiled(value):
            observed.append(aot_config.backward_pass_autocast)
            return value + 1

        with aot_config.patch(backward_pass_autocast="same_as_forward"):
            with mock.patch.object(torch, "compile", return_value=lazily_compiled):
                compiled = layer_ops.compile_with_backward_amp_off(lambda value: value + 1)
            self.assertEqual(aot_config.backward_pass_autocast, "same_as_forward")
            self.assertEqual(compiled(2), 3)
            self.assertEqual(observed, ["off"])
            self.assertEqual(aot_config.backward_pass_autocast, "same_as_forward")


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"),
                     "CUDA and Triton are required for compiled training parity")
class CompiledTrainingTests(unittest.TestCase):
    def test_compiled_bf16_forward_and_non_autocast_backward_match_eager_over_sgd_updates(self):
        from torch._functorch import config as aot_config
        if not hasattr(aot_config, "backward_pass_autocast"):
            self.skipTest("PyTorch does not expose the compiled-backward AMP setting")
        names = ("fused_chunk_ttt_step", "fused_slow_projections", "fused_fast_q_proj",
                 "swiglu_fast_forward", "apply_rope", "_COMPILED_FLEX_ATTN")
        eager_kernels = {name: getattr(layer_ops, name) for name in names}
        ambient_backward_amp = aot_config.backward_pass_autocast
        try:
            maybe_compile_inner_loop({"torch_compile": True, "compile_mode": "default"}, torch.device("cuda"))
            compiled_kernels = {name: getattr(layer_ops, name) for name in names}
            for name in names[:-1]:
                self.assertIsNot(compiled_kernels[name], eager_kernels[name], f"{name} silently stayed eager")

            for checkpointed in (False, True):
                for use_cache in (False, True):
                    for update_fast_weights in (False, True):
                        with self.subTest(checkpointed=checkpointed, amp_cache=use_cache,
                                          update_fast_weights=update_fast_weights):
                            torch.manual_seed(985)
                            config = uTTTConfig(vocab_size=512, d_model=128, n_layers=2, n_heads=1,
                                                n_kv_heads=1, d_ff_fast=256, chunk_size=8,
                                                sliding_window_size=32, gradient_checkpointing=checkpointed)
                            reference = uTTTModel(config).cuda().train()
                            optimized = uTTTModel(config).cuda().train()
                            optimized.load_state_dict(reference.state_dict())
                            batches = [torch.randint(6, 512, (1, 16), device="cuda") for _ in range(2)]
                            with mock.patch.multiple(layer_ops, **eager_kernels):
                                expected = self._trajectory(reference, batches, use_cache, update_fast_weights)
                            with mock.patch.multiple(layer_ops, **compiled_kernels):
                                actual = self._trajectory(optimized, batches, use_cache, update_fast_weights)
                            for step, (wanted, observed) in enumerate(zip(expected, actual), 1):
                                torch.testing.assert_close(observed["loss"], wanted["loss"], atol=0.005, rtol=0.005)
                                self.assertEqual(wanted["gradients"].keys(), observed["gradients"].keys())
                                self._assert_relative_error(observed["gradients"], wanted["gradients"], 0.03,
                                                            f"gradient error after update {step}")
                                self._assert_relative_error(observed["updates"], wanted["updates"], 0.05,
                                                            f"parameter update error after update {step}")
                            self.assertEqual(aot_config.backward_pass_autocast, ambient_backward_amp)
        finally:
            for name, function in eager_kernels.items():
                setattr(layer_ops, name, function)

    def _trajectory(self, model, batches, use_cache, update_fast_weights):
        # SGD maps gradient differences linearly into parameter updates. Adam's
        # first update is g/(abs(g)+eps), amplifying BF16 sign changes near zero;
        # it is measured separately below rather than loosening the parity gate.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.01)
        initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        trajectory = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            with accumulation_autocast("cuda", torch.bfloat16, use_cache):
                for ids in batches:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = model(ids, labels=ids, update_fast_weights=update_fast_weights,
                                       surprise_aux_weight=0.0, return_fast_state=False)
                        loss = result.loss / len(batches)
                    with (torch.autocast("cuda", enabled=False) if use_cache else nullcontext()):
                        loss.backward()
                    losses.append(result.loss.detach())
            gradients = {name: parameter.grad.detach().float().cpu().clone()
                         for name, parameter in model.named_parameters() if parameter.grad is not None}
            for gradient in gradients.values():
                self.assertTrue(torch.isfinite(gradient).all())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True, error_if_nonfinite=True)
            optimizer.step()
            parameters = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
            trajectory.append({"loss": torch.stack(losses).mean().cpu(), "gradients": gradients,
                               "parameters": parameters,
                               "updates": {name: parameters[name] - initial[name] for name in parameters}})
        return trajectory

    def test_adam_updates_follow_each_measured_gradient_and_report_compilation_sensitivity(self):
        from torch._functorch import config as aot_config
        if not hasattr(aot_config, "backward_pass_autocast"):
            self.skipTest("PyTorch does not expose the compiled-backward AMP setting")
        names = ("fused_chunk_ttt_step", "fused_slow_projections", "fused_fast_q_proj",
                 "swiglu_fast_forward", "apply_rope", "_COMPILED_FLEX_ATTN")
        eager_kernels = {name: getattr(layer_ops, name) for name in names}
        torch.manual_seed(985)
        config = uTTTConfig(vocab_size=512, d_model=128, n_layers=2, n_heads=1,
                            n_kv_heads=1, d_ff_fast=256, chunk_size=8,
                            sliding_window_size=32, gradient_checkpointing=False)
        reference = uTTTModel(config).cuda().train()
        optimized = uTTTModel(config).cuda().train()
        optimized.load_state_dict(reference.state_dict())
        ids = torch.randint(6, 512, (1, 16), device="cuda")
        try:
            eager = self._first_adam_update(reference, ids)
            maybe_compile_inner_loop({"torch_compile": True}, torch.device("cuda"), False)
            compiled = self._first_adam_update(optimized, ids)
        finally:
            for name, function in eager_kernels.items():
                setattr(layer_ops, name, function)
        self._assert_relative_error(compiled["gradients"], eager["gradients"], 0.03, "Adam input gradient error")
        actual_difference = compiled["update"] - eager["update"]
        predicted_difference = compiled["predicted_update"] - eager["predicted_update"]
        # This derives an absolute error bound from FP32 parameter rounding,
        # rather than declaring the observed 10% Adam difference acceptable.
        torch.testing.assert_close(actual_difference, predicted_difference, rtol=0.0,
                                   atol=4 * torch.finfo(torch.float32).eps * float(eager["initial"].abs().max()))
        sign_flips = eager["gradient_vector"] * compiled["gradient_vector"] < 0
        squared_difference = actual_difference.double().square()
        diagnostic = {
            "adam_update_relative_l2_error": float(actual_difference.norm() / eager["update"].norm()),
            "adam_update_absolute_rms_error": float(actual_difference.square().mean().sqrt()),
            "adam_update_absolute_max_error": float(actual_difference.abs().max()),
            "gradient_sign_flip_fraction": float(sign_flips.float().mean()),
            "adam_error_energy_on_sign_flips": float(squared_difference[sign_flips].sum()
                                                     / squared_difference.sum().clamp_min(1e-30)),
        }
        print("COMPILED_ADAM_DIAGNOSTIC " + json.dumps(diagnostic), flush=True)

    def _first_adam_update(self, model, ids):
        lr, eps, decay = 2e-4, 1e-8, 0.01
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=decay,
                                     betas=(0.9, 0.95), fused=True)
        initial_by_name = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ids, labels=ids, update_fast_weights=False, surprise_aux_weight=0.0,
                         return_fast_state=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True, error_if_nonfinite=True)
        gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        initial = torch.cat([initial_by_name[name].flatten() for name in gradients])
        gradient_vector = torch.cat([gradient.flatten() for gradient in gradients.values()])
        predicted = initial * (1 - lr * decay) - lr * gradient_vector / (gradient_vector.abs() + eps)
        optimizer.step()
        updated_by_name = dict(model.named_parameters())
        updated = torch.cat([updated_by_name[name].detach().cpu().flatten() for name in gradients])
        torch.testing.assert_close(updated, predicted, rtol=0.0,
                                   atol=2 * torch.finfo(torch.float32).eps * float(initial.abs().max()))
        return {"initial": initial, "gradients": gradients, "gradient_vector": gradient_vector,
                "update": updated - initial, "predicted_update": predicted - initial}

    def test_default_lazy_flex_attention_uses_the_same_backward_amp_guard(self):
        from torch._functorch import config as aot_config
        if not layer_ops._HAS_FLEX_ATTENTION or not hasattr(aot_config, "backward_pass_autocast"):
            self.skipTest("Guarded FlexAttention unavailable")
        torch.manual_seed(431)
        attention = layer_ops.SlidingWindowAttention(d_model=128, n_heads=1, n_kv_heads=1,
                                                    window_size=32).cuda().train()
        ids = torch.randn(1, 128, 128, device="cuda", requires_grad=True)
        # This deliberately bypasses maybe_compile_inner_loop: hardware compile
        # may be disabled while the long-sequence attention path still compiles.
        with mock.patch.object(layer_ops, "_COMPILED_FLEX_ATTN", None):
            with mock.patch.object(layer_ops, "compile_with_backward_amp_off",
                                   wraps=layer_ops.compile_with_backward_amp_off) as guarded:
                with mock.patch.object(layer_ops.F, "scaled_dot_product_attention",
                                       wraps=layer_ops.F.scaled_dot_product_attention) as fallback:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output, _ = attention(ids)
                        loss = output.float().square().mean()
                    fallback.assert_not_called()
                loss.backward()
                guarded.assert_called_once()
                self.assertTrue(torch.isfinite(ids.grad).all())

    def _assert_relative_error(self, actual, expected, tolerance, message):
        error = sum((actual[name] - expected[name]).double().square().sum() for name in expected).sqrt()
        reference = sum(value.double().square().sum() for value in expected.values()).sqrt()
        self.assertGreater(float(reference), 0.0, "Training did not produce any gradients/updates")
        self.assertLessEqual(float(error / reference), tolerance, message)


if __name__ == "__main__":
    unittest.main()

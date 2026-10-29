"""Inference startup must prepare only the work the checkpoint can execute."""
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import engine.personalizer as personalization
from engine.personalizer import uTTTPersonalizer
from model.uttt_model import uTTTConfig


class InferenceWarmupTests(unittest.TestCase):
    def personalizer(self, online):
        result = uTTTPersonalizer.__new__(uTTTPersonalizer)
        result.device = torch.device("cuda")
        result.dtype = torch.bfloat16
        result.online_updates_enabled = online
        result._warmup_complete = False
        result._warmup_compiled = False
        result.model = MagicMock()
        result.model.stream_generate.side_effect = lambda **kwargs: iter(())
        return result

    def contexts(self):
        original_ones = torch.ones
        def cpu_ones(*args, **kwargs):
            kwargs.pop("device", None)
            return original_ones(*args, **kwargs)
        return (
            patch.object(torch, "ones", side_effect=cpu_ones),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.amp, "autocast", side_effect=lambda *args, **kwargs: contextlib.nullcontext()),
        )

    def test_language_checkpoint_has_no_ttt_stream(self):
        manager = SimpleNamespace(
            model=MagicMock(), config=uTTTConfig(vocab_size=512),
            device=torch.device("cuda"), dtype=torch.bfloat16,
            training_metadata={"training_stage": "sft"},
        )
        with patch.object(torch.cuda, "Stream") as stream:
            result = uTTTPersonalizer(manager)
        self.assertIsNone(result._ttt_stream)
        stream.assert_not_called()

    def test_language_warmup_skips_memory_write_and_is_idempotent(self):
        result = self.personalizer(False)
        def generate(**kwargs):
            # Match serving dispatch; inference_mode would need a second compile.
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference_mode_enabled())
            return iter(())
        result.model.stream_generate.side_effect = generate
        ones, synchronize, autocast = self.contexts()
        with ones as tensors, synchronize, autocast:
            result.warmup(compile_kernels=False)
            result.warmup(compile_kernels=False)
        result.model.stream_generate.assert_called_once()
        result.model.init_fast_state.assert_called_once()
        result.model.test_time_update.assert_not_called()
        # Only the prompt is allocated; no update tokens or mask.
        self.assertEqual(tensors.call_count, 1)

    def test_meta_warmup_can_add_write_kernels_after_language_warmup(self):
        language, meta = self.personalizer(False), self.personalizer(True)
        ones, synchronize, autocast = self.contexts()
        with patch.object(personalization, "_INFERENCE_KERNELS_COMPILED", set()), ones, synchronize, autocast:
            with patch.object(torch, "compile", side_effect=lambda function, **kwargs: function) as compile_fn:
                language.warmup()
                self.assertEqual(compile_fn.call_count, 4)
                meta.warmup()
                self.assertEqual(compile_fn.call_count, 6)
                meta.warmup()
                self.assertEqual(compile_fn.call_count, 6)
        meta.model.test_time_update.assert_called_once()

    def test_eager_warmup_does_not_prevent_later_compiled_warmup(self):
        result = self.personalizer(False)
        ones, synchronize, autocast = self.contexts()
        with patch.object(personalization, "_INFERENCE_KERNELS_COMPILED", set()), ones, synchronize, autocast:
            with patch.object(torch, "compile", side_effect=lambda function, **kwargs: function) as compile_fn:
                result.warmup(compile_kernels=False)
                result.warmup(compile_kernels=True)
                self.assertEqual(compile_fn.call_count, 4)
        self.assertEqual(result.model.stream_generate.call_count, 2)

    def test_model_preparation_precedes_kernel_compilation(self):
        result = self.personalizer(False)
        order = []
        result.model.prepare_for_inference.side_effect = lambda: order.append("prepare")
        ones, synchronize, autocast = self.contexts()
        with patch.object(personalization, "_INFERENCE_KERNELS_COMPILED", set()), ones, synchronize, autocast:
            def compile_kernel(function, **kwargs):
                order.append("compile")
                return function
            with patch.object(torch, "compile", side_effect=compile_kernel):
                result.warmup()
        self.assertEqual(order[0], "prepare")
        self.assertEqual(order.count("compile"), 4)


if __name__ == "__main__":
    unittest.main()

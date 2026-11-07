"""Projection packing is an independent, reversible inference startup choice."""
import contextlib
import os
import unittest
from unittest.mock import MagicMock, patch

import torch

import engine.personalizer as personalization
from engine.personalizer import uTTTPersonalizer
from model.uttt_model import uTTTConfig, uTTTModel


class InferencePackingControlTests(unittest.TestCase):
    def test_server_cli_passes_explicit_setting_without_starting_service(self):
        import server.app as server
        for flag, expected in (("--packed-projections", "1"), ("--no-packed-projections", "0")):
            with self.subTest(flag=flag), patch.dict(os.environ, {}, clear=False), \
                    patch("sys.argv", ["server.app", flag, "--no-compile"]), patch("uvicorn.run") as run:
                server.main()
                self.assertEqual(os.environ["UTTT_PACKED_PROJECTIONS"], expected)
                self.assertEqual(os.environ["UTTT_NO_COMPILE"], "1")
                run.assert_called_once()
        with patch.dict(os.environ, {"UTTT_PACKED_PROJECTIONS": "1"}), \
                patch("sys.argv", ["server.app"]), patch("uvicorn.run"):
            server.main()
            self.assertEqual(os.environ["UTTT_PACKED_PROJECTIONS"], "1")

    def personalizer(self):
        result = uTTTPersonalizer.__new__(uTTTPersonalizer)
        result.device = torch.device("cuda")
        result.dtype = torch.bfloat16
        result.online_updates_enabled = False
        result._warmup_complete = False
        result._warmup_compiled = False
        result._warmup_packed = False
        result.model = MagicMock()
        result.model.stream_generate.side_effect = lambda **kwargs: iter(())
        return result

    @contextlib.contextmanager
    def fake_cuda_warmup(self):
        original_ones = torch.ones

        def cpu_ones(*args, **kwargs):
            kwargs.pop("device", None)
            return original_ones(*args, **kwargs)

        with patch.dict(os.environ, {}, clear=False), \
                patch.object(torch, "ones", side_effect=cpu_ones), \
                patch.object(torch.cuda, "synchronize"), \
                patch.object(torch.amp, "autocast", side_effect=lambda *args, **kwargs: contextlib.nullcontext()), \
                patch.object(personalization, "_INFERENCE_KERNELS_COMPILED", set()), \
                patch.object(torch, "compile", side_effect=lambda function, **kwargs: function) as compile_fn:
            os.environ.pop("UTTT_PACKED_PROJECTIONS", None)
            yield compile_fn

    def test_packing_without_compilation_can_be_changed_and_rewarmed(self):
        result = self.personalizer()
        with self.fake_cuda_warmup() as compile_fn:
            result.warmup(compile_kernels=False, packed_projections=False)
            result.warmup(compile_kernels=False, packed_projections=True)
            result.warmup(compile_kernels=False, packed_projections=True)
            result.warmup(compile_kernels=False, packed_projections=False)
            compile_fn.assert_not_called()
        self.assertEqual(result.model.stream_generate.call_count, 3)
        self.assertEqual([call.kwargs["enabled"] for call in result.model.prepare_for_inference.call_args_list],
                         [False, True, True, False])
        self.assertFalse(result._warmup_packed)

    def test_compilation_can_run_without_projection_packs(self):
        result = self.personalizer()
        with self.fake_cuda_warmup() as compile_fn:
            result.warmup(compile_kernels=True, packed_projections=False)
            self.assertEqual(compile_fn.call_count, 4)
        result.model.prepare_for_inference.assert_called_once_with(enabled=False)
        self.assertTrue(result._warmup_compiled)

    def test_cpu_tracks_applied_packing_without_cuda_warmup(self):
        result = self.personalizer()
        result.device = torch.device("cpu")
        with patch.object(torch.cuda, "synchronize") as synchronize:
            result.warmup(compile_kernels=False, packed_projections=True)
            self.assertTrue(result._warmup_packed)
            result.warmup(compile_kernels=False, packed_projections=False)
            self.assertFalse(result._warmup_packed)
        synchronize.assert_not_called()
        result.model.init_fast_state.assert_not_called()

    def test_environment_can_enable_packing_and_explicit_argument_wins(self):
        result = self.personalizer()
        with self.fake_cuda_warmup() as compile_fn:
            os.environ["UTTT_PACKED_PROJECTIONS"] = "1"
            result.warmup(compile_kernels=False)
            result.model.prepare_for_inference.assert_called_with(enabled=True)
            result.warmup(compile_kernels=False, packed_projections=False)
            result.model.prepare_for_inference.assert_called_with(enabled=False)
            compile_fn.assert_not_called()

    def test_unset_environment_keeps_legacy_defaults(self):
        with self.fake_cuda_warmup():
            for compile_kernels in (False, True):
                result = self.personalizer()
                result.warmup(compile_kernels=compile_kernels)
                result.model.prepare_for_inference.assert_called_once_with(enabled=compile_kernels)

    def test_implicit_eager_warmup_preserves_previously_enabled_packs(self):
        result = self.personalizer()
        with self.fake_cuda_warmup() as compile_fn:
            result.warmup(compile_kernels=True)
            result.warmup(compile_kernels=False)
            self.assertEqual(compile_fn.call_count, 4)
            self.assertEqual(result.model.stream_generate.call_count, 1)
            self.assertTrue(result._warmup_packed)
            self.assertEqual([call.kwargs["enabled"] for call in result.model.prepare_for_inference.call_args_list],
                             [True, True])
            # External preparation before the first warmup is preserved too.
            external = self.personalizer()
            external.model.fast_weight_pool.fused_inference_readout = True
            external.warmup(compile_kernels=False)
            external.model.prepare_for_inference.assert_called_once_with(enabled=True)

    def test_invalid_switches_fail_before_preparing_or_allocating(self):
        result = self.personalizer()
        with patch.dict(os.environ, {"UTTT_PACKED_PROJECTIONS": "yes"}):
            with self.assertRaisesRegex(ValueError, "UTTT_PACKED_PROJECTIONS"):
                result.warmup(compile_kernels=False)
        with self.assertRaisesRegex(ValueError, "packed_projections"):
            result.warmup(packed_projections=1)
        result.model.prepare_for_inference.assert_not_called()
        result.model.init_fast_state.assert_not_called()

    def test_real_model_round_trip_preserves_logits_weights_and_state_dict(self):
        torch.manual_seed(908)
        model = uTTTModel(uTTTConfig(
            vocab_size=512, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1,
            d_ff_fast=32, chunk_size=4, sliding_window_size=8,
            gradient_checkpointing=False,
        )).eval().requires_grad_(False)
        ids = torch.randint(6, 512, (1, 12))
        state = model.init_fast_state(detach=True)
        weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        with torch.no_grad():
            expected = model(ids, fast_state=state, update_fast_weights=False).logits
            model._cuda_graph_decoder = object()
            model.prepare_for_inference(enabled=True)
            self.assertIsNone(model._cuda_graph_decoder)
            packed = model(ids, fast_state=state, update_fast_weights=False).logits
            torch.testing.assert_close(packed, expected)

            existing_graph = object()
            model._cuda_graph_decoder = existing_graph
            model.prepare_for_inference(enabled=True)
            self.assertIs(model._cuda_graph_decoder, existing_graph)
            # A reloaded projection also invalidates a graph holding its old pack.
            model.layers[0].attn.wq.weight.add_(0.01)
            model.prepare_for_inference(enabled=True)
            self.assertIsNone(model._cuda_graph_decoder)
            model.layers[0].attn.wq.weight.copy_(weights["layers.0.attn.wq.weight"])
            model.prepare_for_inference(enabled=True)

            model._cuda_graph_decoder = object()
            model.prepare_for_inference(enabled=False)
            self.assertIsNone(model._cuda_graph_decoder)
            self.assertFalse(model.fast_weight_pool.fused_inference_readout)
            self.assertTrue(all(block.attn._packed_qkv is None and block.attn._packed_qkv_key is None
                                for block in model.layers))
            unpacked = model(ids, fast_state=state, update_fast_weights=False).logits
            torch.testing.assert_close(unpacked, expected)
        self.assertEqual(set(model.state_dict()), set(weights))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            model.prepare_for_inference(enabled=1)


if __name__ == "__main__":
    unittest.main()

"""The isolated meta probe refuses live GPUs and never restores old Adam state.

These guard/ownership tests use only the standard library and cannot touch CUDA.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_probe_with_stubs():
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=Mock(return_value=True), current_device=Mock(return_value=0),
        get_device_properties=Mock(return_value=SimpleNamespace(total_memory=32 * 1024 ** 3, uuid="gpu")),
        mem_get_info=Mock(return_value=(31 * 1024 ** 3, 32 * 1024 ** 3)),
        set_per_process_memory_fraction=Mock(),
    )
    fake_torch.load = Mock()
    fake_torch.save = Mock(side_effect=AssertionError("Probe must never save checkpoint weights"))
    fake_model = ModuleType("model.uttt_model")
    fake_model.uTTTConfig = Mock()
    fake_model.uTTTModel = Mock()
    fake_tokenizer = ModuleType("model.tokenizer")
    fake_tokenizer.build_tokenizer = Mock()
    path = Path(__file__).resolve().parents[1] / "deploy/probe_selective_meta.py"
    spec = importlib.util.spec_from_file_location("_selective_meta_probe_guards", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake_torch, "model.uttt_model": fake_model, "model.tokenizer": fake_tokenizer}):
        spec.loader.exec_module(module)
    return module, fake_torch


class SelectiveMetaProbeGuardsTests(unittest.TestCase):
    def setUp(self):
        self.probe, self.torch = load_probe_with_stubs()

    def memory(self, global_gib=31, runtime_gib=31):
        gib = self.probe.GIB
        return {"global_free_bytes": int(global_gib * gib), "runtime_free_bytes": int(runtime_gib * gib),
                "free_bytes": int(min(global_gib, runtime_gib) * gib)}

    def test_either_low_global_or_low_runtime_free_refuses_before_allocator_change(self):
        for global_gib, runtime_gib in ((3, 31), (31, 3), (27.99, 31)):
            with self.subTest(global_gib=global_gib, runtime_gib=runtime_gib), \
                    patch.object(self.probe, "query_available_gpu_memory", return_value=self.memory(global_gib, runtime_gib)):
                with self.assertRaisesRegex(self.probe.GpuMemoryError, "28 GiB"):
                    self.probe.guard_cuda_allocator()
        self.torch.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_allocator_is_capped_at_28_gib_relative_to_total_not_free(self):
        with patch.object(self.probe, "query_available_gpu_memory", return_value=self.memory()):
            device, _, budget = self.probe.guard_cuda_allocator()
        self.assertEqual(device, 0)
        self.assertEqual(budget, 28 * self.probe.GIB)
        self.torch.cuda.set_per_process_memory_fraction.assert_called_once_with(28 / 32, device=0)

    def test_exact_minimum_free_preserves_driver_headroom(self):
        with patch.object(self.probe, "query_available_gpu_memory", return_value=self.memory(28, 31)):
            _, _, budget = self.probe.guard_cuda_allocator()
        self.assertEqual(budget, int(26.5 * self.probe.GIB))
        self.torch.cuda.set_per_process_memory_fraction.assert_called_once_with(26.5 / 32, device=0)

    def test_unknown_global_measurement_refuses_instead_of_using_runtime_alone(self):
        with patch.object(self.probe, "query_available_gpu_memory", side_effect=self.probe.GpuMemoryError("no global measurement")):
            with self.assertRaisesRegex(self.probe.GpuMemoryError, "no global"):
                self.probe.guard_cuda_allocator()
        self.torch.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_checkpoint_is_mmap_read_and_optimizer_payload_is_never_exposed(self):
        class UnusableOptimizer:
            def __getattribute__(self, name):
                raise AssertionError("Checkpoint Adam state must not be inspected or restored")

        payload = {"config": {"tokenizer_json": "actual-bpe"}, "slow_weights": {},
                   "init_fast_weights": {}, "step": 55800, "optimizer_state": UnusableOptimizer()}
        self.torch.load.return_value = payload
        checkpoint = self.probe.load_probe_checkpoint(Path("readonly.pt"))
        self.torch.load.assert_called_once_with(Path("readonly.pt"), map_location="cpu", weights_only=False, mmap=True)
        self.assertNotIn("optimizer_state", checkpoint)
        self.assertEqual(checkpoint["step"], 55800)
        self.assertIs(checkpoint["slow_weights"], payload["slow_weights"])
        self.assertIn("optimizer_state", payload)
        self.torch.save.assert_not_called()

    def test_memory_scope_freezes_attention_embedding_and_external_norms(self):
        names = ("tok_embeddings.weight", "layers.0.attn.wq.weight", "layers.0.uttt_norm.weight",
                 "layers.0.uttt.w_q.weight", "layers.0.uttt.w_k.weight", "fast_weight_pool.w_up_init")
        parameters = [(name, Mock()) for name in names]
        model = SimpleNamespace(named_parameters=lambda: iter(parameters))
        selected = self.probe.scoped_parameters(model, "memory")
        self.assertEqual([name for name, _ in selected], list(names[3:]))
        for index, (_, parameter) in enumerate(parameters):
            parameter.requires_grad_.assert_called_once_with(index >= 3)

    def test_one_chunk_or_one_adam_step_refuses_before_checkpoint_or_cuda(self):
        for options in ({"seq_len": 129}, {"steps": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.probe.probe(Path("never-read.pt"), **options)
        self.torch.load.assert_not_called()
        self.torch.cuda.is_available.assert_not_called()


if __name__ == "__main__":
    unittest.main()

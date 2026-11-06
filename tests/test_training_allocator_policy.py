"""A measured allocator policy must survive launch and refuse unsafe budgets."""
from contextlib import ExitStack
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy.run_recovery_training import child_environment
from deploy.gpu_memory import GpuMemoryError
import train_5090


class TrainingAllocatorPolicyTests(unittest.TestCase):
    def test_launcher_applies_policy_before_import_and_retains_revision(self):
        with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "legacy"}):
            result = child_environment("revision", "run", {"cuda_allocator_conf": "expandable_segments:True"})
        self.assertEqual(result["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", result)
        self.assertEqual(result["UTTT_CODE_REVISION"], "revision")
        self.assertEqual(result["UTTT_RUN_ID"], "run")

    def test_direct_training_refuses_a_different_allocator_before_cuda(self):
        with patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": "different"}), \
                patch.object(train_5090.torch.cuda, "is_available") as available:
            with self.assertRaisesRegex(ValueError, "before launching"):
                train_5090.configure_rtx5090_environment({"cuda_allocator_conf": "expandable_segments:True"})
        available.assert_not_called()

    def configure(self, free):
        stack = ExitStack()
        self.addCleanup(stack.close)
        cuda = train_5090.torch.cuda
        properties = SimpleNamespace(total_memory=32 * 1024**3, uuid="test-gpu")
        for name, value in (("is_available", True), ("get_device_name", "Test GPU"),
                            ("get_device_capability", (12, 0)), ("get_device_properties", properties),
                            ("mem_get_info", (31 * 1024**3, 32 * 1024**3))):
            stack.enter_context(patch.object(cuda, name, return_value=value))
        stack.enter_context(patch("deploy.gpu_memory.query_available_gpu_memory", return_value={"free_bytes": free}))
        fraction = stack.enter_context(patch.object(cuda, "set_per_process_memory_fraction"))
        return fraction

    def test_enforced_budget_is_relative_to_total_capacity(self):
        fraction = self.configure(31 * 1024**3)
        train_5090.configure_rtx5090_environment({"vram_limit_gb": 28, "enforce_vram_limit": True,
                                                "tf32_matmul": False})
        fraction.assert_called_once_with(28 / 32, device=0)

    def test_driver_headroom_is_required_even_when_runtime_reports_more_memory(self):
        fraction = self.configure(29 * 1024**3)
        with self.assertRaises(GpuMemoryError):
            train_5090.configure_rtx5090_environment({"vram_limit_gb": 28, "enforce_vram_limit": True,
                                                    "tf32_matmul": False})
        fraction.assert_not_called()


if __name__ == "__main__":
    unittest.main()

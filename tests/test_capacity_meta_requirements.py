"""A capacity claim must exercise learned writers and persistent Adam state."""
import unittest
from unittest.mock import patch

from deploy.profile_capacity import measure
from model.uttt_model import uTTTConfig


class MetaCapacityRequirementsTests(unittest.TestCase):
    def test_single_chunk_is_rejected_before_touching_cuda(self):
        config = uTTTConfig(chunk_size=128)
        with patch("torch.cuda.empty_cache", side_effect=AssertionError("CUDA must remain untouched")):
            for length in (32, 128, 129):
                with self.subTest(length=length), self.assertRaisesRegex(ValueError, "at least two chunks"):
                    measure(config, length, 28.0)

    def test_one_step_cannot_claim_steady_state_capacity(self):
        with patch("torch.cuda.empty_cache", side_effect=AssertionError("CUDA must remain untouched")):
            with self.assertRaisesRegex(ValueError, "at least two optimizer steps"):
                measure(uTTTConfig(chunk_size=128), 256, 28.0, steps=1)

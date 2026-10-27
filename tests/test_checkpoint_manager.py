"""Regression coverage for checkpoint-manager startup and persona signatures."""
import tempfile
import unittest
from pathlib import Path

import torch

from engine.checkpoint_manager import DualTrackCheckpointManager
from model.uttt_model import uTTTConfig, uTTTModel


class TestCheckpointManagerStartup(unittest.TestCase):
    def test_loaded_base_signature_and_persona_round_trip(self):
        config = uTTTConfig(
            vocab_size=64, d_model=16, n_layers=1, n_heads=2,
            n_kv_heads=1, d_ff_fast=32, chunk_size=8,
            sliding_window_size=16, gradient_checkpointing=False,
        )
        model = uTTTModel(config)
        # Use exact values so the expected signature is independent of RNG/device.
        with torch.no_grad():
            model.fast_weight_pool.w_up_init.fill_(0.25)
        payloads = {
            "dual": {
                "slow_weights": model.slow_state_dict(),
                "init_fast_weights": model.init_fast_state_dict(),
            },
            "full": {"model_state_dict": model.state_dict()},
        }
        for checkpoint_format, weights in payloads.items():
            with self.subTest(checkpoint_format=checkpoint_format), tempfile.TemporaryDirectory() as tmp:
                base_path = Path(tmp) / "base.pt"
                torch.save({"config": config.to_dict(), **weights}, base_path)
                options = dict(
                    base_checkpoint_path=base_path,
                    personas_dir=Path(tmp) / "personas",
                    device="cpu", dtype=torch.float32,
                    auto_create_base_if_missing=False,
                )
                manager = DualTrackCheckpointManager(**options)
                self.assertEqual(manager.base_signature, "16x32:128.000000")
                state = manager.get_user_state("regression_user")
                state.w_up.add_(0.5)
                state.step = 7
                manager.save_user_state("regression_user")

                reloaded = DualTrackCheckpointManager(**options)
                restored = reloaded.get_user_state("regression_user")
                self.assertEqual(restored.step, 7)
                torch.testing.assert_close(restored.w_up, state.w_up)

                # A different base must still invalidate saved persona weights.
                with torch.no_grad():
                    model.fast_weight_pool.w_up_init.fill_(0.75)
                torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict()}, base_path)
                changed = DualTrackCheckpointManager(**options)
                fresh = changed.get_user_state("regression_user")
                self.assertEqual(fresh.step, 0)
                torch.testing.assert_close(fresh.w_up, torch.full_like(fresh.w_up, 0.75))
                with torch.no_grad():
                    model.fast_weight_pool.w_up_init.fill_(0.25)


if __name__ == "__main__":
    unittest.main()

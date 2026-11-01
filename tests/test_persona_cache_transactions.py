"""Incoming personas survive full-cache eviction and failed device transfers."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from engine.checkpoint_manager import DualTrackCheckpointManager
from model.uttt_layer import FastWeightState
from model.uttt_model import uTTTConfig, uTTTModel


class PersonaCacheTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = uTTTConfig(vocab_size=512, d_model=16, n_layers=1, n_heads=2,
                           n_kv_heads=1, d_ff_fast=32, chunk_size=4, sliding_window_size=8)
        model = uTTTModel(config)
        self.base = self.root / "base.pt"
        torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict()}, self.base)

    def manager(self, name="personas"):
        return DualTrackCheckpointManager(self.base, self.root / name, device="cpu",
                                          max_gpu_users=1, max_cpu_users=1,
                                          auto_create_base_if_missing=False)

    def warm_pair(self, manager):
        alice = manager.get_user_state("alice")
        alice.step = 4
        alice.w_up.add_(0.1)
        manager._user_metadata["alice"]["turns"] = 7
        manager.get_user_state("bob")
        return manager._cpu_lru["alice"]

    def test_alternating_users_hit_cpu_cache_without_disk_reads(self):
        manager = self.manager()
        self.warm_pair(manager)
        with patch("engine.checkpoint_manager.torch.load", side_effect=AssertionError("unexpected disk reload")):
            for _ in range(3):
                state = manager.get_user_state("alice")
                self.assertEqual(state.step, 4)
                self.assertEqual(manager._user_metadata["alice"]["turns"], 7)
                self.assertEqual(list(manager._cpu_lru), ["bob"])
                manager.get_user_state("bob")
                self.assertEqual(list(manager._cpu_lru), ["alice"])

    def test_failed_promotion_restores_pending_cpu_state_and_can_retry(self):
        manager = self.manager()
        pending = self.warm_pair(manager)
        original_to = FastWeightState.to
        def transfer(state, *args, **kwargs):
            if state is pending:
                raise RuntimeError("device allocation failed")
            return original_to(state, *args, **kwargs)
        with patch.object(FastWeightState, "to", autospec=True, side_effect=transfer):
            with self.assertRaisesRegex(RuntimeError, "allocation"):
                manager.get_user_state("alice")
        self.assertIs(manager._cpu_lru["alice"], pending)
        self.assertEqual(manager._user_metadata["alice"]["turns"], 7)
        self.assertEqual(manager.get_user_state("alice").step, 4)

    def test_failed_offload_keeps_active_state_and_decoder_until_copy_succeeds(self):
        manager = self.manager()
        current = manager.get_user_state("alice")
        decoder = type("Decoder", (), {"fast_state": current})()
        manager.model._cuda_graph_decoder = decoder
        with patch.object(FastWeightState, "to", side_effect=RuntimeError("host allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "host allocation"):
                manager.get_user_state("bob")
        self.assertIs(manager._gpu_lru["alice"], current)
        self.assertIs(manager.model._cuda_graph_decoder, decoder)

    def test_failed_update_or_reset_of_cpu_user_preserves_previous_state_and_metadata(self):
        for operation in ("update", "reset"):
            with self.subTest(operation=operation):
                manager = self.manager(operation)
                pending = self.warm_pair(manager)
                candidate = pending.clone(detach=True)
                candidate.step = 9
                write = manager._save_state_to_disk
                def persist(user, state):
                    if user == "alice":
                        raise OSError("persona disk full")
                    return write(user, state)
                with patch.object(manager, "_save_state_to_disk", side_effect=persist):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        if operation == "update":
                            manager.update_user_state("alice", candidate)
                        else:
                            manager.reset_user_state("alice")
                self.assertIs(manager._cpu_lru["alice"], pending)
                self.assertEqual(manager._user_metadata["alice"]["turns"], 7)
                self.assertEqual(manager.get_user_state("alice").step, 4)

    def test_disk_persona_transfer_failure_does_not_fall_back_to_fresh_state(self):
        first = self.manager()
        state = first.get_user_state("alice").to(dtype=torch.float32)
        state.step = 11
        first.update_user_state("alice", state)
        path, _ = first.get_persona_paths("alice")
        saved = path.read_bytes()
        reloaded = self.manager()
        with patch.object(FastWeightState, "to", side_effect=RuntimeError("GPU out of memory")), \
                patch.object(reloaded.model, "init_fast_state") as fresh:
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                reloaded.get_user_state("alice")
            fresh.assert_not_called()
        self.assertNotIn("alice", reloaded._gpu_lru)
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual(reloaded.get_user_state("alice").step, 11)

    def test_drift_cache_tracks_inplace_edits_and_base_changes_at_the_same_step(self):
        manager = self.manager()
        state = manager.get_user_state("alice")
        pool = manager.model.fast_weight_pool
        with patch.object(pool, "compute_drift", wraps=pool.compute_drift) as compute:
            self.assertEqual(manager.compute_user_drift("alice")["drift_frobenius"], 0.)
            manager.compute_user_drift("alice")
            self.assertEqual(compute.call_count, 1)
            state.w_up.add_(0.1)
            changed = manager.compute_user_drift("alice")["drift_frobenius"]
            self.assertGreater(changed, 0.)
            self.assertEqual(compute.call_count, 2)
            state.w_up = state.w_up + 0.1
            self.assertGreater(manager.compute_user_drift("alice")["drift_frobenius"], changed)
            with torch.no_grad():
                pool.w_up_init.add_(0.2)
            manager.compute_user_drift("alice")
            self.assertEqual(compute.call_count, 4)
            self.assertEqual(state.step, 0)

    def test_inference_tensor_drift_does_not_access_missing_version_counters(self):
        manager = self.manager()
        state = manager.get_user_state("alice")
        with torch.inference_mode():
            manager._gpu_lru["alice"] = state.clone(detach=True)
        pool = manager.model.fast_weight_pool
        with patch.object(pool, "compute_drift", wraps=pool.compute_drift) as compute:
            manager.compute_user_drift("alice")
            manager.compute_user_drift("alice")
            self.assertEqual(compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()

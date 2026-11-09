"""Meta memory rejection preserves files and caches until an explicit reset."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from engine.checkpoint_manager import DualTrackCheckpointManager, PersonaCompatibilityError
from model.uttt_model import uTTTConfig, uTTTModel


STATE_NAMES = ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down", "ln_weight", "ln_bias")


class PersonaCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = uTTTConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2,
                                 n_kv_heads=1, d_ff_fast=32, chunk_size=8,
                                 sliding_window_size=16, gradient_checkpointing=False)
        model = uTTTModel(self.config)
        for stage in ("meta", "sft"):
            torch.save({"config": self.config.to_dict(), "slow_weights": model.slow_state_dict(),
                        "init_fast_weights": model.init_fast_state_dict(),
                        "metrics": {"training_stage": stage, "inner_loop_protocol": model.INNER_LOOP_PROTOCOL}},
                       self.root / (stage + ".pt"))

    def manager(self, stage="meta", directory="users"):
        return DualTrackCheckpointManager(self.root / (stage + ".pt"), self.root / directory,
                                          device="cpu", dtype=torch.bfloat16, max_gpu_users=1,
                                          max_cpu_users=1, auto_create_base_if_missing=False)

    @staticmethod
    def state(manager, dtype=torch.float32):
        state = manager.model.init_fast_state(batch_size=1, device="cpu", dtype=dtype, detach=True)
        state.w_up.add_(0.000001)
        state.s_gate.add_(0.000003)
        state.ln_bias.add_(0.000007)
        state.step = 7
        state.last_surprise_loss = .25
        state.surprise_history = [.25]
        return state

    @staticmethod
    def payload(manager, state, user="alice"):
        return {"checkpoint_type": "uTTT_User_Persona_State", "user_id": user,
                "base_signature": manager.base_signature, "base_fingerprint": manager.base_fingerprint,
                "fast_state": state.to_state_dict(), "metadata": {"user_id": user, "turns": 3}}

    def files(self):
        return {str(path.relative_to(self.root)): path.read_bytes() if path.is_file() else None
                for path in self.root.rglob("*")}

    def assert_rejected_without_changes(self, manager, operation, message):
        before_files = self.files()
        before_metadata = copy.deepcopy(manager._user_metadata)
        before_caches = [list(manager._gpu_lru.items()), list(manager._cpu_lru.items())]
        before_tensors = {id(state): {name: getattr(state, name).clone() for name in STATE_NAMES}
                          for cache in before_caches for _, state in cache}
        with mock.patch.object(manager, "_reserve_gpu_slot", wraps=manager._reserve_gpu_slot) as reserve:
            with self.assertRaisesRegex(PersonaCompatibilityError, message):
                operation()
            reserve.assert_not_called()
        self.assertEqual(manager._user_metadata, before_metadata)
        self.assertEqual(self.files(), before_files)
        for expected, actual in zip(before_caches, (manager._gpu_lru, manager._cpu_lru)):
            self.assertEqual(list(actual), [user for user, _ in expected])
            for user, state in expected:
                self.assertIs(actual[user], state)
                for name in STATE_NAMES:
                    torch.testing.assert_close(getattr(state, name), before_tensors[id(state)][name], rtol=0, atol=0)

    def test_exact_fp32_meta_save_reload_preserves_every_master_bit(self):
        manager = self.manager()
        original = self.state(manager)
        manager.update_user_state("alice", original)
        restored = self.manager().get_user_state("alice")
        for name in STATE_NAMES:
            self.assertEqual(getattr(restored, name).dtype, torch.float32)
            torch.testing.assert_close(getattr(restored, name), getattr(original, name), rtol=0, atol=0)
        self.assertEqual(restored.step, 7)
        self.assertEqual(restored.surprise_history, [.25])

    def test_matching_fingerprint_rejects_each_rounded_tensor_before_eviction(self):
        manager = self.manager()
        manager.get_user_state("bob")
        path, _ = manager.get_persona_paths("alice")
        for name in STATE_NAMES:
            with self.subTest(tensor=name):
                rounded = self.state(manager)
                setattr(rounded, name, getattr(rounded, name).to(torch.bfloat16))
                torch.save(self.payload(manager, rounded), path)
                self.assert_rejected_without_changes(manager, lambda: manager.get_user_state("alice"),
                                                     "FP32.*" + name)

    def test_signature_only_missing_or_foreign_fingerprint_never_auto_resets_meta(self):
        manager = self.manager()
        manager.get_user_state("bob")
        path, _ = manager.get_persona_paths("alice")
        for fingerprint in ("missing", None, "a-different-checkpoint"):
            with self.subTest(fingerprint=fingerprint):
                payload = self.payload(manager, self.state(manager))
                if fingerprint == "missing":
                    payload.pop("base_fingerprint")
                else:
                    payload["base_fingerprint"] = fingerprint
                torch.save(payload, path)
                self.assert_rejected_without_changes(manager, lambda: manager.get_user_state("alice"),
                                                     "explicit matching base_fingerprint")

    def test_invalid_update_keeps_cpu_persona_and_active_user_with_and_without_save(self):
        manager = self.manager()
        manager.update_user_state("alice", self.state(manager))
        manager.get_user_state("bob")
        candidate = manager._cpu_lru["alice"].clone(detach=True)
        candidate.ln_bias = candidate.ln_bias.to(torch.bfloat16)
        for auto_save in (False, True):
            with self.subTest(auto_save=auto_save):
                self.assert_rejected_without_changes(
                    manager, lambda: manager.update_user_state("alice", candidate, auto_save=auto_save),
                    "FP32.*ln_bias")

    def test_save_and_snapshot_reject_invalid_cached_precision_without_new_files(self):
        manager = self.manager()
        manager.update_user_state("alice", self.state(manager))
        cached = manager._gpu_lru["alice"]
        cached.s_down = cached.s_down.to(torch.bfloat16)
        operations = (lambda: manager.save_user_state("alice"),
                      lambda: manager.save_snapshot("alice", "invalid"),
                      lambda: manager._save_state_to_disk("alice", cached))
        for operation in operations:
            with self.subTest(operation=operation):
                self.assert_rejected_without_changes(manager, operation, "FP32.*s_down")

    def test_snapshot_restore_rejects_precision_or_fingerprint_without_publication(self):
        manager = self.manager()
        manager.update_user_state("alice", self.state(manager))
        snapshot = manager.save_snapshot("alice", "stage")
        original = torch.load(snapshot, map_location="cpu", weights_only=False)
        for fault in ("precision", "legacy", "foreign"):
            with self.subTest(fault=fault):
                payload = copy.deepcopy(original)
                if fault == "precision":
                    payload["fast_state"]["ln_weight"] = payload["fast_state"]["ln_weight"].to(torch.bfloat16)
                    message = "FP32.*ln_weight"
                else:
                    if fault == "legacy":
                        payload.pop("base_fingerprint")
                    else:
                        payload["base_fingerprint"] = "other"
                    message = "explicit matching base_fingerprint"
                torch.save(payload, snapshot)
                self.assert_rejected_without_changes(manager, lambda: manager.load_snapshot("alice", "stage"),
                                                     message)

    def test_explicit_reset_recovers_legacy_or_rounded_meta_and_persists_fp32(self):
        for fault in ("legacy", "rounded"):
            with self.subTest(fault=fault):
                manager = self.manager(directory="recover-" + fault)
                payload = self.payload(manager, self.state(manager, dtype=torch.bfloat16 if fault == "rounded"
                                                            else torch.float32))
                if fault == "legacy":
                    payload.pop("base_fingerprint")
                path, _ = manager.get_persona_paths("alice")
                torch.save(payload, path)
                with self.assertRaises(PersonaCompatibilityError):
                    manager.get_user_state("alice")
                reset = manager.reset_user_state("alice")
                expected = manager.model.init_fast_state(batch_size=1, dtype=torch.float32, detach=True)
                restored = self.manager(directory="recover-" + fault).get_user_state("alice")
                for name in STATE_NAMES:
                    self.assertEqual(getattr(restored, name).dtype, torch.float32)
                    torch.testing.assert_close(getattr(restored, name), getattr(expected, name), rtol=0, atol=0)
                self.assertEqual(reset.step, 0)
                self.assertEqual(restored.step, 0)
                self.assertEqual(torch.load(path, map_location="cpu", weights_only=False)["base_fingerprint"],
                                 manager.base_fingerprint)

    def test_sft_signature_only_bf16_legacy_state_remains_supported(self):
        manager = self.manager(stage="sft")
        state = self.state(manager, dtype=torch.bfloat16)
        payload = self.payload(manager, state)
        payload.pop("base_fingerprint")
        path, _ = manager.get_persona_paths("alice")
        torch.save(payload, path)
        restored = manager.get_user_state("alice")
        for name in STATE_NAMES:
            self.assertEqual(getattr(restored, name).dtype, torch.bfloat16)
            torch.testing.assert_close(getattr(restored, name), getattr(state, name), rtol=0, atol=0)
        manager.update_user_state("alice", restored)
        manager.save_snapshot("alice", "legacy")
        self.assertEqual(manager.load_snapshot("alice", "legacy").w_up.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

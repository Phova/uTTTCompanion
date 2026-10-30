"""Completed turns, failed writes and cancelled requests keep persona state coherent."""
import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from engine.checkpoint_manager import DualTrackCheckpointManager
from engine.personalizer import uTTTPersonalizer
from model.uttt_model import uTTTConfig, uTTTModel
import server.app as service


class LifecycleAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = uTTTConfig(
            vocab_size=512, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
            d_ff_fast=32, chunk_size=4, sliding_window_size=8,
            max_seq_len=64, gradient_checkpointing=False,
        )
        model = uTTTModel(self.config)
        self.base = self.root / "base.pt"
        torch.save({"config": self.config.to_dict(), "model_state_dict": model.state_dict(),
                    "metrics": {"training_stage": "meta", "inner_loop_protocol": model.INNER_LOOP_PROTOCOL}}, self.base)

    def manager(self, **kwargs):
        return DualTrackCheckpointManager(
            self.base, self.root / "personas", device="cpu", auto_create_base_if_missing=False, **kwargs,
        )

    def test_failed_update_and_reset_preserve_active_state_metadata_and_disk(self):
        manager = self.manager()
        original = manager.get_user_state("alice")
        original.step = 3
        path = manager.save_user_state("alice")
        before = path.read_bytes()
        metadata = dict(manager._user_metadata["alice"])
        decoder = SimpleNamespace(fast_state=original)
        manager.model._cuda_graph_decoder = decoder
        candidate = original.clone(detach=True)
        candidate.w_up.add_(0.1)
        candidate.step = 4
        with patch.object(manager, "_write_payload_files", side_effect=OSError("disk full")):
            for operation in (lambda: manager.update_user_state("alice", candidate),
                              lambda: manager.reset_user_state("alice")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    operation()
                self.assertIs(manager.get_user_state("alice"), original)
                self.assertEqual(manager._user_metadata["alice"], metadata)
                self.assertEqual(path.read_bytes(), before)
                self.assertIs(manager.model._cuda_graph_decoder, decoder)

    def test_every_gated_turn_persists_metrics_without_changing_weights_or_step(self):
        manager = self.manager()
        engine = uTTTPersonalizer(manager, surprise_threshold=1.0, default_max_new_tokens=2)
        state = manager.get_user_state("alice")
        original = state.clone(detach=True)
        path = manager.save_user_state("alice")
        candidate = state.clone(detach=True)
        candidate.w_gate.add_(1)
        candidate.step = 99
        for loss in (0.1, 0.2):
            with patch.object(manager.model, "stream_generate", return_value=iter(engine.tokenizer.encode("ok", add_bos=False))), \
                    patch.object(manager.model, "test_time_update", return_value=(candidate, loss, [loss])):
                result = engine.interact_and_evolve("alice", "hello")
            self.assertFalse(result.gate_triggered)
            self.assertTrue(result.persona_saved)
            self.assertIn("Gate Skipped", result.status_bar)
        payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["metadata"]["turns"], 2)
        self.assertEqual(payload["fast_state"]["last_surprise_loss"], 0.2)
        self.assertEqual(payload["fast_state"]["surprise_history"], [0.1, 0.2])
        restored = self.manager().get_user_state("alice")
        self.assertEqual(restored.step, original.step)
        for name in ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down"):
            torch.testing.assert_close(getattr(restored, name), getattr(original, name), atol=0, rtol=0)

    def test_failed_gated_metrics_write_rolls_back_scalar_history_and_turn_count(self):
        manager = self.manager()
        state = manager.get_user_state("alice")
        manager.record_user_turn("alice", 0.1)
        with patch.object(manager, "_write_payload_files", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.record_user_turn("alice", 0.2)
        self.assertEqual(state.last_surprise_loss, 0.1)
        self.assertEqual(state.surprise_history, [0.1])
        self.assertEqual(manager._user_metadata["alice"]["turns"], 1)

    def test_non_finite_updates_and_saved_personas_are_rejected(self):
        manager = self.manager()
        original = manager.get_user_state("alice")
        path = manager.save_user_state("alice")
        candidate = original.clone(detach=True)
        candidate.s_gate.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "finite"):
            manager.update_user_state("alice", candidate)
        self.assertIs(manager.get_user_state("alice"), original)
        payload = torch.load(path, weights_only=False)
        payload["fast_state"]["w_down"].fill_(float("nan"))
        payload["fast_state"]["step"] = 17
        torch.save(payload, path)
        restored = self.manager().get_user_state("alice")
        self.assertEqual(restored.step, 0)
        self.assertTrue(torch.isfinite(restored.w_down).all().item())

    def test_graph_references_are_released_on_eviction_reset_and_state_replacement(self):
        manager = self.manager(max_gpu_users=1)
        original = manager.get_user_state("alice")
        manager.model._cuda_graph_decoder = SimpleNamespace(fast_state=original)
        manager.get_user_state("bob")
        self.assertIsNone(manager.model._cuda_graph_decoder)
        state = manager.get_user_state("alice")
        manager.model._cuda_graph_decoder = SimpleNamespace(fast_state=state)
        manager.reset_user_state("alice", persist=False)
        self.assertIsNone(manager.model._cuda_graph_decoder)
        state = manager.get_user_state("alice")
        manager.model._cuda_graph_decoder = SimpleNamespace(fast_state=state)
        manager.update_user_state("alice", state, auto_save=False)
        self.assertIsNone(manager.model._cuda_graph_decoder)

    def test_fp32_master_survives_bf16_serving_offload_save_reload_and_snapshot(self):
        manager = self.manager(dtype=torch.bfloat16, max_gpu_users=1)
        master = manager.get_user_state("alice").to(dtype=torch.float32)
        master.w_up.add_(0.00001)
        master.s_gate.fill_(0.000001)
        manager.update_user_state("alice", master)
        manager.save_snapshot("alice", "master")
        manager.get_user_state("bob")
        promoted = manager.get_user_state("alice")
        reloaded_manager = self.manager(dtype=torch.bfloat16)
        reloaded = reloaded_manager.get_user_state("alice")
        snapshot = reloaded_manager.load_snapshot("alice", "master")
        for state in (promoted, reloaded, snapshot):
            self.assertEqual(state.w_up.dtype, torch.float32)
            self.assertEqual(state.s_gate.dtype, torch.float32)
            torch.testing.assert_close(state.w_up, master.w_up, atol=0, rtol=0)
            torch.testing.assert_close(state.s_gate, master.s_gate, atol=0, rtol=0)

    def test_partial_generation_closes_inner_iterator_without_test_time_update(self):
        manager = self.manager()
        engine = uTTTPersonalizer(manager, default_max_new_tokens=2)

        class Tokens:
            def __init__(self, ids):
                self.tokens = iter(ids)
                self.closed = False
            def __iter__(self):
                return self
            def __next__(self):
                return next(self.tokens)
            def close(self):
                self.closed = True

        tokens = Tokens(engine.tokenizer.encode("ok", add_bos=False))
        with patch.object(manager.model, "stream_generate", return_value=tokens), \
                patch.object(manager.model, "test_time_update") as update:
            stream = engine.stream_interact_and_evolve("alice", "hello")
            self.assertEqual(next(stream)["type"], "token")
            stream.close()
            self.assertTrue(tokens.closed)
            update.assert_not_called()
        self.assertEqual(manager._user_metadata["alice"]["turns"], 0)

    def test_non_finite_surprise_does_not_commit_and_invalid_gate_config_is_rejected(self):
        manager = self.manager()
        engine = uTTTPersonalizer(manager, default_max_new_tokens=2)
        state = manager.get_user_state("alice")
        with patch.object(manager.model, "stream_generate", return_value=iter(engine.tokenizer.encode("ok", add_bos=False))), \
                patch.object(manager.model, "test_time_update", return_value=(state, float("nan"), [])):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                engine.interact_and_evolve("alice", "hello")
        self.assertEqual(state.last_surprise_loss, 0)
        self.assertEqual(manager._user_metadata["alice"]["turns"], 0)
        for kwargs in ({"surprise_threshold": float("nan")}, {"elastic_anchor_lambda": 1.1},
                       {"max_weight_drift_ratio": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                uTTTPersonalizer(manager, **kwargs)

    def test_memory_writes_require_explicit_compatible_meta_protocol(self):
        manager = self.manager()
        for metadata in ({}, {"training_stage": "meta"},
                         {"training_stage": "meta", "inner_loop_protocol": "obsolete"},
                         {"training_stage": "pretrain", "inner_loop_protocol": manager.model.INNER_LOOP_PROTOCOL},
                         {"training_stage": "sft", "inner_loop_protocol": manager.model.INNER_LOOP_PROTOCOL}):
            with self.subTest(metadata=metadata):
                manager.training_metadata = metadata
                engine = uTTTPersonalizer(manager)
                self.assertFalse(engine.online_updates_enabled)
        manager.training_metadata = {"training_stage": "meta", "inner_loop_protocol": manager.model.INNER_LOOP_PROTOCOL}
        self.assertTrue(uTTTPersonalizer(manager).online_updates_enabled)

    def test_partial_atomic_write_is_cleaned_up_without_replacing_persona(self):
        manager = self.manager()
        manager.get_user_state("alice")
        path = manager.save_user_state("alice")
        before = path.read_bytes()
        def failed_save(payload, destination):
            destination.write(b"partial")
            raise OSError("disk full")
        with patch("engine.checkpoint_manager.torch.save", side_effect=failed_save):
            with self.assertRaises(OSError):
                manager.save_user_state("alice")
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(Path(str(path) + ".tmp").exists())


class ServerCancellationAuditTests(unittest.TestCase):
    def test_repeated_cancellation_keeps_operation_lock_until_worker_finishes(self):
        entered, release = threading.Event(), threading.Event()
        class Engine:
            ckpt_manager = SimpleNamespace(pool_status=lambda: {})
            def warmup(self, **kwargs):
                pass
        engine = Engine()
        def blocking_operation():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("worker release timed out")

        async def exercise():
            with patch.object(service, "get_personalizer", return_value=engine):
                async with service.lifespan(service.app):
                    task = asyncio.create_task(service._run_engine(blocking_operation))
                    await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3)
                    try:
                        task.cancel()
                        await asyncio.sleep(0)
                        task.cancel()
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        self.assertTrue(service.app.state.engine_lock.locked())
                    finally:
                        release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertFalse(service.app.state.engine_lock.locked())
        try:
            asyncio.run(exercise())
        finally:
            release.set()

    def test_health_pool_lock_does_not_block_event_loop(self):
        entered, release = threading.Event(), threading.Event()
        def blocking_pool_status():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("health release timed out")
            return {}
        engine = SimpleNamespace(ckpt_manager=SimpleNamespace(pool_status=blocking_pool_status),
                                 config=SimpleNamespace(name="test"))

        async def exercise():
            with patch.object(service, "get_personalizer", return_value=engine):
                task = asyncio.create_task(service.health_check())
                try:
                    self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3))
                    self.assertFalse(task.done())
                finally:
                    release.set()
                self.assertEqual((await task)["status"], "ok")
        try:
            asyncio.run(exercise())
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()

"""Regression tests for persistence, causal inference, and request isolation."""
import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from engine.checkpoint_manager import DualTrackCheckpointManager
from engine.personalizer import uTTTPersonalizer
from model.uttt_layer import SharedFastWeightPool, SlidingWindowAttention
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel
import train_5090 as training
import server.app as service


def tiny_config(**kwargs):
    return uTTTConfig(**{
        "vocab_size": 512, "d_model": 16, "n_layers": 3,
        "n_heads": 2, "n_kv_heads": 1, "d_ff_fast": 32,
        "chunk_size": 4, "sliding_window_size": 4,
        "max_seq_len": 32, "gradient_checkpointing": False, **kwargs,
    })


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = tiny_config()
        self.model = uTTTModel(self.config)
        self.base = self.root / "base.pt"
        self.write_base()

    def write_base(self):
        torch.save({"config": self.config.to_dict(), "model_state_dict": self.model.state_dict()}, self.base)

    def manager(self, **kwargs):
        return DualTrackCheckpointManager(
            base_checkpoint_path=self.base, personas_dir=self.root / "personas",
            device="cpu", auto_create_base_if_missing=False, **kwargs,
        )

    def test_updates_are_durable_ordered_and_save_errors_propagate(self):
        manager = self.manager()
        state = manager.get_user_state("alice").clone()
        for step in (1, 2, 3):
            state.step = step
            path = manager.update_user_state("alice", state)
            self.assertEqual(torch.load(path, weights_only=False)["fast_state"]["step"], step)
        manager.reset_user_state("alice")
        self.assertEqual(self.manager().get_user_state("alice").step, 0)
        with patch.object(manager, "_write_payload_files", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                manager.update_user_state("alice", state)

    def test_reset_respects_capacity_and_failed_eviction_keeps_state(self):
        manager = self.manager(max_gpu_users=1)
        manager.get_user_state("alice")
        with patch.object(manager, "_write_payload_files", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.get_user_state("bob")
        self.assertEqual(manager.pool_status()["gpu_active_users"], ["alice"])
        manager.reset_user_state("bob", persist=False)
        self.assertEqual(manager.pool_status()["gpu_active_users"], ["bob"])
        self.assertEqual(manager.pool_status()["cpu_cached_users"], ["alice"])

    def test_drift_cache_invalidated_at_same_step(self):
        manager = self.manager()
        state = manager.get_user_state("alice").clone()
        self.assertEqual(manager.compute_user_drift("alice")["drift_pct"], 0)
        state.w_up.add_(0.5)
        manager.update_user_state("alice", state, auto_save=False)
        self.assertGreater(manager.compute_user_drift("alice")["drift_pct"], 0)

    def test_full_fingerprint_detects_gate_change_and_snapshot_mismatch(self):
        manager = self.manager()
        state = manager.get_user_state("alice")
        state.step = 7
        manager.save_user_state("alice")
        manager.save_snapshot("alice", "v1")
        with torch.no_grad():
            self.model.fast_weight_pool.w_gate_init.add_(1)
        self.write_base()
        changed = self.manager()
        self.assertEqual(changed.base_signature, manager.base_signature)
        self.assertEqual(changed.get_user_state("alice").step, 0)
        with self.assertRaisesRegex(ValueError, "Snapshot base"):
            changed.load_snapshot("alice", "v1")

    def test_new_personas_survive_dtype_change(self):
        manager = self.manager(dtype=torch.float32)
        manager.get_user_state("alice").step = 7
        manager.save_user_state("alice")
        reloaded = self.manager(dtype=torch.bfloat16)
        self.assertEqual(reloaded.get_user_state("alice").step, 7)

    def test_user_aliases_and_snapshot_names_do_not_collide(self):
        manager = self.manager()
        manager.get_user_state("alice").step = 3
        manager.save_user_state("alice")
        manager.get_user_state("alice_state").step = 9
        manager.save_user_state("alice_state")
        reloaded = self.manager()
        self.assertEqual(reloaded.get_user_state("alice").step, 3)
        self.assertEqual(reloaded.get_user_state("alice_state").step, 9)
        first = manager.save_snapshot("alice", "a_b")
        second = manager.save_snapshot("alice_a", "b")
        self.assertNotEqual(first, second)
        with self.assertRaises(FileExistsError):
            manager.save_snapshot("alice", "a_b")
        with self.assertRaises(ValueError):
            manager.get_user_state("alice/bob")
        with self.assertRaises(ValueError):
            manager.save_snapshot("alice", "..")

    def test_unsupported_base_and_malformed_persona(self):
        manager = self.manager()
        manager.get_user_state("alice").step = 7
        path = manager.save_user_state("alice")
        payload = torch.load(path, weights_only=False)
        payload["fast_state"]["w_gate"] = torch.zeros(1, 1, 1)
        torch.save(payload, path)
        self.assertEqual(self.manager().get_user_state("alice").step, 0)
        torch.save({"config": self.config.to_dict()}, self.base)
        with self.assertRaisesRegex(ValueError, "Unsupported base"):
            self.manager()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_cached_multi_token_attention_matches_full_sequence(self):
        attn = SlidingWindowAttention(16, 2, 1, window_size=4).eval()
        x = torch.randn(1, 10, 16)
        with torch.no_grad():
            expected, _ = attn(x)
            _, cache = attn(x[:, :6])
            actual, cache = attn(x[:, 6:], cache, start_pos=6)
        torch.testing.assert_close(actual, expected[:, 6:], atol=1e-6, rtol=1e-5)
        self.assertEqual(cache[0].shape[2], 4)

    def test_causal_logits_and_gradients_with_checkpointing(self):
        for length in (4, 12):
            reference = uTTTModel(tiny_config()).train()
            ids = torch.randint(6, 512, (1, length))
            changed = ids.clone()
            changed[:, length // 2:] = (changed[:, length // 2:] + 13) % 512
            expected = reference(ids, labels=ids)
            expected.loss.backward()
            for checkpointed, segments in ((False, 1), (True, 1), (True, 2), (False, 2)):
                with self.subTest(length=length, checkpointed=checkpointed, segments=segments):
                    model = uTTTModel(tiny_config(gradient_checkpointing=checkpointed)).train()
                    model.load_state_dict(reference.state_dict())
                    for block in model.layers:
                        block.uttt.chunk_checkpoint_segments = segments
                    out = model(ids, labels=ids)
                    other = model(changed)
                    torch.testing.assert_close(out.logits[:, :length // 2], other.logits[:, :length // 2])
                    torch.testing.assert_close(out.logits, expected.logits)
                    out.loss.backward()
                    for (name, parameter), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
                        if ref.grad is not None:
                            self.assertIsNotNone(parameter.grad, name)
                            torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-5, rtol=2e-4, msg=name)

    def test_speculative_decoding_matches_greedy_with_sliding_cache(self):
        for seed in (1, 12, 123):
            torch.manual_seed(seed)
            model = uTTTModel(tiny_config()).eval()
            state = model.init_fast_state(detach=True)
            ids = torch.randint(6, 512, (1, 7))
            options = dict(input_ids=ids, fast_state=state, max_new_tokens=8, temperature=0.0, eos_token_id=None)
            expected = list(model.stream_generate(**options, speculative_draft_k=0))
            actual = list(model.stream_generate(**options, speculative_draft_k=3, dspark_conf_threshold=0))
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), 8)

    def test_all_ignored_labels_produce_finite_loss(self):
        model = uTTTModel(tiny_config()).train()
        ids = torch.ones((1, 4), dtype=torch.long)
        out = model(ids, labels=torch.full_like(ids, -100))
        self.assertEqual(out.ce_loss.item(), 0)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()

    def test_padding_does_not_decay_or_apply_momentum(self):
        pool = SharedFastWeightPool(16, 32)
        state = pool.init_state(batch_size=2, detach=True)
        state.s_up.fill_(0.1)
        state.s_gate.fill_(0.2)
        state.s_down.fill_(0.3)
        tensors = [torch.randn(2, 4, 16) for _ in range(3)]
        eta, alpha = torch.rand(2, 4, 1) * 0.01, torch.rand(2, 4, 1) * 0.01
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
        _, padded, _ = pool.chunk_step(state, *tensors, eta, alpha, mask)
        _, unpadded, _ = pool.chunk_step(state, *(t[:, :2] for t in tensors), eta[:, :2], alpha[:, :2], mask[:, :2])
        for name in ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down"):
            torch.testing.assert_close(getattr(padded, name)[0], getattr(unpadded, name)[0])
            torch.testing.assert_close(getattr(padded, name)[1], getattr(state, name)[1])


class TrainingTests(unittest.TestCase):
    def test_small_dataset_training_and_completed_resume_preserve_checkpoint(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.txt"
            data.write_text("<|user|> a <|assistant|> b", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "model": tiny_config().to_dict(),
                "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
                "training": {"micro_batch_size": 8, "grad_accum_steps": 1, "max_steps": 1,
                             "seq_len": 16, "pack_sequences": True},
            }))
            path = training.train(str(config), str(data), str(root / "base.pt"))
            before = path.read_bytes()
            training.train(str(config), str(data), str(path))
            self.assertEqual(before, path.read_bytes())

    def test_explicit_missing_or_empty_dataset_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            with self.assertRaises(FileNotFoundError):
                training.PersonaMetaDataset(SimpleUTF8Tokenizer(), data_path=str(path))
            path.touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                training.PersonaMetaDataset(SimpleUTF8Tokenizer(), data_path=str(path))

    def test_background_save_freezes_cpu_weights_and_surfaces_errors(self):
        model = uTTTModel(tiny_config())
        original = model.fast_weight_pool.w_up_init.detach().clone()
        entered, release = threading.Event(), threading.Event()
        save = torch.save

        def delayed_save(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test writer timed out")
            return save(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base.pt"
            try:
                with patch.object(training.torch, "save", side_effect=delayed_save):
                    training.export_base_checkpoint(model, None, model.config, path, 1, {}, async_save=True)
                    self.assertTrue(entered.wait(5))
                    with torch.no_grad():
                        model.fast_weight_pool.w_up_init.add_(1)
                    release.set()
                    training._wait_for_active_save()
            finally:
                release.set()
                training._wait_for_active_save()
            payload = torch.load(path, weights_only=False)
            torch.testing.assert_close(payload["init_fast_weights"]["w_up_init"], original)
            with patch.object(training.torch, "save", side_effect=OSError("disk full")):
                training.export_base_checkpoint(model, None, model.config, path, 2, {}, async_save=True)
                with self.assertRaisesRegex(RuntimeError, "Background checkpoint"):
                    training._wait_for_active_save()


class FakeEngine:
    def __init__(self):
        self.ckpt_manager = self
        self.threads = []
        self.closed = []

    def warmup(self, **kwargs):
        pass

    def pool_status(self):
        return {}

    def get_user_state(self, user_id):
        return None

    def compute_user_drift(self, user_id):
        return {"step": 0, "drift_pct": 0}

    def stream_interact_and_evolve(self, **kwargs):
        try:
            self.threads.append(threading.get_ident())
            yield {"type": "token", "content": "hello"}
            self.threads.append(threading.get_ident())
            yield {"type": "ttt_status", "result": {"persona_saved": True}}
        finally:
            self.closed.append(threading.get_ident())


class ServerTests(unittest.TestCase):
    def test_stream_holds_operation_lock_until_closed(self):
        engine = FakeEngine()

        async def exercise():
            with patch.object(service, "get_personalizer", return_value=engine):
                async with service.lifespan(service.app):
                    stream = service._stream_engine(engine, service.ChatRequest(prompt="hello"))
                    await anext(stream)
                    ran = threading.Event()
                    pending = asyncio.create_task(service._run_engine(ran.set))
                    await asyncio.sleep(0)
                    self.assertFalse(ran.is_set())
                    await stream.aclose()
                    await asyncio.wait_for(pending, 2)
                    self.assertTrue(ran.is_set())
        asyncio.run(exercise())
        self.assertEqual(engine.threads, engine.closed)

    def test_websocket_validates_recovers_and_keeps_generator_on_one_thread(self):
        engine = FakeEngine()
        with patch.object(service, "get_personalizer", return_value=engine), TestClient(service.app) as client:
            with client.websocket_connect("/ws/chat/alice") as ws:
                self.assertEqual(ws.receive_json()["type"], "session_ready")
                for invalid in ({"prompt": "hello", "max_new_tokens": -1}, {"prompt": "hello", "top_p": 2}, [], {"prompt": " "}):
                    ws.send_json(invalid)
                    self.assertEqual(ws.receive_json()["type"], "error")
                ws.send_json({"prompt": "hello"})
                self.assertEqual(ws.receive_json()["type"], "token")
                self.assertEqual(ws.receive_json()["type"], "ttt_status")
                ws.send_text("hello again")
                self.assertEqual(ws.receive_json()["type"], "token")
                self.assertEqual(ws.receive_json()["type"], "ttt_status")
        self.assertEqual(len(engine.closed), 2)
        self.assertEqual(len(set(engine.threads + engine.closed)), 1)


if __name__ == "__main__":
    unittest.main()

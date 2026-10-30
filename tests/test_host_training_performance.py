"""Host-memory optimizations must preserve sample order and immutable exports."""
from pathlib import Path
import tempfile
import threading
import tracemalloc
import unittest
from unittest import mock

try:
    import torch
    from engine.corpus import ResumableBatchSampler
    from model.uttt_model import uTTTConfig, uTTTModel
    import train_5090 as training
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch dependencies unavailable")
class SamplerPerformanceTests(unittest.TestCase):
    def legacy(self, size, batch_size, seed, epoch=0, offset=0):
        order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        limit = offset + ((size - offset) // batch_size) * batch_size if size >= batch_size else size
        return [order[start:min(start + batch_size, limit)] for start in range(offset, limit, batch_size)]

    def test_order_and_dropped_tails_match_legacy_for_small_and_large_batches(self):
        for size, batch_size in ((1, 4), (10, 1), (19, 4), (2503, 64)):
            for epoch in (0, 3):
                with self.subTest(size=size, batch_size=batch_size, epoch=epoch):
                    sampler = ResumableBatchSampler(size, batch_size, 39)
                    sampler.epoch = epoch
                    self.assertEqual(list(sampler), self.legacy(size, batch_size, 39, epoch))

    def test_rollback_reuses_permutation_but_seed_and_epoch_changes_refresh_it(self):
        sampler = ResumableBatchSampler(101, 4, 13)
        before = torch.get_rng_state().clone()
        with mock.patch.object(torch, "randperm", wraps=torch.randperm) as generate:
            original = list(sampler)
            sampler.advance(4)
            cursor = sampler.state_dict()
            self.assertEqual(list(sampler), original[1:])
            sampler.advance(4)
            sampler.load_state_dict(cursor)
            self.assertEqual(list(sampler), original[1:])
            self.assertEqual(generate.call_count, 1)
            sampler.load_state_dict({**cursor, "epoch": 1, "offset": 0})
            self.assertEqual(list(sampler), self.legacy(101, 4, 13, 1))
            self.assertEqual(generate.call_count, 3)  # sampler plus reference
            sampler.load_state_dict({**cursor, "seed": 19, "epoch": 1, "offset": 0})
            self.assertEqual(list(sampler), self.legacy(101, 4, 19, 1))
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_batch_resize_and_existing_prefetch_keep_their_own_order(self):
        sampler = ResumableBatchSampler(19, 4, 11)
        first_iterator = iter(sampler)
        first = next(first_iterator)
        sampler.advance(4)
        sampler.set_batch_size(3)
        expected = self.legacy(19, 3, 11, offset=4)
        batches = list(sampler)
        self.assertEqual(batches, expected)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual([len(batch) for batch in batches], [3] * 5)
        for batch in batches:
            sampler.advance(len(batch))
        self.assertEqual(sampler.state_dict()["epoch"], 1)
        self.assertEqual(sampler.state_dict()["offset"], 0)
        self.assertEqual([first, *list(first_iterator)], self.legacy(19, 4, 11))

    def test_resume_after_batch_resize_drops_only_the_incomplete_remaining_tail(self):
        sampler = ResumableBatchSampler(20, 3, 31)
        sampler.advance(3)
        sampler.advance(3)
        cursor = sampler.state_dict()
        restored = ResumableBatchSampler(20, 4, 99)
        restored.load_state_dict(cursor)
        batches = list(restored)
        self.assertEqual(batches, self.legacy(20, 4, 31, offset=6))
        self.assertEqual(len(batches), len(restored))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 4])
        for batch in batches:
            restored.advance(len(batch))
        self.assertEqual((restored.epoch, restored.offset), (1, 0))

    def test_large_sampler_does_not_allocate_a_python_integer_for_every_document(self):
        # Tracemalloc measures the boxed objects that caused the old memory
        # blow-up. The permutation's compact tensor allocation is separate.
        size = 150_000
        sampler = ResumableBatchSampler(size, 1, 7)
        tracemalloc.start()
        try:
            first = next(iter(sampler))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(first), 1)
        self.assertLess(peak, size * 8)


@unittest.skipIf(torch is None, "PyTorch dependencies unavailable")
class CheckpointPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = uTTTConfig(vocab_size=32, d_model=8, n_layers=1, n_heads=2,
                                 n_kv_heads=1, d_ff_fast=16, chunk_size=4)
        torch.manual_seed(12)
        self.model = uTTTModel(self.config)
        self.full, self.best = self.root / "full.pt", self.root / "best.pt"

    def tearDown(self):
        training._wait_for_active_save()
        self.directory.cleanup()

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def assert_same_weights(self, left, right):
        for partition in ("slow_weights", "init_fast_weights"):
            self.assertEqual(left[partition].keys(), right[partition].keys())
            for name in left[partition]:
                torch.testing.assert_close(left[partition][name], right[partition][name], atol=0, rtol=0)

    def test_full_and_best_share_one_snapshot_and_tied_storage(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001)
        self.model(torch.tensor([[6, 7, 8, 9]]), labels=torch.tensor([[6, 7, 8, 9]]),
                   update_fast_weights=False).loss.backward()
        optimizer.step()
        captured = []
        original_save = torch.save

        def record(payload, destination):
            captured.append(payload)
            return original_save(payload, destination)

        with mock.patch.object(torch, "save", record), mock.patch.object(self.model, "state_dict", wraps=self.model.state_dict) as snapshot:
            training.export_base_checkpoint(self.model, optimizer, self.config, self.full, 7,
                                            {"ce_loss": 1.2}, training_state={"cursor": 5}, best_save_path=self.best)
        self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(len(captured), 2)
        for name in captured[0]["slow_weights"]:
            self.assertIs(captured[0]["slow_weights"][name], captured[1]["slow_weights"][name])
        full, best = self.load(self.full), self.load(self.best)
        self.assert_same_weights(full, best)
        self.assertIn("optimizer_state", full)
        self.assertIn("training_state", full)
        self.assertNotIn("optimizer_state", best)
        self.assertNotIn("training_state", best)
        self.assertEqual(full["step"], best["step"])
        self.assertEqual(full["metrics"], best["metrics"])
        self.assertEqual(best["slow_weights"]["tok_embeddings.weight"].data_ptr(),
                         best["slow_weights"]["lm_head.weight"].data_ptr())
        restored = uTTTModel(self.config)
        restored.load_dual_base_state(full["slow_weights"], full["init_fast_weights"])
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
        restored_optimizer.load_state_dict(full["optimizer_state"])
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))

    def test_equal_but_untied_parameters_are_not_merged(self):
        self.config.tie_word_embeddings = False
        model = uTTTModel(self.config)
        with torch.no_grad():
            model.lm_head.weight.copy_(model.tok_embeddings.weight)
        training.export_base_checkpoint(model, None, self.config, self.full, 1, {})
        slow = self.load(self.full)["slow_weights"]
        self.assertNotEqual(slow["tok_embeddings.weight"].data_ptr(), slow["lm_head.weight"].data_ptr())

    def test_async_pair_is_immutable_after_the_training_thread_continues(self):
        entered, release = threading.Event(), threading.Event()
        original_save = torch.save
        expected = {key: value.clone() for key, value in self.model.state_dict().items()}
        metrics = {"samples": [{"reply": "before"}]}
        state = {"rng": torch.tensor([1, 2, 3])}

        def pause(payload, destination):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test writer was not released")
            return original_save(payload, destination)

        try:
            with mock.patch.object(torch, "save", pause):
                training.export_base_checkpoint(self.model, None, self.config, self.full, 1, metrics,
                                                async_save=True, training_state=state, best_save_path=self.best)
                self.assertTrue(entered.wait(1))
                self.assertTrue(training._ACTIVE_SAVE_THREAD.is_alive())
                with torch.no_grad():
                    self.model.tok_embeddings.weight.add_(9)
                metrics["samples"][0]["reply"] = "after"
                state["rng"].zero_()
                release.set()
                training._wait_for_active_save()
        finally:
            release.set()
        full, best = self.load(self.full), self.load(self.best)
        self.assert_same_weights(full, best)
        torch.testing.assert_close(best["slow_weights"]["tok_embeddings.weight"], expected["tok_embeddings.weight"], atol=0, rtol=0)
        self.assertEqual(best["metrics"]["samples"][0]["reply"], "before")
        torch.testing.assert_close(full["training_state"]["rng"], torch.tensor([1, 2, 3]))

    def test_failed_primary_write_preserves_both_files_and_removes_temporary_file(self):
        training.export_base_checkpoint(self.model, None, self.config, self.full, 1, {}, best_save_path=self.best)
        before = (self.full.read_bytes(), self.best.read_bytes())
        with mock.patch.object(torch, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                training.export_base_checkpoint(self.model, None, self.config, self.full, 2, {}, best_save_path=self.best)
        self.assertEqual((self.full.read_bytes(), self.best.read_bytes()), before)
        self.assertFalse(Path(str(self.full) + ".tmp").exists())

    def test_async_best_failure_is_reported_and_old_best_remains_readable(self):
        training.export_base_checkpoint(self.model, None, self.config, self.full, 1, {}, best_save_path=self.best)
        original_save = torch.save
        count = 0

        def fail_best(payload, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("best disk full")
            return original_save(payload, destination)

        with mock.patch.object(torch, "save", fail_best):
            training.export_base_checkpoint(self.model, None, self.config, self.full, 2, {},
                                            async_save=True, best_save_path=self.best)
            training._ACTIVE_SAVE_THREAD.join()
        with self.assertRaisesRegex(RuntimeError, "Background checkpoint save failed"):
            training._raise_active_save_error()
        self.assertEqual(self.load(self.full)["step"], 2)
        self.assertEqual(self.load(self.best)["step"], 1)
        self.assertFalse(Path(str(self.best) + ".tmp").exists())

    def test_best_path_cannot_overwrite_the_full_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            training.export_base_checkpoint(self.model, None, self.config, self.full, 1, {}, best_save_path=self.full)
        self.assertFalse(self.full.exists())

    def test_failed_backup_link_preserves_the_previous_recovery_file(self):
        training.export_base_checkpoint(self.model, None, self.config, self.full, 1, {})
        training.export_base_checkpoint(self.model, None, self.config, self.full, 2, {})
        previous = Path(str(self.full) + ".prev")
        self.assertEqual(self.load(previous)["step"], 1)
        with mock.patch.object(training.os, "link", side_effect=OSError("hardlinks unavailable")):
            training.export_base_checkpoint(self.model, None, self.config, self.full, 3, {})
        self.assertEqual(self.load(self.full)["step"], 3)
        self.assertEqual(self.load(previous)["step"], 1)
        self.assertFalse(Path(str(previous) + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()

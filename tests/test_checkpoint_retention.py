"""Compact storage keeps atomic recovery and model-only selection distinct."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import signal
import tempfile
import threading
import unittest
from unittest import mock

try:
    import torch
    import yaml
    import train_5090 as training
    from model.uttt_model import uTTTConfig, uTTTModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class CheckpointRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = uTTTConfig(vocab_size=32, d_model=8, n_layers=1,
                                 n_heads=2, n_kv_heads=1, d_ff_fast=16, chunk_size=4)
        torch.manual_seed(29)
        self.model = uTTTModel(self.config)
        self.full, self.best = self.root / "model.pt", self.root / "model.best.pt"

    def tearDown(self):
        training._wait_for_active_save()
        self.directory.cleanup()

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def export(self, step, **options):
        return training.export_base_checkpoint(self.model, None, self.config, self.full,
                                               step, {}, best_save_path=self.best, **options)

    def test_defaults_remain_legacy_and_options_require_real_booleans(self):
        self.assertEqual(training.checkpoint_save_policy({}), {
            "checkpoint_keep_previous": True,
            "checkpoint_full_on_validation_improvement": True,
            "checkpoint_save_initial_full": False,
        })
        for name in training.checkpoint_save_policy({}):
            for value in (None, 0, 1, "false"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    training.checkpoint_save_policy({name: value})
        self.export(1)
        self.export(2)
        self.assertEqual(self.load(Path(str(self.full) + ".prev"))["step"], 1)
        self.assertEqual(self.load(Path(str(self.best) + ".prev"))["step"], 1)
        with self.assertRaisesRegex(ValueError, "requires best_save_path"):
            training.export_base_checkpoint(self.model, None, self.config, self.full, 3, {}, save_full=False)

    def test_compact_commit_removes_only_its_previous_files_after_success(self):
        self.export(1)
        self.export(2)
        unrelated = self.root / "unrelated.prev"
        unrelated.write_bytes(b"must survive")
        self.export(3, keep_previous=False, best_before_full=True)
        self.assertEqual(self.load(self.full)["step"], 3)
        self.assertEqual(self.load(self.best)["step"], 3)
        self.assertEqual(unrelated.read_bytes(), b"must survive")
        self.assertFalse(Path(str(self.full) + ".prev").exists())
        self.assertFalse(Path(str(self.best) + ".prev").exists())
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_write_fsync_and_replace_failures_preserve_current_and_previous(self):
        self.export(1)
        self.export(2)
        paths = [self.full, self.best, Path(str(self.full) + ".prev"), Path(str(self.best) + ".prev")]
        before = {path: path.read_bytes() for path in paths}
        for target, name in ((torch, "save"), (training.os, "fsync"), (training.os, "replace")):
            with self.subTest(failure=name), mock.patch.object(target, name, side_effect=OSError(name)):
                with self.assertRaisesRegex(OSError, name):
                    training.export_base_checkpoint(self.model, None, self.config, self.full, 3, {},
                                                    keep_previous=False)
            self.assertEqual({path: path.read_bytes() for path in paths}, before)
            self.assertFalse(list(self.root.glob("*.tmp")))

    def test_best_only_never_snapshots_optimizer_or_training_state_or_rewrites_latest(self):
        self.export(1, keep_previous=False)
        before = self.full.read_bytes()
        optimizer = mock.Mock()
        optimizer.state_dict.side_effect = AssertionError("optimizer snapshot is unnecessary")
        state = {"sentinel": torch.tensor([99])}
        copied = []
        original = training._cpu_copy_nested

        def copy(value, memo=None):
            self.assertIsNot(value, state)
            copied.append(value)
            return original(value, memo)

        with mock.patch.object(training, "_cpu_copy_nested", side_effect=copy):
            result = training.export_base_checkpoint(self.model, optimizer, self.config, self.full,
                                                     2, {"validation_ce": .7}, training_state=state,
                                                     best_save_path=self.best, save_full=False,
                                                     keep_previous=False)
        self.assertEqual(result, self.best)
        self.assertEqual(self.full.read_bytes(), before)
        saved = self.load(self.best)
        self.assertEqual(saved["step"], 2)
        self.assertNotIn("optimizer_state", saved)
        self.assertNotIn("training_state", saved)
        optimizer.state_dict.assert_not_called()
        self.assertTrue(copied)

    def test_best_only_low_memory_stream_does_not_clone_or_start_a_background_thread(self):
        original_save = torch.save
        captured = []
        optimizer = mock.Mock()
        optimizer.state_dict.side_effect = AssertionError("optimizer should not be read")

        def stream(payload, destination):
            self.assertEqual(payload["slow_weights"]["tok_embeddings.weight"].data_ptr(),
                             self.model.tok_embeddings.weight.data_ptr())
            self.assertNotIn("optimizer_state", payload)
            self.assertNotIn("training_state", payload)
            captured.append(destination.name)
            return original_save(payload, destination)

        with mock.patch.object(training, "_checkpoint_storage_bytes", return_value=(100, True)), \
                mock.patch.object(training, "_host_memory_status", return_value={"available": 1}), \
                mock.patch.object(training, "_cpu_copy_nested", side_effect=AssertionError("full CPU clone")), \
                mock.patch.object(training.threading, "Thread", side_effect=AssertionError("background thread")), \
                mock.patch.object(torch, "save", side_effect=stream):
            training.export_base_checkpoint(self.model, optimizer, self.config, self.full, 3, {},
                                            training_state={"large_unused_state": torch.ones(10)},
                                            best_save_path=self.best, save_full=False,
                                            keep_previous=False, async_save=True)
        self.assertEqual(len(captured), 1)
        self.assertEqual(self.load(self.best)["step"], 3)
        self.assertFalse(self.full.exists())

    def test_async_best_only_is_immutable_when_training_continues(self):
        self.export(1, keep_previous=False)
        latest = self.full.read_bytes()
        expected = self.model.tok_embeddings.weight.detach().clone()
        entered, release = threading.Event(), threading.Event()
        save = torch.save

        def pause(payload, destination):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("writer not released")
            return save(payload, destination)

        try:
            with mock.patch.object(torch, "save", side_effect=pause):
                self.export(2, save_full=False, keep_previous=False, async_save=True)
                self.assertTrue(entered.wait(1))
                with torch.no_grad():
                    self.model.tok_embeddings.weight.add_(8)
                release.set()
                training._wait_for_active_save()
        finally:
            release.set()
        self.assertEqual(self.full.read_bytes(), latest)
        torch.testing.assert_close(self.load(self.best)["slow_weights"]["tok_embeddings.weight"],
                                   expected, atol=0, rtol=0)

    def metadata(self, score):
        return {"training_pipeline": 3, "training_stage": "sft", "dataset_fingerprint": "data-A",
                "run_id": "retention-test", "validation_selection_metric": "validation_ce",
                "validation_ce": score, "best_validation_ce": score, "best_validation_score": score}

    def test_best_first_then_failed_full_leaves_valid_newer_best_without_losing_latest(self):
        training.export_base_checkpoint(self.model, None, self.config, self.full, 1, self.metadata(1.),
                                        training_state={"data_cursor": {"offset": 1}},
                                        best_save_path=self.best, keep_previous=False, best_before_full=True)
        before = self.full.read_bytes()
        save = torch.save

        def fail_full(payload, destination):
            if Path(destination.name) == Path(str(self.full) + ".tmp"):
                raise OSError("full write interrupted")
            return save(payload, destination)

        with mock.patch.object(torch, "save", side_effect=fail_full):
            with self.assertRaisesRegex(OSError, "full write interrupted"):
                training.export_base_checkpoint(self.model, None, self.config, self.full, 2, self.metadata(.5),
                                                training_state={"data_cursor": {"offset": 2}},
                                                best_save_path=self.best, keep_previous=False, best_before_full=True)
        self.assertEqual(self.full.read_bytes(), before)
        self.assertEqual(self.load(self.best)["step"], 2)
        self.assertEqual(training.resume_compact_validation_best(self.load(self.full), self.best,
                                                                "validation_ce", 10), (.5, .5))
        self.assertEqual(self.load(self.full)["training_state"]["data_cursor"]["offset"], 1)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_compact_resume_rejects_missing_foreign_and_inconsistent_selection_files(self):
        saved = {"config": self.config.to_dict(), "step": 1, "metrics": self.metadata(1.)}
        with self.assertRaisesRegex(ValueError, "missing"):
            training.resume_compact_validation_best(saved, self.best, "validation_ce", 10)
        for change, message in (({"dataset_fingerprint": "other"}, "dataset_fingerprint"),
                                ({"code_revision": "foreign-revision"}, "code_revision"),
                                ({"parameter_scope": "fast"}, "parameter_scope"),
                                ({"best_validation_score": .2}, "matching validation score"),
                                ({"validation_ce": 1.2, "best_validation_score": 1.2,
                                  "best_validation_ce": 1.2}, "older than")):
            torch.save({"config": self.config.to_dict(), "step": 2,
                        "metrics": {**self.metadata(.5), **change}}, self.best)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                training.resume_compact_validation_best(saved, self.best, "validation_ce", 10)

    def fixture(self, directory, **settings):
        directory.mkdir(parents=True, exist_ok=True)
        data = directory / "data.jsonl"
        data.write_text("\n".join(json.dumps({"text": f"<|user|> Q{index}?\n<|assistant|> A{index}.<|eos|>"})
                                  for index in range(20)), encoding="utf-8")
        content = {
            "model": {"vocab_size": 512, "d_model": 8, "d_ff_fast": 16, "n_layers": 1,
                      "n_heads": 2, "n_kv_heads": 1, "chunk_size": 8,
                      "sliding_window_size": 16, "max_seq_len": 64, "dropout": .1},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": "sft", "micro_batch_size": 1, "grad_accum_steps": 1,
                         "max_steps": 4, "seq_len": 64, "seed": 714, "warmup_steps": 0,
                         "outer_lr": 2.e-5, "meta_fast_lr": 1.e-5, "weight_decay": .01,
                         "pack_sequences": False, "gradient_checkpointing": False,
                         "trim_sft_padding": False, "validation_fraction": .5,
                         "eval_interval": 1, "eval_samples": 8, "save_interval": 3, "log_interval": 1,
                         "checkpoint_keep_previous": False,
                         "checkpoint_full_on_validation_improvement": False,
                         "checkpoint_save_initial_full": True, **settings},
        }
        config = directory / "config.yaml"
        config.write_text(yaml.safe_dump(content), encoding="utf-8")
        return config, data

    @staticmethod
    def report(score):
        return {"validation_ce": score, "validation_frozen_ce": score,
                "validation_targets": 10, "validation_examples": 2}

    def test_actual_schedule_writes_initial_periodic_final_full_and_intermediate_best_only(self):
        config, data = self.fixture(self.root)
        observations = []
        export = training.export_base_checkpoint

        def record(*args, **kwargs):
            result = export(*args, **kwargs)
            training._wait_for_active_save()
            observations.append((kwargs["step"], self.load(self.full)["step"], self.load(self.best)["step"],
                                 kwargs["save_full"], kwargs["training_state"] is not None))
            return result

        with redirect_stdout(io.StringIO()), \
                mock.patch.object(training, "evaluate_language_model", side_effect=[self.report(x) for x in (1., .8, .9, .7)]), \
                mock.patch.object(training, "export_base_checkpoint", side_effect=record):
            training.train(str(config), str(data), str(self.full), resume=False)
        self.assertEqual(observations, [(1, 1, 1, True, True), (2, 1, 2, False, False),
                                        (3, 3, 2, True, True), (4, 4, 4, True, True)])
        self.assertFalse(list(self.root.glob("*.prev")))
        self.assertEqual(self.load(self.full)["training_state"]["data_cursor"]["offset"], 4)

    def test_resuming_older_latest_preserves_newer_best_and_exact_training_trajectory(self):
        config, data = self.fixture(self.root / "phase")
        expected_path = self.root / "expected.pt"
        with redirect_stdout(io.StringIO()), mock.patch.object(
                training, "evaluate_language_model", side_effect=[self.report(x) for x in (1., .8, .9, .95)]):
            training.train(str(config), str(data), str(expected_path), resume=False)
        export = training.export_base_checkpoint

        def power_loss(*args, **kwargs):
            result = export(*args, **kwargs)
            training._wait_for_active_save()
            if kwargs["step"] == 2:
                raise RuntimeError("simulated VM loss after best commit")
            return result

        with redirect_stdout(io.StringIO()), \
                mock.patch.object(training, "evaluate_language_model", side_effect=[self.report(x) for x in (1., .8)]), \
                mock.patch.object(training, "export_base_checkpoint", side_effect=power_loss):
            with self.assertRaisesRegex(RuntimeError, "simulated VM loss"):
                training.train(str(config), str(data), str(self.full), resume=False)
        self.assertEqual(self.load(self.full)["step"], 1)
        best_before = self.best.read_bytes()
        self.assertEqual(self.load(self.best)["step"], 2)
        with redirect_stdout(io.StringIO()), mock.patch.object(
                training, "evaluate_language_model", side_effect=[self.report(x) for x in (.9, .85, .95)]):
            training.train(str(config), str(data), str(self.full), resume=True)
        actual, expected = self.load(self.full), self.load(expected_path)
        self.assertEqual(self.best.read_bytes(), best_before)
        self.assertEqual(actual["metrics"]["best_validation_score"], .8)
        self.assertEqual(actual["training_state"]["data_cursor"], expected["training_state"]["data_cursor"])
        for partition in ("slow_weights", "init_fast_weights"):
            for key, value in expected[partition].items():
                torch.testing.assert_close(actual[partition][key], value, atol=0, rtol=0)

    def test_signal_during_validation_always_saves_full_resume_state(self):
        config, data = self.fixture(self.root, save_interval=100)
        calls = 0

        def validate(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                training._graceful_signal_handler(signal.SIGTERM, None)
            return self.report(1. if calls == 1 else 1.2)

        with redirect_stdout(io.StringIO()), mock.patch.object(training, "evaluate_language_model", side_effect=validate):
            with self.assertRaises(SystemExit) as stopped:
                training.train(str(config), str(data), str(self.full), resume=False)
        self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
        saved = self.load(self.full)
        self.assertEqual(saved["step"], 2)
        self.assertTrue(saved["optimizer_state"]["state"])
        self.assertEqual(saved["training_state"]["data_cursor"]["offset"], 2)
        self.assertIn("rng", saved["training_state"])
        self.assertEqual(self.load(self.best)["step"], 1)


if __name__ == "__main__":
    unittest.main()

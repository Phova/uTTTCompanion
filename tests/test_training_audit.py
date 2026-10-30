"""Training replay, failure recovery and deployment ownership regressions."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from deploy import stop_training as stopping
from deploy.stop_training import select_training_processes

try:
    import torch
    import yaml
    from engine.corpus import ResumableBatchSampler
    import train_5090 as training
except ImportError:
    torch = None


class DeploymentAuditTests(unittest.TestCase):
    def test_stop_selects_session_descendants_in_workspace_and_pipeline_before_child(self):
        workspace = Path("/project").resolve()
        table = {
            10: {"parent": 1, "cwd": workspace, "scripts": {"bash"}},
            11: {"parent": 10, "cwd": workspace / "runs" / "snapshot", "scripts": {"run_training_pipeline.py"}},
            12: {"parent": 11, "cwd": workspace / "runs" / "snapshot", "scripts": {"train_5090.py"}},
            13: {"parent": 10, "cwd": workspace.parent / "other", "scripts": {"train_5090.py"}},
            14: {"parent": 12, "cwd": workspace / "runs" / "snapshot", "scripts": {"train_5090.py"}},
            15: {"parent": 14, "cwd": workspace / "runs" / "snapshot", "scripts": {"train_5090.py"}},
            20: {"parent": 1, "cwd": workspace, "scripts": {"train_5090.py"}},
        }
        self.assertEqual(select_training_processes(table, [10], workspace), [11, 12])

    def test_stop_signals_pipeline_then_primary_and_leaves_dataloader_workers_to_trainer(self):
        workspace = Path("/project").resolve()
        table = {
            10: {"parent": 1, "cwd": workspace, "scripts": {"bash"}, "start_time": 100},
            11: {"parent": 10, "cwd": workspace, "scripts": {"run_training_pipeline.py"}, "start_time": 101},
            12: {"parent": 11, "cwd": workspace, "scripts": {"train_5090.py"}, "start_time": 102},
            13: {"parent": 12, "cwd": workspace, "scripts": {"train_5090.py"}, "start_time": 103},
            14: {"parent": 12, "cwd": workspace, "scripts": {"train_5090.py"}, "start_time": 104},
        }
        with mock.patch.object(stopping, "process_table", side_effect=[table, {}]), mock.patch.object(stopping, "_signal_process") as send:
            self.assertEqual(stopping.stop([10], workspace), [11, 12])
        self.assertEqual([call.args[0] for call in send.call_args_list], [11, 12])

    def test_live_foreign_session_or_stale_reused_pid_is_refused_without_signals(self):
        workspace = Path("/project").resolve()
        for scripts in ({"bash"}, {"train_5090.py"}):
            with self.subTest(scripts=scripts):
                table = {10: {"parent": 1, "cwd": workspace.parent / "different-project",
                              "scripts": scripts, "start_time": 201}}
                with mock.patch.object(stopping, "process_table", return_value=table), mock.patch.object(stopping, "_signal_process") as send:
                    with self.assertRaisesRegex(RuntimeError, "No owned primary process.*preserving"):
                        stopping.stop([10], workspace)
                send.assert_not_called()

    def test_dead_pid_file_does_not_select_an_unrelated_trainer(self):
        workspace = Path("/project").resolve()
        table = {20: {"parent": 1, "cwd": workspace, "scripts": {"train_5090.py"}, "start_time": 300}}
        with mock.patch.object(stopping, "process_table", return_value=table), mock.patch.object(stopping, "_signal_process") as send:
            self.assertEqual(stopping.stop([999], workspace), [])
        send.assert_not_called()

    def test_pid_reuse_between_selection_and_signal_is_rejected(self):
        workspace = Path("/project").resolve()
        initial = {12: {"parent": 1, "cwd": workspace, "scripts": {"train_5090.py"}, "start_time": 400}}
        reused = {12: {**initial[12], "start_time": 500}}
        with mock.patch.object(stopping, "process_table", side_effect=[initial, reused]), \
                mock.patch.object(stopping.os, "pidfd_open", return_value=77, create=True), \
                mock.patch.object(stopping.signal, "pidfd_send_signal", create=True) as send, \
                mock.patch.object(stopping.os, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "PID 12 was reused"):
                stopping.stop([12], workspace)
        send.assert_not_called()
        close.assert_called_once_with(77)

    def test_wait_does_not_follow_an_unrelated_process_reusing_the_trainers_pid(self):
        workspace = Path("/project").resolve()
        initial = {12: {"parent": 1, "cwd": workspace, "scripts": {"train_5090.py"}, "start_time": 600}}
        reused = {12: {**initial[12], "start_time": 700}}
        with mock.patch.object(stopping, "process_table", side_effect=[initial, reused]), mock.patch.object(stopping, "_signal_process"):
            self.assertEqual(stopping.stop([12], workspace), [12])

    def test_pipeline_rejects_invalid_run_id_and_nonpositive_steps_before_launching(self):
        script = Path(__file__).resolve().parents[1] / "deploy" / "run_training_pipeline.py"
        for arguments in (("--run-id", "../escape"), ("--pretrain-steps", "0"), ("--sft-steps", "-1")):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class TrainingAuditTests(unittest.TestCase):
    def fixture(self, root, *, steps=3, accumulation=2, dropout=0.0):
        data = root / "data.txt"
        data.write_text("\n".join(f"<|user|> {index}? <|assistant|> {index}." for index in range(12)), encoding="utf-8")
        config = root / "config.yaml"
        config.write_text(yaml.safe_dump({
            "model": {"vocab_size": 512, "d_model": 16, "n_layers": 1, "n_heads": 2,
                      "n_kv_heads": 1, "d_ff_fast": 32, "chunk_size": 4, "dropout": dropout},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": "pretrain", "micro_batch_size": 1, "grad_accum_steps": accumulation,
                         "max_steps": steps, "seq_len": 16, "save_interval": 1,
                         "gradient_checkpointing": False, "seed": 173},
        }), encoding="utf-8")
        return config, data

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def assert_weights_close(self, actual, expected):
        for partition in ("slow_weights", "init_fast_weights"):
            for name in expected[partition]:
                torch.testing.assert_close(actual[partition][name], expected[partition][name],
                                           rtol=1e-6, atol=1e-7, msg=f"{partition}.{name}")

    def test_prefetch_does_not_commit_data_and_rollback_replays_exact_batches(self):
        sampler = ResumableBatchSampler(10, 3, seed=5)
        prefetched = list(iter(sampler))
        self.assertEqual(sampler.state_dict()["offset"], 0)
        sampler.advance(3)
        checkpoint = sampler.state_dict()
        restored = ResumableBatchSampler(10, 3, seed=999)
        restored.load_state_dict(checkpoint)
        self.assertEqual(list(iter(restored)), prefetched[1:])
        sampler.advance(3)
        sampler.load_state_dict(checkpoint)
        self.assertEqual(list(iter(sampler)), prefetched[1:])
        restored.advance(3)
        restored.advance(3)
        self.assertEqual(restored.state_dict()["epoch"], 1)
        self.assertEqual(restored.state_dict()["offset"], 0)

    def test_signal_saves_complete_step_and_resume_matches_uninterrupted_dropout_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root, dropout=0.1)
            expected_path = training.train(str(config), str(data), str(root / "reference.pt"))
            optimizer_step = torch.optim.AdamW.step

            def stop_after_update(optimizer, *args, **kwargs):
                result = optimizer_step(optimizer, *args, **kwargs)
                training._graceful_signal_handler(signal.SIGTERM, None)
                return result

            resumed_path = root / "resumed.pt"
            with mock.patch.object(torch.optim.AdamW, "step", stop_after_update):
                with self.assertRaises(SystemExit) as stopped:
                    training.train(str(config), str(data), str(resumed_path))
            self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
            self.assertIsNone(training._STOP_REQUESTED)
            self.assertEqual(self.load(resumed_path)["step"], 1)
            training.train(str(config), str(data), str(resumed_path))
            expected, actual = self.load(expected_path), self.load(resumed_path)
            self.assertEqual(actual["training_state"]["data_cursor"], expected["training_state"]["data_cursor"])
            self.assertEqual(actual["metrics"]["tokens_seen"], expected["metrics"]["tokens_seen"])
            self.assert_weights_close(actual, expected)

    def test_partial_accumulation_oom_retries_consumed_samples_and_preserves_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root, steps=1)
            reference = self.load(training.train(str(config), str(data), str(root / "reference.pt")))
            original = training.uTTTModel.forward
            observed = []

            def fail_second_microbatch(model, *args, **kwargs):
                observed.append(kwargs["input_ids"].detach().clone())
                if len(observed) == 2:
                    raise torch.OutOfMemoryError("injected partial accumulation failure")
                return original(model, *args, **kwargs)

            with mock.patch.object(training.uTTTModel, "forward", fail_second_microbatch):
                recovered = self.load(training.train(str(config), str(data), str(root / "recovered.pt")))
            self.assertEqual(len(observed), 4)
            torch.testing.assert_close(observed[0], observed[2])
            torch.testing.assert_close(observed[1], observed[3])
            self.assertEqual(recovered["metrics"]["tokens_seen"], reference["metrics"]["tokens_seen"])
            self.assertEqual(recovered["training_state"]["data_cursor"], reference["training_state"]["data_cursor"])
            self.assert_weights_close(recovered, reference)

    def test_corrupt_or_incompatible_resume_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root, steps=1)
            corrupt = root / "corrupt.pt"
            corrupt.write_bytes(b"unfinished checkpoint")
            with self.assertRaisesRegex(RuntimeError, "No readable checkpoint"):
                training.train(str(config), str(data), str(corrupt))
            self.assertEqual(corrupt.read_bytes(), b"unfinished checkpoint")
            good = training.train(str(config), str(data), str(root / "good.pt"))
            saved = self.load(good)
            saved["config"]["n_heads"] *= 2
            torch.save(saved, good)
            before = good.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "checkpoint preserved"):
                training.train(str(config), str(data), str(good), override_steps=2)
            self.assertEqual(good.read_bytes(), before)

    def test_meta_resume_rejects_legacy_inner_loop_without_changing_language_stage_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root, steps=1)
            checkpoint = training.train(str(config), str(data), str(root / "base.pt"))
            saved = self.load(checkpoint)
            self.assertEqual(saved["metrics"]["inner_loop_protocol"], training.uTTTModel.INNER_LOOP_PROTOCOL)
            saved["metrics"].pop("inner_loop_protocol")
            # Pretrain/SFT checkpoints remain loadable: the protocol changes
            # concern learned inner-loop adaptation, which is off in pretrain.
            for stage in ("pretrain", "sft"):
                saved["metrics"]["training_stage"] = stage
                torch.save(saved, checkpoint)
                training.train(str(config), str(data), str(checkpoint), stage=stage)
            saved["metrics"]["training_stage"] = "meta"
            torch.save(saved, checkpoint)
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "inner-loop protocol.*--init-from"):
                training.train(str(config), str(data), str(checkpoint), stage="meta", override_steps=2)
            self.assertEqual(checkpoint.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

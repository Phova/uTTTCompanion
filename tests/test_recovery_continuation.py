"""A fresh recovery phase can inherit full state without mutating its parent."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy import run_recovery_training as runner


class FakeProcess:
    pid = 4132
    stdout = ()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def poll(self):
        return None

    def send_signal(self, signum):
        pass

    def wait(self):
        return 0


class RecoveryContinuationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=cache)))
        (self.root / "deploy").mkdir()
        self.source = self.root / "parent.pt"
        self.source.write_bytes(b"immutable-full-parent")
        self.output = self.root / "continued.pt"
        self.config, self.data, self.tasks = (self.root / name for name in ("config.yaml", "data.jsonl", "tasks.jsonl"))
        self.config.write_text("config", encoding="utf-8")
        self.data.write_text("data", encoding="utf-8")
        self.tasks.write_text("\n".join(json.dumps({"prompt": str(i), "accepted": [str(i)]}) for i in range(4)), encoding="utf-8")
        self.settings = {"training": {"stage": "sft", "max_steps": 6, "lr_schedule_start_step": 2,
                                      "micro_batch_size": 1, "grad_accum_steps": 16, "seed": 20261107,
                                      "validation_fraction": .02, "validation_selection_metric": "validation_example_ce"}}
        self.parent = {"step": 2, "has_optimizer": True, "has_training_state": True,
                       "metrics": {"training_stage": "sft", "training_pipeline": 3, "run_id": "parent",
                                   "code_revision": "parent-revision", "dataset_fingerprint": "fixed-data",
                                   "parameter_scope": "all", "total_loss": 1.2,
                                   "validation_selection_metric": "validation_example_ce"},
                       "training_state_summary": {"micro_batch_size": 1, "grad_accum_steps": 16,
                                                  "data_cursor": {"version": 1, "size": 1000, "seed": 20261107,
                                                                  "epoch": 0, "offset": 32}}}
        self.argv = ["--config", str(self.config), "--data-path", str(self.data),
                     "--init-from", str(self.source), "--output-ckpt", str(self.output),
                     "--tasks", str(self.tasks), "--steps", "6", "--run-id", "child"]
        self.commands = []
        self.fail_train = False
        self.stack.enter_context(patch.object(runner, "__file__", str(self.root / "deploy/run_recovery_training.py")))
        self.stack.enter_context(patch.object(runner, "load_config", return_value=self.settings))
        self.stack.enter_context(patch.object(runner, "dataset_identity", return_value={
            "fingerprint": "fixed-data", "documents": 1000, "window_protocol": "conditioned_turn_v1"}))
        self.stack.enter_context(patch.object(runner, "checkpoint_info", side_effect=self.checkpoint_info))
        self.stack.enter_context(patch.dict(os.environ, {"UTTT_CODE_REVISION": "child-revision"}))
        self.stack.enter_context(patch.dict(sys.modules, {"fcntl": SimpleNamespace(
            LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)}))
        self.stack.enter_context(patch.object(runner.signal, "signal", return_value=None))
        self.stack.enter_context(patch.object(runner.subprocess, "Popen", side_effect=self.spawn))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def checkpoint_info(self, path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        info = json.loads(json.dumps(self.parent))
        if Path(path) != self.source:
            info["step"] = 6
            info["metrics"].update(run_id="child", code_revision="child-revision")
            info["training_state_summary"]["data_cursor"]["offset"] = 96
        return info

    def spawn(self, command, **kwargs):
        self.commands.append(command)
        if "train_5090.py" in command:
            if self.fail_train:
                self.fail_train = False
                raise RuntimeError("simulated failure before the first child checkpoint")
            self.output.write_bytes(b"child-full")
            self.output.with_suffix(".best.pt").write_bytes(b"child-best")
        else:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            count = 2 if checkpoint == self.source else 3
            report = {"results": [{"prompt": str(i), "exact_match": i < count, "eos_generated": True}
                                  for i in range(4)]}
            Path(command[command.index("--output") + 1]).write_text(json.dumps(report), encoding="utf-8")
        return FakeProcess()

    def state(self):
        return json.loads((self.root / "logs/recovery_child/state.json").read_text(encoding="utf-8"))

    def fresh(self):
        return runner.main(self.argv + ["--continue-from", str(self.source)])

    def train_commands(self):
        return [command for command in self.commands if "train_5090.py" in command]

    def test_fresh_phase_pins_parent_identity_and_routes_full_state_without_weight_only_initialization(self):
        before = self.source.read_bytes()
        self.assertEqual(self.fresh(), 0)
        state = self.state()
        lineage = state["binding"]["continuation"]
        self.assertEqual(lineage["sha256"], runner.digest(self.source))
        self.assertEqual(lineage["step"], 2)
        self.assertEqual(lineage["run_id"], "parent")
        self.assertEqual(lineage["code_revision"], "parent-revision")
        self.assertEqual(lineage["data_cursor"]["offset"], 32)
        self.assertEqual(state["resume_action"], "new_continuation")
        command = self.train_commands()[0]
        self.assertEqual(command[command.index("--continue-from") + 1], str(self.source))
        self.assertNotIn("--init-from", command)
        self.assertNotIn("--no-resume", command)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertTrue(state["quality_passed"])
        self.assertFalse(state["online_writes_allowed"])

    def test_resume_uses_child_checkpoint_and_retains_the_original_continuation_binding(self):
        self.fresh()
        original = self.state()["binding"]
        self.assertEqual(runner.main(self.argv + ["--resume"]), 0)
        self.assertEqual(self.state()["binding"], original)
        self.assertEqual(self.state()["resume_action"], "full_checkpoint")
        command = self.train_commands()[-1]
        self.assertNotIn("--continue-from", command)
        self.assertNotIn("--init-from", command)
        self.assertNotIn("--no-resume", command)

    def test_changed_parent_is_rejected_before_restarting_or_mutating_run_state(self):
        self.fresh()
        state_path = self.root / "logs/recovery_child/state.json"
        original = state_path.read_bytes()
        commands = len(self.commands)
        self.source.write_bytes(b"replaced-parent")
        with self.assertRaisesRegex(ValueError, "identity or SHA256 changed"):
            runner.main(self.argv + ["--resume"])
        self.assertEqual(state_path.read_bytes(), original)
        self.assertEqual(len(self.commands), commands)

    def test_missing_child_after_saved_evidence_never_falls_back_to_parent(self):
        self.fresh()
        self.output.unlink()
        count = len(self.train_commands())
        with self.assertRaisesRegex(RuntimeError, "full SFT checkpoint is missing"):
            runner.main(self.argv + ["--resume"])
        self.assertEqual(len(self.train_commands()), count)

    def test_retry_before_first_child_checkpoint_replays_only_the_pinned_full_parent(self):
        self.fail_train = True
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            self.fresh()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(runner.main(self.argv + ["--resume"]), 0)
        self.assertEqual(self.state()["resume_action"], "continuation_retry_no_checkpoint")
        self.assertIn("--continue-from", self.train_commands()[-1])
        self.assertNotIn("--init-from", self.train_commands()[-1])

    def test_cli_continuation_is_fresh_only_and_parent_must_match_baseline_path(self):
        for extra in (["--resume"], ["--resume", "--resume-from-revision", "old"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(self.argv + ["--continue-from", str(self.source), *extra])
        other = self.root / "other.pt"
        other.write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "same parent checkpoint"):
            runner.main(self.argv + ["--continue-from", str(other)])
        self.assertFalse(self.commands)

    def test_source_stage_full_state_scope_data_metric_batch_schedule_and_target_are_checked(self):
        original = json.loads(json.dumps(self.parent))
        changes = [
            ("has_optimizer", False), ("has_training_state", False),
            ("metrics", {**original["metrics"], "training_stage": "meta"}),
            ("metrics", {**original["metrics"], "parameter_scope": "memory"}),
            ("metrics", {**original["metrics"], "dataset_fingerprint": "other-data"}),
            ("metrics", {**original["metrics"], "validation_selection_metric": "validation_ce"}),
            ("metrics", {**original["metrics"], "run_id": "child"}),
            ("metrics", {**original["metrics"], "code_revision": "unknown"}),
            ("step", 6), ("training_state_summary", {**original["training_state_summary"], "micro_batch_size": 2}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                self.parent = json.loads(json.dumps(original))
                self.parent[key] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    self.fresh()
                self.assertFalse(self.commands)
        self.parent = original
        for key, value in (("lr_schedule_start_step", 0), ("lr_schedule_start_step", "2"),
                           ("micro_batch_size", 2), ("grad_accum_steps", 8), ("seed", 4),
                           ("validation_fraction", 0)):
            with self.subTest(config=key, value=value):
                before = self.settings["training"][key]
                self.settings["training"][key] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    self.fresh()
                self.settings["training"][key] = before
                self.assertFalse(self.commands)

    def test_old_non_continuation_binding_and_command_remain_unchanged(self):
        self.assertEqual(runner.main(self.argv), 0)
        self.assertNotIn("continuation", self.state()["binding"])
        command = self.train_commands()[0]
        self.assertIn("--init-from", command)
        self.assertIn("--no-resume", command)
        self.assertNotIn("--continue-from", command)


if __name__ == "__main__":
    unittest.main()

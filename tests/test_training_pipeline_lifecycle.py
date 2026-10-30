"""Requested pipeline stops exit cleanly without hiding unrelated failures."""
import contextlib
import io
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deploy import run_training_pipeline as pipeline


class FakeProcess:
    pid = 12345

    def __init__(self, lines=(), code=0):
        self.stdout = iter(lines)
        self.code = code
        self.sent_signals = []

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def poll(self):
        return None

    def send_signal(self, signum):
        self.sent_signals.append(signum)

    def wait(self):
        return self.code


class TrainingPipelineLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        for path in ("deploy", "checkpoints", "data/pretrain_test/tokenized", "data/instruction_test"):
            (self.root / path).mkdir(parents=True)
        (self.root / "data/pretrain_test/tokenized/manifest.json").touch()
        (self.root / "data/instruction_test/train.jsonl").touch()
        (self.root / "checkpoints/uttt_1.584b_test_pretrain.best.pt").touch()
        self.handlers = {}
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(pipeline, "__file__", str(self.root / "deploy/run_training_pipeline.py")))
        self.stack.enter_context(patch.object(pipeline.os, "chdir"))
        self.stack.enter_context(patch.object(sys, "argv", ["run_training_pipeline.py", "--run-id", "test",
                                                           "--pretrain-steps", "10", "--sft-steps", "2"]))
        # The runner targets Linux; stub only its OS registration and lock on
        # Windows so the same lifecycle tests also run on development hosts.
        self.stack.enter_context(patch.object(pipeline.signal, "SIGHUP", getattr(signal, "SIGHUP", 1), create=True))
        self.stack.enter_context(patch.object(pipeline.signal, "signal", side_effect=self.handlers.__setitem__))
        self.stack.enter_context(patch.dict(sys.modules, {"fcntl": SimpleNamespace(
            LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)}))
        self.popen = self.stack.enter_context(patch.object(pipeline.subprocess, "Popen"))
        self.metrics = self.stack.enter_context(patch.object(pipeline, "checkpoint_metrics", return_value={
            "step": 10, "metrics": {"training_stage": "pretrain", "training_pipeline": 3}, "path": "saved.pt"}))

    def state(self):
        return json.loads((self.root / "logs/pipeline_test.json").read_text(encoding="utf-8"))

    def stop(self, signum=signal.SIGTERM):
        self.handlers[signum](signum, None)

    def test_requested_term_exits_cleanly_and_does_not_launch_sft(self):
        def lines():
            yield "training step complete\n"
            self.stop()

        process = FakeProcess(lines=lines())
        self.popen.return_value = process
        self.assertIsNone(pipeline.main())
        self.assertEqual(process.sent_signals, [signal.SIGTERM])
        self.popen.assert_called_once()
        self.metrics.assert_not_called()
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertEqual(self.state()["stage"], "pretrain")
        self.assertIn("[Pipeline] Interrupted:", self.output.getvalue())
        self.assertNotIn("TRAINING_PIPELINE_COMPLETE", self.output.getvalue())

    def test_interrupted_child_exit_also_exits_cleanly(self):
        self.popen.return_value = FakeProcess(code=-signal.SIGTERM)
        self.assertIsNone(pipeline.main())
        self.popen.assert_called_once()
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertIn("next stage was not started", self.state()["error"])

    def test_stop_during_checkpoint_verification_does_not_launch_sft(self):
        self.popen.return_value = FakeProcess()

        def verify(path):
            self.stop()
            return {"step": 10, "metrics": {"training_stage": "pretrain", "training_pipeline": 3}}

        self.metrics.side_effect = verify
        self.assertIsNone(pipeline.main())
        self.popen.assert_called_once()
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertIn("checking the checkpoint", self.state()["error"])

    def test_unexpected_interrupted_error_is_a_failure_and_propagates(self):
        self.popen.side_effect = InterruptedError("unexpected OS interruption")
        with self.assertRaisesRegex(InterruptedError, "unexpected OS interruption"):
            pipeline.main()
        self.assertEqual(self.state()["status"], "failed")
        self.assertNotIn("[Pipeline] Interrupted:", self.output.getvalue())

    def test_other_error_during_requested_stop_is_not_hidden(self):
        def lines():
            self.stop()
            raise OSError("stage log read failed")
            yield

        self.popen.return_value = FakeProcess(lines=lines())
        with self.assertRaisesRegex(OSError, "stage log read failed"):
            pipeline.main()
        self.popen.assert_called_once()
        self.assertEqual(self.state()["status"], "failed")

    def test_incomplete_checkpoint_remains_a_failure(self):
        self.popen.return_value = FakeProcess()
        self.metrics.return_value["step"] = 9
        with self.assertRaisesRegex(RuntimeError, "before target 10"):
            pipeline.main()
        self.popen.assert_called_once()
        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.state()["saved_step"], 9)


if __name__ == "__main__":
    unittest.main()

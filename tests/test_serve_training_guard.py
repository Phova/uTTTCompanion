"""Serving must refuse live trainers without stopping or restarting anything."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from deploy import serve_guard as guard


ROOT = Path(__file__).resolve().parents[1]


class ServingTrainingGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "runs" / "frozen_source"
        self.run.mkdir(parents=True)
        self.records = self.root / "logs" / ".wsl_owners"
        self.records.mkdir(parents=True)

    @staticmethod
    def identity(pid, start=100, command=None):
        return {"pid": pid, "start_time": start, "uid": 0, "boot_id": "test-boot",
                "cmdline": command or ["python3", "train_5090.py"]}

    def record(self, identity, *, phase="running"):
        record = {"workspace": str(self.root), "session": "independent_owner", "phase": phase,
                  "command": ["/venv/python", "-u", str(self.run / "deploy" / "run_recovery_training.py")],
                  "owner": identity}
        (self.records / "independent_owner.json").write_text(json.dumps(record), encoding="utf-8")

    def test_live_frozen_source_pipeline_and_trainer_block_without_tmux_or_pid_files(self):
        table = {
            10: {"parent": 1, "cwd": self.run, "start_time": 100, "scripts": {"run_recovery_training.py"}},
            11: {"parent": 10, "cwd": self.run, "start_time": 100, "scripts": {"train_5090.py"}},
            12: {"parent": 11, "cwd": self.run, "start_time": 100, "scripts": {"train_5090.py"}},
        }
        identities = {10: self.identity(10, command=["python3", "run_recovery_training.py"]),
                      11: self.identity(11), 12: self.identity(12)}
        result = guard.training_conflicts(self.run, table=table, identity=identities.get)
        self.assertEqual(result, ["training PID 10", "training PID 11"])

    def test_verified_live_wsl_owner_blocks_before_trainer_exists(self):
        expected = self.identity(42, command=["python3", "wsl_training_owner.py"])
        self.record(expected, phase="starting")
        result = guard.training_conflicts(self.run / "deploy", table={}, identity={42: expected}.get)
        self.assertEqual(len(result), 1)
        self.assertIn("WSL training owner independent_owner (PID 42)", result[0])

    def test_dead_reused_or_previous_boot_owner_records_do_not_block(self):
        expected = self.identity(42, command=["python3", "wsl_training_owner.py"])
        self.record(expected)
        for current in (None, {**expected, "start_time": 101}, {**expected, "boot_id": "previous-boot"},
                        {**expected, "cmdline": ["python3", "unrelated.py"]}):
            with self.subTest(current=current):
                self.assertEqual(guard.training_conflicts(self.root, table={}, identity=lambda _: current), [])

    def test_completed_owner_does_not_block_even_before_owner_process_exits(self):
        expected = self.identity(42, command=["python3", "wsl_training_owner.py"])
        self.record(expected, phase="exited")
        (self.records / "broken.json").write_text("{", encoding="utf-8")
        self.assertEqual(guard.training_conflicts(self.root, table={}, identity={42: expected}.get), [])

    def test_native_pid_exit_or_reuse_and_unrelated_workspace_do_not_block(self):
        table = {11: {"parent": 1, "cwd": self.run, "start_time": 100, "scripts": {"train_5090.py"}},
                 20: {"parent": 1, "cwd": self.root.parent / "other_project", "start_time": 100,
                      "scripts": {"train_5090.py"}}}
        for current in (None, self.identity(11, start=101)):
            with self.subTest(current=current):
                self.assertEqual(guard.training_conflicts(self.root, table=table, identity=lambda _: current), [])

    def test_assertion_and_cli_refuse_without_signalling_processes(self):
        with mock.patch.object(guard.sys, "platform", "linux"), \
                mock.patch.object(guard, "training_conflicts", return_value=["training PID 42"]), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaisesRegex(guard.TrainingActiveError, "Wait for training to finish"):
                guard.assert_serving_available(self.run)
            self.assertEqual(guard.main(["--workspace", str(self.root)]), 2)
        self.assertIn("Refusing inference startup", errors.getvalue())

    def test_native_windows_keeps_existing_startup_behavior(self):
        with mock.patch.object(guard.sys, "platform", "win32"), mock.patch.object(guard, "training_conflicts") as scan:
            guard.assert_serving_available(self.root)
        scan.assert_not_called()

    def test_direct_server_refuses_before_configuration_or_checkpoint_loading(self):
        # Run the real entry function in isolation: no torch import or model load.
        source = ast.parse((ROOT / "server" / "app.py").read_text(encoding="utf-8"))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "get_personalizer")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 function], type_ignores=[])
        blocker = mock.Mock(side_effect=guard.TrainingActiveError("training remains active"))
        checkpoint = mock.Mock()
        namespace = {"_personalizer": None, "assert_serving_available": blocker,
                     "DualTrackCheckpointManager": checkpoint}
        exec(compile(ast.fix_missing_locations(module), "server/app.py", "exec"), namespace)
        with self.assertRaisesRegex(guard.TrainingActiveError, "training remains active"):
            namespace["get_personalizer"]()
        blocker.assert_called_once_with()
        checkpoint.assert_not_called()

    def test_missing_service_checkpoint_is_never_cold_initialized(self):
        source = ast.parse((ROOT / "server" / "app.py").read_text(encoding="utf-8"))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == "get_personalizer")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 function], type_ignores=[])
        missing = self.root / "misspelled_model.pt"

        def load_checkpoint(**kwargs):
            if kwargs["auto_create_base_if_missing"]:
                missing.write_bytes(b"untrained replacement")
                self.fail("Serving allowed a random replacement checkpoint")
            raise FileNotFoundError(f"Base checkpoint not found at {kwargs['base_checkpoint_path']}")

        checkpoint = mock.Mock(side_effect=load_checkpoint)
        namespace = {"_personalizer": None, "assert_serving_available": mock.Mock(), "Path": Path,
                     "os": os, "DualTrackCheckpointManager": checkpoint}
        exec(compile(ast.fix_missing_locations(module), "server/app.py", "exec"), namespace)
        with mock.patch.dict(os.environ, {"UTTT_CONFIG": str(self.root / "missing_config.yaml"),
                                          "UTTT_BASE_CKPT": str(missing)}), \
                self.assertRaisesRegex(FileNotFoundError, "Base checkpoint not found"):
            namespace["get_personalizer"]()
        self.assertFalse(missing.exists())
        checkpoint.assert_called_once()
        self.assertEqual(checkpoint.call_args.kwargs["base_checkpoint_path"], str(missing))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux/WSL shell startup")
    def test_shell_refusal_preserves_existing_service_and_logs(self):
        source = (ROOT / "deploy" / "remote_ctl.sh").read_text(encoding="utf-8")
        body = source.split("<<'REMOTE_SERVE'\n", 1)[1].split("\nREMOTE_SERVE", 1)[0]
        executable = self.root / ".venv" / "bin" / "python3"
        executable.parent.mkdir(parents=True)
        calls = self.root / "python_calls"
        executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> " + shlex.quote(str(calls))
                              + "\ncase \"$1\" in deploy/serve_guard.py) exit 2;; *) exit 0;; esac\n")
        executable.chmod(0o700)
        log = self.root / "logs" / "serve_5090.log"
        log.write_text("existing service log\n")
        result = subprocess.run(["bash", "-s", "--", str(self.root), "uttt_serve", "0", "unused", "8000"],
                                input=body, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(calls.read_text().splitlines(), [f"deploy/serve_guard.py --workspace {self.root}"])
        self.assertEqual(log.read_text(), "existing service log\n")
        self.assertNotIn("Started", result.stdout)


if __name__ == "__main__":
    unittest.main()

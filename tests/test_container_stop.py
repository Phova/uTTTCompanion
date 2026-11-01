"""Docker training stop is independent of the host CLI's process ancestry."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy import stop_training as stopping


class ContainerStopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.container_id = "c" * 64

    def container(self, *, running=True, source=None):
        return {"Id": self.container_id, "State": {"Running": running},
                "Mounts": [{"Type": "bind", "Source": str(source or self.root / "checkpoints"),
                            "Destination": "/workspace/uTTT/checkpoints"}]}

    @staticmethod
    def inspected(record):
        return SimpleNamespace(returncode=0, stdout=json.dumps([record]), stderr="")

    def test_cli_docker_stop_does_not_require_native_training_descendants(self):
        table = {10: {"cwd": self.root, "scripts": {"bash", "docker"}, "parent": 1}}
        with mock.patch.object(sys, "argv",
                               ["stop_training.py", "--workspace", str(self.root),
                                "--session", "uttt_train", "--container", "uttt_train"]), \
                mock.patch.object(stopping.shutil, "which", return_value="tmux"), \
                mock.patch.object(stopping.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="10\n")), \
                mock.patch.object(stopping, "process_table", return_value=table), \
                mock.patch.object(stopping, "stop") as native, \
                mock.patch.object(stopping, "stop_container") as container, \
                contextlib.redirect_stdout(io.StringIO()):
            stopping.main()
        native.assert_not_called()
        container.assert_called_once_with("uttt_train", 300, self.root)

    def test_stop_pins_id_before_signalling_and_waiting(self):
        with mock.patch.object(stopping.subprocess, "run", side_effect=[
            self.inspected(self.container()), SimpleNamespace(returncode=0),
            self.inspected(self.container(running=False)),
        ]) as run:
            stopping.stop_container("uttt_train", 300, self.root)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[1], ["docker", "kill", "--signal", "TERM", self.container_id])
        self.assertEqual(commands[2][-1], self.container_id)

    def test_foreign_mount_is_refused_before_any_signal(self):
        foreign = self.root / "other-project" / "checkpoints"
        with mock.patch.object(stopping.subprocess, "run", return_value=self.inspected(self.container(source=foreign))) as run:
            with self.assertRaisesRegex(RuntimeError, "another workspace"):
                stopping.stop_container("uttt_train", 300, self.root)
        self.assertEqual(run.call_count, 1)

    def test_missing_container_is_idempotent_but_daemon_failure_is_not_success(self):
        for message, missing in (("Error: No such container: uttt_train", True),
                                 ("connect /var/run/docker.sock: no such file or directory", False),
                                 ("permission denied while connecting to Docker", False)):
            with self.subTest(message=message), mock.patch.object(stopping.subprocess, "run", return_value=
                    SimpleNamespace(returncode=1, stdout="", stderr=message)) as run:
                if missing:
                    stopping.stop_container("uttt_train", 300, self.root)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Cannot inspect"):
                        stopping.stop_container("uttt_train", 300, self.root)
                self.assertEqual(run.call_count, 1)

    def test_timeout_preserves_saving_container_instead_of_force_kill(self):
        with mock.patch.object(stopping.subprocess, "run", side_effect=[
                self.inspected(self.container()), SimpleNamespace(returncode=0), self.inspected(self.container())]) as run, \
                mock.patch.object(stopping.time, "monotonic", side_effect=[0, 301]):
            with self.assertRaisesRegex(RuntimeError, "still stopping"):
                stopping.stop_container("uttt_train", 300, self.root)
        self.assertEqual([call.args[0][1] for call in run.call_args_list], ["inspect", "kill", "inspect"])
        self.assertEqual(run.call_args_list[1].args[0][3], "TERM")

    @unittest.skipUnless(os.name == "posix", "Linux checkpoint symlink")
    def test_snapshot_checkpoint_symlink_matches_the_actual_bind_source(self):
        physical = self.root / "shared" / "checkpoints"
        physical.mkdir(parents=True)
        snapshot = self.root / "runs" / "immutable"
        snapshot.mkdir(parents=True)
        (snapshot / "checkpoints").symlink_to(physical, target_is_directory=True)
        with mock.patch.object(stopping.subprocess, "run", side_effect=[
            self.inspected(self.container(source=physical)), SimpleNamespace(returncode=0),
            self.inspected(self.container(running=False, source=physical)),
        ]):
            stopping.stop_container("uttt_train", 300, snapshot)


if __name__ == "__main__":
    unittest.main()

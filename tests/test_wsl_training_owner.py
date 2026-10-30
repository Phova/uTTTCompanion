"""Foreground WSL owner protocol; no GPU, training, WMI or live tmux required."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy import wsl_training_owner as owner


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.run = self.root / "snapshot with spaces"
        self.run.mkdir()
        self.uid = os.getuid() if hasattr(os, "getuid") else 0
        self.identities = {pid: {"pid": pid, "start_time": pid * 10, "uid": self.uid,
                                 "boot_id": "boot", "cmdline": ["python", "owner"]}
                           for pid in (os.getpid(), 42)}
        self.calls = []

    def tearDown(self):
        self.directory.cleanup()

    def fake_tmux(self, *args, check=True):
        self.calls.append(args)
        if args[0] == "has-session":
            return SimpleNamespace(returncode=1, stdout="")
        if args[0] == "new-session":
            return SimpleNamespace(returncode=0, stdout="$3 %4 42\n")
        if args[0] == "show-options":
            return SimpleNamespace(returncode=0, stdout="token\n")
        if args[0] == "list-panes":
            return SimpleNamespace(returncode=0, stdout="42\n")
        return SimpleNamespace(returncode=0, stdout="")

    def finishing_waiter(self, *_, **__):
        finished = False
        def wait(*_, **__):
            nonlocal finished
            record_path = self.root / "logs" / ".wsl_owners" / "uttt_train.json"
            record = json.loads(record_path.read_text())
            result = record_path.with_name(f"uttt_train.{record['token']}.result.json")
            owner.atomic_json(result, {"token": record["token"], "exit_code": 7})
            finished = True
            return 0
        return SimpleNamespace(wait=wait, poll=lambda: 0 if finished else None)

    def launch(self, **patches):
        command = [str(Path(subprocess.__file__).resolve()), "hello world", "literal; $(no_command)"]
        with mock.patch.object(owner, "OwnerLock", return_value=nullcontext()), \
                mock.patch.object(owner, "process_identity", side_effect=self.identities.get), \
                mock.patch.object(owner, "tmux", side_effect=patches.get("tmux", self.fake_tmux)), \
                mock.patch.object(owner.subprocess, "Popen", side_effect=patches.get("waiter", self.finishing_waiter)):
            return owner.run_owner(self.root, "uttt_train", self.run,
                                   self.root / "logs" / "train.log", command)

    def test_ready_start_finish_ack_uses_real_pane_and_returns_training_status(self):
        self.assertEqual(self.launch(), 7)
        record = json.loads((self.root / "logs" / ".wsl_owners" / "uttt_train.json").read_text())
        self.assertEqual(record["pane"]["pid"], 42)
        self.assertEqual(record["session_id"], "$3")
        self.assertEqual(record["phase"], "exited")
        waits = [args for args in self.calls if args[0] == "wait-for"]
        self.assertTrue(waits[0][-1].endswith("-ready"))
        self.assertTrue(waits[1][-1].endswith("-start"))
        self.assertTrue(waits[2][-1].endswith("-ack"))
        self.assertFalse(any(args[0] in ("kill-pane", "kill-session") for args in self.calls))

    def test_startup_failure_logs_trace_and_removes_only_its_verified_pane(self):
        def failing(*args, **kwargs):
            if args[0] == "set-option":
                raise subprocess.CalledProcessError(1, ["tmux", *args])
            return self.fake_tmux(*args, **kwargs)
        with self.assertRaises(subprocess.CalledProcessError):
            self.launch(tmux=failing)
        log = self.root / "logs" / ".wsl_owners" / "uttt_train.owner.log"
        self.assertIn("CalledProcessError", log.read_text())
        self.assertIn(("kill-pane", "-t", "%4"), self.calls)

    def test_existing_session_is_rejected_without_modifying_it(self):
        with mock.patch.object(owner, "OwnerLock", return_value=nullcontext()), \
                mock.patch.object(owner, "tmux", return_value=SimpleNamespace(returncode=0)), \
                mock.patch.object(owner.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                owner.run_owner(self.root, "uttt_train", self.run, self.root / "train.log", [str(self.run / "true")])
            launch.assert_not_called()

    def test_signal_before_barrier_cancels_without_starting_training(self):
        stopped = mock.Mock()
        def waiter(*args, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return SimpleNamespace(poll=lambda: None, terminate=stopped, wait=lambda **_: 0)
        self.assertEqual(self.launch(waiter=waiter), 143)
        self.assertFalse(any(args[0] == "wait-for" and args[-1].endswith("-start") for args in self.calls))
        self.assertIn(("kill-pane", "-t", "%4"), self.calls)
        stopped.assert_called_once()

    def test_signal_during_release_remains_pending_until_primary_exists(self):
        def releasing(*args, **kwargs):
            result = self.fake_tmux(*args, **kwargs)
            if args[0] == "wait-for" and args[-1].endswith("-start"):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return result
        def waiter(*args, **kwargs):
            child = self.finishing_waiter()
            finish = child.wait
            child.wait = mock.Mock(side_effect=[subprocess.TimeoutExpired("tmux", .2), None])
            def wait(**kw):
                value = child.wait(**kw)
                return finish() if value is None else value
            return SimpleNamespace(poll=child.poll, wait=wait)
        with mock.patch.object(owner, "stop_owned_session", side_effect=[RuntimeError("No owned primary process"), None]) as stop:
            self.assertEqual(self.launch(tmux=releasing, waiter=waiter), 7)
            self.assertEqual(stop.call_count, 2)

    def test_wrapper_quotes_arguments_and_has_no_idle_sleep(self):
        wrapper = owner.build_wrapper(["/usr/bin/python3", "space argument", "; $(literal)"], self.run,
                                      self.root / "train.log", self.root / "result.json", "token",
                                      ("ready", "start", "finished", "ack"))
        self.assertIn("'space argument' '; $(literal)'", wrapper)
        self.assertIn("trap owner_finished EXIT", wrapper)
        self.assertIn("tmux wait-for ack", wrapper)
        self.assertNotIn("sleep", wrapper)

    def record(self):
        return {"workspace": str(self.root), "session": "uttt_train", "session_id": "$3", "pane_id": "%4",
                "token": "token", "owner": self.identities[os.getpid()], "pane": self.identities[42]}

    def test_safe_stop_delegates_only_after_identity_token_pane_and_lock_checks(self):
        _, lock_path, record_path = owner.owner_paths(self.root, "uttt_train")
        owner.atomic_json(record_path, self.record())
        with mock.patch.object(owner, "OwnerLock", side_effect=owner.OwnerBusyError("busy")):
            calls = []
            owner.stop_owned_session(self.root, "uttt_train", 300, identity=self.identities.get,
                                     tmux_call=self.fake_tmux, runner=lambda *args, **kwargs: calls.append(args))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ([42], self.root, 300))
        self.assertIn(("list-panes", "-t", "$3:", "-F", "#{pane_pid}"), self.calls)

    def test_stop_refuses_pid_reuse_token_change_or_workspace_mismatch(self):
        original = self.record()
        for field in ("owner", "pane", "token", "workspace"):
            changed = copy.deepcopy(original)
            if field in ("owner", "pane"):
                changed[field]["start_time"] += 1
            else:
                changed[field] = "different"
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                owner.validate_owned_session(changed, self.root, "uttt_train",
                                             identity=self.identities.get, tmux_call=self.fake_tmux)

    @unittest.skipIf(owner.fcntl is None, "Linux flock")
    def test_lock_refuses_duplicate_owner_and_releases_on_exit(self):
        path = self.root / "owner.lock"
        with owner.OwnerLock(path):
            with self.assertRaises(owner.OwnerBusyError):
                with owner.OwnerLock(path):
                    self.fail("duplicate lock accepted")
        with owner.OwnerLock(path):
            pass


if __name__ == "__main__":
    unittest.main()

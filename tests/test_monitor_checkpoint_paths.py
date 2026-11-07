"""The dashboard follows an owner's explicit output, including another drive."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy.monitor_status import Monitor, checkpoint_status, owner_checkpoint_output


class MonitorCheckpointPathTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.base = Path(self.directory.name)
        self.root = self.base / "workspace"
        self.root.mkdir()
        self.owner_path = self.root / "logs" / ".wsl_owners" / "uttt_train.json"
        self.owner_path.parent.mkdir(parents=True)
        self.legacy = self.root / "checkpoints" / "old.pt"
        self.legacy.parent.mkdir()
        self.legacy.write_bytes(b"old model")

    def tearDown(self):
        self.directory.cleanup()

    def owner(self, command, **changes):
        record = {"workspace": str(self.root), "session": "uttt_train", "command": command,
                  "token": "owner-secret-must-not-appear", "phase": "starting"}
        record.update(changes)
        self.owner_path.write_text(json.dumps(record), encoding="utf-8")

    def test_absolute_external_output_wins_even_when_legacy_or_best_is_newer(self):
        output = self.base / "other_drive" / "current.pt"
        output.parent.mkdir()
        output.write_bytes(b"new model")
        output.with_suffix(".best.pt").write_bytes(b"best")
        os.utime(output, (100, 100))
        self.owner(["/python", "train.py", "--output-ckpt", str(output)])
        description = checkpoint_status(self.root)
        self.assertTrue(description.startswith("current.pt ("))
        self.assertIn("GiB updated", description)
        self.assertNotIn("old.pt", description)
        self.assertNotIn("owner-secret", description)

    def test_monitor_metadata_uses_explicit_output_without_printing_owner_record(self):
        output = self.base / "external.pt"
        output.write_bytes(b"model")
        self.owner(["/python", "runner.py", "--output-ckpt=" + str(output)])
        stream = io.StringIO()
        monitor = Monitor(self.root, 200, stream)
        try:
            with mock.patch("deploy.monitor_status.subprocess.run", return_value=SimpleNamespace(stdout="uttt_train\n")):
                monitor.refresh_metadata()
            monitor.render(10)
            self.assertIn("external.pt (", stream.getvalue())
            self.assertNotIn("owner-secret", stream.getvalue())
        finally:
            monitor.close()

    def test_missing_explicit_current_waits_instead_of_showing_old_or_backup(self):
        output = self.base / "not_saved.pt"
        output.with_suffix(".best.pt").write_bytes(b"weights")
        Path(str(output) + ".prev").write_bytes(b"previous")
        self.owner(["/python", "runner.py", "--output-ckpt", str(output)])
        self.assertEqual(checkpoint_status(self.root), "not_saved.pt (waiting for first save)")

    def test_no_owner_or_no_explicit_output_keeps_legacy_fallback(self):
        self.assertIn("old.pt (", checkpoint_status(self.root))
        self.owner(["/python", "runner.py", "--config", "train.yaml"])
        self.assertIn("old.pt (", checkpoint_status(self.root))
        self.legacy.unlink()
        self.assertEqual(checkpoint_status(self.root), "none")

    def test_relative_output_uses_workspace_or_recorded_run_directory(self):
        for working in (None, str(self.root / "runs" / "current"), "runs/current"):
            with self.subTest(working=working):
                base = self.root if working is None else self.root / "runs" / "current"
                output = base / "results" / "relative.pt"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"weights")
                changes = {} if working is None else {"run_directory": working}
                self.owner(["/python", "runner.py", "--output-ckpt", "results/relative.pt"], **changes)
                explicit, path = owner_checkpoint_output(self.root)
                self.assertTrue(explicit)
                self.assertEqual(path, output)
                self.assertIn("relative.pt (", checkpoint_status(self.root))

    def test_last_output_option_matches_argument_parser_precedence(self):
        output = self.base / "last.pt"
        output.write_bytes(b"last")
        self.owner(["/python", "--output-ckpt", str(self.legacy), "--output-ckpt=" + str(output)])
        self.assertTrue(checkpoint_status(self.root).startswith("last.pt ("))

    def test_invalid_explicit_paths_do_not_select_another_run(self):
        for command in (["/python", "--output-ckpt"], ["/python", "--output-ckpt="],
                        ["/python", "--output-ckpt", "--config"],
                        ["/python", "--output-ckpt", "a\0b"]):
            with self.subTest(command=command):
                self.owner(command)
                self.assertIn("invalid output path", checkpoint_status(self.root))
                self.assertNotIn("old.pt", checkpoint_status(self.root))

    def test_malformed_oversized_and_foreign_owner_records_are_bounded_and_stable(self):
        records = [b"{invalid", b"[]", b'"string"', b"\xff", b"[" * 4000,
                   b" " * 65537,
                   json.dumps({"workspace": str(self.root), "session": "uttt_train", "command": "not argv"}).encode(),
                   json.dumps({"workspace": str(self.base / "foreign"), "session": "uttt_train",
                               "command": ["--output-ckpt", "foreign.pt"]}).encode()]
        for contents in records:
            with self.subTest(size=len(contents)):
                self.owner_path.write_bytes(contents)
                self.assertIn("old.pt (", checkpoint_status(self.root))
        self.owner(["/python", "--output-ckpt", "bad.pt"], session="other_train")
        self.assertIn("old.pt (", checkpoint_status(self.root))

    def test_non_regular_explicit_output_waits_and_control_chars_are_not_rendered(self):
        directory = self.base / "directory.pt"
        directory.mkdir()
        self.owner(["/python", "--output-ckpt", str(directory)])
        self.assertEqual(checkpoint_status(self.root), "directory.pt (waiting for first save)")
        self.owner(["/python", "--output-ckpt", str(self.base / "unsafe\x1bname.pt")])
        self.assertNotIn("\x1b", checkpoint_status(self.root))


if __name__ == "__main__":
    unittest.main()

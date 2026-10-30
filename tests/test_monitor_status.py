"""Dashboard refresh labels and session-scoped disconnect cleanup."""
import argparse
import io
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from deploy.monitor_status import Monitor, compact_training_line, latest_training_line, positive_interval, target_hz


class MonitorDisplayTests(unittest.TestCase):
    def test_non_pty_output_uses_forwarded_panel_width_without_wrapping(self):
        output = io.StringIO()
        monitor = Monitor('.', 10, output, columns=80)
        try:
            monitor.render(10)
            clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', output.getvalue())
            self.assertTrue(all(len(line.strip('\r')) < 80 for line in clean.splitlines()))
            self.assertIn('target 100Hz / actual', clean)
        finally:
            monitor.close()

    def test_refresh_label_follows_interval_and_reports_measured_rate(self):
        self.assertEqual(target_hz(10), "100")
        self.assertEqual(target_hz(200), "5")
        self.assertEqual(target_hz(300), "3.33333")
        output = io.StringIO()
        monitor = Monitor(".", 10, output)
        try:
            monitor.render(10)
            monitor.render(10.02)
            self.assertIn("target 100Hz / actual 50.0Hz", output.getvalue())
        finally:
            monitor.close()

    def test_invalid_interval_rejected_instead_of_busy_loop(self):
        for value in ("0", "-10", "hello", "60001"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_interval(value)
        self.assertEqual(positive_interval("10"), 10)

    def test_log_selects_latest_progress_after_multiline_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("[Step 1/10] CE Loss: 3.5\nSample: hello\nlong sample\n", encoding="utf-8")
            self.assertEqual(latest_training_line(path), "[Step 1/10] CE Loss: 3.5")

    def test_compact_training_keeps_rate_and_throughput_with_disabled_ttt(self):
        line = ("[Step 15070/100000] Stage: pretrain | Total Loss: 3.4728 | CE Loss: 3.4728 | "
                "Surprise Loss: disabled | GradNorm: 0.606 | LR: 1.90e-04 | Speed: 7,300 tok/s | Peak VRAM: 25 GB")
        compact = compact_training_line(line)
        self.assertEqual(compact, "15070/100000 pretrain CE3.4728 LR1.90e-04 7,300 tok/s")
        self.assertNotIn("Surprise", compact)
        self.assertEqual(compact_training_line("waiting for log..."), "waiting for log...")


@unittest.skipUnless(os.name == "posix" and Path("/proc").exists(), "Linux process lifecycle regression")
class MonitorCleanupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake_smi = self.bin / "nvidia-smi"
        fake_smi.write_text(f"#!{sys.executable}\n" + '''import os, signal, time
from pathlib import Path
child = os.fork()
if child == 0:
    while True:
        time.sleep(1)
def finish(*_):
    try:
        os.kill(child, signal.SIGTERM)
        os.waitpid(child, 0)
    except ProcessLookupError:
        pass
    raise SystemExit(0)
signal.signal(signal.SIGTERM, finish)
Path(os.environ['MONITOR_TEST_PIDS']).write_text(str(os.getpid()) + ' ' + str(child))
while True:
    print('90, 30, 1000, 32000, 500, 2900, 60', flush=True)
    time.sleep(.01)
''', encoding="utf-8")
        fake_smi.chmod(0o755)
        self.processes = []
        self.pty_masters = {}

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        for master in self.pty_masters.values():
            os.close(master)
        self.directory.cleanup()

    def start_monitor(self, name, *, pty_output=False):
        pid_file = self.root / (name + ".pids")
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   MONITOR_TEST_PIDS=str(pid_file))
        script = Path(__file__).resolve().parents[1] / "deploy" / "monitor_status.py"
        arguments = [sys.executable, "-u", str(script), "--root", str(self.root), "--interval-ms", "10"]
        if pty_output:
            import pty
            master, slave = pty.openpty()
            try:
                process = subprocess.Popen(arguments, stdin=slave, stdout=slave, stderr=slave, env=env)
            finally:
                os.close(slave)
            self.pty_masters[process.pid] = master
        else:
            process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.processes.append(process)
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.01)
        self.assertTrue(pid_file.exists(), "GPU collector did not start")
        pids = [int(item) for item in pid_file.read_text().split()]
        return process, pids

    def assert_stopped(self, process, pids):
        process.wait(timeout=3)
        deadline = time.monotonic() + 2
        while any(self.is_running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(any(self.is_running(pid) for pid in pids), f"collector leaked: {pids}")

    @staticmethod
    def is_running(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
        except FileNotFoundError:
            return False

    def test_hup_and_term_clean_collector_descendants(self):
        for sig in (signal.SIGHUP, signal.SIGTERM):
            with self.subTest(sig=sig):
                process, pids = self.start_monitor(str(sig))
                process.send_signal(sig)
                self.assert_stopped(process, pids)

    def test_broken_ssh_output_cleans_collector_descendants(self):
        process, pids = self.start_monitor("output")
        process.stdout.close()
        self.assert_stopped(process, pids)

    def test_closed_ssh_pty_cleans_collector_descendants(self):
        process, pids = self.start_monitor("pty", pty_output=True)
        os.close(self.pty_masters.pop(process.pid))
        self.assert_stopped(process, pids)

    def test_closing_one_monitor_preserves_other_session(self):
        first, first_pids = self.start_monitor("first")
        second, second_pids = self.start_monitor("second")
        first.terminate()
        self.assert_stopped(first, first_pids)
        self.assertIsNone(second.poll())
        self.assertTrue(all(self.is_running(pid) for pid in second_pids))
        second.terminate()
        self.assert_stopped(second, second_pids)


if __name__ == "__main__":
    unittest.main()

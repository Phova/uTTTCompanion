"""Exercise the real shell viewer commands through an isolated SSH/PTY stub."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(os.name == "posix" and Path("/proc").exists(), "Linux SSH/PTY integration")
class RemoteViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "workspace with spaces"
        (self.root / "logs").mkdir(parents=True)
        (self.root / "logs" / "train_5090.log").write_text("training continues\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.processes = []
        self.records = []
        self.script = Path(__file__).resolve().parents[1] / "deploy" / "remote_ctl.sh"
        self.write_executable("ssh", '''import errno, json, os, pty, select, signal, sys, time
from pathlib import Path
assert '-tt' in sys.argv, sys.argv
assert sys.stdin.read() == '', 'viewer SSH must receive EOF without ending its PTY'
pid, master = pty.fork()
if pid == 0:
    os.execvp('bash', ['bash', '-c', sys.argv[-1]])
record = Path(os.environ['VIEW_RECORD'])
temporary = record.with_suffix('.tmp')
temporary.write_text(json.dumps({'ssh_pid': os.getpid(), 'remote_pid': pid, 'args': sys.argv[1:]}))
temporary.replace(record)
stopping = False
def stop(*_):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGHUP, stop)
try:
    while not stopping:
        ready, _, _ = select.select([master], [], [], .05)
        if ready:
            try:
                data = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
finally:
    os.close(master)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if os.waitpid(pid, os.WNOHANG)[0]:
            break
        time.sleep(.01)
    else:
        raise RuntimeError('remote viewer survived PTY close')
''')
        self.write_executable("nvidia-smi", '''import os, signal, time
from pathlib import Path
Path(os.environ['VIEW_COLLECTOR']).write_text(str(os.getpid()))
while True:
    print('95, 33, 28000, 32000, 550, 3000, 70', flush=True)
    time.sleep(.01)
''')
        self.write_executable("tmux", '''import os, sys
if 'display-message' in sys.argv:
    print(os.environ.get('VIEW_TRAIN_PID', ''))
else:
    print('uttt_train')
''')

    def write_executable(self, name, source):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
        path.chmod(0o755)

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        for record_path, collector_path in self.records:
            if record_path.exists():
                record = json.loads(record_path.read_text())
                for pid in (record["ssh_pid"], record["remote_pid"]):
                    if self.running(pid):
                        os.kill(pid, signal.SIGHUP)
            if collector_path.exists():
                pid = int(collector_path.read_text())
                if self.running(pid):
                    os.killpg(pid, signal.SIGTERM)
        self.temp.cleanup()

    @staticmethod
    def running(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
        except FileNotFoundError:
            return False

    def launch(self, arguments, *, script=None, trainer=None):
        record_path = self.root / f"view{len(self.records)}.json"
        collector_path = record_path.with_suffix(".collector")
        self.records.append((record_path, collector_path))
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   REMOTE_HOST="isolated-test", REMOTE_USER="test", REMOTE_DIR=str(self.root),
                   SSH_KEY=str(self.root / "no-key"), VIEW_RECORD=str(record_path),
                   VIEW_COLLECTOR=str(collector_path), VIEW_TRAIN_PID=str(trainer.pid) if trainer else "")
        process = subprocess.Popen(["bash", str(script or self.script), *arguments],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.processes.append(process)
        deadline = time.monotonic() + 3
        while not record_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(record_path.exists(), "viewer SSH did not start")
        record = json.loads(record_path.read_text())
        return process, record, collector_path

    def assert_detached(self, process, record):
        process.terminate()
        process.wait(timeout=5)
        deadline = time.monotonic() + 2
        while any(self.running(record[key]) for key in ("ssh_pid", "remote_pid")) and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(self.running(record["ssh_pid"]), "local SSH client leaked")
        self.assertFalse(self.running(record["remote_pid"]), "remote viewer leaked")

    def wait_remote_program(self, record, name):
        path = Path(f"/proc/{record['remote_pid']}/comm")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                if path.read_text().strip() == name:
                    return
            except FileNotFoundError:
                break
            time.sleep(.01)
        self.fail(f"remote viewer did not execute {name}")

    def test_status_command_survives_stdin_eof_then_cleans_on_panel_close(self):
        process, record, collector = self.launch(["status", "10"])
        deadline = time.monotonic() + 3
        while not collector.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(collector.exists(), "encoded Python monitor did not execute")
        self.assertIsNone(process.poll(), "stdin EOF terminated the forced PTY monitor")
        collector_pid = int(collector.read_text())
        self.assert_detached(process, record)
        self.assertFalse(self.running(collector_pid), "GPU collector leaked after local panel close")

    def test_idle_log_view_exits_when_panel_closes(self):
        process, record, _ = self.launch(["logs", "train"])
        self.assertIn("exec tail", record["args"][-1])
        self.wait_remote_program(record, "tail")
        self.assert_detached(process, record)

    def test_training_attach_detaches_without_signalling_trainer_and_exits_at_completion(self):
        # Exercise cmd_train's actual attach code while replacing only its
        # launch/sync actions, so this test never starts a real training run.
        source = self.script.read_text(encoding="utf-8").rsplit('\nmain "$@"', 1)[0]
        harness = self.root / "attach.sh"
        harness.write_text(source + '\ncmd_sync() { :; }\nremote_exec() { cat >/dev/null; }\ncmd_train\n')
        trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.processes.append(trainer)
        first, first_record, _ = self.launch([], script=harness, trainer=trainer)
        self.assertIn("--pid=", first_record["args"][-1])
        self.wait_remote_program(first_record, "tail")
        self.assert_detached(first, first_record)
        self.assertIsNone(trainer.poll(), "detaching log view stopped the trainer")
        second, second_record, _ = self.launch([], script=harness, trainer=trainer)
        self.wait_remote_program(second_record, "tail")
        trainer.terminate()
        trainer.wait(timeout=2)
        second.wait(timeout=5)
        self.assertFalse(self.running(second_record["remote_pid"]), "tail outlived its training PID")


if __name__ == "__main__":
    unittest.main()

"""Exercise real viewer commands through an isolated non-PTY SSH pipe stub."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(os.name == "posix" and Path("/proc").exists(), "Linux SSH pipe lifecycle integration")
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
        self.write_executable("ssh", '''import errno, json, os, selectors, signal, subprocess, sys
from pathlib import Path
command = sys.argv[-1]
assert len(command.encode()) < 1024, 'SSH exec command exceeds the Windows bootstrap budget'
is_bootstrap = command.startswith('python3 -c ')
payload = sys.stdin.buffer.read() if is_bootstrap else b''
with Path(os.environ['VIEW_CALLS']).open('a') as trace:
    trace.write(json.dumps({'args': sys.argv[1:], 'stdin_bytes': len(payload)}) + '\\n')
if is_bootstrap:
    assert '-tt' not in sys.argv, sys.argv
    if os.environ.get('VIEW_BOOTSTRAP_FAIL') == '1':
        raise SystemExit(73)
    raise SystemExit(subprocess.run(['bash', '-c', command], input=payload).returncode)
assert '-T' in sys.argv and '-tt' not in sys.argv, sys.argv
if os.environ.get('VIEW_VIEWER_FAIL') == '1':
    raise SystemExit(255)
remote = subprocess.Popen(['bash', '-c', command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, start_new_session=True)
record = Path(os.environ['VIEW_RECORD'])
temporary = record.with_suffix('.tmp')
temporary.write_text(json.dumps({'ssh_pid': os.getpid(), 'remote_pid': remote.pid, 'args': sys.argv[1:]}))
temporary.replace(record)
stopping = False
def stop(*_):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGHUP, stop)
selector = selectors.DefaultSelector()
selector.register(sys.stdin.buffer, selectors.EVENT_READ, 'stdin')
selector.register(remote.stdout, selectors.EVENT_READ, 'stdout')
try:
    while not stopping:
        for key, _ in selector.select(.05):
            data = os.read(key.fd, 65536)
            if key.data == 'stdin':
                if data:
                    remote.stdin.write(data)
                    remote.stdin.flush()
                else:
                    remote.stdin.close()
                    selector.unregister(key.fileobj)
            elif not data:
                stopping = True
                break
            else:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
finally:
    selector.close()
    if not remote.stdin.closed:
        remote.stdin.close()
    try:
        remote.wait(timeout=4)
    except subprocess.TimeoutExpired:
        os.killpg(remote.pid, signal.SIGHUP)
        remote.wait(timeout=4)
    remote.stdout.close()
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

    def launch(self, arguments, *, script=None, trainer=None, expect_viewer=True,
               bootstrap_failure=False, viewer_failure=False):
        record_path = self.root / f"view{len(self.records)}.json"
        collector_path = record_path.with_suffix(".collector")
        self.records.append((record_path, collector_path))
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   REMOTE_HOST="isolated-test", REMOTE_USER="test", REMOTE_DIR=str(self.root),
                   SSH_KEY=str(self.root / "no-key"), VIEW_RECORD=str(record_path),
                   VIEW_COLLECTOR=str(collector_path), VIEW_TRAIN_PID=str(trainer.pid) if trainer else "",
                   VIEW_CALLS=str(record_path.with_suffix('.calls')),
                   VIEW_BOOTSTRAP_FAIL=str(int(bootstrap_failure)), VIEW_VIEWER_FAIL=str(int(viewer_failure)))
        process = subprocess.Popen(["bash", str(script or self.script), *arguments],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.processes.append(process)
        if not expect_viewer:
            process.wait(timeout=5)
            self.assertFalse(record_path.exists(), "a failed bootstrap started a viewer")
            return process, record_path, collector_path
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
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            pending, seen = [record['remote_pid']], set()
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    if Path(f'/proc/{pid}/comm').read_text().strip() == name:
                        return pid
                    pending.extend(int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
                except FileNotFoundError:
                    continue
            time.sleep(.01)
        self.fail(f"remote viewer did not execute {name}")

    def wait_collector(self, process, collector):
        deadline = time.monotonic() + 3
        while not collector.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(collector.exists(), "uploaded Python monitor did not execute")
        return int(collector.read_text())

    def assert_remote_exited(self, record, *extra):
        pids = [record['ssh_pid'], record['remote_pid'], *extra]
        deadline = time.monotonic() + 5
        while any(self.running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(any(self.running(pid) for pid in pids), f'view process leaked: {pids}')

    def test_status_short_commands_keep_stdin_open_then_clean_on_panel_close(self):
        process, record, collector = self.launch(["status", "10"])
        collector_pid = self.wait_collector(process, collector)
        trace_path = self.records[-1][0].with_suffix('.calls')
        calls = [json.loads(line) for line in trace_path.read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('-tt' not in call['args'] for call in calls))
        self.assertIn('-T', calls[-1]['args'])
        helper = self.script.with_name('monitor_status.py').read_bytes()
        self.assertEqual(calls[0]['stdin_bytes'], len(helper))
        self.assertEqual(calls[1]['stdin_bytes'], self.script.with_name('remote_view.py').stat().st_size)
        self.assertEqual(calls[-1]['stdin_bytes'], 0)
        self.assertNotIn('base64', calls[-1]['args'][-1])
        self.assertTrue(all(len(call['args'][-1].encode()) < 1024 for call in calls))
        installed = list((self.root / '.cache' / 'remote-view').glob('*-monitor_status.py'))
        self.assertEqual(len(installed), 1)
        self.assertEqual(installed[0].read_bytes(), helper)
        self.assertIsNone(process.poll(), "the non-PTY monitor lost its stdin keepalive")
        self.assert_detached(process, record)
        self.assertFalse(self.running(collector_pid), "GPU collector leaked after local panel close")

    def test_failed_monitor_upload_does_not_launch_a_viewer(self):
        process, record_path, collector = self.launch(["status", "10"],
                                                     expect_viewer=False, bootstrap_failure=True)
        self.assertEqual(process.returncode, 1)
        self.assertIn('Viewer helper upload failed', process.stderr.read().decode())
        self.assertFalse(collector.exists())
        calls = [json.loads(line) for line in record_path.with_suffix('.calls').read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertNotIn('-tt', calls[0]['args'])

    def test_remote_exec_rejection_is_reported_as_failure(self):
        process, record_path, collector = self.launch(["status", "10"],
                                                     expect_viewer=False, viewer_failure=True)
        self.assertEqual(process.returncode, 255)
        self.assertFalse(collector.exists())
        calls = [json.loads(line) for line in record_path.with_suffix('.calls').read_text().splitlines()]
        self.assertEqual(len(calls), 3)
        self.assertIn('-T', calls[-1]['args'])

    def test_parent_sigkill_closes_the_only_stdin_writer_and_reaps_remote_monitor(self):
        process, record, collector = self.launch(['status', '10'])
        collector_pid = self.wait_collector(process, collector)
        process.kill()
        process.wait(timeout=3)
        self.assert_remote_exited(record, collector_pid)

    def test_ssh_transport_loss_delivers_stdin_eof_to_remote_supervisor(self):
        process, record, collector = self.launch(['status', '10'])
        collector_pid = self.wait_collector(process, collector)
        os.kill(record['ssh_pid'], signal.SIGKILL)
        process.wait(timeout=5)
        self.assert_remote_exited(record, collector_pid)

    def test_idle_log_view_exits_when_panel_closes(self):
        process, record, _ = self.launch(["logs", "train"])
        self.assertIn("-- tail", record["args"][-1])
        tail_pid = self.wait_remote_program(record, "tail")
        self.assert_detached(process, record)
        self.assertFalse(self.running(tail_pid), 'log tail leaked')

    def test_training_attach_detaches_without_signalling_trainer_and_exits_at_completion(self):
        # Exercise cmd_train's actual attach code while replacing only its
        # launch/sync actions, so this test never starts a real training run.
        source = self.script.read_text(encoding="utf-8").rsplit('\nmain "$@"', 1)[0]
        harness = self.root / "attach.sh"
        harness.write_text(source + '\nSCRIPT_DIR=' + shlex.quote(str(self.script.parent))
                           + '\ncmd_sync() { :; }\nremote_exec() { if [[ "$*" == bash\\ -s* ]]; then cat >/dev/null; '
                           + 'else ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "$@"; fi; }\ncmd_train\n')
        trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.processes.append(trainer)
        first, first_record, _ = self.launch([], script=harness, trainer=trainer)
        self.assertIn("--train-session", first_record["args"][-1])
        self.wait_remote_program(first_record, "tail")
        self.assert_detached(first, first_record)
        self.assertIsNone(trainer.poll(), "detaching log view stopped the trainer")
        second, second_record, _ = self.launch([], script=harness, trainer=trainer)
        self.wait_remote_program(second_record, "tail")
        trainer.terminate()
        trainer.wait(timeout=2)
        second.wait(timeout=5)
        self.assertFalse(self.running(second_record["remote_pid"]), "tail outlived its training PID")

    def test_supervisor_forwards_literal_arguments_including_spaces(self):
        source = self.script.read_text(encoding='utf-8').rsplit('\nmain "$@"', 1)[0]
        harness = self.root / 'argv.sh'
        payload = "import json,sys;print(json.dumps(sys.argv[1:]))"
        literal = ['argument with spaces', 'semi;literal', '$(not-a-command)']
        command = ' '.join(shlex.quote(value) for value in ['python3', '-c', payload, *literal])
        harness.write_text(source + '\nSCRIPT_DIR=' + shlex.quote(str(self.script.parent))
                           + '\nvalidate_config\nremote_view_stream -- ' + command + '\n')
        process, _, _ = self.launch([], script=harness)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertIn(json.dumps(literal), stdout.decode())


if __name__ == "__main__":
    unittest.main()

"""Non-PTY disconnect cleanup remains responsive under output backpressure."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(os.name == 'posix' and Path('/proc').exists(), 'Linux SSH supervisor lifecycle')
class RemoteViewSupervisorTests(unittest.TestCase):
    def running(self, pid):
        try:
            return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
        except FileNotFoundError:
            return False

    def test_stdin_eof_reaps_child_even_when_output_is_full_and_unread(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / 'child.pid'
            program = ("import os,sys;from pathlib import Path;"
                       "Path(sys.argv[1]).write_text(str(os.getpid()));"
                       "exec(\"while True: os.write(1,b'x'*65536)\")")
            script = Path(__file__).resolve().parents[1] / 'deploy' / 'remote_view.py'
            process = subprocess.Popen([sys.executable, str(script), '--', sys.executable, '-c', program, str(pid_file)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            child_pid = None
            try:
                deadline = time.monotonic() + 3
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(pid_file.exists(), 'viewer child did not start')
                child_pid = int(pid_file.read_text())
                # Output fills the supervisor's bounded buffer and pipe. Do not
                # drain it: stdin EOF must still be serviced independently.
                time.sleep(.2)
                process.stdin.close()
                process.wait(timeout=5)
                self.assertEqual(process.returncode, 0, process.stderr.read().decode())
                self.assertFalse(self.running(child_pid), 'owned output producer leaked')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                if not process.stdin.closed:
                    process.stdin.close()
                process.stdout.close()
                process.stderr.close()
                if child_pid is not None and self.running(child_pid):
                    os.killpg(child_pid, signal.SIGKILL)

    def test_backpressured_monitor_finishes_signal_cleanup_of_separate_gpu_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bin').mkdir()
            (root / 'logs').mkdir()
            # One status line is large enough to fill both the bounded relay
            # and pipe buffers in a few frames, independently of scheduler Hz.
            (root / 'logs' / 'train_5090.log').write_text('backlog ' + 'X' * 65000 + '\n')
            pid_file = root / 'collector.pids'
            smi = root / 'bin' / 'nvidia-smi'
            smi.write_text(f'#!{sys.executable}\n' + '''import os,signal,time
from pathlib import Path
child=os.fork()
if child==0:
 while True:time.sleep(1)
def finish(*_):
 try:
  os.kill(child,signal.SIGTERM)
  os.waitpid(child,0)
 except ProcessLookupError:pass
 raise SystemExit(0)
signal.signal(signal.SIGTERM,finish)
p=Path(os.environ['COLLECTOR_PID_FILE'])
t=p.with_suffix('.tmp')
t.write_text(f'{os.getppid()} {os.getpid()} {child}')
t.replace(p)
while True:
 print('90,30,1000,32000,500,2900,60',flush=True)
 time.sleep(.001)
''')
            smi.chmod(0o755)
            deploy = Path(__file__).resolve().parents[1] / 'deploy'
            environment = dict(os.environ, PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'],
                               COLLECTOR_PID_FILE=str(pid_file))
            process = subprocess.Popen([sys.executable, str(deploy / 'remote_view.py'), '--',
                                        sys.executable, str(deploy / 'monitor_status.py'),
                                        '--root', str(root), '--interval-ms', '1'],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=environment)
            monitor_pid = collector_pid = grandchild_pid = None
            try:
                deadline = time.monotonic() + 3
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(pid_file.exists(), 'monitor collector did not start')
                monitor_pid, collector_pid, grandchild_pid = map(int, pid_file.read_text().split())
                deadline = time.monotonic() + 3
                blocked = False
                while time.monotonic() < deadline:
                    channel = Path(f'/proc/{monitor_pid}/wchan').read_text()
                    if 'pipe' in channel and 'write' in channel:
                        blocked = True
                        break
                    time.sleep(.01)
                self.assertTrue(blocked, 'monitor did not reach output backpressure')
                process.stdin.close()
                process.wait(timeout=5)
                self.assertEqual(process.returncode, 0, process.stderr.read().decode())
                self.assertFalse(any(self.running(pid) for pid in (monitor_pid, collector_pid, grandchild_pid)),
                                 'monitor/GPU collector group leaked after stdin EOF')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                if not process.stdin.closed:
                    process.stdin.close()
                process.stdout.close()
                process.stderr.close()
                if collector_pid is not None and self.running(collector_pid):
                    os.killpg(collector_pid, signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()

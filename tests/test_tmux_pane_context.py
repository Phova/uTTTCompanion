"""Exercise pane discovery against an isolated real tmux server, never training."""
import contextlib
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import uuid

from deploy import remote_view, stop_training


@unittest.skipUnless(os.name == 'posix' and shutil.which('tmux'), 'Linux tmux required')
class TmuxPaneContextTests(unittest.TestCase):
    def test_log_follow_and_graceful_stop_find_the_actual_pane_without_a_client(self):
        socket = 'uttt-test-' + uuid.uuid4().hex
        raw_run = subprocess.run

        def isolated(arguments, *args, **kwargs):
            if arguments[0] == 'tmux':
                arguments = ['tmux', '-L', socket, '-f', '/dev/null', *arguments[1:]]
            return raw_run(arguments, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'train_5090.py'
            script.write_text("import pathlib,signal,sys\n"
                              "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n"
                              "pathlib.Path('started').touch()\n"
                              "while True: signal.pause()\n")
            command = 'exec ' + shlex.join([sys.executable, str(script)])
            try:
                launched = isolated(['tmux', 'new-session', '-d', '-s', 'uttt_train', '-c', str(root),
                                     '-P', '-F', '#{pane_pid}', command], capture_output=True, text=True, check=True)
                pane = int(launched.stdout.strip())
                deadline = time.monotonic() + 5
                while not (root / 'started').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((root / 'started').exists())
                with mock.patch.object(remote_view.subprocess, 'run', side_effect=isolated):
                    follow = remote_view.training_command(root, 'uttt_train')
                self.assertIsNotNone(follow)
                self.assertIn('--pid=' + str(pane), follow)
                with mock.patch.object(sys, 'argv', ['stop_training.py', '--workspace', str(root),
                                                    '--session', 'uttt_train', '--timeout', '5']), \
                        mock.patch.object(stop_training.subprocess, 'run', side_effect=isolated), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    stop_training.main()
                self.assertIn(str(pane), output.getvalue())
                self.assertNotIn(pane, stop_training.process_table())
            finally:
                isolated(['tmux', 'kill-server'], capture_output=True)


if __name__ == '__main__':
    unittest.main()

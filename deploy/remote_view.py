#!/usr/bin/env python3
"""Supervise an SSH viewer and reap only its own process group on disconnect."""
import argparse
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time


def training_command(root, session):
    """Follow one training supervisor without ever sending it a signal."""
    pid = None
    if shutil.which('tmux'):
        result = subprocess.run(['tmux', 'display-message', '-p', '-t', '=' + session, '#{pane_pid}'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip()
    if pid is None:
        try:
            pid = (root / 'logs' / (session + '.pid')).read_text().strip()
        except OSError:
            pass
    if not pid or not pid.isdecimal() or int(pid) < 1:
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return None
    return ['tail', '--pid=' + pid, '-n', '+1', '-F', '--', str(root / 'logs' / 'train_5090.log')]


class RemoteView:
    def __init__(self, command):
        self.command = command
        self.child = None
        self.stopping = False

    def stop(self, *_):
        self.stopping = True

    def close(self):
        child = self.child
        if child is None:
            return
        # The child's new group contains this viewer only. The real trainer
        # lives outside it, including when our child is tail --pid=<trainer>.
        if child.returncode is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # A monitor can be blocked in flush() when SSH output is full.
            # Drain/discard its output during the TERM grace period so its
            # signal handler can finish and reap its own GPU collector group.
            # Never communicate(): an unbounded producer must not fill RAM.
            deadline = time.monotonic() + 3
            drain = selectors.DefaultSelector()
            try:
                os.set_blocking(child.stdout.fileno(), False)
                drain.register(child.stdout, selectors.EVENT_READ)
                while child.poll() is None and time.monotonic() < deadline:
                    for key, _ in drain.select(min(.05, max(0, deadline - time.monotonic()))):
                        try:
                            data = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not data:
                            drain.unregister(key.fileobj)
            finally:
                drain.close()
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=1)
        child.stdout.close()

    def run(self):
        selector = selectors.DefaultSelector()
        panel_fd = sys.stdout.fileno()
        was_blocking = os.get_blocking(panel_fd)
        pending = bytearray()
        child_eof, reading_output, writing_panel = False, True, False
        try:
            # A stalled SSH output pipe must not prevent us observing stdin EOF.
            os.set_blocking(panel_fd, False)
            selector.register(sys.stdin.buffer, selectors.EVENT_READ, 'stdin')
            self.child = subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            selector.register(self.child.stdout, selectors.EVENT_READ, 'output')
            while not self.stopping:
                for key, _ in selector.select(.1):
                    if key.data == 'stdin':
                        data = os.read(key.fd, 65536)
                        if not data:
                            self.stopping = True
                            break
                    elif key.data == 'output':
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(self.child.stdout)
                            reading_output, child_eof = False, True
                        else:
                            pending.extend(data)
                            if len(pending) >= 262144:
                                selector.unregister(self.child.stdout)
                                reading_output = False
                            if not writing_panel:
                                selector.register(panel_fd, selectors.EVENT_WRITE, 'panel')
                                writing_panel = True
                    else:
                        try:
                            written = os.write(panel_fd, pending)
                        except BlockingIOError:
                            continue
                        del pending[:written]
                        if not pending:
                            selector.unregister(panel_fd)
                            writing_panel = False
                        if not reading_output and not child_eof and len(pending) < 262144:
                            selector.register(self.child.stdout, selectors.EVENT_READ, 'output')
                            reading_output = True
                if child_eof and not pending:
                    return self.child.wait()
            return 0
        except OSError as exc:
            if exc.errno not in (5, 32):
                raise
            return 0
        finally:
            selector.close()
            self.close()
            try:
                os.set_blocking(panel_fd, was_blocking)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path)
    parser.add_argument('--train-session')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.train_root is not None:
        if command or not args.train_session:
            parser.error('training view requires a session and no explicit command')
        command = training_command(args.train_root, args.train_session)
        if command is None:
            print('[Remote] Training session is no longer active.', flush=True)
            return 0
    if not command:
        parser.error('a viewer command is required')
    viewer = RemoteView(command)
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP', 'SIGPIPE'):
        signal.signal(getattr(signal, name), viewer.stop)
    try:
        return viewer.run()
    finally:
        try:
            sys.stdout.flush()
        except OSError:
            sys.stdout = open(os.devnull, 'w')


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only SSH dashboard with session-scoped GPU collector cleanup."""
import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time


def positive_interval(value):
    try:
        interval = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("refresh interval must be a positive integer in milliseconds") from exc
    if interval < 1 or interval > 60000:
        raise argparse.ArgumentTypeError("refresh interval must be between 1 and 60000 milliseconds")
    return interval


def target_hz(interval_ms):
    return f"{1000 / interval_ms:g}"


def latest_training_line(path):
    try:
        with path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 65536))
            lines = log.read().decode("utf-8", errors="replace").splitlines()
        progress = [line for line in lines if "[Step " in line]
        return (progress or lines or ["waiting for log..."])[-1]
    except OSError:
        return "waiting for log..."


def compact_training_line(line):
    """Keep the complete learning rate and throughput visible on narrow panels."""
    step = re.search(r"\[Step\s+([^\]]+)\]", line)
    if step is None:
        return line
    parts = [step.group(1)]
    for label, short in (("Stage", ""), ("CE Loss", "CE"), ("LR", "LR"), ("Speed", "")):
        match = re.search(r"(?:^|\||\])\s*" + label + r":\s*([^|]+)", line)
        if match:
            parts.append(short + match.group(1).strip())
    return " ".join(parts)


def owner_checkpoint_output(root):
    """Return (explicit_output, path) without exposing owner credentials.

    The current owner protocol records workspace and argv, but not its cwd.
    Relative outputs therefore use that workspace unless a future record also
    supplies run_directory. An explicit missing/invalid output must not select
    a different run's old checkpoint.
    """
    root = Path(root)
    try:
        with (root / "logs" / ".wsl_owners" / "uttt_train.json").open("rb") as source:
            contents = source.read(65537)
        if len(contents) > 65536:
            return False, None
        record = json.loads(contents.decode("utf-8"))
        if not isinstance(record, dict) or record.get("session") != "uttt_train":
            return False, None
        workspace = record.get("workspace")
        if not isinstance(workspace, str) or Path(workspace).resolve() != root.resolve():
            return False, None
        command = record.get("command")
        if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
            return False, None
        explicit, value = False, None
        for index, argument in enumerate(command):
            if argument == "--output-ckpt":
                explicit = True
                value = command[index + 1] if index + 1 < len(command) else None
                if value is not None and value.startswith("--"):
                    value = None
            elif argument.startswith("--output-ckpt="):
                explicit, value = True, argument.partition("=")[2]
        if not explicit:
            return False, None
        if not value or not value.strip() or "\0" in value:
            return True, None
        output = Path(value)
        if not output.is_absolute():
            working = record.get("run_directory", workspace)
            if not isinstance(working, str) or not working.strip() or "\0" in working:
                return True, None
            working = Path(working)
            if not working.is_absolute():
                working = root / working
            output = working / output
        return True, output
    except (OSError, ValueError, UnicodeError, RecursionError, RuntimeError):
        return False, None


def _checkpoint_description(path):
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            return None
        name = "".join(character if character.isprintable() else "?" for character in path.name)
        return (f"{name} ({info.st_size / 1024**3:.2f} GiB updated "
                f"{datetime.fromtimestamp(info.st_mtime):%H:%M:%S})")
    except (OSError, ValueError, OverflowError):
        return None


def checkpoint_status(root):
    explicit, output = owner_checkpoint_output(root)
    if explicit:
        if output is None:
            return "waiting for configured checkpoint (invalid output path)"
        description = _checkpoint_description(output)
        name = "".join(character if character.isprintable() else "?" for character in output.name)
        return description or f"{name} (waiting for first save)"
    # Legacy/manual runs without an explicit owner output keep the old display.
    candidates = []
    try:
        for path in (Path(root) / "checkpoints").glob("*.pt"):
            try:
                info = path.stat()
                if stat.S_ISREG(info.st_mode):
                    candidates.append((info.st_mtime, path))
            except OSError:
                continue
    except OSError:
        return "none"
    return (_checkpoint_description(max(candidates, key=lambda item: item[0])[1]) or "none") if candidates else "none"


class Monitor:
    def __init__(self, root, interval_ms, output=None, columns=None):
        self.root = Path(root)
        self.interval_ms = interval_ms
        self.output = output if output is not None else sys.stdout
        self.columns = columns
        self.stopping = False
        self.collector = None
        self.selector = selectors.DefaultSelector()
        self.metrics = ["--"] * 7
        self.pending = b""
        self.frames = deque()
        self.sessions = "checking..."
        self.checkpoint = "checking..."
        self.training = "waiting for log..."

    def stop(self, *_):
        self.stopping = True

    def start_collector(self):
        try:
            self.collector = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.selector.register(self.collector.stdout, selectors.EVENT_READ)
        except OSError:
            self.metrics = ["unavailable"] + ["--"] * 6

    def read_collector(self, key):
        data = os.read(key.fd, 65536)
        if not data:
            self.selector.unregister(key.fileobj)
            self.metrics = ["unavailable"] + ["--"] * 6
            return
        self.pending += data
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            fields = [item.strip() for item in line.decode("utf-8", errors="replace").split(",")]
            if len(fields) == 7:
                self.metrics = fields

    def refresh_metadata(self):
        try:
            result = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                                    capture_output=True, text=True, timeout=0.5)
            self.sessions = ",".join(result.stdout.splitlines()) or "none"
        except (OSError, subprocess.TimeoutExpired):
            self.sessions = "none"
        self.checkpoint = checkpoint_status(self.root)
        self.training = latest_training_line(self.root / "logs" / "train_5090.log")

    def render(self, now):
        self.frames.append(now)
        while len(self.frames) > 2 and self.frames[1] < now - 1:
            self.frames.popleft()
        actual = "--"
        if len(self.frames) > 1 and now > self.frames[0]:
            actual = f"{(len(self.frames) - 1) / (now - self.frames[0]):.1f}"
        gpu, bandwidth, used, total, power, clock, temperature = self.metrics
        lines = [
            f"[uTTT 5090 target {target_hz(self.interval_ms)}Hz / actual {actual}Hz] "
            f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} | tmux: {self.sessions} | Ctrl+C to exit",
            f"GPU  : Util: {gpu}% | MemBW: {bandwidth}% | VRAM: {used}/{total} MiB | Power: {power}W | {clock}MHz | {temperature}C",
            f"Ckpt : {self.checkpoint}",
            f"Train: {compact_training_line(self.training)}",
            "-" * 108,
        ]
        if self.columns is not None or self.output.isatty():
            columns = max(20, self.columns if self.columns is not None
                          else shutil.get_terminal_size(fallback=(140, 24)).columns)
            lines = [line[:columns - 1] for line in lines]
        self.output.write("\033[5A" + "".join("\r\033[2K" + line + "\n" for line in lines))
        self.output.flush()

    def close(self):
        self.selector.close()
        process = self.collector
        if process is None:
            return
        # The dedicated group contains only this dashboard's collector and its
        # descendants. Other monitors and the trainer are never selected.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=1)
        if process.stdout is not None:
            process.stdout.close()

    def run(self):
        parent_pid = os.getppid()
        try:
            self.start_collector()
            self.output.write("\n" * 5)
            self.output.flush()
            deadline = time.monotonic()
            metadata_deadline = 0.0
            interval = self.interval_ms / 1000
            while not self.stopping and os.getppid() == parent_pid:
                now = time.monotonic()
                if now >= metadata_deadline:
                    self.refresh_metadata()
                    metadata_deadline = time.monotonic() + 1
                now = time.monotonic()
                if now >= deadline:
                    self.render(now)
                    deadline = now + interval
                # Bound the wait so HUP/TERM and parent loss are handled even
                # when the requested refresh interval is very long.
                for key, _ in self.selector.select(min(0.1, max(0, deadline - time.monotonic()))):
                    self.read_collector(key)
        except (BrokenPipeError, OSError) as exc:
            if isinstance(exc, OSError) and exc.errno not in (5, 32):
                raise
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--interval-ms", type=positive_interval, default=200)
    parser.add_argument("--columns", type=int, help="Local panel width when SSH has no remote PTY")
    args = parser.parse_args()
    if args.columns is not None and not 20 <= args.columns <= 10000:
        parser.error("columns must be between 20 and 10000")
    monitor = Monitor(args.root, args.interval_ms, columns=args.columns)
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGPIPE"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), monitor.stop)
    try:
        monitor.run()
    finally:
        # Avoid a second flush on an already disconnected SSH output stream.
        if monitor.output is sys.stdout:
            try:
                sys.stdout.flush()
            except OSError:
                sys.stdout = open(os.devnull, "w")


if __name__ == "__main__":
    main()

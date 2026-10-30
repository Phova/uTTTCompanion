#!/usr/bin/env python3
"""Keep a foreground WSL client alive until its owned tmux training pane exits."""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import traceback
import uuid

try:
    import fcntl
except ImportError:
    fcntl = None


def session_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise argparse.ArgumentTypeError("session must contain only letters, digits, _ or - (1-64 characters)")
    return value


def process_identity(pid):
    """Include the kernel start counter so a reused PID is never trusted."""
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_time": int(fields[19]), "uid": root.stat().st_uid,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "cmdline": [os.fsdecode(value) for value in (root / "cmdline").read_bytes().split(b"\0") if value]}
    except (OSError, ValueError, IndexError):
        return None


def same_process(current, expected):
    return current is not None and all(current.get(key) == expected.get(key)
                                       for key in ("pid", "start_time", "uid", "boot_id", "cmdline"))


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class OwnerBusyError(RuntimeError):
    pass


class OwnerLock:
    def __init__(self, path):
        self.path = path
        self.descriptor = None

    def __enter__(self):
        if fcntl is None:
            raise RuntimeError("run this helper inside Linux/WSL, not Windows Python")
        self.descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        details = os.fstat(self.descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid() or details.st_nlink != 1:
            os.close(self.descriptor)
            self.descriptor = None
            raise RuntimeError("unsafe owner lock file; refusing to start")
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.descriptor)
            self.descriptor = None
            raise OwnerBusyError("another foreground owner already holds this workspace/session") from exc
        return self

    def __exit__(self, *_):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def owner_paths(workspace, session):
    directory = workspace / "logs" / ".wsl_owners"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory, directory / (session + ".lock"), directory / (session + ".json")


def tmux(*arguments, check=True):
    return subprocess.run(["tmux", *arguments], check=check, text=True, capture_output=True)


def build_wrapper(command, working_directory, log_path, result_path, token, channels):
    """The acknowledgment keeps tmux alive even if the command exits instantly."""
    ready, start, finished, acknowledged = map(shlex.quote, channels)
    result = shlex.quote(str(result_path))
    temporary = shlex.quote(str(result_path) + ".tmp")
    return f"""#!/usr/bin/env bash
set -u
umask 077
owner_finished() {{
    owner_exit=$?
    trap - EXIT HUP INT TERM
    printf '{{"token":"{token}","exit_code":%d}}\\n' "$owner_exit" > {temporary}
    mv -f -- {temporary} {result}
    tmux wait-for -S {finished}
    tmux wait-for {acknowledged}
    exit "$owner_exit"
}}
trap owner_finished EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
tmux wait-for -S {ready}
tmux wait-for {start}
cd -- {shlex.quote(str(working_directory))} || exit 125
set -o pipefail
{shlex.join(command)} 2>&1 | tee -a -- {shlex.quote(str(log_path))}
exit "${{PIPESTATUS[0]}}"
"""


def validate_owned_session(record, workspace, session, *, identity=process_identity, tmux_call=tmux):
    if record.get("workspace") != str(workspace) or record.get("session") != session:
        raise RuntimeError("owner record belongs to a different workspace/session")
    if hasattr(os, "getuid") and record["owner"]["uid"] != os.getuid():
        raise RuntimeError("foreground owner belongs to a different Linux user")
    if not same_process(identity(record["owner"]["pid"]), record["owner"]):
        raise RuntimeError("foreground owner identity is stale; refusing to signal any process")
    if not same_process(identity(record["pane"]["pid"]), record["pane"]):
        raise RuntimeError("tmux pane identity changed; refusing stop")
    actual_token = tmux_call("show-options", "-qv", "-t", record["session_id"], "@uttt_wsl_owner").stdout.strip()
    if actual_token != record.get("token"):
        raise RuntimeError("tmux session is not owned by this record; refusing stop")
    actual_panes = tmux_call("list-panes", "-t", record["session_id"] + ":", "-F", "#{pane_pid}").stdout.splitlines()
    if actual_panes != [str(record["pane"]["pid"])]:
        raise RuntimeError("tmux session pane layout changed; refusing stop")


def stop_owned_session(workspace, session, timeout, *, identity=process_identity, tmux_call=tmux, runner=None):
    _, lock_path, record_path = owner_paths(workspace, session)
    if not record_path.exists():
        raise RuntimeError("no foreground owner record exists for this workspace/session")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    validate_owned_session(record, workspace, session, identity=identity, tmux_call=tmux_call)
    # A live PID record alone is insufficient: its owner must still hold its
    # lock. Never unlink locks, signal the owner, or kill a whole tmux session.
    try:
        with OwnerLock(lock_path):
            raise RuntimeError("owner record has no live lock; refusing stop")
    except OwnerBusyError:
        pass
    if runner is None:
        module_name = (__package__ + ".stop_training") if __package__ else "stop_training"
        runner = importlib.import_module(module_name).stop
    # Keep the already verified pane as the root. Re-discovering by session
    # name here could select a replacement session created after validation.
    runner([record["pane"]["pid"]], workspace, timeout)


def run_owner(workspace, session, working_directory, log_path, command, timeout=300):
    workspace = workspace.resolve(strict=True)
    working_directory = working_directory.resolve(strict=True)
    log_path = log_path.resolve()
    if not working_directory.is_relative_to(workspace) or not log_path.is_relative_to(workspace):
        raise ValueError("run directory and training log must be inside the workspace")
    if not command or not Path(command[0]).is_absolute():
        raise ValueError("training command must begin with an absolute Linux executable path")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    directory, lock_path, record_path = owner_paths(workspace, session)
    with OwnerLock(lock_path):
        if tmux("has-session", "-t", "=" + session, check=False).returncode == 0:
            raise RuntimeError("tmux session already exists; refusing duplicate launch or automatic adoption")
        token = uuid.uuid4().hex
        channels = tuple("uttt-owner-" + token + "-" + part for part in ("ready", "start", "finished", "ack"))
        wrapper_path = directory / (session + "." + token + ".sh")
        result_path = directory / (session + "." + token + ".result.json")
        wrapper_path.write_text(build_wrapper(command, working_directory, log_path, result_path, token, channels), encoding="utf-8")
        wrapper_path.chmod(0o700)
        record = {"workspace": str(workspace), "session": session, "token": token,
                  "owner": process_identity(os.getpid()), "phase": "starting", "command": command}
        pane_created = False
        training_started = False
        stop_requested = False
        old_handlers = {}
        waiter = None

        def record_event(message):
            try:
                with (directory / (session + ".owner.log")).open("a", encoding="utf-8") as output:
                    output.write(message + "\n")
                    output.flush()
            except OSError:
                pass
            try:
                print(message, flush=True)
            except OSError:
                pass

        def request_stop(signum, _):
            nonlocal stop_requested
            # Signal handlers only latch intent. A primary process may not yet
            # exist while the pane is at/releasing its startup barrier.
            stop_requested = True

        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            if not hasattr(signal, name):
                continue
            signum = getattr(signal, name)
            old_handlers[signum] = signal.signal(signum, request_stop)
        try:
            launched = tmux("new-session", "-d", "-s", session, "-c", str(working_directory),
                            "-P", "-F", "#{session_id} #{pane_id} #{pane_pid}", "bash", str(wrapper_path))
            pane_created = True
            session_id, pane_id, pane_pid_text = launched.stdout.strip().split()
            record.update(session_id=session_id, pane_id=pane_id)
            pane_pid = int(pane_pid_text)
            tmux("wait-for", channels[0])
            record["pane"] = process_identity(pane_pid)
            if record["pane"] is None:
                raise RuntimeError("new tmux pane disappeared before launch authorization")
            tmux("set-option", "-t", record["session_id"], "@uttt_wsl_owner", token)
            atomic_json(record_path, record)
            if stop_requested:
                return 143
            record["phase"] = "running"
            atomic_json(record_path, record)
            # The foreground waiter has no busy loop or sleep child. The pane
            # waits for our final acknowledgment, so even /bin/true is safe.
            waiter = subprocess.Popen(["tmux", "wait-for", channels[2]])
            if stop_requested:
                return 143
            # Releasing the start barrier is the launch commit point. A signal
            # during this tmux call remains pending until the primary exists;
            # it is never cleared merely because stop found no trainer yet.
            training_started = True
            tmux("wait-for", "-S", channels[1])
            record_event(f"Foreground WSL owner {os.getpid()} holds tmux {session}, pane PID {pane_pid}, token {token}.")
            stop_completed = False
            stop_error = None
            while waiter.poll() is None:
                if stop_requested and not stop_completed:
                    try:
                        stop_owned_session(workspace, session, timeout)
                        stop_completed = True
                    except Exception as exc:
                        message = str(exc)
                        if message != stop_error:
                            record_event(f"Safe stop pending; foreground owner retained: {message}")
                            stop_error = message
                try:
                    waiter.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass

            exit_code = 125
            if not result_path.exists() and process_identity(pane_pid) is not None:
                raise RuntimeError("completion waiter exited before its live pane; retaining foreground owner")
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("token") != token or not isinstance(result.get("exit_code"), int):
                    raise RuntimeError("unexpected completion record; refusing acknowledgment")
                exit_code = result["exit_code"]
                if not 0 <= exit_code <= 255:
                    raise RuntimeError("invalid training exit status")
                tmux("wait-for", "-S", channels[3], check=False)
            record.update(phase="exited", exit_code=exit_code)
            atomic_json(record_path, record)
            record_event(f"Training pane completed with status {exit_code}; foreground WSL owner exiting.")
            return exit_code
        except BaseException:
            record_event("Foreground owner error:\n" + traceback.format_exc())
            # Do not drop the last Windows WSL client on an unexpected Python
            # error while its pane still exists. A finished pane waits for ack;
            # an active one retains the owner until completion or a safe stop.
            if training_started:
                while process_identity(record["pane"]["pid"]) is not None:
                    if stop_requested:
                        try:
                            stop_owned_session(workspace, session, timeout)
                        except Exception as exc:
                            record_event(f"Safe stop remains pending during owner recovery: {exc}")
                    if result_path.exists():
                        tmux("wait-for", "-S", channels[3], check=False)
                        break
                    waiting = tmux("wait-for", channels[2], check=False)
                    if waiting.returncode and process_identity(record["pane"]["pid"]) is not None:
                        record_event("tmux wait failed; retaining foreground owner until an operator signal.")
                        signal.pause()
                record.update(phase="exited", exit_code=125)
                try:
                    atomic_json(record_path, record)
                except OSError:
                    pass
                return 125
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            if waiter is not None and waiter.poll() is None:
                waiter.terminate()
                waiter.wait(timeout=5)
            if not training_started:
                if pane_created and record.get("pane") and same_process(process_identity(record["pane"]["pid"]), record["pane"]):
                    tmux("kill-pane", "-t", record["pane_id"], check=False)
                wrapper_path.unlink(missing_ok=True)
            elif record.get("phase") == "exited":
                wrapper_path.unlink(missing_ok=True)
                result_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "stop"):
        command_parser = subparsers.add_parser(mode)
        command_parser.add_argument("--workspace", type=Path, required=True)
        command_parser.add_argument("--session", type=session_name, default="uttt_train")
        command_parser.add_argument("--timeout", type=float, default=300)
        if mode == "run":
            command_parser.add_argument("--cwd", type=Path, required=True)
            command_parser.add_argument("--log", type=Path, required=True)
            command_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    try:
        if args.mode == "stop":
            stop_owned_session(args.workspace.resolve(strict=True), args.session, args.timeout)
            return 0
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return run_owner(args.workspace, args.session, args.cwd, args.log, command, args.timeout)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[WSL owner] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

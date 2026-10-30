"""Stop only a selected tmux/nohup training tree and wait for checkpoint writes."""
import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


TRAINING_SCRIPTS = {"train_5090.py", "run_training_pipeline.py"}


def process_table():
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                continue
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
            table[int(entry.name)] = {"parent": int(stat[1]), "cwd": (entry / "cwd").resolve(),
                                      "start_time": int(stat[19]),
                                      "scripts": {Path(os.fsdecode(arg)).name for arg in arguments if arg}}
        except (OSError, ValueError, IndexError):
            continue
    return table


def select_training_processes(table, roots, workspace, scripts=TRAINING_SCRIPTS):
    descendants = set(roots)
    while True:
        found = {pid for pid, info in table.items() if info["parent"] in descendants}
        if found <= descendants:
            break
        descendants.update(found)
    candidates = {pid for pid in descendants if pid in table
                  and table[pid]["cwd"].is_relative_to(workspace)
                  and table[pid]["scripts"] & scripts}

    def below_trainer(pid):
        seen = {pid}
        parent = table[pid]["parent"]
        while parent in table and parent not in seen:
            if parent in candidates and "train_5090.py" in table[parent]["scripts"]:
                return True
            seen.add(parent)
            parent = table[parent]["parent"]
        return False

    # Forked DataLoader workers inherit the trainer's full command line. They
    # must stay alive until the trainer finishes its optimizer step and save.
    selected = [pid for pid in candidates if not below_trainer(pid)]
    # Notify the pipeline before its child so it cannot advance to the next stage.
    return sorted(selected, key=lambda pid: ("run_training_pipeline.py" not in table[pid]["scripts"], pid))


def _same_process(current, expected):
    return current is not None and current.get("start_time") == expected.get("start_time")


def _signal_process(pid, expected):
    """Pin the Linux process identity before signaling, even if a PID is reused."""
    descriptor = None
    try:
        use_pidfd = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
        if use_pidfd:
            descriptor = os.pidfd_open(pid)
        current = process_table().get(pid)
        if current is None:
            return
        if not _same_process(current, expected):
            raise RuntimeError(f"PID {pid} was reused during stop; session preserved")
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def stop(roots, workspace, timeout=300, scripts=TRAINING_SCRIPTS):
    if any(pid <= 0 for pid in roots):
        raise ValueError("Process roots must be positive PIDs")
    table = process_table()
    selected = select_training_processes(table, roots, workspace.resolve(), scripts)
    # A stale PID file or a shared tmux name must never turn "no owned
    # trainer" into permission for remote_ctl to kill another session.
    if not selected and any(pid in table for pid in roots):
        raise RuntimeError("No owned primary process found under the live session/PID roots; refusing stop and preserving the session")
    for pid in selected:
        _signal_process(pid, table[pid])
    deadline = time.monotonic() + timeout
    while True:
        current = process_table()
        if not any(_same_process(current.get(pid), table[pid]) for pid in selected):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Training is still saving or stopping; session preserved. Check its log before retrying.")
        time.sleep(0.2)
    return selected


def stop_container(name, timeout):
    def running():
        result = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                                capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "true"
    if not running():
        return
    subprocess.run(["docker", "kill", "--signal", "TERM", name], check=True)
    deadline = time.monotonic() + timeout
    while running():
        if time.monotonic() >= deadline:
            raise RuntimeError("Container is still stopping; it was preserved for inspection")
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--kind", choices=["training", "serve"], default="training")
    parser.add_argument("--container", help="Also gracefully stop this exact named Docker container")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    roots = []
    tmux = subprocess.run(["tmux", "list-panes", "-t", "=" + args.session + ":", "-F", "#{pane_pid}"],
                          capture_output=True, text=True) if shutil.which("tmux") else None
    if tmux is not None and tmux.returncode == 0:
        roots.extend(int(value) for value in tmux.stdout.splitlines())
    if args.pid_file is not None and args.pid_file.exists():
        roots.append(int(args.pid_file.read_text().strip()))
    scripts = TRAINING_SCRIPTS if args.kind == "training" else {"server.app"}
    stopped = stop(roots, args.workspace, args.timeout, scripts)
    if args.container:
        stop_container(args.container, args.timeout)
    print(f"Stopped selected {args.kind} processes: {stopped}", flush=True)


if __name__ == "__main__":
    main()

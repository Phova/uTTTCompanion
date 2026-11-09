"""Refuse Linux/WSL inference startup while this workspace is training.

This is a startup check, not a cross-process GPU reservation or a stop command.
Windows-native serving keeps its existing behavior.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__:
    from .stop_training import TRAINING_SCRIPTS, process_table, select_training_processes
    from .wsl_training_owner import process_identity, same_process
else:
    from stop_training import TRAINING_SCRIPTS, process_table, select_training_processes
    from wsl_training_owner import process_identity, same_process


class TrainingActiveError(RuntimeError):
    pass


def resolve_workspace(path):
    """Frozen sources at WORKSPACE/runs/RUN still share WORKSPACE's GPU."""
    path = Path(path).resolve()
    for candidate in (path, *path.parents):
        if candidate.parent.name == "runs":
            return candidate.parent.parent
    return path


def training_conflicts(workspace, *, table=None, identity=None):
    """Return live native trainers/pipelines and verified WSL startup owners."""
    workspace = resolve_workspace(workspace)
    table = process_table() if table is None else table
    identity = process_identity if identity is None else identity
    conflicts = []
    selected = select_training_processes(table, list(table), workspace)
    for pid in selected:
        current = identity(pid)
        if (current is not None and current.get("start_time") == table[pid]["start_time"]
                and {Path(arg).name for arg in current.get("cmdline", ())} & TRAINING_SCRIPTS):
            conflicts.append(f"training PID {pid}")

    # An owner may be at its launch barrier or between pipeline stages, before
    # the next trainer exists. JSON phase/PID alone cannot establish liveness.
    for path in (workspace / "logs" / ".wsl_owners").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or record.get("phase") not in {"starting", "running"}
                    or record.get("workspace") != str(workspace)
                    or not {Path(arg).name for arg in record.get("command", ())} & TRAINING_SCRIPTS):
                continue
            expected = record["owner"]
            if not isinstance(expected, dict) or type(expected.get("pid")) is not int:
                continue
            if same_process(identity(expected["pid"]), expected):
                conflicts.append(f"WSL training owner {record.get('session', path.stem)} (PID {expected['pid']})")
        except (OSError, ValueError, KeyError, TypeError):
            # Completed or partially written historical records are not a lock.
            continue
    return conflicts


def assert_serving_available(workspace=None):
    if not sys.platform.startswith("linux"):
        return
    root = resolve_workspace(Path.cwd() if workspace is None else workspace)
    conflicts = training_conflicts(root)
    if conflicts:
        raise TrainingActiveError(
            "Refusing inference startup while this workspace is training: "
            + "; ".join(conflicts)
            + ". Wait for training to finish, or save and stop it explicitly before serving."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        assert_serving_available(args.workspace)
    except (OSError, TrainingActiveError) as exc:
        print(f"[Serve guard] {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

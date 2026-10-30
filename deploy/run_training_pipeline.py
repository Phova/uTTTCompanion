"""Persistent pretrain -> instruction tuning pipeline with verifiable stage completion."""
import argparse
import datetime
import json
import os
import pickle
import signal
import subprocess
import sys
from pathlib import Path


def checkpoint_metrics(path):
    import torch
    error = None
    for candidate in (Path(path), Path(str(path) + ".prev")):
        try:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False, mmap=True)
            return {"step": int(checkpoint.get("step", 0)), "metrics": checkpoint.get("metrics", {}),
                    "path": str(candidate)}
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as caught:
            error = caught
    raise RuntimeError(f"No readable stage checkpoint: {path}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261029")
    parser.add_argument("--pretrain-steps", type=int, default=100000)
    parser.add_argument("--sft-steps", type=int, default=6000)
    args = parser.parse_args()
    if args.pretrain_steps < 1 or args.sft_steps < 1:
        parser.error("Training stage steps must be positive")
    if not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.run_id):
        parser.error("run-id may contain only letters, numbers, underscores and hyphens")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    state_path = root / "logs" / f"pipeline_{args.run_id}.json"
    state_path.parent.mkdir(exist_ok=True)
    # This runner targets the Linux/WSL host. Keep one owner of each run's
    # checkpoints even when launched from different terminals or tmux sessions.
    import fcntl
    run_lock = state_path.with_suffix(".lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Training run {args.run_id} is already active") from error
    state = {"run_id": args.run_id, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "pid": os.getpid(),
             "code_revision": os.environ.get("UTTT_CODE_REVISION", "unknown")}
    active_process = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if active_process is not None and active_process.poll() is None:
            active_process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)

    def update(**values):
        state.update(values, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    def run(stage, command, expected_checkpoint, expected_steps):
        nonlocal active_process, stopping
        if stopping:
            raise InterruptedError("Pipeline was stopped; no further stage started")
        print(f"[Pipeline] Starting {stage}: {' '.join(command)}", flush=True)
        update(stage=stage, status="running", command=command, checkpoint=str(expected_checkpoint))
        with (root / "logs" / f"{stage}_{args.run_id}.log").open("a", encoding="utf-8") as stage_log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "RAYON_NUM_THREADS": "4", "UTTT_RUN_ID": args.run_id}) as process:
                active_process = process
                if stopping:
                    process.send_signal(signal.SIGTERM)
                update(child_pid=process.pid)
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    stage_log.write(line)
                    stage_log.flush()
                code = process.wait()
                active_process = None
        if stopping or code in (129, 130, 143, -signal.SIGTERM, -signal.SIGHUP, -signal.SIGINT):
            stopping = True
            update(status="interrupted")
            raise InterruptedError(f"{stage} was stopped; next stage was not started")
        if code != 0:
            raise RuntimeError(f"{stage} exited with code {code}")
        if expected_checkpoint is None:
            update(status="stage_complete")
            return
        saved = checkpoint_metrics(expected_checkpoint)
        if stopping:
            update(status="interrupted")
            raise InterruptedError("Pipeline stopped while checking the checkpoint")
        if saved["step"] < expected_steps:
            update(status="interrupted", saved_step=saved["step"])
            raise RuntimeError(f"{stage} stopped at step {saved['step']} before target {expected_steps}; next stage was not started")
        if saved["metrics"].get("training_stage") != stage or saved["metrics"].get("training_pipeline") != 3:
            raise RuntimeError(f"{stage} checkpoint has incompatible training provenance; next stage was not started")
        update(status="stage_complete", saved_step=saved["step"], last_metrics=saved["metrics"])

    try:
        corpus = root / "data" / f"pretrain_{args.run_id}" / "tokenized" / "manifest.json"
        instructions = root / "data" / f"instruction_{args.run_id}" / "train.jsonl"
        if not corpus.exists() or not instructions.exists():
            raise FileNotFoundError("Prepared corpus or instruction mixture is missing")
        pretrain = root / "checkpoints" / f"uttt_1.584b_{args.run_id}_pretrain.pt"
        sft = root / "checkpoints" / f"uttt_1.584b_{args.run_id}_sft.pt"
        python = sys.executable
        run("pretrain", [python, "-u", "train_5090.py", "--config", f"configs/5090_pretrain_{args.run_id}.yaml",
                         "--data-path", str(corpus), "--steps", str(args.pretrain_steps), "--output-ckpt", str(pretrain)],
            pretrain, args.pretrain_steps)
        best_pretrain = pretrain.with_suffix(".best.pt")
        if not best_pretrain.exists():
            raise FileNotFoundError("Validated best pretraining checkpoint is missing")
        run("sft", [python, "-u", "train_5090.py", "--config", f"configs/5090_sft_{args.run_id}.yaml",
                    "--data-path", str(instructions), "--steps", str(args.sft_steps), "--init-from", str(best_pretrain),
                    "--output-ckpt", str(sft)], sft, args.sft_steps)
        best_sft = sft.with_suffix(".best.pt")
        run("evaluate", [python, "-u", "deploy/evaluate_model.py", "--checkpoint", str(best_sft),
                         "--output", f"logs/quality_{args.run_id}.json"], None, None)
        if stopping:
            raise InterruptedError("Pipeline was stopped")
        update(stage="complete", status="complete", selected_checkpoint=str(best_sft))
        print("TRAINING_PIPELINE_COMPLETE", flush=True)
    except BaseException as error:
        update(status="interrupted" if stopping or state.get("status") == "interrupted" else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()

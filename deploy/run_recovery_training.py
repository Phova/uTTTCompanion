"""Run an isolated SFT recovery experiment and report its heldout quality gate.

This runner never deploys a model. Passing its narrow curriculum evaluation is
not evidence of general assistant quality or learned online memory.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def child_environment(revision, run_id, hardware):
    """Apply allocator policy before a child imports Torch or initializes CUDA."""
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
                   "RAYON_NUM_THREADS": "4", "UTTT_CODE_REVISION": revision, "UTTT_RUN_ID": run_id}
    allocator = hardware.get("cuda_allocator_conf")
    if allocator is not None:
        if not isinstance(allocator, str) or not allocator or any(char in allocator for char in "\0\r\n"):
            raise ValueError("hardware.cuda_allocator_conf must be a nonempty allocator configuration")
        environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        environment["PYTORCH_ALLOC_CONF"] = allocator
    return environment


def load_config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("training"), dict):
        raise ValueError("Config must contain a training mapping")
    return value


def dataset_identity(config, path, stage):
    # Reuse the trainer's deduplication, split and versioned window fingerprint.
    # The constructor does not tokenize or create GPU tensors.
    from train_5090 import PersonaMetaDataset
    settings = config["training"]
    dataset = PersonaMetaDataset(
        None, seq_len=int(settings.get("seq_len", 1024)), data_path=str(path),
        mode=stage, split="train" if settings.get("validation_fraction", 0) else "all",
        validation_fraction=float(settings.get("validation_fraction", 0)),
        split_seed=int(settings.get("seed", 42)), pack_sequences=False,
    )
    return {"fingerprint": dataset.fingerprint, "documents": len(dataset),
            "window_protocol": dataset.SFT_WINDOW_PROTOCOL if stage == "sft" else None}


def checkpoint_info(path):
    import torch
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    optimizer = saved.get("optimizer_state")
    state = saved.get("training_state")
    return {"step": int(saved.get("step", 0)), "metrics": saved.get("metrics", {}),
            "has_optimizer": isinstance(optimizer, dict) and bool(optimizer.get("state"))
                             and bool(optimizer.get("param_groups")),
            "has_training_state": isinstance(state, dict) and "data_cursor" in state and "rng" in state}


def validate_checkpoint(path, *, stage, fingerprint, run_id, revision, target=None, full=True, parameter_scope="all", saved=None):
    saved = checkpoint_info(path) if saved is None else saved
    metrics = saved["metrics"]
    expected = {"training_stage": stage, "training_pipeline": 3,
                "dataset_fingerprint": fingerprint, "run_id": run_id, "code_revision": revision}
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Checkpoint has incompatible stage, data protocol or run provenance: {path}")
    if metrics.get("parameter_scope", "all") != parameter_scope:
        raise RuntimeError(f"Checkpoint uses a different parameter scope: {path}")
    if not math.isfinite(float(metrics.get("total_loss", float("nan")))):
        raise RuntimeError(f"Checkpoint has non-finite training loss: {path}")
    if saved["step"] < 1 or (target is not None and saved["step"] != target):
        raise RuntimeError(f"Checkpoint stopped at step {saved['step']}; expected {target}: {path}")
    if full and not (saved["has_optimizer"] and saved["has_training_state"]):
        raise RuntimeError(f"Full optimizer, data cursor and RNG checkpoint is required: {path}")
    return saved


def resume_checkpoint_source(path, **expected):
    """A readable conflicting checkpoint must never be bypassed by a backup."""
    failures = []
    for candidate in (Path(path), Path(str(path) + ".prev")):
        if not candidate.exists():
            continue
        try:
            saved = checkpoint_info(candidate)
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
            failures.append((candidate, error))
            continue
        # Metadata/scope/full-state validation deliberately occurs outside the
        # read-error handler. An incompatible full is an explicit conflict.
        validate_checkpoint(candidate, saved=saved, **expected)
        return candidate
    if failures:
        raise RuntimeError("No readable full recovery checkpoint; initialization was not attempted") from failures[-1][1]
    return None


def quality_comparison(baseline, candidate, tasks, settings):
    def summary(report):
        rows = report.get("profile_results", {}).get("raw_greedy", report.get("results", []))
        if [row.get("prompt") for row in rows] != [task["prompt"] for task in tasks]:
            raise ValueError("Evaluation report does not match the fixed heldout tasks")
        judged = [row for row, task in zip(rows, tasks) if "accepted" in task]
        if not judged or any(type(row.get("exact_match")) is not bool for row in judged):
            raise ValueError("Heldout evaluation needs accepted answers and exact-match results")
        return {"accuracy": sum(row["exact_match"] for row in judged) / len(judged),
                "accepted_tasks": len(judged),
                "eos_rate": sum(bool(row.get("eos_generated")) for row in rows) / len(rows)}

    before, after = summary(baseline), summary(candidate)
    minimum = float(settings.get("min_accuracy", .5))
    improvement = float(settings.get("min_accuracy_improvement", 0))
    eos_drop = float(settings.get("max_eos_rate_drop", .1))
    if not all(math.isfinite(value) for value in (minimum, improvement, eos_drop)) or not (
            0 <= minimum <= 1 and 0 <= improvement <= 1 and 0 <= eos_drop <= 1):
        raise ValueError("Recovery quality thresholds must be finite fractions in [0, 1]")
    reasons = []
    if after["accuracy"] < minimum:
        reasons.append("heldout_accuracy_below_minimum")
    if after["accuracy"] <= before["accuracy"] + improvement:
        reasons.append("heldout_accuracy_did_not_improve")
    if after["eos_rate"] + eos_drop < before["eos_rate"]:
        reasons.append("eos_rate_regressed")
    return {"quality_passed": not reasons, "reasons": reasons, "baseline": before, "candidate": after,
            "thresholds": {"min_accuracy": minimum, "min_accuracy_improvement": improvement,
                           "max_eos_rate_drop": eos_drop},
            "scope": "This heldout curriculum only; not a general assistant or online-memory benchmark."}


def validate_meta_probe(path, config):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    training = config["training"]
    scope = training.get("parameter_scope", "all")
    if (report.get("status") != "ok" or report.get("finite") is not True
            or scope not in ("all", "memory") or report.get("scope") != scope or len(report.get("steps", [])) < 2
            or not report.get("all_writers_have_nonzero_ce_grad")):
        raise ValueError("Meta requires a successful two-step capacity/gradient probe with the same parameter scope")
    model = {**config.get("model", {}), **config.get("uttt", {})}
    def setting(name, default):
        return training.get(name, model.get(name, default))
    for key, expected in (("seq_len", int(training.get("seq_len", 1024))),
                          ("batch_size", int(training.get("micro_batch_size", 1))),
                          ("chunk_size", int(model.get("chunk_size", 64))),
                          ("gradient_checkpointing", setting("gradient_checkpointing", True)),
                          ("gradient_checkpointing_every", int(setting("gradient_checkpointing_every", 1))),
                          ("meta_checkpoint_layers", int(setting("meta_checkpoint_layers", 0))),
                          ("meta_checkpoint_time_chunks", setting("meta_checkpoint_time_chunks", True)),
                          ("meta_checkpoint_reentrant", setting("meta_checkpoint_reentrant", False))):
        if report.get(key) != expected:
            raise ValueError(f"Meta capacity probe does not match {key}")
    if report.get("surprise_loss_measured", True) is not True:
        raise ValueError("Meta capacity probe omitted the measured surprise graph used by training")
    expected_allocator = config.get("hardware", {}).get("cuda_allocator_conf", "")
    if report.get("allocator_configuration", "") != expected_allocator:
        raise ValueError("Meta capacity probe does not match cuda_allocator_conf")
    expected_protocol = training.get("expected_inner_loop_protocol")
    if expected_protocol and report.get("inner_loop_protocol") != expected_protocol:
        raise ValueError("Meta capacity probe does not match the inner-loop protocol")
    for name in ("d_model", "d_ff_fast", "n_layers", "n_heads", "n_kv_heads", "vocab_size"):
        if name in model and report.get("model_shape", {}).get(name) != model[name]:
            raise ValueError(f"Meta capacity probe does not match model {name}")
    tokenizer_path = training.get("tokenizer_path")
    if tokenizer_path and digest(tokenizer_path) != report.get("tokenizer_sha256"):
        raise ValueError("Meta capacity probe does not match the tokenizer")
    budget = float(config.get("hardware", {}).get("vram_limit_gb", 28)) * 1024 ** 3
    if report.get("peak_allocated_bytes", float("inf")) > budget or report.get("peak_reserved_bytes", float("inf")) > budget:
        raise ValueError("Meta capacity probe exceeded the configured GPU memory budget")
    if not all(step.get("finite") and step.get("all_writers_have_nonzero_ce_grad") for step in report["steps"]):
        raise ValueError("Meta capacity probe has incomplete writer gradients")
    return report


def check_meta_language_retention(sft_quality, meta_quality, settings):
    """End-to-end improvement must not conceal a regression from the new SFT."""
    maximum_drop = float(settings.get("max_meta_accuracy_drop", .05))
    eos_drop = float(settings.get("max_eos_rate_drop", .1))
    if not math.isfinite(maximum_drop) or not 0 <= maximum_drop <= 1:
        raise ValueError("max_meta_accuracy_drop must be a finite fraction in [0, 1]")
    reference, candidate = sft_quality["candidate"], meta_quality["candidate"]
    result = {**meta_quality, "reasons": list(meta_quality["reasons"])}
    if candidate["accuracy"] + maximum_drop < reference["accuracy"]:
        result["reasons"].append("meta_accuracy_regressed_from_sft")
    if candidate["eos_rate"] + eos_drop < reference["eos_rate"]:
        result["reasons"].append("meta_eos_regressed_from_sft")
    result["quality_passed"] = meta_quality["quality_passed"] and not result["reasons"]
    result["sft_retention"] = {"reference": reference, "maximum_accuracy_drop": maximum_drop,
                               "maximum_eos_rate_drop": eos_drop}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-path", "init-from", "output-ckpt", "tasks", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", action="store_true", help="Require the existing full checkpoint; do not initialize weights again")
    parser.add_argument("--meta-config")
    parser.add_argument("--meta-data")
    parser.add_argument("--meta-output-ckpt")
    parser.add_argument("--meta-capacity-probe")
    args = parser.parse_args(argv)
    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.run_id):
        parser.error("run-id may contain only letters, numbers, underscores and hyphens")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    config_path, data_path, initial, output, task_path = (
        Path(value).resolve() for value in (args.config, args.data_path, args.init_from, args.output_ckpt, args.tasks))
    for path in (config_path, data_path, initial, task_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output == initial:
        raise ValueError("Recovery output must differ from initialization checkpoint")
    config = load_config(config_path)
    if config["training"].get("stage") != "sft":
        raise ValueError("Recovery config training.stage must be sft")
    steps = args.steps if args.steps is not None else int(config["training"].get("max_steps", 2000))
    if steps < 1:
        raise ValueError("Recovery training steps must be positive")
    tasks = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tasks or any(not isinstance(task.get("prompt"), str) or not task["prompt"].strip() for task in tasks):
        raise ValueError("Heldout tasks must contain nonempty prompts")
    settings = config.get("recovery", {})
    # Validate quality settings before any GPU work.
    dummy = {"results": [{"prompt": task["prompt"], "exact_match": False, "eos_generated": False} for task in tasks]}
    dummy_quality = quality_comparison(dummy, dummy, tasks, settings)
    check_meta_language_retention(dummy_quality, dummy_quality, settings)
    identity = dataset_identity(config, data_path, "sft")
    revision = os.environ.get("UTTT_CODE_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    meta_values = (args.meta_config, args.meta_data, args.meta_output_ckpt, args.meta_capacity_probe)
    if any(meta_values) and not all(meta_values):
        raise ValueError("All four meta arguments are required together; meta is disabled by default")
    meta = None
    if all(meta_values):
        meta_paths = [Path(value).resolve() for value in meta_values]
        meta_config = load_config(meta_paths[0])
        if meta_config["training"].get("stage") != "meta":
            raise ValueError("Meta config training.stage must be meta")
        if meta_paths[2] in (initial, output):
            raise ValueError("Meta output must be a separate checkpoint")
        validate_meta_probe(meta_paths[3], meta_config)
        memory_heldout = meta_paths[1].parent / "heldout.jsonl"
        memory_manifest = meta_paths[1].parent / "manifest.json"
        for path in (memory_heldout, memory_manifest):
            if not path.is_file():
                raise FileNotFoundError(f"Meta requires a separate heldout curriculum and manifest: {path}")
        manifest = json.loads(memory_manifest.read_text(encoding="utf-8"))
        if (manifest.get("files", {}).get("train", {}).get("file") != meta_paths[1].name
                or manifest.get("files", {}).get("heldout", {}).get("file") != memory_heldout.name
                or manifest["files"]["train"].get("sha256") != digest(meta_paths[1])
                or manifest["files"]["heldout"].get("sha256") != digest(memory_heldout)):
            raise ValueError("Meta training and heldout files must match their curriculum manifest")
        meta = {"config": meta_paths[0], "data": meta_paths[1], "output": meta_paths[2],
                "config_sha256": digest(meta_paths[0]), "data_sha256": digest(meta_paths[1]),
                "probe_sha256": digest(meta_paths[3]),
                "heldout": memory_heldout, "heldout_sha256": digest(memory_heldout),
                "manifest": memory_manifest, "manifest_sha256": digest(memory_manifest),
                "steps": int(meta_config["training"].get("max_steps", 2000)),
                "parameter_scope": meta_config["training"].get("parameter_scope", "all"),
                "identity": dataset_identity(meta_config, meta_paths[1], "meta")}
        if meta["steps"] < 1:
            raise ValueError("Meta training steps must be positive")
    run_dir = root / "logs" / f"recovery_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (run_dir / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    state_path = run_dir / "state.json"
    binding = {"run_id": args.run_id, "config": str(config_path), "config_sha256": digest(config_path),
               "data_path": str(data_path), "data_sha256": digest(data_path), "dataset": identity,
               "init_from": str(initial), "output_checkpoint": str(output), "steps": steps,
               "tasks": str(task_path), "tasks_sha256": digest(task_path), "code_revision": revision,
               "meta": ({key: str(value) if isinstance(value, Path) else value for key, value in meta.items()} if meta else None)}
    active = None
    stopping = False
    handlers = {}
    state = None

    def update(**values):
        state.update(values, updated_at=utc_now())
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)

    def check_stop():
        if stopping:
            raise InterruptedError("Recovery pipeline stopped; no further stage was started")

    def run(stage, command):
        nonlocal active, stopping
        check_stop()
        record = {"stage": stage, "started_at": utc_now(), "status": "running", "command": command,
                  "log": str(run_dir / f"{len(state['stages']) + 1:03d}_{stage}.log")}
        state["stages"].append(record)
        update(stage=stage, status="running")
        print(f"[Recovery] Starting {stage}: {' '.join(command)}", flush=True)
        stage_hardware = (meta_config if meta and stage.startswith(("meta", "memory")) else config).get("hardware", {})
        with Path(record["log"]).open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, env=child_environment(revision, args.run_id, stage_hardware)) as process:
                active = process
                if stopping:
                    process.send_signal(signal.SIGTERM)
                update(child_pid=process.pid)
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
                code = process.wait()
                active = None
        record.update(finished_at=utc_now(), exit_code=code)
        if stopping or code in (129, 130, 143, -signal.SIGTERM, -signal.SIGINT):
            stopping = True
            record["status"] = "interrupted"
            raise InterruptedError(f"{stage} stopped; its next stage was not started")
        if code:
            record["status"] = "failed"
            raise RuntimeError(f"{stage} exited with code {code}")
        record["status"] = "complete"
        update(child_pid=None)
        check_stop()

    def report_path(stage):
        # A failed evaluation can leave a partial JSON file. Its next attempt
        # gets a new name, preserving both the old evidence and the new result.
        return run_dir / f"{len(state['stages']) + 1:03d}_{stage}.json"

    def evaluate(stage, checkpoint):
        report = report_path(stage)
        run(stage, [sys.executable, "-u", "deploy/evaluate_model.py", "--checkpoint", str(checkpoint),
                    "--tasks", str(task_path), "--output", str(report), "--mode", "chat", "--seed", "42",
                    "--greedy-only",
                    "--max-new-tokens", str(int(settings.get("eval_max_new_tokens", 64)))])
        return json.loads(report.read_text(encoding="utf-8")), report

    def evaluate_memory(key, checkpoint, *, allow_untrained=False):
        previous = state.get("memory_evaluations", {}).get(key)
        if args.resume and previous:
            path = Path(previous["report"])
            if digest(path) != previous["sha256"]:
                raise ValueError("A recorded memory evaluation report changed")
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("checkpoint_sha256") != digest(checkpoint):
                raise ValueError("The checkpoint for a recorded memory evaluation changed")
        else:
            stage = "memory_baseline_evaluate" if allow_untrained else "meta_memory_evaluate"
            path = report_path(stage)
            command = [sys.executable, "-u", "deploy/evaluate_memory.py", "--checkpoint", str(checkpoint),
                       "--data-path", str(meta["heldout"]), "--output", str(path), "--max-samples", "64"]
            if allow_untrained:
                command.append("--allow-untrained-baseline")
            run(stage, command)
            report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("data_sha256") != meta["heldout_sha256"]
                or report.get("free_generation_accuracy_measured") is not False
                or report.get("online_writes_allowed") is not False
                or (not allow_untrained and report.get("meta_trained") is not True)):
            raise ValueError("Memory report must use this heldout set and remain a teacher-forced diagnostic")
        evaluations = dict(state.get("memory_evaluations", {}))
        evaluations[key] = {"report": str(path), "sha256": digest(path),
                            "checkpoint_sha256": report.get("checkpoint_sha256"),
                            "data_sha256": report["data_sha256"], "summary": report.get("summary"),
                            "groups": report.get("groups"), "free_generation_accuracy_measured": False,
                            "online_writes_allowed": False}
        update(memory_evaluations=evaluations, online_memory_quality_verified=False,
               online_writes_allowed=False)

    try:
        resume_source = None
        if args.resume:
            if not state_path.is_file():
                raise ValueError("--resume requires this run's existing state manifest")
            previous_state = json.loads(state_path.read_text(encoding="utf-8"))
            if previous_state.get("binding") != binding:
                raise ValueError("Resume config, data, tasks, code or output differs from the original run")
            state = previous_state
            resume_source = resume_checkpoint_source(output, stage="sft", fingerprint=identity["fingerprint"],
                                                     run_id=args.run_id, revision=revision)
            if resume_source is None and (state.get("training_complete") or state.get("saved_step")
                    or output.with_suffix(".best.pt").exists()
                    or any(stage.get("stage") == "sft" and stage.get("status") == "complete"
                           for stage in state.get("stages", []))):
                raise RuntimeError("The full SFT checkpoint is missing after training/save evidence; refusing to reinitialize or substitute best/previous weights")
        else:
            if state_path.exists() or any(Path(str(output) + suffix).exists() for suffix in ("", ".prev")) or output.with_suffix(".best.pt").exists():
                raise FileExistsError("New recovery run refuses existing state or checkpoints; use --resume for this exact run")
            if meta and (meta["output"].exists() or Path(str(meta["output"]) + ".prev").exists()
                         or meta["output"].with_suffix(".best.pt").exists()):
                raise FileExistsError("New recovery run refuses an existing meta checkpoint")
            state = {"protocol": "uttt-recovery-run-v1", "binding": binding, "created_at": utc_now(),
                     "stages": [], "training_complete": False, "quality_passed": False,
                     "online_writes_allowed": False}
        update(pid=os.getpid(), status="starting", online_writes_allowed=False,
               resume_action=(("full_checkpoint" if resume_source == output else "previous_full_checkpoint") if resume_source else
                              "initialization_retry_no_checkpoint" if args.resume else "new_initialization"),
               resume_checkpoint=str(resume_source) if resume_source else None)
        for signum in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", signal.SIGTERM)):
            if signum not in handlers:
                handlers[signum] = signal.signal(signum, stop)
        baseline_path = Path(state.get("baseline_report", str(run_dir / "baseline.json")))
        if args.resume and state.get("baseline_sha256"):
            if digest(baseline_path) != state["baseline_sha256"]:
                raise ValueError("The baseline report changed; refusing a different comparison")
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            baseline, baseline_path = evaluate("baseline_evaluate", initial)
            update(baseline_report=str(baseline_path), baseline_sha256=digest(baseline_path))
        command = [sys.executable, "-u", "train_5090.py", "--config", str(config_path), "--data-path", str(data_path),
                   "--steps", str(steps), "--output-ckpt", str(output)]
        if resume_source is None:
            command.extend(["--init-from", str(initial), "--no-resume"])
        run("sft", command)
        saved = validate_checkpoint(output, stage="sft", fingerprint=identity["fingerprint"],
                                    run_id=args.run_id, revision=revision, target=steps)
        check_stop()
        best = output.with_suffix(".best.pt")
        validate_checkpoint(best, stage="sft", fingerprint=identity["fingerprint"],
                            run_id=args.run_id, revision=revision, full=False)
        update(training_complete=True, saved_step=saved["step"])
        candidate, candidate_path = evaluate("sft_evaluate", best)
        comparison = quality_comparison(baseline, candidate, tasks, settings)
        update(quality=comparison, quality_passed=comparison["quality_passed"], selected_checkpoint=str(best),
               sft_quality_report=str(candidate_path), sft_quality_sha256=digest(candidate_path))
        if meta and comparison["quality_passed"]:
            evaluate_memory("baseline", best, allow_untrained=True)
            meta_command = [sys.executable, "-u", "train_5090.py", "--config", str(meta["config"]),
                            "--data-path", str(meta["data"]), "--steps", str(meta["steps"]),
                            "--output-ckpt", str(meta["output"])]
            meta_resume_source = (resume_checkpoint_source(
                meta["output"], stage="meta", fingerprint=meta["identity"]["fingerprint"],
                run_id=args.run_id, revision=revision, parameter_scope=meta["parameter_scope"])
                if args.resume else None)
            if meta_resume_source is None:
                if (meta["output"].with_suffix(".best.pt").exists() or state.get("meta_training_complete")
                        or any(stage.get("stage") == "meta" and stage.get("status") == "complete"
                               for stage in state.get("stages", []))):
                    raise RuntimeError("The full meta checkpoint is missing after training/save evidence; refusing weight-only reinitialization")
                meta_command.extend(["--init-from", str(best), "--no-resume"])
            update(meta_resume_checkpoint=str(meta_resume_source) if meta_resume_source else None)
            run("meta", meta_command)
            validate_checkpoint(meta["output"], stage="meta", fingerprint=meta["identity"]["fingerprint"],
                                run_id=args.run_id, revision=revision, target=meta["steps"],
                                parameter_scope=meta["parameter_scope"])
            meta_best = meta["output"].with_suffix(".best.pt")
            validate_checkpoint(meta_best, stage="meta", fingerprint=meta["identity"]["fingerprint"],
                                run_id=args.run_id, revision=revision, full=False,
                                parameter_scope=meta["parameter_scope"])
            meta_report, meta_report_path = evaluate("meta_evaluate", meta_best)
            meta_quality = quality_comparison(baseline, meta_report, tasks, settings)
            meta_quality = check_meta_language_retention(comparison, meta_quality, settings)
            update(meta_quality=meta_quality, quality_passed=meta_quality["quality_passed"],
                   meta_quality_report=str(meta_report_path), meta_quality_sha256=digest(meta_report_path),
                   selected_checkpoint=str(meta_best if meta_quality["quality_passed"] else best), online_writes_allowed=False,
                   meta_training_complete=True, online_memory_quality_verified=False,
                   online_writes_blocked_reason="Separate memory heldout evaluation of dynamic versus frozen state is required")
            evaluate_memory("meta", meta_best)
            update(online_writes_blocked_reason="Teacher-forced memory CE comparisons do not establish free-generation memory quality or authorize online writes")
        elif meta:
            update(meta_skipped_reason="SFT heldout quality gate failed")
        check_stop()
        update(stage="complete", status="complete" if state["quality_passed"] else "quality_failed")
        print("RECOVERY_TRAINING_COMPLETE " + json.dumps({"training_complete": state["training_complete"],
              "quality_passed": state["quality_passed"], "online_writes_allowed": state["online_writes_allowed"]}), flush=True)
        return 0
    except InterruptedError as error:
        if not stopping:
            if state is not None:
                update(status="failed", error=str(error), online_writes_allowed=False)
            raise
        update(status="interrupted", error=str(error), online_writes_allowed=False, child_pid=None)
        print(f"[Recovery] Interrupted: {error}", flush=True)
        return 0
    except BaseException as error:
        if state is not None:
            update(status="failed", error=str(error), online_writes_allowed=False)
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

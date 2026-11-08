"""Recovery runs keep checkpoints, stage completion and heldout quality separate."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy import run_recovery_training as runner


class FakeProcess:
    pid = 54321

    def __init__(self, lines=(), code=0):
        self.stdout = iter(lines)
        self.code = code
        self.sent_signals = []

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def poll(self):
        return None

    def send_signal(self, signum):
        self.sent_signals.append(signum)

    def wait(self):
        return self.code


class RecoveryDatasetIdentityTests(unittest.TestCase):
    def test_grouped_identity_matches_the_trainer_and_changes_legacy_fingerprint(self):
        try:
            from train_5090 import PersonaMetaDataset
        except ImportError:
            self.skipTest("Trainer's PyTorch dependencies unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grouped.jsonl"
            rows = [{"text": f"<|user|> Question {group}, form {form}?<|assistant|> Answer {group}.<|eos|>",
                     "validation_group_id": f"semantic-{group}"}
                    for group in range(40) for form in range(3)]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            training = {"seq_len": 64, "validation_fraction": .25, "seed": 271828,
                        "validation_group_key": "validation_group_id"}
            expected = PersonaMetaDataset(None, seq_len=64, data_path=str(path), mode="sft", split="train",
                                          validation_fraction=.25, split_seed=271828, pack_sequences=False,
                                          validation_group_key="validation_group_id")
            actual = runner.dataset_identity({"training": training}, path, "sft")
            self.assertEqual(actual, {"fingerprint": expected.fingerprint, "documents": len(expected),
                                      "window_protocol": expected.SFT_WINDOW_PROTOCOL})
            legacy = PersonaMetaDataset(None, seq_len=64, data_path=str(path), mode="sft", split="train",
                                        validation_fraction=.25, split_seed=271828, pack_sequences=False)
            self.assertNotEqual(actual["fingerprint"], legacy.fingerprint)


class RecoveryLifecycleTests(unittest.TestCase):
    def test_meta_improvement_over_old_model_cannot_hide_sft_regression(self):
        sft = {"candidate": {"accuracy": .8, "eos_rate": 1.0}}
        improved_over_old = {"quality_passed": True, "reasons": [],
                             "candidate": {"accuracy": .5, "eos_rate": .95}}
        result = runner.check_meta_language_retention(sft, improved_over_old, {})
        self.assertFalse(result["quality_passed"])
        self.assertIn("meta_accuracy_regressed_from_sft", result["reasons"])
        self.assertEqual(improved_over_old["reasons"], [])
        improved_over_old["candidate"]["accuracy"] = .78
        self.assertTrue(runner.check_meta_language_retention(sft, improved_over_old, {})["quality_passed"])
        with self.assertRaisesRegex(ValueError, "max_meta_accuracy_drop"):
            runner.check_meta_language_retention(sft, improved_over_old, {"max_meta_accuracy_drop": float("nan")})

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / "deploy").mkdir()
        self.initial = self.root / "old_sft.best.pt"
        self.initial.touch()
        self.config = self.root / "recovery.yaml"
        self.config.write_text("config", encoding="utf-8")
        self.data = self.root / "new_data.jsonl"
        self.data.write_text("data", encoding="utf-8")
        self.tasks_path = self.root / "heldout.jsonl"
        self.tasks = [{"prompt": f"question {i}", "accepted": [str(i)]} for i in range(4)]
        self.tasks_path.write_text("\n".join(json.dumps(task) for task in self.tasks), encoding="utf-8")
        self.output = self.root / "new_sft.pt"
        self.meta_output = self.root / "meta.pt"
        self.argv = ["--config", str(self.config), "--data-path", str(self.data), "--init-from", str(self.initial),
                     "--output-ckpt", str(self.output), "--tasks", str(self.tasks_path), "--steps", "10", "--run-id", "test"]
        self.settings = {"training": {"stage": "sft", "max_steps": 10}}
        self.info = {"step": 10, "metrics": {"training_stage": "sft", "training_pipeline": 3,
                    "dataset_fingerprint": "newdata", "run_id": "test", "code_revision": "revision",
                    "total_loss": 1.2}, "has_optimizer": True, "has_training_state": True}
        self.handlers = {}
        self.stack.enter_context(patch.object(runner, "__file__", str(self.root / "deploy/run_recovery_training.py")))
        self.stack.enter_context(patch.object(runner, "load_config", return_value=self.settings))
        self.stack.enter_context(patch.object(runner, "dataset_identity", return_value={
            "fingerprint": "newdata", "documents": 30, "window_protocol": "conditioned_turn_v1"}))
        self.stack.enter_context(patch.object(runner, "checkpoint_info", side_effect=self.saved_info))
        self.stack.enter_context(patch.dict(os.environ, {"UTTT_CODE_REVISION": "revision"}))
        self.stack.enter_context(patch.dict(sys.modules, {"fcntl": SimpleNamespace(
            LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)}))
        self.stack.enter_context(patch.object(runner.signal, "signal", side_effect=self.register_handler))
        self.console = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.baseline_correct = 0
        self.candidate_correct = 3
        self.final_correct = None
        self.final_eos = True
        self.processes = []
        self.popen = self.stack.enter_context(patch.object(runner.subprocess, "Popen", side_effect=self.spawn))

    def register_handler(self, signum, handler):
        previous = self.handlers.get(signum)
        self.handlers[signum] = handler
        return previous

    def spawn(self, command, **kwargs):
        if "deploy/evaluate_model.py" in command:
            report_path = Path(command[command.index("--output") + 1])
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            correct = self.baseline_correct if checkpoint == self.initial else self.candidate_correct
            if checkpoint == self.output and self.final_correct is not None:
                correct = self.final_correct
            report_path.write_text(json.dumps(self.report(correct, eos=self.final_eos if checkpoint == self.output else True)),
                                   encoding="utf-8")
        elif "deploy/evaluate_memory.py" in command:
            report_path = Path(command[command.index("--output") + 1])
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            data = Path(command[command.index("--data-path") + 1])
            report_path.write_text(json.dumps({
                "checkpoint_sha256": runner.digest(checkpoint), "data_sha256": runner.digest(data),
                "meta_trained": "--allow-untrained-baseline" not in command,
                "free_generation_accuracy_measured": False, "online_writes_allowed": False,
                "summary": {"paired_ce_gain": .1}, "groups": {}}), encoding="utf-8")
        elif "deploy/evaluate_chat_curriculum.py" in command:
            from deploy.evaluate_chat_curriculum import validate_tasks
            report_path = Path(command[command.index("--output") + 1])
            task_path = Path(command[command.index("--tasks") + 1])
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            with task_path.open(encoding="utf-8") as stream:
                development = validate_tasks([json.loads(line) for line in stream if line.strip()])
            identity = runner.hashlib.sha256(json.dumps(development, ensure_ascii=False, sort_keys=True,
                                                        separators=(",", ":")).encode()).hexdigest()
            report_path.write_text(json.dumps({
                "protocol": "chat-curriculum-diagnostics-v1", "checkpoint": str(checkpoint), "tasks_sha256": identity,
                "results": [{"id": task["id"], "automatic_pass": None if task["category"] == "chat" else True}
                            for task in development],
                "summary": {"category:chat": {"manual_review_required": 1, "automatic_checks": 0}}}), encoding="utf-8")
        elif "train_5090.py" in command:
            output = Path(command[command.index("--output-ckpt") + 1])
            output.touch()
            output.with_suffix(".best.pt").touch()
        process = FakeProcess()
        self.processes.append((command, process))
        return process

    def saved_info(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        if str(path).startswith(str(self.meta_output.with_suffix(""))):
            info = json.loads(json.dumps(self.info))
            info["metrics"].update(training_stage="meta", parameter_scope="memory")
            return info
        return self.info

    def meta_arguments(self):
        directory = self.root / "memory"
        directory.mkdir()
        config, data, heldout, manifest, probe = (directory / name for name in
                                                ("meta.yaml", "train.jsonl", "heldout.jsonl", "manifest.json", "probe.json"))
        config.write_text("meta-config", encoding="utf-8")
        data.write_text("memory-train", encoding="utf-8")
        heldout.write_text("memory-heldout", encoding="utf-8")
        manifest.write_text(json.dumps({"files": {
            "train": {"file": data.name, "sha256": runner.digest(data)},
            "heldout": {"file": heldout.name, "sha256": runner.digest(heldout)}}}), encoding="utf-8")
        probe.write_text("probe", encoding="utf-8")
        meta_settings = {"training": {"stage": "meta", "parameter_scope": "memory", "max_steps": 10}}
        self.stack.enter_context(patch.object(runner, "load_config", side_effect=lambda path:
                                             meta_settings if Path(path) == config else self.settings))
        self.stack.enter_context(patch.object(runner, "validate_meta_probe", return_value={"status": "ok"}))
        return ["--meta-config", str(config), "--meta-data", str(data), "--meta-output-ckpt", str(self.meta_output),
                "--meta-capacity-probe", str(probe)]

    def report(self, correct, eos=True):
        return {"results": [{"prompt": task["prompt"], "exact_match": i < correct,
                             "eos_generated": eos} for i, task in enumerate(self.tasks)]}

    def state(self):
        return json.loads((self.root / "logs/recovery_test/state.json").read_text(encoding="utf-8"))

    def stop(self):
        self.handlers[signal.SIGTERM](signal.SIGTERM, None)

    def test_complete_sft_reports_quality_without_enabling_online_writes(self):
        self.assertEqual(runner.main(self.argv), 0)
        state = self.state()
        self.assertEqual(state["status"], "complete")
        self.assertTrue(state["training_complete"])
        self.assertTrue(state["quality_passed"])
        self.assertFalse(state["online_writes_allowed"])
        self.assertEqual([stage["stage"] for stage in state["stages"]], ["baseline_evaluate", "sft", "sft_evaluate"])
        self.assertTrue(all(stage["status"] == "complete" for stage in state["stages"]))
        command = self.processes[1][0]
        self.assertIn("--init-from", command)
        self.assertIn("--no-resume", command)
        self.assertNotIn("development_tasks", state["binding"])
        self.assertNotIn("development_evaluations", state)

    def development_arguments(self):
        self.development_path = self.root / "development.jsonl"
        rows = [{"id": "chat-one", "category": "chat", "task": "daily", "language": "zh", "family": "dev:chat",
                 "prompt": "第一行\u2028仍在同一JSON字符串内。"},
                {"id": "code-one", "category": "coding", "task": "function", "language": "en", "family": "dev:code",
                 "prompt": "Define f(x) returning x.", "code_check": {"protocol": "bounded_python_ast_v1",
                    "function": "f", "cases": [{"args": [2], "expected": 2}]}}]
        self.development_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return ["--development-tasks", str(self.development_path)]

    def test_optional_development_records_identity_after_old_gate_without_changing_quality(self):
        self.final_candidate_monitor()
        runner.main(self.argv + self.development_arguments())
        state = self.state()
        self.assertEqual(state["binding"]["development_tasks"], {
            "path": str(self.development_path), "sha256": runner.digest(self.development_path)})
        self.assertEqual([entry["stage"] for entry in state["stages"]], [
            "baseline_evaluate", "development_baseline_evaluate", "sft", "sft_evaluate", "sft_final_evaluate",
            "development_best_ce_evaluate", "development_final_evaluate"])
        self.assertEqual(set(state["development_evaluations"]), {"baseline", "best_ce", "final"})
        self.assertEqual(state["quality"]["candidate"]["accuracy"], 14 / 60)
        self.assertFalse(state["quality_passed"])
        self.assertFalse(state["online_writes_allowed"])
        for record in state["development_evaluations"].values():
            self.assertEqual(record["report_sha256"], runner.digest(record["report"]))
            self.assertEqual(record["checkpoint_sha256"], runner.digest(record["checkpoint"]))
            report = json.loads(Path(record["report"]).read_text(encoding="utf-8"))
            self.assertIsNone(report["results"][0]["automatic_pass"])
            self.assertEqual(record["summary"]["category:chat"]["automatic_checks"], 0)

    def test_resume_reuses_pinned_development_reports(self):
        self.final_candidate_monitor()
        extra = self.development_arguments()
        runner.main(self.argv + extra)
        records = self.state()["development_evaluations"]
        contents = {Path(value["report"]): Path(value["report"]).read_bytes() for value in records.values()}
        self.processes.clear()
        runner.main(self.argv + extra + ["--resume"])
        self.assertEqual(len(self.processes), 1)
        self.assertIn("train_5090.py", self.processes[0][0])
        self.assertEqual(self.state()["development_evaluations"], records)
        for path, content in contents.items():
            self.assertEqual(path.read_bytes(), content)

    def test_changed_development_binding_is_rejected_before_resume_launch(self):
        extra = self.development_arguments()
        runner.main(self.argv + extra)
        state_path = self.root / "logs/recovery_test/state.json"
        original = state_path.read_bytes()
        with self.development_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        self.processes.clear()
        with self.assertRaisesRegex(ValueError, "differs from the original run"):
            runner.main(self.argv + extra + ["--resume"])
        self.assertEqual(self.processes, [])
        self.assertEqual(state_path.read_bytes(), original)

    def test_changed_development_report_is_not_silently_recomputed(self):
        extra = self.development_arguments()
        runner.main(self.argv + extra)
        path = Path(self.state()["development_evaluations"]["baseline"]["report"])
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.processes.clear()
        with self.assertRaisesRegex(ValueError, "development evaluation report changed"):
            runner.main(self.argv + extra + ["--resume"])
        self.assertEqual(self.processes, [])
        self.assertEqual(self.state()["status"], "failed")

    def test_development_baseline_stop_retries_unique_report_before_training(self):
        extra = self.development_arguments()
        original = self.spawn
        def interrupted(command, **kwargs):
            process = original(command, **kwargs)
            if "deploy/evaluate_chat_curriculum.py" in command:
                def lines():
                    self.stop()
                    yield "development baseline interrupted\n"
                process.stdout = lines()
            return process
        self.popen.side_effect = interrupted
        runner.main(self.argv + extra)
        state = self.state()
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["stage"], "development_baseline_evaluate")
        self.assertFalse(self.output.exists())
        old_path = Path(state["stages"][-1]["log"]).with_suffix(".json")
        contents = old_path.read_bytes()
        self.processes.clear()
        self.popen.side_effect = original
        runner.main(self.argv + extra + ["--resume"])
        self.assertIn("deploy/evaluate_chat_curriculum.py", self.processes[0][0])
        self.assertIn("--init-from", self.processes[1][0])
        self.assertEqual(old_path.read_bytes(), contents)
        self.assertNotEqual(self.state()["development_evaluations"]["baseline"]["report"], str(old_path))

    def test_invalid_development_tasks_fail_before_any_gpu_child(self):
        extra = self.development_arguments()
        self.development_path.write_text('{"category":"coding","task":"bad","language":"en","family":"dev","prompt":"Bad"}\n',
                                         encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Coding tasks require"):
            runner.main(self.argv + extra)
        self.popen.assert_not_called()

    def test_training_completion_is_distinct_from_failed_quality(self):
        self.candidate_correct = 1
        runner.main(self.argv)
        self.assertTrue(self.state()["training_complete"])
        self.assertFalse(self.state()["quality_passed"])
        self.assertEqual(self.state()["status"], "quality_failed")
        self.assertFalse(self.state()["online_writes_allowed"])

    def final_candidate_monitor(self, *, baseline=2, best=2, final=14):
        self.settings["recovery"] = {"evaluate_final_checkpoint": True}
        self.tasks = [{"prompt": f"question {i}", "accepted": [str(i)]} for i in range(60)]
        self.tasks_path.write_text("\n".join(json.dumps(task) for task in self.tasks), encoding="utf-8")
        self.baseline_correct, self.candidate_correct, self.final_correct = baseline, best, final

    def test_final_improvement_is_retained_without_bypassing_meta_gate(self):
        self.final_candidate_monitor()
        runner.main(self.argv + self.meta_arguments())
        state = self.state()
        self.assertEqual(state["selected_checkpoint"], str(self.output))
        self.assertEqual(state["sft_selection"]["selection_status"], "intermediate_only")
        self.assertFalse(state["sft_selection"]["may_start_meta"])
        self.assertFalse(state["quality_passed"])
        self.assertEqual(state["quality"]["candidate"]["accuracy"], 14 / 60)
        self.assertEqual(state["status"], "quality_failed")
        self.assertEqual([entry["stage"] for entry in state["stages"]],
                         ["baseline_evaluate", "sft", "sft_evaluate", "sft_final_evaluate"])
        self.assertFalse(any("deploy/evaluate_memory.py" in command for command, _ in self.processes))
        self.assertEqual(set(state["sft_candidates"]), {"best_ce", "final"})
        for candidate in state["sft_candidates"].values():
            self.assertEqual(candidate["checkpoint_sha256"], runner.digest(candidate["checkpoint"]))
            self.assertEqual(candidate["report_sha256"], runner.digest(candidate["report"]))
            self.assertEqual(candidate["code_revision"], "revision")
            self.assertEqual(candidate["step"], 10)

    def test_passing_final_candidate_supplies_meta_initialization_and_retention_reference(self):
        self.final_candidate_monitor(final=31)
        runner.main(self.argv + self.meta_arguments())
        state = self.state()
        self.assertEqual(state["sft_selection"]["selection_status"], "gate_passed")
        self.assertTrue(state["sft_selection"]["may_start_meta"])
        self.assertEqual(state["sft_selected_checkpoint"], str(self.output))
        memory = next(command for command, _ in self.processes if "--allow-untrained-baseline" in command)
        self.assertEqual(memory[memory.index("--checkpoint") + 1], str(self.output))
        meta = next(command for command, _ in self.processes
                    if "train_5090.py" in command and command[command.index("--output-ckpt") + 1] == str(self.meta_output))
        self.assertEqual(meta[meta.index("--init-from") + 1], str(self.output))
        self.assertIn("--no-resume", meta)
        self.assertEqual(state["meta_quality"]["sft_retention"]["reference"]["accuracy"], 31 / 60)
        # Fake meta stays at the old best's score and fails; retain the selected
        # final SFT candidate, rather than accidentally falling back to best CE.
        self.assertFalse(state["meta_quality"]["quality_passed"])
        self.assertEqual(state["selected_checkpoint"], str(self.output))

    def test_higher_final_accuracy_cannot_hide_eos_regression(self):
        self.final_candidate_monitor(final=31)
        self.final_eos = False
        runner.main(self.argv + self.meta_arguments())
        state = self.state()
        self.assertEqual(state["sft_selection"]["selected_candidate"], "best_ce")
        self.assertEqual(state["selected_checkpoint"], str(self.output.with_suffix(".best.pt")))
        self.assertIn("eos_rate_regressed", state["sft_candidates"]["final"]["quality"]["reasons"])
        self.assertFalse(state["sft_selection"]["may_start_meta"])
        self.assertEqual(len(self.processes), 4)

    def test_candidate_accuracy_ties_prefer_best_ce(self):
        self.settings["recovery"] = {"evaluate_final_checkpoint": True}
        self.final_correct = self.candidate_correct
        runner.main(self.argv)
        state = self.state()
        self.assertEqual(state["sft_selection"]["eligible_candidates"], ["best_ce", "final"])
        self.assertEqual(state["sft_selection"]["selected_candidate"], "best_ce")
        self.assertTrue(state["quality_passed"])

    def test_no_nonregressing_candidate_keeps_best_ce_as_unapproved_fallback(self):
        self.final_candidate_monitor(baseline=20, best=2, final=14)
        runner.main(self.argv)
        self.assertEqual(self.state()["sft_selection"]["selection_status"], "fallback_only")
        self.assertEqual(self.state()["sft_selection"]["safe_intermediate_candidates"], [])
        self.assertEqual(self.state()["sft_selection"]["selected_candidate"], "best_ce")
        self.assertFalse(self.state()["quality_passed"])

    def test_resume_reuses_both_pinned_candidate_evaluations(self):
        self.final_candidate_monitor()
        runner.main(self.argv)
        records = self.state()["sft_candidates"]
        contents = {Path(value["report"]): Path(value["report"]).read_bytes() for value in records.values()}
        self.processes.clear()
        runner.main(self.argv + ["--resume"])
        self.assertEqual(len(self.processes), 1)
        self.assertIn("train_5090.py", self.processes[0][0])
        self.assertEqual(self.state()["sft_candidates"], records)
        self.assertEqual(self.state()["selected_checkpoint"], str(self.output))
        for path, value in contents.items():
            self.assertEqual(path.read_bytes(), value)

    def test_changed_cached_candidate_report_is_not_silently_reevaluated(self):
        self.final_candidate_monitor()
        runner.main(self.argv)
        report = Path(self.state()["sft_candidates"]["best_ce"]["report"])
        report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.processes.clear()
        with self.assertRaisesRegex(ValueError, "candidate evaluation report changed"):
            runner.main(self.argv + ["--resume"])
        self.assertEqual(len(self.processes), 1)
        self.assertEqual(self.state()["status"], "failed")

    def test_interrupted_final_evaluation_reuses_best_and_preserves_partial_report(self):
        self.final_candidate_monitor(final=31)
        spawn = self.spawn

        def interrupted_final(command, **kwargs):
            process = spawn(command, **kwargs)
            if ("deploy/evaluate_model.py" in command
                    and command[command.index("--checkpoint") + 1] == str(self.output)):
                def lines():
                    self.stop()
                    yield "final candidate evaluation interrupted\n"
                process.stdout = lines()
            return process

        self.popen.side_effect = interrupted_final
        runner.main(self.argv)
        interrupted = self.state()
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(set(interrupted["sft_candidates"]), {"best_ce"})
        old_report = Path(interrupted["stages"][-1]["log"]).with_suffix(".json")
        old_contents = old_report.read_bytes()
        self.processes.clear()
        self.popen.side_effect = spawn
        runner.main(self.argv + ["--resume"])
        self.assertEqual(len(self.processes), 2)
        self.assertIn("train_5090.py", self.processes[0][0])
        self.assertEqual(self.processes[1][0][self.processes[1][0].index("--checkpoint") + 1], str(self.output))
        self.assertEqual(old_report.read_bytes(), old_contents)
        self.assertNotEqual(self.state()["sft_candidates"]["final"]["report"], str(old_report))
        self.assertEqual(self.state()["selected_checkpoint"], str(self.output))
        self.assertTrue(self.state()["quality_passed"])

    def test_changed_cached_candidate_weights_are_rejected(self):
        self.final_candidate_monitor()
        runner.main(self.argv)
        self.output.write_bytes(b"different weights with the same mocked metadata")
        self.processes.clear()
        with self.assertRaisesRegex(ValueError, "candidate checkpoint changed"):
            runner.main(self.argv + ["--resume"])
        self.assertEqual(len(self.processes), 1)

    def test_checkpoint_change_during_candidate_evaluation_is_rejected(self):
        self.final_candidate_monitor()
        spawn = self.spawn

        def change_weights(command, **kwargs):
            result = spawn(command, **kwargs)
            if ("deploy/evaluate_model.py" in command
                    and command[command.index("--checkpoint") + 1] == str(self.output.with_suffix(".best.pt"))):
                self.output.with_suffix(".best.pt").write_bytes(b"replaced during evaluation")
            return result

        self.popen.side_effect = change_weights
        with self.assertRaisesRegex(ValueError, "changed during evaluation"):
            runner.main(self.argv)
        self.assertEqual(self.state()["status"], "failed")
        self.assertFalse(self.state()["online_writes_allowed"])
        self.assertNotIn("sft_candidates", self.state())

    def test_final_candidate_flag_requires_a_real_boolean_before_gpu_work(self):
        for value in ("true", 1, None):
            with self.subTest(value=value):
                self.settings["recovery"] = {"evaluate_final_checkpoint": value}
                with self.assertRaisesRegex(ValueError, "evaluate_final_checkpoint must be a boolean"):
                    runner.main(self.argv)
        self.popen.assert_not_called()

    def test_requested_stop_forwards_term_and_launches_no_next_stage(self):
        original = self.spawn

        def spawn(command, **kwargs):
            process = original(command, **kwargs)
            if "train_5090.py" in command:
                def lines():
                    yield "saving checkpoint\n"
                    self.stop()
                process.stdout = lines()
            return process

        self.popen.side_effect = spawn
        self.assertEqual(runner.main(self.argv), 0)
        self.assertEqual(len(self.processes), 2)
        self.assertEqual(self.processes[-1][1].sent_signals, [signal.SIGTERM])
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertFalse(self.state()["training_complete"])

    def test_baseline_stop_can_resume_before_any_training_checkpoint_exists(self):
        original = self.spawn
        def interrupted(command, **kwargs):
            process = original(command, **kwargs)
            if "deploy/evaluate_model.py" in command:
                def lines():
                    self.stop()
                    yield "baseline interrupted\n"
                process.stdout = lines()
            return process
        self.popen.side_effect = interrupted
        runner.main(self.argv)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.state()["stage"], "baseline_evaluate")
        old_reports = {path: path.read_bytes() for path in (self.root / "logs/recovery_test").glob("*.json")}
        self.processes.clear()
        self.popen.side_effect = original
        runner.main(self.argv + ["--resume"])
        self.assertTrue(self.state()["training_complete"])
        self.assertEqual(self.state()["resume_action"], "initialization_retry_no_checkpoint")
        self.assertIn("--init-from", self.processes[1][0])
        for path, contents in old_reports.items():
            if path.name != "state.json":
                self.assertEqual(path.read_bytes(), contents)

    def test_sft_stop_before_first_save_retries_initialization_and_reuses_baseline(self):
        original = self.spawn
        def interrupted(command, **kwargs):
            if "train_5090.py" not in command:
                return original(command, **kwargs)
            def lines():
                self.stop()
                yield "stopped before first optimizer step\n"
            process = FakeProcess(lines=lines())
            self.processes.append((command, process))
            return process
        self.popen.side_effect = interrupted
        runner.main(self.argv)
        self.assertFalse(self.output.exists())
        baseline_report = Path(self.state()["baseline_report"])
        baseline_bytes = baseline_report.read_bytes()
        self.processes.clear()
        self.popen.side_effect = original
        runner.main(self.argv + ["--resume"])
        self.assertEqual(len(self.processes), 2)
        self.assertIn("train_5090.py", self.processes[0][0])
        self.assertIn("--init-from", self.processes[0][0])
        self.assertEqual(baseline_report.read_bytes(), baseline_bytes)

    def test_read_error_can_use_valid_previous_full_but_conflicting_metadata_cannot(self):
        self.output.touch()
        previous = Path(str(self.output) + ".prev")
        previous.touch()
        expected = {"stage": "sft", "fingerprint": "newdata", "run_id": "test", "revision": "revision"}
        def corrupt(path):
            if Path(path) == self.output:
                raise RuntimeError("unreadable torch archive")
            return self.info
        with patch.object(runner, "checkpoint_info", side_effect=corrupt):
            self.assertEqual(runner.resume_checkpoint_source(self.output, **expected), previous)
        conflict = json.loads(json.dumps(self.info))
        conflict["metrics"]["dataset_fingerprint"] = "different-data"
        with patch.object(runner, "checkpoint_info", side_effect=lambda path: conflict if Path(path) == self.output else self.info) as reader:
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                runner.resume_checkpoint_source(self.output, **expected)
            reader.assert_called_once_with(self.output)

    def test_failed_training_never_evaluates_stale_best(self):
        original = self.spawn

        def spawn(command, **kwargs):
            process = original(command, **kwargs)
            if "train_5090.py" in command:
                process.code = 1
            return process

        self.popen.side_effect = spawn
        with self.assertRaisesRegex(RuntimeError, "sft exited"):
            runner.main(self.argv)
        self.assertEqual(len(self.processes), 2)
        self.assertEqual(self.state()["status"], "failed")
        self.assertFalse(self.state()["online_writes_allowed"])

    def test_checkpoint_requires_final_step_data_version_and_full_state(self):
        for field in ("step", "fingerprint", "optimizer"):
            with self.subTest(field=field):
                info = json.loads(json.dumps(self.info))
                if field == "step":
                    info["step"] = 9
                elif field == "fingerprint":
                    info["metrics"]["dataset_fingerprint"] = "old-windowing-policy"
                else:
                    info["has_optimizer"] = False
                with patch.object(runner, "checkpoint_info", return_value=info):
                    with self.assertRaises(RuntimeError):
                        runner.validate_checkpoint(self.output, stage="sft", fingerprint="newdata",
                                                   run_id="test", revision="revision", target=10)

    def test_new_run_refuses_an_old_output_without_launching_any_process(self):
        self.output.touch()
        with self.assertRaises(FileExistsError):
            runner.main(self.argv)
        self.popen.assert_not_called()

    def test_resume_keeps_history_and_omits_weight_initialization(self):
        runner.main(self.argv)
        interrupted = self.state()
        interrupted.update(status="interrupted", error="prior interruption")
        (self.root / "logs/recovery_test/state.json").write_text(json.dumps(interrupted), encoding="utf-8")
        first_count = len(interrupted["stages"])
        old_logs = {Path(stage["log"]): Path(stage["log"]).read_bytes() for stage in interrupted["stages"]}
        spawn = self.spawn

        def check_healthy_resume(command, **kwargs):
            self.assertIsNone(self.state()["error"])
            self.assertEqual(self.state()["status"], "running")
            return spawn(command, **kwargs)

        self.popen.side_effect = check_healthy_resume
        self.processes.clear()
        runner.main(self.argv + ["--resume"])
        self.assertEqual(len(self.processes), 2)
        command = self.processes[0][0]
        self.assertIn("train_5090.py", command)
        self.assertNotIn("--init-from", command)
        self.assertNotIn("--no-resume", command)
        self.assertIsNone(self.state()["error"])
        self.assertEqual(self.state()["stages"][:first_count], interrupted["stages"])
        self.assertEqual(len(self.state()["stages"]), first_count + 2)
        self.assertEqual(len(list((self.root / "logs/recovery_test").glob("*_sft.log"))), 2)
        for path, contents in old_logs.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_changed_resume_binding_preserves_the_original_manifest(self):
        runner.main(self.argv)
        path = self.root / "logs/recovery_test/state.json"
        original = path.read_bytes()
        self.data.write_text("changed data", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the original run"):
            runner.main(self.argv + ["--resume"])
        self.assertEqual(path.read_bytes(), original)

    def interrupted_sft_for_migration(self, step=4):
        self.info["step"] = step
        original = self.spawn
        def interrupted(command, **kwargs):
            process = original(command, **kwargs)
            if "train_5090.py" in command:
                def lines():
                    self.stop()
                    yield "full Adam/cursor checkpoint saved before stop\n"
                process.stdout = lines()
            return process
        self.popen.side_effect = interrupted
        runner.main(self.argv)
        self.assertEqual(self.state()["stage"], "sft")
        self.assertFalse(self.state()["training_complete"])
        self.processes.clear()
        self.popen.side_effect = original
        return self.root / "logs/recovery_test/state.json"

    def test_code_only_migration_reruns_baseline_and_continues_full_checkpoint_without_initialization(self):
        path = self.interrupted_sft_for_migration()
        old_state = self.state()
        old_baseline = Path(old_state["baseline_report"])
        old_baseline_bytes = old_baseline.read_bytes()
        old_checkpoint = json.loads(json.dumps(self.info))
        trained = False
        original = self.spawn
        def spawn(command, **kwargs):
            nonlocal trained
            process = original(command, **kwargs)
            if "train_5090.py" in command:
                self.assertNotIn("--init-from", command)
                self.assertNotIn("--no-resume", command)
                self.assertEqual(kwargs["env"]["UTTT_CODE_REVISION"], "new-revision")
                trained = True
            return process
        def saved(path):
            if Path(path) == self.output and trained:
                value = json.loads(json.dumps(old_checkpoint))
                value["step"] = 10
                value["metrics"]["code_revision"] = "new-revision"
                return value
            return old_checkpoint
        self.popen.side_effect = spawn
        with patch.dict(os.environ, {"UTTT_CODE_REVISION": "new-revision"}), patch.object(runner, "checkpoint_info", side_effect=saved):
            runner.main(self.argv + ["--resume", "--resume-from-revision", "revision"])
        state = self.state()
        self.assertEqual(state["binding"]["code_revision"], "new-revision")
        self.assertEqual([entry["stage"] for entry in state["stages"]][-3:], ["baseline_evaluate", "sft", "sft_evaluate"])
        self.assertEqual(state["code_migrations"][0]["from"], "revision")
        self.assertEqual(state["code_migrations"][0]["to"], "new-revision")
        self.assertEqual(state["code_migrations"][0]["saved_step"], 4)
        self.assertEqual(state["code_migrations"][0]["discarded_baseline_report"], str(old_baseline))
        self.assertEqual(set(state["accepted_code_revisions"]), {"revision", "new-revision"})
        self.assertNotEqual(state["baseline_report"], str(old_baseline))
        self.assertEqual(old_baseline.read_bytes(), old_baseline_bytes)
        self.assertTrue(state["training_complete"])
        # The old best remains truthfully labeled with its original source.
        self.assertEqual(old_checkpoint["metrics"]["code_revision"], "revision")
        self.assertFalse(state["online_writes_allowed"])

    def test_code_migration_rejects_config_or_wrong_old_revision_without_manifest_changes(self):
        path = self.interrupted_sft_for_migration()
        original = path.read_bytes()
        with patch.dict(os.environ, {"UTTT_CODE_REVISION": "new-revision"}):
            with self.assertRaisesRegex(ValueError, "old source revision"):
                runner.main(self.argv + ["--resume", "--resume-from-revision", "unrelated-revision"])
            self.assertEqual(path.read_bytes(), original)
            self.config.write_text("changed config", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Code-only migration cannot change"):
                runner.main(self.argv + ["--resume", "--resume-from-revision", "revision"])
            self.assertEqual(path.read_bytes(), original)
        self.assertFalse(self.processes)

    def test_code_migration_rejects_completed_or_meta_runs_and_weight_only_checkpoint(self):
        path = self.interrupted_sft_for_migration()
        original = path.read_bytes()
        with patch.dict(os.environ, {"UTTT_CODE_REVISION": "new-revision"}):
            self.info["step"] = 10
            with self.assertRaisesRegex(ValueError, "strictly below"):
                runner.main(self.argv + ["--resume", "--resume-from-revision", "revision"])
            self.assertEqual(path.read_bytes(), original)
            self.info["step"] = 4
            self.info["has_optimizer"] = False
            with self.assertRaisesRegex(RuntimeError, "Full optimizer"):
                runner.main(self.argv + ["--resume", "--resume-from-revision", "revision"])
            self.assertEqual(path.read_bytes(), original)
            self.info["has_optimizer"] = True
            state = json.loads(original)
            state["stage"] = "meta"
            path.write_text(json.dumps(state), encoding="utf-8")
            meta_state = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "unfinished SFT"):
                runner.main(self.argv + ["--resume", "--resume-from-revision", "revision"])
            self.assertEqual(path.read_bytes(), meta_state)
        self.assertFalse(self.processes)

    def test_revision_migration_flag_requires_resume_and_default_resume_remains_strict(self):
        path = self.interrupted_sft_for_migration()
        original = path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(self.argv + ["--resume-from-revision", "revision"])
        self.assertEqual(path.read_bytes(), original)
        with patch.dict(os.environ, {"UTTT_CODE_REVISION": "new-revision"}), self.assertRaisesRegex(ValueError, "differs from the original run"):
            runner.main(self.argv + ["--resume"])
        self.assertEqual(path.read_bytes(), original)

    def test_checkpoint_revision_sets_follow_only_continuous_explicit_migrations(self):
        state = {"binding": {"code_revision": "third"}, "code_migrations": [
            {"from": "first", "to": "second"}, {"from": "second", "to": "third"}]}
        self.assertEqual(runner.accepted_checkpoint_revisions(state), {"first", "second", "third"})
        runner.validate_checkpoint(self.output, stage="sft", fingerprint="newdata", run_id="test",
                                   revision={"revision", "new-revision"}, saved=self.info)
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            runner.validate_checkpoint(self.output, stage="sft", fingerprint="newdata", run_id="test",
                                       revision="new-revision", saved=self.info)
        state["code_migrations"][0]["to"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "continuous revision chain"):
            runner.accepted_checkpoint_revisions(state)

    def test_meta_records_separate_heldout_ablation_and_reuses_verified_reports_on_resume(self):
        arguments = self.argv + self.meta_arguments()
        runner.main(arguments)
        stages = [entry["stage"] for entry in self.state()["stages"]]
        self.assertEqual(stages, ["baseline_evaluate", "sft", "sft_evaluate", "memory_baseline_evaluate",
                                  "meta", "meta_evaluate", "meta_memory_evaluate"])
        self.assertEqual(set(self.state()["memory_evaluations"]), {"baseline", "meta"})
        self.assertFalse(self.state()["online_writes_allowed"])
        self.assertFalse(self.state()["online_memory_quality_verified"])
        baseline_command = self.processes[3][0]
        self.assertIn("--allow-untrained-baseline", baseline_command)
        self.assertEqual(baseline_command[baseline_command.index("--max-samples") + 1], "64")
        report_files = {Path(command[command.index("--output") + 1]) for command, _ in self.processes if "--output" in command}
        self.assertEqual(len(report_files), 5)
        previous_contents = {path: path.read_bytes() for path in report_files}
        self.processes.clear()
        runner.main(arguments + ["--resume"])
        self.assertFalse(any("deploy/evaluate_memory.py" in command for command, _ in self.processes))
        self.assertTrue(all("--init-from" not in command for command, _ in self.processes if "train_5090.py" in command))
        for path, contents in previous_contents.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertFalse(self.state()["online_writes_allowed"])

    def test_failed_sft_quality_does_not_start_meta_or_memory_evaluation(self):
        arguments = self.argv + self.meta_arguments()
        self.candidate_correct = 1
        runner.main(arguments)
        self.assertEqual(len(self.processes), 3)
        self.assertEqual(self.state()["meta_skipped_reason"], "SFT heldout quality gate failed")

    def test_memory_manifest_is_required_and_bound_into_resume_identity(self):
        arguments = self.argv + self.meta_arguments()
        runner.main(arguments)
        manifest = self.root / "memory/manifest.json"
        contents = json.loads(manifest.read_text(encoding="utf-8"))
        contents["changed_note"] = True
        manifest.write_text(json.dumps(contents), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the original run"):
            runner.main(arguments + ["--resume"])

    def test_improvement_and_eos_gate_cannot_be_hidden_by_completion(self):
        baseline = self.report(2)
        same = runner.quality_comparison(baseline, self.report(2), self.tasks, {})
        self.assertFalse(same["quality_passed"])
        self.assertIn("heldout_accuracy_did_not_improve", same["reasons"])
        regressed = runner.quality_comparison(baseline, self.report(4, eos=False), self.tasks, {})
        self.assertFalse(regressed["quality_passed"])
        self.assertIn("eos_rate_regressed", regressed["reasons"])

    def test_wrong_heldout_report_is_rejected(self):
        report = self.report(4)
        report["results"][0]["prompt"] = "a different task"
        with self.assertRaisesRegex(ValueError, "fixed heldout tasks"):
            runner.quality_comparison(self.report(0), report, self.tasks, {})

    def test_memory_only_probe_cannot_authorize_full_parameter_meta(self):
        path = self.root / "probe.json"
        path.write_text(json.dumps({"status": "ok", "finite": True, "scope": "memory",
                                   "steps": [{}, {}], "all_writers_have_nonzero_ce_grad": True}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same parameter scope"):
            runner.validate_meta_probe(path, self.settings)

    def test_probe_geometry_must_match_meta_training(self):
        config = {"training": {"stage": "meta", "parameter_scope": "memory", "seq_len": 1024,
                               "micro_batch_size": 1, "meta_checkpoint_layers": 4,
                               "meta_checkpoint_reentrant": True}, "uttt": {"chunk_size": 128}}
        report = {"status": "ok", "finite": True, "scope": "memory",
                  "steps": [{"finite": True, "all_writers_have_nonzero_ce_grad": True}] * 2,
                  "all_writers_have_nonzero_ce_grad": True, "seq_len": 1024, "batch_size": 1,
                  "chunk_size": 128, "gradient_checkpointing": True, "gradient_checkpointing_every": 1,
                  "meta_checkpoint_layers": 4, "meta_checkpoint_time_chunks": True,
                  "meta_checkpoint_reentrant": True, "surprise_loss_measured": True,
                  "peak_allocated_bytes": 27 * 1024 ** 3, "peak_reserved_bytes": 27 * 1024 ** 3}
        path = self.root / "probe.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        self.assertEqual(runner.validate_meta_probe(path, config)["scope"], "memory")
        for key, value in (("seq_len", 256), ("meta_checkpoint_reentrant", False),
                           ("surprise_loss_measured", False), ("peak_reserved_bytes", 29 * 1024 ** 3)):
            with self.subTest(key=key):
                path.write_text(json.dumps({**report, key: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.validate_meta_probe(path, config)


if __name__ == "__main__":
    unittest.main()

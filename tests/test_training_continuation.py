"""A new SFT phase preserves full state without touching its completed parent."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
    import yaml
    import train_5090 as training
    from engine.corpus import ResumableBatchSampler
    from model.uttt_model import uTTTConfig
    from model.tokenizer import build_tokenizer
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class TrainingContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data.jsonl"
        self.data.write_text("\n".join(json.dumps({
            "text": f"<|user|> Copy {index}.<|assistant|> {index}.<|eos|>"
        }) for index in range(40)), encoding="utf-8")
        self.content = {
            "model": {"vocab_size": 512, "d_model": 8, "d_ff_fast": 16, "n_layers": 1,
                      "n_heads": 2, "n_kv_heads": 1, "chunk_size": 8,
                      "sliding_window_size": 16, "max_seq_len": 64, "dropout": 0.},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": "sft", "micro_batch_size": 1, "grad_accum_steps": 2,
                         "max_steps": 2, "seq_len": 64, "seed": 933, "warmup_steps": 1,
                         "outer_lr": 2.e-5, "meta_fast_lr": 1.e-5, "weight_decay": .01,
                         "gradient_checkpointing": False, "pack_sequences": False,
                         "trim_sft_padding": True, "validation_fraction": .25,
                         "validation_selection_metric": "validation_example_ce",
                         "eval_interval": 1, "eval_samples": 4, "save_interval": 1, "log_interval": 1},
        }
        self.parent_config = self.write_config("parent", self.content)
        self.parent = self.root / "parent.pt"
        self.call(self.parent_config, self.parent, scores=[.5, .4], resume=False,
                  run_id="parent-run", revision="parent-revision")
        self.parent_bytes = self.parent.read_bytes()
        self.saved = self.load(self.parent)
        self.phase_content = json.loads(json.dumps(self.content))
        self.phase_content["training"].update(max_steps=4, outer_lr=5.e-6, meta_fast_lr=2.5e-6,
                                               lr_schedule_start_step=2, lr_warmup_start_ratio=.4)
        self.phase_config = self.write_config("phase", self.phase_content)

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self, name, content):
        path = self.root / (name + ".yaml")
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
        return path

    @staticmethod
    def report(score):
        return {"validation_ce": score + .1, "validation_frozen_ce": score + .1,
                "validation_example_ce": score, "validation_targets": 12, "validation_examples": 4}

    def call(self, config, output, *, scores, run_id="phase-run", revision="phase-revision", **kwargs):
        log = io.StringIO()
        with redirect_stdout(log), mock.patch.dict(os.environ, {
            "UTTT_RUN_ID": run_id, "UTTT_CODE_REVISION": revision,
        }), mock.patch.object(training, "evaluate_language_model", side_effect=[self.report(s) for s in scores]):
            result = training.train(str(config), str(self.data), str(output), **kwargs)
        return result, log.getvalue()

    @staticmethod
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_full_state_continues_from_next_samples_without_mutating_parent(self):
        config = uTTTConfig.from_dict(self.saved["config"])
        tokenizer = build_tokenizer(config)
        dataset = training.PersonaMetaDataset(tokenizer, seq_len=64, data_path=str(self.data),
                                              pack_sequences=False, mode="sft", split="train",
                                              validation_fraction=.25, split_seed=933)
        sampler = ResumableBatchSampler(len(dataset), 1, 933)
        sampler.load_state_dict(self.saved["training_state"]["data_cursor"])
        expected = [dataset[index[0]]["input_ids"].tolist() for index in list(iter(sampler))[:4]]
        calls, rng_states = [], []
        original = training.uTTTModel.forward

        def observe(model, *args, **kwargs):
            calls.append(kwargs["input_ids"][0].tolist())
            rng_states.append(torch.get_rng_state().clone())
            return original(model, *args, **kwargs)

        output = self.root / "phase.pt"
        with mock.patch.object(training.uTTTModel, "forward", new=observe):
            self.call(self.phase_config, output, scores=[.8, .9], continue_from=str(self.parent))
        final, best = self.load(output), self.load(output.with_suffix(".best.pt"))
        self.assertEqual(calls, expected)
        torch.testing.assert_close(rng_states[0], self.saved["training_state"]["rng"]["cpu"], rtol=0, atol=0)
        self.assertEqual(final["step"], 4)
        self.assertTrue(all(int(value["step"]) == 4 for value in final["optimizer_state"]["state"].values()))
        self.assertEqual(final["training_state"]["data_cursor"]["offset"], 8)
        self.assertGreater(final["metrics"]["tokens_seen"], self.saved["metrics"]["tokens_seen"])
        record = final["metrics"]["continuation"]
        self.assertEqual(record["sha256"], hashlib.sha256(self.parent_bytes).hexdigest())
        self.assertEqual(record["step"], 2)
        self.assertEqual(record["source_run_id"], "parent-run")
        self.assertEqual(record["source_code_revision"], "parent-revision")
        self.assertTrue(record["phase_best_reset"])
        self.assertEqual(record["historical_best_validation_score"], .4)
        self.assertEqual(final["metrics"]["best_validation_score"], .8)
        self.assertEqual(best["step"], 3)  # New phase best exists despite not beating the parent's .4.
        self.assertEqual(final["metrics"]["run_id"], "phase-run")
        self.assertEqual(self.parent.read_bytes(), self.parent_bytes)

    def test_normal_resume_keeps_phase_history_and_matches_uninterrupted_phase(self):
        phase = self.root / "phase.pt"
        self.call(self.phase_config, phase, scores=[.8, .9], continue_from=str(self.parent))
        reference = self.load(phase)
        resumed = self.root / "resumed.pt"
        # Independent copies of tiny test files, never hard links to a source.
        shutil.copyfile(str(phase) + ".prev", resumed)
        shutil.copyfile(phase.with_suffix(".best.pt"), resumed.with_suffix(".best.pt"))
        self.call(self.phase_config, resumed, scores=[.9], resume=True)
        actual = self.load(resumed)
        self.assertEqual(actual["step"], 4)
        self.assertEqual(actual["metrics"]["continuation"], reference["metrics"]["continuation"])
        self.assertEqual(actual["metrics"]["best_validation_score"], .8)
        self.assertEqual(self.load(resumed.with_suffix(".best.pt"))["step"], 3)
        self.assertEqual(actual["training_state"]["data_cursor"], reference["training_state"]["data_cursor"])
        for section in ("slow_weights", "init_fast_weights"):
            for name, value in actual[section].items():
                torch.testing.assert_close(value, reference[section][name], rtol=0, atol=0)
        for index, state in actual["optimizer_state"]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, reference["optimizer_state"]["state"][index][name], rtol=0, atol=0)
        with mock.patch.object(training.uTTTModel, "load_dual_base_state") as load:
            with self.assertRaisesRegex(ValueError, "different phase LR schedule"):
                self.call(self.phase_config, resumed, scores=[], override_steps=5, resume=True)
            load.assert_not_called()
        self.assertEqual(self.parent.read_bytes(), self.parent_bytes)

    def test_incomplete_or_incompatible_sources_fail_before_hardware_or_output(self):
        mutations = {
            "optimizer": lambda p: p.pop("optimizer_state"),
            "rng": lambda p: p["training_state"].pop("rng"),
            "cursor": lambda p: p["training_state"]["data_cursor"].update(seed=934),
            "cursor_step": lambda p: p["training_state"]["data_cursor"].update(offset=3),
            "accumulation": lambda p: p["training_state"].update(grad_accum_steps=4),
            "data": lambda p: p["metrics"].update(dataset_fingerprint="wrong"),
            "metric": lambda p: p["metrics"].update(validation_selection_metric="validation_ce"),
            "math": lambda p: p["config"].update(rope_theta=20000.),
            "tokenizer": lambda p: p["config"].update(tokenizer_json="different"),
            "provenance": lambda p: p["metrics"].update(run_id=None),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = self.load(self.parent)
                mutate(payload)
                bad = self.root / (name + ".pt")
                torch.save(payload, bad)
                output = self.root / ("out_" + name + ".pt")
                with mock.patch.object(training, "configure_rtx5090_environment") as hardware:
                    with self.assertRaises(ValueError):
                        self.call(self.phase_config, output, scores=[], continue_from=str(bad))
                    hardware.assert_not_called()
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".best.pt").exists())
        self.assertEqual(self.parent.read_bytes(), self.parent_bytes)

    def test_parameter_conflicts_existing_outputs_and_stage_offsets_are_rejected(self):
        output = self.root / "output.pt"
        for kwargs in ({"init_from": str(self.parent)}, {"resume": False}, {"lightweight_preset": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "cannot be combined"):
                self.call(self.phase_config, output, scores=[], continue_from=str(self.parent), **kwargs)
        for suffix in ("", ".prev", ".best.pt", ".best.pt.prev"):
            path = output if suffix == "" else (output.with_suffix(".best.pt") if suffix == ".best.pt"
                                                else Path(str(output.with_suffix(".best.pt")) + ".prev")
                                                if suffix == ".best.pt.prev" else Path(str(output) + suffix))
            path.write_bytes(b"preserve")
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "new output"):
                self.call(self.phase_config, output, scores=[], continue_from=str(self.parent))
            self.assertEqual(path.read_bytes(), b"preserve")
            path.unlink()
        with self.assertRaisesRegex(ValueError, "new output"):
            self.call(self.phase_config, self.parent, scores=[], continue_from=str(self.parent))
        with self.assertRaisesRegex(ValueError, "requires --continue-from"):
            self.call(self.phase_config, output, scores=[], resume=False)
        with redirect_stderr(io.StringIO()), mock.patch.object(sys, "argv", ["train", "--continue-from", "a", "--no-resume"]):
            with self.assertRaises(SystemExit):
                training.parse_args()

    def test_explicit_schedule_starts_at_parent_floor_and_legacy_formula_is_exact(self):
        for total in (2, 7, 2000):
            for warmup in (0, 1, min(100, total)):
                spec = training.training_lr_schedule("sft", 0, 0., total, warmup, 2.e-5, 1.e-5)
                for step in (1, max(1, total // 2), total):
                    self.assertEqual(training.scheduled_training_lr(step, 2.e-5, spec),
                                     training.get_cosine_lr(step - 1, total, warmup, 2.e-5))
        spec = training.training_lr_schedule("sft", 2000, .4, 6000, 100, 5.e-6, 2.5e-6)
        self.assertAlmostEqual(training.scheduled_training_lr(2001, 5.e-6, spec), 2.e-6)
        self.assertAlmostEqual(training.scheduled_training_lr(2001, 2.5e-6, spec), 1.e-6)
        self.assertAlmostEqual(training.scheduled_training_lr(2101, 5.e-6, spec), 5.e-6)
        self.assertLess(training.scheduled_training_lr(6000, 5.e-6, spec), 5.1e-7)
        for stage, offset, ratio in (("meta", 2, .4), ("pretrain", 0, .4), ("sft", True, 0.),
                                     ("sft", -1, 0.), ("sft", 4, 0.), ("sft", 2, float("nan"))):
            with self.subTest(stage=stage, offset=offset, ratio=ratio), self.assertRaises(ValueError):
                training.training_lr_schedule(stage, offset, ratio, 4, 1, 5.e-6, 2.5e-6)


if __name__ == "__main__":
    unittest.main()

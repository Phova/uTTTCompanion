"""Tiny CPU training verifies fused execution and transactional OOM replay."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    import yaml
    import train_5090 as training
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class SftExecutionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, root, name, execution_size=None, *, long_rows=False, **execution_options):
        data = root / ("long.jsonl" if long_rows else "train.jsonl")
        if not data.exists():
            # Seven rows force B1/A4 step two to cross an epoch boundary. Their
            # answer lengths differ, exposing accidental token-weighted CE.
            answers = ("a", "bcde", "f" * 12, "gh", "ijklmnop", "q" * 23, "rst")
            if long_rows:
                answers = tuple(chr(97 + index) * 480 for index in range(7))
            data.write_text("\n".join(json.dumps({
                "text": f"<|user|> {index}?<|assistant|> {answer}<|eos|>"
            }) for index, answer in enumerate(answers)), encoding="utf-8")
        content = {
            "model": {"vocab_size": 512, "d_model": 8, "d_ff_fast": 16, "n_layers": 1,
                      "n_heads": 2, "n_kv_heads": 1, "chunk_size": 8,
                      "sliding_window_size": 16, "max_seq_len": 512, "dropout": 0.},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": "sft", "micro_batch_size": 1, "grad_accum_steps": 4,
                         "max_steps": 2, "seq_len": 512 if long_rows else 64,
                         "seed": 419, "warmup_steps": 0, "outer_lr": 2.e-5,
                         "meta_fast_lr": 1.e-5, "weight_decay": .01,
                         "max_outer_grad_norm": 1., "pack_sequences": False,
                         "gradient_checkpointing": False, "trim_sft_padding": True,
                         "save_interval": 2, "log_interval": 1},
        }
        if execution_size is not None:
            content["training"].update(sft_execution_batch_size=execution_size,
                                        sft_padding_multiple=32, sft_sort_by_length=True)
            content["training"].update(execution_options)
        config = root / (name + ".yaml")
        config.write_text(yaml.safe_dump(content), encoding="utf-8")
        return config, data, root / (name + ".pt")

    def run_training(self, fixture, forward_hook=None):
        config, data, output = fixture
        log = io.StringIO()
        context = (mock.patch.object(training.uTTTModel, "forward", new=forward_hook)
                   if forward_hook is not None else mock.patch.object(training, "_STOP_REQUESTED", 0))
        with redirect_stdout(log), context:
            path = training.train(str(config), str(data), str(output), resume=False)
        return torch.load(path, map_location="cpu", weights_only=False), log.getvalue()

    def assert_equivalent(self, actual, expected, *, exact=False):
        self.assertEqual(actual["step"], expected["step"])
        self.assertEqual(actual["step"], 2)
        for section in ("slow_weights", "init_fast_weights"):
            self.assertEqual(actual[section].keys(), expected[section].keys())
            for name, value in actual[section].items():
                torch.testing.assert_close(value, expected[section][name],
                                           atol=0 if exact else 2.e-6,
                                           rtol=0 if exact else 2.e-5,
                                           msg=section + "." + name)
        left, right = actual["optimizer_state"], expected["optimizer_state"]
        self.assertEqual(left["param_groups"], right["param_groups"])
        self.assertEqual(left["state"].keys(), right["state"].keys())
        for index, state in left["state"].items():
            self.assertEqual(state.keys(), right["state"][index].keys())
            self.assertEqual(int(state["step"]), 2)
            for name, value in state.items():
                torch.testing.assert_close(value, right["state"][index][name],
                                           atol=0 if exact else 2.e-7,
                                           rtol=0 if exact else 3.e-4,
                                           msg=f"optimizer[{index}].{name}")
        for name in ("total_loss", "ce_loss"):
            self.assertAlmostEqual(actual["metrics"][name], expected["metrics"][name], places=5)
        for name in ("tokens_seen", "dataset_fingerprint", "training_stage", "parameter_scope"):
            self.assertEqual(actual["metrics"][name], expected["metrics"][name], name)
        self.assertEqual(actual["training_state"]["data_cursor"], expected["training_state"]["data_cursor"])
        cursor = actual["training_state"]["data_cursor"]
        self.assertEqual((cursor["size"], cursor["epoch"], cursor["offset"]), (7, 1, 1))
        for name in ("micro_batch_size", "grad_accum_steps"):
            self.assertEqual(actual["training_state"][name], expected["training_state"][name])
        self.assertEqual(actual["training_state"]["micro_batch_size"], 1)
        self.assertEqual(actual["training_state"]["grad_accum_steps"], 4)
        self.assertIn("rng", actual["training_state"])

    def test_two_cpu_optimizer_steps_keep_original_loss_weights_adam_and_cross_epoch_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected, _ = self.run_training(self.fixture(root, "legacy"))
            self.assertIsNone(expected["metrics"]["sft_execution"])
            for execution_size in (2, 4):
                with self.subTest(execution_size=execution_size):
                    actual, log = self.run_training(self.fixture(root, f"fused{execution_size}", execution_size))
                    self.assert_equivalent(actual, expected)
                    self.assertEqual(actual["metrics"]["sft_execution"]["execution_batch_size"], execution_size)
                    self.assertEqual(actual["metrics"]["sft_execution"]["ce_reduction"], "sequence_mean")
                    self.assertIn("SFTExecution=", log)

    def test_partial_execution_oom_reduces_group_and_replays_complete_optimizer_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected, _ = self.run_training(self.fixture(root, "reference", 1))
            original_forward = training.uTTTModel.forward
            calls, failure = [], []

            def fail_second_group(model, *args, **kwargs):
                calls.append(tuple(kwargs["input_ids"].shape))
                if not failure and len(calls) == 2:
                    failure.append(True)
                    raise torch.OutOfMemoryError("simulated after one fused backward")
                return original_forward(model, *args, **kwargs)

            actual, log = self.run_training(self.fixture(root, "oom", 2), fail_second_group)
            self.assertEqual(failure, [True])
            self.assertEqual([shape[0] for shape in calls[:2]], [2, 2])
            self.assertEqual([shape[0] for shape in calls[2:]], [1] * 8)
            self.assertIn("Reduced SFT execution group to 1", log)
            self.assertEqual(actual["metrics"]["sft_execution"]["execution_batch_size"], 1)
            self.assert_equivalent(actual, expected, exact=True)

    def test_physical_token_budget_splits_execution_groups_without_changing_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected, _ = self.run_training(self.fixture(root, "budget_reference", 2))
            original_forward = training.uTTTModel.forward
            shapes = []

            def observe(model, *args, **kwargs):
                shapes.append(tuple(kwargs["input_ids"].shape))
                return original_forward(model, *args, **kwargs)

            # All valid rows fit one 32-token bucket. A 64-physical-token budget
            # makes requested B4 execute as B2, preserving both optimizer steps.
            actual, _ = self.run_training(
                self.fixture(root, "budget", 4, sft_max_group_tokens=64), observe)
            self.assertEqual(shapes, [(2, 32)] * 4)
            self.assertEqual(actual["metrics"]["sft_execution"]["execution_batch_size"], 4)
            self.assertEqual(actual["metrics"]["sft_execution"]["max_tokens_per_group"], 64)
            self.assert_equivalent(actual, expected, exact=True)


if __name__ == "__main__":
    unittest.main()

"""Validation reports preserve token CE while SFT can select per-example CE."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
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
class ValidationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def controlled_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_embeddings = torch.nn.Embedding(32, 4)
                self.fast_weight_pool = SimpleNamespace(readonly_state=lambda **kwargs: object())
                self.calls = []

            def forward(self, input_ids, update_fast_weights=False, **kwargs):
                marker = int(input_ids[0, 0])
                self.calls.append((marker, update_fast_weights))
                loss = {10: 1., 20: 3., 30: 99.}[marker] + (2. if update_fast_weights else 0.)
                return SimpleNamespace(ce_loss=torch.tensor(loss))

        return Model().train()

    def controlled_rows(self):
        rows = []
        for marker, count in ((10, 2), (20, 6), (30, 0)):
            ids = torch.full((8,), marker, dtype=torch.long)
            labels = torch.full_like(ids, -100)
            labels[1:1 + count] = 6
            rows.append({"input_ids": ids, "labels": labels, "attention_mask": torch.ones(8)})
        return rows

    def test_example_mean_and_token_mean_differ_and_zero_target_rows_are_excluded(self):
        for updates in (False, True):
            with self.subTest(updates=updates):
                model = self.controlled_model()
                before_rng = torch.get_rng_state().clone()
                report = training.evaluate_language_model(
                    model, self.controlled_rows(), torch.device("cpu"), torch.float32, updates,
                )
                self.assertEqual(report["validation_ce"], 2.5 + (2. if updates else 0.))
                self.assertEqual(report["validation_frozen_ce"], 2.5)
                self.assertEqual(report["validation_targets"], 8)
                self.assertEqual(report["validation_example_ce"], 2. + (2. if updates else 0.))
                self.assertEqual(report["validation_examples"], 2)
                self.assertFalse(any(marker == 30 for marker, _ in model.calls))
                self.assertTrue(model.training)
                torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)

    def test_default_selection_is_compatible_and_example_selection_is_sft_only(self):
        for stage in ("sft", "pretrain", "meta"):
            self.assertEqual(training.validate_validation_selection_metric(stage), "validation_ce")
            self.assertEqual(training.validate_validation_selection_metric(stage, "validation_ce"), "validation_ce")
        self.assertEqual(training.validate_validation_selection_metric("sft", "validation_example_ce"),
                         "validation_example_ce")
        for stage in ("pretrain", "meta"):
            with self.assertRaisesRegex(ValueError, "requires stage=sft"):
                training.validate_validation_selection_metric(stage, "validation_example_ce")
        for value in (None, True, 1, "example_mean", "ce", "validation_frozen_ce"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "validation_selection_metric"):
                training.validate_validation_selection_metric("sft", value)

    def test_resume_preserves_metric_and_historical_score_with_legacy_defaults(self):
        self.assertEqual(training.validate_resume_validation_selection({"best_validation_ce": 2.5}, "validation_ce"), 2.5)
        self.assertEqual(training.validate_resume_validation_selection({}, "validation_ce"), float("inf"))
        metadata = {"validation_selection_metric": "validation_example_ce", "best_validation_score": 1.4,
                    "best_validation_ce": .9, "validation_example_ce": 1.7}
        self.assertEqual(training.validate_resume_validation_selection(metadata, "validation_example_ce"), 1.4)
        for old, requested in (({}, "validation_example_ce"),
                               (metadata, "validation_ce"),
                               ({"validation_selection_metric": None}, "validation_ce")):
            with self.assertRaisesRegex(ValueError, "different validation_selection_metric"):
                training.validate_resume_validation_selection(old, requested)
        with self.assertRaisesRegex(ValueError, "missing historical best_validation_score"):
            training.validate_resume_validation_selection(
                {"validation_selection_metric": "validation_example_ce", "validation_example_ce": 1.4},
                "validation_example_ce")
        for value in (None, "broken", float("nan"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid best_validation_score"):
                training.validate_resume_validation_selection({"best_validation_score": value}, "validation_ce")

    def fixture(self, root, metric=None):
        data = root / "train.jsonl"
        if not data.exists():
            data.write_text("\n".join(json.dumps({
                "text": f"<|user|> Question {index}?<|assistant|> Answer {index}.<|eos|>"
            }) for index in range(12)), encoding="utf-8")
        content = {
            "model": {"vocab_size": 512, "d_model": 8, "d_ff_fast": 16, "n_layers": 1,
                      "n_heads": 2, "n_kv_heads": 1, "chunk_size": 8,
                      "sliding_window_size": 16, "max_seq_len": 64, "dropout": 0.},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": "sft", "micro_batch_size": 1, "grad_accum_steps": 1,
                         "max_steps": 3, "seq_len": 64, "seed": 714, "warmup_steps": 0,
                         "outer_lr": 2.e-5, "meta_fast_lr": 1.e-5, "weight_decay": .01,
                         "pack_sequences": False, "gradient_checkpointing": False,
                         "trim_sft_padding": True, "validation_fraction": .5,
                         "eval_interval": 1, "eval_samples": 8, "save_interval": 3, "log_interval": 1},
        }
        if metric is not None:
            content["training"]["validation_selection_metric"] = metric
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
        return path, data, content

    @staticmethod
    def reports():
        return [dict(validation_ce=token, validation_frozen_ce=token, validation_targets=10,
                     validation_example_ce=example, validation_examples=2)
                for token, example in ((1., 1.), (.9, 1.2), (.8, 1.3))]

    def test_actual_training_selects_requested_metric_and_resume_retains_best_score(self):
        for metric, best_step, best_score in ((None, 3, .8), ("validation_example_ce", 1, 1.)):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, data, _ = self.fixture(root, metric)
                output = root / "model.pt"
                with redirect_stdout(io.StringIO()), mock.patch.object(
                        training, "evaluate_language_model", side_effect=self.reports()):
                    training.train(str(config), str(data), str(output), resume=False)
                final = torch.load(output, map_location="cpu", weights_only=False)
                best = torch.load(output.with_suffix(".best.pt"), map_location="cpu", weights_only=False)
                self.assertEqual(best["step"], best_step)
                self.assertEqual(final["metrics"]["best_validation_ce"], .8)
                self.assertEqual(final["metrics"]["best_validation_score"], best_score)
                self.assertEqual(final["metrics"]["validation_selection_metric"], metric or "validation_ce")
                resumed_report = dict(validation_ce=.7, validation_frozen_ce=.7,
                                      validation_targets=10, validation_example_ce=1.1, validation_examples=2)
                with redirect_stdout(io.StringIO()), mock.patch.object(
                        training, "evaluate_language_model", return_value=resumed_report):
                    training.train(str(config), str(data), str(output), override_steps=4, resume=True)
                resumed = torch.load(output, map_location="cpu", weights_only=False)
                resumed_best = torch.load(output.with_suffix(".best.pt"), map_location="cpu", weights_only=False)
                self.assertEqual(resumed["step"], 4)
                self.assertEqual(resumed["metrics"]["best_validation_ce"], .7)
                self.assertEqual(resumed["metrics"]["best_validation_score"], .7 if metric is None else 1.)
                self.assertEqual(resumed_best["step"], 4 if metric is None else 1)

    def test_invalid_metric_or_stage_fails_before_hardware_initialization(self):
        for stage, metric in (("sft", "example_mean"), ("pretrain", "validation_example_ce"),
                              ("meta", "validation_example_ce")):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                config, data, content = self.fixture(Path(tmp), metric)
                content["training"]["stage"] = stage
                config.write_text(yaml.safe_dump(content), encoding="utf-8")
                with mock.patch.object(training, "configure_rtx5090_environment") as configure:
                    with self.assertRaises(ValueError):
                        training.train(str(config), str(data), str(Path(tmp) / "out.pt"), resume=False)
                    configure.assert_not_called()

    def test_resume_metric_mismatch_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data, content = self.fixture(root)
            output = root / "model.pt"
            with redirect_stdout(io.StringIO()), mock.patch.object(
                    training, "evaluate_language_model", side_effect=self.reports()):
                training.train(str(config), str(data), str(output), resume=False)
            before = output.read_bytes()
            content["training"]["validation_selection_metric"] = "validation_example_ce"
            config.write_text(yaml.safe_dump(content), encoding="utf-8")
            with mock.patch.object(training.uTTTModel, "load_dual_base_state") as load:
                with self.assertRaisesRegex(ValueError, "different validation_selection_metric"):
                    training.train(str(config), str(data), str(output), override_steps=4, resume=True)
                load.assert_not_called()
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

"""Training diagnostics must expose disabled TTT and preserve training randomness."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy.diagnose_training import diagnose, parse_training_log
from deploy.training_diagnostics import repeated_ngram_fraction, repetition_metrics

try:
    import torch
    import yaml
    import train_5090 as training
except ImportError:
    torch = None


class LogDiagnosticsTests(unittest.TestCase):
    def test_ngram_diagnostics_find_cycles_without_penalizing_short_or_unique_text(self):
        self.assertEqual(repeated_ngram_fraction([1, 2]), 0.0)
        self.assertEqual(repeated_ngram_fraction(list(range(12))), 0.0)
        repeated = repetition_metrics([7] * 20, "abc " * 10)
        self.assertGreater(repeated["text_4gram_repetition"], 0.8)
        self.assertGreater(repeated["token_4gram_repetition"], 0.9)
        self.assertEqual(repeated["max_token_run"], 20)
        with self.assertRaises(ValueError):
            repeated_ngram_fraction([], n=0)

    def test_legacy_pretrain_zero_is_unmeasured_and_ce_trend_is_separate_from_repetition(self):
        samples = {"prompt": "Science", "reply": "science science science science science", "eos_generated": False}
        lines = [
            "[Train] Starting pretrain from step 1 to 100000 | MicroBatch=1",
            "[Validation 500] CE=5.2000 | Frozen CE=5.2000 | Tokens seen=16,384,000",
            "[Validation 15000] CE=3.4000 | Frozen CE=3.4000 | Tokens seen=491,520,000",
            "[Validation 15500] CE=3.3688 | Frozen CE=3.3688 | Tokens seen=507,904,000",
            "[Validation 16000] CE=3.3604 | Frozen CE=3.3604 | Tokens seen=524,288,000",
            "[Validation sample] " + json.dumps(samples),
            "[Step 16010/100000] Total Loss: 3.44 | CE Loss: 3.44 | Surprise Loss: 0.0000 | Speed: 7,300 tok/s",
        ]
        report = parse_training_log(lines, recent=2)
        self.assertFalse(report["ttt_updates_enabled"])
        self.assertFalse(report["surprise_loss_measured"])
        self.assertEqual(report["latest_training"]["tokens_per_second"], 7300)
        self.assertEqual(report["validation"]["latest"]["tokens_seen"], 524288000)
        self.assertAlmostEqual(report["validation"]["recent_ce_delta"], -0.0084)
        self.assertEqual(report["samples"]["repetition_flagged"], 1)
        self.assertTrue(any("placeholder" in note for note in report["notes"]))
        self.assertNotIn("token_4gram_repetition", report["samples"]["recent"][0])

    def test_pipeline_stages_do_not_mix_validation_trends_and_bad_samples_are_skipped(self):
        lines = [
            "[Train] Starting pretrain from step 1 to 100000",
            "[Validation 100000] CE=3.1000 | Frozen CE=3.1000 | Tokens seen=3,276,800,000",
            "[Train] Starting sft from step 1 to 6000",
            "[Validation 250] CE=2.3000 | Frozen CE=2.3000 | Tokens seen=4,000,000 | Stage: sft | TTT: disabled",
            '[Validation sample] {"reply":123}',
            "[Validation sample] incomplete-json",
            "[Step 0250/6000] Stage: sft | Total Loss: 2.3 | CE Loss: 2.3 | Surprise Loss: disabled | Speed: 500 tok/s",
        ]
        latest = parse_training_log(lines)
        self.assertEqual(latest["stage"], "sft")
        self.assertEqual(latest["validation"]["count"], 1)
        self.assertEqual(latest["validation"]["latest"]["ce"], 2.3)
        self.assertIsNone(latest["latest_training"]["surprise_loss"])
        self.assertEqual(latest["samples"]["malformed_rows"], 2)
        self.assertEqual(parse_training_log(lines, stage="pretrain")["validation"]["count"], 1)

    def test_meta_stage_and_unknown_stage_do_not_invent_disabled_status(self):
        line = "[Step 1/10] Stage: meta | CE Loss: 3.0 | Surprise Loss: 0.000001 | Speed: 123 tok/s"
        report = parse_training_log([line])
        self.assertTrue(report["surprise_loss_measured"])
        self.assertEqual(report["latest_training"]["surprise_loss"], 0.000001)
        self.assertIsNone(parse_training_log(["[Step 1/10] Surprise Loss: 0.0000"])["surprise_loss_measured"])

    def test_nonfinite_losses_are_reported_even_when_the_last_parseable_validation_was_finite(self):
        report = parse_training_log([
            "[Train] Starting meta from step 1 to 100",
            "[Validation 50] CE=3.5 | Frozen CE=3.7 | Tokens seen=1000",
            "[Validation 100] CE=nan | Frozen CE=inf | Tokens seen=2000",
            "[Step 100/100] CE Loss: nan | GradNorm: inf",
        ])
        self.assertEqual(report["nonfinite_metrics"]["count"], 2)
        self.assertTrue(any("Non-finite" in note for note in report["notes"]))

    def test_readonly_report_does_not_modify_source_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.log"
            content = "[Train] Starting pretrain from step 1 to 20\n"
            path.write_text(content, encoding="utf-8")
            result = diagnose(path)
            self.assertEqual(result["stage"], "pretrain")
            self.assertGreaterEqual(result["log_age_seconds"], 0.0)
            self.assertEqual(path.read_text(encoding="utf-8"), content)


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class SamplingDiagnosticsTests(unittest.TestCase):
    def test_dual_decoding_is_repeatable_and_preserves_training_rng(self):
        class Tokenizer:
            EOS_ID = 2

            def encode(self, text, **kwargs):
                return [1, 6, 7]

            def decode(self, tokens):
                return " ".join(map(str, tokens))

        class Model:
            config = SimpleNamespace(max_seq_len=64, repetition_penalty=1.7)
            training = True
            fast_weight_pool = SimpleNamespace(readonly_state=lambda **kwargs: None)

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def init_fast_state(self, **kwargs):
                raise AssertionError("Validation should reuse a read-only state")

            def stream_generate(self, inputs, state, temperature, **kwargs):
                if temperature == 0:
                    yield from [8] * 8
                else:
                    yield from torch.randint(6, 30, (8,)).tolist()

        torch.manual_seed(193)
        state = torch.get_rng_state().clone()
        arguments = (Model(), Tokenizer(), ["prompt"], "pretrain", torch.device("cpu"), torch.float32)
        first = training.generate_validation_samples(*arguments, max_new_tokens=8, sample_seed=17)
        torch.testing.assert_close(torch.get_rng_state(), state)
        second = training.generate_validation_samples(*arguments, max_new_tokens=8, sample_seed=17)
        self.assertEqual(first, second)
        self.assertEqual([item["decoding"] for item in first], ["greedy", "sampled"])
        self.assertEqual(first[0]["max_token_run"], 8)
        self.assertEqual(first[1]["seed"], 17)
        self.assertNotEqual(first[0]["reply"], first[1]["reply"])

    def fixture(self, root, stage, checkpointing):
        config = root / "config.yaml"
        config.write_text(yaml.safe_dump({
            "model": {"vocab_size": 512, "d_model": 16, "n_layers": 1, "n_heads": 2,
                      "n_kv_heads": 1, "d_ff_fast": 32, "chunk_size": 4},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": stage, "micro_batch_size": 1, "grad_accum_steps": 1,
                         "max_steps": 1, "seq_len": 8, "gradient_checkpointing": checkpointing},
        }), encoding="utf-8")
        return config

    def test_disabled_surprise_is_logged_explicitly_without_changing_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root, "pretrain", False)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                checkpoint = training.train(str(config), output_ckpt=str(root / "base.pt"))
            self.assertIn("Stage: pretrain", output.getvalue())
            self.assertIn("Surprise Loss: disabled", output.getvalue())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["metrics"]["surprise_loss"], 0.0)
            self.assertFalse(saved["metrics"]["surprise_loss_measured"])

    def test_meta_oom_does_not_retry_ineffective_per_layer_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root, "meta", True)
            with mock.patch.object(training.uTTTModel, "forward", side_effect=torch.OutOfMemoryError("injected")) as forward:
                with self.assertRaisesRegex(RuntimeError, "all effective checkpoint/cache mitigations"):
                    training.train(str(config), output_ckpt=str(root / "meta.pt"))
            self.assertEqual(forward.call_count, 1)
            self.assertFalse((root / "meta.pt").exists())


if __name__ == "__main__":
    unittest.main()

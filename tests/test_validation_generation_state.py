"""Raw generation diagnostics must leave training weights, mode and RNG intact."""
import unittest
from unittest import mock

try:
    import torch
    from model.uttt_model import uTTTConfig, uTTTModel
    from train_5090 import generate_validation_samples
except ImportError:
    torch = None


class Tokenizer:
    EOS_ID = 2

    def encode(self, text, **kwargs):
        return [1, 6, 7]

    def decode(self, values):
        return " ".join(map(str, values))


@unittest.skipIf(torch is None, "PyTorch dependencies unavailable")
class ValidationGenerationStateTests(unittest.TestCase):
    def model(self):
        config = uTTTConfig(vocab_size=32, d_model=8, n_layers=1, n_heads=2, n_kv_heads=1,
                            d_ff_fast=16, max_seq_len=64, sliding_window_size=16,
                            gradient_checkpointing=False, dropout=.25,
                            repetition_penalty=1.7, dspark_speculative_k=4)
        torch.manual_seed(29)
        return uTTTModel(config).train()

    def test_controls_are_raw_ar_and_one_readonly_state_is_reused(self):
        model = self.model()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        states = []

        def generate(inputs, state, **kwargs):
            self.assertFalse(model.training)
            self.assertEqual(model.config.repetition_penalty, 1.0)
            self.assertEqual(kwargs["speculative_draft_k"], 0)
            self.assertFalse(kwargs["use_cuda_graph"])
            states.append(state)
            # The expanded momentum view occupies one scalar, not a full MLP.
            self.assertEqual(state.s_up.untyped_storage().nbytes(), 4)
            torch.rand(3)
            yield 8
            yield 2

        with mock.patch.object(model, "stream_generate", generate), \
                mock.patch.object(model, "init_fast_state", side_effect=AssertionError("unneeded clone")), \
                mock.patch.object(model.fast_weight_pool, "readonly_state", wraps=model.fast_weight_pool.readonly_state) as readonly:
            result = generate_validation_samples(model, Tokenizer(), ["first", "second"], "pretrain",
                                                 torch.device("cpu"), torch.float32, max_new_tokens=4)
        self.assertEqual(readonly.call_count, 1)
        self.assertEqual(len(states), 4)
        self.assertTrue(all(value is states[0] for value in states))
        self.assertTrue(model.training)
        self.assertEqual(model.config.repetition_penalty, 1.7)
        self.assertTrue(all(item["generation_backend"] == "standard_autoregressive" for item in result))
        self.assertTrue(all(item["repetition_penalty"] == 1.0 for item in result))
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_exception_restores_config_mode_and_rng(self):
        model = self.model()
        for originally_training in (False, True):
            model.train(originally_training)
            rng = torch.get_rng_state().clone()

            def fail(*args, **kwargs):
                self.assertFalse(model.training)
                torch.rand(3)
                raise RuntimeError("generation failure")

            with mock.patch.object(model, "stream_generate", fail):
                with self.assertRaisesRegex(RuntimeError, "generation failure"):
                    generate_validation_samples(model, Tokenizer(), ["prompt"], "pretrain",
                                                torch.device("cpu"), torch.float32, max_new_tokens=4)
            self.assertEqual(model.training, originally_training)
            self.assertEqual(model.config.repetition_penalty, 1.7)
            torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_real_dropout_model_samples_are_repeatable_without_modifying_weights(self):
        model = self.model()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        arguments = (model, Tokenizer(), ["prompt"], "pretrain", torch.device("cpu"), torch.float32)
        first = generate_validation_samples(*arguments, max_new_tokens=8, sample_seed=53)
        second = generate_validation_samples(*arguments, max_new_tokens=8, sample_seed=53)
        self.assertEqual(first, second)
        self.assertTrue(model.training)
        self.assertEqual(model.config.dspark_speculative_k, 4)
        self.assertEqual(model.config.repetition_penalty, 1.7)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

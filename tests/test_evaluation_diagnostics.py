"""Evaluation keeps raw model repetition observable beside decoding controls."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy.training_diagnostics import generation_summary, repeated_ngram_continuations

try:
    import torch
    from deploy import evaluate_model as evaluation
    from model.uttt_model import uTTTConfig, uTTTModel
except ImportError:
    torch = None


class EvaluationMetricTests(unittest.TestCase):
    def test_no_repeat_candidates_follow_matching_prefix_and_allow_short_histories(self):
        self.assertEqual(repeated_ngram_continuations([6, 7, 8, 6, 7], 3), {8})
        self.assertEqual(repeated_ngram_continuations([6, 6], 2), {6})
        self.assertEqual(repeated_ngram_continuations([6, 7], 4), set())
        self.assertEqual(repeated_ngram_continuations([6, 7], 1), {6, 7})
        with self.assertRaises(ValueError):
            repeated_ngram_continuations([6], 0)

    def test_summary_distinguishes_eos_and_token_budget_without_quality_claim(self):
        rows = [dict(eos_generated=True, stop_reason='eos', generated_tokens=3,
                     token_4gram_repetition=0, text_4gram_repetition=.2, exact_match=True),
                dict(eos_generated=False, stop_reason='token_limit', generated_tokens=8,
                     token_4gram_repetition=.8, text_4gram_repetition=.6, exact_match=False)]
        result = generation_summary(rows)
        self.assertEqual(result['eos_rate'], .5)
        self.assertEqual(result['token_limit_rate'], .5)
        self.assertEqual(result['accepted_exact_matches'], 1)
        self.assertEqual(result['mean_generated_tokens'], 5.5)


@unittest.skipIf(torch is None, 'PyTorch evaluation dependencies unavailable')
class EvaluationDecodingTests(unittest.TestCase):
    def fixture(self, stage='sft'):
        config = uTTTConfig(vocab_size=128, d_model=8, n_layers=1, n_heads=1,
                            n_kv_heads=1, d_ff_fast=16, max_seq_len=256,
                            gradient_checkpointing=False, repetition_penalty=1.8)
        model = uTTTModel(config).eval().requires_grad_(False)

        class Tokenizer:
            EOS_ID, USER_ID, ASSISTANT_ID = 2, 4, 5

            def __init__(self):
                self.prompts = []

            def encode(self, text, **kwargs):
                self.prompts.append(text)
                return [1, 6]

            def decode(self, tokens):
                return ''.join('a' if token == 6 else '' for token in tokens)

        def stream(inputs, state, max_new_tokens, temperature, top_p, **kwargs):
            recent = []
            for _ in range(max_new_tokens):
                logits = torch.full((1, config.vocab_size), -80.)
                logits[:, 6], logits[:, 2] = 8., 0.
                token = model._sample_next_token(logits, temperature=temperature, top_p=top_p,
                                                 recent_tokens=recent,
                                                 repetition_penalty=config.repetition_penalty)
                number = int(token.item())
                if number in (4, 5):
                    break
                recent.append(number)
                yield number
                if number == 2:
                    break

        model.stream_generate = stream
        manager = SimpleNamespace(model=model, config=config, device=torch.device('cpu'),
                                  dtype=torch.float32, training_metadata={'training_stage': stage})
        return manager, Tokenizer()

    def run_evaluation(self, manager, tokenizer, tasks, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(evaluation, 'DualTrackCheckpointManager', return_value=manager), \
                    mock.patch.object(evaluation, 'build_tokenizer', return_value=tokenizer), \
                    contextlib.redirect_stdout(io.StringIO()):
                return evaluation.evaluate('fake.pt', tasks, Path(directory) / 'report.json',
                                           max_new_tokens=8, **kwargs)

    def test_raw_scores_remain_primary_when_controlled_decoding_reduces_repetition(self):
        manager, tokenizer = self.fixture()
        report = self.run_evaluation(manager, tokenizer, [{'prompt': 'answer', 'accepted': ['aa']}],
                                     controlled=True, controlled_ngram=2)
        self.assertEqual(report['results'][0]['reply'], 'aaaaaaaa')
        self.assertEqual(report['results'][0]['stop_reason'], 'token_limit')
        self.assertEqual(report['basic_instruction_matches'], 0)
        controlled = report['profile_results']['controlled_sampled'][0]
        self.assertEqual(controlled['reply'], 'aa')
        self.assertEqual(controlled['stop_reason'], 'eos')
        self.assertTrue(controlled['exact_match'])
        self.assertEqual(manager.config.repetition_penalty, 1.8)
        self.assertNotIn('_sample_next_token', manager.model.__dict__)
        self.assertEqual(report['results'][0]['decoding_parameters']['repetition_penalty'], 1.)

    def test_pretraining_defaults_to_continuations_without_a_chat_template(self):
        manager, tokenizer = self.fixture('pretrain')
        report = self.run_evaluation(manager, tokenizer, None)
        self.assertEqual(report['default_evaluation_mode'], 'continuation')
        self.assertTrue(all('<|user|>' not in prompt for prompt in tokenizer.prompts))
        self.assertEqual(report['basic_instruction_total'], 0)
        self.assertEqual(len(report['results']), len(evaluation.DEFAULT_CONTINUATIONS))

    def test_sampled_results_are_repeatable_and_training_rng_is_restored(self):
        manager, tokenizer = self.fixture()
        torch.manual_seed(193)
        before = torch.get_rng_state().clone()
        first = self.run_evaluation(manager, tokenizer, [{'prompt': 'answer'}], seed=17)
        torch.testing.assert_close(torch.get_rng_state(), before)
        second = self.run_evaluation(manager, tokenizer, [{'prompt': 'answer'}], seed=17)
        self.assertEqual(first['profile_results'], second['profile_results'])

    def test_policy_restores_sampler_after_failure_and_never_blocks_eos(self):
        manager, _ = self.fixture()
        profile = dict(name='test', repetition_penalty=1., no_repeat_ngram_size=1)
        with evaluation.decoding_policy(manager.model, profile, 2):
            logits = torch.full((1, 128), -80.)
            logits[:, 2] = 10.
            result = manager.model._sample_next_token(logits, temperature=0, recent_tokens=[2])
            self.assertEqual(int(result.item()), 2)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with evaluation.decoding_policy(manager.model, profile, 2):
                raise RuntimeError('injected')
        self.assertNotIn('_sample_next_token', manager.model.__dict__)

    def test_role_boundary_is_recorded_separately_from_a_missing_eos(self):
        manager, tokenizer = self.fixture()

        def stream(*args, temperature, top_p, **kwargs):
            scores = torch.full((1, 128), -80.)
            scores[:, 4] = 80.
            manager.model._sample_next_token(scores, temperature=temperature, top_p=top_p)
            return iter(())

        manager.model.stream_generate = stream
        report = self.run_evaluation(manager, tokenizer, [{'prompt': 'answer'}])
        self.assertEqual(report['results'][0]['stop_reason'], 'role_boundary')
        self.assertFalse(report['results'][0]['eos_generated'])
        self.assertEqual(report['results'][0]['generated_tokens'], 0)


if __name__ == '__main__':
    unittest.main()

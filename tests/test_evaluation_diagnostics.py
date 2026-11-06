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
class AnswerMatchingTests(unittest.TestCase):
    def test_case_conversion_and_copy_contracts_are_case_sensitive(self):
        for task, expected, wrong in [('uppercase', 'ABC', 'abc'), ('lowercase', 'abc', 'ABC'),
                                      ('copy', 'aBc', 'ABC'), ('reverse', 'cBa', 'cba'),
                                      ('extract', 'CaseValue', 'casevalue')]:
            item = {'task': task, 'accepted': [expected]}
            with self.subTest(task=task):
                self.assertEqual(evaluation.answer_match_mode(item), 'strict_text')
                self.assertTrue(evaluation.answer_matches(' \n' + expected + '\t', item))
                self.assertFalse(evaluation.answer_matches(wrong, item))
                self.assertFalse(evaluation.answer_matches(expected + '.', item))

    def test_arithmetic_and_lengths_do_not_accept_extra_punctuation_or_explanation(self):
        for task in ('add', 'subtract', 'multiply', 'length'):
            item = {'task': task, 'accepted': ['12']}
            with self.subTest(task=task):
                self.assertTrue(evaluation.answer_matches(' 12\n', item))
                self.assertFalse(evaluation.answer_matches('12.', item))
                self.assertFalse(evaluation.answer_matches('The answer is 12', item))

    def test_sort_accepts_json_layout_but_retains_order_types_and_case(self):
        item = {'task': 'sort', 'accepted': ['[1,2,3]']}
        self.assertEqual(evaluation.answer_match_mode(item), 'json')
        self.assertTrue(evaluation.answer_matches(' \n[1, 2,\n3] ', item))
        for wrong in ('[true,2,3]', '[1.0,2,3]', '["1",2,3]', '[3,2,1]', '[1,2]', '[1,2,3].'):
            with self.subTest(reply=wrong):
                self.assertFalse(evaluation.answer_matches(wrong, item))
        nested = {'match_mode': 'json', 'accepted': ['{"a":[true,null],"b":"Case"}']}
        self.assertTrue(evaluation.answer_matches('{"b":"Case", "a":[true, null]}', nested))
        self.assertFalse(evaluation.answer_matches('{"a":[1,null],"b":"Case"}', nested))
        self.assertFalse(evaluation.answer_matches('{"a":[true,null],"b":"case"}', nested))

    def test_nonstandard_or_ambiguous_json_cannot_match(self):
        for reply in ('NaN', 'Infinity', '1e999', '{"a":1,"a":2}', '[] trailing'):
            with self.subTest(reply=reply):
                self.assertFalse(evaluation.answer_matches(reply, {'match_mode': 'json', 'accepted': ['[]']}))
        with self.assertRaisesRegex(ValueError, 'Accepted answer'):
            evaluation.answer_matches('[]', {'match_mode': 'json', 'accepted': ['not JSON']})

    def test_explicit_modes_override_task_defaults_and_unknown_modes_fail(self):
        item = {'task': 'uppercase', 'match_mode': 'legacy_casefold', 'accepted': ['ABC']}
        self.assertTrue(evaluation.answer_matches('abc!', item))
        self.assertEqual(evaluation.answer_match_mode(item), 'legacy_casefold')
        strict = {'match_mode': 'strict_text', 'accepted': ['Hello']}
        self.assertFalse(evaluation.answer_matches('hello', strict))
        for mode in ('unknown', '', None, True, 7):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Unknown answer match_mode'):
                evaluation.answer_matches('ABC', {'task': 'uppercase', 'match_mode': mode, 'accepted': ['ABC']})

    def test_legacy_basic_probe_scoring_is_preserved(self):
        for reply in ('hello', 'HELLO!', ' Hello。\n'):
            self.assertTrue(evaluation.answer_matches(reply, {'accepted': ['Hello', 'Hi']}))
        self.assertTrue(evaluation.answer_matches('你好！', {'accepted': ['你好']}))
        self.assertFalse(evaluation.answer_matches('Well, Hello', {'accepted': ['Hello']}))


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

    def test_boundary_comparison_keeps_fixed_prompt_as_primary_and_labels_legacy(self):
        manager, tokenizer = self.fixture()
        report = self.run_evaluation(manager, tokenizer, [{'prompt': 'answer', 'accepted': ['aa']}],
                                     compare_prompt_boundaries=True)
        self.assertIn('<|user|> answer\n<|assistant|>', tokenizer.prompts)
        self.assertIn('<|user|> answer\n<|assistant|> ', tokenizer.prompts)
        self.assertEqual(report['results'], report['profile_results']['raw_greedy'])
        self.assertEqual(report['results'][0]['prompt_boundary'], 'assistant_role')
        legacy = report['profile_results']['legacy_space_raw_greedy'][0]
        self.assertEqual(legacy['prompt_boundary'], 'legacy_assistant_space')
        self.assertEqual(legacy['decoding_parameters']['repetition_penalty'], 1.)
        self.assertIsNone(legacy['seed'])
        self.assertTrue(report['prompt_boundary_comparison']['same_checkpoint'])
        self.assertEqual(report['prompt_boundary_comparison']['decoding'], 'unpenalized_greedy')

    def test_results_record_task_specific_matching_policy(self):
        manager, tokenizer = self.fixture()
        tokenizer.decode = lambda tokens: 'abc'
        report = self.run_evaluation(manager, tokenizer, [
            {'prompt': 'uppercase probe', 'task': 'uppercase', 'accepted': ['ABC']},
            {'prompt': 'legacy probe', 'accepted': ['ABC']},
        ], greedy_only=True)
        self.assertEqual(report['results'][0]['answer_match_mode'], 'strict_text')
        self.assertFalse(report['results'][0]['exact_match'])
        self.assertEqual(report['results'][1]['answer_match_mode'], 'legacy_casefold')
        self.assertTrue(report['results'][1]['exact_match'])

    def test_unknown_answer_policy_rejects_before_generation(self):
        manager, tokenizer = self.fixture()
        with mock.patch.object(manager.model, 'stream_generate') as generate:
            with self.assertRaisesRegex(ValueError, 'Unknown answer match_mode'):
                self.run_evaluation(manager, tokenizer, [{'prompt': 'answer', 'match_mode': 'unsupported'}])
            generate.assert_not_called()

    def test_greedy_monitor_omits_sampling_and_keeps_boundary_ablation(self):
        manager, tokenizer = self.fixture()
        report = self.run_evaluation(manager, tokenizer, [{'prompt': 'answer'}],
                                     greedy_only=True, compare_prompt_boundaries=True)
        self.assertEqual(set(report['profile_results']), {'raw_greedy', 'legacy_space_raw_greedy'})
        self.assertTrue(all(row['seed'] is None for rows in report['profile_results'].values() for row in rows))
        with self.assertRaisesRegex(ValueError, 'controlled'):
            self.run_evaluation(manager, tokenizer, [{'prompt': 'answer'}], greedy_only=True, controlled=True)

    def test_boundary_comparison_does_not_add_a_space_to_continuation_prompts(self):
        manager, tokenizer = self.fixture('pretrain')
        report = self.run_evaluation(manager, tokenizer, [{'prompt': 'continue here'}],
                                     compare_prompt_boundaries=True)
        self.assertEqual(tokenizer.prompts, ['continue here'])
        self.assertEqual(report['profile_results']['legacy_space_raw_greedy'], [])
        self.assertEqual(report['results'][0]['prompt_boundary'], 'continuation')

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

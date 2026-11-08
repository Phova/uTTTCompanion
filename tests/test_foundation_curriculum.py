"""Verify lesson answers, semantic exclusions, and complete bilingual families."""
from collections import Counter, defaultdict
import datetime
import json
from pathlib import Path
import tempfile
import unittest

from deploy.evaluate_chat_curriculum import restricted_python_check
from deploy.prepare_foundation_curriculum import (
    CODE_BEHAVIORS, canonical_key, coding_lesson, coding_specs, collect_reference_replay, generate,
    load_exclusions, reasoning_lesson, reasoning_reference,
)
from deploy.prepare_recovery_curriculum import parse_conversation


class FoundationCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.report = generate(count=512, seed=20261109)

    def test_semantic_families_bind_both_languages_and_answer_formats(self):
        self.assertEqual(Counter(row['category'] for row in self.rows), {'reasoning': 320, 'coding': 192})
        families = defaultdict(list)
        for row in self.rows:
            families[row['family']].append(row)
            self.assertEqual(parse_conversation(row['text']), [(row['prompt'], row['answer'])])
            self.assertEqual(row['text'].count('<|eos|>'), 1)
            self.assertTrue(row['reference_verified'])
        self.assertEqual(len(families), 128)
        for rows in families.values():
            self.assertEqual(len(rows), 4)
            self.assertEqual({(row['language'], row['response_style']) for row in rows},
                             {(lang, style) for lang in ('zh', 'en') for style in ('direct', 'brief_explanation')})
            self.assertEqual(len({row['semantic_key'] for row in rows}), 1)
            self.assertEqual(len({json.dumps(row['arguments'], sort_keys=True) for row in rows}), 1)

    def test_numeric_calendar_and_boolean_references_have_independent_answers(self):
        cases = [('add', {'a': 8, 'b': 6}, '14'), ('subtract', {'a': 0, 'b': 5}, '-5'),
                 ('multiply', {'a': 9, 'b': 0}, '0'), ('divide', {'a': 27, 'b': 3}, '9'),
                 ('max', {'a': 3, 'b': 18}, '18'), ('min', {'a': 3, 'b': 18}, '3'),
                 ('quantity', {'initial': 2, 'gained': 5, 'removed': 4}, '3'),
                 ('order', {'values': [8, 2, 5], 'query': 'max'}, '8')]
        for operation, arguments, expected in cases:
            for language in ('zh', 'en'):
                for explain in (False, True):
                    with self.subTest(operation=operation, language=language, explain=explain):
                        prompt, answer, final = reasoning_lesson(operation, arguments, language, explain, 27)
                        self.assertEqual(final, expected)
                        self.assertTrue(answer.endswith(('答案：' if language == 'zh' else 'Answer: ') + expected) if explain
                                        else answer == expected)
                        self.assertTrue(prompt)
        for hour, minute, delta in ((23, 55, 10), (0, 0, 1), (12, 45, 30)):
            expected = (datetime.datetime(2020, 1, 1, hour, minute) + datetime.timedelta(minutes=delta)).strftime('%H:%M')
            self.assertEqual(reasoning_reference('clock', dict(hour=hour, minute=minute, delta=delta)), expected)
        self.assertEqual(reasoning_reference('weekday', {'weekday': 'Friday', 'offset': 3}, 'zh'), '星期一')
        truth = {'and': (False, False, False, True), 'or': (False, True, True, True),
                 'xor': (False, True, True, False)}
        for operation, expected in truth.items():
            actual = [reasoning_reference('boolean', dict(operator=operation, left=a, right=b))
                      for a, b in ((False, False), (False, True), (True, False), (True, True))]
            self.assertEqual(actual, list(map(str, expected)))

    def test_code_templates_cover_boundaries_without_executing_generated_code(self):
        specifications = coding_specs()
        self.assertEqual(set(specifications), set(CODE_BEHAVIORS))
        for behavior, parameters in specifications.items():
            for arguments in (parameters[0], parameters[len(parameters) // 2], parameters[-1]):
                lesson = coding_lesson(behavior, arguments)
                cases = [{'args': values, 'expected': lesson['reference'](*values)} for values in lesson['inputs']]
                result = restricted_python_check(lesson['code'], lesson['function'], cases)
                with self.subTest(behavior=behavior, arguments=arguments):
                    self.assertTrue(result['passed'], result)
                    self.assertFalse(result['generated_code_executed'])
        self.assertEqual(coding_lesson('nth_default', {'index': -2, 'default': None})['reference']([1]), None)
        self.assertEqual(coding_lesson('slice_text', {'side': 'prefix', 'length': 0})['reference']('abcd'), '')
        self.assertEqual(coding_lesson('safe_divide', {'default': -1})['reference'](3, 0), -1)
        self.assertIsNone(coding_lesson('lookup_default', {'default': -1})['reference']({'a': None}, 'a'))

    def test_generated_code_obeys_requested_response_format_and_function_cases(self):
        for row in self.rows:
            if row['category'] != 'coding':
                continue
            check = row['code_check']
            self.assertTrue(restricted_python_check(row['answer'], check['function'], check['cases'])['passed'])
            if row['requested_format'] == 'code_only':
                self.assertTrue(row['answer'].startswith('def '))
                self.assertNotIn('```', row['answer'])
            else:
                self.assertEqual(row['answer'].count('```'), 2)
            if row['language'] == 'zh':
                self.assertIn('调用示例', row['prompt'])
                self.assertIn('内置函数', row['prompt'])

    def test_typed_and_legacy_exclusions_remove_whole_parameter_families(self):
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(cache.resolve()))
            path = Path(directory) / 'heldout.jsonl'
            records = [{'category': 'reasoning', 'foundation_operation': 'add', 'arguments': {'a': 2, 'b': 3},
                        'prompt': 'reserved wording'},
                       {'category': 'coding', 'foundation_behavior': 'affine', 'arguments': {'a': 3, 'b': 4}},
                       {'family': 'dev20261108:reason:sum'}]
            path.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8')
            blocked, _, _, hashes = load_exclusions([path])
            keys = {canonical_key('reasoning', 'add', {'a': 3, 'b': 2}),
                    canonical_key('coding', 'affine', {'a': 3, 'b': 4}),
                    canonical_key('reasoning', 'add', {'a': 5, 'b': 7})}
            self.assertTrue(keys <= blocked)
            self.assertTrue(any(canonical_key('reasoning', 'add', {'a': 2, 'b': value}) not in blocked
                                for value in range(21)), 'Individual small operands must remain learnable')
            rows, report = generate(count=512, seed=20261109, exclude_paths=[path])
            self.assertFalse(keys & {row['semantic_key'] for row in rows})
            self.assertIn(str(path.resolve()), hashes)
            self.assertGreater(report['rejected_families']['reasoning:excluded_family'], 0)

    def test_new_evaluation_parameter_keys_are_absent_from_candidates(self):
        folder = Path(__file__).resolve().parents[1] / 'deploy'
        blocked = set()
        for name in ('foundation_development_tasks_20261108.jsonl', 'foundation_holdout_tasks_20261108.jsonl'):
            blocked.update(json.loads(line)['semantic_key'] for line in (folder / name).read_text(encoding='utf-8').splitlines())
        self.assertFalse(blocked & {row['semantic_key'] for row in self.rows})

    def test_repeat_seed_is_reproducible_and_invalid_or_excessive_capacity_fails(self):
        rows, report = generate(count=512, seed=20261109)
        self.assertEqual(rows, self.rows)
        self.assertEqual(report, self.report)
        for count in (0, 31, 33, True):
            with self.assertRaises(ValueError):
                generate(count=count)
        with self.assertRaisesRegex(ValueError, 'Insufficient distinct semantic families'):
            generate(count=10**7)

    def test_replay_recomputes_answers_preserves_groups_and_never_duplicates(self):
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            path = Path(directory) / 'old.jsonl'
            row = {'source': 'uttt:test', 'task': 'uppercase', 'arguments': {'text': 'aBc'},
                   'language': 'zh', 'validation_group_id': 'original-family',
                   'text': '<|user|> 转成大写：aBc\n<|assistant|> ABC<|eos|>'}
            path.write_text('\n'.join(json.dumps(row) for _ in range(2)), encoding='utf-8')
            replay, proof = collect_reference_replay(path, 1)
            self.assertEqual(replay[0]['family'], 'original-family')
            self.assertEqual(replay[0]['text'], row['text'])
            self.assertTrue(replay[0]['reference_verified'])
            self.assertEqual(proof['rows'], 1)
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                collect_reference_replay(path, 2)
            row['text'] = row['text'].replace(' ABC<', ' abc<')
            path.write_text(json.dumps(row), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'failed verification'):
                collect_reference_replay(path, 1)


if __name__ == '__main__':
    unittest.main()

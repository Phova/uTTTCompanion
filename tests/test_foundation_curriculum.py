"""Verify lesson answers, semantic exclusions, and complete bilingual families."""
from collections import Counter, defaultdict
import ast
import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import re
import unittest

from deploy.evaluate_chat_curriculum import restricted_python_check
from deploy.prepare_foundation_curriculum import (
    CODE_BEHAVIORS, canonical_key, coding_lesson, coding_specs, collect_reference_replay, generate,
    load_exclusions, reasoning_lesson, reasoning_reference,
)
from deploy.prepare_recovery_curriculum import parse_conversation
from deploy.prepare_chat_curriculum import prepare, read_jsonl


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

    def test_commutative_presentations_balance_order_without_changing_truth_or_family(self):
        arguments = {'a': 2, 'b': 9}
        for seed in (27, 20261109):
            for operation, expected in (('add', '11'), ('multiply', '18'), ('max', '9'), ('min', '2')):
                orders = Counter()
                key = canonical_key('reasoning', operation, arguments)
                for language in ('en', 'zh'):
                    for explain in (False, True):
                        prompt, answer, final = reasoning_lesson(operation, arguments, language, explain, seed)
                        presented = tuple(map(int, re.findall(r'\d+', prompt)))
                        orders[presented] += 1
                        self.assertEqual(final, expected)
                        self.assertEqual(arguments, {'a': 2, 'b': 9})
                        self.assertEqual(canonical_key('reasoning', operation, arguments), key)
                        if explain:
                            a, b = presented
                            symbol = {'add': '+', 'multiply': '*'}.get(operation)
                            step = (f'{a} {symbol} {b} = {expected}.' if symbol else
                                    f'{a} {">" if a > b else "<"} {b}.')
                            self.assertTrue(answer.startswith(step), answer)
                        else:
                            self.assertEqual(answer, expected)
                self.assertEqual(orders, {(2, 9): 2, (9, 2): 2})

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
        families = defaultdict(list)
        for row in self.rows:
            if row['category'] != 'coding':
                continue
            check = row['code_check']
            families[row['family']].append(row)
            self.assertTrue(restricted_python_check(row['answer'], check['function'], check['cases'])['passed'])
            if row['requested_format'] == 'code_only':
                self.assertTrue(row['answer'].startswith('def '))
                self.assertNotIn('```', row['answer'])
                code = row['answer']
                self.assertIn('只输出Python代码' if row['language'] == 'zh' else 'Return Python code only', row['prompt'])
            else:
                self.assertEqual(row['answer'].count('```'), 2)
                code = row['answer'].split('```python\n')[1].split('\n```')[0]
                self.assertNotIn('只输出Python代码', row['prompt'])
                self.assertNotIn('Return Python code only', row['prompt'])
            definition = ast.parse(code).body[0]
            self.assertEqual(definition.name, check['function'])
            signature = code.splitlines()[0].removeprefix('def ').removesuffix(':')
            self.assertIn(signature, row['prompt'])
            base = coding_lesson(row['foundation_behavior'], row['arguments'])
            base_definition = ast.parse(base['code']).body[0]
            definition.name = base_definition.name
            self.assertEqual(ast.dump(definition), ast.dump(base_definition))
            self.assertEqual(check['cases'], [{'args': values, 'expected': base['reference'](*values)} for values in base['inputs']])
            if row['language'] == 'zh':
                self.assertIn('调用示例', row['prompt'])
                self.assertIn('内置函数', row['prompt'])
        for rows in families.values():
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row['code_check']['function'] for row in rows}), 4)
            self.assertEqual(len({row['semantic_key'] for row in rows}), 1)
            self.assertEqual(len({json.dumps(row['arguments'], sort_keys=True) for row in rows}), 1)
        self.assertEqual(self.report['reference_code_case_checks'],
                         sum(len(row['code_check']['cases']) for row in self.rows if row['category'] == 'coding'))

    def test_selection_priority_preserves_elementary_arithmetic_without_changing_groups(self):
        for row in self.rows:
            elementary = row['category'] == 'reasoning' and row['task'] in {'add', 'subtract', 'multiply', 'divide', 'max', 'min'}
            self.assertEqual(row['selection_priority'], 0 if elementary else 1)
            self.assertEqual(row['family'], 'foundation:' + canonical_key(row['category'], row['task'], row['arguments']))
            self.assertEqual(row['foundation_skill'], row['category'] + ':' + row['task'])

    def test_default_lesson_content_retains_previous_checksum_apart_from_skill_metadata(self):
        previous_rows = [{key: value for key, value in row.items() if key != 'foundation_skill'} for row in self.rows]
        payload = json.dumps(previous_rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         '28c2c7e77080fd13a448d50acea0016d0a52187903740e351efb2420cd5b69cd')

    def test_optional_code_only_mode_keeps_four_distinct_verified_presentations_per_family(self):
        rows, report = generate(count=512, seed=20261109, coding_code_only=True)
        self.assertTrue(report['coding_code_only'])
        self.assertEqual(report['rows'], self.report['rows'])
        self.assertEqual(report['distinct_families'], self.report['distinct_families'])
        self.assertEqual({row['semantic_key'] for row in rows}, {row['semantic_key'] for row in self.rows})
        self.assertEqual([row for row in rows if row['category'] == 'reasoning'],
                         [row for row in self.rows if row['category'] == 'reasoning'])
        families = defaultdict(list)
        for row in rows:
            if row['category'] != 'coding':
                continue
            families[row['family']].append(row)
            self.assertEqual(row['requested_format'], 'code_only')
            self.assertNotIn('```', row['answer'])
            self.assertTrue(row['answer'].startswith('def '))
            definition = ast.parse(row['answer']).body[0]
            self.assertEqual(definition.name, row['code_check']['function'])
            self.assertIn(row['answer'].splitlines()[0][4:-1], row['prompt'])
            self.assertIn('只输出Python代码' if row['language'] == 'zh' else 'Return Python code only', row['prompt'])
            self.assertNotIn('先用一句话说明', row['prompt'])
            self.assertNotIn('Explain in one sentence', row['prompt'])
            check = row['code_check']
            result = restricted_python_check(row['answer'], check['function'], check['cases'])
            self.assertTrue(result['passed'], result)
            self.assertFalse(result['generated_code_executed'])
            base = coding_lesson(row['task'], row['arguments'])
            self.assertEqual(check['cases'], [{'args': values, 'expected': base['reference'](*values)} for values in base['inputs']])
        for family_rows in families.values():
            self.assertEqual(len(family_rows), 4)
            self.assertEqual(len({row['prompt'] for row in family_rows}), 4)
            self.assertEqual(len({row['code_check']['function'] for row in family_rows}), 4)
            self.assertEqual({row['language'] for row in family_rows}, {'en', 'zh'})
            self.assertEqual({row['response_style'] for row in family_rows}, {'direct', 'direct_alternate'})
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            generate(count=512, coding_code_only=1)

    def test_more_interface_names_keep_exact_counts_and_single_semantic_families(self):
        for count, variants in ((256, 4), (512, 8)):
            rows, report = generate(count=count, seed=20261109, coding_code_only=True, coding_name_variants=variants)
            self.assertEqual(len(rows), count)
            self.assertEqual(Counter(row['category'] for row in rows), {'reasoning': count * 5 // 8, 'coding': count * 3 // 8})
            self.assertEqual(report['coding_presentations_per_family'], variants * 2)
            self.assertEqual(report['coding_name_variants_per_language'], variants)
            self.assertEqual(report['distinct_families_by_category']['coding'], count * 3 // 8 // (variants * 2))
            self.assertEqual(report['distinct_families'], len({row['family'] for row in rows}))
            families = defaultdict(list)
            for row in rows:
                families[row['family']].append(row)
                if row['category'] != 'coding':
                    continue
                self.assertEqual(row['requested_format'], 'code_only')
                definition = ast.parse(row['answer']).body[0]
                self.assertEqual(definition.name, row['code_check']['function'])
                self.assertIn(row['answer'].splitlines()[0][4:-1], row['prompt'])
                base_definition = ast.parse(coding_lesson(row['task'], row['arguments'])['code']).body[0]
                definition.name = base_definition.name
                self.assertEqual(ast.dump(definition), ast.dump(base_definition))
                check = row['code_check']
                result = restricted_python_check(row['answer'], check['function'], check['cases'])
                self.assertTrue(result['passed'], result)
                self.assertFalse(result['generated_code_executed'])
            for items in families.values():
                expected = variants * 2 if items[0]['category'] == 'coding' else 4
                self.assertEqual(len(items), expected)
                self.assertEqual(len({row['semantic_key'] for row in items}), 1)
                if items[0]['category'] != 'coding':
                    continue
                self.assertEqual(len({row['prompt'] for row in items}), expected)
                self.assertEqual(len({(row['language'], row['response_style']) for row in items}), expected)
                for language in ('en', 'zh'):
                    names = {row['code_check']['function'] for row in items if row['language'] == language}
                    self.assertEqual(len(names), variants)
                self.assertEqual(len({json.dumps(row['code_check']['cases'], sort_keys=True) for row in items}), 1)
            repeated, repeated_report = generate(count=count, seed=20261109, coding_code_only=True, coding_name_variants=variants)
            self.assertEqual(rows, repeated)
            self.assertEqual(report, repeated_report)
        explicit_default, _ = generate(count=512, seed=20261109, coding_name_variants=2)
        self.assertEqual(explicit_default, self.rows)
        for variants in (0, 1, 3, 16, True):
            with self.subTest(variants=variants), self.assertRaisesRegex(ValueError, 'must be 2, 4 or 8'):
                generate(count=128, coding_code_only=True, coding_name_variants=variants)
        with self.assertRaisesRegex(ValueError, 'requires coding_code_only'):
            generate(count=128, coding_name_variants=8)
        with self.assertRaisesRegex(ValueError, 'divisible.*partial families'):
            generate(count=160, coding_code_only=True, coding_name_variants=8)

    def test_all_sixteen_interface_presentations_remain_in_one_actual_builder_partition(self):
        rows, _ = generate(count=512, seed=20261109, coding_code_only=True, coding_name_variants=8)
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            root = Path(directory)
            candidates = root / 'candidates.jsonl'
            candidates.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
            report = prepare(candidates, root / 'selected', count=128, ratios=(0, 5, 3), seed=92,
                             validation_fraction=.2, heldout_fraction=.25, heldout_count=32,
                             min_multi_turn_fraction=0, balance_key='foundation_skill', length_fn=lambda text: 16)
            selected = read_jsonl(root / 'selected/train.jsonl') + read_jsonl(root / 'selected/heldout.jsonl')
            self.assertEqual(report['group_overlap'], 0)
            partitions = defaultdict(set)
            for row in selected:
                partitions[row['family']].add(row['split'])
            self.assertTrue(all(len(splits) == 1 for splits in partitions.values()))
            self.assertEqual({row['split'] for row in selected}, {'train', 'validation', 'heldout'})

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
            self.assertEqual(replay[0]['selection_priority'], 1)
            self.assertEqual(replay[0]['foundation_skill'], 'replay:uppercase')
            self.assertEqual(proof['rows'], 1)
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                collect_reference_replay(path, 2)
            row['text'] = row['text'].replace(' ABC<', ' abc<')
            path.write_text(json.dumps(row), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'failed verification'):
                collect_reference_replay(path, 1)


if __name__ == '__main__':
    unittest.main()

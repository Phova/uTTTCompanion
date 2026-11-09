"""This round's fixed same-domain holdout, not an end-to-end unseen-parameter test.

Prior checkpoints may have seen these parameters. The next curriculum excludes
their complete canonical families. No model answers are used in these tests.
"""
from collections import Counter
import json
from pathlib import Path
import unittest

from deploy.evaluate_chat_curriculum import restricted_python_check, validate_tasks
from deploy.prepare_foundation_curriculum import canonical_key, load_exclusions
from tests.test_foundation_development_tasks import (
    DAYS, reference_function, reference_reasoning_answers, semantic_identity,
)


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'deploy/foundation_holdout_tasks_20261109.jsonl'
PREVIOUS = ('foundation_development_tasks_20261108.jsonl', 'foundation_holdout_tasks_20261108.jsonl')


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def independent_code_answer(row, values):
    """Test-owned calculations, separate from lesson code and recorded targets."""
    args, behavior = row['arguments'], row['foundation_behavior']
    value = values[0]
    if behavior == 'affine':
        product = sum([value] * abs(args['a']))
        return (product if args['a'] > 0 else -product) + args['b']
    if behavior == 'clamp':
        return sorted((args['low'], value, args['high']))[1]
    if behavior == 'interval':
        start = args['low'] + int(not args['include_low'])
        stop = args['high'] + int(args['include_high'])
        return value in range(start, stop)
    if behavior == 'mod_equals':
        return divmod(value, args['divisor'])[1] == args['remainder']
    if behavior == 'length_test':
        relation = (len(value) > args['limit']) - (len(value) < args['limit'])
        return {'eq': relation == 0, 'ge': relation >= 0, 'lt': relation < 0}[args['operator']]
    if behavior == 'nth_default':
        try:
            return value[args['index']]
        except IndexError:
            return args['default']
    if behavior == 'slice_text':
        count = min(args['length'], len(value))
        indices = range(count) if args['side'] == 'prefix' else range(len(value) - count, len(value))
        return ''.join(value[index] for index in indices)
    if behavior == 'safe_divide':
        return args['default'] if values[1] == 0 else values[0] / values[1]
    raise AssertionError('Unexpected behavior: ' + behavior)


class FoundationBalancedHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = validate_tasks(read(PATH))

    def test_fixed_mix_metadata_and_canonical_families_are_explicit(self):
        self.assertEqual(len(self.rows), 64)
        self.assertEqual(Counter(row['category'] for row in self.rows), {'reasoning': 40, 'coding': 24})
        self.assertEqual(Counter(row['language'] for row in self.rows), {'zh': 48, 'en': 16})
        self.assertEqual(Counter(row['foundation_operation'] for row in self.rows if row['category'] == 'reasoning'),
                         {operation: 4 for operation in ('add', 'subtract', 'multiply', 'divide', 'max', 'min',
                                                        'quantity', 'clock', 'weekday', 'order')})
        self.assertEqual(Counter(row['foundation_behavior'] for row in self.rows if row['category'] == 'coding'),
                         {'affine': 4, 'clamp': 4, 'interval': 3, 'mod_equals': 3, 'length_test': 3,
                          'nth_default': 3, 'slice_text': 3, 'safe_divide': 1})
        for row in self.rows:
            self.assertEqual(row['evaluation_partition'], 'parameter_holdout_20261109')
            self.assertIs(row['evaluation_only'], True)
            self.assertIs(row['used_for_model_selection'], False)
            self.assertEqual(row['parameter_seed'], 2026110901)
            self.assertIn('prior checkpoint exposure', row['evaluation_scope'])
            self.assertIn('Not new skills', row['evaluation_scope'])
            self.assertEqual(row['semantic_key'], semantic_identity(row))
            operation = row.get('foundation_operation', row.get('foundation_behavior'))
            self.assertEqual(row['semantic_key'], canonical_key(row['category'], operation, row['arguments']))
            self.assertEqual(row['family'], 'foundation:' + row['semantic_key'])
        for field in ('id', 'family', 'semantic_key', 'prompt'):
            self.assertEqual(len({row[field] for row in self.rows}), 64, field)

    def test_old_two_cohorts_are_disjoint_and_next_prep_can_exclude_complete_families(self):
        previous = [row for name in PREVIOUS for row in read(ROOT / 'deploy' / name)]
        for field in ('semantic_key', 'family', 'id', 'prompt'):
            self.assertFalse({row[field] for row in self.rows} & {row[field] for row in previous}, field)
        blocked, prompts, _, hashes = load_exclusions([PATH])
        self.assertIn(str(PATH.resolve()), hashes)
        for row in self.rows:
            self.assertIn(row['semantic_key'], blocked)
            args = dict(row['arguments'])
            operation = row.get('foundation_operation', row.get('foundation_behavior'))
            if row['category'] == 'reasoning' and operation in {'add', 'multiply', 'max', 'min'}:
                args['a'], args['b'] = args['b'], args['a']
            self.assertIn(canonical_key(row['category'], operation, args), blocked)

    def test_reasoning_answers_have_independent_references_and_same_small_domains(self):
        for row in self.rows:
            if row['category'] != 'reasoning':
                continue
            with self.subTest(id=row['id']):
                self.assertEqual(set(row['accepted']), reference_reasoning_answers(row))
                args, op = row['arguments'], row['foundation_operation']
                if op in {'add', 'subtract', 'max', 'min'}:
                    self.assertTrue(all(type(args[key]) is int and 0 <= args[key] <= 20 for key in ('a', 'b')))
                elif op == 'multiply':
                    self.assertTrue(all(0 <= args[key] <= 10 for key in ('a', 'b')))
                elif op == 'divide':
                    self.assertTrue(1 <= args['b'] <= 10 and 0 <= args['a'] // args['b'] <= 10)
                    self.assertEqual(args['a'] % args['b'], 0)
                elif op == 'quantity':
                    self.assertTrue(0 <= args['initial'] <= 10 and 0 <= args['gained'] <= 5 and 0 <= args['removed'] <= 5)
                    self.assertGreaterEqual(args['initial'] + args['gained'], args['removed'])
                elif op == 'clock':
                    self.assertTrue(0 <= args['hour'] < 24 and 0 <= args['minute'] < 60 and args['minute'] % 5 == 0)
                    self.assertTrue(1 <= args['delta'] <= 30)
                elif op == 'weekday':
                    self.assertIn(args['weekday'], DAYS)
                    self.assertTrue(1 <= args['offset'] <= 7)
                elif op == 'order':
                    self.assertEqual(len(set(args['values'])), 3)
                    self.assertTrue(all(0 <= value <= 20 for value in args['values']))

    def test_code_cases_are_independently_correct_and_reference_code_uses_exact_interface(self):
        names = set()
        for row in self.rows:
            if row['category'] != 'coding':
                continue
            with self.subTest(id=row['id']):
                check = row['code_check']
                names.add(check['function'])
                self.assertEqual(row['format_requirement'], 'code_only')
                self.assertIs(row['format_scored_by_functional_check'], False)
                self.assertIn(check['function'] + '(', row['prompt'])
                self.assertGreaterEqual(len(check['cases']), 4)
                for case in check['cases']:
                    expected = independent_code_answer(row, case['args'])
                    self.assertIs(type(case['expected']), type(expected))
                    self.assertEqual(case['expected'], expected)
                source = (f"def {check['function']}(a, b):\n    return None if b == 0 else a / b"
                          if row['foundation_behavior'] == 'safe_divide' else reference_function(row))
                result = restricted_python_check(source, check['function'], check['cases'])
                self.assertTrue(result['passed'], result)
                self.assertEqual(result['cases_passed'], len(check['cases']))
                self.assertFalse(result['generated_code_executed'])
        self.assertEqual(len(names), 8)

    def test_boundary_cases_detect_wrong_sides_inclusivity_indexing_and_zero_division(self):
        for row in self.rows:
            if row['category'] != 'coding':
                continue
            args, behavior = row['arguments'], row['foundation_behavior']
            cases = row['code_check']['cases']
            values = [case['args'][0] for case in cases]
            if behavior in {'clamp', 'interval'}:
                self.assertTrue({args['low'] - 1, args['low'], args['high'], args['high'] + 1} <= set(values))
            elif behavior == 'affine':
                self.assertIn(0, values)
                self.assertTrue(any(value < 0 for value in values) and any(value > 0 for value in values))
            elif behavior == 'mod_equals':
                self.assertIn(0, values)
                self.assertTrue(any(value < 0 for value in values))
                self.assertEqual({case['expected'] for case in cases}, {True, False})
            elif behavior == 'length_test':
                self.assertTrue({max(args['limit'] - 1, 0), args['limit'], args['limit'] + 1} <= {len(value) for value in values})
                self.assertEqual({type(value) for value in values}, {str, list})
            elif behavior == 'nth_default':
                self.assertIn([], values)
                self.assertTrue(any(-len(value) <= args['index'] < len(value) for value in values))
            elif behavior == 'slice_text':
                self.assertIn('', values)
                self.assertIn('你好呀', values)
                self.assertTrue(any(len(value) > args['length'] for value in values))
                if args['length']:
                    self.assertTrue(any(value[:args['length']] != value[-args['length']:] for value in values if len(value) > args['length']))
            elif behavior == 'safe_divide':
                self.assertTrue(any(case['args'][1] == 0 for case in cases))
                self.assertTrue(any(case['args'][1] < 0 for case in cases))
                self.assertTrue(any(case['args'][1] > 0 for case in cases))


if __name__ == '__main__':
    unittest.main()

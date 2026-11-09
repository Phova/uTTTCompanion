"""Independent checks of the explicit facts in grounded dialogue targets."""
from collections import Counter
import unittest

from deploy.prepare_grounded_dialogues import DAYS, ZH_DAYS, generate, lesson
from deploy.prepare_recovery_curriculum import parse_conversation


class GroundedDialogueTests(unittest.TestCase):
    def test_every_target_follows_stated_facts(self):
        rows, manifest = generate()
        self.assertEqual(manifest['rows'], len(rows))
        self.assertEqual(set(manifest['by_skill']), {
            'chat:available_day', 'chat:correction', 'chat:budget_choice', 'chat:listed_ingredients'})
        seen = Counter()
        contexts = {}
        for row in rows:
            args, language = row['arguments'], row['language']
            turns = parse_conversation(row['text'])
            answer = turns[-1][1]
            seen[row['family']] += 1
            skill = row['grounded_operation']
            if skill == 'available_day':
                busy = set(args['days']) - {args['free']}
                free, = set(args['days']) - busy
                expected = (ZH_DAYS if language == 'zh' else DAYS)[free]
                self.assertIn(expected, answer)
                self.assertEqual(len(turns), 2)
            elif skill == 'correction':
                expected = (ZH_DAYS if language == 'zh' else DAYS)[args['after']]
                old = (ZH_DAYS if language == 'zh' else DAYS)[args['before']]
                self.assertIn(expected, answer)
                self.assertNotIn(old, answer)
                self.assertIn(f"{args['hour']:02d}:00", answer)
                self.assertEqual(len(turns), 3)
            elif skill == 'budget_choice':
                choices = [i for i, value in enumerate(args['costs']) if value <= args['budget']]
                self.assertEqual(choices, [args['choice']])
                labels = ('笔记本', '马克杯', '收纳盒') if language == 'zh' else ('notebook', 'mug', 'storage box')
                expected = labels[choices[0]]
                self.assertIn(expected, answer)
                self.assertTrue(all(label not in answer for label in labels if label != expected))
            else:
                self.assertNotEqual(args['avoid'], args['other'])
                labels = ('组合甲', '组合乙') if language == 'zh' else ('option A', 'option B')
                expected = labels[args['choice']]
                self.assertIn(expected, answer)
                self.assertNotIn(labels[1 - args['choice']], answer)
            self.assertEqual(row['expected_choice'], expected)
            for index, (_, response) in enumerate(turns):
                context = tuple(turns[:index]) + (turns[index][0],)
                self.assertEqual(contexts.setdefault(context, response), response)
        self.assertEqual(set(seen.values()), {2})
        self.assertEqual(len(seen), manifest['semantic_families'])

    def test_reproducible_and_capacity_bounded(self):
        first, proof = generate(seed=17, max_families_per_skill=8)
        self.assertEqual((first, proof), generate(seed=17, max_families_per_skill=8))
        self.assertEqual(len(first), 64)
        self.assertEqual(set(Counter(row['foundation_skill'] for row in first).values()), {16})
        self.assertEqual(len({row['text'] for row in first}), len(first))
        self.assertNotEqual(first, generate(seed=18, max_families_per_skill=8)[0])

    def test_invalid_requests_fail(self):
        for count in (0, -1, True, '8'):
            with self.assertRaises(ValueError):
                generate(max_families_per_skill=count)
        with self.assertRaises(ValueError):
            lesson('unknown', {}, 'zh')


if __name__ == '__main__':
    unittest.main()

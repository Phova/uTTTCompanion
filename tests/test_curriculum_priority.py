"""Curator priorities change selection, never category quotas or split ownership."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import deploy.prepare_chat_curriculum as builder


class CurriculumPriorityTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.directory.name)
        self.source = self.root / "candidates.jsonl"
        self.rows = []
        for category in builder.CATEGORIES:
            for index in range(180):
                turns = [(f"Priority {category} question {index}.", f"Answer {category} {index}.")]
                if index % 2 == 0:
                    turns.append(("Could you explain that?", f"Explanation {category} {index}."))
                self.rows.append({"category": category, "language": "en", "source": "fixture:" + category,
                                  "copyright": "fixture terms", "family": f"{category}:priority:{index}",
                                  "text": builder.render_turns(turns)})

    def tearDown(self):
        self.directory.cleanup()

    def build(self, name, rows=None, observer=None, **changes):
        with self.source.open("w", encoding="utf-8", newline="\n") as stream:
            for row in self.rows if rows is None else rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        options = dict(count=60, ratios=(.6, .2, .2), seed=92, max_tokens=512,
                       validation_fraction=.2, heldout_fraction=.25, heldout_count=15,
                       min_multi_turn_fraction=.25, length_fn=lambda text: len(text) + 2)
        options.update(changes)
        output = self.root / name
        original = builder._select

        def select(pool, required, **settings):
            if observer is not None:
                observer.extend((settings["split"], row) for rows in pool.values() for row in rows)
            return original(pool, required, **settings)

        with patch.object(builder, "_select", side_effect=select):
            report = builder.prepare(self.source, output, **options)
        selected = builder.read_jsonl(output / "train.jsonl") + builder.read_jsonl(output / "heldout.jsonl")
        return output, report, selected

    @staticmethod
    def selected_identity(rows):
        return {(row["text_sha256"], row["split"]) for row in rows}

    def assert_partitions(self, rows):
        groups = defaultdict(set)
        families = defaultdict(set)
        for row in rows:
            groups[row["validation_group_id"]].add(row["split"])
            families[row["family"]].add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        self.assertEqual(len({row["text_sha256"] for row in rows}), len(rows))

    def test_absent_priority_retains_frozen_pre_feature_output_files(self):
        output, _, selected = self.build("legacy")
        # Recorded from this fixture using the unmodified selector, before the
        # feature was added. This guards the actual output, not a copied sort.
        expected = {
            "train.jsonl": "98705ffb4af1029dc538d50bdac9fdb56a70b19fe60c2706c64ab57b14031323",
            "validation.jsonl": "0adbd8229006d93399fc908e0277847b3e0cba75d7632184ef06110abbbf623f",
            "heldout.jsonl": "cb4e1e19c2a5805523bbd0039453a02aab08f98c25b987820de679ebba1baeec",
        }
        self.assertEqual({name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in expected}, expected)
        self.assertTrue(all("selection_priority" not in row for row in selected))

    def test_explicit_zero_matches_default_and_equal_priorities_remain_seeded(self):
        _, _, baseline = self.build("default")
        rows = [{**row, "selection_priority": 0} for row in self.rows]
        first, _, explicit = self.build("zero", rows)
        second, _, reordered = self.build("zero-reversed", list(reversed(rows)))
        self.assertEqual(self.selected_identity(explicit), self.selected_identity(baseline))
        self.assertEqual(self.selected_identity(reordered), self.selected_identity(explicit))
        for name in ("train.jsonl", "validation.jsonl", "heldout.jsonl"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        _, _, reseeded = self.build("other-seed", rows, seed=93)
        self.assertNotEqual(self.selected_identity(explicit), self.selected_identity(reseeded))

    def test_priority_admits_previously_unselected_candidates_without_crossing_splits(self):
        observed = []
        _, baseline_report, baseline = self.build("before", observer=observed)
        chosen_before = {row["family"] for row in baseline}
        prior_split = {row["family"]: split for split, row in observed}
        targets = set()
        for split in ("train", "validation", "heldout"):
            for category in builder.CATEGORIES:
                candidates = [row for source_split, row in observed
                              if source_split == split and row["category"] == category
                              and row["family"] not in chosen_before
                              and (category != "chat" or row["turns"] > 1)]
                self.assertTrue(candidates)
                targets.add(candidates[0]["family"])
        rows = [{**row, "selection_priority": 0 if row["family"] in targets else 7}
                for row in self.rows]
        _, after_report, after = self.build("after", rows)
        self.assertTrue(targets.issubset({row["family"] for row in after}))
        self.assertTrue(all(row["split"] == prior_split[row["family"]] for row in after))
        for split in ("training", "validation", "heldout"):
            self.assertEqual(after_report[split]["category"], baseline_report[split]["category"])
            self.assertEqual(after_report[split]["rows"], baseline_report[split]["rows"])
        self.assertEqual(Counter(row["category"] for row in after if row["split"] != "heldout"),
                         {"chat": 36, "reasoning": 12, "coding": 12})
        self.assert_partitions(after)

    def test_priority_never_displaces_required_multi_turn_conversations(self):
        rows = [{**row, "selection_priority": 50 if row["text"].count("<|assistant|>") > 1 else 0}
                for row in self.rows]
        _, report, selected = self.build("prefer-single", rows)
        for split in ("training", "validation", "heldout"):
            self.assertGreaterEqual(report[split]["multi_turn"], math.ceil(report[split]["rows"] * .25))
        self.assertTrue(any(row["selection_priority"] == 0 for row in selected))
        self.assertTrue(any(row["selection_priority"] == 50 and row["turns"] > 1 for row in selected))
        self.assertEqual(report["all_mixture"]["category"], {"chat": 36, "reasoning": 12, "coding": 12})
        self.assert_partitions(selected)

    def test_prioritized_family_variants_stay_in_one_partition(self):
        target = "reasoning:priority:1"
        rows = [{**row, "selection_priority": 0 if row["family"] == target else 9} for row in self.rows]
        original = next(row for row in rows if row["family"] == target)
        rows.append({**original, "text": builder.render_turns([
            ("Here is a differently worded condition for this family.", "The reference outcome is unchanged.")])})
        observed = []
        _, _, selected = self.build("family", rows, observer=observed)
        variants = [row for row in selected if row["family"] == target]
        self.assertEqual(len(variants), 2)
        self.assertEqual(len({row["split"] for row in variants}), 1)
        self.assertEqual(len({row["validation_group_id"] for row in variants}), 1)
        self.assertEqual(len({split for split, row in observed if row["family"] == target}), 1)
        self.assert_partitions(selected)

    def test_invalid_priority_is_rejected_before_creating_output(self):
        for index, value in enumerate((-1, True, False, 0.0, 1.5, "0", None, [], {})):
            name = f"invalid-{index}"
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "selection_priority.*nonnegative integer"):
                self.build(name, [*self.rows[:-1], {**self.rows[-1], "selection_priority": value}])
            self.assertFalse((self.root / name).exists())


if __name__ == "__main__":
    unittest.main()

"""Curriculum selection must retain conditions/EOS and isolate family identities."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from deploy.prepare_chat_curriculum import (
    CATEGORIES, complete_prefix, contextual_prompts, is_initial_generic_greeting, prepare, quotas,
    read_jsonl, render_turns, validation_group_split,
)
from deploy.prepare_recovery_curriculum import parse_conversation


class ChatCurriculumTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.directory.name)
        self.candidates = self.root / "candidates.jsonl"
        self.rows = []
        for category in CATEGORIES:
            for index in range(180):
                turns = [(f"{category} unique question {index}", f"correct answer {index}")]
                if index % 2 == 0:
                    turns.append(("Explain your previous answer briefly.", f"{category} explanation {index}"))
                self.rows.append({"category": category, "language": "en", "source": "fixture:" + category,
                                  "copyright": "retain source terms", "family": f"{category}:family:{index}",
                                  "text": render_turns(turns)})
        self.write(self.candidates, self.rows)

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def write(path, rows):
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def build(self, name="new", **changes):
        options = dict(count=60, seed=92, max_tokens=512, validation_fraction=.2,
                       heldout_fraction=.25, heldout_count=15, min_multi_turn_fraction=.25,
                       length_fn=lambda text: len(text) + 2)
        options.update(changes)
        output = self.root / name
        report = prepare(self.candidates, output, **options)
        return output, report

    def test_exact_mixture_no_upsampling_and_trainer_hash_matches_inspection_validation(self):
        output, report = self.build()
        mixture = read_jsonl(output / "train.jsonl")
        heldout = read_jsonl(output / "heldout.jsonl")
        validation = read_jsonl(output / "validation.jsonl")
        self.assertEqual(len(mixture), 60)
        self.assertEqual(Counter(row["category"] for row in mixture), {"chat": 36, "reasoning": 12, "coding": 12})
        self.assertEqual(len({row["text_sha256"] for row in mixture + heldout}), 75)
        self.assertEqual(len(heldout), 15)
        self.assertEqual(len(validation), 12)
        self.assertEqual(report["group_overlap"], 0)
        val_ids = {row["text_sha256"] for row in validation}
        for row in mixture:
            expected = validation_group_split(row["validation_group_id"], 92, .2)
            self.assertEqual(row["split"], expected)
            self.assertEqual(row["text_sha256"] in val_ids, expected == "validation")
        for split in ("training", "validation", "heldout"):
            self.assertGreaterEqual(report[split]["multi_turn"], report[split]["rows"] * .25)
        self.assertTrue(all(row["copyright"] == "retain source terms" for row in mixture + heldout))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for actual trainer split")
    def test_actual_trainer_loads_exactly_the_reported_validation_and_training_rows(self):
        from train_5090 import PersonaMetaDataset
        output, report = self.build()
        rows = read_jsonl(output / "train.jsonl")
        kwargs = dict(seq_len=512, data_path=str(output / "train.jsonl"), pack_sequences=False,
                      mode="sft", validation_fraction=.2, split_seed=92,
                      validation_group_key="validation_group_id")
        for split, name in (("train", "training"), ("validation", "validation")):
            dataset = PersonaMetaDataset(None, split=split, **kwargs)
            self.assertEqual(len(dataset), report[name]["rows"])
            self.assertEqual(set(dataset.external_texts), {row["text"] for row in rows if row["split"] == split})

    def test_reproducibility_and_input_order_do_not_change_selected_examples_or_splits(self):
        first, _ = self.build("one")
        self.write(self.candidates, list(reversed(self.rows)))
        second, _ = self.build("two")
        for name in ("train.jsonl", "validation.jsonl", "heldout.jsonl"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_real_eos_and_context_are_kept_when_only_a_complete_prefix_fits(self):
        turns = [("my initial condition", "acknowledged"), ("a follow-up question", "x" * 500)]
        text = render_turns(turns)
        length = lambda value: len(value) + 2
        prefix = complete_prefix(text, turns, length(render_turns(turns[:1])), length)
        self.assertEqual(prefix, (render_turns(turns[:1]), turns[:1], True))
        self.assertEqual(parse_conversation(prefix[0]), turns[:1])
        self.assertEqual(prefix[0].count("<|eos|>"), 1)
        self.assertIsNone(complete_prefix(text, turns, 16, length))
        self.assertNotEqual(contextual_prompts([("first context", "one"), ("you?", "two")])[-1],
                            contextual_prompts([("other context", "one"), ("you?", "two")])[-1])

    def test_over_budget_conversation_cannot_become_generic_opening_only(self):
        opening = "你好，你能帮我解答一个问题吗？"
        long = {**self.rows[0], "family": "long-opening", "text": render_turns([
            (opening, "当然，请说说你的问题。"),
            ("我想理解函数参数的传递方式。", "详细解释" * 300)])}
        complete_single = {**self.rows[0], "family": "complete-opening", "text": render_turns([
            ("你好，我想问你一个问题。", "可以，请告诉我你的问题。")])}
        self.write(self.candidates, self.rows + [long, complete_single])
        output, report = self.build()
        self.assertEqual(report["rejected"]["shortened_to_generic_opening_only"], 1)
        # A genuinely complete short opening is not rejected by this guard.
        available = sum(sum(counts.values()) for counts in report["available"].values())
        self.assertEqual(available, len(self.rows) + 1)
        selected = read_jsonl(output / "train.jsonl") + read_jsonl(output / "heldout.jsonl")
        self.assertFalse(any(row["family"] == "long-opening" for row in selected))
        self.assertTrue(is_initial_generic_greeting(opening))
        self.assertTrue(is_initial_generic_greeting("你好，我想问你一个问题。"))
        self.assertFalse(is_initial_generic_greeting("你好，我想问你一个关于Python装饰器的问题。"))

    def test_exclusion_from_jsonl_and_nested_diagnosis_and_semantic_id_applies_before_selection(self):
        blocked_rows = [self.rows[index] for index in (0, 181, 362)]
        exclusion = self.root / "old_heldout.jsonl"
        diagnosis = self.root / "diagnosis.json"
        self.write(exclusion, [{"text": blocked_rows[0]["text"]}, {"semantic_key": blocked_rows[2]["family"]}])
        diagnosis.write_text(json.dumps({"product_probes": [{"prompt": parse_conversation(blocked_rows[1]["text"])[0][0]}]}))
        output, report = self.build(exclusions=[exclusion, diagnosis])
        self.assertEqual(report["rejected"]["excluded_probe_or_family"], 3)
        selected = read_jsonl(output / "train.jsonl") + read_jsonl(output / "heldout.jsonl")
        self.assertFalse({row["family"] for row in selected} & {row["family"] for row in blocked_rows})

    def test_conflicting_answers_are_both_rejected_and_duplicates_are_not_upsampled(self):
        row = {**self.rows[0], "deterministic_answer": True}
        conflict = {**row, "family": "conflict-alias", "text": row["text"].replace("correct answer 0", "a conflicting answer")}
        duplicate = {**self.rows[1], "family": "duplicate-alias", "source": "fixture:second-source"}
        self.write(self.candidates, [row, *self.rows[1:], conflict, duplicate])
        output, report = self.build()
        self.assertEqual(report["rejected"]["conflicting_answer_for_same_context"], 2)
        self.assertEqual(report["rejected"]["duplicate_text"], 1)
        selected = read_jsonl(output / "train.jsonl") + read_jsonl(output / "heldout.jsonl")
        self.assertFalse(any(item["family"] in {row["family"], "conflict-alias"} for item in selected))
        self.assertEqual(len({item["text_sha256"] for item in selected}), len(selected))

    def test_open_question_alternate_answers_are_retained_in_one_group(self):
        topic = "Which hobby should I try after work?"
        alternatives = [
            {**self.rows[0], "family": "open-choice-a", "text": render_turns([(topic, "Try a short walk.")])},
            {**self.rows[0], "family": "open-choice-b", "text": render_turns([(topic, "You could try sketching.")])},
        ]
        self.write(self.candidates, self.rows + alternatives)
        # Select the entire pool using a private selection observer, so this
        # checks candidate acceptance/grouping rather than random quota luck.
        import deploy.prepare_chat_curriculum as builder
        from unittest.mock import patch
        observed = []
        original = builder._select
        def select(pool, required, **options):
            observed.extend(row for group in pool.values() for row in group)
            return original(pool, required, **options)
        with patch.object(builder, "_select", side_effect=select):
            _, report = self.build()
        rows = [row for row in observed if row["family"] in {"open-choice-a", "open-choice-b"}]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["validation_group_id"] for row in rows}), 1)
        self.assertNotIn("conflicting_answer_for_same_context", report["rejected"])

    def test_generic_initial_greetings_do_not_merge_unrelated_topics(self):
        import deploy.prepare_chat_curriculum as builder
        from unittest.mock import patch
        topics = [
            {**self.rows[0], "family": "greeting-cooking", "text": render_turns([
                ("你好！", "你好，今天想聊什么？"), ("我想学做一份简单晚饭。", "可以从番茄炒蛋开始。")])},
            {**self.rows[0], "family": "greeting-reading", "text": render_turns([
                ("你好！", "嗨，很高兴和你聊天。"), ("怎么建立读书习惯？", "先每天固定读十分钟。")])},
        ]
        self.write(self.candidates, self.rows + topics)
        observed = []
        original = builder._select
        def select(pool, required, **options):
            observed.extend(row for group in pool.values() for row in group)
            return original(pool, required, **options)
        with patch.object(builder, "_select", side_effect=select):
            _, report = self.build()
        rows = [row for row in observed if row["family"] in {"greeting-cooking", "greeting-reading"}]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["validation_group_id"] for row in rows}), 2)
        self.assertEqual(report["grouping"]["initial_greeting_links_skipped"], 2)
        self.assertTrue(is_initial_generic_greeting("  Hi! "))
        self.assertFalse(is_initial_generic_greeting("你好，我想学Python"))

    def test_family_and_context_aliases_cannot_cross_any_partition(self):
        # Rephrase a conversation but explicitly keep its semantic family.
        extra = {**self.rows[10], "text": self.rows[10]["text"].replace("unique question", "new phrasing")}
        # Identical contextual question in a distinct declared family connects it.
        alias = {**self.rows[20], "family": "alias-family"}
        self.write(self.candidates, self.rows + [extra, alias])
        output, _ = self.build()
        selected = read_jsonl(output / "train.jsonl") + read_jsonl(output / "heldout.jsonl")
        families, questions = {}, {}
        for row in selected:
            families.setdefault(row["family"], set()).add(row["split"])
            for context in contextual_prompts(parse_conversation(row["text"])):
                questions.setdefault(context, set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        self.assertTrue(all(len(splits) == 1 for splits in questions.values()))

    def test_insufficient_data_does_not_create_output_or_duplicate_rows(self):
        output = self.root / "shortage"
        with self.assertRaisesRegex(ValueError, "Insufficient distinct candidates"):
            self.build("shortage", count=10000)
        self.assertFalse(output.exists())
        single_turn = [{**row, "text": render_turns(parse_conversation(row["text"])[:1])} for row in self.rows]
        self.write(self.candidates, single_turn)
        with self.assertRaisesRegex(ValueError, "Insufficient distinct multi-turn"):
            self.build("no-multi")
        self.assertFalse((self.root / "no-multi").exists())

    def test_metadata_is_explicit_and_unicode_json_line_separators_are_preserved(self):
        self.write(self.candidates, [{**self.rows[0], "text": self.rows[0]["text"].replace("condition", "condition\u2028still text")}])
        self.assertEqual(len(read_jsonl(self.candidates)), 1)
        self.write(self.candidates, [{key: value for key, value in self.rows[0].items() if key != "category"}])
        with self.assertRaisesRegex(ValueError, "explicit category"):
            self.build()
        self.assertEqual(quotas(11, (.6, .2, .2)), {"chat": 7, "reasoning": 2, "coding": 2})
        for ratios in ((1, 1), (-1, 1, 1), (float("nan"), 1, 1), (0, 0, 0)):
            with self.assertRaisesRegex(ValueError, "ratios"):
                quotas(10, ratios)


if __name__ == "__main__":
    unittest.main()

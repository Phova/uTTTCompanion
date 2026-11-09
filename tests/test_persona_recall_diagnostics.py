"""Diagnostic token groups and gold-cache alignment must preserve actual positions."""
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy import diagnose_persona_recall as diagnostic

try:
    import torch
except ImportError:
    torch = None


class OffsetTokenizer:
    """ASCII fixture with explicit offsets, optional whitespace/payload BPE merge."""
    BOS_ID, EOS_ID, USER_ID, ASSISTANT_ID, PAD_ID = 1, 2, 4, 5, 0
    SPECIAL = {"<|user|>": 4, "<|assistant|>": 5, "<|eos|>": 2}

    def __init__(self, merged=False, mixed_suffix=False):
        self.backend = self
        self.merged, self.mixed_suffix = merged, mixed_suffix

    def encode(self, text, add_bos=True, add_eos=False, add_special_tokens=None):
        ids, offsets, index = [], [], 0
        while index < len(text):
            match = next(((token, value) for token, value in self.SPECIAL.items() if text.startswith(token, index)), None)
            if match:
                piece, value = match
            elif self.merged and text.startswith(" AB", index):
                piece, value = " AB", 400
            elif self.mixed_suffix and text.startswith("C READY", index):
                piece, value = "C READY", 401
            else:
                piece, value = text[index], 16 + ord(text[index])
            ids.append(value)
            offsets.append((index, index + len(piece)))
            index += len(piece)
        if add_special_tokens is not None:
            return SimpleNamespace(ids=ids, offsets=offsets)
        return ([self.BOS_ID] if add_bos else []) + ids + ([self.EOS_ID] if add_eos else [])


def row(payload="ABC", ready=False):
    return {"episode_id": "dev-" + payload, "family": "same-query-family",
            "write_text": f"<|user|> My code is {payload}.\n<|assistant|> Recorded.<|eos|>",
            "query_text": "What is my code?", "answer": payload + (" READY" if ready else ""),
            "memory_task": {"entity": "person-A", "payload": payload, "task": "reply_prefix" if ready else "recall",
                            "language": "en", "split": "dev"}}


class RecallDiagnosticLayoutTests(unittest.TestCase):
    def test_whitespace_is_separate_and_ready_eos_are_not_payload(self):
        layout = diagnostic.answer_layout(OffsetTokenizer(), row(ready=True))
        values = {position["index"]: 1.0 for position in layout["positions"]}
        summary = diagnostic.summarize_nll(layout["positions"], values)
        self.assertEqual(summary["payload_first"]["tokens"], 1)
        self.assertEqual(summary["payload_rest"]["tokens"], 2)
        self.assertEqual(summary["ready"]["tokens"], 5)
        self.assertEqual(summary["eos"]["tokens"], 1)
        self.assertEqual(summary["format_whitespace"]["tokens"], 2)
        self.assertEqual(summary["payload_all"]["total_nll"], 3.0)
        first = next(position for position in layout["positions"] if position["group"] == "payload_first")
        self.assertEqual(first["piece"], "A")
        self.assertGreater(first["index"], len(layout["prefix_ids"]))

    def test_whitespace_merged_with_payload_is_correctly_classified_by_overlap(self):
        layout = diagnostic.answer_layout(OffsetTokenizer(merged=True), row())
        first = next(position for position in layout["positions"] if position["group"] == "payload_first")
        self.assertEqual(first["piece"], " AB")
        self.assertEqual(first["index"], len(layout["prefix_ids"]))
        self.assertEqual([value["piece"] for value in layout["positions"] if value["group"] == "payload_rest"], ["C"])

    def test_bpe_token_crossing_payload_and_ready_is_disclosed_and_counted_once(self):
        layout = diagnostic.answer_layout(OffsetTokenizer(mixed_suffix=True), row(ready=True))
        self.assertEqual(len(layout["mixed_payload_ready_tokens"]), 1)
        mixed = layout["mixed_payload_ready_tokens"][0]
        self.assertEqual(next(value for value in layout["positions"] if value["index"] == mixed)["group"], "payload_rest")
        self.assertFalse(any(value["group"] == "ready" for value in layout["positions"]))

    def test_idle_guard_rejects_unknown_or_other_processes_without_external_execution(self):
        for output in ("22\n", "[N/A]\n"):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                diagnostic.guard_idle_gpu(runner=mock.Mock(return_value=SimpleNamespace(stdout=output, returncode=0)), own_pid=11)
        runner = mock.Mock(return_value=SimpleNamespace(stdout="11\n", returncode=0))
        self.assertEqual(diagnostic.guard_idle_gpu(runner=runner, own_pid=11)["other_compute_pids"], [])
        self.assertEqual(runner.call_count, 1)

    def test_oracle_is_four_separate_current_query_copy_examples(self):
        examples = diagnostic.oracle_examples([row()])
        self.assertEqual(len(examples), 4)
        self.assertEqual([value["language"] for value in examples], ["zh", "en", "zh", "en"])
        self.assertTrue(all(value["expected"] in value["query"] for value in examples))

    def test_counterfactual_requires_identical_query_and_entity(self):
        changed = row("XYZ")
        changed["query_text"] = "Different query"
        with self.assertRaisesRegex(ValueError, "same query/entity"):
            diagnostic.diagnose_episode(None, None, row(), changed, dtype=None)

    def test_default_four_cases_cover_ready_and_both_languages(self):
        rows = []
        for task in ("recall", "correction", "reply_prefix"):
            for language in ("en", "zh"):
                value = row(ready=task == "reply_prefix")
                value["episode_id"] = task + "-" + language
                value["memory_task"].update(task=task, language=language)
                rows.append(value)
        chosen = diagnostic.select_diagnostic_rows(rows, 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual({value["memory_task"]["task"] for value in chosen}, {"recall", "correction", "reply_prefix"})
        self.assertEqual({value["memory_task"]["language"] for value in chosen}, {"en", "zh"})


@unittest.skipIf(torch is None, "CPU model tests require Torch")
class RecallDiagnosticModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self):
        from model.uttt_model import uTTTConfig, uTTTModel
        torch.manual_seed(245)
        return uTTTModel(uTTTConfig(vocab_size=512, d_model=8, d_ff_fast=16, n_layers=2,
                                    n_heads=2, n_kv_heads=1, chunk_size=128, sliding_window_size=256,
                                    max_seq_len=256, dropout=0.0)).eval().requires_grad_(False)

    def test_padded_support_masks_preserve_state_and_gold_cache_matches_full_query(self):
        model, tokenizer, example = self.model(), OffsetTokenizer(), row()
        state, lengths = diagnostic.write_supports(model, tokenizer, example, dtype=torch.float32)
        padded, _ = diagnostic.write_supports(model, tokenizer, example, dtype=torch.float32, padded=True)
        self.assertLess(lengths[0], 128)
        comparison = diagnostic.compare_states(state, padded)
        self.assertTrue(comparison["step_equal"])
        self.assertLess(max(value["max_abs_difference"] for value in comparison["fields"].values()), 1e-5)
        layout = diagnostic.answer_layout(tokenizer, example)
        full = diagnostic.teacher_logits(model, state, layout["ids"], dtype=torch.float32)
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            alignment = diagnostic.cached_gold_alignment(model, state, layout, full, dtype=torch.float32)
        calls = forward.call_args_list
        self.assertEqual(calls[0].args[0].tolist(), [layout["prefix_ids"]])
        self.assertEqual([call.args[0].item() for call in calls[1:]],
                         [value["token_id"] for value in layout["positions"][:-1]])
        self.assertEqual([call.kwargs["start_pos"] for call in calls[1:]],
                         [value["index"] for value in layout["positions"][:-1]])
        self.assertTrue(all(not call.kwargs["update_fast_weights"] for call in calls))
        self.assertLess(alignment["max_abs_difference"], 1e-5)

    def test_complete_episode_diagnostic_is_finite_and_never_changes_model_parameters(self):
        model, tokenizer = self.model(), OffsetTokenizer()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        result = diagnostic.diagnose_episode(model, tokenizer, row(), row("XYZ"), dtype=torch.float32)
        self.assertEqual(result["correct_memory_nll"]["payload_all"]["tokens"], 3)
        self.assertTrue(torch.isfinite(torch.tensor(result["payload_nll_gain_from_correct_memory"])))
        self.assertEqual(result["counterfactual_episode_id"], "dev-XYZ")
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()

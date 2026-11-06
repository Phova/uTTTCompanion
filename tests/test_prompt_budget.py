"""Long chat input retains its answering role and never consumes output space."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from engine.personalizer import uTTTPersonalizer
from engine.prompting import encode_chat_prompt
from model.tokenizer import BPETokenizer, SPECIAL_TOKENS
from model.uttt_model import SimpleUTF8Tokenizer


class PromptBudgetTests(unittest.TestCase):
    def tokenizers(self):
        backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(["hello there; question at the end"], trainer=trainers.BpeTrainer(
            vocab_size=300, min_frequency=2, special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ))
        return SimpleUTF8Tokenizer(32000), BPETokenizer(backend.to_str(), 512)

    def test_short_prompt_ends_at_assistant_token(self):
        for tokenizer in self.tokenizers():
            with self.subTest(tokenizer=type(tokenizer).__name__):
                expected = tokenizer.encode("<|user|> hello 世界\n<|assistant|>", add_bos=True)
                self.assertEqual(encode_chat_prompt(tokenizer, "  hello 世界\n", 64), expected)

    def test_long_prompt_preserves_roles_and_recent_question_with_exact_budget(self):
        for tokenizer in self.tokenizers():
            with self.subTest(tokenizer=type(tokenizer).__name__):
                prompt = "outdated words " * 100 + "FINAL QUESTION"
                ids = encode_chat_prompt(tokenizer, prompt, 32)
                self.assertEqual(len(ids), 32)
                self.assertEqual(ids[:2], [tokenizer.BOS_ID, tokenizer.USER_ID])
                self.assertEqual(ids.count(tokenizer.ASSISTANT_ID), 1)
                decoded = tokenizer.decode(ids, skip_special_tokens=False)
                self.assertIn("FINAL QUESTION", decoded)
                self.assertTrue(decoded.endswith("\n<|assistant|>"), decoded)

    def test_insufficient_template_space_is_rejected(self):
        tokenizer = SimpleUTF8Tokenizer(32000)
        for budget in (0, 1, 2, 3, 4, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                encode_chat_prompt(tokenizer, "hello", budget)
        ids = encode_chat_prompt(tokenizer, "x" * 100, 5)
        self.assertEqual(ids, [1, 4, 16 + ord("x"), 16 + ord("\n"), 5])

    def engine(self):
        engine = uTTTPersonalizer.__new__(uTTTPersonalizer)
        engine.config = SimpleNamespace(max_seq_len=64)
        engine.default_max_new_tokens = 4
        engine.temperature, engine.top_p = 0.0, 0.9
        engine.dtype, engine.device = torch.float32, torch.device("cpu")
        engine.tokenizer = SimpleUTF8Tokenizer(32000)
        engine.online_updates_enabled = False
        engine.surprise_threshold = 0.01
        engine.ckpt_manager = Mock()
        engine.ckpt_manager.compute_user_drift.return_value = {"drift_frobenius": 0.0, "drift_pct": 0.0, "step": 0}
        engine.ckpt_manager.get_persona_paths.return_value = (Path("unused.pt"), Path("unused.json"))
        engine.model = Mock()
        engine.model.stream_generate.return_value = iter(engine.tokenizer.encode("ok", add_bos=False))
        return engine

    def test_personalizer_sends_bounded_template_to_generation(self):
        engine = self.engine()
        list(engine.stream_interact_and_evolve("alice", "old text " * 100 + "FINAL QUESTION", max_new_tokens=32))
        inputs = engine.model.stream_generate.call_args.kwargs["input_ids"]
        self.assertEqual(tuple(inputs.shape), (1, 32))
        text = engine.tokenizer.decode(inputs[0], skip_special_tokens=False)
        self.assertTrue(text.endswith("FINAL QUESTION\n<|assistant|>"), text)

    def test_invalid_budget_cannot_load_or_evict_a_persona(self):
        for max_tokens in (60, 63, True):
            engine = self.engine()
            with self.subTest(max_tokens=max_tokens), self.assertRaises(ValueError):
                list(engine.stream_interact_and_evolve("alice", "hello", max_new_tokens=max_tokens))
            engine.ckpt_manager.get_user_state.assert_not_called()
            engine.model.stream_generate.assert_not_called()

    def test_budget_completion_keeps_unicode_tail_after_leading_whitespace(self):
        for tokenizer in self.tokenizers():
            for eos in (False, True):
                with self.subTest(tokenizer=type(tokenizer).__name__, eos=eos):
                    engine = self.engine()
                    engine.tokenizer = tokenizer
                    text = "  reply\ufffd"
                    engine.model.stream_generate.return_value = iter(tokenizer.encode(text, add_bos=False, add_eos=eos))
                    events = list(engine.stream_interact_and_evolve("alice", "hello", max_new_tokens=32))
                    streamed = "".join(event["content"] for event in events if event["type"] == "token")
                    self.assertEqual(streamed, text)
                    self.assertEqual(events[-1]["result"]["reply"], text.strip())


if __name__ == "__main__":
    unittest.main()

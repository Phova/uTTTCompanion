"""Generation must be a token prefix of complete SFT examples, before answers."""
import importlib.util
from pathlib import Path
import unittest

# The helper itself is dependency-free; avoid engine.__init__ importing Torch
# when checking prompt text on a lightweight deployment workstation.
spec = importlib.util.spec_from_file_location("prompting_under_test", Path(__file__).resolve().parents[1] / "engine" / "prompting.py")
prompting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompting)
encode_chat_prompt, format_chat_prompt = prompting.encode_chat_prompt, prompting.format_chat_prompt

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None


@unittest.skipIf(Tokenizer is None, "Byte-level BPE dependency unavailable")
class PromptBoundaryBPETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        special = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|user|>", "<|assistant|>"]
        backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        corpus = [
            "<|user|> Hello\n<|assistant|> Hello! How can I help you today?<|eos|>",
            "<|user|> 1+1?\n<|assistant|> 2<|eos|>",
            "<|user|> 翻译 hello\n<|assistant|> 你好<|eos|>",
        ] * 20
        backend.train_from_iterator(corpus, trainer=trainers.BpeTrainer(
            vocab_size=512, min_frequency=2, special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ))

        class Adapter:
            BOS_ID, ASSISTANT_ID = 1, 5

            def encode(self, text, add_bos=True, add_eos=False):
                return ([1] if add_bos else []) + backend.encode(text, add_special_tokens=False).ids + ([2] if add_eos else [])

        cls.tokenizer = Adapter()

    def test_generation_is_a_training_token_prefix_for_different_answer_types(self):
        for prompt, answer in [("Hello", "Hello! How can I help you today?"),
                               ("1+1?", "2"), ("翻译 hello", "你好")]:
            with self.subTest(answer=answer):
                generated = encode_chat_prompt(self.tokenizer, prompt, 128)
                trained = self.tokenizer.encode(format_chat_prompt(prompt) + " " + answer, add_eos=True)
                self.assertEqual(generated[-1], self.tokenizer.ASSISTANT_ID)
                self.assertEqual(trained[:len(generated)], generated)

    def test_old_trailing_space_reproduces_the_english_and_number_mismatch(self):
        for prompt, answer in [("Hello", "Hello!"), ("1+1?", "2")]:
            with self.subTest(answer=answer):
                legacy = self.tokenizer.encode(format_chat_prompt(prompt) + " ")
                trained = self.tokenizer.encode(format_chat_prompt(prompt) + " " + answer)
                self.assertNotEqual(trained[:len(legacy)], legacy)


class PromptTextTests(unittest.TestCase):
    def test_format_retains_internal_whitespace_and_ends_at_role(self):
        self.assertEqual(format_chat_prompt("  first\nsecond  "), "<|user|> first\nsecond\n<|assistant|>")

    def test_blank_and_nontext_inputs_fail_before_tokenization(self):
        for prompt in ("", " \n", None, 7):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                format_chat_prompt(prompt)


if __name__ == "__main__":
    unittest.main()

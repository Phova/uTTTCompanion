import unittest
from unittest.mock import MagicMock

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from engine.inference import TokenTextStream
from model.tokenizer import BPETokenizer, SPECIAL_TOKENS
from model.uttt_model import SimpleUTF8Tokenizer


class IncrementalDecodingTests(unittest.TestCase):
    def bpe_tokenizer(self):
        backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        # Byte-only vocabulary intentionally splits emoji and Chinese characters.
        backend.train_from_iterator(["hello world"], trainer=trainers.BpeTrainer(
            vocab_size=262, min_frequency=2, special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ))
        return BPETokenizer(backend.to_str(), 512)

    def test_byte_bpe_stream_matches_full_decode_without_replacement_characters(self):
        tokenizer = self.bpe_tokenizer()
        text = "  Hello 世界! 🙂 éλ\nAnother line."
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        expected = tokenizer.decode(ids)
        tokenizer.decode = MagicMock(wraps=tokenizer.decode)
        stream = TokenTextStream(tokenizer)
        native = stream.incremental
        pieces = [stream.push(token) for token in ids]
        self.assertEqual("".join(pieces), expected)
        self.assertTrue(all("\ufffd" not in piece for piece in pieces))
        if native:
            tokenizer.decode.assert_not_called()

    def test_legacy_hybrid_tokenizer_keeps_exact_streamed_text(self):
        tokenizer = SimpleUTF8Tokenizer(32000)
        text = "Hello，世界！emoji🙂 and λ"
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        stream = TokenTextStream(tokenizer)
        self.assertFalse(stream.incremental)
        self.assertEqual("".join(stream.push(token) for token in ids), tokenizer.decode(ids))

    def test_streams_do_not_share_partial_utf8_state(self):
        tokenizer = self.bpe_tokenizer()
        first_ids = tokenizer.encode("🙂中文", add_bos=False)
        second_ids = tokenizer.encode("λé", add_bos=False)
        first, second = TokenTextStream(tokenizer), TokenTextStream(tokenizer)
        first_text, second_text = "", ""
        for index in range(max(len(first_ids), len(second_ids))):
            if index < len(first_ids):
                first_text += first.push(first_ids[index])
            if index < len(second_ids):
                second_text += second.push(second_ids[index])
        self.assertEqual(first_text, "🙂中文")
        self.assertEqual(second_text, "λé")

    def test_native_failure_falls_back_without_duplicating_emitted_prefix(self):
        tokenizer = self.bpe_tokenizer()
        text = "Hello 🙂中文"
        ids = tokenizer.encode(text, add_bos=False)
        if not TokenTextStream(tokenizer).incremental:
            self.skipTest("Installed tokenizers has no DecodeStream")
        # Exercise failures before, during and after a multi-byte UTF-8 character.
        for failure_at in range(1, len(ids) + 1):
            with self.subTest(failure_at=failure_at):
                stream = TokenTextStream(tokenizer)
                original = stream._native
                calls = 0
                def step(backend, token):
                    nonlocal calls
                    calls += 1
                    if calls == failure_at:
                        raise RuntimeError("unsupported stream decoder")
                    return original.step(backend, token)
                stream._native = MagicMock()
                stream._native.step.side_effect = step
                self.assertEqual("".join(stream.push(token) for token in ids), text)
                self.assertFalse(stream.incremental)


if __name__ == "__main__":
    unittest.main()

"""Long-context phases, cache invalidation and decoding boundary regressions."""
import unittest
from unittest import mock
import weakref

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from engine.inference import CUDAGraphDecoder, TokenTextStream
from model.tokenizer import BPETokenizer, SPECIAL_TOKENS
from model.uttt_layer import RotaryEmbedding, SlidingWindowAttention
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel


def small_model():
    return uTTTModel(uTTTConfig(vocab_size=128, d_model=16, n_layers=2, n_heads=2,
                              n_kv_heads=1, d_ff_fast=32, max_seq_len=8,
                              sliding_window_size=8, gradient_checkpointing=False)).eval()


class LongContextPhaseTests(unittest.TestCase):
    def test_bf16_extension_preserves_prefix_and_uses_distinct_fp32_positions(self):
        rope = RotaryEmbedding(8, max_seq_len=16).bfloat16()
        before = rope.cos_cached.clone()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            cosine, sine = rope(torch.zeros(1, 1, 2, 8, dtype=torch.bfloat16), start_pos=4096)
        torch.testing.assert_close(rope.cos_cached[:16], before, atol=0, rtol=0)
        positions = torch.tensor([4096., 4097.])
        torch.testing.assert_close(cosine[0, 0, :, 0], positions.cos().bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(sine[0, 0, :, 0], positions.sin().bfloat16(), atol=0, rtol=0)
        self.assertFalse(torch.equal(cosine[:, :, 0], cosine[:, :, 1]))
        self.assertEqual(rope.cos_cached.dtype, torch.bfloat16)
        self.assertEqual(rope.inv_freq.dtype, torch.float32)

    def test_fp16_positions_beyond_integer_range_remain_finite(self):
        rope = RotaryEmbedding(8, max_seq_len=4).half()
        cosine, sine = rope(torch.zeros(1, 1, 2, 8, dtype=torch.float16), start_pos=65536)
        self.assertTrue(torch.isfinite(cosine).all())
        self.assertTrue(torch.isfinite(sine).all())
        torch.testing.assert_close(cosine[0, 0, :, 0], torch.tensor([65536., 65537.]).cos().half(),
                                   atol=0, rtol=0)

    def test_negative_position_is_rejected_before_cache_slicing(self):
        rope = RotaryEmbedding(8, max_seq_len=8)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            rope(torch.zeros(1, 1, 1, 8), start_pos=-1)

    def test_incremental_cache_matches_full_prefix_across_rope_extension(self):
        torch.manual_seed(91)
        model = small_model()
        state = model.init_fast_state(detach=True)
        ids = torch.randint(6, 128, (1, 14))
        caches = None
        with torch.no_grad():
            for length in range(6, 15):
                full = model(ids[:, :length], fast_state=state, update_fast_weights=False,
                             return_fast_state=False, logits_to_keep=1)
                actual = full if caches is None else model(ids[:, length - 1:length], fast_state=state,
                    kv_caches=caches, start_pos=length - 1, update_fast_weights=False,
                    return_fast_state=False, logits_to_keep=1)
                caches = actual.kv_caches
                torch.testing.assert_close(actual.logits, full.logits, atol=2e-5, rtol=2e-4)


class CacheSemanticInvalidationTests(unittest.TestCase):
    def test_sliding_mask_is_rebuilt_after_window_change(self):
        attention = SlidingWindowAttention(8, 2, 1, window_size=4)
        old = attention._get_sliding_causal_mask(8, 8, torch.device('cpu'))
        attention.window_size = 2
        updated = attention._get_sliding_causal_mask(8, 8, torch.device('cpu'))
        self.assertIsNot(updated, old)
        self.assertEqual(updated[-1].tolist(), [False] * 6 + [True, True])

    def test_cuda_graph_turn_key_includes_window_and_normalization_constants(self):
        model = small_model()
        state = model.init_fast_state(detach=True)
        decoder = CUDAGraphDecoder.__new__(CUDAGraphDecoder)
        decoder.model, decoder.fast_state = model, state
        targets = [(model.layers[0].attn, 'window_size'), (model.layers[0].attn_norm, 'eps'),
                   (model.layers[0].uttt_norm, 'eps'), (model.layers[0].uttt.q_norm, 'eps'),
                   (model.norm, 'eps'), (model.fast_weight_pool, 'eps')]
        for target, name in targets:
            with self.subTest(attribute=name, module=type(target).__name__):
                decoder._captured_model_key = decoder._model_key(model)
                decoder._captured_state_key = decoder._state_key(state)
                self.assertTrue(decoder.matches(model, state))
                original = getattr(target, name)
                setattr(target, name, original * 2)
                self.assertFalse(decoder.matches(model, state))
                setattr(target, name, original)


class SamplingBoundaryTests(unittest.TestCase):
    def test_forbidden_vocabulary_never_wins_against_very_negative_legal_logits(self):
        logits = torch.full((2, 16), -1e20)
        logits[:, 2] = -1e19
        logits[:, :2], logits[:, 3], logits[:, 8:] = 100., 100., 100.
        for temperature in (0., 1.):
            sampled = uTTTModel._sample_next_token(logits, temperature=temperature, top_p=1.,
                                                  active_vocab_limit=8, repetition_penalty=1.)
            self.assertEqual(sampled.tolist(), [2, 2])

    def test_nonfinite_or_invalid_sampling_settings_are_rejected(self):
        logits = torch.zeros(1, 128)
        for settings in ({'temperature': float('nan')}, {'temperature': -1.},
                         {'top_p': 0.}, {'top_p': float('inf')},
                         {'repetition_penalty': float('nan')}, {'active_vocab_limit': 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                uTTTModel._sample_next_token(logits, **settings)

    def test_zero_budget_skips_prefill_and_preserves_training_mode(self):
        model = small_model().train()
        state = model.init_fast_state(detach=True)
        with mock.patch.object(model, 'forward', side_effect=AssertionError('unneeded prefill')):
            self.assertEqual(list(model.stream_generate(torch.ones(1, 2, dtype=torch.long), state,
                                                       max_new_tokens=0)), [])
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, 'one non-empty prompt'):
            list(model.stream_generate(torch.ones(2, 2, dtype=torch.long), state, max_new_tokens=1))
        self.assertTrue(model.training)

    def test_old_persona_graph_is_released_before_another_graph_is_allocated(self):
        model = small_model().requires_grad_(False)
        state = model.init_fast_state(detach=True)

        class Previous:
            def matches(self, *_):
                return False

        model._cuda_graph_decoder = Previous()
        previous = weakref.ref(model._cuda_graph_decoder)
        owner = self

        class Replacement:
            @staticmethod
            def is_supported(*_):
                return True

            def __init__(self, *args):
                owner.assertIsNone(previous(), 'old graph was retained during capture allocation')

            def step(self, token):
                return torch.zeros((1, 128))

        with mock.patch('engine.inference.CUDAGraphDecoder', Replacement), \
                mock.patch.object(model, '_sample_next_token', return_value=torch.tensor([6])):
            tokens = list(model.stream_generate(torch.ones(1, 2, dtype=torch.long), state,
                                               max_new_tokens=3, eos_token_id=None, use_cuda_graph=True))
        self.assertEqual(tokens, [6, 6, 6])


class TextFidelityTests(unittest.TestCase):
    def tokenizer(self):
        backend = Tokenizer(models.BPE(unk_token='<|unk|>'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(['abc'], trainer=trainers.BpeTrainer(vocab_size=262,
            special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        return BPETokenizer(backend.to_str(), 512)

    def test_literal_replacement_character_is_not_deleted_from_complete_text(self):
        for tokenizer in (self.tokenizer(), SimpleUTF8Tokenizer(32000)):
            for text in ('before\ufffdafter', 'trailing\ufffd'):
                with self.subTest(tokenizer=type(tokenizer).__name__, text=text):
                    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                    stream = TokenTextStream(tokenizer)
                    self.assertEqual(''.join(stream.push(token) for token in ids), tokenizer.decode(ids))


if __name__ == '__main__':
    unittest.main()

"""Per-request sampling generators isolate concurrent conversations' RNG state."""
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from model.uttt_model import uTTTConfig, uTTTModel


class RequestSamplingRNGTests(unittest.TestCase):
    def probabilities(self, shape):
        logits = torch.full((*shape, 16), -80.)
        logits[..., 6:9] = torch.tensor([.55, .30, .15]).log()
        return logits

    def sample_many(self, generator, top_p=.9):
        return [int(uTTTModel._sample_next_token(self.probabilities((1,)), temperature=.8,
                                               top_p=top_p, generator=generator).item()) for _ in range(24)]

    def test_local_generator_is_independent_of_global_seed_for_both_nucleus_paths(self):
        for top_p in (.9, 1.):
            with self.subTest(top_p=top_p):
                torch.manual_seed(123)
                first = self.sample_many(torch.Generator().manual_seed(17), top_p)
                torch.manual_seed(9876)
                second = self.sample_many(torch.Generator().manual_seed(17), top_p)
                self.assertEqual(first, second)

    def test_local_generator_does_not_advance_the_global_rng(self):
        torch.manual_seed(21)
        before = torch.get_rng_state().clone()
        self.sample_many(torch.Generator().manual_seed(42))
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)

    def test_default_none_preserves_existing_global_sampling_trajectory(self):
        logits = self.probabilities((1,))
        torch.manual_seed(52)
        expected = [uTTTModel._sample_next_token(logits, temperature=.8).item() for _ in range(16)]
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(52)
        observed = [uTTTModel._sample_next_token(logits, temperature=.8, generator=None).item() for _ in range(16)]
        self.assertEqual(observed, expected)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)

    def fixture(self):
        torch.manual_seed(87)
        model = uTTTModel(uTTTConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=1,
                                    n_kv_heads=1, d_ff_fast=16, max_seq_len=64,
                                    gradient_checkpointing=False)).eval().requires_grad_(False)
        state = model.init_fast_state(detach=True)
        return model, state

    def stream(self, model, state, *, speculative, generator, temperature=.8):
        # A controlled, prefix-independent target isolates RNG semantics from
        # normal GEMM/SDPA batch-shape rounding. Proposals still use argmax;
        # every committed target must use the supplied local generator.
        def forward(ids=None, input_ids=None, **kwargs):
            inputs = input_ids if input_ids is not None else ids
            return SimpleNamespace(logits=self.probabilities(tuple(inputs.shape)), kv_caches=None)

        def draft(hidden):
            return self.probabilities(tuple(hidden.shape[:-1]))

        with mock.patch.object(model, 'forward', side_effect=forward) as target, \
                mock.patch.object(model.lm_head, 'forward', side_effect=draft):
            tokens = list(model.stream_generate(torch.tensor([[1, 6]]), state,
                                                max_new_tokens=20, temperature=temperature, top_p=.9,
                                                eos_token_id=None, speculative_draft_k=3 if speculative else 0,
                                                dspark_conf_threshold=0., generator=generator))
            if speculative:
                self.assertTrue(any(call.kwargs.get('input_ids', torch.empty(1, 0)).shape[1] > 1
                                    for call in target.call_args_list[1:]), 'verification path was not exercised')
            return tokens

    def test_ordinary_and_forced_speculative_streams_share_the_local_target_rng(self):
        model, state = self.fixture()
        torch.manual_seed(109)
        before = torch.get_rng_state().clone()
        plain = self.stream(model, state, speculative=False, generator=torch.Generator().manual_seed(19))
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        torch.manual_seed(999)
        other_before = torch.get_rng_state().clone()
        speculative = self.stream(model, state, speculative=True, generator=torch.Generator().manual_seed(19))
        self.assertEqual(speculative, plain)
        self.assertEqual(len(plain), 20)
        torch.testing.assert_close(torch.get_rng_state(), other_before, atol=0, rtol=0)

    def test_greedy_speculative_acceptance_does_not_consume_a_local_generator(self):
        model, state = self.fixture()
        generator = torch.Generator().manual_seed(31)
        before = generator.get_state().clone()
        result = self.stream(model, state, speculative=True, generator=generator, temperature=0.)
        self.assertEqual(result, [6] * 20)
        torch.testing.assert_close(generator.get_state(), before, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()

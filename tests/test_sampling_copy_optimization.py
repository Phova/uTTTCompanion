"""One-copy working logits preserve all dtypes, masks and RNG trajectories."""
import unittest

import torch

from deploy.benchmark_sampling_copy import baseline_sampler
from model.uttt_model import uTTTModel


class SamplingCopyTests(unittest.TestCase):
    def test_dtype_layout_and_probability_paths_keep_exact_samples_rng_and_input(self):
        original = baseline_sampler()
        settings = [dict(temperature=0., top_p=.9, repetition_penalty=1.),
                    dict(temperature=.8, top_p=.9, repetition_penalty=1.),
                    dict(temperature=.8, top_p=1., repetition_penalty=1.),
                    dict(temperature=.8, top_p=.9, repetition_penalty=1.1, recent_tokens=[6, 7, 8, 6])]
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for strided in (False, True):
                torch.manual_seed(71)
                values = torch.randn(3, 64, dtype=dtype)
                values = values[:, ::2] if strided else values[:, :32].contiguous()
                before = values.clone()
                version = values._version
                for options in settings:
                    with self.subTest(dtype=dtype, strided=strided, options=options):
                        first, second = (torch.Generator().manual_seed(41) for _ in range(2))
                        expected = [original(values, generator=first, active_vocab_limit=24, **options)
                                    for _ in range(8)]
                        actual = [uTTTModel._sample_next_token(values, generator=second,
                                                              active_vocab_limit=24, **options)
                                  for _ in range(8)]
                        torch.testing.assert_close(torch.stack(actual), torch.stack(expected), atol=0, rtol=0)
                        torch.testing.assert_close(second.get_state(), first.get_state(), atol=0, rtol=0)
                        torch.testing.assert_close(values, before, atol=0, rtol=0)
                        self.assertEqual(values._version, version)

    def test_ties_and_extreme_negative_valid_logits_keep_hard_masks_and_lowest_legal_index(self):
        original = baseline_sampler()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = torch.full((1, 16), -1000., dtype=dtype)
            values[:, 2], values[:, 6] = -500., -500.
            values[:, [0, 1, 3, 8, 9]] = 1000.
            for function in (original, uTTTModel._sample_next_token):
                result = function(values, temperature=0., active_vocab_limit=8)
                self.assertEqual(result.tolist(), [2])


if __name__ == '__main__':
    unittest.main()

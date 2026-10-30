"""CUDA Graph decode must preserve positions, window masks and persona isolation."""
import unittest
from unittest.mock import patch

import torch

from engine.inference import CUDAGraphDecoder
from model.uttt_model import uTTTConfig, uTTTModel


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
class CUDAGraphDecodeTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(572)
        config = uTTTConfig(
            vocab_size=512, d_model=64, n_layers=2, n_heads=4,
            n_kv_heads=2, d_ff_fast=128, chunk_size=4,
            sliding_window_size=8, max_seq_len=64,
            gradient_checkpointing=False, dropout=0.0,
        )
        model = uTTTModel(config).to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        with torch.no_grad():
            # Make attention positions/masking visible above BF16 rounding noise.
            for block in model.layers:
                block.attn.wq.weight.mul_(3)
                block.attn.wk.weight.mul_(3)
        prepare = getattr(model, "prepare_for_inference", None)
        if callable(prepare):
            prepare()
        state = model.init_fast_state(batch_size=1, device="cuda", dtype=torch.bfloat16, detach=True)
        return model, state

    def tokens(self, length, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(6, 512, (1, length), generator=generator).cuda()

    def greedy_without_stops(self, logits, **kwargs):
        # Stop tokens are excluded so both paths run across the full window.
        scores = logits.float().clone()
        scores[:, :6] = -torch.inf
        limit = kwargs.get("active_vocab_limit")
        if limit is not None:
            scores[:, limit:] = -torch.inf
        return scores.argmax(dim=-1)

    def prefill(self, model, state, ids):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(
                ids, fast_state=state, start_pos=0, update_fast_weights=False,
                logits_to_keep=1, return_fast_state=False,
            )
        return result.kv_caches

    def check_steps(self, model, state, decoder, caches, start_pos, ids):
        # The graph must copy its initialization, rather than modifying caller caches.
        reference_caches = [(key.clone(), value.clone()) for key, value in caches]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in range(ids.shape[1]):
                token = ids[:, index:index + 1]
                reference = model(
                    token, fast_state=state, kv_caches=reference_caches,
                    start_pos=start_pos + index, update_fast_weights=False,
                    logits_to_keep=1, return_fast_state=False,
                )
                reference_caches = reference.kv_caches
                actual = decoder.step(token)
                self.assertEqual(tuple(actual.shape), (1, model.config.vocab_size))
                self.assertTrue(torch.isfinite(actual).all().item())
                # A ring permutes attention's reduction order, so BF16 need not be bitwise equal.
                torch.testing.assert_close(
                    actual.float(), reference.logits[:, -1, :].float(),
                    atol=0.015, rtol=0.025,
                    msg=f"Graph decode differs at absolute position {start_pos + index}",
                )
                self.assertEqual(decoder.next_position, start_pos + index + 1)

    def test_short_and_long_prefill_match_eager_across_multiple_window_wraps(self):
        model, state = self.make_model()
        before = {name: getattr(state, name).clone() for name in ("w_up", "w_gate", "w_down")}
        # Three tokens leave five invalid static slots; thirteen require dropping history.
        for prompt_length in (3, 13):
            with self.subTest(prompt_length=prompt_length):
                prompt = self.tokens(prompt_length, 900 + prompt_length)
                caches = self.prefill(model, state, prompt)
                self.assertTrue(CUDAGraphDecoder.is_supported(model, state, caches, prompt_length))
                decoder = CUDAGraphDecoder(model, state, caches, prompt_length)
                self.assertEqual(decoder.next_position, prompt_length)
                self.check_steps(model, state, decoder, caches, prompt_length, self.tokens(19, 152))
        for name, expected in before.items():
            torch.testing.assert_close(getattr(state, name), expected, atol=0, rtol=0)

    def test_reset_replaces_previous_turn_cache_and_position(self):
        model, state = self.make_model()
        first_caches = self.prefill(model, state, self.tokens(13, 127))
        decoder = CUDAGraphDecoder(model, state, first_caches, 13)
        self.check_steps(model, state, decoder, first_caches, 13, self.tokens(11, 128))
        next_caches = self.prefill(model, state, self.tokens(2, 129))
        decoder.reset(next_caches, 2)
        self.assertEqual(decoder.next_position, 2)
        self.assertTrue(decoder.matches(model, state))
        self.check_steps(model, state, decoder, next_caches, 2, self.tokens(13, 130))

    def test_persona_pointer_and_in_place_mutation_invalidate_capture(self):
        model, state = self.make_model()
        caches = self.prefill(model, state, self.tokens(3, 200))
        decoder = CUDAGraphDecoder(model, state, caches, 3)
        self.assertTrue(decoder.matches(model, state))
        other_persona = state.clone(detach=True)
        self.assertFalse(decoder.matches(model, other_persona))
        with torch.no_grad():
            state.w_up.add_(0.125)
        self.assertFalse(decoder.matches(model, state))
        with self.assertRaises(ValueError):
            decoder.step(self.tokens(1, 201))

    def test_model_weight_mutation_invalidates_capture(self):
        model, state = self.make_model()
        caches = self.prefill(model, state, self.tokens(3, 300))
        decoder = CUDAGraphDecoder(model, state, caches, 3)
        self.assertTrue(decoder.matches(model, state))
        with torch.no_grad():
            model.layers[0].attn.wq.weight.add_(0.125)
        self.assertFalse(decoder.matches(model, state))

    def test_fp32_persona_master_uses_bf16_graph_reads_without_losing_master_bits(self):
        model, state = self.make_model()
        state = state.to(dtype=torch.float32)
        state.w_up.add_(1e-5)
        before = state.w_up.clone()
        caches = self.prefill(model, state, self.tokens(3, 371))
        self.assertTrue(CUDAGraphDecoder.is_supported(model, state, caches, 3))
        decoder = CUDAGraphDecoder(model, state, caches, 3)
        self.check_steps(model, state, decoder, caches, 3, self.tokens(13, 372))
        self.assertEqual(state.w_up.dtype, torch.float32)
        torch.testing.assert_close(state.w_up, before, atol=0, rtol=0)

    def test_exported_window_resumes_eager_in_chronological_order(self):
        model, state = self.make_model()
        caches = self.prefill(model, state, self.tokens(3, 400))
        decoder = CUDAGraphDecoder(model, state, caches, 3)
        self.check_steps(model, state, decoder, caches, 3, self.tokens(19, 401))
        exported = decoder.export_kv_caches()
        self.assertEqual(len(exported), len(model.layers))
        for block, (key, value) in zip(model.layers, exported):
            expected = (1, block.attn.n_kv_heads, block.attn.window_size, block.attn.head_dim)
            self.assertEqual(tuple(key.shape), expected)
            self.assertEqual(tuple(value.shape), expected)
            self.assertEqual(key.untyped_storage().nbytes(), key.numel() * key.element_size())
            self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
        position = decoder.next_position
        token = self.tokens(1, 402)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager = model(
                token, fast_state=state, kv_caches=exported, start_pos=position,
                update_fast_weights=False, logits_to_keep=1, return_fast_state=False,
            )
            continuation = decoder.step(token)
        torch.testing.assert_close(
            continuation.float(), eager.logits[:, -1, :].float(), atol=0.015, rtol=0.025,
        )

    def test_supported_guard_rejects_trainable_fp32_and_training_models_under_autocast(self):
        model, state = self.make_model()
        caches = self.prefill(model, state, self.tokens(3, 500))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.assertTrue(CUDAGraphDecoder.is_supported(model, state, caches, 3))
            parameter = next(model.parameters())
            parameter.requires_grad_(True)
            self.assertFalse(CUDAGraphDecoder.is_supported(model, state, caches, 3))
            parameter.requires_grad_(False)
            model.train()
            self.assertFalse(CUDAGraphDecoder.is_supported(model, state, caches, 3))
            model.eval().float()
            fp32_state = state.to(dtype=torch.float32)
            fp32_caches = [(key.float(), value.float()) for key, value in caches]
            self.assertFalse(CUDAGraphDecoder.is_supported(model, fp32_state, fp32_caches, 3))

    def test_stream_generation_preserves_greedy_tokens_and_defers_graph_until_after_first_yield(self):
        model, state = self.make_model()
        max_tokens = 14

        def generation(prompt, graph):
            return model.stream_generate(
                prompt, fast_state=state, max_new_tokens=max_tokens,
                temperature=0.0, eos_token_id=None, speculative_draft_k=0,
                use_cuda_graph=graph,
            )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            with patch.object(model, "_sample_next_token", side_effect=self.greedy_without_stops):
                first_prompt = self.tokens(3, 600)
                expected = list(generation(first_prompt, False))
                self.assertEqual(len(expected), max_tokens)
                stream = generation(first_prompt, True)
                first = next(stream)
                self.assertIsNone(getattr(model, "_cuda_graph_decoder", None))
                actual = [first] + list(stream)
                self.assertEqual(actual, expected)
                decoder = model._cuda_graph_decoder
                self.assertIsNotNone(decoder)

                second_prompt = self.tokens(5, 601)
                expected = list(generation(second_prompt, False))
                self.assertEqual(len(expected), max_tokens)
                with patch.object(decoder, "reset", wraps=decoder.reset) as reset:
                    stream = generation(second_prompt, True)
                    first = next(stream)
                    reset.assert_not_called()
                    self.assertIs(model._cuda_graph_decoder, decoder)
                    actual = [first] + list(stream)
                    reset.assert_called_once()
                self.assertIs(model._cuda_graph_decoder, decoder)
                self.assertEqual(actual, expected)

    def test_capture_failure_falls_back_once_and_preserves_greedy_generation(self):
        model, state = self.make_model()
        prompt = self.tokens(3, 700)
        kwargs = {
            "fast_state": state, "max_new_tokens": 10, "temperature": 0.0,
            "eos_token_id": None, "speculative_draft_k": 0,
        }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            with patch.object(model, "_sample_next_token", side_effect=self.greedy_without_stops):
                expected = list(model.stream_generate(prompt, use_cuda_graph=False, **kwargs))
                with patch("engine.inference.CUDAGraphDecoder") as decoder_type:
                    decoder_type.is_supported.return_value = True
                    decoder_type.side_effect = RuntimeError("simulated capture failure")
                    actual = list(model.stream_generate(prompt, use_cuda_graph=True, **kwargs))
                    decoder_type.assert_called_once()
                self.assertEqual(len(actual), 10)
                self.assertEqual(actual, expected)
                self.assertIsNone(getattr(model, "_cuda_graph_decoder", None))


if __name__ == "__main__":
    unittest.main()

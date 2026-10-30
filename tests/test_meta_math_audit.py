"""Numerical checks for differentiable uTTT updates and master-state precision."""
import unittest

import torch

from model.uttt_layer import (
    SharedFastWeightPool,
    analytical_swiglu_chunk_grad,
    fused_chunk_ttt_step,
    swiglu_fast_forward,
)


class MetaMathAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(193)

    @staticmethod
    def _reference_step(q, k, v, eta, alpha, weights, momentum, ln_w, ln_b,
                        mask=None, beta=0.7, max_norm=0.0, eps=1e-4):
        prediction = swiglu_fast_forward(k, *weights, ln_w, ln_b, eps)
        token_loss = 0.5 * (prediction - v).square().sum(-1) / k.shape[-1]
        if mask is None:
            weighted = (token_loss * eta.squeeze(-1)).sum() / k.shape[1]
            surprise = token_loss.mean()
            alpha_bar = alpha.mean(1, keepdim=True)
        else:
            counts = mask.sum(-1, keepdim=True).clamp_min(1)
            weighted = (token_loss * eta.squeeze(-1) * mask / counts).sum()
            surprise = (token_loss * mask).sum() / mask.sum().clamp_min(1)
            alpha_bar = (alpha * mask.unsqueeze(-1)).sum(1, keepdim=True) / counts.unsqueeze(-1)
        gradients = torch.autograd.grad(weighted, weights, create_graph=True)
        if max_norm > 0:
            squared = sum(gradient.square().sum((1, 2)) for gradient in gradients)
            scale = (max_norm * torch.rsqrt(squared + 1e-8)).clamp(max=1).view(-1, 1, 1)
            gradients = tuple(gradient * scale for gradient in gradients)
        new_momentum = tuple(beta * previous - gradient for previous, gradient in zip(momentum, gradients))
        new_weights = tuple((1 - alpha_bar) * weight + change for weight, change in zip(weights, new_momentum))
        if mask is not None:
            active = mask.any(-1).view(-1, 1, 1)
            new_weights = tuple(torch.where(active, new, old) for new, old in zip(new_weights, weights))
            new_momentum = tuple(torch.where(active, new, old) for new, old in zip(new_momentum, momentum))
        readout = swiglu_fast_forward(q, *weights, ln_w, ln_b, eps)
        return readout, new_weights, new_momentum, surprise

    def test_double_analytical_gradient_matches_autograd_with_dynamic_eta_and_mask(self):
        batch, length, dim, hidden = 2, 3, 4, 5
        k = torch.randn(batch, length, dim, dtype=torch.float64)
        v = torch.randn_like(k)
        eta = torch.rand(batch, length, 1, dtype=torch.float64) * 0.08
        weights = tuple((torch.randn(*shape, dtype=torch.float64) * 0.3).requires_grad_()
                        for shape in ((batch, dim, hidden), (batch, dim, hidden), (batch, hidden, dim)))
        ln_w = torch.randn(dim, dtype=torch.float64)
        ln_b = torch.randn(dim, dtype=torch.float64)
        for mask in (None, torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.float64)):
            with self.subTest(masked=mask is not None):
                actual = analytical_swiglu_chunk_grad(k, v, eta, *weights, ln_w, ln_b,
                                                     token_mask=mask, normalize_by_dim=True, eps=1e-4)
                prediction = swiglu_fast_forward(k, *weights, ln_w, ln_b, 1e-4)
                losses = 0.5 * (prediction - v).square().sum(-1) / dim
                if mask is None:
                    loss = (losses * eta.squeeze(-1)).sum() / length
                    surprise = losses.mean()
                else:
                    loss = (losses * eta.squeeze(-1) * mask / mask.sum(-1, keepdim=True)).sum()
                    surprise = (losses * mask).sum() / mask.sum()
                reference = torch.autograd.grad(loss, weights)
                for result, expected in zip(actual[:3], reference):
                    torch.testing.assert_close(result, expected, atol=1e-11, rtol=1e-9)
                torch.testing.assert_close(actual[3], surprise, atol=1e-12, rtol=1e-10)

    def test_exact_two_step_meta_gradients_match_differentiable_autograd_reference(self):
        batch, length, dim, hidden = 1, 3, 4, 5

        def leaf(shape, scale=0.3):
            return (torch.randn(*shape, dtype=torch.float64) * scale).requires_grad_()

        initial = tuple(leaf(shape) for shape in ((batch, dim, hidden), (batch, dim, hidden), (batch, hidden, dim)))
        momentum = tuple(leaf(weight.shape, 0.01) for weight in initial)
        ln_w = leaf((dim,), 0.8)
        ln_b = leaf((dim,), 0.1)
        chunks = [tuple(leaf((batch, length, dim)) for _ in range(3)) +
                  (leaf((batch, length, 1), 0.02), leaf((batch, length, 1), 0.01)) for _ in range(2)]
        mask = torch.tensor([[1, 1, 0]], dtype=torch.float64)
        leaves = list(initial + momentum + (ln_w, ln_b)) + [tensor for chunk in chunks for tensor in chunk]
        for max_norm in (0.0, 0.01):
            with self.subTest(max_norm=max_norm):
                actual_weights, actual_momentum = initial, momentum
                expected_weights, expected_momentum = initial, momentum
                actual_loss = torch.zeros((), dtype=torch.float64)
                expected_loss = torch.zeros((), dtype=torch.float64)
                for q, k, v, eta, alpha in chunks:
                    actual = fused_chunk_ttt_step(
                        q, k, v, eta, alpha, *actual_weights, *actual_momentum, ln_w, ln_b,
                        mask, 0.7, max_norm, True, 1e-4,
                    )
                    expected = self._reference_step(
                        q, k, v, eta, alpha, expected_weights, expected_momentum, ln_w, ln_b,
                        mask=mask, max_norm=max_norm,
                    )
                    actual_weights, actual_momentum = actual[1:4], actual[4:7]
                    expected_weights, expected_momentum = expected[1:3]
                    torch.testing.assert_close(actual[0], expected[0], atol=1e-11, rtol=1e-9)
                    for result, reference in zip(actual_weights + actual_momentum,
                                                 expected_weights + expected_momentum):
                        torch.testing.assert_close(result, reference, atol=1e-11, rtol=1e-9)
                    actual_loss = actual_loss + actual[0].square().mean() + 0.07 * actual[7]
                    expected_loss = expected_loss + expected[0].square().mean() + 0.07 * expected[3]
                actual_loss = actual_loss + sum(weight.square().mean() for weight in actual_weights + actual_momentum)
                expected_loss = expected_loss + sum(weight.square().mean() for weight in expected_weights + expected_momentum)
                actual_grad = torch.autograd.grad(actual_loss, leaves, retain_graph=True)
                expected_grad = torch.autograd.grad(expected_loss, leaves, retain_graph=True)
                for index, (result, reference) in enumerate(zip(actual_grad, expected_grad)):
                    torch.testing.assert_close(result, reference, atol=1e-10, rtol=1e-8, msg=f"leaf {index}")

    def test_double_outer_derivative_agrees_with_finite_difference(self):
        dim, hidden, length = 4, 5, 3
        weights = tuple(torch.randn(*shape, dtype=torch.float64) * 0.3
                        for shape in ((1, dim, hidden), (1, dim, hidden), (1, hidden, dim)))
        q, k, v = (torch.randn(1, length, dim, dtype=torch.float64) for _ in range(3))
        eta = torch.full((1, length, 1), 0.03, dtype=torch.float64)
        alpha = torch.full_like(eta, 0.01)
        ln_w, ln_b = torch.ones(dim, dtype=torch.float64), torch.zeros(dim, dtype=torch.float64)

        def objective(value):
            changed = weights[0].clone()
            changed[0, 0, 0] = value
            state = (changed, weights[1], weights[2])
            result = fused_chunk_ttt_step(
                q, k, v, eta, alpha, *state, *(torch.zeros_like(weight) for weight in state),
                ln_w, ln_b, None, 0.7, 0.0, True, 1e-4,
            )
            return result[1].square().sum() + 0.2 * result[2].square().sum() + 0.1 * result[7]

        value = weights[0][0, 0, 0].clone().requires_grad_()
        derivative, = torch.autograd.grad(objective(value), value)
        delta = 1e-5
        finite = (objective(value.detach() + delta) - objective(value.detach() - delta)) / (2 * delta)
        torch.testing.assert_close(derivative, finite, atol=2e-8, rtol=2e-6)

    def test_momentum_gradient_survives_two_steps(self):
        dim, hidden = 3, 4
        zero_tokens = torch.zeros(1, 2, dim, dtype=torch.float64)
        weights = tuple(torch.ones(*shape, dtype=torch.float64)
                        for shape in ((1, dim, hidden), (1, dim, hidden), (1, hidden, dim)))
        momentum = tuple(torch.zeros_like(weight).requires_grad_() for weight in weights)
        eta = torch.zeros(1, 2, 1, dtype=torch.float64)
        alpha = torch.full_like(eta, 0.01)
        for _ in range(2):
            result = fused_chunk_ttt_step(
                zero_tokens, zero_tokens, zero_tokens, eta, alpha, *weights, *momentum,
                torch.ones(dim, dtype=torch.float64), torch.zeros(dim, dtype=torch.float64),
                None, 0.7, 0.0, True, 1e-4,
            )
            weights, momentum = result[1:4], result[4:7]
            if _ == 0:
                first_momentum = momentum[0]
        gradient, = torch.autograd.grad(weights[0].sum(), first_momentum)
        torch.testing.assert_close(gradient, torch.full_like(gradient, 0.99 + 0.7), atol=1e-12, rtol=1e-12)

    def test_bf16_inputs_accumulate_small_decay_and_momentum_in_fp32_master_state(self):
        dim, hidden = 3, 4
        tokens = torch.zeros(1, 2, dim, dtype=torch.bfloat16)
        initial = tuple(torch.ones(*shape, dtype=torch.bfloat16)
                        for shape in ((1, dim, hidden), (1, dim, hidden), (1, hidden, dim)))
        zero_momentum = tuple(torch.zeros_like(weight) for weight in initial)
        eta = torch.zeros(1, 2, 1, dtype=torch.bfloat16)
        alpha = torch.full_like(eta, 0.001)
        ln_w, ln_b = torch.ones(dim, dtype=torch.bfloat16), torch.zeros(dim, dtype=torch.bfloat16)
        weights, momentum = initial, zero_momentum
        with torch.no_grad():
            for _ in range(10):
                result = fused_chunk_ttt_step(
                    tokens, tokens, tokens, eta, alpha, *weights, *momentum,
                    ln_w, ln_b, None, 0.0, 0.0, True, 1e-4,
                )
                weights, momentum = result[1:4], result[4:7]
                self.assertTrue(all(tensor.dtype == torch.float32 for tensor in weights + momentum))
            expected = (1 - alpha.float().mean()).pow(10)
            for weight in weights:
                torch.testing.assert_close(weight, expected.expand_as(weight), atol=2e-7, rtol=1e-7)
                self.assertTrue(bool((weight < 0.999).all()))
            small_momentum = tuple(torch.full_like(weight, 1e-5) for weight in initial)
            result = fused_chunk_ttt_step(
                tokens, tokens, tokens, eta, torch.zeros_like(alpha), *initial, *small_momentum,
                ln_w, ln_b, None, 1.0, 0.0, True, 1e-4,
            )
            torch.testing.assert_close(result[1], initial[0].float() + small_momentum[0].float(), atol=0, rtol=0)
            self.assertTrue(bool((result[1] > 1).all()))

    def test_separate_chunk_read_state_does_not_replace_the_write_cursor(self):
        pool = SharedFastWeightPool(4, 5, normalize_loss_by_dim=True).double()
        cursor = pool.init_state(dtype=torch.float64)
        chunk_start = cursor.clone()
        chunk_start.w_up = chunk_start.w_up + torch.randn_like(chunk_start.w_up) * 0.1
        q, k, v = (torch.randn(1, 3, 4, dtype=torch.float64) for _ in range(3))
        eta = torch.full((1, 3, 1), 0.03, dtype=torch.float64)
        alpha = torch.full_like(eta, 0.01)
        actual, updated, loss = pool.chunk_step(cursor, q, k, v, eta, alpha, read_state=chunk_start)
        _, ordinary_updated, ordinary_loss = pool.chunk_step(cursor, q, k, v, eta, alpha)
        expected = swiglu_fast_forward(q, chunk_start.w_up, chunk_start.w_gate, chunk_start.w_down,
                                      cursor.ln_weight, cursor.ln_bias, pool.eps)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(loss, ordinary_loss, atol=0, rtol=0)
        for name in ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down"):
            torch.testing.assert_close(getattr(updated, name), getattr(ordinary_updated, name), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

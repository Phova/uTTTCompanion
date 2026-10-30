"""Shared memory must advance over chunks and depth without leaking future tokens."""
import unittest
from unittest.mock import patch

import torch

from model.uttt_model import uTTTConfig, uTTTModel


class SharedScheduleAuditTests(unittest.TestCase):
    def model(self, checkpointed=False):
        torch.manual_seed(971)
        return uTTTModel(uTTTConfig(
            vocab_size=128, d_model=16, d_ff_fast=32, n_layers=3,
            n_heads=2, n_kv_heads=1, chunk_size=4, sliding_window_size=8,
            gradient_checkpointing=checkpointed, max_inner_grad_norm=0,
        ))

    def test_chunk_first_writes_share_cursor_and_read_snapshot(self):
        model = self.model().eval()
        records = []
        original = model.fast_weight_pool.chunk_step

        def track(*args, **kwargs):
            state, read = kwargs["state"], kwargs["read_state"]
            records.append((state.step, read.step, state.w_up.clone(), read.w_up.clone()))
            return original(*args, **kwargs)

        ids = torch.randint(6, 128, (1, 12))
        with torch.no_grad(), patch.object(model.fast_weight_pool, "chunk_step", side_effect=track):
            result = model(ids)
        self.assertEqual(result.fast_state.step, 9)
        self.assertEqual([row[0] for row in records], list(range(9)))
        self.assertEqual([row[1] for row in records], [0] * 3 + [3] * 3 + [6] * 3)
        for start in (0, 3, 6):
            for row in records[start:start + 3]:
                torch.testing.assert_close(row[3], records[start][3], atol=0, rtol=0)
        self.assertFalse(torch.equal(records[3][3], records[0][3]))

    def test_deep_write_changes_next_chunks_shallow_read_only(self):
        model = self.model().eval()
        ids = torch.randint(6, 128, (1, 8))
        original = model.fast_weight_pool.chunk_step

        def execute(intervene):
            reads = []
            def track(*args, **kwargs):
                if intervene and kwargs["state"].step == 2:
                    kwargs["v_chunk"] = kwargs["v_chunk"] + torch.linspace(-1, 1, 16)
                result = original(*args, **kwargs)
                reads.append(result[0].clone())
                return result
            with torch.no_grad(), patch.object(model.fast_weight_pool, "chunk_step", side_effect=track):
                model(ids)
            return reads

        baseline, changed = execute(False), execute(True)
        for index in range(3):
            torch.testing.assert_close(baseline[index], changed[index], atol=0, rtol=0)
        self.assertGreater(float((baseline[3] - changed[3]).abs().max()), 1e-7)

    def test_meta_and_test_time_writes_use_identical_dynamics(self):
        model = self.model().eval()
        ids = torch.randint(6, 128, (1, 11))
        state = model.init_fast_state(detach=True)
        with torch.no_grad():
            direct = model(ids, fast_state=state.clone(detach=True)).fast_state
            interactive, _, _ = model.test_time_update(ids, state)
        for name in ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down"):
            torch.testing.assert_close(getattr(direct, name), getattr(interactive, name), atol=0, rtol=0)
        self.assertEqual(direct.step, interactive.step)

    def test_checkpoint_preserves_cross_chunk_gradients_and_causality(self):
        ids = torch.randint(6, 128, (1, 12))
        changed = ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 11) % 128
        reference = self.model().train()
        optimized = self.model(True).train()
        expected = reference(ids, labels=ids)
        actual = optimized(ids, labels=ids)
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.logits[:, :6], optimized(changed).logits[:, :6])
        expected.loss.backward()
        actual.loss.backward()
        for (name, left), (_, right) in zip(reference.named_parameters(), optimized.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(right.grad, left.grad, atol=2e-5, rtol=2e-4, msg=name)

    def test_unsupported_schedule_is_not_silently_ignored(self):
        model = self.model().eval()
        model.config.write_schedule = "aggregated"
        with self.assertRaisesRegex(ValueError, "aggregated"):
            model(torch.ones(1, 4, dtype=torch.long))

    def test_surprise_gate_statistic_ignores_padded_chunks_and_partial_tail_padding(self):
        model = self.model().eval()
        # Fixed state isolates aggregation from changes in the write trajectory.
        for block in model.layers:
            block.uttt.base_inner_lr = 0
            block.uttt.base_weight_decay = 0
        ids = torch.randint(6, 128, (1, 5))
        padded = torch.cat([ids, torch.zeros(1, 7, dtype=torch.long)], dim=1)
        with torch.no_grad():
            expected = model(ids, attention_mask=torch.ones_like(ids))
            actual = model(padded, attention_mask=(padded != 0))
        torch.testing.assert_close(actual.surprise_loss, expected.surprise_loss)

    def test_masked_rows_and_trailing_empty_chunks_preserve_nonzero_momentum(self):
        model = self.model().eval()
        initial = model.init_fast_state(batch_size=2, detach=True)
        for name in ("s_up", "s_gate", "s_down"):
            getattr(initial, name).normal_(std=0.01)
        ids = torch.randint(6, 128, (2, 8))
        mask = torch.tensor([[0] * 8, [1] * 5 + [0] * 3])
        padded = torch.cat((ids, torch.zeros(2, 8, dtype=torch.long)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(2, 8, dtype=mask.dtype)), dim=1)
        with torch.no_grad():
            expected = model(ids, fast_state=initial.clone(detach=True), attention_mask=mask)
            actual = model(padded, fast_state=initial.clone(detach=True), attention_mask=padded_mask)
        for name in ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down"):
            with self.subTest(name=name):
                torch.testing.assert_close(getattr(actual.fast_state, name), getattr(expected.fast_state, name),
                                           atol=0, rtol=0)
                torch.testing.assert_close(getattr(actual.fast_state, name)[0], getattr(initial, name)[0],
                                           atol=0, rtol=0)
        self.assertFalse(torch.equal(actual.fast_state.w_up[1], initial.w_up[1]))
        torch.testing.assert_close(actual.surprise_loss, expected.surprise_loss)


if __name__ == "__main__":
    unittest.main()

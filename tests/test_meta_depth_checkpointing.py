"""Depth recomputation preserves exact shared-memory CE/meta gradients and causality."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.uttt_model import uTTTConfig, uTTTModel


STATE_NAMES = ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down", "ln_weight", "ln_bias")


class MetaDepthCheckpointingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, *, checkpointed=False, segment_layers=0, clip=0.0, dropout=0.0,
              time_checkpoint=True, reentrant=False):
        torch.manual_seed(8723)
        return uTTTModel(uTTTConfig(
            vocab_size=64, d_model=12, d_ff_fast=20, n_layers=5,
            n_heads=3, n_kv_heads=1, chunk_size=3, sliding_window_size=5,
            max_seq_len=32, gradient_checkpointing=checkpointed,
            meta_checkpoint_layers=segment_layers, max_inner_grad_norm=clip,
            meta_checkpoint_time_chunks=time_checkpoint, meta_checkpoint_reentrant=reentrant,
            dropout=dropout,
        )).train()

    def assert_gradients_match(self, expected, actual):
        left_parameters = dict(expected.named_parameters())
        right_parameters = dict(actual.named_parameters())
        self.assertEqual(left_parameters.keys(), right_parameters.keys())
        writer_gradients = []
        for name, left in left_parameters.items():
            right = right_parameters[name]
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                self.assertTrue(torch.isfinite(right.grad).all(), name)
                torch.testing.assert_close(right.grad, left.grad, atol=2e-5, rtol=3e-4, msg=name)
                if any(part in name for part in (".w_k.", ".w_v.", ".w_eta.", ".w_alpha.")):
                    writer_gradients.append(float(left.grad.norm()))
        # CE-only comparisons must really exercise writers through later chunks,
        # rather than passing because the write graph was detached in both paths.
        self.assertTrue(writer_gradients)
        self.assertTrue(all(norm > 0 for norm in writer_gradients))

    def test_multiple_depth_boundaries_match_whole_chunk_ce_and_all_meta_gradients(self):
        for batch_size in (1, 2):
            for clip in (0.0, 0.01):
                torch.manual_seed(391)
                ids = torch.randint(6, 64, (batch_size, 8))
                mask = torch.ones_like(ids)
                mask[-1, -1] = 0
                labels = ids.masked_fill(mask == 0, -100)
                reference = self.model(clip=clip)
                expected = reference(ids, labels=labels, attention_mask=mask,
                                     surprise_aux_weight=0.0)
                expected.loss.backward()
                # 2/3 split a five-layer stack unevenly; 1 exercises every layer,
                # and 8 exercises a segment larger than the network itself.
                for segment_layers in (0, 1, 2, 3, 8):
                    with self.subTest(batch=batch_size, clip=clip, segment_layers=segment_layers):
                        actual_model = self.model(checkpointed=True, segment_layers=segment_layers, clip=clip)
                        actual = actual_model(ids, labels=labels, attention_mask=mask,
                                              surprise_aux_weight=0.0)
                        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                        torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
                        torch.testing.assert_close(actual.surprise_loss, expected.surprise_loss, atol=0, rtol=0)
                        self.assertEqual(actual.fast_state.step, 15)
                        for name in STATE_NAMES:
                            torch.testing.assert_close(getattr(actual.fast_state, name),
                                                       getattr(expected.fast_state, name), atol=0, rtol=0)
                        actual.loss.backward()
                        self.assert_gradients_match(reference, actual_model)

    def test_auxiliary_loss_and_dropout_replay_preserve_gradients_and_rng(self):
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
        reference = self.model(clip=0.02, dropout=0.15)
        actual_model = self.model(checkpointed=True, segment_layers=2, clip=0.02, dropout=0.15)
        torch.manual_seed(3991)
        expected = reference(ids, labels=ids, surprise_aux_weight=0.2)
        expected_rng = torch.get_rng_state().clone()
        expected.loss.backward()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        torch.manual_seed(3991)
        actual = actual_model(ids, labels=ids, surprise_aux_weight=0.2)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        actual.loss.backward()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        self.assert_gradients_match(reference, actual_model)

    def test_depth_only_and_flattened_reentrant_backwards_match_exact_reference(self):
        for batch_size in (1, 2):
            for memory_only in (False, True):
                for time_checkpoint, reentrant in ((False, False), (True, True), (False, True)):
                    with self.subTest(batch=batch_size, memory_only=memory_only,
                                      time_checkpoint=time_checkpoint, reentrant=reentrant):
                        reference = self.model(clip=0.01)
                        actual_model = self.model(checkpointed=True, segment_layers=2, clip=0.01,
                                                  time_checkpoint=time_checkpoint, reentrant=reentrant)
                        if memory_only:
                            for model in (reference, actual_model):
                                for name, parameter in model.named_parameters():
                                    parameter.requires_grad_(name.startswith("fast_weight_pool.") or ".uttt." in name)
                        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]]).expand(batch_size, -1)
                        expected = reference(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
                        actual = actual_model(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
                        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
                        torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
                        torch.testing.assert_close(actual.surprise_loss, expected.surprise_loss, atol=0, rtol=0)
                        expected.loss.backward()
                        actual.loss.backward()
                        self.assert_gradients_match(reference, actual_model)

    def test_reentrant_dropout_and_auxiliary_meta_gradient_replay(self):
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
        reference = self.model(clip=0.01, dropout=0.1)
        actual_model = self.model(checkpointed=True, segment_layers=2, clip=0.01,
                                  dropout=0.1, reentrant=True)
        torch.manual_seed(902)
        expected = reference(ids, labels=ids, surprise_aux_weight=0.2)
        expected_rng = torch.get_rng_state().clone()
        expected.loss.backward()
        torch.manual_seed(902)
        actual = actual_model(ids, labels=ids, surprise_aux_weight=0.2)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
        actual.loss.backward()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        self.assert_gradients_match(reference, actual_model)

    def test_single_effective_chunk_falls_back_and_keeps_unused_writers_out_of_adam(self):
        # C=3: the label at index 3 is still predicted from the FIRST chunk.
        # Cases include one physical chunk, a final input-only second chunk,
        # padded physical chunks, and unmasked tails with ignored CE labels.
        for batch_size in (1, 3):
            for token_count, valid_targets_through, right_padding in (
                    (3, 2, False), (4, 3, False), (8, 3, True), (8, 3, False)):
                with self.subTest(batch=batch_size, token_count=token_count,
                                  valid_targets_through=valid_targets_through, right_padding=right_padding):
                    model = self.model(checkpointed=True, segment_layers=2, clip=0.01, reentrant=True)
                    ids = torch.arange(6, 6 + token_count).unsqueeze(0).expand(batch_size, -1)
                    labels = ids.clone()
                    labels[:, valid_targets_through + 1:] = -100
                    mask = torch.ones_like(ids)
                    if right_padding:
                        mask[:, valid_targets_through + 1:] = 0
                    writers = [(name, parameter) for name, parameter in model.named_parameters()
                               if any(part in name for part in (".w_k.", ".w_v.", ".w_eta.", ".w_alpha.", ".k_norm."))]
                    original_weights = {name: parameter.detach().clone() for name, parameter in writers}
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        with patch("model.uttt_model.checkpoint", wraps=checkpoint) as checkpoint_calls:
                            result = model(ids, labels=labels, attention_mask=mask,
                                           surprise_aux_weight=0.0, return_fast_state=False)
                        self.assertTrue(checkpoint_calls.call_args_list)
                        self.assertFalse(any(call.kwargs.get("use_reentrant", True)
                                             for call in checkpoint_calls.call_args_list))
                        if batch_size <= 2:
                            expected_ce = F.cross_entropy(result.logits[:, :-1].reshape(-1, 64),
                                                          labels[:, 1:].reshape(-1), ignore_index=-100)
                            torch.testing.assert_close(result.ce_loss, expected_ce, atol=1e-6, rtol=1e-6)
                        result.loss.backward()
                        for name, parameter in writers:
                            self.assertIsNone(parameter.grad, name)
                        optimizer.step()
                        for name, parameter in writers:
                            self.assertNotIn(parameter, optimizer.state, name)
                            torch.testing.assert_close(parameter, original_weights[name], atol=0, rtol=0, msg=name)

    def test_reentrant_admission_requires_future_supervised_prediction_not_just_second_chunk(self):
        model = self.model(checkpointed=True, segment_layers=2, reentrant=True)
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
        labels = torch.full_like(ids, -100)
        labels[:, 3] = ids[:, 3]  # Prediction index 2 belongs to the first C=3 chunk.
        self.assertEqual(model._meta_checkpoint_supervision(labels, None, ids.shape[1]), (False, True))
        labels[:, 4] = ids[:, 4]
        self.assertEqual(model._meta_checkpoint_supervision(labels, None, ids.shape[1]), (True, False))
        mask = torch.ones_like(ids)
        mask[:, 4:] = 0
        # Invalid labels on padding cannot establish safe admission; their CE
        # semantics are retained instead of silently dropping those targets.
        self.assertEqual(model._meta_checkpoint_supervision(labels, mask, ids.shape[1]), (False, False))
        self.assertEqual(model._meta_checkpoint_supervision(None, None, ids.shape[1]), (False, False))

    def test_segments_read_chunk_start_snapshot_and_advance_one_writer_cursor(self):
        model = self.model(checkpointed=True, segment_layers=2)
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
        records = []
        original = model.fast_weight_pool.chunk_step

        def track(*args, **kwargs):
            state, read = kwargs["state"], kwargs["read_state"]
            records.append((state.step, read.step, read.w_up.detach().clone()))
            return original(*args, **kwargs)

        with patch.object(model.fast_weight_pool, "chunk_step", side_effect=track):
            output = model(ids, return_fast_state=False)
        self.assertIsNone(output.fast_state)
        self.assertEqual([row[0] for row in records], list(range(15)))
        self.assertEqual([row[1] for row in records], [0] * 5 + [5] * 5 + [10] * 5)
        for start in (0, 5, 10):
            for row in records[start:start + 5]:
                torch.testing.assert_close(row[2], records[start][2], atol=0, rtol=0)
        self.assertFalse(torch.equal(records[0][2], records[5][2]))

    def test_nested_checkpoint_is_causal_across_depth_and_time(self):
        model = self.model(checkpointed=True, segment_layers=2)
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
        original = model(ids).logits
        for changed_from in (2, 3, 5, 6):
            changed = ids.clone()
            changed[:, changed_from:] += 17
            with self.subTest(changed_from=changed_from):
                actual = model(changed).logits
                torch.testing.assert_close(actual[:, :changed_from], original[:, :changed_from], atol=0, rtol=0)

    def test_segments_are_disposable_training_policy_and_do_not_change_eval_or_frozen_reads(self):
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
        baseline = self.model(checkpointed=True).eval()
        segmented = self.model(checkpointed=True, segment_layers=2).eval()
        self.assertEqual(baseline.state_dict().keys(), segmented.state_dict().keys())
        with torch.no_grad():
            expected = baseline(ids)
            actual = segmented(ids)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        for model in (baseline.train(), segmented.train()):
            model.zero_grad(set_to_none=True)
        expected = baseline(ids, labels=ids, update_fast_weights=False, return_fast_state=False)
        actual = segmented(ids, labels=ids, update_fast_weights=False, return_fast_state=False)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        expected.loss.backward()
        actual.loss.backward()
        for (name, left), (_, right) in zip(baseline.named_parameters(), segmented.named_parameters()):
            self.assertEqual(left.grad is None, right.grad is None, name)
            if left.grad is not None:
                torch.testing.assert_close(right.grad, left.grad, atol=0, rtol=0, msg=name)

    def test_config_accepts_nested_training_policy_and_rejects_invalid_segments(self):
        config = uTTTConfig.from_dict({"model": {"n_layers": 5}, "training": {"meta_checkpoint_layers": 2}})
        self.assertEqual(config.meta_checkpoint_layers, 2)
        self.assertEqual(uTTTConfig.from_dict(config.to_dict()).meta_checkpoint_layers, 2)
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                uTTTModel(uTTTConfig(meta_checkpoint_layers=value))
        with self.assertRaisesRegex(ValueError, "requires positive"):
            uTTTModel(uTTTConfig(meta_checkpoint_reentrant=True))
        for field in ("meta_checkpoint_time_chunks", "meta_checkpoint_reentrant"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "must be boolean"):
                uTTTModel(uTTTConfig(**{field: "true"}))


if __name__ == "__main__":
    unittest.main()

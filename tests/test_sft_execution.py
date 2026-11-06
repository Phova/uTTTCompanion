"""Within-step execution regrouping preserves semantic B1 sample weighting."""
import unittest

try:
    import torch
    from torch.nn import functional as F
    from engine.sft_execution import (
        SftExecutionPolicy,
        build_sft_execution_step,
        validate_sft_execution,
    )
except ImportError:
    torch = None


def source_batch(length, marker=11, width=512, targets=2, mask_dtype=None):
    ids = torch.zeros(1, width, dtype=torch.long)
    ids[0, :length] = (torch.arange(length) % 43) + 6
    ids[0, 0], ids[0, length - 1] = marker, 2
    labels = torch.full_like(ids, -100)
    begin = max(1, length - targets)
    labels[0, begin:length] = ids[0, begin:length]
    mask = torch.zeros(1, width, dtype=mask_dtype or torch.float32)
    mask[0, :length] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@unittest.skipIf(torch is None, "PyTorch unavailable")
class SftExecutionTests(unittest.TestCase):
    def test_long_rows_reduce_execution_size_without_truncation_or_loss_reweighting(self):
        batches = [source_batch(510, marker=40 + index) for index in range(16)]
        plan = build_sft_execution_step(batches, SftExecutionPolicy(4, 32, True), pad_token_id=0)
        self.assertEqual([group.sample_count for group in plan.groups], [2] * 8)
        self.assertEqual(plan.actual_tokens, 510 * 16)
        self.assertEqual(plan.physical_tokens, 512 * 16)
        self.assertEqual([index for group in plan.groups for index in group.source_indices], list(range(16)))
        self.assertAlmostEqual(sum(group.loss_weight for group in plan.groups), 1.)
        for value in (0, 1, True, 1536.):
            with self.assertRaisesRegex(ValueError, "max_tokens_per_group"):
                SftExecutionPolicy(4, 32, True, value)

    def validate(self, execution_batch_size=None, padding_multiple=128, sort_by_length=False, **changes):
        settings = dict(stage="sft", dropout=0., torch_compile=False,
                        micro_batch_size=1, trim_sft_padding=True)
        settings.update(changes)
        return validate_sft_execution(execution_batch_size, padding_multiple, sort_by_length, **settings)

    def test_omission_preserves_legacy_stages_and_explicit_policy_is_frozen_sft_only(self):
        self.assertIsNone(self.validate(stage="meta", dropout=.2, torch_compile=True, micro_batch_size=4,
                                        trim_sft_padding=False))
        self.assertEqual(self.validate(1), SftExecutionPolicy(1, 128, False))
        self.assertEqual(self.validate(4, 32, True), SftExecutionPolicy(4, 32, True))
        for settings in (dict(stage="pretrain"), dict(stage="meta"), dict(dropout=.1),
                         dict(torch_compile=True), dict(torch_compile=0), dict(micro_batch_size=2),
                         dict(micro_batch_size=True), dict(trim_sft_padding=False)):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "requires stage=sft"):
                self.validate(2, **settings)

    def test_invalid_policy_values_and_implicit_opt_in_are_rejected(self):
        for value in (0, -1, 1.5, "2", True):
            with self.subTest(batch=value), self.assertRaisesRegex(ValueError, "execution_batch_size"):
                self.validate(value)
        for value in (0, 8, 17, 256, 32., True):
            with self.subTest(multiple=value), self.assertRaisesRegex(ValueError, "padding_multiple"):
                self.validate(2, value)
        for value in (0, 1, "false", None):
            with self.subTest(sort=value), self.assertRaisesRegex(ValueError, "sort_by_length"):
                self.validate(2, 64, value)
        for options in ((None, 64, False), (None, 128, True)):
            with self.assertRaisesRegex(ValueError, "requires execution_batch_size"):
                self.validate(*options)

    def test_default_b1_rounding_matches_previous_trim_and_caps_partial_bucket(self):
        batches = [source_batch(30), source_batch(129), source_batch(129, width=130),
                   source_batch(511)]
        plan = build_sft_execution_step(batches, SftExecutionPolicy(), pad_token_id=0)
        self.assertEqual([tuple(group.batch["input_ids"].shape) for group in plan.groups],
                         [(1, 128), (1, 256), (1, 130), (1, 512)])
        self.assertEqual([group.source_indices for group in plan.groups], [(0,), (1,), (2,), (3,)])
        self.assertEqual(plan.actual_tokens, 799)
        self.assertEqual(plan.physical_tokens, 1026)
        self.assertEqual(plan.total_samples, 4)

    def test_regrouping_keeps_all_rows_real_eos_labels_and_input_memory_unchanged(self):
        batches = [source_batch(length, marker=40 + index, targets=index + 1)
                   for index, length in enumerate((17, 65, 33, 129, 66))]
        before = [{name: tensor.clone() for name, tensor in batch.items()} for batch in batches]
        rng = torch.get_rng_state().clone()
        plan = build_sft_execution_step(batches, SftExecutionPolicy(2, 32), pad_token_id=9)
        self.assertEqual([group.sample_count for group in plan.groups], [2, 2, 1])
        self.assertEqual([tuple(group.batch["input_ids"].shape) for group in plan.groups],
                         [(2, 96), (2, 160), (1, 96)])
        self.assertEqual([group.source_indices for group in plan.groups], [(0, 1), (2, 3), (4,)])
        self.assertEqual([group.loss_weight for group in plan.groups], [.4, .4, .2])
        self.assertAlmostEqual(sum(group.loss_weight for group in plan.groups), 1.)
        self.assertEqual(plan.actual_tokens, 310)
        self.assertEqual(plan.physical_tokens, 608)
        for group in plan.groups:
            for row, index in enumerate(group.source_indices):
                length = int(batches[index]["attention_mask"].sum())
                for name in ("input_ids", "labels", "attention_mask"):
                    torch.testing.assert_close(group.batch[name][row, :length], before[index][name][0, :length])
                    self.assertTrue(group.batch[name].is_contiguous())
                    self.assertFalse(group.batch[name].is_pinned())
                self.assertEqual(int(group.batch["labels"][row, length - 1]), 2)
                self.assertTrue(bool((group.batch["input_ids"][row, length:] == 9).all()))
                self.assertTrue(bool((group.batch["labels"][row, length:] == -100).all()))
                self.assertTrue(bool((group.batch["attention_mask"][row, length:] == 0).all()))
        for original, expected in zip(batches, before):
            for name in original:
                torch.testing.assert_close(original[name], expected[name])
        torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_sorting_is_stable_confined_to_one_step_and_reduces_padding(self):
        batches = [source_batch(length, marker=40 + index)
                   for index, length in enumerate((17, 220, 17, 240, 31, 230))]
        unsorted = build_sft_execution_step(batches, SftExecutionPolicy(2, 32), pad_token_id=0)
        sorted_plan = build_sft_execution_step(batches, SftExecutionPolicy(2, 32, True), pad_token_id=0)
        self.assertEqual([group.source_indices for group in sorted_plan.groups], [(0, 2), (4, 1), (5, 3)])
        self.assertLess(sorted_plan.physical_tokens, unsorted.physical_tokens)
        self.assertEqual(sorted_plan.actual_tokens, unsorted.actual_tokens)
        self.assertEqual(sorted(index for group in sorted_plan.groups for index in group.source_indices),
                         list(range(6)))
        other_step = build_sft_execution_step([source_batch(400, marker=61)],
                                             SftExecutionPolicy(4, 16, True), pad_token_id=0)
        self.assertEqual(other_step.groups[0].source_indices, (0,))
        self.assertEqual(other_step.total_samples, 1)
        self.assertEqual(sorted_plan.total_samples, 6)

    def test_group_weights_preserve_b1_ce_and_gradient_with_unequal_targets_and_tail(self):
        batches = [source_batch(length, marker=40 + index, targets=targets)
                   for index, (length, targets) in enumerate(((19, 1), (64, 15), (35, 2), (128, 25), (22, 3)))]
        initial = torch.linspace(-.6, .9, 64, dtype=torch.float64)
        before, after = initial.clone().requires_grad_(), initial.clone().requires_grad_()

        def sample_mean_ce(batch, weight):
            rows = []
            for row in range(batch["input_ids"].shape[0]):
                target = batch["labels"][row, 1:]
                # A simple shared trainable classifier makes the gradient test
                # sensitive to target count and to erroneous tail-group weights.
                logits = weight.expand(target.shape[0], -1)
                rows.append(F.cross_entropy(logits, target, ignore_index=-100))
            return torch.stack(rows).mean()

        reference = torch.stack([sample_mean_ce(batch, before) for batch in batches]).mean()
        reference.backward()
        for size in (1, 2, 4, 16):
            after.grad = None
            plan = build_sft_execution_step(batches, SftExecutionPolicy(size, 32, True), pad_token_id=0)
            grouped = sum(sample_mean_ce(group.batch, after) * group.loss_weight for group in plan.groups)
            grouped.backward()
            torch.testing.assert_close(grouped, reference, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(after.grad, before.grad, rtol=1e-12, atol=1e-12)

    def test_binary_masks_and_all_padding_buckets_are_supported(self):
        for multiple, expected_width in ((16, 48), (32, 64), (64, 64), (128, 128)):
            for mask_dtype in (torch.bool, torch.float32, torch.long):
                with self.subTest(multiple=multiple, dtype=mask_dtype):
                    plan = build_sft_execution_step([source_batch(35, mask_dtype=mask_dtype)],
                                                   SftExecutionPolicy(2, multiple), pad_token_id=0)
                    self.assertEqual(plan.groups[0].batch["attention_mask"].dtype, mask_dtype)
                    self.assertEqual(plan.groups[0].batch["input_ids"].shape, (1, expected_width))

    def test_empty_malformed_unlabeled_and_non_cpu_inputs_are_rejected(self):
        policy = SftExecutionPolicy(2, 32)
        with self.assertRaisesRegex(ValueError, "nonempty materialized"):
            build_sft_execution_step([], policy, pad_token_id=0)
        with self.assertRaisesRegex(ValueError, "validated SftExecutionPolicy"):
            build_sft_execution_step([source_batch(30)], None, pad_token_id=0)
        for value in (-1, True, 0.):
            with self.assertRaisesRegex(ValueError, "pad_token_id"):
                build_sft_execution_step([source_batch(30)], policy, pad_token_id=value)
        changes = [
            ("matching semantic B1", lambda batch: {name: tensor.repeat(2, 1) for name, tensor in batch.items()}),
            ("matching semantic B1", lambda batch: {**batch, "labels": batch["labels"][:, :-1]}),
            ("torch.long", lambda batch: {**batch, "input_ids": batch["input_ids"].float()}),
            ("requires exactly", lambda batch: {**batch, "sample_id": torch.tensor([1])}),
            ("requires tensors", lambda batch: {**batch, "attention_mask": None}),
            ("CPU", lambda batch: {name: tensor.to("meta") for name, tensor in batch.items()}),
        ]
        for message, mutate in changes:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_sft_execution_step([mutate(source_batch(30))], policy, pad_token_id=0)
        for bad_value in (.5, float("nan")):
            batch = source_batch(30)
            batch["attention_mask"][0, 8] = bad_value
            with self.assertRaisesRegex(ValueError, "binary"):
                build_sft_execution_step([batch], policy, pad_token_id=0)
        batch = source_batch(30)
        batch["attention_mask"][0, 8] = 0
        with self.assertRaisesRegex(ValueError, "right padding"):
            build_sft_execution_step([batch], policy, pad_token_id=0)
        batch = source_batch(30)
        batch["labels"][0, 80] = 5
        with self.assertRaisesRegex(ValueError, "supervised padding"):
            build_sft_execution_step([batch], policy, pad_token_id=0)
        batch = source_batch(30)
        batch["labels"].fill_(-100)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            build_sft_execution_step([batch], policy, pad_token_id=0)
        batch = source_batch(30)
        batch["attention_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_sft_execution_step([batch], policy, pad_token_id=0)
        with self.assertRaisesRegex(ValueError, "dtypes must agree"):
            build_sft_execution_step([source_batch(30), source_batch(40, mask_dtype=torch.bool)],
                                     policy, pad_token_id=0)


if __name__ == "__main__":
    unittest.main()

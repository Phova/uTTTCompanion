"""Batched frozen-write SFT must preserve the one-sequence training objective."""
import unittest

try:
    import torch
    from model.uttt_model import uTTTConfig, uTTTModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch model dependencies unavailable")
class SequenceMeanCeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def models(self):
        torch.manual_seed(717)
        config = uTTTConfig(vocab_size=64, d_model=16, n_layers=2, n_heads=2,
                            n_kv_heads=1, d_ff_fast=32, chunk_size=8,
                            sliding_window_size=16, dropout=0., gradient_checkpointing=False)
        left, right = uTTTModel(config).double().train(), uTTTModel(config).double().train()
        right.load_state_dict(left.state_dict())
        return left, right

    def batch(self, size):
        generator = torch.Generator().manual_seed(438)
        lengths, target_counts = (11, 27, 38, 50), (2, 4, 12, 35)
        ids = torch.zeros(size, 64, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros(size, 64, dtype=torch.float64)
        for row in range(size):
            length, count = lengths[row], target_counts[row]
            ids[row, :length] = torch.randint(6, 64, (length,), generator=generator)
            ids[row, length - 1] = 2
            labels[row, length - count:length] = ids[row, length - count:length]
            mask[row, :length] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    def assert_gradients(self, left, right):
        for (name, expected), (other_name, actual) in zip(left.named_parameters(), right.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(expected.grad is None, actual.grad is None, name)
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, atol=3e-10, rtol=3e-8, msg=name)

    def test_default_token_average_is_unchanged_for_b1_b2_and_sliced_b4(self):
        for size in (1, 2, 4):
            with self.subTest(batch_size=size):
                implicit, explicit = self.models()
                batch = self.batch(size)
                before = implicit(**batch, update_fast_weights=False, return_fast_state=False)
                after = explicit(**batch, update_fast_weights=False, return_fast_state=False,
                                 ce_reduction="token_mean")
                torch.testing.assert_close(after.ce_loss, before.ce_loss, rtol=0, atol=0)
                before.loss.backward()
                after.loss.backward()
                self.assert_gradients(implicit, explicit)

    def test_sequence_average_matches_individual_losses_and_all_parameter_gradients(self):
        for size in (1, 2, 4):
            with self.subTest(batch_size=size):
                individual, batched = self.models()
                batch = self.batch(size)
                one_losses = []
                counts = []
                for row in range(size):
                    length = int(batch["attention_mask"][row].sum())
                    one = {key: value[row:row + 1, :length] for key, value in batch.items()}
                    result = individual(**one, update_fast_weights=False, return_fast_state=False)
                    one_losses.append(result.loss)
                    counts.append(int((one["labels"][:, 1:] != -100).sum()))
                expected = torch.stack(one_losses).mean()
                result = batched(**batch, update_fast_weights=False, return_fast_state=False,
                                 ce_reduction="sequence_mean")
                torch.testing.assert_close(result.loss, expected, atol=3e-10, rtol=3e-8)
                expected.backward()
                result.loss.backward()
                self.assert_gradients(individual, batched)
                if size > 1:
                    # Unequal answer lengths make this a different objective
                    # from one token-weighted CE over the enlarged microbatch.
                    weighted = sum(loss.detach() * count for loss, count in zip(one_losses, counts)) / sum(counts)
                    self.assertGreater(float((weighted - expected.detach()).abs()), 1e-5)

    def test_sliced_head_retains_per_sequence_weights_across_both_slices(self):
        _, model = self.models()
        shapes = []
        handle = model.lm_head.register_forward_hook(lambda module, inputs, output: shapes.append(tuple(inputs[0].shape)))
        try:
            output = model(**self.batch(4), update_fast_weights=False, return_fast_state=False,
                           ce_reduction="sequence_mean")
        finally:
            handle.remove()
        self.assertEqual(shapes, [(2 * 63, 16), (2 * 63, 16)])
        self.assertEqual(tuple(output.logits.shape), (4, 0, 64))
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.tok_embeddings.weight.grad).all())

    def test_zero_target_row_matches_legacy_one_sequence_zero_contribution(self):
        individual, batched = self.models()
        batch = self.batch(2)
        batch["labels"][0].fill_(-100)
        losses = [individual(**{key: value[row:row + 1] for key, value in batch.items()},
                             update_fast_weights=False, return_fast_state=False).loss for row in range(2)]
        expected = torch.stack(losses).mean()
        actual = batched(**batch, update_fast_weights=False, return_fast_state=False,
                         ce_reduction="sequence_mean").loss
        torch.testing.assert_close(losses[0], torch.zeros_like(losses[0]))
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-8)
        expected.backward()
        actual.backward()
        self.assert_gradients(individual, batched)

    def test_sequence_mean_is_explicit_and_cannot_change_dynamic_meta_objectives(self):
        model, _ = self.models()
        batch = self.batch(1)
        for reduction in ("mean", "sum", None, 1):
            with self.subTest(reduction=reduction), self.assertRaisesRegex(ValueError, "ce_reduction"):
                model(**batch, update_fast_weights=False, ce_reduction=reduction)
        with self.assertRaisesRegex(ValueError, "update_fast_weights=False"):
            model(**batch, update_fast_weights=True, ce_reduction="sequence_mean")
        unlabeled = {key: value for key, value in batch.items() if key != "labels"}
        with self.assertRaisesRegex(ValueError, "requires labels"):
            model(**unlabeled, update_fast_weights=False, ce_reduction="sequence_mean")


if __name__ == "__main__":
    unittest.main()

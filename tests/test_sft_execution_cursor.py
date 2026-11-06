"""Execution fusion retains B1 sample/RNG order across epochs and OOM replay."""
import unittest

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from engine.corpus import ResumableBatchSampler
    from engine.sft_execution import SftExecutionPolicy, materialize_sft_execution_step
except ImportError:
    torch = None
    Dataset = object


class RandomWindowDataset(Dataset):
    def __init__(self, size=19):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Model random window/turn selection in the trainer's main process.
        length = int(torch.randint(9, 47, ()).item())
        ids = torch.zeros(64, dtype=torch.long)
        ids[:length] = torch.randint(6, 62, (length,))
        ids[0], ids[length - 1] = 100 + index, 2
        labels = torch.full_like(ids, -100)
        labels[length - 3:length] = ids[length - 3:length]
        mask = torch.zeros(64)
        mask[:length] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@unittest.skipIf(torch is None, "PyTorch unavailable")
class SftExecutionCursorTests(unittest.TestCase):
    def loader(self, size=19):
        sampler = ResumableBatchSampler(size, 1, seed=814)
        loader = DataLoader(RandomWindowDataset(size), batch_sampler=sampler,
                            num_workers=0, generator=torch.Generator().manual_seed(993))
        return loader, sampler, iter(loader)

    @staticmethod
    def valid_row(batch, row=0):
        length = int(batch["attention_mask"][row].sum())
        return {name: value[row, :length].tolist() for name, value in batch.items()}

    def ordered_plan_rows(self, plan):
        rows = {}
        for group in plan.groups:
            for row, index in enumerate(group.source_indices):
                self.assertNotIn(index, rows)
                rows[index] = self.valid_row(group.batch, row)
        self.assertEqual(sorted(rows), list(range(plan.total_samples)))
        return [rows[index] for index in range(plan.total_samples)]

    def reference_step(self, loader, iterator, sampler, count=16):
        rows = []
        for _ in range(count):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            rows.append(self.valid_row(batch))
            sampler.advance(1)
        return rows, iterator

    def test_two_steps_cross_epoch_without_dropping_tail_or_changing_random_windows(self):
        torch.manual_seed(771)
        initial_rng = torch.get_rng_state().clone()
        loader, sampler, iterator = self.loader()
        expected = []
        for _ in range(2):
            rows, iterator = self.reference_step(loader, iterator, sampler)
            expected.append(rows)
        expected_cursor, expected_rng = sampler.state_dict(), torch.get_rng_state().clone()

        torch.set_rng_state(initial_rng)
        loader, sampler, iterator = self.loader()
        actual = []
        for _ in range(2):
            plan, iterator = materialize_sft_execution_step(
                loader, iterator, sampler, 16, SftExecutionPolicy(4, 32, True), pad_token_id=0)
            actual.append(self.ordered_plan_rows(plan))
            self.assertEqual(plan.total_samples, 16)
            self.assertAlmostEqual(sum(group.loss_weight for group in plan.groups), 1.)
        self.assertEqual(actual, expected)
        self.assertEqual(sampler.state_dict(), expected_cursor)
        self.assertEqual((sampler.epoch, sampler.offset), (1, 13))
        self.assertEqual(len({row["input_ids"][0] for rows in actual for row in rows} ), 19)
        self.assertEqual(len(actual[0] + actual[1][:3]), 19)
        self.assertEqual(len({row["input_ids"][0] for row in actual[0] + actual[1][:3]}), 19)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_oom_after_partial_group_execution_replays_entire_step_from_cursor_and_rng(self):
        torch.manual_seed(176)
        loader, sampler, iterator = self.loader()
        # Begin close to an epoch boundary to exercise its remaining B1 tail.
        _, iterator = self.reference_step(loader, iterator, sampler, count=13)
        start_cursor, start_rng = sampler.state_dict(), torch.get_rng_state().clone()
        first, iterator = materialize_sft_execution_step(
            loader, iterator, sampler, 16, SftExecutionPolicy(4, 32, True), pad_token_id=0)
        expected_rows = self.ordered_plan_rows(first)
        expected_cursor, expected_rng = sampler.state_dict(), torch.get_rng_state().clone()
        partial = []
        try:
            for index, group in enumerate(first.groups):
                if index == 1:
                    raise torch.OutOfMemoryError("simulated after one fused backward")
                partial.extend(group.source_indices)
        except torch.OutOfMemoryError:
            self.assertEqual(len(partial), 4)
            sampler.load_state_dict(start_cursor)
            torch.set_rng_state(start_rng)
            iterator = iter(loader)
        # A reduced execution group is allowed on retry; all 16 samples replay.
        replay, iterator = materialize_sft_execution_step(
            loader, iterator, sampler, 16, SftExecutionPolicy(2, 32, True), pad_token_id=0)
        self.assertEqual(self.ordered_plan_rows(replay), expected_rows)
        self.assertEqual(sampler.state_dict(), expected_cursor)
        self.assertEqual((sampler.epoch, sampler.offset), (1, 10))
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_materialization_failure_can_restore_samples_and_random_windows(self):
        torch.manual_seed(991)
        loader, sampler, iterator = self.loader()
        start_cursor, start_rng = sampler.state_dict(), torch.get_rng_state().clone()
        expected, iterator = materialize_sft_execution_step(
            loader, iterator, sampler, 16, SftExecutionPolicy(2, 32), pad_token_id=0)
        expected_rows = self.ordered_plan_rows(expected)
        expected_rng = torch.get_rng_state().clone()
        sampler.load_state_dict(start_cursor)
        torch.set_rng_state(start_rng)
        iterator = iter(loader)

        def fail_after_three(original):
            for index in range(16):
                if index == 3:
                    raise torch.OutOfMemoryError("simulated while materializing")
                yield next(original)

        with self.assertRaises(torch.OutOfMemoryError):
            materialize_sft_execution_step(loader, fail_after_three(iterator), sampler, 16,
                                           SftExecutionPolicy(2, 32), pad_token_id=0)
        self.assertEqual(sampler.offset, 3)
        sampler.load_state_dict(start_cursor)
        torch.set_rng_state(start_rng)
        iterator = iter(loader)
        replay, iterator = materialize_sft_execution_step(
            loader, iterator, sampler, 16, SftExecutionPolicy(2, 32), pad_token_id=0)
        self.assertEqual(self.ordered_plan_rows(replay), expected_rows)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_invalid_accumulation_and_non_b1_sampler_fail_before_consuming_data(self):
        torch.manual_seed(881)
        loader, sampler, iterator = self.loader()
        initial_cursor, initial_rng = sampler.state_dict(), torch.get_rng_state().clone()
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "grad_accum_steps"):
                materialize_sft_execution_step(loader, iterator, sampler, count,
                                               SftExecutionPolicy(2), pad_token_id=0)
        sampler.batch_size = 2
        with self.assertRaisesRegex(ValueError, "original B1 sampler"):
            materialize_sft_execution_step(loader, iterator, sampler, 16,
                                           SftExecutionPolicy(2), pad_token_id=0)
        sampler.batch_size = 1
        self.assertEqual(sampler.state_dict(), initial_cursor)
        torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

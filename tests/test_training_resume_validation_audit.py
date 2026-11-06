"""Full Adam resumes, immutable validation and truthful padded-token timing."""
from contextlib import redirect_stdout
import gc
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
import weakref
from unittest import mock

try:
    import torch
    import yaml
    import train_5090 as training
    from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies unavailable")
class ResumeValidationAuditTests(unittest.TestCase):
    def tiny_model(self):
        torch.manual_seed(85)
        return uTTTModel(uTTTConfig(vocab_size=512, d_model=8, n_layers=1, n_heads=2,
                                    n_kv_heads=1, d_ff_fast=16, chunk_size=4,
                                    sliding_window_size=16, gradient_checkpointing=False))

    def optimizer_fixture(self):
        used = torch.nn.Parameter(torch.ones(2, 3))
        unused = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.AdamW([
            {"name": "used", "params": [used]}, {"name": "inactive", "params": [unused]},
        ])
        used.sum().backward()
        optimizer.step()
        return optimizer

    def test_sparse_adam_states_resume_but_missing_moments_or_bad_steps_fail(self):
        optimizer = self.optimizer_fixture()
        state = optimizer.state_dict()
        self.assertEqual(len(state["state"]), 1)  # Inactive writers need no state.
        training.validate_resume_optimizer_state(optimizer, state, 7)
        for value in (None, {}, {"state": {}, "param_groups": state["param_groups"]}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "full checkpoint.*AdamW"):
                    training.validate_resume_optimizer_state(optimizer, value, 7)
        key = next(iter(state["state"]))
        original_step = state["state"][key]["step"]
        for step in (None, torch.tensor(float("nan")), torch.tensor(8.), torch.tensor(.5), torch.zeros(2)):
            with self.subTest(step=step):
                state["state"][key]["step"] = step
                with self.assertRaisesRegex(ValueError, "AdamW step"):
                    training.validate_resume_optimizer_state(optimizer, state, 7)
        state["state"][key]["step"] = original_step
        state["state"][key]["exp_avg"] = torch.zeros(3, 2)
        with self.assertRaisesRegex(ValueError, "exp_avg shape"):
            training.validate_resume_optimizer_state(optimizer, state, 7)

    def test_reordered_or_orphan_adam_groups_fail_before_load(self):
        optimizer = self.optimizer_fixture()
        state = optimizer.state_dict()
        state["param_groups"].reverse()
        with self.assertRaisesRegex(ValueError, "parameter groups"):
            training.validate_resume_optimizer_state(optimizer, state, 1)
        state = optimizer.state_dict()
        state["state"][999] = state["state"].pop(next(iter(state["state"])))
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            training.validate_resume_optimizer_state(optimizer, state, 1)

    def fixture(self, root):
        data = root / "train.txt"
        data.write_text("a\nb\n", encoding="utf-8")
        config = root / "config.yaml"
        config.write_text(yaml.safe_dump({
            "model": {"vocab_size": 512, "d_model": 8, "n_layers": 1, "n_heads": 2,
                      "n_kv_heads": 1, "d_ff_fast": 16, "chunk_size": 4},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": "pretrain", "micro_batch_size": 4, "grad_accum_steps": 1,
                         "max_steps": 1, "seq_len": 16, "pack_sequences": False,
                         "gradient_checkpointing": False},
        }), encoding="utf-8")
        return config, data

    def test_weight_only_resume_preserves_file_and_never_loads_model_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root)
            checkpoint = training.train(str(config), str(data), str(root / "full.pt"))
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved.pop("optimizer_state")
            torch.save(saved, checkpoint)
            before = checkpoint.read_bytes()
            with mock.patch.object(training.uTTTModel, "load_dual_base_state") as load:
                with self.assertRaisesRegex(RuntimeError, "Could not restore") as raised:
                    training.train(str(config), str(data), str(checkpoint), override_steps=2)
            self.assertIn("--init-from", str(raised.exception.__cause__))
            load.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_resume_releases_mapped_rng_tensor_before_the_next_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root)
            checkpoint = training.train(str(config), str(data), str(root / "full.pt"))
            restore = training.restore_training_rng
            forward = training.uTTTModel.forward
            rng_refs, resumed_forwards = [], []

            def observe_restore(state):
                rng_refs.append(weakref.ref(state["cpu"]))
                restore(state)

            def verify_released(model, *args, **kwargs):
                gc.collect()
                self.assertEqual(len(rng_refs), 1)
                self.assertIsNone(rng_refs[0](), "resume frame still owns the checkpoint's mapped RNG tensor")
                resumed_forwards.append(True)
                return forward(model, *args, **kwargs)

            # Plain replacement functions deliberately avoid Mock call records:
            # recording the restore arguments would itself retain the mmap.
            with mock.patch.object(training, "restore_training_rng", new=observe_restore), \
                    mock.patch.object(training.uTTTModel, "forward", new=verify_released):
                training.train(str(config), str(data), str(checkpoint), override_steps=2)
            self.assertTrue(resumed_forwards)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(saved["step"], 2)
            self.assertTrue(all(int(value["step"]) == 2 for value in saved["optimizer_state"]["state"].values()))

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/proc/self/maps").exists(), "Linux mmap lifetime")
    def test_tiny_rng_reference_keeps_a_rotated_checkpoint_mapping_until_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.pt"
            torch.save({"weights": torch.zeros(16384), "rng": torch.get_rng_state()}, path)

            def mapped_lines():
                return [line for line in Path("/proc/self/maps").read_text().splitlines() if str(path) in line]

            saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            rng = saved["rng"]
            rng_ref = weakref.ref(rng)
            self.assertTrue(mapped_lines())
            training.restore_training_rng({"cpu": rng})
            del saved
            path.unlink()  # Simulate the inode losing its final checkpoint name.
            gc.collect()
            self.assertIsNotNone(rng_ref())
            self.assertTrue(any("(deleted)" in line for line in mapped_lines()))
            del rng
            gc.collect()
            self.assertIsNone(rng_ref())
            self.assertFalse(mapped_lines())

    def test_speed_counts_actual_unpadded_tokens_and_actual_small_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root)
            output = io.StringIO()
            with redirect_stdout(output), mock.patch.object(training.time, "perf_counter", side_effect=[10., 12., 12.]):
                checkpoint = training.train(str(config), str(data), str(root / "full.pt"))
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            tokens = saved["metrics"]["tokens_seen"]
            speed = int(re.search(r"Speed: ([\d,]+) tok/s", output.getvalue()).group(1).replace(",", ""))
            self.assertEqual(tokens, 6)  # Two rows, each BOS / body / EOS, rather than 4*16.
            self.assertEqual(speed, tokens // 2)

    def dataset(self):
        return [{"input_ids": torch.tensor([1, 6, 7, 8, 9, 2, 0, 0]),
                 "labels": torch.tensor([-100, 6, 7, 8, 9, 2, -100, -100]),
                 "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 0, 0])},
                {"input_ids": torch.tensor([1, 10, 11, 2, 0, 0, 0, 0]),
                 "labels": torch.tensor([-100, 10, 11, 2, -100, -100, -100, -100]),
                 "attention_mask": torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])}]

    def legacy_validation(self, model, dataset, dtype, updates):
        totals, frozen_totals, targets = 0., 0., 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in dataset:
                    inputs = {name: value.unsqueeze(0) for name, value in batch.items()}
                    count = int((batch["labels"][1:] != -100).sum())
                    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
                        result = model(**inputs, update_fast_weights=updates, surprise_aux_weight=0.)
                        frozen = model(**inputs, update_fast_weights=False, surprise_aux_weight=0.) if updates else result
                    totals += float(result.ce_loss) * count
                    frozen_totals += float(frozen.ce_loss) * count
                    targets += count
            return {"validation_ce": totals / targets, "validation_frozen_ce": frozen_totals / targets,
                    "validation_targets": targets}
        finally:
            model.train(was_training)

    def test_validation_matches_independent_mutable_state_baseline_and_reuses_amp_scope(self):
        for dtype in (torch.float32, torch.bfloat16):
            for updates in (False, True):
                with self.subTest(dtype=dtype, updates=updates):
                    model = self.tiny_model().train()
                    dataset = self.dataset()
                    expected = self.legacy_validation(model, dataset, dtype, updates)
                    before_rng = torch.get_rng_state().clone()
                    before = {key: value.clone() for key, value in model.state_dict().items()}
                    with mock.patch.object(torch, "autocast", wraps=torch.autocast) as amp, \
                            mock.patch.object(model, "init_fast_state", wraps=model.init_fast_state) as mutable, \
                            mock.patch.object(model.fast_weight_pool, "readonly_state", wraps=model.fast_weight_pool.readonly_state) as readonly, \
                            mock.patch.object(model, "forward", wraps=model.forward) as forward:
                        actual = training.evaluate_language_model(model, dataset, torch.device("cpu"), dtype, updates)
                    self.assertEqual(actual, expected)
                    self.assertTrue(model.training)
                    self.assertEqual(amp.call_count, 1)
                    self.assertTrue(all(call.kwargs["return_fast_state"] is False for call in forward.call_args_list))
                    self.assertEqual(mutable.call_count, len(dataset) if updates else 0)
                    self.assertEqual(readonly.call_count, 1)
                    frozen_calls = [call for call in forward.call_args_list if not call.kwargs["update_fast_weights"]]
                    self.assertTrue(all(call.kwargs["fast_state"] is frozen_calls[0].kwargs["fast_state"] for call in frozen_calls))
                    torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_validation_failure_restores_training_mode_and_exits_amp(self):
        model = self.tiny_model()
        for was_training in (False, True):
            with self.subTest(was_training=was_training):
                model.train(was_training)
                with mock.patch.object(model, "forward", side_effect=RuntimeError("injected validation failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        training.evaluate_language_model(model, self.dataset(), torch.device("cpu"), torch.bfloat16)
                self.assertEqual(model.training, was_training)
                self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_long_chat_validation_retains_assistant_suffix_and_pretrain_stays_raw(self):
        model = self.tiny_model()
        model.config.max_seq_len = 32
        tokenizer = SimpleUTF8Tokenizer(512)
        observed = []

        def generate(ids, *_args, **_kwargs):
            observed.append(ids[0].tolist())
            yield tokenizer.EOS_ID

        prompt = "old " * 20 + "recent question"
        with mock.patch.object(model, "stream_generate", side_effect=generate):
            training.generate_validation_samples(model, tokenizer, [prompt], "sft", torch.device("cpu"),
                                                 torch.float32, max_new_tokens=4)
            training.generate_validation_samples(model, tokenizer, [prompt], "pretrain", torch.device("cpu"),
                                                 torch.float32, max_new_tokens=4)
        self.assertEqual(observed[0], training.encode_chat_prompt(tokenizer, prompt, 28))
        self.assertIn(tokenizer.ASSISTANT_ID, observed[0][-3:])
        self.assertTrue(tokenizer.decode(observed[0]).rstrip().endswith("recent question"))
        self.assertEqual(observed[2], tokenizer.encode(prompt, add_bos=True, max_length=28))


if __name__ == "__main__":
    unittest.main()

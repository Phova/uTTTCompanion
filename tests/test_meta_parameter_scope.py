"""Memory-only meta training preserves CE gradients and refuses ambiguous resumes."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

import train_5090 as training
from model.uttt_model import uTTTConfig, uTTTModel


class MetaParameterScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, checkpointed=False):
        torch.manual_seed(851)
        return uTTTModel(uTTTConfig(
            vocab_size=64, d_model=12, d_ff_fast=20, n_layers=3,
            n_heads=3, n_kv_heads=1, chunk_size=3, sliding_window_size=5,
            max_seq_len=32, gradient_checkpointing=checkpointed,
            meta_checkpoint_layers=2 if checkpointed else 0,
            meta_checkpoint_reentrant=checkpointed, meta_checkpoint_time_chunks=True,
            max_inner_grad_norm=0.1,
        )).train()

    def test_scoped_ce_gradients_match_full_backbone_and_two_adam_steps_leave_frozen_weights_unchanged(self):
        ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
        for checkpointed in (False, True):
            with self.subTest(checkpointed=checkpointed):
                reference = self.model(checkpointed)
                reference(ids, labels=ids, update_fast_weights=True, surprise_aux_weight=0.0,
                          return_fast_state=False).loss.backward()
                model = self.model(checkpointed)
                stats = training.apply_training_parameter_scope(model, "meta", "memory")
                named = dict(model.named_parameters())
                selected = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
                frozen = {name: parameter.detach().clone() for name, parameter in named.items() if not parameter.requires_grad}
                self.assertTrue(frozen)
                self.assertTrue(all(name.startswith("fast_weight_pool.") or ".uttt." in name for name in selected))
                self.assertTrue(all(".attn." not in name and ".uttt_norm." not in name for name in selected))
                self.assertEqual(stats["trainable"], sum(parameter.numel() for parameter in selected.values()))
                self.assertEqual(stats["total"], stats["trainable"] + stats["frozen"])
                optimizer = torch.optim.AdamW([
                    {"name": "slow_weights", "params": [p for p in model.get_slow_parameters() if p.requires_grad]},
                    {"name": "init_fast_weights", "params": [p for p in model.get_meta_fast_parameters() if p.requires_grad]},
                ], lr=1e-3)
                for step in range(1, 3):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(ids, labels=ids, update_fast_weights=True, surprise_aux_weight=0.0,
                                   return_fast_state=False)
                    output.loss.backward()
                    for name, parameter in selected.items():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        if step == 1:
                            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                                       atol=2e-6, rtol=3e-4, msg=name)
                    writers = [parameter for name, parameter in selected.items()
                               if any(marker in name for marker in (".w_k.", ".w_v.", ".w_eta.", ".w_alpha."))]
                    self.assertTrue(writers)
                    self.assertTrue(all(float(parameter.grad.norm()) > 0 for parameter in writers))
                    self.assertTrue(all(named[name].grad is None for name in frozen))
                    optimizer.step()
                    self.assertEqual(set(optimizer.state), set(selected.values()))
                    self.assertTrue(all(int(state["step"]) == step for state in optimizer.state.values()))
                    for name, expected in frozen.items():
                        torch.testing.assert_close(named[name], expected, atol=0, rtol=0, msg=name)

    def fixture(self, root, scope="memory", stage="meta"):
        data = root / "train.txt"
        data.write_text("\n".join(f"<|user|> {i}?<|assistant|> {i}." for i in range(8)), encoding="utf-8")
        config = root / "config.yaml"
        content = {
            "model": {"vocab_size": 512, "d_model": 8, "d_ff_fast": 16, "n_layers": 2,
                      "n_heads": 2, "n_kv_heads": 1, "chunk_size": 4, "sliding_window_size": 8},
            "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
            "training": {"stage": stage, "parameter_scope": scope, "micro_batch_size": 1,
                         "grad_accum_steps": 1, "max_steps": 2, "seq_len": 16, "save_interval": 2,
                         "gradient_checkpointing": False, "surprise_aux_loss_weight": 0.0},
        }
        config.write_text(yaml.safe_dump(content), encoding="utf-8")
        return config, data, content

    def test_training_persists_scope_counts_and_exact_optimizer_parameter_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data, _ = self.fixture(root)
            log = io.StringIO()
            with redirect_stdout(log):
                path = training.train(str(config), str(data), str(root / "meta.pt"))
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["metrics"]["parameter_scope"], "memory")
            model = uTTTModel(uTTTConfig.from_dict(saved["config"]))
            stats = training.apply_training_parameter_scope(model, "meta", "memory")
            self.assertEqual(saved["metrics"]["trainable_parameters"], stats["trainable"])
            self.assertEqual(saved["metrics"]["frozen_parameters"], stats["frozen"])
            groups = saved["optimizer_state"]["param_groups"]
            expected = [[p for p in model.get_slow_parameters() if p.requires_grad],
                        [p for p in model.get_meta_fast_parameters() if p.requires_grad]]
            self.assertEqual([len(group["params"]) for group in groups], list(map(len, expected)))
            self.assertEqual(len(saved["optimizer_state"]["state"]), sum(map(len, expected)))
            self.assertTrue(all(int(state["step"]) == 2 for state in saved["optimizer_state"]["state"].values()))
            self.assertIn("ParameterScope=memory", log.getvalue())
            self.assertIn(f"Trainable={stats['trainable']:,}", log.getvalue())

    def test_invalid_scope_and_memory_language_training_fail_before_model_creation(self):
        for scope, stage in (("other", "meta"), (None, "meta"), ("memory", "sft"), ("memory", "pretrain")):
            with self.subTest(scope=scope, stage=stage), tempfile.TemporaryDirectory() as tmp:
                config, data, _ = self.fixture(Path(tmp), scope, stage)
                with mock.patch.object(training, "configure_rtx5090_environment") as configure:
                    with self.assertRaisesRegex(ValueError, "parameter_scope"):
                        training.train(str(config), str(data), str(Path(tmp) / "out.pt"))
                    configure.assert_not_called()

    def test_resume_scope_mismatch_is_rejected_before_loading_model_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data, content = self.fixture(root)
            path = training.train(str(config), str(data), str(root / "meta.pt"))
            content["training"]["parameter_scope"] = "all"
            config.write_text(yaml.safe_dump(content), encoding="utf-8")
            before = path.read_bytes()
            with mock.patch.object(training.uTTTModel, "load_dual_base_state") as load:
                with self.assertRaisesRegex(ValueError, "Resume checkpoint.*parameter_scope"):
                    training.train(str(config), str(data), str(path), override_steps=3)
                load.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_legacy_missing_scope_means_all_and_cannot_be_reinterpreted_as_memory(self):
        training.validate_resume_parameter_scope({}, "all")
        training.validate_resume_parameter_scope({"parameter_scope": "memory"}, "memory")
        for saved in ({}, {"parameter_scope": "all"}, {"parameter_scope": None}):
            with self.subTest(saved=saved), self.assertRaisesRegex(ValueError, "parameter_scope.*--init-from"):
                training.validate_resume_parameter_scope(saved, "memory")
        model = self.model()
        training.apply_training_parameter_scope(model, "meta", "memory")
        counts = training.apply_training_parameter_scope(model, "meta", "all")
        self.assertEqual(counts["frozen"], 0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

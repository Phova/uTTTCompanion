"""Query CE must train every writer through independent, cache-free support turns."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import default_collate
import yaml

import train_5090 as training
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel


class EpisodeMetaTests(unittest.TestCase):
    def test_published_meta_configuration_passes_stage_and_selection_contracts(self):
        for name in ('5090_persona_meta_20261109.yaml','5090_persona_meta_paired_20261109.yaml'):
            with self.subTest(config=name):
                path=Path(__file__).resolve().parents[1]/'configs'/name
                raw=yaml.safe_load(path.read_text(encoding='utf-8'))
                settings=raw['training'];config=uTTTConfig.from_yaml(str(path))
                stage=settings['stage']
                scope=training.validate_parameter_scope(stage,settings['parameter_scope'])
                self.assertEqual(training.validate_meta_objective(stage,settings['meta_objective'],scope,config,settings),
                                 training.EPISODE_META_PROTOCOL)
                self.assertEqual(training.validate_validation_selection_metric(stage,settings['validation_selection_metric']),
                                 'validation_ce')

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def row(split="train", correction=False):
        row = {"write_text": "<|user|> My code is K7V.\n<|assistant|> Recorded.<|eos|>",
               "query_text": "What is my code?", "answer": "P2R" if correction else "K7V",
               "memory_task": {"task": "correction" if correction else "recall", "language": "en",
                               "split": split, "validation_group_id": "family-" + split}}
        if correction:
            row["correction_text"] = "<|user|> Replace my old code with P2R.\n<|assistant|> Updated.<|eos|>"
        return row

    def batch(self, correction=True):
        return default_collate([training.EpisodeMetaDataset.encode_row(SimpleUTF8Tokenizer(512), self.row(correction=correction), 128)])

    def model(self, checkpointed=False):
        torch.manual_seed(173)
        model = uTTTModel(uTTTConfig(vocab_size=512, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1,
                                    d_ff_fast=16, chunk_size=32, sliding_window_size=16, max_seq_len=128,
                                    gradient_checkpointing=checkpointed, meta_checkpoint_layers=1 if checkpointed else 0,
                                    meta_checkpoint_reentrant=False, max_inner_grad_norm=0.1, dropout=0.0)).double().train()
        training.apply_training_parameter_scope(model, "meta", "writer")
        return model

    def test_explicit_splits_whole_turn_encoding_and_fingerprint_bindings(self):
        tokenizer = SimpleUTF8Tokenizer(512)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episodes.jsonl"
            rows = [self.row("train"), self.row("dev", correction=True)]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            train = training.EpisodeMetaDataset(tokenizer, 128, path, "train")
            dev = training.EpisodeMetaDataset(tokenizer, 128, path, "dev")
            self.assertEqual((len(train), len(dev)), (1, 1))
            self.assertNotEqual(train.fingerprint, dev.fingerprint)
            self.assertNotEqual(train.fingerprint, training.EpisodeMetaDataset(tokenizer, 127, path, "train").fingerprint)
            item = train[0]
            labels = item["labels"][item["labels"] != -100].tolist()
            self.assertEqual(tokenizer.decode(labels, skip_special_tokens=False), " K7V<|eos|>")
            self.assertEqual(item["write_input_ids"].shape, (1, 128))
            rows[0]["memory_task"]["entity"] = "changed"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            self.assertNotEqual(train.fingerprint, training.EpisodeMetaDataset(tokenizer, 128, path, "train").fingerprint)
            rows[1]["memory_task"]["validation_group_id"] = "family-train"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple explicit splits"):
                training.EpisodeMetaDataset(tokenizer, 128, path, "train")
        with self.assertRaisesRegex(ValueError, "cropping is forbidden"):
            training.EpisodeMetaDataset.encode_row(tokenizer, self.row(), 8)

    def test_returned_state_is_opt_in_differentiable_and_query_has_no_support_cache(self):
        model, batch = self.model(), self.batch()
        support = batch["write_input_ids"][:, 0]
        mask = batch["write_attention_mask"][:, 0]
        legacy = model(support, attention_mask=mask, update_fast_weights=True, logits_to_keep=-1)
        self.assertFalse(legacy.fast_state.w_up.requires_grad)
        retained = model(support, attention_mask=mask, update_fast_weights=True, logits_to_keep=-1,
                         detach_returned_fast_state=False)
        self.assertTrue(retained.fast_state.w_up.requires_grad)
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            result = training.episode_forward(model, batch)
        self.assertEqual(forward.call_count, 3)
        query = forward.call_args
        self.assertFalse(query.kwargs["update_fast_weights"])
        self.assertNotIn("kv_caches", query.kwargs)
        self.assertIsNone(query.kwargs["fast_state"]._init_state)
        self.assertIsNone(query.kwargs["fast_state"]._chunk_states)
        self.assertTrue(torch.equal(query.args[0], batch["input_ids"]))
        self.assertIs(result.loss, result.ce_loss)

    def test_all_writer_gradients_and_second_derivative_survive_checkpointing(self):
        batch = self.batch()
        reference = self.model(False)
        checked = copy.deepcopy(reference)
        checked.gradient_checkpointing = True
        checked.config.meta_checkpoint_layers = 1
        gradients = []
        # Math SDPA supports the higher-order audit on CPU; the production
        # capacity probe independently measures its actual BF16 kernel path.
        with sdpa_kernel(SDPBackend.MATH):
            for model in (reference, checked):
                selected = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
                self.assertEqual(len(selected), 12)  # K/V weights + eta/alpha weight/bias, in two layers.
                loss = training.episode_forward(model, batch).loss
                first = torch.autograd.grad(loss, [value for _, value in selected], create_graph=True)
                for (name, _), gradient in zip(selected, first):
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                    self.assertGreater(float(gradient.detach().abs().sum()), 0, name)
                second = torch.autograd.grad(first[0].square().sum(), selected[0][1])[0]
                self.assertTrue(torch.isfinite(second).all())
                self.assertGreater(float(second.abs().sum()), 0)
                gradients.append([value.detach() for value in first])
        for expected, actual in zip(*gradients):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-7)

    def test_two_adam_steps_keep_every_nonwriter_and_unwritten_logits_bitwise_fixed(self):
        model, batch = self.model(True), self.batch()
        frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
        selected = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(selected, lr=1e-4)
        with torch.no_grad():
            before = training.episode_forward(model, batch, apply_writes=False).logits.clone()
        for step in range(1, 3):
            optimizer.zero_grad(set_to_none=True)
            training.episode_forward(model, batch).loss.backward()
            self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all()
                                and float(value.grad.abs().sum()) > 0 for value in selected))
            optimizer.step()
            self.assertTrue(all(int(state["step"]) == step for state in optimizer.state.values()))
            for name, value in model.named_parameters():
                if name in frozen:
                    self.assertTrue(torch.equal(value, frozen[name]), name)
                    self.assertIsNone(value.grad, name)
            with torch.no_grad():
                after = training.episode_forward(model, batch, apply_writes=False).logits
            self.assertTrue(torch.equal(before, after))

    def test_episode_config_refuses_reentrant_aux_loss_and_legacy_resume(self):
        config = self.model().config
        for settings in ({"surprise_aux_loss_weight": 0.1}, {"pack_sequences": True}, {"validation_fraction": 0.1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                training.validate_meta_objective("meta", training.EPISODE_META_PROTOCOL, "writer", config, settings)
        config.meta_checkpoint_reentrant = True
        with self.assertRaisesRegex(ValueError, "non-reentrant"):
            training.validate_meta_objective("meta", training.EPISODE_META_PROTOCOL, "writer", config, {})
        with self.assertRaisesRegex(ValueError, "different meta_objective"):
            training.validate_resume_meta_objective({}, training.EPISODE_META_PROTOCOL)
        training.validate_resume_meta_objective({}, "chunk_lm_v1")

    def test_real_training_records_episode_protocol_and_two_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "episodes.jsonl"
            data.write_text("\n".join(json.dumps(self.row(split)) for split in ("train", "dev")), encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({
                "model": {"vocab_size": 512, "d_model": 8, "d_ff_fast": 16, "n_layers": 2,
                          "n_heads": 2, "n_kv_heads": 1, "chunk_size": 32, "sliding_window_size": 16, "max_seq_len": 128},
                "hardware": {"device": "cpu", "precision": "float32", "torch_compile": False},
                "training": {"stage": "meta", "parameter_scope": "writer", "meta_objective": training.EPISODE_META_PROTOCOL,
                             "micro_batch_size": 1, "grad_accum_steps": 1, "max_steps": 2, "seq_len": 128,
                             "pack_sequences": False, "gradient_checkpointing": True, "meta_checkpoint_layers": 1,
                             "meta_checkpoint_reentrant": False, "save_interval": 2, "eval_interval": 1},
            }), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                output = training.train(str(config), str(data), str(root / "meta.pt"))
            saved = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(saved["step"], 2)
            self.assertEqual(saved["metrics"]["parameter_scope"], "writer")
            self.assertEqual(saved["metrics"]["meta_objective"], training.EPISODE_META_PROTOCOL)
            self.assertFalse(saved["metrics"]["online_memory_quality_verified"])
            self.assertEqual(len(saved["optimizer_state"]["state"]), 12)
            self.assertTrue(all(int(value["step"]) == 2 for value in saved["optimizer_state"]["state"].values()))


if __name__ == "__main__":
    unittest.main()

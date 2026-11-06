"""Heldout isolation, paired state independence and final-answer-only scoring."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deploy import evaluate_memory as evaluator
from deploy.prepare_memory_curriculum import PROTOCOL, SPECIAL_TOKENS, measure

try:
    import torch
    from model.uttt_model import uTTTConfig, uTTTModel
except ImportError:
    torch = None

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None


def write_curriculum(root, train, heldout, tokenizer_hash="tokenizer", *, chunk=4, window=4, seq=256):
    for name, rows in (("train", train), ("heldout", heldout)):
        (root / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    manifest = {"protocol": PROTOCOL, "tokenizer_sha256": tokenizer_hash, "chunk_size": chunk,
                "sliding_window": window, "seq_len": seq,
                "identifier_intersections": {"entities": 0, "payloads": 0},
                "files": {name: {"file": f"{name}.jsonl", "rows": len(rows),
                                 "sha256": evaluator.file_hash(root / f"{name}.jsonl")}
                          for name, rows in (("train", train), ("heldout", heldout))}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class MemoryEvaluationGuardsTests(unittest.TestCase):
    def rows(self):
        def row(entity, answer):
            return {"text": f"facts {entity} {answer}", "memory_task": {
                "task": "recall", "language": "en", "entity": entity, "answer": answer}}
        return [row("train-person", "TRAIN-CODE")], [row("heldout-person", "HELDOUT-CODE")]

    def test_heldout_hash_and_actual_identifier_overlap_are_both_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, heldout = self.rows()
            write_curriculum(root, train, heldout)
            args = {"chunk_size": 4, "sliding_window": 4, "context_limit": 256}
            rows, _, evidence = evaluator.load_heldout(root / "heldout.jsonl", "tokenizer", **args)
            self.assertEqual(rows, heldout)
            self.assertEqual(evidence["verified_identifier_intersections"], {"entities": 0, "payloads": 0})
            (root / "heldout.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file hash"):
                evaluator.load_heldout(root / "heldout.jsonl", "tokenizer", **args)
            heldout[0]["memory_task"]["answer"] = train[0]["memory_task"]["answer"]
            # A manifest claiming zero overlap cannot hide an actual payload leak.
            write_curriculum(root, train, heldout)
            with self.assertRaisesRegex(ValueError, "identifiers overlap"):
                evaluator.load_heldout(root / "heldout.jsonl", "tokenizer", **args)

    def test_training_file_tokenizer_and_geometry_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_curriculum(root, *self.rows())
            for path, tokenizer_hash, window in (("train.jsonl", "tokenizer", 4),
                                                  ("heldout.jsonl", "different", 4),
                                                  ("heldout.jsonl", "tokenizer", 8)):
                with self.subTest(path=path, tokenizer_hash=tokenizer_hash, window=window), self.assertRaises(ValueError):
                    evaluator.load_heldout(root / path, tokenizer_hash, chunk_size=4,
                                           sliding_window=window, context_limit=256)

    def test_sampling_preserves_task_language_groups(self):
        rows = [{"memory_task": {"task": task, "language": language}}
                for _ in range(16) for task in ("recall", "correction") for language in ("en", "zh")]
        selected = evaluator.sample_indices(rows, 8)
        groups = [(rows[index]["memory_task"]["task"], rows[index]["memory_task"]["language"]) for index in selected]
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(set(groups), {("recall", "en"), ("recall", "zh"), ("correction", "en"), ("correction", "zh")})
        self.assertEqual(selected, evaluator.sample_indices(rows, 8))
        with self.assertRaises(ValueError):
            evaluator.sample_indices(rows, 0)

    def test_gpu_guard_uses_global_free_and_caps_allocator(self):
        calls = []
        cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                               get_device_properties=lambda index: SimpleNamespace(total_memory=32 * evaluator.GIB, uuid="unused"),
                               mem_get_info=lambda index: (31 * evaluator.GIB, 32 * evaluator.GIB),
                               set_per_process_memory_fraction=lambda fraction, **kwargs: calls.append(fraction),
                               reset_peak_memory_stats=lambda index: None)
        memory = {"free_bytes": evaluator.GIB, "global_free_bytes": evaluator.GIB,
                  "runtime_free_bytes": 31 * evaluator.GIB, "device_uuid": "unused"}
        with patch("deploy.gpu_memory.query_available_gpu_memory", return_value=memory):
            with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                evaluator.guard_gpu(SimpleNamespace(cuda=cuda), SimpleNamespace(index=0), evaluator.GIB)
        self.assertFalse(calls)
        with patch("deploy.gpu_memory.query_available_gpu_memory", return_value={**memory, "free_bytes": 31 * evaluator.GIB}):
            evidence = evaluator.guard_gpu(SimpleNamespace(cuda=cuda), SimpleNamespace(index=0), evaluator.GIB)
        self.assertEqual(calls, [28 / 32])
        self.assertNotIn("device_uuid", evidence)


@unittest.skipIf(torch is None or Tokenizer is None, "Torch and actual BPE dependencies are required")
class MemoryEvaluationCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fixture(self, root, *, stage="meta", protocol=None):
        backend = Tokenizer(models.BPE(unk_token="<|unk|>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(["Small independent memory evaluation."], trainer=trainers.BpeTrainer(
            vocab_size=262, special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        config = uTTTConfig(vocab_size=262, d_model=8, d_ff_fast=12, n_layers=2,
                            n_heads=2, n_kv_heads=1, sliding_window_size=4, chunk_size=4,
                            max_seq_len=256, tokenizer_json=backend.to_str(), dropout=0)
        torch.manual_seed(728)
        model = uTTTModel(config)
        def row(entity, answer):
            text = f"<|user|> Code for {entity} is {answer}.\n\nA calm reader studies a small garden.\n\nWhich code?\n<|assistant|> {answer}<|eos|>"
            geometry = measure(backend, text, text.index("\n\n"), 4)
            return {"text": text, "memory_task": {"task": "recall", "language": "en", "entity": entity,
                                                  "answer": answer, "obsolete_answer": None, **geometry}}
        train, heldout = [row("person-A", "KEY-TRAIN")], [row("person-B", "KEY-HELDOUT"), row("person-C", "KEY-UNSEEN")]
        write_curriculum(root, train, heldout, hashlib.sha256(config.tokenizer_json.encode()).hexdigest())
        checkpoint = root / "model.pt"
        torch.save({"config": config.to_dict(), "slow_weights": model.slow_state_dict(),
                    "init_fast_weights": model.init_fast_state_dict(), "step": 9,
                    "metrics": {"training_stage": stage,
                                "inner_loop_protocol": protocol or model.INNER_LOOP_PROTOCOL}}, checkpoint)
        return model, config, heldout, checkpoint

    def test_pair_uses_four_independent_fresh_states_and_changes_no_parameters(self):
        from model.tokenizer import build_tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            model, config, rows, _ = self.fixture(Path(tmp))
            tokenizer = build_tokenizer(config)
            ids, labels, _ = evaluator.prepare_document(rows[0], tokenizer, config)
            assistant = ids.index(tokenizer.ASSISTANT_ID)
            self.assertTrue(all(value == -100 for value in labels[:assistant + 1]))
            self.assertEqual(labels[-2:], [tokenizer.EOS_ID, -100])
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            states = []
            original = model.init_fast_state
            def fresh(**settings):
                state = original(**settings)
                self.assertEqual(state.step, 0)
                self.assertEqual(int(torch.count_nonzero(state.s_up)), 0)
                torch.testing.assert_close(state.w_up[0], model.fast_weight_pool.w_up_init)
                states.append(state)
                return state
            with patch.object(model, "init_fast_state", side_effect=fresh):
                first = evaluator.evaluate_pair(model, ids, labels, device=torch.device("cpu"), activation_dtype=torch.float32)
                second = evaluator.evaluate_pair(model, ids, labels, device=torch.device("cpu"), activation_dtype=torch.float32)
            self.assertEqual(first, second)
            self.assertEqual(len({state.w_up.data_ptr() for state in states}), 4)
            self.assertTrue(model.training)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            with torch.no_grad():
                output = model(torch.tensor([ids]), fast_state=original(dtype=torch.float32, detach=True),
                               labels=torch.tensor([labels]), update_fast_weights=False, surprise_aux_weight=0,
                               return_fast_state=False)
            self.assertAlmostEqual(first["frozen"]["answer_ce"], float(output.ce_loss), places=6)

    def test_real_cpu_report_omits_raw_answers_and_is_teacher_forced_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, rows, checkpoint = self.fixture(root)
            before = evaluator.file_hash(checkpoint)
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                report = evaluator.evaluate(checkpoint, root / "heldout.jsonl", root / "result.json", 2, "cpu")
            self.assertEqual(report["summary"]["documents"], 2)
            self.assertEqual(report["checkpoint_step"], 9)
            self.assertFalse(report["free_generation_accuracy_measured"])
            self.assertFalse(report["online_writes_allowed"])
            self.assertTrue(report["meta_trained"])
            self.assertEqual(evaluator.file_hash(checkpoint), before)
            serialized = json.dumps(report) + console.getvalue()
            for row in rows:
                self.assertNotIn(row["memory_task"]["answer"], serialized)
                self.assertNotIn(row["memory_task"]["entity"], serialized)
            with self.assertRaises(FileExistsError):
                evaluator.evaluate(checkpoint, root / "heldout.jsonl", root / "result.json", 2, "cpu")

    def test_protocol_is_strict_and_untrained_baseline_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, checkpoint = self.fixture(root, protocol="old-protocol")
            with self.assertRaisesRegex(ValueError, "inner-loop protocol"):
                evaluator.evaluate(checkpoint, root / "heldout.jsonl", root / "report.json", 1, "cpu",
                                   allow_untrained_baseline=True)
            _, _, _, checkpoint = self.fixture(root, stage="sft")
            with self.assertRaisesRegex(ValueError, "meta checkpoint"):
                evaluator.evaluate(checkpoint, root / "heldout.jsonl", root / "report.json", 1, "cpu")
            with contextlib.redirect_stdout(io.StringIO()):
                report = evaluator.evaluate(checkpoint, root / "heldout.jsonl", root / "report.json", 1, "cpu",
                                            allow_untrained_baseline=True)
            self.assertTrue(report["untrained_baseline"])
            self.assertFalse(report["online_writes_allowed"])


if __name__ == "__main__":
    unittest.main()

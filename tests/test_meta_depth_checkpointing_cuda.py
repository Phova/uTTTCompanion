"""Bounded CUDA BF16 parity for exact reentrant depth/time meta recomputation.

Run in a separate test process. This builds only tiny random models and never
loads checkpoints. Parameters, parameter gradients and dynamic W/S remain FP32;
only activations use the same CUDA BF16 autocast policy as real training.
"""
import gc
import json
import math
import os
import unittest
from unittest.mock import patch

try:
    import torch
    from deploy.gpu_memory import GpuMemoryError, query_available_gpu_memory
    from deploy.probe_selective_meta import scoped_parameters
    from model.uttt_model import uTTTConfig, uTTTModel
except ImportError:
    torch = None


MIB = 1024 ** 2
ALLOCATOR_LIMIT = 512 * MIB
MIN_FREE_BYTES = 1024 * MIB
STATE_NAMES = ("w_up", "w_gate", "w_down", "s_up", "s_gate", "s_down", "ln_weight", "ln_bias")
WRITER_MARKERS = (".w_k.", ".w_v.", ".w_eta.", ".w_alpha.")
# Checkpoint recomputation can change where BF16 branch gradients accumulate
# before casting into FP32 parameters. Bound each element by ONE BF16 epsilon
# at that parameter's peak reference-gradient scale, with NO extra relative
# allowance. Additionally cap EACH parameter's relative gradient L2 error at
# 1%, preventing the peak-scale allowance from hiding smaller writer gradients.
BF16_EPS = 2 ** -7
GRAD_RELATIVE_L2 = 0.01
FP32_GRAD_ATOL, FP32_GRAD_RTOL, FP32_GRAD_RELATIVE_L2 = 2e-5, 3e-4, 1e-5
FP32_NORM_NOISE_PER_ELEMENT = 1e-9


@unittest.skipUnless(torch is not None and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                     "CUDA BF16 test dependencies unavailable")
class MetaDepthCheckpointingCUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(cls.device)
        runtime_free, _ = torch.cuda.mem_get_info(cls.device)
        try:
            available = query_available_gpu_memory(
                runtime_free, device_uuid=getattr(properties, "uuid", None), device_index=cls.device,
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        except GpuMemoryError as error:
            raise unittest.SkipTest(f"Independent global GPU headroom check unavailable: {error}") from error
        if available["free_bytes"] < MIN_FREE_BYTES:
            raise unittest.SkipTest("Tiny CUDA parity test requires 1 GiB free in both runtime and global measurements")
        if torch.cuda.memory_allocated(cls.device) > ALLOCATOR_LIMIT // 4:
            raise unittest.SkipTest("Run CUDA parity in an isolated process; existing allocations exceed 128 MiB")
        getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        if getter is None:
            raise unittest.SkipTest("CUDA allocator fraction getter required to restore the previous limit")
        cls.previous_fraction = float(getter(cls.device))
        fraction = min(cls.previous_fraction, ALLOCATOR_LIMIT / int(properties.total_memory))
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(fraction, cls.device)
        cls.addClassCleanup(torch.cuda.set_per_process_memory_fraction, cls.previous_fraction, cls.device)
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", cls.previous_tf32)
        cls.addClassCleanup(torch.cuda.empty_cache)
        torch.cuda.reset_peak_memory_stats(cls.device)

    def config(self, checkpointed):
        return uTTTConfig(
            vocab_size=96, d_model=32, d_ff_fast=48, n_layers=5,
            n_heads=4, n_kv_heads=2, chunk_size=4, sliding_window_size=6,
            max_seq_len=32, max_inner_grad_norm=0.01, dropout=0.0,
            gradient_checkpointing=checkpointed,
            meta_checkpoint_layers=2 if checkpointed else 0,
            meta_checkpoint_time_chunks=True, meta_checkpoint_reentrant=checkpointed,
        )

    def capture(self, initial, ids, mask, *, checkpointed, scope, explicit_state=True, activation_dtype=None):
        activation_dtype = torch.bfloat16 if activation_dtype is None else activation_dtype
        model = uTTTModel(self.config(checkpointed)).float()
        model.load_state_dict(initial, strict=True)
        model.to(device=torch.device("cuda", self.device)).train()
        scoped_parameters(model, scope)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
        labels = ids.masked_fill(mask == 0, -100)
        # Pin the tested protocol explicitly: BF16 activations read FP32 master
        # states. Do not let automatic state initialization silently choose the
        # activation dtype and turn this into a lower-precision comparison.
        fast_state = (model.init_fast_state(batch_size=ids.shape[0], device=ids.device,
                                            dtype=torch.float32, detach=False) if explicit_state else None)
        first_read = []
        original_step = model.fast_weight_pool.chunk_step

        def inspect_first_read(*args, **kwargs):
            if not first_read:
                read = kwargs["read_state"]
                values = {name: getattr(read, name).detach().cpu().clone() for name in STATE_NAMES}
                for name in STATE_NAMES:
                    self.assertEqual(values[name].dtype, torch.float32, "first read " + name)
                for name, parameter_name in (("w_up", "w_up_init"), ("w_gate", "w_gate_init"),
                                              ("w_down", "w_down_init")):
                    expected = getattr(model.fast_weight_pool, parameter_name).detach().cpu()
                    expected = expected.unsqueeze(0).expand(ids.shape[0], -1, -1)
                    # Exact comparison catches the BF16 -> FP32 round trip that
                    # a final-state dtype assertion alone would miss.
                    torch.testing.assert_close(values[name], expected, atol=0, rtol=0, msg="initial " + name)
                for name in ("ln_weight", "ln_bias"):
                    expected = getattr(model.fast_weight_pool, name).detach().cpu()
                    torch.testing.assert_close(values[name], expected, atol=0, rtol=0, msg="initial " + name)
                for name in ("s_up", "s_gate", "s_down"):
                    torch.testing.assert_close(values[name], torch.zeros_like(values[name]), atol=0, rtol=0, msg=name)
                self.assertFalse(torch.equal(values["w_up"], values["w_up"].bfloat16().float()),
                                 "Fixture must distinguish FP32 master values from rounded BF16 values")
                first_read.append(values)
            return original_step(*args, **kwargs)

        with patch.object(model.fast_weight_pool, "chunk_step", side_effect=inspect_first_read):
            with torch.autocast("cuda", dtype=activation_dtype, enabled=activation_dtype != torch.float32):
                output = model(ids, labels=labels, attention_mask=mask, fast_state=fast_state,
                               update_fast_weights=True, surprise_aux_weight=0.0,
                               return_fast_state=True)
        self.assertEqual(output.logits.dtype, activation_dtype)
        self.assertEqual(output.ce_loss.dtype, torch.float32)
        for name in STATE_NAMES:
            self.assertEqual(getattr(output.fast_state, name).dtype, torch.float32, name)
        self.assertEqual(output.fast_state.step, 15)  # Three time chunks, five writers each.
        self.assertEqual(len(first_read), 1)
        output.loss.backward()
        torch.cuda.synchronize(self.device)
        gradients, frozen, writer_norms = {}, [], {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                frozen.append(name)
                self.assertIsNone(parameter.grad, name)
            if parameter.requires_grad and any(marker in name for marker in WRITER_MARKERS):
                self.assertIsNotNone(parameter.grad, name)
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.dtype, torch.float32, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                gradient = parameter.grad.detach().cpu().clone()
                gradients[name] = gradient
                if any(marker in name for marker in WRITER_MARKERS):
                    writer_norms[name] = float(gradient.norm())
            else:
                gradients[name] = None
        self.assertTrue(writer_norms)
        # A particular eta/alpha bias may legitimately have zero derivative in
        # a tiny clipped/BF16 fixture. Every writer is still compared, including
        # its None status and exact zero/reference behavior below. Require the
        # K/V writer matrices that establish the actual CE-through-write path.
        essential = {name: norm for name, norm in writer_norms.items()
                     if name.endswith((".w_k.weight", ".w_v.weight"))}
        self.assertEqual(len(essential), 2 * model.config.n_layers)
        self.assertTrue(all(norm > 0 for norm in essential.values()), essential)
        if scope == "memory":
            self.assertIn("tok_embeddings.weight", frozen)
            self.assertTrue(any(".attn." in name for name in frozen))
        result = {
            "logits": output.logits.detach().float().cpu().clone(),
            "ce": output.ce_loss.detach().cpu().clone(),
            "states": {name: getattr(output.fast_state, name).detach().cpu().clone() for name in STATE_NAMES},
            "gradients": gradients, "writer_gradients": len(writer_norms),
            "writer_nonzero_gradients": sum(norm > 0 for norm in writer_norms.values()),
            "zero_writer_names": [name for name, norm in writer_norms.items() if norm == 0],
            "frozen_parameters": len(frozen),
        }
        del output, model, labels, fast_state
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def compare_gradients(self, expected, actual, *, allow_frozen=False, precision="bfloat16"):
        self.assertEqual(expected.keys(), actual.keys())
        rows = []
        for name, left in expected.items():
            right = actual[name]
            if allow_frozen and right is None:
                continue
            self.assertEqual(left is None, right is None, name)
            if left is None:
                continue
            difference = (right - left).abs()
            norm = float(left.norm())
            scale = float(left.abs().max())
            max_abs = float(difference.max())
            difference_norm = float(difference.norm())
            relative_l2 = difference_norm / max(norm, 1e-30)
            max_relative = float((difference / left.abs().clamp_min(1e-8)).max())
            absolute_budget = BF16_EPS * scale if precision == "bfloat16" else FP32_GRAD_ATOL
            relative_budget = 0.0 if precision == "bfloat16" else FP32_GRAD_RTOL
            l2_budget = GRAD_RELATIVE_L2 if precision == "bfloat16" else FP32_GRAD_RELATIVE_L2
            absolute_norm_budget = 0.0 if precision == "bfloat16" else FP32_NORM_NOISE_PER_ELEMENT * math.sqrt(left.numel())
            norm_budget = l2_budget * norm + absolute_norm_budget
            # Relative errors are unstable for cancellation-near-zero bias
            # gradients. Report them only when the relative norm term, rather
            # than the absolute FP32 noise floor, controls the comparison.
            relative_report = relative_l2 if norm > 0 and l2_budget * norm >= absolute_norm_budget else None
            rows.append({"name": name, "max_abs": max_abs,
                         "max_rel": max_relative if relative_report is not None else None,
                         "relative_l2": relative_report, "reference_norm": norm,
                         "relative_metrics_suppressed_near_zero": relative_report is None,
                         "gradient_diff_norm": difference_norm, "norm_error_budget": norm_budget,
                         "absolute_norm_noise_budget": absolute_norm_budget,
                         "reference_linf": scale, "absolute_budget": absolute_budget,
                         "max_abs_in_bf16_peak_ulps": max_abs / max(BF16_EPS * scale, 1e-30)})
            details = json.dumps(rows[-1])
            torch.testing.assert_close(right, left, atol=absolute_budget, rtol=relative_budget, msg=details)
            if precision == "float32":
                self.assertLessEqual(difference_norm, norm_budget, details)
            elif norm > 1e-20:
                self.assertLessEqual(relative_l2, l2_budget, details)
            else:
                self.assertEqual(max_abs, 0.0, details)
        self.assertTrue(rows)
        return {
            "compared_gradients": len(rows),
            "worst_absolute": max(rows, key=lambda row: row["max_abs"]),
            "worst_relative": max(rows, key=lambda row: row["max_rel"] if row["max_rel"] is not None else -1),
            "worst_relative_l2": max(rows, key=lambda row: row["relative_l2"] if row["relative_l2"] is not None else -1),
            "worst_norm_error_budget_fraction": max(rows, key=lambda row: row["gradient_diff_norm"] / max(row["norm_error_budget"], 1e-30)),
        }

    def test_bf16_reentrant_time_segments_match_full_and_memory_scope_meta_gradients(self):
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(8723)
            initial = {name: value.detach().cpu().clone()
                       for name, value in uTTTModel(self.config(False)).float().state_dict().items()}
            for batch in (1, 2):
                generator = torch.Generator(device="cpu").manual_seed(391 + batch)
                ids = torch.randint(6, 96, (batch, 11), generator=generator).to(torch.device("cuda", self.device))
                mask = torch.ones_like(ids)
                mask[-1, -2:] = 0
                if batch == 2:
                    mask[0, -1] = 0
                full_reference = None
                for scope in ("all", "memory"):
                    with self.subTest(batch=batch, scope=scope):
                        reference = self.capture(initial, ids, mask, checkpointed=False, scope=scope)
                        actual = self.capture(initial, ids, mask, checkpointed=True, scope=scope)
                        torch.testing.assert_close(actual["ce"], reference["ce"], atol=2e-6, rtol=2e-6)
                        torch.testing.assert_close(actual["logits"], reference["logits"], atol=1e-4, rtol=0.008)
                        for name in STATE_NAMES:
                            torch.testing.assert_close(actual["states"][name], reference["states"][name],
                                                       atol=2e-6, rtol=2e-5, msg=name)
                        diagnostics = self.compare_gradients(reference["gradients"], actual["gradients"])
                        frozen_flow = None
                        if scope == "all":
                            full_reference = reference
                        else:
                            # Freezing attention/embeddings must not suppress the
                            # gradient flowing THROUGH them into earlier memory
                            # writers. Compare surviving gradients to the full run.
                            frozen_flow = self.compare_gradients(full_reference["gradients"], reference["gradients"],
                                                                 allow_frozen=True)
                        print("CUDA_META_CHECKPOINT_PARITY " + json.dumps({
                            "batch": batch, "scope": scope, "time_chunks": 3, "depth_segment_layers": 2,
                            "initial_dynamic_state_dtype": "float32", "activation_dtype": "bfloat16",
                            "ce": float(actual["ce"]),
                            "ce_abs_error": float((actual["ce"] - reference["ce"]).abs()),
                            "logits_max_abs_error": float((actual["logits"] - reference["logits"]).abs().max()),
                            "writer_parameter_gradients": actual["writer_gradients"],
                            "writer_nonzero_gradients": actual["writer_nonzero_gradients"],
                            "zero_writer_names": actual["zero_writer_names"],
                            "frozen_parameters": actual["frozen_parameters"],
                            "gradient_tolerances": {"atol": "BF16_EPS * parameter_reference_gradient_Linf",
                                                    "bf16_eps": BF16_EPS, "rtol": 0.0,
                                                    "max_parameter_relative_l2": GRAD_RELATIVE_L2},
                            "gradients": diagnostics, "gradient_flow_through_frozen_layers": frozen_flow,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                        }), flush=True)
                        del actual, reference
                del full_reference, ids, mask
        self.assertLessEqual(torch.cuda.max_memory_reserved(self.device), ALLOCATOR_LIMIT)

    def test_implicit_write_initialization_preserves_fp32_masters_before_first_read(self):
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(7401)
            initial = {name: value.detach().cpu().clone()
                       for name, value in uTTTModel(self.config(False)).float().state_dict().items()}
            ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]],
                               device=torch.device("cuda", self.device))
            mask = torch.ones_like(ids)
            mask[-1, -2:] = 0
            explicit = self.capture(initial, ids, mask, checkpointed=True, scope="memory", explicit_state=True)
            implicit = self.capture(initial, ids, mask, checkpointed=True, scope="memory", explicit_state=False)
            torch.testing.assert_close(implicit["ce"], explicit["ce"], atol=0, rtol=0)
            torch.testing.assert_close(implicit["logits"], explicit["logits"], atol=0, rtol=0)
            for name in STATE_NAMES:
                torch.testing.assert_close(implicit["states"][name], explicit["states"][name], atol=0, rtol=0, msg=name)
            diagnostics = self.compare_gradients(explicit["gradients"], implicit["gradients"])
            print("CUDA_META_INITIAL_STATE_PARITY " + json.dumps({
                "batch": 2, "scope": "memory", "implicit_vs_explicit_fp32": True,
                "first_read_master_values_exact": True, "first_read_eight_tensors_float32": True,
                "ce": float(implicit["ce"]), "gradients": diagnostics,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }), flush=True)
        self.assertLessEqual(torch.cuda.max_memory_reserved(self.device), ALLOCATOR_LIMIT)

    def test_cuda_fp32_recomputation_matches_with_strict_gradient_l2(self):
        """Separate structural gradient errors from BF16 accumulation rounding."""
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(8723)
            initial = {name: value.detach().cpu().clone()
                       for name, value in uTTTModel(self.config(False)).float().state_dict().items()}
            for batch in (1, 2):
                generator = torch.Generator(device="cpu").manual_seed(391 + batch)
                ids = torch.randint(6, 96, (batch, 11), generator=generator).to(torch.device("cuda", self.device))
                mask = torch.ones_like(ids)
                mask[-1, -2:] = 0
                if batch == 2:
                    mask[0, -1] = 0
                for scope in ("all", "memory"):
                    with self.subTest(batch=batch, scope=scope):
                        reference = self.capture(initial, ids, mask, checkpointed=False, scope=scope,
                                                 activation_dtype=torch.float32)
                        actual = self.capture(initial, ids, mask, checkpointed=True, scope=scope,
                                              activation_dtype=torch.float32)
                        torch.testing.assert_close(actual["ce"], reference["ce"], atol=2e-7, rtol=2e-7)
                        torch.testing.assert_close(actual["logits"], reference["logits"], atol=1e-6, rtol=1e-5)
                        for name in STATE_NAMES:
                            torch.testing.assert_close(actual["states"][name], reference["states"][name],
                                                       atol=1e-7, rtol=1e-6, msg=name)
                        diagnostics = self.compare_gradients(reference["gradients"], actual["gradients"], precision="float32")
                        print("CUDA_FP32_META_CHECKPOINT_PARITY " + json.dumps({
                            "batch": batch, "scope": scope, "time_chunks": 3, "depth_segment_layers": 2,
                            "activation_dtype": "float32", "tf32_enabled": False,
                            "ce": float(actual["ce"]),
                            "ce_abs_error": float((actual["ce"] - reference["ce"]).abs()),
                            "logits_max_abs_error": float((actual["logits"] - reference["logits"]).abs().max()),
                            "gradient_tolerances": {"atol": FP32_GRAD_ATOL, "rtol": FP32_GRAD_RTOL,
                                                    "gradient_norm_bound": "1e-5 * reference_norm + 1e-9 * sqrt(numel)",
                                                    "relative_norm_factor": FP32_GRAD_RELATIVE_L2,
                                                    "absolute_norm_noise_per_element": FP32_NORM_NOISE_PER_ELEMENT},
                            "gradients": diagnostics,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                        }), flush=True)
                        del actual, reference
                del ids, mask
        self.assertLessEqual(torch.cuda.max_memory_reserved(self.device), ALLOCATOR_LIMIT)


if __name__ == "__main__":
    unittest.main()

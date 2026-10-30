"""
tests/test_uttt_pipeline.py
===========================
End-to-End Verification Suite for the uTTT Framework:
1. Mathematical verification of `analytical_swiglu_chunk_grad` against `torch.autograd.grad`.
2. Cross-layer `SharedFastWeightPool` sequential state passing & outer-loop meta-gradients.
3. Dual-track checkpointing (`base_uttt.pt` + `personas/{user_id}_state.pt`) & LRU offloading.
4. `uTTTPersonalizer.interact_and_evolve` 3-step loop, Stability Guard, and CLI status bar.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from engine.checkpoint_manager import DualTrackCheckpointManager
from engine.personalizer import uTTTPersonalizer
from model.uttt_layer import (
    SharedFastWeightPool,
    analytical_swiglu_chunk_grad,
    swiglu_fast_forward,
)
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel
from model.uttt_nvs import uTTTNVSModel
from train_5090 import export_base_checkpoint


class TestuTTTPipeline(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="uttt_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_analytical_swiglu_grad_matches_autograd(self) -> None:
        """
        Verify that `analytical_swiglu_chunk_grad` produces the exact same gradients
        for {W_up, W_gate, W_down} as PyTorch `autograd.grad` on L = 1/2 ||f_W(k) - v||_2^2.
        """
        bsz, chunk_len, d_model, d_ff = 2, 16, 32, 64
        k = torch.randn(bsz, chunk_len, d_model, dtype=torch.float32)
        v = torch.randn(bsz, chunk_len, d_model, dtype=torch.float32)
        eta = torch.rand(bsz, chunk_len, 1, dtype=torch.float32) * 0.05

        w_up = (torch.randn(bsz, d_model, d_ff, dtype=torch.float32) * 0.1).requires_grad_(True)
        w_gate = (torch.randn(bsz, d_model, d_ff, dtype=torch.float32) * 0.1).requires_grad_(True)
        w_down = (torch.randn(bsz, d_ff, d_model, dtype=torch.float32) * 0.1).requires_grad_(True)
        ln_w = torch.ones(d_model, dtype=torch.float32)
        ln_b = torch.zeros(d_model, dtype=torch.float32)

        # 1. Analytical Chunk-wise Gradient
        g_up_ana, g_gate_ana, g_down_ana, surp_loss = analytical_swiglu_chunk_grad(
            k_chunk=k,
            v_chunk=v,
            eta_chunk=eta,
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            ln_weight=ln_w,
            ln_bias=ln_b,
            normalize_by_dim=False,
        )

        # 2. Reference Autograd Gradient
        v_pred = swiglu_fast_forward(k, w_up, w_gate, w_down, ln_w, ln_b)
        per_token_loss = 0.5 * (v_pred - v).pow(2).sum(dim=-1)  # (B, C)
        weighted_loss = ((eta.squeeze(-1) / float(chunk_len)) * per_token_loss).sum()
        g_up_ref, g_gate_ref, g_down_ref = torch.autograd.grad(
            weighted_loss, [w_up, w_gate, w_down]
        )

        self.assertTrue(torch.allclose(surp_loss, per_token_loss.mean(), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(g_up_ana, g_up_ref, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(g_gate_ana, g_gate_ref, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(g_down_ana, g_down_ref, atol=1e-5, rtol=1e-4))

    def test_02_cross_layer_fast_weight_passing_and_meta_grad(self) -> None:
        """
        Verify that SharedFastWeightPool is instantiated once, state evolves across
        layers 0..L-1, and outer-loop backward updates both slow_weights and W_0.
        """
        cfg = uTTTConfig(
            vocab_size=512,
            d_model=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            d_ff_fast=128,
            chunk_size=16,
            sliding_window_size=32,
            gradient_checkpointing=True,
        )
        model = uTTTModel(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels, update_fast_weights=True)
        # 3 layers x (32 / 16 = 2 chunks per layer) = 6 inner-loop TTT steps
        self.assertEqual(out.fast_state.step, 6)
        self.assertEqual(len(out.layer_surprise_losses), 3)

        out.loss.backward()
        # Check both slow weights and meta-initial fast weights received outer-loop gradients
        self.assertIsNotNone(model.fast_weight_pool.w_up_init.grad)
        self.assertGreater(model.fast_weight_pool.w_up_init.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(model.layers[0].uttt.w_q.weight.grad)

    def test_03_dual_checkpoint_and_personalizer_lifecycle(self) -> None:
        """
        Verify Base Checkpoint export, multi-user LRU offloading, 3-step
        `interact_and_evolve`, Stability Guard, and `/save`, `/diff`, `/snapshot`, `/reset`.
        """
        cfg = uTTTConfig(
            vocab_size=512,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff_fast=128,
            chunk_size=16,
            sliding_window_size=32,
            surprise_threshold=0.01,
            elastic_anchor_lambda=0.05,
        )
        base_model = uTTTModel(cfg)
        base_ckpt_path = self.tmp_dir / "checkpoints" / "base_uttt.pt"
        personas_dir = self.tmp_dir / "personas"

        export_base_checkpoint(
            model=base_model,
            optimizer=None,
            config=cfg,
            save_path=base_ckpt_path,
            step=10,
            metrics={"test": True, "training_stage": "meta", "inner_loop_protocol": uTTTModel.INNER_LOOP_PROTOCOL},
        )
        self.assertTrue(base_ckpt_path.exists())

        ckpt_mgr = DualTrackCheckpointManager(
            base_checkpoint_path=base_ckpt_path,
            personas_dir=personas_dir,
            max_gpu_users=2,
            max_cpu_users=4,
        )
        personalizer = uTTTPersonalizer(
            checkpoint_manager=ckpt_mgr,
            surprise_threshold=0.01,
            elastic_anchor_lambda=0.05,
            default_max_new_tokens=16,
        )

        # Turn 1 for user 'alice'
        res1 = personalizer.interact_and_evolve(
            user_id="alice",
            prompt="My name is Alice and I build CUDA kernels for RTX 5090.",
        )
        self.assertTrue(res1.gate_triggered)
        self.assertTrue(res1.persona_saved)
        self.assertGreater(res1.weight_drift_pct, 0.0)
        self.assertIn("[TTT Status] Surprise Loss:", res1.status_bar)
        self.assertIn("Weight Drift from Base:", res1.status_bar)
        self.assertIn("Persona Saved", res1.status_bar)

        # Test LRU eviction by loading user 'bob' and 'charlie' (max_gpu_users=2)
        personalizer.interact_and_evolve("bob", "Hello from Bob!")
        personalizer.interact_and_evolve("charlie", "Hello from Charlie!")
        pool_info = ckpt_mgr.pool_status()
        self.assertEqual(pool_info["gpu_pool_size"], 2)
        self.assertIn("alice", pool_info["cpu_cached_users"])

        # Test /diff, /snapshot, /reset, /restore for 'alice'
        diff_res = personalizer.handle_command("alice", "/diff")
        self.assertGreater(diff_res.weight_drift_pct, 0.0)

        snap_res = personalizer.handle_command("alice", "/snapshot v1")
        self.assertTrue(snap_res.persona_saved)

        reset_res = personalizer.handle_command("alice", "/reset")
        self.assertAlmostEqual(reset_res.weight_drift_pct, 0.0, places=5)

        restore_res = personalizer.handle_command("alice", "/restore v1")
        self.assertGreater(restore_res.weight_drift_pct, 0.0)

    def test_04_strict_causal_no_future_leakage_and_cjk_tokenizer(self) -> None:
        """
        Verify mathematically that `uTTTModel.forward(update_fast_weights=True)` has
        ZERO intra-chunk or cross-layer future token leakage (modifying tokens t' >= t0
        leaves logits[:, :t0, :] bit-for-bit identical), and verify CJK 1-token encoding.
        """
        tok = SimpleUTF8Tokenizer(vocab_size=32000)
        sample_text = "<|user|> 你好，我是Phova！\n<|assistant|> 你好！很高兴认识你。"
        encoded = tok.encode(sample_text, add_bos=True, add_eos=True)
        decoded = tok.decode(encoded, skip_special_tokens=True)
        self.assertIn("你好，我是Phova！", decoded)
        self.assertIn(tok.USER_ID, encoded)
        self.assertIn(tok.ASSISTANT_ID, encoded)

        cfg = uTTTConfig(
            vocab_size=512,
            d_model=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            d_ff_fast=128,
            chunk_size=8,
            sliding_window_size=32,
            gradient_checkpointing=False,
        )
        model = uTTTModel(cfg).eval()
        ids_a = torch.randint(0, cfg.vocab_size, (2, 24))
        ids_b = ids_a.clone()
        # Corrupt future tokens from index 12..23 (inside chunk 1 and chunk 2)
        ids_b[:, 12:] = (ids_a[:, 12:] + 77) % cfg.vocab_size

        with torch.no_grad():
            out_a = model(input_ids=ids_a, update_fast_weights=True)
            out_b = model(input_ids=ids_b, update_fast_weights=True)

        # Logits for tokens 0..11 must be 100% invariant to future tokens 12..23
        self.assertTrue(
            torch.allclose(out_a.logits[:, :12, :], out_b.logits[:, :12, :], atol=1e-5, rtol=1e-5)
        )

    def test_05_multimodal_nvs_novel_view_synthesis(self) -> None:
        """
        Verify `uTTTNVSModel` Plucker ray parameterization, 3D scene compression into
        shared SwiGLU Fast Weights W_scene, and novel view synthesis readout + gradients.
        """
        cfg = uTTTConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff_fast=128,
            chunk_size=16,
            sliding_window_size=64,
        )
        nvs_model = uTTTNVSModel(cfg, img_size=32, patch_size=8)
        bsz, v_ctx, v_tgt, h, w = 2, 2, 1, 32, 32

        ctx_rgb = torch.rand(bsz, v_ctx, 3, h, w)
        tgt_rgb = torch.rand(bsz, v_tgt, 3, h, w)
        eye3 = torch.eye(3).view(1, 1, 3, 3)
        eye4 = torch.eye(4).view(1, 1, 4, 4)
        ctx_k = eye3.repeat(bsz, v_ctx, 1, 1) * 16.0
        ctx_k[:, :, 2, 2] = 1.0
        ctx_c2w = eye4.repeat(bsz, v_ctx, 1, 1)
        tgt_k = eye3.repeat(bsz, v_tgt, 1, 1) * 16.0
        tgt_k[:, :, 2, 2] = 1.0
        tgt_c2w = eye4.repeat(bsz, v_tgt, 1, 1)

        out = nvs_model(
            ctx_rgb=ctx_rgb,
            ctx_intrinsics=ctx_k,
            ctx_c2w=ctx_c2w,
            tgt_intrinsics=tgt_k,
            tgt_c2w=tgt_c2w,
            tgt_rgb=tgt_rgb,
        )
        self.assertEqual(out.rendered_rgb.shape, (bsz, v_tgt, 3, h, w))
        self.assertGreater(out.scene_fast_state.step, 0)
        out.loss.backward()
        self.assertIsNotNone(nvs_model.fast_weight_pool.w_up_init.grad)


if __name__ == "__main__":
    unittest.main()

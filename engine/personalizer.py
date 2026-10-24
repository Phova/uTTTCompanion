"""
engine/personalizer.py
======================
Interactive Test-Time Personalization Engine (`uTTTPersonalizer`).

Implements the complete closed-loop Test-Time Shaping state machine:
1. Cold-Start / Session Load:
   Loads user's `FastWeightState` W_user (or clones W_0 from `base_uttt.pt` on first connect)
   while sharing the single frozen Slow-Weight backbone in RTX 5090 VRAM.
2. `interact_and_evolve(user_id, prompt)`:
   - Step 1 (Forward): Freeze W_user, stream-generate reply tokens.
   - Step 2 (Test-Time Update): Concatenate `[User Prompt + Assistant Reply]` and run
     analytical chunk-wise TTT update to compute Surprise Loss and candidate W_updated.
   - Step 3 (Stability Guard):
     * Surprise Threshold Gate (tau): Only commit update when Surprise Loss > tau.
     * Slow-Anchor Elastic Pull (lambda):
           W_new <- (1 - lambda) * W_updated + lambda * W_0
       preventing catastrophic forgetting or weight collapse over long dialogues.
3. State Control Commands:
   Supports `/save`, `/reset`, `/snapshot [tag]`, `/restore <tag>`, and `/diff`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import torch

from engine.checkpoint_manager import DualTrackCheckpointManager
from model.uttt_layer import FastWeightState
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel


@dataclass
class EvolutionResult:
    """Structured result returned after an interactive turn or slash command."""
    user_id: str
    prompt: str
    reply: str
    surprise_loss: float
    surprise_threshold: float
    gate_triggered: bool
    weight_drift_frobenius: float
    weight_drift_pct: float
    persona_saved: bool
    persona_path: str
    step: int
    status_bar: str
    layer_surprise_losses: List[float] = field(default_factory=list)
    is_command: bool = False
    command_name: Optional[str] = None
    drift_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_ttt_status_bar(
    surprise_loss: float,
    weight_drift_pct: float,
    persona_saved: bool = True,
    gate_triggered: bool = True,
) -> str:
    """
    Format the canonical single-line CLI status bar required by MODULE 2:
        `[TTT Status] Surprise Loss: 0.42 | Weight Drift from Base: 3.18% | Persona Saved`
    """
    if persona_saved:
        save_label = "Persona Saved"
    elif not gate_triggered:
        save_label = "Gate Skipped (Low Surprise)"
    else:
        save_label = "In-Memory Updated"

    return (
        f"[TTT Status] Surprise Loss: {surprise_loss:.2f} | "
        f"Weight Drift from Base: {weight_drift_pct:.2f}% | {save_label}"
    )


class uTTTPersonalizer:
    """
    High-level engine orchestrating multi-user Test-Time Personalization on RTX 5090.
    """

    def __init__(
        self,
        checkpoint_manager: Optional[DualTrackCheckpointManager] = None,
        base_checkpoint_path: str = "checkpoints/base_uttt.pt",
        personas_dir: str = "personas",
        config_path: str = "configs/5090_config.yaml",
        surprise_threshold: Optional[float] = None,
        elastic_anchor_lambda: Optional[float] = None,
        max_weight_drift_ratio: Optional[float] = None,
        auto_save_every_turn: bool = True,
        default_max_new_tokens: int = 96,
        temperature: float = 0.7,
        top_p: float = 0.9,
        lightweight_fallback: bool = False,
    ) -> None:
        if checkpoint_manager is None:
            self.ckpt_manager = DualTrackCheckpointManager(
                base_checkpoint_path=base_checkpoint_path,
                personas_dir=personas_dir,
                config_path=config_path,
                lightweight_fallback=lightweight_fallback,
            )
        else:
            self.ckpt_manager = checkpoint_manager

        self.model: uTTTModel = self.ckpt_manager.model
        self.config: uTTTConfig = self.ckpt_manager.config
        self.device = self.ckpt_manager.device
        self.dtype = self.ckpt_manager.dtype
        self.tokenizer = SimpleUTF8Tokenizer(vocab_size=self.config.vocab_size)

        # Stability Guard hyperparameters
        self.surprise_threshold = (
            float(surprise_threshold)
            if surprise_threshold is not None
            else float(self.config.surprise_threshold)
        )
        self.elastic_anchor_lambda = (
            float(elastic_anchor_lambda)
            if elastic_anchor_lambda is not None
            else float(self.config.elastic_anchor_lambda)
        )
        self.max_weight_drift_ratio = (
            float(max_weight_drift_ratio)
            if max_weight_drift_ratio is not None
            else float(self.config.max_weight_drift_ratio)
        )
        self.auto_save_every_turn = auto_save_every_turn
        self.default_max_new_tokens = default_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        # DeepInfer Asynchronous CUDA Stream for background TTT fast-weight updates
        self._ttt_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    # --------------------------------------------------------------------------
    # Slash Command Execution (/save, /reset, /snapshot, /restore, /diff)
    # --------------------------------------------------------------------------

    def is_slash_command(self, text: str) -> bool:
        stripped = text.strip()
        return stripped.startswith(("/save", "/reset", "/snapshot", "/restore", "/diff"))

    def handle_command(self, user_id: str, command_text: str) -> EvolutionResult:
        """Execute state control commands (`/save`, `/reset`, `/snapshot`, `/diff`)."""
        parts = command_text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None

        if cmd == "/save":
            saved_path = self.ckpt_manager.save_user_state(user_id)
            drift = self.ckpt_manager.compute_user_drift(user_id)
            reply_msg = (
                f"Persona state persisted to `{saved_path}` "
                f"(Step={drift['step']}, Drift={drift['drift_pct']:.2f}%)."
            )
            saved_flag = True

        elif cmd == "/reset":
            self.ckpt_manager.reset_user_state(user_id, persist=True)
            drift = self.ckpt_manager.compute_user_drift(user_id)
            reply_msg = (
                "Persona fast weights and momentum buffers have been reset to Base Checkpoint W_0 "
                f"(Drift={drift['drift_pct']:.2f}%)."
            )
            saved_flag = True

        elif cmd == "/snapshot":
            snap_path = self.ckpt_manager.save_snapshot(user_id, snapshot_tag=arg)
            drift = self.ckpt_manager.compute_user_drift(user_id)
            reply_msg = (
                f"Persona snapshot saved to `{snap_path}` "
                f"(Step={drift['step']}, Drift={drift['drift_pct']:.2f}%)."
            )
            saved_flag = True

        elif cmd == "/restore":
            if not arg:
                drift = self.ckpt_manager.compute_user_drift(user_id)
                reply_msg = "Usage: `/restore <snapshot_tag>`"
                saved_flag = False
            else:
                self.ckpt_manager.load_snapshot(user_id, snapshot_tag=arg)
                drift = self.ckpt_manager.compute_user_drift(user_id)
                reply_msg = (
                    f"Restored persona snapshot `{arg}` "
                    f"(Step={drift['step']}, Drift={drift['drift_pct']:.2f}%)."
                )
                saved_flag = True

        elif cmd == "/diff":
            drift = self.ckpt_manager.compute_user_drift(user_id)
            reply_msg = (
                "Fast-Weight Drift Report ||W_user - W_0||_F:\n"
                f"  - Total Frobenius Drift : {drift['drift_frobenius']:.4f} "
                f"/ Base {drift['base_frobenius']:.4f} ({drift['drift_pct']:.2f}%)\n"
                f"  - W_up   Drift          : {drift['w_up_drift']:.4f} ({drift['w_up_drift_pct']:.2f}%)\n"
                f"  - W_gate Drift          : {drift['w_gate_drift']:.4f} ({drift['w_gate_drift_pct']:.2f}%)\n"
                f"  - W_down Drift          : {drift['w_down_drift']:.4f} ({drift['w_down_drift_pct']:.2f}%)\n"
                f"  - Accumulated TTT Steps : {drift['step']}\n"
                f"  - Last Surprise Loss    : {drift['last_surprise_loss']:.4f}"
            )
            saved_flag = drift.get("file_exists", False)

        else:
            drift = self.ckpt_manager.compute_user_drift(user_id)
            reply_msg = (
                f"Unknown command `{cmd}`. Supported commands: `/save`, `/reset`, "
                "`/snapshot [tag]`, `/restore <tag>`, `/diff`."
            )
            saved_flag = False

        primary_path, _ = self.ckpt_manager.get_persona_paths(user_id)
        status_bar = format_ttt_status_bar(
            surprise_loss=float(drift.get("last_surprise_loss", 0.0)),
            weight_drift_pct=float(drift.get("drift_pct", 0.0)),
            persona_saved=saved_flag,
            gate_triggered=True,
        )
        return EvolutionResult(
            user_id=user_id,
            prompt=command_text,
            reply=reply_msg,
            surprise_loss=float(drift.get("last_surprise_loss", 0.0)),
            surprise_threshold=self.surprise_threshold,
            gate_triggered=False,
            weight_drift_frobenius=float(drift.get("drift_frobenius", 0.0)),
            weight_drift_pct=float(drift.get("drift_pct", 0.0)),
            persona_saved=saved_flag,
            persona_path=str(primary_path),
            step=int(drift.get("step", 0)),
            status_bar=status_bar,
            is_command=True,
            command_name=cmd,
            drift_details=drift,
        )

    # --------------------------------------------------------------------------
    # Core Personalization Loop: interact_and_evolve & stream_interact_and_evolve
    # --------------------------------------------------------------------------

    def stream_interact_and_evolve(
        self,
        user_id: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Generator[Dict[str, Any], None, EvolutionResult]:
        """
        Streaming generator version of `interact_and_evolve` for WebSocket/CLI streaming:
        1. Yields `{"type": "token", "content": token_str}` as tokens are generated with
           frozen `W_user`.
        2. Runs the Test-Time Update + Stability Guard on `[User Prompt + Assistant Reply]`.
        3. Yields `{"type": "ttt_status", "result": EvolutionResult.to_dict()}` and returns
           the final `EvolutionResult`.
        """
        if self.is_slash_command(prompt):
            cmd_res = self.handle_command(user_id, prompt)
            yield {"type": "token", "content": cmd_res.reply}
            yield {"type": "ttt_status", "result": cmd_res.to_dict()}
            return cmd_res

        max_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p

        # Retrieve user's isolated FastWeightState on GPU
        user_state = self.ckpt_manager.get_user_state(user_id)

        # Format prompt & encode
        formatted_prompt = f"<|user|> {prompt.strip()}\n<|assistant|> "
        prompt_ids = self.tokenizer.encode(
            formatted_prompt,
            add_bos=True,
            add_eos=False,
            max_length=self.config.max_seq_len - max_tokens,
        )
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # ======================================================================
        # Step 1 (Forward): Freeze Fast Weights & Stream-Generate Reply Tokens
        # ======================================================================
        generated_ids: List[int] = []
        autocast_dev = "cuda" if self.device.type == "cuda" else "cpu"

        with torch.amp.autocast(autocast_dev, dtype=self.dtype):
            for tok_id in self.model.stream_generate(
                input_ids=input_tensor,
                fast_state=user_state,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=tp,
                eos_token_id=self.tokenizer.EOS_ID,
            ):
                generated_ids.append(tok_id)
                piece = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                if piece:
                    yield {"type": "token", "content": piece}

        reply_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not reply_text:
            reply_text = "[Acknowledged]"
            yield {"type": "token", "content": reply_text}

        # ======================================================================
        # Step 2 (Test-Time Update): Chunk TTT Update on [User Prompt + Reply]
        # ======================================================================
        full_turn_text = f"<|user|> {prompt.strip()}\n<|assistant|> {reply_text}"
        turn_ids = self.tokenizer.encode(
            full_turn_text,
            add_bos=True,
            add_eos=True,
            max_length=self.config.max_seq_len,
        )
        # Pad up to a multiple of chunk_size if shorter than chunk_size for optimal GEMM alignment
        chunk_sz = self.config.chunk_size
        valid_len = len(turn_ids)
        padded_len = ((valid_len + chunk_sz - 1) // chunk_sz) * chunk_sz
        pad_count = padded_len - valid_len

        padded_ids = turn_ids + [self.tokenizer.PAD_ID] * pad_count
        attn_mask = [1.0] * valid_len + [0.0] * pad_count

        interaction_tensor = torch.tensor([padded_ids], dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor([attn_mask], dtype=torch.float32, device=self.device)

        # DeepInfer Async Stream Execution: isolate TTT inner-loop update on dedicated CUDA stream
        if self._ttt_stream is not None:
            self._ttt_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._ttt_stream), torch.amp.autocast(autocast_dev, dtype=self.dtype):
                candidate_state, surprise_loss, layer_surprises = self.model.test_time_update(
                    interaction_ids=interaction_tensor,
                    fast_state=user_state,
                    attention_mask=mask_tensor,
                )
            torch.cuda.current_stream(self.device).wait_stream(self._ttt_stream)
        else:
            with torch.amp.autocast(autocast_dev, dtype=self.dtype):
                candidate_state, surprise_loss, layer_surprises = self.model.test_time_update(
                    interaction_ids=interaction_tensor,
                    fast_state=user_state,
                    attention_mask=mask_tensor,
                )

        # ======================================================================
        # Step 3 (Stability Guard): Surprise Threshold Gate + Elastic Anchor Pull
        # ======================================================================
        gate_triggered = bool(surprise_loss > self.surprise_threshold)
        persona_saved = False
        primary_path, _ = self.ckpt_manager.get_persona_paths(user_id)

        if gate_triggered:
            # Apply Slow-Anchor Elastic Constraint:
            #   W_new <- (1 - lambda) * W_updated + lambda * W_0
            stabilized_state = self.model.fast_weight_pool.apply_elastic_pull(
                state=candidate_state,
                anchor_lambda=self.elastic_anchor_lambda,
                max_drift_ratio=self.max_weight_drift_ratio,
            )
            saved_path = self.ckpt_manager.update_user_state(
                user_id=user_id,
                new_state=stabilized_state,
                auto_save=self.auto_save_every_turn,
            )
            persona_saved = saved_path is not None
        else:
            # Below surprise threshold tau: skip weight modification, record surprise metric
            user_state.last_surprise_loss = surprise_loss
            if self.auto_save_every_turn and not primary_path.exists():
                self.ckpt_manager.save_user_state(user_id)
                persona_saved = True

        drift_metrics = self.ckpt_manager.compute_user_drift(user_id)
        drift_fro = float(drift_metrics["drift_frobenius"])
        drift_pct = float(drift_metrics["drift_pct"])
        cur_step = int(drift_metrics["step"])

        status_bar = format_ttt_status_bar(
            surprise_loss=surprise_loss,
            weight_drift_pct=drift_pct,
            persona_saved=persona_saved,
            gate_triggered=gate_triggered,
        )

        result = EvolutionResult(
            user_id=user_id,
            prompt=prompt,
            reply=reply_text,
            surprise_loss=surprise_loss,
            surprise_threshold=self.surprise_threshold,
            gate_triggered=gate_triggered,
            weight_drift_frobenius=drift_fro,
            weight_drift_pct=drift_pct,
            persona_saved=persona_saved,
            persona_path=str(primary_path),
            step=cur_step,
            status_bar=status_bar,
            layer_surprise_losses=layer_surprises,
            is_command=False,
            drift_details=drift_metrics,
        )

        yield {"type": "ttt_status", "result": result.to_dict()}
        return result

    def interact_and_evolve(
        self,
        user_id: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> EvolutionResult:
        """
        Execute the full 3-step Test-Time Personalization cycle for `(user_id, prompt)`:
        1. Forward generation with frozen fast weights `W_user`.
        2. Analytical Chunk-wise TTT update on `[User Prompt + Assistant Reply]`.
        3. Stability Guard (Surprise Threshold Gate `tau` + Slow-Anchor Elastic Pull `lambda`).
        """
        gen = self.stream_interact_and_evolve(
            user_id=user_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        final_result: Optional[EvolutionResult] = None
        try:
            while True:
                event = next(gen)
                if event.get("type") == "ttt_status":
                    res_dict = event["result"]
                    final_result = EvolutionResult(**res_dict)
        except StopIteration as stop:
            if stop.value is not None:
                final_result = stop.value

        if final_result is None:
            raise RuntimeError("interact_and_evolve did not produce an EvolutionResult")
        return final_result

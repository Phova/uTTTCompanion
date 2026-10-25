"""
server/app.py
=============
FastAPI + WebSocket Streaming Inference & Real-Time Test-Time Personalization Server.

Endpoints:
- `GET  /health`                  : GPU VRAM and Multi-User LRU Persona Pool status.
- `POST /v1/chat`                 : Synchronous chat + Test-Time Personalization turn.
- `POST /v1/command`              : Execute `/save`, `/reset`, `/snapshot`, `/restore`, `/diff`.
- `GET  /v1/persona/{user_id}/diff`: Inspect Frobenius norm drift ||W_user - W_0||_F.
- `WS   /ws/chat/{user_id}`       : Full-duplex WebSocket streaming token generation +
                                    real-time post-turn Chunk TTT update & status bar push.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from engine.checkpoint_manager import DualTrackCheckpointManager
from engine.personalizer import uTTTPersonalizer


# ==============================================================================
# 1. Request / Response Schemas
# ==============================================================================

class ChatRequest(BaseModel):
    user_id: str = Field(default="default_user", description="Unique user identifier")
    prompt: str = Field(..., description="User message or slash command (/save, /reset, /snapshot, /diff)")
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=1024)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class CommandRequest(BaseModel):
    user_id: str = Field(default="default_user", description="Unique user identifier")
    command: str = Field(..., description="Slash command: /save, /reset, /snapshot [tag], /restore <tag>, /diff")


# ==============================================================================
# 2. Global Singleton Engine & Lifespan Initialization
# ==============================================================================

_personalizer: Optional[uTTTPersonalizer] = None


def get_personalizer() -> uTTTPersonalizer:
    global _personalizer
    if _personalizer is None:
        config_path = os.environ.get("UTTT_CONFIG", "configs/5090_config.yaml")
        base_ckpt = os.environ.get("UTTT_BASE_CKPT", "checkpoints/base_uttt.pt")
        personas_dir = os.environ.get("UTTT_PERSONAS_DIR", "personas")
        lightweight = os.environ.get("UTTT_LIGHTWEIGHT", "0") == "1"

        max_gpu_users = 16
        max_cpu_users = 128
        auto_save = True
        max_new_tokens = 96
        temperature = 0.7
        top_p = 0.9

        if Path(config_path).exists():
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
            p_cfg = raw_cfg.get("personalization", {})
            t_cfg = raw_cfg.get("training", {})
            base_ckpt = os.environ.get(
                "UTTT_BASE_CKPT", t_cfg.get("base_checkpoint_path", base_ckpt)
            )
            personas_dir = os.environ.get(
                "UTTT_PERSONAS_DIR", p_cfg.get("personas_dir", personas_dir)
            )
            max_gpu_users = int(p_cfg.get("max_gpu_users", max_gpu_users))
            max_cpu_users = int(p_cfg.get("max_cpu_users", max_cpu_users))
            auto_save = bool(p_cfg.get("auto_save_every_turn", auto_save))
            max_new_tokens = int(p_cfg.get("default_max_new_tokens", max_new_tokens))
            temperature = float(p_cfg.get("temperature", temperature))
            top_p = float(p_cfg.get("top_p", top_p))

        if not os.environ.get("UTTT_BASE_CKPT"):
            ckpt_dir = Path(base_ckpt).parent
            if ckpt_dir.exists():
                pt_files = sorted(
                    ckpt_dir.glob("*.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if pt_files:
                    base_ckpt = str(pt_files[0])

        if os.environ.get("UTTT_MAX_GPU_USERS"):
            max_gpu_users = int(os.environ["UTTT_MAX_GPU_USERS"])
        if os.environ.get("UTTT_MAX_NEW_TOKENS"):
            max_new_tokens = int(os.environ["UTTT_MAX_NEW_TOKENS"])
        if os.environ.get("UTTT_TEMPERATURE"):
            temperature = float(os.environ["UTTT_TEMPERATURE"])
        surprise_threshold: Optional[float] = (
            float(os.environ["UTTT_SURPRISE_THRESHOLD"])
            if os.environ.get("UTTT_SURPRISE_THRESHOLD")
            else None
        )

        ckpt_manager = DualTrackCheckpointManager(
            base_checkpoint_path=base_ckpt,
            personas_dir=personas_dir,
            config_path=config_path,
            max_gpu_users=max_gpu_users,
            max_cpu_users=max_cpu_users,
            auto_create_base_if_missing=True,
            lightweight_fallback=lightweight,
        )
        _personalizer = uTTTPersonalizer(
            checkpoint_manager=ckpt_manager,
            config_path=config_path,
            surprise_threshold=surprise_threshold,
            auto_save_every_turn=auto_save,
            default_max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    return _personalizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize shared Slow Weights & LRU Fast-Weight Pool on startup
    engine = get_personalizer()
    compile_kernels = os.environ.get("UTTT_NO_COMPILE", "0") != "1"
    engine.warmup(compile_kernels=compile_kernels)
    prewarm_user = os.environ.get("UTTT_DEFAULT_USER", "").strip()
    if prewarm_user:
        engine.ckpt_manager.get_user_state(prewarm_user)
        print(f"[Server] Pre-warmed persona state for user_id='{prewarm_user}' in GPU VRAM.")
    print(f"[Server] uTTT Personalization Engine online | Pool: {engine.ckpt_manager.pool_status()}")
    yield
    print("[Server] Shutting down uTTT Personalization Server...")


app = FastAPI(
    title="uTTT Interactive Personalization Server",
    description=(
        "Universal Test-Time Training (uTTT) inference & real-time Fast-Weight "
        "shaping service optimized for NVIDIA RTX 5090."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# 3. REST & WebSocket Endpoints
# ==============================================================================

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    engine = get_personalizer()
    pool = engine.ckpt_manager.pool_status()
    vram_allocated_gb = (
        torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )
    vram_reserved_gb = (
        torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )
    return {
        "status": "ok",
        "model_name": engine.config.name,
        "vram_allocated_gb": round(vram_allocated_gb, 3),
        "vram_reserved_gb": round(vram_reserved_gb, 3),
        "pool": pool,
    }


@app.post("/v1/chat")
async def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    engine = get_personalizer()
    try:
        result = await asyncio.to_thread(
            engine.interact_and_evolve,
            user_id=req.user_id,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/command")
async def command_endpoint(req: CommandRequest) -> Dict[str, Any]:
    engine = get_personalizer()
    try:
        result = await asyncio.to_thread(
            engine.handle_command,
            user_id=req.user_id,
            command_text=req.command,
        )
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/persona/{user_id}/diff")
async def persona_diff_endpoint(user_id: str) -> Dict[str, Any]:
    engine = get_personalizer()
    try:
        return await asyncio.to_thread(engine.ckpt_manager.compute_user_drift, user_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.websocket("/ws/chat/{user_id}")
async def websocket_chat_endpoint(websocket: WebSocket, user_id: str) -> None:
    """
    Full-duplex WebSocket streaming endpoint:
    - Accepts JSON `{"prompt": "...", "max_new_tokens": 96, "temperature": 0.7}` or plain text.
    - Streams `{"type": "token", "content": "..."}` during frozen-fast-weight generation.
    - Sends `{"type": "ttt_status", "result": {...}}` once Chunk TTT update & Stability Guard finish.
    """
    await websocket.accept()
    engine = get_personalizer()

    # Warm up user's FastWeightState into GPU LRU pool on connect
    await asyncio.to_thread(engine.ckpt_manager.get_user_state, user_id)
    init_drift = await asyncio.to_thread(engine.ckpt_manager.compute_user_drift, user_id)
    await websocket.send_json(
        {
            "type": "session_ready",
            "user_id": user_id,
            "step": init_drift["step"],
            "weight_drift_pct": init_drift["drift_pct"],
        }
    )

    try:
        while True:
            raw_msg = await websocket.receive_text()
            if not raw_msg.strip():
                continue

            try:
                payload = json.loads(raw_msg)
                if isinstance(payload, dict):
                    prompt = str(payload.get("prompt", "")).strip()
                    max_new_tokens = payload.get("max_new_tokens")
                    temperature = payload.get("temperature")
                    top_p = payload.get("top_p")
                else:
                    prompt = str(payload).strip()
                    max_new_tokens = None
                    temperature = None
                    top_p = None
            except json.JSONDecodeError:
                prompt = raw_msg.strip()
                max_new_tokens = None
                temperature = None
                top_p = None

            if not prompt:
                continue

            # Stream events from the generator without blocking the event loop
            gen = engine.stream_interact_and_evolve(
                user_id=user_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            def _next_event(it):
                try:
                    return next(it)
                except StopIteration:
                    return None

            while True:
                event = await asyncio.to_thread(_next_event, gen)
                if event is None:
                    break
                await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="uTTT FastAPI + WebSocket Personalization Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server listen port")
    parser.add_argument("--config", type=str, default="configs/5090_config.yaml", help="YAML config path")
    parser.add_argument(
        "--base-ckpt",
        "--save-path",
        "--output-ckpt",
        dest="base_ckpt",
        type=str,
        default=None,
        help="Base checkpoint path (defaults to latest .pt in checkpoints/ or base_uttt.pt)",
    )
    parser.add_argument("--personas-dir", type=str, default="personas", help="User persona fast-weight directory")
    parser.add_argument("--user-id", type=str, default="", help="Optional default user ID to pre-warm in GPU VRAM")
    parser.add_argument("--max-gpu-users", type=int, default=None, help="Override Tier-1 GPU VRAM LRU pool size")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override default max generated tokens per turn")
    parser.add_argument("--temperature", type=float, default=None, help="Override default sampling temperature")
    parser.add_argument("--surprise-threshold", type=float, default=None, help="Override Surprise Gate threshold tau")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile kernel fusion on startup")
    parser.add_argument("--lightweight", action="store_true", help="Use lightweight fallback model when cold-starting")
    args, _ = parser.parse_known_args()

    os.environ["UTTT_CONFIG"] = args.config
    if args.base_ckpt:
        os.environ["UTTT_BASE_CKPT"] = args.base_ckpt
    os.environ["UTTT_PERSONAS_DIR"] = args.personas_dir
    if args.user_id:
        os.environ["UTTT_DEFAULT_USER"] = args.user_id
    if args.max_gpu_users is not None:
        os.environ["UTTT_MAX_GPU_USERS"] = str(args.max_gpu_users)
    if args.max_new_tokens is not None:
        os.environ["UTTT_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    if args.temperature is not None:
        os.environ["UTTT_TEMPERATURE"] = str(args.temperature)
    if args.surprise_threshold is not None:
        os.environ["UTTT_SURPRISE_THRESHOLD"] = str(args.surprise_threshold)
    if args.no_compile:
        os.environ["UTTT_NO_COMPILE"] = "1"
    if args.lightweight:
        os.environ["UTTT_LIGHTWEIGHT"] = "1"

    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

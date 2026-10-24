"""
client_chat.py
==============
Interactive CLI Chat Client for uTTT (Universal Test-Time Training).

Features:
1. Real-time token streaming over WebSocket (`ws://localhost:8000/ws/chat/{user_id}`)
   or in-process local execution (`--local`).
2. Displays the real-time TTT Status Bar after every dialogue turn:
       `[TTT Status] Surprise Loss: 0.42 | Weight Drift from Base: 3.18% | Persona Saved`
3. Interactive Persona Lifecycle Commands:
   - `/save`           : Persist current user fast weights to `personas/{user_id}_state.pt`
   - `/reset`          : Reset user fast weights & momentum back to Base Checkpoint W_0
   - `/snapshot [tag]` : Save a named persona snapshot
   - `/restore <tag>`  : Restore a saved persona snapshot
   - `/diff`           : View Frobenius norm drift Delta ||W_user - W_0||_F
   - `/help`           : List supported commands
   - `/quit` or `/exit`: Exit the client
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Optional

from engine.personalizer import format_ttt_status_bar, uTTTPersonalizer


BANNER = """================================================================================
  uTTT Interactive Personalization Client (RTX 5090 Fast-Weight Shaping)
  Commands: /save | /reset | /snapshot [tag] | /restore <tag> | /diff | /quit
================================================================================"""


def print_help() -> None:
    print(
        "\nSupported Interactive Commands:\n"
        "  /save             Persist current personalized fast weights W_user to disk\n"
        "  /reset            Reset fast weights & momentum back to Base Checkpoint W_0\n"
        "  /snapshot [tag]   Save an immutable snapshot of current persona state\n"
        "  /restore <tag>    Restore a previously saved persona snapshot\n"
        "  /diff             Inspect Frobenius norm drift ||W_user - W_0||_F per matrix\n"
        "  /help             Show this help message\n"
        "  /quit             Exit interactive session\n"
    )


async def run_websocket_session(
    ws_url: str,
    user_id: str,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    demo_prompts: Optional[List[str]] = None,
) -> None:
    """Connect to the remote/tunneled FastAPI WebSocket endpoint and stream chat + TTT updates."""
    import websockets

    endpoint = f"{ws_url.rstrip('/')}/ws/chat/{user_id}"
    print(BANNER)
    print(f"[Client] Connecting to {endpoint} (user_id='{user_id}')...")

    async with websockets.connect(endpoint, ping_interval=25) as ws:
        # Receive initial session_ready frame
        init_raw = await ws.recv()
        init_msg = json.loads(init_raw)
        if init_msg.get("type") == "session_ready":
            print(
                f"[Session Ready] User: {init_msg.get('user_id')} | "
                f"Accumulated TTT Steps: {init_msg.get('step', 0)} | "
                f"Current Drift: {init_msg.get('weight_drift_pct', 0.0):.2f}%\n"
            )

        prompt_idx = 0
        while True:
            if demo_prompts is not None:
                if prompt_idx >= len(demo_prompts):
                    break
                user_input = demo_prompts[prompt_idx]
                prompt_idx += 1
                print(f"\nYou ({user_id}) > {user_input}")
            else:
                try:
                    user_input = input(f"\nYou ({user_id}) > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Client] Session closed.")
                    break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit"):
                print("[Client] Saving session and exiting...")
                await ws.send(json.dumps({"prompt": "/save"}))
                break
            if user_input.lower() == "/help":
                print_help()
                continue

            await ws.send(
                json.dumps(
                    {
                        "prompt": user_input,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    }
                )
            )

            print("Assistant > ", end="", flush=True)
            while True:
                raw_event = await ws.recv()
                event = json.loads(raw_event)
                ev_type = event.get("type")

                if ev_type == "token":
                    print(event.get("content", ""), end="", flush=True)
                elif ev_type == "ttt_status":
                    res = event.get("result", {})
                    status_bar = res.get("status_bar") or format_ttt_status_bar(
                        surprise_loss=float(res.get("surprise_loss", 0.0)),
                        weight_drift_pct=float(res.get("weight_drift_pct", 0.0)),
                        persona_saved=bool(res.get("persona_saved", True)),
                        gate_triggered=bool(res.get("gate_triggered", True)),
                    )
                    print(f"\n{status_bar}", flush=True)
                    break
                elif ev_type == "error":
                    print(f"\n[Server Error] {event.get('message')}", flush=True)
                    break


def run_local_session(
    user_id: str,
    config_path: str = "configs/5090_config.yaml",
    base_ckpt: str = "checkpoints/base_uttt.pt",
    personas_dir: str = "personas",
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    lightweight: bool = False,
    demo_prompts: Optional[List[str]] = None,
) -> None:
    """Run interactive chat directly against an in-process `uTTTPersonalizer`."""
    print(BANNER)
    personalizer = uTTTPersonalizer(
        base_checkpoint_path=base_ckpt,
        personas_dir=personas_dir,
        config_path=config_path,
        default_max_new_tokens=max_new_tokens,
        temperature=temperature,
        lightweight_fallback=lightweight,
    )
    if not lightweight:
        personalizer.warmup(compile_kernels=True)
    init_drift = personalizer.ckpt_manager.compute_user_drift(user_id)
    print(
        f"[Local Session Ready] User: {user_id} | "
        f"Accumulated TTT Steps: {init_drift['step']} | "
        f"Current Drift: {init_drift['drift_pct']:.2f}%\n"
    )

    prompt_idx = 0
    while True:
        if demo_prompts is not None:
            if prompt_idx >= len(demo_prompts):
                break
            user_input = demo_prompts[prompt_idx]
            prompt_idx += 1
            print(f"\nYou ({user_id}) > {user_input}")
        else:
            try:
                user_input = input(f"\nYou ({user_id}) > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Client] Session closed.")
                break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            personalizer.handle_command(user_id, "/save")
            print("[Client] Persona state saved. Goodbye!")
            break
        if user_input.lower() == "/help":
            print_help()
            continue

        print("Assistant > ", end="", flush=True)
        for event in personalizer.stream_interact_and_evolve(
            user_id=user_id,
            prompt=user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            ev_type = event.get("type")
            if ev_type == "token":
                print(event.get("content", ""), end="", flush=True)
            elif ev_type == "ttt_status":
                res = event["result"]
                print(f"\n{res['status_bar']}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="uTTT Interactive Personalization CLI Client")
    parser.add_argument(
        "--user-id",
        type=str,
        default="alex",
        help="User ID for isolated Fast-Weight persona state (`personas/{user_id}_state.pt`).",
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://localhost:8000",
        help="WebSocket server base URL (tunneled from remote RTX 5090 via remote_ctl.sh serve).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in-process local Personalizer instead of connecting to WebSocket server.",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight model dimensions when initializing local cold-start model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/5090_config.yaml",
        help="Path to YAML config (used in --local mode).",
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="checkpoints/base_uttt.pt",
        help="Path to Base Checkpoint (used in --local mode).",
    )
    parser.add_argument(
        "--personas-dir",
        type=str,
        default="personas",
        help="Directory for user persona checkpoints (used in --local mode).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a scripted multi-turn demonstration showing TTT updates and /diff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_script: Optional[List[str]] = None
    if args.demo:
        demo_script = [
            "Hi! My name is Dr. Lin, I am a CUDA kernel architect working on Blackwell RTX 5090.",
            "Please always reply using concise mathematical notation and PyTorch bmm examples.",
            "/diff",
            "/snapshot stage1_cuda_persona",
            "/save",
        ]

    if args.local:
        run_local_session(
            user_id=args.user_id,
            config_path=args.config,
            base_ckpt=args.base_ckpt,
            personas_dir=args.personas_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            lightweight=args.lightweight,
            demo_prompts=demo_script,
        )
    else:
        try:
            asyncio.run(
                run_websocket_session(
                    ws_url=args.ws_url,
                    user_id=args.user_id,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    demo_prompts=demo_script,
                )
            )
        except Exception as exc:
            print(
                f"\n[Client] Could not reach WebSocket server at {args.ws_url} ({exc}).\n"
                "[Client] Falling back to in-process local Personalizer mode (--local)...\n"
            )
            run_local_session(
                user_id=args.user_id,
                config_path=args.config,
                base_ckpt=args.base_ckpt,
                personas_dir=args.personas_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                lightweight=args.lightweight,
                demo_prompts=demo_script,
            )


if __name__ == "__main__":
    main()

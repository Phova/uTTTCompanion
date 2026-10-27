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
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import sys
from typing import List, Optional
from urllib.parse import urlsplit, quote


def format_ttt_status_bar(
    surprise_loss: float,
    weight_drift_pct: float,
    persona_saved: bool = True,
    gate_triggered: bool = True,
) -> str:
    """Format the canonical single-line CLI status bar without requiring torch on thin clients."""
    if persona_saved:
        save_label = "Persona Saved"
    elif not gate_triggered:
        save_label = "Gate Skipped (Low Surprise)"
    else:
        save_label = "In-Memory Updated"

    loss_str = f"{surprise_loss:.4f}" if 0.0 < surprise_loss < 0.1 else f"{surprise_loss:.2f}"
    return (
        f"[TTT Status] Surprise Loss: {loss_str} | "
        f"Weight Drift from Base: {weight_drift_pct:.2f}% | {save_label}"
    )


class _StdlibWebSocket:
    """
    Zero-dependency RFC 6455 WebSocket client built on Python's standard library (`socket`),
    allowing `client_chat.py` to run on any local Windows/WSL Python installation without
    requiring `pip install websockets` or `torch`.
    """

    def __init__(self, url: str, timeout: float = 120.0) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError("WebSocket URL must use ws:// or wss://")
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += f"?{parsed.query}"
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        try:
            if parsed.scheme == "wss":
                self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)
            self._handshake()
        except BaseException:
            self.sock.close()
            raise

    def _handshake(self) -> None:
        raw_key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {raw_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode("ascii"))
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket handshake closed unexpectedly by server.")
            resp += chunk
        header_block, self._buf = resp.split(b"\r\n\r\n", 1)
        status_line = header_block.split(b"\r\n", 1)[0].decode("latin1", errors="replace")
        if len(status_line.split()) < 2 or status_line.split()[1] != "101":
            raise ConnectionError(f"WebSocket handshake failed: {status_line}")
        headers = {}
        for line in header_block.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            headers[name.strip().lower()] = value.strip()
        expected = base64.b64encode(hashlib.sha1((raw_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest())
        if headers.get(b"sec-websocket-accept") != expected:
            raise ConnectionError("Invalid WebSocket Sec-WebSocket-Accept response")

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise ConnectionError("WebSocket connection closed by server.")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        fin_and_op = 0x80 | (opcode & 0x0F)
        mask_key = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", fin_and_op, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", fin_and_op, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", fin_and_op, 0x80 | 127, length)
        masked = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        self.sock.sendall(header + mask_key + masked)

    async def send(self, text: str) -> None:
        self._send_frame(0x1, text.encode("utf-8"))

    async def recv(self) -> str:
        fragments: List[bytes] = []
        while True:
            b1, b2 = struct.unpack("!BB", self._recv_exact(2))
            fin = bool(b1 & 0x80)
            opcode = b1 & 0x0F
            masked = bool(b2 & 0x80)
            length = b2 & 0x7F
            if length == 126:
                (length,) = struct.unpack("!H", self._recv_exact(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._recv_exact(8))
            mask_key = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length)
            if masked:
                payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))

            if opcode == 0x8:  # Close
                raise ConnectionError("Server closed WebSocket connection.")
            if opcode == 0x9:  # Ping -> Pong
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:  # Pong
                continue
            if opcode in (0x0, 0x1, 0x2):
                fragments.append(payload)
                if fin:
                    return b"".join(fragments).decode("utf-8", errors="replace")

    async def __aenter__(self) -> "_StdlibWebSocket":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            self._send_frame(0x8, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


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
    endpoint = f"{ws_url.rstrip('/')}/ws/chat/{quote(user_id, safe='')}"
    print(BANNER)
    print(f"[Client] Connecting to {endpoint} (user_id='{user_id}')...")

    try:
        import websockets  # type: ignore
        ws_ctx = websockets.connect(endpoint, ping_interval=25)
    except ImportError:
        ws_ctx = _StdlibWebSocket(endpoint)

    async with ws_ctx as ws:
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
                    user_input = (await asyncio.to_thread(input, f"\nYou ({user_id}) > ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[Client] Session closed.")
                    break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit"):
                print("[Client] Saving session and exiting...")
                saved = False
                try:
                    await ws.send(json.dumps({"prompt": "/save"}))
                    while True:
                        raw_ack = await ws.recv()
                        ack = json.loads(raw_ack)
                        if ack.get("type") in ("ttt_status", "error"):
                            saved = ack.get("type") == "ttt_status" and bool(ack.get("result", {}).get("persona_saved"))
                            break
                except Exception:
                    pass
                print("[Client] Persona state saved. Goodbye!" if saved else "[Client] Save was not confirmed. Check the server before closing.")
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
    from engine.personalizer import uTTTPersonalizer

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
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[Client] Session closed. Goodbye!")
            return
        except Exception as exc:
            try:
                import torch as _  # noqa: F401
                has_local_torch = True
            except ImportError:
                has_local_torch = False

            if has_local_torch:
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
            else:
                print(
                    f"\n[Client Error] Could not connect to WebSocket server at {args.ws_url} ({exc}).\n"
                    "Please make sure the remote server & SSH tunnel are running in another terminal:\n"
                    f"  ./deploy/remote_ctl.sh serve --user-id {args.user_id}\n"
                )
                sys.exit(1)


if __name__ == "__main__":
    main()

"""A failed remote connection must never create or load a local model."""
import asyncio
import contextlib
import io
import sys
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, patch

import client_chat as client


def immediate(coroutine):
    """Run mocked, already-complete IO without creating sockets or an event loop."""
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    else:
        raise AssertionError("Test WebSocket coroutine unexpectedly suspended")
    finally:
        coroutine.close()


class ClientConnectionModeTests(unittest.TestCase):
    def test_successful_remote_mode_keeps_requested_options_and_never_loads_local(self):
        arguments = ["client_chat.py", "--user-id", "Phova", "--ws-url", "ws://localhost:8010",
                     "--max-new-tokens", "256", "--temperature", "0.3"]
        with patch.object(sys, "argv", arguments), \
                patch.object(client, "run_websocket_session", new=AsyncMock()) as remote, \
                patch.object(client.asyncio, "run", side_effect=immediate), \
                patch.object(client, "run_local_session") as local:
            self.assertIsNone(client.main())
        remote.assert_awaited_once_with(ws_url="ws://localhost:8010", user_id="Phova", max_new_tokens=256,
                                        temperature=0.3, demo_prompts=None)
        local.assert_not_called()

    def test_remote_failure_exits_nonzero_even_when_torch_is_available(self):
        arguments = ["client_chat.py", "--ws-url", "ws://localhost:8010"]
        with patch.object(sys, "argv", arguments), \
                patch.dict(sys.modules, {"torch": ModuleType("torch")}), \
                patch.object(client, "run_websocket_session", new=AsyncMock(side_effect=ConnectionError("connection refused"))) as remote, \
                patch.object(client.asyncio, "run", side_effect=immediate), \
                patch.object(client, "run_local_session") as local, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as stopped:
                client.main()
        self.assertNotEqual(stopped.exception.code, 0)
        remote.assert_awaited_once()
        local.assert_not_called()
        self.assertIn("ws://localhost:8010", errors.getvalue())
        self.assertIn("connection refused", errors.getvalue())
        self.assertIn("SSH tunnel", errors.getvalue())

    def test_explicit_local_mode_still_forwards_model_and_generation_options(self):
        arguments = ["client_chat.py", "--local", "--user-id", "Phova", "--config", "chosen.yaml",
                     "--base-ckpt", "chosen.best.pt", "--personas-dir", "chosen_personas", "--max-new-tokens", "256",
                     "--temperature", "0.3", "--lightweight"]
        with patch.object(sys, "argv", arguments), \
                patch.object(client, "run_websocket_session", new=AsyncMock()) as remote, \
                patch.object(client.asyncio, "run") as run, \
                patch.object(client, "run_local_session") as local:
            self.assertIsNone(client.main())
        local.assert_called_once_with(user_id="Phova", config_path="chosen.yaml", base_ckpt="chosen.best.pt",
                                      personas_dir="chosen_personas", max_new_tokens=256, temperature=0.3,
                                      lightweight=True, demo_prompts=None)
        remote.assert_not_called()
        run.assert_not_called()

    def test_user_interrupt_and_cancellation_keep_graceful_exit(self):
        for interrupted in (KeyboardInterrupt, asyncio.CancelledError):
            with self.subTest(interrupted=interrupted), \
                    patch.object(sys, "argv", ["client_chat.py"]), \
                    patch.object(client, "run_websocket_session", new=AsyncMock(side_effect=interrupted)), \
                    patch.object(client.asyncio, "run", side_effect=immediate), \
                    patch.object(client, "run_local_session") as local, \
                    contextlib.redirect_stdout(io.StringIO()) as output, \
                    contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertIsNone(client.main())
            local.assert_not_called()
            self.assertIn("Session closed", output.getvalue())
            self.assertEqual(errors.getvalue(), "")


if __name__ == "__main__":
    unittest.main()

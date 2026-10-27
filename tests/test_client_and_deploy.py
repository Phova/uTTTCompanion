import asyncio
import base64
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from client_chat import _StdlibWebSocket


class ClientTests(unittest.TestCase):
    def test_wss_uses_tls_and_validates_handshake(self):
        key_bytes = b"a" * 16
        key = base64.b64encode(key_bytes).decode("ascii")
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        sock, tls = MagicMock(), MagicMock()
        tls.recv.return_value = b"HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n"
        context = MagicMock()
        context.wrap_socket.return_value = tls
        with patch("client_chat.socket.create_connection", return_value=sock), patch("client_chat.ssl.create_default_context", return_value=context), patch("client_chat.os.urandom", return_value=key_bytes):
            ws = _StdlibWebSocket("wss://example.com/ws/chat/alice")
            self.assertIs(ws.sock, tls)
            context.wrap_socket.assert_called_once_with(sock, server_hostname="example.com")
            tls.recv.return_value = b"HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: invalid\r\n\r\n"
            with self.assertRaisesRegex(ConnectionError, "Sec-WebSocket-Accept"):
                _StdlibWebSocket("wss://example.com/ws/chat/alice")
            tls.close.assert_called_once()


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class DeploymentTests(unittest.TestCase):
    def run_serve(self, health):
        source = (Path(__file__).resolve().parents[1] / "deploy" / "remote_ctl.sh").read_text()
        # Load real functions, replacing only external effects with deterministic mocks.
        source = source.rsplit('main "$@"', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "remote_ctl.sh"
            script.write_text(source)
            harness = f'''
source "{script}"
validate_config() {{ REMOTE_TARGET=example; SSH_OPTS=(); }}
cmd_sync() {{ :; }}
remote_exec() {{
    case "$*" in
        *curl*) return {health} ;;
        *grep*) return 0 ;;
        *hostname*) echo 127.0.0.1 ;;
        *) return 0 ;;
    esac
}}
ssh() {{ return 23; }}
STARTUP_TIMEOUT=1
cmd_serve
'''
            return subprocess.run(["bash", "-c", harness], capture_output=True, text=True, timeout=10)

    def test_failed_startup_is_reported_immediately(self):
        result = self.run_serve(1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed during application startup", result.stderr)

    def test_tunnel_failure_is_not_reported_as_success(self):
        result = self.run_serve(0)
        self.assertEqual(result.returncode, 23)


if __name__ == "__main__":
    unittest.main()

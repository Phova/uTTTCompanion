"""Recent conversation is explicit, bounded and committed only on completion."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from client_chat import ConversationHistory, run_local_session, run_websocket_session
from engine.prompting import (MAX_HISTORY_CHARS, MAX_HISTORY_MESSAGE_CHARS, MAX_HISTORY_MESSAGES,
                              encode_chat_prompt, format_chat_prompt, validate_chat_history)


def pair(user, assistant):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def completed(prompt, reply="received"):
    return {"prompt": prompt, "reply": reply, "is_command": False, "status_bar": "done"}


class LiteralTokenizer:
    def encode(self, text, add_bos=True, add_eos=False):
        return (["BOS"] if add_bos else []) + list(text) + (["EOS"] if add_eos else [])


class HistoryPromptTests(unittest.TestCase):
    def test_complete_multiturn_matches_training_prefix_without_trailing_answer_space(self):
        history = pair(" I am Mei. ", " Hello, Mei! ") + pair("Keep it short.", "Understood.")
        expected = ("<|user|> I am Mei.\n<|assistant|> Hello, Mei!<|eos|>\n"
                    "<|user|> Keep it short.\n<|assistant|> Understood.<|eos|>\n"
                    "<|user|> Recommend breakfast.\n<|assistant|>")
        self.assertEqual(format_chat_prompt("Recommend breakfast.", history), expected)
        self.assertEqual(encode_chat_prompt(LiteralTokenizer(), "Recommend breakfast.", 1000, history),
                         ["BOS"] + list(expected))

    def test_empty_history_preserves_single_turn_tokens(self):
        tokenizer = LiteralTokenizer()
        expected = ["BOS"] + list("<|user|> hello 世界\n<|assistant|>")
        self.assertEqual(encode_chat_prompt(tokenizer, " hello 世界 ", 200), expected)
        self.assertEqual(encode_chat_prompt(tokenizer, " hello 世界 ", 200, []), expected)

    def test_budget_removes_whole_oldest_pair_and_keeps_current_turn(self):
        history = pair("obsolete" * 10, "old answer") + pair("recent question", "recent answer")
        expected = ("<|user|> recent question\n<|assistant|> recent answer<|eos|>\n"
                    "<|user|> now\n<|assistant|>")
        self.assertEqual(encode_chat_prompt(LiteralTokenizer(), "now", len(expected) + 1, history),
                         ["BOS"] + list(expected))

    def test_oversized_current_turn_uses_existing_single_turn_tail_truncation(self):
        tokenizer = LiteralTokenizer()
        prompt = "outdated " * 100 + "CURRENT QUESTION"
        budget = 70
        with_history = encode_chat_prompt(tokenizer, prompt, budget, pair("old", "answer"))
        self.assertEqual(with_history, encode_chat_prompt(tokenizer, prompt, budget))
        self.assertEqual(len(with_history), budget)
        self.assertTrue("".join(with_history[1:]).endswith("CURRENT QUESTION\n<|assistant|>"))

    def test_invalid_or_unbounded_history_and_reserved_markers_are_rejected(self):
        invalid = [None, (), pair("q", "a")[:1], [{"role": "system", "content": "x"}] * 2,
                   pair("", "a"), pair("q", 1), [{"role": "user", "content": "q", "extra": "x"},
                                                       {"role": "assistant", "content": "a"}],
                   pair("q", "a") * (MAX_HISTORY_MESSAGES // 2 + 1),
                   pair("q" * (MAX_HISTORY_MESSAGE_CHARS + 1), "a"),
                   pair("q" * MAX_HISTORY_MESSAGE_CHARS, "a" * MAX_HISTORY_MESSAGE_CHARS) * 3]
        for value in invalid:
            with self.subTest(history=str(value)[:100]), self.assertRaises(ValueError):
                validate_chat_history(value)
        for marker in ("<|user|>", "<|assistant|>", "<|eos|>", "<|bos|>", "<|pad|>", "<|unk|>"):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, "reserved template"):
                    format_chat_prompt("ordinary " + marker)
                with self.assertRaisesRegex(ValueError, "reserved template"):
                    validate_chat_history(pair("ordinary", marker))


class ClientHistoryTests(unittest.TestCase):
    def test_lightweight_client_does_not_import_torch_and_engine_exports_remain_available(self):
        script = '''
import sys, types
import client_chat
import engine
assert 'torch' not in sys.modules
assert set(engine.__all__) <= set(dir(engine))
checkpoint = types.ModuleType('engine.checkpoint_manager')
personalizer = types.ModuleType('engine.personalizer')
for name in engine.__all__:
    setattr(checkpoint if name == 'DualTrackCheckpointManager' else personalizer, name, object())
sys.modules['engine.checkpoint_manager'] = checkpoint
sys.modules['engine.personalizer'] = personalizer
from engine import DualTrackCheckpointManager, EvolutionResult, format_ttt_status_bar, uTTTPersonalizer
assert DualTrackCheckpointManager is checkpoint.DualTrackCheckpointManager
assert EvolutionResult is personalizer.EvolutionResult
assert format_ttt_status_bar is personalizer.format_ttt_status_bar
assert uTTTPersonalizer is personalizer.uTTTPersonalizer
try:
    engine.unknown_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown engine name was accepted')
'''
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_retention_is_pair_bounded_and_requests_cannot_mutate_local_state(self):
        history = ConversationHistory()
        for i in range(12):
            history.completed(str(i), completed(str(i)))
        self.assertEqual(len(history.messages), MAX_HISTORY_MESSAGES)
        self.assertEqual(history.messages[0]["content"], "4")
        request = history.request_messages()
        request[0]["content"] = "changed"
        self.assertEqual(history.messages[0]["content"], "4")
        for _ in range(4):
            prompt = "u" * MAX_HISTORY_MESSAGE_CHARS
            history.completed(prompt, completed(prompt, "a" * MAX_HISTORY_MESSAGE_CHARS))
        self.assertLessEqual(sum(len(x["content"]) for x in history.messages), MAX_HISTORY_CHARS)
        self.assertEqual(len(history.messages) % 2, 0)

    def test_only_confirmed_reset_clears_and_commands_never_enter_history(self):
        history = ConversationHistory()
        history.completed("hello", completed("hello"))
        expected = history.request_messages()
        history.completed("/reset", {"prompt": "/reset", "is_command": True, "command_name": "/reset", "persona_saved": False})
        history.completed("/save", {"prompt": "/save", "is_command": True, "persona_saved": True})
        history.completed("ignored", {"prompt": "different", "reply": "no", "is_command": False})
        self.assertEqual(history.messages, expected)
        history.completed("/reset", {"prompt": "/reset", "is_command": True, "command_name": "/reset", "persona_saved": True})
        self.assertEqual(history.messages, [])

    def test_remote_stream_errors_and_local_clear_do_not_become_conversation(self):
        class Socket:
            def __init__(self):
                self.sent = []
                self.events = [{"type": "session_ready"}]
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_):
                return False
            async def send(self, raw):
                payload = json.loads(raw)
                self.sent.append(copy.deepcopy(payload))
                prompt = payload["prompt"]
                if prompt == "failed":
                    self.events.extend([{"type": "token", "content": "partial"}, {"type": "error", "message": "failed"}])
                else:
                    result = (dict(prompt=prompt, reply="saved", is_command=True, command_name="/save", persona_saved=True,
                                   status_bar="saved") if prompt == "/save" else completed(prompt))
                    self.events.extend([{"type": "token", "content": "received"}, {"type": "ttt_status", "result": result}])
            async def recv(self):
                return json.dumps(self.events.pop(0))

        socket = Socket()
        module = SimpleNamespace(connect=mock.Mock(return_value=socket))
        with mock.patch.dict(sys.modules, {"websockets": module}), contextlib.redirect_stdout(io.StringIO()):
            # All mocked socket operations complete synchronously, so this
            # needs neither a network-capable event loop nor a real socket.
            coroutine = run_websocket_session("ws://localhost:8000", "alice",
                                                demo_prompts=["first", "failed", "/save", "next", "/clear", "new topic"])
            try:
                with self.assertRaises(StopIteration):
                    coroutine.send(None)
            finally:
                coroutine.close()
        self.assertEqual([x["prompt"] for x in socket.sent], ["first", "failed", "/save", "next", "new topic"])
        self.assertEqual(socket.sent[0]["history"], [])
        for index in (1, 2, 3):
            self.assertEqual(socket.sent[index]["history"], pair("first", "received"))
        self.assertEqual(socket.sent[4]["history"], [])

    def test_local_mode_uses_same_completion_reset_and_error_rules(self):
        calls = []
        def stream(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            prompt = kwargs["prompt"]
            if prompt == "failed":
                yield {"type": "token", "content": "partial"}
                raise ValueError("failure after partial generation")
            if prompt == "/reset":
                result = dict(prompt=prompt, is_command=True, command_name="/reset", persona_saved=True,
                              status_bar="reset")
            else:
                result = completed(prompt)
            yield {"type": "ttt_status", "result": result}
        engine = SimpleNamespace(stream_interact_and_evolve=stream, ckpt_manager=SimpleNamespace(
            compute_user_drift=lambda _: {"step": 0, "drift_pct": 0.0}))
        module = SimpleNamespace(uTTTPersonalizer=mock.Mock(return_value=engine))
        with mock.patch.dict(sys.modules, {"engine.personalizer": module}), contextlib.redirect_stdout(io.StringIO()):
            run_local_session("alice", lightweight=True,
                              demo_prompts=["first", "failed", "next", "/reset", "new topic", "/clear", "last"])
        self.assertEqual(calls[1]["history"], pair("first", "received"))
        self.assertEqual(calls[2]["history"], pair("first", "received"))
        self.assertEqual(calls[4]["history"], [])
        self.assertEqual(calls[5]["history"], [])


if __name__ == "__main__":
    unittest.main()

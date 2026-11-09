"""Optional local sampling streams survive HTTP and WebSocket request plumbing."""
import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError
import torch

from server.app import ChatRequest
import server.app as service
import tests.test_prompt_budget as prompt_tests


class SeedRequestValidationTests(TestCase):
    def test_seed_is_optional_and_rejects_lossy_integer_coercion(self):
        self.assertIsNone(ChatRequest(user_id="alice", prompt="hello").seed)
        for value in (0, 2 ** 63 - 1):
            self.assertEqual(ChatRequest(user_id="alice", prompt="hello", seed=value).seed, value)
        for value in (-1, 2 ** 63, True, 3.5, "42"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ChatRequest(user_id="alice", prompt="hello", seed=value)

    def test_personalizer_uses_local_generator_and_echoes_seed(self):
        engine = prompt_tests.PromptBudgetTests().engine()
        before = torch.random.get_rng_state().clone()
        events = list(engine.stream_interact_and_evolve("alice", "hello", seed=42))
        generator = engine.model.stream_generate.call_args.kwargs["generator"]
        self.assertEqual(generator.initial_seed(), 42)
        self.assertEqual(events[-1]["result"]["seed"], 42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_personalizer_bad_seed_does_not_load_persona(self):
        for seed in (-1, 2 ** 63, True, 3.5):
            engine = prompt_tests.PromptBudgetTests().engine()
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                list(engine.stream_interact_and_evolve("alice", "hello", seed=seed))
            engine.ckpt_manager.get_user_state.assert_not_called()


class SeedEndpointTests(IsolatedAsyncioTestCase):
    async def test_http_and_stream_forward_request_seed(self):
        engine = SimpleNamespace(interact_and_evolve=Mock(),
                                 stream_interact_and_evolve=Mock(return_value=(event for event in [{"type": "token", "content": "ok"}])))
        request = ChatRequest(user_id="alice", prompt="hello", seed=123)
        result = SimpleNamespace(to_dict=lambda: {"reply": "ok", "seed": 123})
        with patch.object(service, "get_personalizer", return_value=engine), \
                patch.object(service, "_run_engine", new=AsyncMock(return_value=result)) as run:
            self.assertEqual(await service.chat_endpoint(request), {"reply": "ok", "seed": 123})
            self.assertEqual(run.call_args.kwargs["seed"], 123)

        async def direct(operation, *args, **kwargs):
            return operation(*args, **kwargs)
        with patch.object(service.app.state, "engine_lock", asyncio.Lock(), create=True), \
                patch.object(service, "_on_engine_thread", side_effect=direct):
            events = [event async for event in service._stream_engine(engine, request)]
        self.assertEqual(events, [{"type": "token", "content": "ok"}])
        self.assertEqual(engine.stream_interact_and_evolve.call_args.kwargs["seed"], 123)


class HistoryRequestTests(IsolatedAsyncioTestCase):
    async def test_http_and_websocket_engine_receive_identical_explicit_history(self):
        history = [{"role": "user", "content": "I cannot eat chili."},
                   {"role": "assistant", "content": "Understood, no chili."}]
        request = ChatRequest(user_id="alice", prompt="Suggest dinner.", history=history)
        engine = SimpleNamespace(interact_and_evolve=Mock(),
                                 stream_interact_and_evolve=Mock(return_value=(event for event in [])))
        result = SimpleNamespace(to_dict=lambda: {"reply": "ok"})
        with patch.object(service, "get_personalizer", return_value=engine), \
                patch.object(service, "_run_engine", new=AsyncMock(return_value=result)) as run:
            await service.chat_endpoint(request)
            self.assertEqual(run.call_args.kwargs["history"], history)
        async def direct(operation, *args, **kwargs):
            return operation(*args, **kwargs)
        with patch.object(service.app.state, "engine_lock", asyncio.Lock(), create=True), \
                patch.object(service, "_on_engine_thread", side_effect=direct):
            self.assertEqual([event async for event in service._stream_engine(engine, request)], [])
        self.assertEqual(engine.stream_interact_and_evolve.call_args.kwargs["history"], history)

    async def test_request_defaults_and_history_validation_are_shared(self):
        first = ChatRequest(prompt="hello")
        first.history.append({"role": "user", "content": "not a complete pair"})
        self.assertEqual(ChatRequest(prompt="hello").history, [])
        invalid = [None, [{"role": "user", "content": "incomplete"}],
                   [{"role": "assistant", "content": "wrong role"}, {"role": "user", "content": "wrong role"}],
                   [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "<|eos|>"}],
                   [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 9]
        for history in invalid:
            with self.subTest(history=history), self.assertRaises(ValidationError):
                ChatRequest(prompt="hello", history=history)

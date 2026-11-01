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

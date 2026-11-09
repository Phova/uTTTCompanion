"""Real CPU service/write/persistence mechanism; no learned-recall claim.

The checkpoint's meta metadata enables the online path for a tiny random model.
Generation and analytical inner writes are real; no updater or state is mocked.
"""
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from fastapi.testclient import TestClient
    from engine.checkpoint_manager import DualTrackCheckpointManager
    from engine.personalizer import uTTTPersonalizer
    from engine.prompting import encode_chat_prompt
    from model.uttt_model import uTTTConfig, uTTTModel
    import server.app as service
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch, FastAPI and TestClient dependencies required")
class TTTServiceRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        previous_rng = torch.random.get_rng_state()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.addClassCleanup(torch.random.set_rng_state, previous_rng)
        torch.set_num_threads(2)

    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='ttt-service-roundtrip-', dir=cache)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        torch.manual_seed(20261125)
        self.config = uTTTConfig(
            name='uTTT-CPU-Service-Mechanism-Test', vocab_size=512, d_model=16,
            n_layers=2, n_heads=2, n_kv_heads=1, d_ff_fast=24,
            chunk_size=4, sliding_window_size=8, max_seq_len=192,
            dropout=0.0, gradient_checkpointing=False,
            surprise_threshold=0.0, elastic_anchor_lambda=0.0,
            max_weight_drift_ratio=10.0,
        )
        model = uTTTModel(self.config)
        self.base = self.root / 'base.pt'
        torch.save({'config': self.config.to_dict(), 'slow_weights': model.slow_state_dict(),
                    'init_fast_weights': model.init_fast_state_dict(), 'step': 1,
                    'metrics': {'training_stage': 'meta', 'inner_loop_protocol': model.INNER_LOOP_PROTOCOL,
                                'fixture_scope': 'mechanism_only_not_trained_recall'}}, self.base)
        self.base_bytes = self.base.read_bytes()
        self.manager = self.new_manager()
        self.engine = uTTTPersonalizer(
            self.manager, surprise_threshold=0.0, elastic_anchor_lambda=0.0,
            max_weight_drift_ratio=10.0, auto_save_every_turn=True,
            default_max_new_tokens=2, temperature=0.0, top_p=1.0,
        )
        self.model_parameters = {name: value.detach().clone() for name, value in self.manager.model.named_parameters()}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {'UTTT_NO_COMPILE': '1', 'UTTT_CUDA_GRAPH': '0',
                                                   'UTTT_PACKED_PROJECTIONS': '0', 'UTTT_DEFAULT_USER': ''}))
        # Only discovery is overridden to keep /health CPU-only even on a GPU
        # host. The real model, generation, updater and lifecycle are untouched.
        stack.enter_context(patch.object(service.torch.cuda, 'is_available', return_value=False))
        stack.enter_context(patch.object(service, '_personalizer', self.engine))
        self.client = stack.enter_context(TestClient(service.app))

    def new_manager(self):
        return DualTrackCheckpointManager(
            self.base, self.root / 'personas', device='cpu', dtype=torch.float32,
            max_gpu_users=2, max_cpu_users=2, auto_create_base_if_missing=False,
        )

    @staticmethod
    def snapshot(state):
        tensors = ('w_up', 'w_gate', 'w_down', 's_up', 's_gate', 's_down', 'ln_weight', 'ln_bias')
        return {**{name: getattr(state, name).detach().clone() for name in tensors},
                'step': state.step, 'last_surprise_loss': state.last_surprise_loss,
                'surprise_history': list(state.surprise_history)}

    def assert_exact_state(self, actual, expected):
        observed = self.snapshot(actual)
        self.assertEqual(observed.keys(), expected.keys())
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(observed[name], value, rtol=0, atol=0, msg=name)
            else:
                self.assertEqual(observed[name], value, name)

    def command(self, command):
        response = self.client.post('/v1/command', json={'user_id': 'alice', 'command': command})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result['is_command'])
        self.assertTrue(result['persona_saved'])
        return result

    def turn(self, prompt):
        response = self.client.post('/v1/chat', json={
            'user_id': 'alice', 'prompt': prompt, 'history': [],
            'max_new_tokens': 2, 'temperature': 0.0, 'top_p': 1.0, 'seed': 7,
        })
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result['gate_triggered'])
        self.assertTrue(result['persona_saved'])
        self.assertGreater(result['surprise_loss'], 0)
        return result

    def test_real_http_writes_isolate_users_and_commands_roundtrip_exact_state_and_momentum(self):
        health = self.client.get('/health')
        self.assertEqual(health.status_code, 200)
        self.assertIs(health.json()['online_ttt_enabled'], True)
        self.assertEqual(health.json()['vram_allocated_gb'], 0)

        # Allocate both users before writing, so shared/aliased cold states
        # cannot accidentally pass the isolation check.
        bob = self.manager.get_user_state('bob')
        bob_base = self.snapshot(bob)
        alice_base = self.snapshot(self.manager.get_user_state('alice'))
        for name in ('w_up', 'w_gate', 'w_down', 's_up', 's_gate', 's_down'):
            self.assertNotEqual(getattr(bob, name).data_ptr(), getattr(self.manager.get_user_state('alice'), name).data_ptr())

        first = self.turn('Remember my code AA.')
        state = self.manager.get_user_state('alice')
        self.assertEqual(state.w_up.device.type, 'cpu')
        self.assertGreater(state.step, 0)
        self.assertTrue(any(not torch.equal(getattr(state, name), alice_base[name])
                            for name in ('w_up', 'w_gate', 'w_down')))
        self.assertGreater(sum(float(getattr(state, name).abs().sum()) for name in ('s_up', 's_gate', 's_down')), 0)
        self.assert_exact_state(self.manager.get_user_state('bob'), bob_base)
        primary, _ = self.manager.get_persona_paths('alice')
        self.assertTrue(primary.is_file())
        persisted = torch.load(primary, map_location='cpu', weights_only=False)
        self.assertEqual(persisted['user_id'], 'alice')
        for name in ('w_up', 'w_gate', 'w_down', 's_up', 's_gate', 's_down'):
            torch.testing.assert_close(persisted['fast_state'][name], getattr(state, name), rtol=0, atol=0)

        # Observe a genuine model prefill without replacing generation. The
        # next request must contain only its own prompt, despite evolved W/S.
        prefills = []
        def observe_embeddings(module, args):
            prefills.append(args[0].detach().cpu().clone())
        # stream_generate invokes forward directly, so observe the real token
        # embedding module rather than a top-level Module.__call__ hook.
        hook = self.manager.model.tok_embeddings.register_forward_pre_hook(observe_embeddings)
        next_prompt = 'What is my code?'
        try:
            second = self.turn(next_prompt)
        finally:
            hook.remove()
        expected = encode_chat_prompt(self.engine.tokenizer, next_prompt, self.config.max_seq_len - 2, history=[])
        self.assertGreaterEqual(len(prefills), 1)
        self.assertEqual(prefills[0].tolist(), [expected])
        self.assertGreater(second['step'], first['step'])
        self.assert_exact_state(self.manager.get_user_state('bob'), bob_base)

        evolved = self.snapshot(self.manager.get_user_state('alice'))
        self.command('/save')
        self.command('/snapshot evolved')
        self.assert_exact_state(self.manager.get_user_state('alice'), evolved)
        fresh_manager = self.new_manager()
        self.assert_exact_state(fresh_manager.get_user_state('alice'), evolved)

        self.command('/reset')
        reset = self.manager.get_user_state('alice')
        self.assert_exact_state(reset, alice_base)
        self.assertEqual(reset.step, 0)
        self.assertTrue(all(torch.count_nonzero(getattr(reset, name)) == 0 for name in ('s_up', 's_gate', 's_down')))
        self.command('/restore evolved')
        self.assert_exact_state(self.manager.get_user_state('alice'), evolved)
        self.assert_exact_state(self.new_manager().get_user_state('alice'), evolved)
        self.assert_exact_state(self.manager.get_user_state('bob'), bob_base)
        for name, value in self.manager.model.named_parameters():
            torch.testing.assert_close(value, self.model_parameters[name], rtol=0, atol=0, msg=name)
        self.assertEqual(self.base.read_bytes(), self.base_bytes)


if __name__ == '__main__':
    unittest.main()

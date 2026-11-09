import tempfile
import unittest
from pathlib import Path
import torch
from engine.checkpoint_manager import DualTrackCheckpointManager
from engine.personalizer import uTTTPersonalizer
from model.uttt_model import uTTTConfig, uTTTModel


class PersonaMasterPrecisionTests(unittest.TestCase):
    def test_explicit_ttt_mode_requires_meta_and_can_disable_valid_meta(self):
        config=uTTTConfig(vocab_size=512,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,d_ff_fast=32)
        model=uTTTModel(config)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for stage in ('sft','meta'):
                path=root/(stage+'.pt')
                torch.save({'config':config.to_dict(),'slow_weights':model.slow_state_dict(),
                            'init_fast_weights':model.init_fast_state_dict(),
                            'metrics':{'training_stage':stage,'inner_loop_protocol':model.INNER_LOOP_PROTOCOL}},path)
                manager=DualTrackCheckpointManager(path,root/stage,device='cpu',auto_create_base_if_missing=False)
                self.assertEqual(uTTTPersonalizer(manager).online_updates_enabled,stage=='meta')
                self.assertEqual(uTTTPersonalizer(manager)._default_cuda_graph,'0' if stage=='meta' else '1')
                self.assertFalse(uTTTPersonalizer(manager,enable_online_ttt=False).online_updates_enabled)
                if stage=='meta':
                    self.assertTrue(uTTTPersonalizer(manager,enable_online_ttt=True).online_updates_enabled)
                else:
                    with self.assertRaisesRegex(ValueError,'meta-trained'):
                        uTTTPersonalizer(manager,enable_online_ttt=True)
                with self.assertRaisesRegex(ValueError,'boolean'):
                    uTTTPersonalizer(manager,enable_online_ttt=1)

    def test_meta_masters_survive_low_precision_reads_cold_reset_and_reload(self):
        config=uTTTConfig(vocab_size=64,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,
                          d_ff_fast=32,chunk_size=8,sliding_window_size=16,gradient_checkpointing=False)
        model=uTTTModel(config)
        with torch.no_grad():
            model.fast_weight_pool.w_up_init.fill_(0.1234567)
        original=model.init_fast_state_dict()
        for full in (False,True):
            with self.subTest(full=full),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path=root/'base.pt'
                payload={'config':config.to_dict(),'metrics':{'training_stage':'meta','inner_loop_protocol':model.INNER_LOOP_PROTOCOL}}
                payload.update({'model_state_dict':model.state_dict()} if full else
                               {'slow_weights':model.slow_state_dict(),'init_fast_weights':original})
                torch.save(payload,path)
                options=dict(base_checkpoint_path=path,personas_dir=root/'users',device='cpu',dtype=torch.bfloat16,
                             max_gpu_users=1,max_cpu_users=1,auto_create_base_if_missing=False)
                manager=DualTrackCheckpointManager(**options)
                self.assertEqual(manager.dtype,torch.bfloat16)
                for name,value in manager.model.fast_weight_pool.state_dict().items():
                    self.assertEqual(value.dtype,torch.float32)
                    torch.testing.assert_close(value,original[name],rtol=0,atol=0)
                alice=manager.get_user_state('alice')
                self.assertEqual(alice.w_up.dtype,torch.float32)
                self.assertEqual(alice.s_up.dtype,torch.float32)
                self.assertEqual(manager.compute_user_drift('alice')['drift_pct'],0.)
                alice.w_up.add_(0.000001)
                alice.step=1
                manager.save_user_state('alice')
                restored=DualTrackCheckpointManager(**options).get_user_state('alice')
                torch.testing.assert_close(alice.w_up,restored.w_up,rtol=0,atol=0)
                reset=manager.reset_user_state('alice',persist=False)
                torch.testing.assert_close(reset.w_up[0],original['w_up_init'],rtol=0,atol=0)
                self.assertEqual(reset.s_up.count_nonzero().item(),0)

    def test_language_only_read_precision_is_unchanged(self):
        config=uTTTConfig(vocab_size=64,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,d_ff_fast=32)
        model=uTTTModel(config)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'base.pt'
            torch.save({'config':config.to_dict(),'slow_weights':model.slow_state_dict(),
                        'init_fast_weights':model.init_fast_state_dict(),'metrics':{'training_stage':'sft'}},path)
            manager=DualTrackCheckpointManager(path,root/'users',device='cpu',dtype=torch.bfloat16,
                                              auto_create_base_if_missing=False)
            self.assertEqual(manager.get_user_state('alice').w_up.dtype,torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 required')
class MetaServingPrecisionCudaTests(unittest.TestCase):
    def test_activation_precision_projections_preserve_exact_writes_and_graph_reads(self):
        from engine.inference import CUDAGraphDecoder
        config=uTTTConfig(vocab_size=512,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,
                          d_ff_fast=64,chunk_size=8,sliding_window_size=8,max_seq_len=64,
                          gradient_checkpointing=False,dropout=0.)
        torch.manual_seed(167)
        reference=uTTTModel(config).cuda().float().eval().requires_grad_(False)
        with torch.no_grad():reference.fast_weight_pool.w_up_init.add_(0.000001)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'base.pt'
            torch.save({'config':config.to_dict(),'slow_weights':reference.slow_state_dict(),
                        'init_fast_weights':reference.init_fast_state_dict(),
                        'metrics':{'training_stage':'meta','inner_loop_protocol':reference.INNER_LOOP_PROTOCOL}},path)
            manager=DualTrackCheckpointManager(path,root/'users',device='cuda',dtype=torch.bfloat16,
                                              auto_create_base_if_missing=False)
            optimized=manager.model
            self.assertIs(optimized.tok_embeddings.weight,optimized.lm_head.weight)
            for name,value in optimized.named_parameters():
                self.assertEqual(value.dtype,torch.float32 if name.startswith('fast_weight_pool.') else torch.bfloat16)
            state=manager.get_user_state('alice')
            before=state.clone(detach=True)
            support=torch.tensor([[1,4,8,12,5,13,2]],device='cuda')
            query=torch.tensor([[1,4,11,5]],device='cuda')
            expected=state.clone(detach=True)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(2):
                    expected,_,_=reference.test_time_update(support,expected)
                    state,_,_=optimized.test_time_update(support,state)
                    for name in ('w_up','w_gate','w_down','s_up','s_gate','s_down','ln_weight','ln_bias'):
                        torch.testing.assert_close(getattr(state,name),getattr(expected,name),rtol=0,atol=0)
                    first=reference(query,fast_state=expected,update_fast_weights=False,logits_to_keep=1)
                    second=optimized(query,fast_state=state,update_fast_weights=False,logits_to_keep=1)
                    torch.testing.assert_close(second.logits,first.logits,rtol=0,atol=0)
                optimized.prepare_for_inference()
                prefill=optimized(query,fast_state=state,update_fast_weights=False,logits_to_keep=1)
                self.assertTrue(CUDAGraphDecoder.is_supported(optimized,state,prefill.kv_caches,4))
                decoder=CUDAGraphDecoder(optimized,state,prefill.kv_caches,4)
                caches=prefill.kv_caches
                for position in range(4,20):
                    token=torch.tensor([[10+position]],device='cuda')
                    eager=optimized(token,fast_state=state,kv_caches=caches,start_pos=position,
                                    update_fast_weights=False,logits_to_keep=1)
                    caches=eager.kv_caches
                    actual=decoder.step(token)
                    torch.testing.assert_close(actual.float(),eager.logits[:,-1].float(),atol=.015,rtol=.025)
                self.assertTrue(decoder.matches(optimized,state))
                state.w_gate.add_(1e-6)
                self.assertFalse(decoder.matches(optimized,state))
            torch.testing.assert_close(before.w_up[0],optimized.fast_weight_pool.w_up_init,rtol=0,atol=0)


if __name__=='__main__':unittest.main()

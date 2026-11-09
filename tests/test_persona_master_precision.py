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


if __name__=='__main__':unittest.main()

"""Real support/query optimizer probe with a tiny model and exact state accounting."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from deploy.probe_persona_meta import choose_episodes, run_episode_steps, storage_ledger
from model.uttt_model import SimpleUTF8Tokenizer, uTTTConfig, uTTTModel
from train_5090 import EpisodeMetaDataset


class PersonaMetaProbeTests(unittest.TestCase):
    def test_two_full_adam_steps_and_hard_episode_budget(self):
        previous=torch.get_num_threads();torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'train.jsonl'
                row={'write_text':'<|user|> My code is A.\n<|assistant|> Recorded.<|eos|>',
                     'query_text':'My code?','answer':'A','validation_group_id':'one',
                     'memory_task':{'split':'train'}}
                path.write_text(json.dumps(row),encoding='utf-8')
                dataset=EpisodeMetaDataset(SimpleUTF8Tokenizer(512),96,path)
                model=uTTTModel(uTTTConfig(vocab_size=512,d_model=8,n_layers=2,n_heads=2,n_kv_heads=1,
                    d_ff_fast=16,max_seq_len=128,chunk_size=96,gradient_checkpointing=True,
                    meta_checkpoint_layers=1,meta_checkpoint_reentrant=False)).float()
                with self.assertRaisesRegex(ValueError,'exceeds'):
                    choose_episodes(dataset,max_total_tokens=191)
                report=run_episode_steps(model,dataset,device='cpu',amp_dtype=torch.float32,max_total_tokens=192)
                self.assertEqual(report['status'],'ok')
                self.assertEqual(report['optimizer_steps'],[2])
                self.assertEqual(report['optimizer_state_parameters'],12)
                self.assertTrue(report['frozen_parameters_unchanged'])
                self.assertTrue(report['no_write_logits_bitwise_unchanged'])
                self.assertEqual(report['query_calls_checked_no_KV_or_support_text'],2)
        finally:
            torch.set_num_threads(previous)

    def test_max_model_persona_and_writer_storage_accounting(self):
        ledger=storage_ledger(2560,6912,34)
        self.assertEqual(ledger['writer_parameters'],445818948)
        self.assertEqual(ledger['writer_parameter_tensors'],204)
        self.assertEqual(ledger['persona_master_fp32_bytes'],424693760)


if __name__=='__main__':unittest.main()

"""Actual-BPE split identities, complete request boundaries and no-crop budgets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deploy.prepare_persona_memory import NEUTRAL, TEMPLATES, measure_episode, prepare, support_texts
from deploy.evaluate_persona_memory import load_episodes

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None


@unittest.skipIf(Tokenizer is None, "Actual BPE tokenizers dependency required")
class PersonaMemoryCurriculumTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / '.cache'
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.directory.name)
        backend = Tokenizer(models.BPE(unk_token='<|unk|>'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        texts = [text.format(entity='profile-ABCDEFGH', value='X7Z') for languages in TEMPLATES.values()
                 for templates in languages.values() for text in templates]
        texts += [text for values in NEUTRAL.values() for text in values] + ['已记录。 Recorded. READY']
        backend.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2048, special_tokens=[
            '<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>', '<|user|>', '<|assistant|>'],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        self.tokenizer = backend
        self.path = self.root / 'tokenizer.json'
        backend.save(str(self.path))

    def tearDown(self):
        self.directory.cleanup()

    def build(self, name='episodes', **changes):
        options = dict(train_episodes=32, dev_episodes=16, test_episodes=16, seq_len=128,
                       payload_length=3, interference_writes=1, seed=92)
        options.update(changes)
        output = self.root / name
        report = prepare(output, self.path, **options)
        rows = [json.loads(line) for filename in ('train.jsonl', 'test.jsonl')
                for line in (output / filename).read_text(encoding='utf-8').splitlines()]
        return output, report, rows

    def test_explicit_split_requests_have_independent_entities_values_and_prompt_families(self):
        output, report, rows = self.build()
        self.assertEqual(Counter(row['memory_task']['split'] for row in rows), {'train': 32, 'dev': 16, 'test': 16})
        self.assertEqual(report['files']['train.jsonl']['rows'], 48)
        self.assertEqual(report['files']['test.jsonl']['rows'], 16)
        identities = {}
        for split in ('train', 'dev', 'test'):
            items = [row for row in rows if row['memory_task']['split'] == split]
            identities[split] = ({row['memory_task']['entity'] for row in items},
                                 {value for row in items for value in (row['memory_task']['payload'], row['memory_task']['obsolete_payload']) if value},
                                 {row['memory_task']['prompt_family'] for row in items})
        for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
            self.assertTrue(all(not a & b for a, b in zip(identities[left], identities[right])))
        template_sets = {split: {text for templates in TEMPLATES[split].values() for text in templates} for split in TEMPLATES}
        self.assertFalse(template_sets['train'] & template_sets['test'])
        for row in rows:
            metadata = row['memory_task']
            self.assertEqual(row['family'], row['validation_group_id'])
            self.assertNotIn('<|', row['query_text'])
            self.assertNotIn(metadata['payload'].casefold(), row['query_text'].casefold())
            self.assertEqual(row['lengths'], measure_episode(row, self.tokenizer, 128))
            self.assertTrue(all(value <= 128 for value in row['lengths']['support_tokens']))
            for text in support_texts(row):
                self.assertEqual(text.count('<|user|>'), 1)
                self.assertEqual(text.count('<|assistant|>'), 1)
                self.assertEqual(text.count('<|eos|>'), 1)
            if metadata['task'] == 'correction':
                self.assertIn(metadata['obsolete_payload'], row['write_text'])
                self.assertIn(metadata['payload'], row['correction_text'])
                self.assertNotEqual(metadata['obsolete_payload'], metadata['payload'])
            expected = metadata['payload'] + ' READY' if metadata['task'] == 'reply_prefix' else metadata['payload']
            self.assertEqual(row['answer'], expected)
        checked, _, evidence = load_episodes(output / 'test.jsonl', self.tokenizer,
                                             hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(len(checked), 16)
        self.assertEqual(evidence['verified_split_intersections'], 0)

    def test_reproducible_bytes_and_short_payload_capacity_never_silently_duplicate(self):
        first, _, _ = self.build('one')
        second, _, _ = self.build('two')
        for name in ('train.jsonl', 'test.jsonl', 'manifest.json'):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        with self.assertRaisesRegex(ValueError, 'Not enough distinct short payloads'):
            self.build('exhausted', train_episodes=2000, payload_length=2)
        self.assertFalse((self.root / 'exhausted').exists())
        with self.assertRaises(FileExistsError):
            self.build('one')

    def test_actual_budget_rejects_crops_and_manifest_hash_cannot_hide_split_leakage(self):
        with self.assertRaisesRegex(ValueError, 'cropping'):
            self.build('too-small', seq_len=16)
        self.assertFalse((self.root / 'too-small').exists())
        output, _, rows = self.build()
        test = output / 'test.jsonl'
        test.write_text(test.read_text(encoding='utf-8') + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs from preparation'):
            load_episodes(test, self.tokenizer, hashlib.sha256(self.path.read_bytes()).hexdigest())
        # Even a self-consistent updated file hash cannot hide actual identity overlap.
        test_rows = [row for row in rows if row['memory_task']['split'] == 'test']
        train = next(row for row in rows if row['memory_task']['split'] == 'train')
        test_rows[0]['memory_task']['prompt_family'] = train['memory_task']['prompt_family']
        test.write_text('\n'.join(json.dumps(row) for row in test_rows), encoding='utf-8')
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        manifest['files']['test.jsonl']['sha256'] = hashlib.sha256(test.read_bytes()).hexdigest()
        (output / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'identities overlap'):
            load_episodes(test, self.tokenizer, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_shared_episode_dataset_encoding_agrees_with_actual_bpe_measurements(self):
        try:
            from model.tokenizer import BPETokenizer
            from train_5090 import EpisodeMetaDataset
        except ImportError:
            self.skipTest('Torch trainer dependency unavailable')
        output, _, rows = self.build()
        tokenizer = BPETokenizer(self.path.read_text(encoding='utf-8'), self.tokenizer.get_vocab_size())
        train = EpisodeMetaDataset(tokenizer, 128, output / 'train.jsonl', split='train')
        dev = EpisodeMetaDataset(tokenizer, 128, output / 'train.jsonl', split='dev')
        self.assertEqual((len(train), len(dev)), (32, 16))
        for row in rows:
            encoded = EpisodeMetaDataset.encode_row(tokenizer, row, 128)
            self.assertEqual(encoded['write_attention_mask'].sum(-1).tolist(), row['lengths']['support_tokens'])
            self.assertEqual(int(encoded['attention_mask'].sum()), row['lengths']['query_with_answer_tokens'])
            self.assertEqual(int((encoded['labels'] != -100).sum()), row['lengths']['answer_tokens_including_eos'])


if __name__ == '__main__':
    unittest.main()

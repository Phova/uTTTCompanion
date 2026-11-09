"""Causal recall controls must separate weights from earlier request contents."""
import copy
import unittest
from unittest.mock import patch

from deploy.evaluate_persona_memory import evaluate_with_adapter


class MemoryAdapter:
    def __init__(self, corrupt_persistence=False):
        self.reads=[]
        self.corrupt_persistence=corrupt_persistence
        self.save_reload_calls=0

    def base_state(self): return {'value':None}
    def clone(self,state): return copy.deepcopy(state)
    def equal(self,left,right): return left==right
    def write(self,state,text):
        state['value']=text.split('=')[1]
        return state
    def no_write(self,state,text): return state
    def read(self,state,query):
        self.reads.append(query)
        return state['value'] or 'UNKNOWN'
    def save_reload(self,state):
        self.save_reload_calls += 1
        return self.base_state() if self.corrupt_persistence else self.clone(state)


class ControlledReplyAdapter(MemoryAdapter):
    def __init__(self, replacements=None):
        super().__init__()
        self.replacements = replacements or {}

    def read(self, state, query):
        reply = super().read(state, query)
        return self.replacements.get(reply, reply)


class PersonaMemoryEvaluationTests(unittest.TestCase):
    def rows(self):
        return [{'episode_id':str(index),'write_text':'value='+old,'correction_text':'value='+value,
                 'query_text':'retrieve profile '+str(index),'answer':value,
                 'memory_task':{'entity':str(index),'task':'correction','language':'en',
                                'payload':value,'obsolete_payload':old}}
                for index,(old,value) in enumerate((('OLD1','NEW1'),('OLD2','NEW2')))]

    def test_only_correct_user_state_supplies_answer_and_replacement_survives_reload(self):
        rows=self.rows();adapter=MemoryAdapter()
        report=evaluate_with_adapter(rows,adapter,max_samples=2)
        summary=report['summary']
        self.assertEqual(summary['arms']['correct_write']['passes'],2)
        self.assertEqual(summary['arms']['save_reload']['passes'],2)
        for arm in ('base_w0','no_write','mismatched_state'):
            self.assertEqual(summary['arms'][arm]['passes'],0)
        self.assertEqual(summary['isolation_failures'],0)
        self.assertEqual(summary['persistence_failures'],0)
        self.assertTrue(all(row['correction']['before_exact_old'] and row['correction']['after_exact_new']
                            for row in report['results']))
        self.assertEqual(set(adapter.reads),{row['query_text'] for row in rows})
        self.assertEqual(report['counterfactual_controls'], {
            'total': 0, 'counterfactual_exact': 0, 'reply_changes_with_memory': 0, 'original_answer_leaked': 0})

    def test_corrupt_disk_state_is_detected_even_when_in_memory_recall_passes(self):
        report=evaluate_with_adapter(self.rows(),MemoryAdapter(corrupt_persistence=True),max_samples=2)
        self.assertEqual(report['summary']['arms']['correct_write']['passes'],2)
        self.assertEqual(report['summary']['persistence_failures'],2)
        self.assertEqual(report['summary']['arms']['save_reload']['passes'],0)

    def test_identical_query_must_follow_the_counterfactual_written_value(self):
        rows=self.rows();paired=copy.deepcopy(rows[0])
        paired['episode_id']='0-cf';paired['answer']='NEW3';paired['correction_text']='value=NEW3'
        paired['memory_task']['payload']='NEW3';rows.append(paired)
        report=evaluate_with_adapter(rows,MemoryAdapter(),max_samples=3)
        controls=report['counterfactual_controls']
        self.assertEqual(controls['total'],2)
        self.assertEqual(controls['counterfactual_exact'],2)
        self.assertEqual(controls['reply_changes_with_memory'],2)
        self.assertEqual(controls['original_answer_leaked'],0)

    def paired_rows(self):
        rows = self.rows()
        rows[0]['family'] = 'memory-pair:0'
        rows[0]['memory_task']['counterfactual_pair'] = rows[0]['family']
        paired = copy.deepcopy(rows[0])
        paired.update(episode_id='0-cf', answer='NEW3', correction_text='value=NEW3')
        paired['memory_task']['payload'] = 'NEW3'
        return rows + [paired]

    def test_one_sampled_member_still_requires_both_separately_written_answers(self):
        rows = self.paired_rows()
        with patch('deploy.evaluate_persona_memory.balanced_indices', return_value=[0]):
            report = evaluate_with_adapter(rows, ControlledReplyAdapter(), max_samples=1)
        controls = report['counterfactual_controls']
        self.assertEqual(controls['total'], 1)
        self.assertEqual(controls['unique_pairs'], 1)
        self.assertEqual(controls['both_members_exact_pairs'], 1)
        self.assertEqual(controls['both_members_exact_accuracy'], 1.0)
        self.assertEqual(controls['original_answer_leak_pairs'], 0)
        row = report['results'][0]
        self.assertTrue(row['arms']['correct_write']['exact_match'])
        self.assertTrue(row['counterfactual_control']['counterfactual_exact'])

    def test_changed_reply_with_one_wrong_member_is_not_pair_recall_success(self):
        report = evaluate_with_adapter(self.paired_rows(), ControlledReplyAdapter({'NEW3': 'WRONG'}), max_samples=3)
        controls = report['counterfactual_controls']
        self.assertEqual(controls['reply_changes_with_memory'], 2)
        self.assertEqual(controls['counterfactual_exact'], 1)
        self.assertEqual(controls['unique_pairs'], 1)
        self.assertEqual(controls['both_members_exact_pairs'], 0)
        self.assertEqual(controls['both_members_exact_accuracy'], 0.0)
        self.assertEqual(controls['original_answer_leak_pairs'], 0)

    def test_sampling_both_members_counts_one_successful_pair(self):
        report = evaluate_with_adapter(self.paired_rows(), ControlledReplyAdapter(), max_samples=3)
        controls = report['counterfactual_controls']
        self.assertEqual(controls['total'], 2)
        self.assertEqual(controls['unique_pairs'], 1)
        self.assertEqual(controls['both_members_exact_pairs'], 1)
        self.assertEqual(controls['both_members_exact_accuracy'], 1.0)
        paired_results = [row for row in report['results'] if 'counterfactual_control' in row]
        self.assertEqual(len({row['counterfactual_control']['pair_id'] for row in paired_results}), 1)

    def test_original_answer_leaks_are_deduplicated_per_pair(self):
        report = evaluate_with_adapter(self.paired_rows(),
                                       ControlledReplyAdapter({'NEW1': 'NEW3', 'NEW3': 'NEW1'}), max_samples=3)
        controls = report['counterfactual_controls']
        self.assertEqual(controls['original_answer_leaked'], 2)
        self.assertEqual(controls['original_answer_leak_pairs'], 1)
        self.assertEqual(controls['both_members_exact_pairs'], 0)

    def test_different_families_with_same_query_are_distinct_pairs(self):
        rows = self.paired_rows()
        rows[1]['query_text'] = rows[0]['query_text']
        rows[1]['family'] = 'memory-pair:1'
        rows[1]['memory_task']['counterfactual_pair'] = rows[1]['family']
        paired = copy.deepcopy(rows[1])
        paired.update(episode_id='1-cf', answer='NEW4', correction_text='value=NEW4')
        paired['memory_task']['payload'] = 'NEW4'
        rows.append(paired)
        report = evaluate_with_adapter(rows, ControlledReplyAdapter(), max_samples=4)
        controls = report['counterfactual_controls']
        self.assertEqual(controls['total'], 4)
        self.assertEqual(controls['unique_pairs'], 2)
        self.assertEqual(controls['both_members_exact_pairs'], 2)
        self.assertEqual(controls['both_members_exact_accuracy'], 1.0)

    def test_persistence_limit_does_real_io_for_only_selected_rows_without_reducing_recall(self):
        adapter = MemoryAdapter()
        report = evaluate_with_adapter(self.paired_rows(), adapter, max_samples=3, persistence_samples=1)
        self.assertEqual(adapter.save_reload_calls, 1)
        self.assertEqual(report['summary']['episodes'], 3)
        self.assertEqual(report['summary']['persistence_checks'], 1)
        self.assertEqual(report['summary']['persistence_failures'], 0)
        self.assertEqual(report['summary']['arms']['save_reload'], {'passes': 1, 'total': 1, 'accuracy': 1.0})
        self.assertEqual(report['summary']['arms']['correct_write'], {'passes': 3, 'total': 3, 'accuracy': 1.0})
        for arm in ('base_w0', 'no_write', 'mismatched_state'):
            self.assertEqual(report['summary']['arms'][arm]['total'], 3)
        for row in report['results'][1:]:
            self.assertNotIn('save_reload', row['arms'])
            self.assertNotIn('save_reload', row['replies'])
            self.assertIsNone(row['save_reload_state_exact'])
            self.assertIsNone(row['save_reload_reply_exact'])
        self.assertEqual(report['counterfactual_controls']['total'], 2)
        self.assertEqual(report['counterfactual_controls']['both_members_exact_pairs'], 1)
        self.assertEqual(report['persistence_sampling']['tested'], 1)

    def test_no_persistence_tests_have_null_accuracy_and_never_count_as_failure_or_success(self):
        adapter = MemoryAdapter(corrupt_persistence=True)
        rows = self.rows()
        rows[1]['memory_task']['language'] = 'zh'
        report = evaluate_with_adapter(rows, adapter, max_samples=2, persistence_samples=0)
        self.assertEqual(adapter.save_reload_calls, 0)
        self.assertEqual(report['summary']['persistence_checks'], 0)
        self.assertEqual(report['summary']['persistence_failures'], 0)
        self.assertEqual(report['summary']['arms']['save_reload'], {'passes': 0, 'total': 0, 'accuracy': None})
        self.assertEqual(report['summary']['arms']['correct_write']['total'], 2)
        for group in report['groups']['language'].values():
            self.assertEqual(group['arms']['save_reload']['total'], 0)
            self.assertIsNone(group['arms']['save_reload']['accuracy'])
            self.assertEqual(group['persistence_checks'], 0)
        for row in report['results']:
            self.assertNotIn('save_reload', row['arms'])
            self.assertIsNone(row['save_reload_state_exact'])
            self.assertIsNone(row['save_reload_reply_exact'])

    def test_only_explicit_persistence_failure_counts_and_limit_cannot_exceed_sample(self):
        adapter = MemoryAdapter(corrupt_persistence=True)
        report = evaluate_with_adapter(self.rows(), adapter, max_samples=2, persistence_samples=1)
        self.assertEqual(adapter.save_reload_calls, 1)
        self.assertEqual(report['summary']['persistence_checks'], 1)
        self.assertEqual(report['summary']['persistence_failures'], 1)
        self.assertEqual(report['summary']['arms']['save_reload'], {'passes': 0, 'total': 1, 'accuracy': 0.0})
        self.assertEqual(report['summary']['arms']['correct_write']['passes'], 2)
        adapter = MemoryAdapter()
        report = evaluate_with_adapter(self.rows(), adapter, max_samples=1, persistence_samples=20)
        self.assertEqual(adapter.save_reload_calls, 1)
        self.assertEqual(report['summary']['persistence_checks'], 1)

    def test_default_and_explicit_none_keep_all_row_persistence_and_nonpaired_metrics(self):
        default_adapter, explicit_adapter = MemoryAdapter(), MemoryAdapter()
        default = evaluate_with_adapter(self.rows(), default_adapter, max_samples=2)
        explicit = evaluate_with_adapter(self.rows(), explicit_adapter, max_samples=2, persistence_samples=None)
        self.assertEqual(default, explicit)
        self.assertEqual(default_adapter.save_reload_calls, 2)
        self.assertEqual(explicit_adapter.save_reload_calls, 2)
        self.assertEqual(default['summary']['persistence_checks'], 2)
        self.assertEqual(default['summary']['arms']['save_reload'], {'passes': 2, 'total': 2, 'accuracy': 1.0})
        self.assertNotIn('persistence_sampling', default)
        self.assertEqual(default['counterfactual_controls'], {
            'total': 0, 'counterfactual_exact': 0, 'reply_changes_with_memory': 0, 'original_answer_leaked': 0})
        for value in (-1, True, 1.5, '2'):
            adapter = MemoryAdapter()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonnegative integer or None'):
                evaluate_with_adapter(self.rows(), adapter, max_samples=2, persistence_samples=value)
            self.assertEqual(adapter.save_reload_calls, 0)


if __name__=='__main__':unittest.main()

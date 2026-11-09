"""Causal recall controls must separate weights from earlier request contents."""
import copy
import unittest

from deploy.evaluate_persona_memory import evaluate_with_adapter


class MemoryAdapter:
    def __init__(self, corrupt_persistence=False):
        self.reads=[]
        self.corrupt_persistence=corrupt_persistence

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
        return self.base_state() if self.corrupt_persistence else self.clone(state)


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

    def test_corrupt_disk_state_is_detected_even_when_in_memory_recall_passes(self):
        report=evaluate_with_adapter(self.rows(),MemoryAdapter(corrupt_persistence=True),max_samples=2)
        self.assertEqual(report['summary']['arms']['correct_write']['passes'],2)
        self.assertEqual(report['summary']['persistence_failures'],2)
        self.assertEqual(report['summary']['arms']['save_reload']['passes'],0)


if __name__=='__main__':unittest.main()

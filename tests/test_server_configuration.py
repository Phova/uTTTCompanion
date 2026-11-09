"""Explicit serving options must respect the selected persona storage volume."""
import os
import unittest
from unittest import mock


class ServerConfigurationTests(unittest.TestCase):
    def invoke(self,arguments):
        from server import app as module
        with mock.patch.dict(os.environ,{},clear=True),mock.patch('sys.argv',['server.app',*arguments]),mock.patch('uvicorn.run'):
            module.main()
            return dict(os.environ)

    def test_unspecified_persona_directory_defers_to_selected_config(self):
        environment=self.invoke(['--config','configs/5090_persona_meta_paired_20261109.yaml'])
        self.assertEqual(environment['UTTT_CONFIG'],'configs/5090_persona_meta_paired_20261109.yaml')
        self.assertNotIn('UTTT_PERSONAS_DIR',environment)

    def test_explicit_storage_and_ttt_generation_options_are_forwarded(self):
        environment=self.invoke(['--personas-dir','/scratch/meta','--enable-ttt','--no-cuda-graph',
                                 '--packed-projections','--no-compile'])
        self.assertEqual(environment['UTTT_PERSONAS_DIR'],'/scratch/meta')
        for name,value in (('UTTT_ONLINE_TTT','1'),('UTTT_CUDA_GRAPH','0'),
                           ('UTTT_PACKED_PROJECTIONS','1'),('UTTT_NO_COMPILE','1')):
            self.assertEqual(environment[name],value)


if __name__=='__main__':unittest.main()

"""Capacity probes must account for other processes under WSL/WDDM."""
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy.gpu_memory import GpuMemoryError, query_available_gpu_memory

try:
    from deploy import profile_capacity
except ImportError:
    profile_capacity = None


GPU_A = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
GPU_B = 'GPU-11234567-89ab-cdef-0123-456789abcdef'
GIB = 1024**3


class GpuMemoryGuardTests(unittest.TestCase):
    def query(self, runtime_free, output, **kwargs):
        runner = mock.Mock(return_value=SimpleNamespace(returncode=0, stdout=output))
        result = query_available_gpu_memory(runtime_free, runner=runner, **kwargs)
        return result, runner

    def test_runtime_optimism_is_limited_by_global_free_memory(self):
        result, _ = self.query(31 * GIB, f'0, {GPU_A}, 3072, 32768\n', device_uuid=GPU_A)
        self.assertEqual(result['free_bytes'], 3 * GIB)
        self.assertEqual(result['runtime_free_bytes'], 31 * GIB)
        self.assertEqual(result['global_free_bytes'], 3 * GIB)

    def test_global_optimism_is_limited_by_runtime_free_memory(self):
        result, _ = self.query(3 * GIB, f'0, {GPU_A}, 31744, 32768\n', device_uuid=GPU_A)
        self.assertEqual(result['free_bytes'], 3 * GIB)

    def test_uuid_selects_physical_gpu_without_assuming_logical_index_order(self):
        result, runner = self.query(4 * GIB, f'3, {GPU_B}, 2048, 32768\n', device_uuid=GPU_B, device_index=0,
                                    visible_devices=f'{GPU_B},{GPU_A}')
        self.assertEqual(result['device_uuid'], GPU_B)
        self.assertEqual(result['gpu_index'], 3)
        self.assertIn('--id=' + GPU_B, runner.call_args.args[0])
        self.assertFalse(runner.call_args.kwargs['shell'])

    def test_full_visibility_uuid_maps_a_device_without_properties_uuid(self):
        result, _ = self.query(10 * GIB, f'1, {GPU_B}, 8192, 32768\n', device_index=1,
                               visible_devices=f'{GPU_A},{GPU_B}')
        self.assertEqual(result['device_uuid'], GPU_B)

    def test_single_gpu_fallback_requires_matching_visible_target(self):
        for visible in (None, '0'):
            with self.subTest(visible=visible):
                result, _ = self.query(10 * GIB, f'0, {GPU_A}, 8192, 32768\n', visible_devices=visible)
                self.assertEqual(result['free_bytes'], 8 * GIB)
        for visible in ('1', '', '-1'):
            with self.subTest(visible=visible), self.assertRaises(GpuMemoryError):
                self.query(10 * GIB, f'0, {GPU_A}, 8192, 32768\n', visible_devices=visible)

    def test_multiple_gpus_without_uuid_are_not_guessed_from_numeric_visibility(self):
        inventory = f'0, {GPU_A}, 8192, 32768\n1, {GPU_B}, 1024, 32768\n'
        for visible in (None, '1,0'):
            with self.subTest(visible=visible), self.assertRaisesRegex(GpuMemoryError, 'ambiguous'):
                self.query(10 * GIB, inventory, visible_devices=visible)

    def test_mig_and_conflicting_mapping_are_refused(self):
        for arguments in ({'device_uuid': 'MIG-' + GPU_A[4:]},
                          {'device_uuid': GPU_A, 'visible_devices': 'MIG-' + GPU_A[4:]},
                          {'device_uuid': GPU_A, 'visible_devices': GPU_B}):
            with self.subTest(arguments=arguments), self.assertRaises(GpuMemoryError):
                self.query(10 * GIB, f'0, {GPU_A}, 8192, 32768\n', **arguments)

    def test_missing_or_failed_nvidia_smi_does_not_fall_back_to_runtime_only(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 5),
                      subprocess.CalledProcessError(1, 'nvidia-smi')):
            with self.subTest(error=type(error).__name__), self.assertRaisesRegex(GpuMemoryError, 'unavailable'):
                query_available_gpu_memory(31 * GIB, device_uuid=GPU_A, runner=mock.Mock(side_effect=error))

    def test_invalid_measurements_and_nonmatching_uuid_are_refused(self):
        for line in (f'0, {GPU_A}, N/A, 32768', f'0, {GPU_A}, 33000, 32768',
                     f'0, {GPU_A}, -1, 32768', f'0, {GPU_A}, 0, 0',
                     f'0, {GPU_B}, 8192, 32768', '', 'malformed'):
            with self.subTest(line=line), self.assertRaises(GpuMemoryError):
                self.query(31 * GIB, line, device_uuid=GPU_A)


@unittest.skipIf(profile_capacity is None, 'PyTorch profiling dependencies unavailable')
class CapacityWorkerGuardTests(unittest.TestCase):
    def test_worker_rechecks_global_free_before_constructing_a_large_model(self):
        memory = {'free_bytes': 3 * GIB, 'runtime_free_bytes': 31 * GIB,
                  'global_free_bytes': 3 * GIB}
        with mock.patch.object(profile_capacity, 'available_memory', return_value=memory), \
                mock.patch.object(profile_capacity.torch.cuda, 'empty_cache'), \
                mock.patch.object(profile_capacity, 'uTTTModel') as constructor:
            result = profile_capacity.measure(None, 1024, 28)
        constructor.assert_not_called()
        self.assertEqual(result['status'], 'insufficient_global_memory')
        self.assertEqual(result['available_memory'], memory)


if __name__ == '__main__':
    unittest.main()

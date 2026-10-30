"""Conservative CUDA capacity guard using independently reported global VRAM.

Some WSL/WDDM runtimes report nearly the entire GPU as free even while another
process occupies it. A capacity probe must use the smaller of CUDA's runtime
figure and nvidia-smi's global figure. These are snapshots, not reservations;
other workloads can still allocate memory after this check.
"""
import csv
from decimal import Decimal, InvalidOperation
import io
import re
import subprocess
import uuid


class GpuMemoryError(RuntimeError):
    """Global availability or the selected CUDA-to-physical mapping is unknown."""


def _gpu_uuid(value):
    if isinstance(value, bytes):
        if len(value) == 16:
            value = str(uuid.UUID(bytes=value))
        else:
            try:
                value = value.decode('ascii')
            except UnicodeDecodeError as error:
                raise GpuMemoryError('Invalid CUDA GPU UUID') from error
    text = str(value).strip()
    if text.upper().startswith('MIG-') or '/' in text:
        raise GpuMemoryError('MIG memory mapping is unsupported; refusing a parent-GPU capacity estimate')
    if text.upper().startswith('GPU-'):
        text = text[4:]
    try:
        return 'GPU-' + str(uuid.UUID(text))
    except (ValueError, AttributeError) as error:
        raise GpuMemoryError('A full GPU UUID is required for an unambiguous capacity estimate') from error


def _visible_target(visible_devices, device_index):
    if not isinstance(device_index, int) or device_index < 0:
        raise GpuMemoryError('CUDA device index must be a nonnegative integer')
    if visible_devices is None:
        return None
    targets = [target.strip() for target in visible_devices.split(',')]
    if device_index >= len(targets) or not targets[device_index]:
        raise GpuMemoryError('CUDA_VISIBLE_DEVICES does not identify the selected device')
    target = targets[device_index]
    if target.upper().startswith('MIG-') or '/' in target:
        raise GpuMemoryError('MIG memory mapping is unsupported; refusing a parent-GPU capacity estimate')
    return target


def _mib_bytes(value):
    if re.fullmatch(r'\d+(?:\.\d+)?', value) is None:
        raise GpuMemoryError('nvidia-smi did not provide a valid global VRAM measurement')
    try:
        return int(Decimal(value) * 1024**2)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise GpuMemoryError('Invalid global VRAM measurement') from error


def query_available_gpu_memory(runtime_free_bytes, *, device_uuid=None, device_index=0,
                               visible_devices=None, runner=None):
    """Return byte counts for one selected CUDA device, or refuse an unsafe guess.

    Pass the current CUDA device's properties.uuid when available, and its
    logical device index plus CUDA_VISIBLE_DEVICES. Without a CUDA UUID,
    a full UUID in the visibility list is usable; numeric fallback requires
    exactly one physical GPU with the matching index.
    """
    if not isinstance(runtime_free_bytes, int) or isinstance(runtime_free_bytes, bool) or runtime_free_bytes < 0:
        raise GpuMemoryError('CUDA runtime free memory must be a nonnegative byte count')
    target = _visible_target(visible_devices, device_index)
    selected_uuid = _gpu_uuid(device_uuid) if device_uuid is not None else None
    if target is not None and not target.isdecimal():
        visible_uuid = _gpu_uuid(target)
        if selected_uuid is not None and selected_uuid != visible_uuid:
            raise GpuMemoryError('CUDA properties and CUDA_VISIBLE_DEVICES disagree about the selected GPU')
        selected_uuid = visible_uuid
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total',
               '--format=csv,noheader,nounits']
    if selected_uuid is not None:
        command.append('--id=' + selected_uuid)
    try:
        result = (runner or subprocess.run)(command, capture_output=True, text=True,
                                            timeout=5, check=True, shell=False)
        if result.returncode != 0:
            raise GpuMemoryError('nvidia-smi could not confirm global VRAM availability')
    except (OSError, subprocess.SubprocessError) as error:
        raise GpuMemoryError('nvidia-smi is unavailable; cannot safely determine global free VRAM') from error
    rows = []
    for fields in csv.reader(io.StringIO(result.stdout)):
        if not fields or all(not field.strip() for field in fields):
            continue
        fields = [field.strip() for field in fields]
        if len(fields) != 4 or not fields[0].isdecimal():
            raise GpuMemoryError('Invalid nvidia-smi GPU inventory')
        physical_index = int(fields[0])
        physical_uuid = _gpu_uuid(fields[1])
        free, total = _mib_bytes(fields[2]), _mib_bytes(fields[3])
        if total <= 0 or free > total:
            raise GpuMemoryError('Inconsistent nvidia-smi global VRAM measurement')
        rows.append((physical_index, physical_uuid, free, total))
    if selected_uuid is not None:
        matching = [row for row in rows if row[1] == selected_uuid]
        if len(rows) != 1 or len(matching) != 1:
            raise GpuMemoryError('nvidia-smi could not uniquely match the selected CUDA GPU UUID')
        selected = matching[0]
    else:
        if len(rows) != 1:
            raise GpuMemoryError('Multiple or unknown GPUs without a CUDA UUID; refusing ambiguous capacity mapping')
        selected = rows[0]
        if (target is not None and int(target) != selected[0]) or (target is None and device_index != 0):
            raise GpuMemoryError('Single-GPU inventory does not match the selected CUDA device')
    index, physical_uuid, global_free, global_total = selected
    return {'free_bytes': min(runtime_free_bytes, global_free),
            'runtime_free_bytes': runtime_free_bytes, 'global_free_bytes': global_free,
            'global_total_bytes': global_total, 'device_uuid': physical_uuid,
            'gpu_index': index, 'global_source': 'nvidia-smi'}

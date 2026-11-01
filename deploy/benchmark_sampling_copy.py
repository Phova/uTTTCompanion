"""Measure one-copy sampling against its exact float().clone() baseline.

No model/checkpoint is loaded. Results isolate sampling overhead and must not be
reported as whole-model throughput. CUDA allocation is capped at 1 GiB.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import textwrap
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import model.uttt_model as model_code
from model.uttt_model import uTTTModel


def baseline_sampler():
    """Change only the working-copy instruction in the currently loaded sampler."""
    source = textwrap.dedent(inspect.getsource(uTTTModel._sample_next_token))
    candidate = 'work_logits = logits.to(dtype=torch.float32, copy=True)'
    if source.count(candidate) != 1:
        raise RuntimeError('Sampler source does not contain the expected one-copy candidate')
    source = source.replace('@staticmethod\n', '', 1).replace(candidate, 'work_logits = logits.float().clone()')
    namespace = vars(model_code).copy()
    exec(compile(source, '<sampling-copy-baseline>', 'exec'), namespace)
    return namespace['_sample_next_token']


def sampling_trajectory(function, logits, settings, seed, iterations):
    generator = torch.Generator(device=logits.device).manual_seed(seed)
    samples = [function(logits, generator=generator, **settings) for _ in range(iterations)]
    return torch.stack(samples), generator.get_state()


def timed(function, logits, settings, iterations):
    generator = torch.Generator(device=logits.device).manual_seed(91)
    start = end = None
    if logits.is_cuda:
        torch.cuda.synchronize(logits.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    begun = time.perf_counter()
    for _ in range(iterations):
        function(logits, generator=generator, **settings)
    if end is not None:
        end.record()
        end.synchronize()
    wall = (time.perf_counter() - begun) * 1e6 / iterations
    return {'wall_us_per_call': wall,
            'cuda_us_per_call': start.elapsed_time(end) * 1000 / iterations if start is not None else None}


def benchmark(device='cuda', dtype='bfloat16', vocabulary=32000, iterations=300, repeats=5):
    if not 4 <= vocabulary <= 131072 or not 1 <= iterations <= 10000 or not 1 <= repeats <= 20:
        raise ValueError('Benchmark dimensions/iterations exceed the bounded sampling probe')
    target = torch.device(device)
    memory = None
    if target.type == 'cuda':
        from deploy.gpu_memory import query_available_gpu_memory
        torch.cuda.init()
        index = target.index if target.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        memory = query_available_gpu_memory(free,
            device_uuid=getattr(torch.cuda.get_device_properties(index), 'uuid', None),
            device_index=index, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if memory['free_bytes'] < 512 * 1024**2:
            raise RuntimeError('Less than 512 MiB globally free; sampling benchmark refused')
        torch.cuda.set_per_process_memory_fraction(min(1., 1024**3 / total), index)
        torch.cuda.reset_peak_memory_stats(index)
    precision = getattr(torch, dtype)
    values = torch.randn((1, vocabulary), device=target, dtype=precision,
                         generator=torch.Generator(device=target).manual_seed(7))
    functions = {'baseline': baseline_sampler(), 'one_copy': uTTTModel._sample_next_token}
    cases = {
        'greedy': dict(temperature=0., top_p=.9, repetition_penalty=1.),
        'nucleus': dict(temperature=.8, top_p=.9, repetition_penalty=1.),
        'full_softmax': dict(temperature=.8, top_p=1., repetition_penalty=1.),
        'recent_penalty': dict(temperature=.8, top_p=.9, repetition_penalty=1.1,
                               recent_tokens=list(range(6, min(24, vocabulary)))),
    }
    rows = {}
    with torch.no_grad():
        for name, settings in cases.items():
            settings['active_vocab_limit'] = vocabulary
            expected, expected_rng = sampling_trajectory(functions['baseline'], values, settings, 42, 32)
            actual, actual_rng = sampling_trajectory(functions['one_copy'], values, settings, 42, 32)
            if not torch.equal(actual, expected) or not torch.equal(actual_rng, expected_rng):
                raise AssertionError(f'{name}: sample or RNG trajectory changed')
            for function in functions.values():
                sampling_trajectory(function, values, settings, 11, 20)
            measurements = {label: [] for label in functions}
            for repetition in range(repeats):
                order = ('baseline', 'one_copy') if repetition % 2 == 0 else ('one_copy', 'baseline')
                for label in order:
                    measurements[label].append(timed(functions[label], values, settings, iterations))
            summaries = {label: {metric: statistics.median(row[metric] for row in entries)
                                 if entries[0][metric] is not None else None
                                 for metric in entries[0]}
                         for label, entries in measurements.items()}
            rows[name] = {'samples_bitwise_equal': True, 'generator_states_equal': True,
                          'settings': settings, 'median': summaries, 'measurements': measurements}
    return {'scope': 'sampling only; no claim about whole-model throughput', 'torch': torch.__version__,
            'device': str(target), 'dtype': dtype, 'vocabulary': vocabulary,
            'iterations_per_measurement': iterations, 'repeats': repeats,
            'eliminated_fp32_copy_bytes_per_call': vocabulary * 4 if precision != torch.float32 else 0,
            'memory_guard': memory,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(target) if target.type == 'cuda' else None,
            'cases': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--vocabulary', type=int, default=32000)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = benchmark(args.device, args.dtype, args.vocabulary, args.iterations, args.repeats)
    content = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n', encoding='utf-8')
    print(content)

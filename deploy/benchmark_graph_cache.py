"""Isolated read-cache memory benchmark; never loads a production checkpoint.

Run from the new source tree on the GPU host. Baseline and optimized workers
run sequentially in separate processes so all CUDA allocations are released.
Latency on a shared training GPU is diagnostic, not a throughput guarantee.
"""
import argparse
from array import array
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline', type=Path, required=True, help='Baseline engine/inference.py from the comparison commit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--width', type=int, default=2560)
    parser.add_argument('--ff', type=int, default=6912)
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--prompt', type=int, default=48)
    parser.add_argument('--tokens', type=int, default=50)
    parser.add_argument('--worker', choices=('baseline', 'optimized'))
    parser.add_argument('--logits-output', type=Path)
    return parser.parse_args()


def worker(args):
    import torch
    sys.path.insert(0, str(args.workspace.resolve()))
    from model.uttt_model import uTTTConfig, uTTTModel
    if args.worker == 'baseline':
        spec = importlib.util.spec_from_file_location('inference_read_cache_baseline', args.baseline.resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Decoder = module.CUDAGraphDecoder
    else:
        from engine.inference import CUDAGraphDecoder as Decoder
    torch.set_num_threads(2)
    torch.manual_seed(1793)
    torch.cuda.init()
    runtime_free, total = torch.cuda.mem_get_info()
    from deploy.gpu_memory import query_available_gpu_memory
    device = torch.cuda.current_device()
    memory = query_available_gpu_memory(
        runtime_free, device_uuid=getattr(torch.cuda.get_device_properties(device), 'uuid', None),
        device_index=device, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    )
    free = memory['free_bytes']
    # About 1.0--1.4 GiB is expected for the default one-layer surrogate.
    # A size-dependent check below also covers custom matrix/context sizes.
    if free < 2.0 * 1024**3:
        raise RuntimeError(f'Only {free / 1024**3:.2f} GiB free; refusing concurrent benchmark')
    config = uTTTConfig(
        vocab_size=512, d_model=args.width, d_ff_fast=args.ff, n_layers=1,
        n_heads=args.width // 128, n_kv_heads=max(1, args.width // 512),
        chunk_size=16, sliding_window_size=args.window,
        max_seq_len=max(256, args.prompt + args.tokens + 1), gradient_checkpointing=False,
    )
    with torch.device('meta'):
        parameter_count = uTTTModel(config).num_parameters()['total']
    matrix_elements = args.width * args.ff
    estimated_peak = (parameter_count * 4 + matrix_elements * 40
                      + args.prompt * args.ff * 32
                      + config.n_heads * args.prompt * (args.window + args.prompt) * 4
                      + config.max_seq_len * 128 * 8)
    if estimated_peak + 512 * 1024**2 > free:
        raise RuntimeError('Insufficient global GPU headroom for the requested surrogate size')
    model = uTTTModel(config).to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    model.prepare_for_inference()
    state = model.init_fast_state(device='cuda', dtype=torch.float32, detach=True)
    state.w_up.add_(1e-5)
    prompt = torch.randint(6, config.vocab_size, (1, args.prompt), device='cuda')
    tokens = torch.randint(6, config.vocab_size, (args.tokens, 1, 1), device='cuda')
    inputs = list(tokens.unbind(0))
    torch.cuda.synchronize()
    before_prefill = torch.cuda.memory_allocated()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prefill = model(prompt, fast_state=state, update_fast_weights=False,
                        logits_to_keep=1, return_fast_state=False)
    caches = prefill.kv_caches
    del prefill
    torch.cuda.synchronize()
    after_prefill = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begun = time.perf_counter()
    decoder = Decoder(model, state, caches, args.prompt)
    torch.cuda.synchronize()
    construct_seconds = time.perf_counter() - begun
    after_graph = torch.cuda.memory_allocated()
    graph_peak = torch.cuda.max_memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for token in inputs:
        decoder.step(token)
    end.record()
    end.synchronize()
    decode_ms = start.elapsed_time(end)
    decoder.reset(caches, args.prompt)
    reference_caches = [(key.clone(), value.clone()) for key, value in caches]
    outputs, eager_outputs = [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, token in enumerate(inputs):
            reference = model(token, fast_state=state, kv_caches=reference_caches,
                              start_pos=args.prompt + index, update_fast_weights=False,
                              logits_to_keep=1, return_fast_state=False)
            reference_caches = reference.kv_caches
            actual = decoder.step(token)
            outputs.append(actual.clone())
            eager_outputs.append(reference.logits[:, -1].clone())
    graph_logits = torch.stack(outputs).float()
    eager_logits = torch.stack(eager_outputs).float()
    difference = graph_logits - eager_logits
    error_l2 = torch.linalg.vector_norm(difference).item()
    reference_l2 = torch.linalg.vector_norm(eager_logits).item()
    rms_error = difference.square().mean().sqrt().item()
    maximum_error = difference.abs().max().item()
    reference_maximum = eager_logits.abs().max().item()
    relative_l2 = error_l2 / max(reference_l2, 1e-12)
    bf16_epsilon = torch.finfo(torch.bfloat16).eps
    # A fixed ring/validity mask and a variable-length chronological SDPA call
    # can choose different attention kernels and reduction orders. At d2560,
    # pointwise relative errors near zero are not a useful acceptance criterion.
    # Keep a global BF16-scale sanity gate, report every metric, and require the
    # actual optimized-vs-baseline outputs to be BITWISE equal in the parent.
    # No production regression-test tolerance is changed by this benchmark.
    if (not torch.isfinite(graph_logits).all().item()
            or not torch.isfinite(eager_logits).all().item()
            or relative_l2 > 2 * bf16_epsilon
            or maximum_error > 2 * bf16_epsilon * max(1, reference_maximum)):
        raise AssertionError(f'Graph/eager global numerical check failed: relative_L2={relative_l2}, '
                             f'max_error={maximum_error}, reference_max={reference_maximum}')
    logits = graph_logits.cpu().reshape(-1).tolist()
    with args.logits_output.open('wb') as target:
        array('f', logits).tofile(target)
    result = {
        'implementation': args.worker, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'width': args.width, 'ff': args.ff, 'layers': 1, 'vocab': config.vocab_size,
        'window': args.window, 'prompt_tokens': args.prompt, 'decode_tokens': args.tokens,
        'persona_dtype': str(state.w_up.dtype), 'model_dtype': str(next(model.parameters()).dtype),
        'free_before_bytes': free, 'global_memory_before': memory, 'total_gpu_bytes': total,
        'before_prefill_allocated_bytes': before_prefill,
        'prefill_read_cache_increment_bytes': after_prefill - before_prefill,
        'after_prefill_allocated_bytes': after_prefill,
        'graph_retained_increment_bytes': after_graph - after_prefill,
        'graph_construction_peak_allocated_bytes': graph_peak,
        'graph_construction_seconds': construct_seconds,
        'decode_gpu_ms': decode_ms, 'decode_gpu_ms_per_token': decode_ms / args.tokens,
        'eager_logit_diagnostics': {
            'global_relative_l2_error': relative_l2,
            'absolute_rms_error': rms_error,
            'absolute_max_error': maximum_error,
            'reference_absolute_max': reference_maximum,
            'relative_l2_in_bf16_epsilon_units': relative_l2 / bf16_epsilon,
            'max_error_in_reference_max_scale_bf16_units': maximum_error / (bf16_epsilon * max(1, reference_maximum)),
            'global_bf16_sanity_check_passed': True,
            'sanity_bound_epsilon_units': 2,
        },
    }
    # Workers exit after releasing every reference; this is also explicit for
    # manual reuse under a debugger rather than relying only on process exit.
    del outputs, eager_outputs, graph_logits, eager_logits, difference, reference, reference_caches, actual
    del decoder, caches, model, state, inputs, tokens, prompt
    gc.collect()
    torch.cuda.empty_cache()
    print('READ_CACHE_RESULT ' + json.dumps(result), flush=True)


def main():
    args = arguments()
    if min(args.width, args.ff, args.window, args.prompt, args.tokens) < 1 or args.width % 128:
        raise ValueError('Positive dimensions are required; width must be a multiple of 128')
    if args.worker:
        worker(args)
        return
    results = []
    with tempfile.TemporaryDirectory(prefix='uttt_read_cache_bench_') as temporary:
        paths = []
        for mode in ('baseline', 'optimized'):
            path = Path(temporary) / (mode + '.f32')
            command = [sys.executable, str(Path(__file__).resolve()), '--workspace', str(args.workspace.resolve()),
                       '--baseline', str(args.baseline.resolve()), '--width', str(args.width), '--ff', str(args.ff),
                       '--window', str(args.window), '--prompt', str(args.prompt), '--tokens', str(args.tokens),
                       '--worker', mode, '--logits-output', str(path)]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       env={**os.environ, 'OMP_NUM_THREADS': '2'})
            if completed.returncode:
                raise RuntimeError(f'{mode} worker failed:\n{completed.stdout}\n{completed.stderr}')
            item = next(json.loads(line.split(' ', 1)[1]) for line in completed.stdout.splitlines()
                        if line.startswith('READ_CACHE_RESULT '))
            results.append(item)
            paths.append(path)
        observed = []
        for path in paths:
            values = array('f')
            with path.open('rb') as source:
                values.fromfile(source, path.stat().st_size // values.itemsize)
            observed.append(values)
        if len(observed[0]) != len(observed[1]):
            raise AssertionError('Baseline/optimized logit shapes differ')
        max_error = max(abs(left - right) for left, right in zip(*observed))
        bitwise_equal = paths[0].read_bytes() == paths[1].read_bytes()
        if not bitwise_equal:
            raise AssertionError(f'Baseline/optimized logits are not bitwise equal (max error {max_error}); '
                                 'no benchmark acceptance or performance claim is made')
    report = {
        'scope': 'one-layer surrogate with production-size shared fast weights; live training GPU shared',
        'baseline_commit': '1fe1f55', 'workers_sequential_isolated': True,
        'baseline_vs_optimized_logit_max_absolute_error': max_error,
        'baseline_vs_optimized_logits_bitwise_equal': bitwise_equal,
        'measured_graph_retained_bytes_saved': results[0]['graph_retained_increment_bytes'] - results[1]['graph_retained_increment_bytes'],
        'expected_duplicate_read_bytes_removed': args.width * args.ff * 2 * 5,
        'latency_caveat': 'Concurrent production training affects timing; memory/parity are the acceptance criteria.',
        'results': results,
    }
    content = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n')
    print(content)


if __name__ == '__main__':
    main()

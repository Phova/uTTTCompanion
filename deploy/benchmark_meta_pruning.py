"""Tiny isolated benchmark; never loads or changes a production checkpoint."""
import argparse
import json
import statistics
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.uttt_model import uTTTConfig, uTTTModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
parser.add_argument('--iterations', type=int, default=5)
args = parser.parse_args()
if args.iterations < 1:
    parser.error("iterations must be positive")
device = torch.device(args.device)
if device.type == 'cuda':
    from deploy.gpu_memory import query_available_gpu_memory
    current_device = torch.cuda.current_device()
    free, _ = torch.cuda.mem_get_info()
    memory = query_available_gpu_memory(
        free, device_uuid=getattr(torch.cuda.get_device_properties(current_device), 'uuid', None),
        device_index=current_device, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    )
    if memory['free_bytes'] < 1024**3:
        raise RuntimeError('Less than 1 GiB global GPU headroom; refusing the microbenchmark')
torch.set_num_threads(2)
torch.manual_seed(345)
config = uTTTConfig(vocab_size=512, d_model=128, d_ff_fast=256, n_layers=4,
                    n_heads=4, n_kv_heads=2, chunk_size=16, sliding_window_size=64,
                    gradient_checkpointing=False, max_inner_grad_norm=1.0)
model = uTTTModel(config).to(device).train()
ids = torch.randint(6, 512, (1, 128), device=device)
timings = {'old_zero_aux_graph': [], 'pruned_zero_aux_graph': []}
gradients = {}

for iteration in range(args.iterations + 2):
    # Alternate which path runs first to reduce drift from the live GPU workload.
    paths = list(timings)
    if iteration % 2:
        paths.reverse()
    for label in paths:
        model.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            result = model(ids, labels=ids, surprise_aux_weight=0.0, return_fast_state=False)
            loss = result.ce_loss + 0.0 * result.surprise_loss if label == 'old_zero_aux_graph' else result.loss
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if iteration >= 2:
            timings[label].append(elapsed)
        gradients[label] = {name: parameter.grad.detach().float().cpu().clone()
                            for name, parameter in model.named_parameters() if parameter.grad is not None}
        del result, loss

assert gradients['old_zero_aux_graph'].keys() == gradients['pruned_zero_aux_graph'].keys()
for name in gradients['old_zero_aux_graph']:
    torch.testing.assert_close(gradients['pruned_zero_aux_graph'][name], gradients['old_zero_aux_graph'][name],
                               rtol=.01 if device.type == 'cuda' else 1e-5,
                               atol=1e-4 if device.type == 'cuda' else 1e-7, msg=name)
medians = {name: statistics.median(values) for name, values in timings.items()}
print(json.dumps({'device': str(device), 'config': config.to_dict(), 'iterations': args.iterations,
                  'median_seconds': medians,
                  'backward_step_speedup': medians['old_zero_aux_graph'] / medians['pruned_zero_aux_graph'],
                  'gradient_parity': True, 'isolated_microbenchmark': True}, indent=2))

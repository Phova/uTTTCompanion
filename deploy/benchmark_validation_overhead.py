"""CPU validation comparison and a small, real inner/outer gradient probe."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trainer", type=Path, required=True, help="Trusted baseline train_5090.py, exported from Git")
    parser.add_argument("--output", type=Path, default=Path("logs/validation_overhead.json"))
    args = parser.parse_args()
    import torch
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import train_5090 as current
    from model.uttt_model import uTTTConfig, uTTTModel

    spec = importlib.util.spec_from_file_location("old_validation_trainer", args.baseline_trainer.resolve())
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    torch.set_num_threads(2)
    torch.manual_seed(921)
    config = uTTTConfig(name="CPU-validation-D128-L2", vocab_size=512, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                       d_ff_fast=256, chunk_size=16, sliding_window_size=64,
                       max_seq_len=256, gradient_checkpointing=False)
    model = uTTTModel(config).train()
    batches = []
    for _ in range(32):
        ids = torch.randint(6, 272, (64,))
        batches.append({"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)})
    results = {}
    for name, function in (("baseline", baseline.evaluate_language_model), ("optimized", current.evaluate_language_model)):
        with patch.object(model.fast_weight_pool, "init_state", wraps=model.fast_weight_pool.init_state) as init, \
                patch.object(model.fast_weight_pool, "readonly_state", wraps=model.fast_weight_pool.readonly_state) as readonly:
            value = function(model, batches, torch.device("cpu"), torch.float32, max_batches=32)
            init_calls = init.call_count
            readonly_calls = readonly.call_count
        for _ in range(2):
            function(model, batches, torch.device("cpu"), torch.float32, max_batches=32)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            function(model, batches, torch.device("cpu"), torch.float32, max_batches=32)
            timings.append(time.perf_counter() - start)
        results[name] = {"validation_ce": value["validation_ce"], "init_state_calls": init_calls,
                         "readonly_state_calls": readonly_calls,
                         "median_seconds": statistics.median(timings), "times_seconds": timings}
    assert abs(results["baseline"]["validation_ce"] - results["optimized"]["validation_ce"]) < 1e-7
    assert results["baseline"]["init_state_calls"] == 32 and results["optimized"]["init_state_calls"] == 0

    torch.manual_seed(70)
    probe = uTTTModel(uTTTConfig(vocab_size=512, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1,
                                d_ff_fast=32, chunk_size=4, sliding_window_size=4,
                                max_seq_len=64, gradient_checkpointing=False)).train()
    ids = torch.randint(6, 272, (1, 12))
    gradient_probe = {}
    for enabled in (False, True):
        probe.zero_grad(set_to_none=True)
        output = probe(ids, labels=ids, update_fast_weights=enabled, surprise_aux_weight=0.0, return_fast_state=False)
        output.loss.backward()
        writers = {name: (None if parameter.grad is None else parameter.grad.float().norm().item())
                   for name, parameter in probe.named_parameters()
                   if any(part in name for part in (".w_k.", ".w_v.", ".w_eta.", ".w_alpha."))}
        gradient_probe[str(enabled)] = {"ce": output.ce_loss.item(), "total": output.loss.item(),
                                       "inner_surprise": output.surprise_loss.item(), "writer_grad_norms": writers}
        assert output.loss.item() == output.ce_loss.item()
        if enabled:
            assert all(value is not None and value > 0 for value in writers.values()), writers
        else:
            assert all(value is None for value in writers.values()), writers

    report = {"scope": "CPU small model; validation allocation overhead only, no RTX 5090 throughput claim",
              "torch": torch.__version__, "validation_model": config.to_dict(), "validation": results,
              "inner_outer_probe": gradient_probe}
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

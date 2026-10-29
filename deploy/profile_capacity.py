"""Measure full AdamW training capacity in isolated subprocesses, without saving weights."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.uttt_model import uTTTConfig, uTTTModel


def candidate_config(width, layers, seq_len=1024):
    heads = width // 128
    return uTTTConfig(
        name=f"uTTT-{width}x{layers}", d_model=width, n_layers=layers,
        n_heads=heads, n_kv_heads=heads // 4,
        d_ff_fast=((width * 8 // 3 + 255) // 256) * 256,
        chunk_size=128, sliding_window_size=512, max_seq_len=max(4096, seq_len),
        gradient_checkpointing=True,
    )


def measure(config, seq_len, budget_gb, steps=2):
    torch.cuda.empty_cache()
    # Reserve space for other live processes and reject candidates before allocation.
    with torch.device("meta"):
        count = uTTTModel(config).num_parameters()["total"]
    if count * 16 / 1024**3 + 1.5 > budget_gb:
        return {"status": "static_budget_exceeded", "parameters": count}
    torch.manual_seed(42)
    model = uTTTModel(config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ids = torch.randint(16, 21000, (1, seq_len), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, labels=ids, update_fast_weights=True, surprise_aux_weight=0.0)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        del out
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    return {"status": "ok" if reserved <= budget_gb else "over_budget", "parameters": count,
            "peak_allocated_gb": peak, "peak_reserved_gb": reserved,
            "seq_len": seq_len, "batch_size": 1, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--budget-gb", type=float, default=29.0)
    parser.add_argument("--output", default="logs/capacity_report.json")
    parser.add_argument("--worker", nargs=2, type=int, metavar=("WIDTH", "LAYERS"))
    args = parser.parse_args()
    if args.worker:
        try:
            result = measure(candidate_config(*args.worker, args.seq_len), args.seq_len, args.budget_gb)
        except torch.OutOfMemoryError:
            result = {"status": "oom"}
        print("CAPACITY_RESULT " + json.dumps(result), flush=True)
        return
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    free, _ = torch.cuda.mem_get_info()
    budget = min(args.budget_gb, free / 1024**3 - 1.5)
    if budget < 4:
        parser.error("Not enough free GPU memory; stop other GPU workloads first")
    candidates = [(1536, 24), (2048, 24), (2560, 24), (2560, 28), (2560, 32), (2560, 33), (2560, 34), (2560, 35), (3072, 24), (3072, 28)]
    reports = []
    for width, layers in candidates:
        completed = subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--worker", str(width), str(layers),
            "--seq-len", str(args.seq_len), "--budget-gb", str(budget),
        ], capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
        result = next((json.loads(line.removeprefix("CAPACITY_RESULT "))
                       for line in completed.stdout.splitlines() if line.startswith("CAPACITY_RESULT ")), None)
        if result is None:
            raise RuntimeError(completed.stdout + completed.stderr)
        result.update(width=width, layers=layers)
        reports.append(result)
        print(json.dumps(result), flush=True)
    passed = [result for result in reports if result["status"] == "ok"]
    report = {"budget_gb": budget, "candidates": reports,
              "largest_tested": max(passed, key=lambda result: result["parameters"]) if passed else None}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print("LARGEST_TESTED " + json.dumps(report["largest_tested"]), flush=True)


if __name__ == "__main__":
    main()

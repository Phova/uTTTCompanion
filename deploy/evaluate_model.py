"""Evaluate actual frozen-memory generation, independently of training loss."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.checkpoint_manager import DualTrackCheckpointManager
from model.tokenizer import build_tokenizer


DEFAULT_TASKS = [
    {"prompt": "Hello"},
    {"prompt": "翻译成中文，只输出译文：hello", "accepted": ["你好", "您好"]},
    {"prompt": "1+1等于多少？只输出数字。", "accepted": ["2"]},
    {"prompt": "请只回答：你好", "accepted": ["你好"]},
    {"prompt": "Translate to English, output only the translation: 你好", "accepted": ["Hello", "Hi"]},
]


def evaluate(checkpoint, tasks, output, max_new_tokens=64, device=None):
    with tempfile.TemporaryDirectory(prefix="uttt_quality_") as personas:
        manager = DualTrackCheckpointManager(checkpoint, personas, device=device, auto_create_base_if_missing=False)
        tokenizer = build_tokenizer(manager.config)
        results = []
        for task in tasks:
            prompt = task["prompt"]
            ids = tokenizer.encode(f"<|user|> {prompt}\n<|assistant|> ", add_bos=True)
            if len(ids) + max_new_tokens > manager.config.max_seq_len:
                raise ValueError("Evaluation prompt exceeds model context budget")
            with torch.autocast(manager.device.type, dtype=manager.dtype, enabled=manager.dtype != torch.float32):
                tokens = list(manager.model.stream_generate(
                    torch.tensor([ids], device=manager.device),
                    manager.model.init_fast_state(device=manager.device, dtype=manager.dtype, detach=True),
                    max_new_tokens=max_new_tokens, temperature=0.0,
                ))
            reply = tokenizer.decode(tokens).strip()
            result = {"prompt": prompt, "reply": reply, "generated_tokens": len(tokens),
                      "eos_generated": tokenizer.EOS_ID in tokens}
            if "accepted" in task:
                result["exact_match"] = reply.casefold().strip("。.!！ \n") in [value.casefold() for value in task["accepted"]]
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        judged = [result for result in results if "exact_match" in result]
        report = {"checkpoint": str(checkpoint), "parameters": manager.model.num_parameters(),
                  "training_metrics": manager.training_metadata,
                  "basic_instruction_matches": sum(result["exact_match"] for result in judged),
                  "basic_instruction_total": len(judged), "results": results}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", help="JSONL with prompt and optional accepted strings")
    parser.add_argument("--output", default="logs/quality_report.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()] if args.tasks else DEFAULT_TASKS
    evaluate(args.checkpoint, tasks, args.output, args.max_new_tokens, args.device)

"""Evaluate actual frozen-memory generation, independently of training loss."""
import argparse
from contextlib import contextmanager
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.checkpoint_manager import DualTrackCheckpointManager
from model.tokenizer import build_tokenizer
from deploy.training_diagnostics import generation_summary, repeated_ngram_continuations, repetition_metrics


DEFAULT_TASKS = [
    {"prompt": "Hello"},
    {"prompt": "翻译成中文，只输出译文：hello", "accepted": ["你好", "您好"]},
    {"prompt": "1+1等于多少？只输出数字。", "accepted": ["2"]},
    {"prompt": "请只回答：你好", "accepted": ["你好"]},
    {"prompt": "Translate to English, output only the translation: 你好", "accepted": ["Hello", "Hi"]},
]


DEFAULT_CONTINUATIONS = [
    {"prompt": "The purpose of scientific research is", "mode": "continuation"},
    {"prompt": "To prepare a simple meal, first", "mode": "continuation"},
    {"prompt": "科学研究的重要意义在于", "mode": "continuation"},
    {"prompt": "为了养成良好的学习习惯，我们可以", "mode": "continuation"},
]


@contextmanager
def decoding_policy(model, profile, eos_id):
    """Apply a labeled evaluation policy and restore the original sampler."""
    sentinel = object()
    previous = model.__dict__.get("_sample_next_token", sentinel)
    original = model._sample_next_token
    trace = {}

    def sample(logits, **settings):
        settings["repetition_penalty"] = profile["repetition_penalty"]
        size = profile["no_repeat_ngram_size"]
        if size:
            blocked = repeated_ngram_continuations(settings.get("recent_tokens") or [], size)
            blocked.discard(eos_id)
            if blocked:
                logits = logits.clone()
                logits[:, sorted(blocked)] = -torch.inf
        result = original(logits, **settings)
        trace["last_sampled"] = result
        return result

    model._sample_next_token = sample
    try:
        yield trace
    finally:
        if previous is sentinel:
            del model._sample_next_token
        else:
            model._sample_next_token = previous


def evaluation_profiles(sample_temperature, sample_top_p, controlled, controlled_penalty, controlled_ngram):
    if not 0 < sample_temperature <= 2 or not 0 < sample_top_p <= 1:
        raise ValueError("Sample temperature/top-p must be positive and within supported bounds")
    if controlled_penalty < 1 or controlled_ngram < 0:
        raise ValueError("Controlled penalty must be at least 1 and n-gram size must be nonnegative")
    profiles = [
        {"name": "raw_greedy", "temperature": 0.0, "top_p": 1.0,
         "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
        {"name": "raw_sampled", "temperature": sample_temperature, "top_p": sample_top_p,
         "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
    ]
    if controlled:
        profiles.append({"name": "controlled_sampled", "temperature": sample_temperature,
                         "top_p": sample_top_p, "repetition_penalty": controlled_penalty,
                         "no_repeat_ngram_size": controlled_ngram})
    return profiles


def evaluate(checkpoint, tasks, output, max_new_tokens=64, device=None, *, mode="auto", seed=42,
             sample_temperature=.8, sample_top_p=.95, controlled=False, controlled_penalty=1.05,
             controlled_ngram=4):
    if max_new_tokens < 1 or mode not in {"auto", "chat", "continuation"}:
        raise ValueError("Positive generation length and a supported evaluation mode are required")
    profiles = evaluation_profiles(sample_temperature, sample_top_p, controlled, controlled_penalty, controlled_ngram)
    with tempfile.TemporaryDirectory(prefix="uttt_quality_") as personas:
        manager = DualTrackCheckpointManager(checkpoint, personas, device=device, auto_create_base_if_missing=False)
        tokenizer = build_tokenizer(manager.config)
        stage = manager.training_metadata.get("training_stage", "unknown")
        default_mode = ("continuation" if stage == "pretrain" else "chat") if mode == "auto" else mode
        if tasks is None:
            tasks = DEFAULT_CONTINUATIONS if default_mode == "continuation" else DEFAULT_TASKS
        if not tasks:
            raise ValueError("At least one evaluation task is required")
        profile_results = {profile["name"]: [] for profile in profiles}
        devices = [manager.device.index if manager.device.index is not None else torch.cuda.current_device()] if manager.device.type == "cuda" else []
        for task_index, task in enumerate(tasks):
            prompt = task["prompt"]
            task_mode = task.get("mode", default_mode)
            if not isinstance(prompt, str) or task_mode not in {"chat", "continuation"}:
                raise ValueError("Each task requires a text prompt and a supported mode")
            encoded_prompt = f"<|user|> {prompt}\n<|assistant|> " if task_mode == "chat" else prompt
            ids = tokenizer.encode(encoded_prompt, add_bos=True)
            if len(ids) + max_new_tokens > manager.config.max_seq_len:
                raise ValueError("Evaluation prompt exceeds model context budget")
            for profile in profiles:
                sample_seed = int(seed) + task_index
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.random.default_generator.manual_seed(sample_seed)
                    if manager.device.type == "cuda":
                        with torch.cuda.device(manager.device):
                            torch.cuda.manual_seed(sample_seed)
                    with decoding_policy(manager.model, profile, tokenizer.EOS_ID) as trace:
                        with torch.autocast(manager.device.type, dtype=manager.dtype, enabled=manager.dtype != torch.float32):
                            tokens = list(manager.model.stream_generate(
                                torch.tensor([ids], device=manager.device),
                                manager.model.fast_weight_pool.readonly_state(device=manager.device, dtype=manager.dtype, detach=True),
                                max_new_tokens=max_new_tokens, temperature=profile["temperature"], top_p=profile["top_p"],
                                speculative_draft_k=0, use_cuda_graph=False,
                            ))
                last = int(trace["last_sampled"].item()) if "last_sampled" in trace else None
                eos = tokenizer.EOS_ID in tokens
                stop_reason = ("eos" if eos else "token_limit" if len(tokens) >= max_new_tokens
                               else "role_boundary" if last in (tokenizer.USER_ID, tokenizer.ASSISTANT_ID) else "stopped")
                reply = tokenizer.decode(tokens).strip()
                result = {"prompt": prompt, "mode": task_mode, "reply": reply, "generated_tokens": len(tokens),
                          "eos_generated": eos, "stop_reason": stop_reason,
                          "decoding": profile["name"], "decoding_parameters": dict(profile),
                          "seed": None if profile["name"] == "raw_greedy" else sample_seed,
                          **repetition_metrics(tokens, reply)}
                if "accepted" in task:
                    result["exact_match"] = reply.casefold().strip("。.!！ \n") in [value.casefold() for value in task["accepted"]]
                profile_results[profile["name"]].append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        results = profile_results["raw_greedy"]
        judged = [result for result in results if "exact_match" in result]
        report = {"checkpoint": str(checkpoint), "parameters": manager.model.num_parameters(),
                  "training_metrics": manager.training_metadata, "default_evaluation_mode": default_mode,
                  "basic_instruction_matches": sum(result["exact_match"] for result in judged),
                  "basic_instruction_total": len(judged), "results": results,
                  "profile_results": profile_results,
                  "decoding_summaries": {name: generation_summary(items) for name, items in profile_results.items()},
                  "notes": ["Primary results and exact-match totals use unpenalized raw greedy decoding.",
                            "Controlled decoding is optional and cannot establish that training repetition is fixed.",
                            "Continuation samples may reach the token budget before a document's EOS; inspect repetition separately."]}
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
    parser.add_argument("--mode", choices=("auto", "chat", "continuation"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-temperature", type=float, default=.8)
    parser.add_argument("--sample-top-p", type=float, default=.95)
    parser.add_argument("--controlled", action="store_true", help="Also report a separately labeled repetition-controlled sample")
    parser.add_argument("--controlled-penalty", type=float, default=1.05)
    parser.add_argument("--controlled-ngram", type=int, default=4)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()] if args.tasks else None
    evaluate(args.checkpoint, tasks, args.output, args.max_new_tokens, args.device, mode=args.mode, seed=args.seed,
             sample_temperature=args.sample_temperature, sample_top_p=args.sample_top_p,
             controlled=args.controlled, controlled_penalty=args.controlled_penalty, controlled_ngram=args.controlled_ngram)

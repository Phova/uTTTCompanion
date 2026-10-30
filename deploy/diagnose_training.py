"""Read training logs without loading a model or using GPU memory.

Reports measured CE trends and repetition heuristics separately. A falling CE
or low repetition does not establish instruction following or memory ability.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.training_diagnostics import repetition_metrics


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_STEP = re.compile(r"^\[Step\s+(\d+)/(\d+)\](.*)")
_VALIDATION = re.compile(rf"^\[Validation\s+(\d+)\]\s+CE=({_NUMBER})")
_START = re.compile(r"^\[Train\] Starting (pretrain|sft|meta)\b")
_STAGE = re.compile(r"\bStage:\s*(pretrain|sft|meta)\b")


def _field(line, name):
    match = re.search(rf"{re.escape(name)}\s*[:=]\s*({_NUMBER})", line)
    return float(match.group(1)) if match else None


def parse_training_log(lines, stage=None, recent=8):
    """Parse both legacy logs and logs with explicit stage/TTT diagnostics."""
    if stage not in (None, "pretrain", "sft", "meta") or recent < 1:
        raise ValueError("Invalid stage or recent-validation count")
    current_stage = stage or "unknown"
    steps, validations, samples = [], [], []
    nonfinite_rows = []
    malformed_samples = 0
    for raw in lines:
        line = raw.strip()
        start = _START.search(line)
        explicit_stage = _STAGE.search(line)
        if start:
            current_stage = start.group(1)
        elif explicit_stage:
            current_stage = explicit_stage.group(1)
        if re.search(r"(?:CE(?: Loss)?|Total Loss|Surprise Loss|GradNorm|Frozen CE)\s*[:=]\s*[+-]?(?:nan|inf(?:inity)?)\b", line, re.IGNORECASE):
            nonfinite_rows.append({"stage": current_stage, "line": line[:500]})
        step = _STEP.search(line)
        if step:
            speed = re.search(r"Speed:\s*([\d,]+(?:\.\d+)?)\s*tok/s", line)
            steps.append({
                "stage": current_stage, "step": int(step.group(1)), "target_steps": int(step.group(2)),
                "ce_loss": _field(line, "CE Loss"), "total_loss": _field(line, "Total Loss"),
                "surprise_loss": _field(line, "Surprise Loss"),
                "tokens_per_second": float(speed.group(1).replace(",", "")) if speed else None,
            })
        validation = _VALIDATION.search(line)
        if validation:
            tokens = re.search(r"Tokens seen=([\d,]+)", line)
            validations.append({
                "stage": current_stage, "step": int(validation.group(1)), "ce": float(validation.group(2)),
                "frozen_ce": _field(line, "Frozen CE"),
                "tokens_seen": int(tokens.group(1).replace(",", "")) if tokens else None,
            })
        marker = "[Validation sample] "
        if line.startswith(marker):
            try:
                sample = json.loads(line[len(marker):])
                if not isinstance(sample, dict) or not isinstance(sample.get("reply"), str):
                    raise ValueError("Missing sample reply")
            except (ValueError, TypeError):
                malformed_samples += 1
                continue
            # Legacy logs have no token IDs. Report their text metric only,
            # without inventing a token repetition score from characters.
            text_metrics = repetition_metrics([], sample["reply"])
            sample = {**sample, "text_4gram_repetition": text_metrics["text_4gram_repetition"],
                      "stage": current_stage,
                      "step": validations[-1]["step"] if validations else None}
            sample["repetition_flagged"] = (sample["text_4gram_repetition"] >= 0.35
                                             or sample.get("max_token_run", 0) >= 8)
            samples.append(sample)
    selected_stage = stage or current_stage
    selected_steps = [item for item in steps if item["stage"] == selected_stage]
    selected_validations = [item for item in validations if item["stage"] == selected_stage]
    selected_samples = [item for item in samples if item["stage"] == selected_stage]
    selected_nonfinite = [item for item in nonfinite_rows if item["stage"] == selected_stage]
    window = selected_validations[-recent:]
    latest = selected_validations[-1] if selected_validations else None
    ttt_enabled = selected_stage == "meta" if selected_stage != "unknown" else None
    notes = []
    if ttt_enabled is False:
        notes.append("TTT writes and surprise measurement are disabled for this language-training stage; legacy zero is a placeholder.")
    elif ttt_enabled is None:
        notes.append("Stage is unknown; pass --stage before interpreting a zero surprise value.")
    if len(window) > 1 and window[-1]["ce"] - window[0]["ce"] > 0.05:
        notes.append("CE increased over the recent validation window; inspect further before changing training.")
    if selected_nonfinite:
        notes.append("Non-finite loss/gradient metrics were logged; inspect training immediately.")
    if any(item["repetition_flagged"] for item in selected_samples[-recent:]):
        notes.append("Recent continuation samples contain repeated text; inspect both greedy and sampled outputs.")
    if selected_samples and not any(item.get("decoding") == "sampled" for item in selected_samples):
        notes.append("Samples are legacy/greedy-only; they do not distinguish greedy decoding loops from general generation quality.")
    if selected_stage == "pretrain":
        notes.append("Pretraining samples test text continuation. Instruction following requires separate SFT and task evaluation.")
    notes.append("This is a log-only report; it does not verify that the training process is alive.")
    return {
        "stage": selected_stage, "ttt_updates_enabled": ttt_enabled,
        "surprise_loss_measured": ttt_enabled, "latest_training": selected_steps[-1] if selected_steps else None,
        "validation": {
            "count": len(selected_validations), "first": selected_validations[0] if selected_validations else None,
            "latest": latest,
            "best": min(selected_validations, key=lambda item: item["ce"]) if selected_validations else None,
            "recent_ce_delta": window[-1]["ce"] - window[0]["ce"] if len(window) > 1 else None,
            "recent": window,
            "latest_perplexity": math.exp(latest["ce"]) if latest and latest["ce"] < 700 else None,
        },
        "samples": {"count": len(selected_samples), "repetition_flagged": sum(item["repetition_flagged"] for item in selected_samples),
                    "recent": selected_samples[-recent:], "malformed_rows": malformed_samples},
        "nonfinite_metrics": {"count": len(selected_nonfinite), "recent": selected_nonfinite[-recent:]},
        "notes": notes,
    }


def diagnose(log_path, stage=None, recent=8):
    path = Path(log_path)
    with path.open(encoding="utf-8", errors="replace") as stream:
        result = parse_training_log(stream, stage=stage, recent=recent)
    result.update(log_path=str(path.resolve()), log_age_seconds=max(0.0, time.time() - path.stat().st_mtime))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/pretrain_20261029.log")
    parser.add_argument("--stage", choices=("pretrain", "sft", "meta"))
    parser.add_argument("--recent", type=int, default=8)
    parser.add_argument("--output", help="Optional JSON report; the input log is never changed")
    args = parser.parse_args()
    if args.recent < 1:
        parser.error("--recent must be positive")
    report = diagnose(args.log, args.stage, args.recent)
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        if output.resolve() == Path(args.log).resolve():
            parser.error("--output must differ from the input log")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()

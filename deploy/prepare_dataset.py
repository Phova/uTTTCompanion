#!/usr/bin/env python3
"""
deploy/prepare_dataset.py
=========================
High-Speed Public Dataset Downloader & Formatter for uTTT (Universal Test-Time Training).

Downloads and formats real open-source Multi-Turn Dialogue & Persona datasets from
Alibaba Cloud ModelScope CDN (`https://www.modelscope.cn`) — with fallback to
`hf-mirror.com` / `huggingface.co` — into `data/uttt_persona_train.jsonl` using
standard `<|user|>` and `<|assistant|>` tags compatible with `train_5090.py --data-path`.

Primary Dataset Source (`AI-ModelScope/ShareGPT-Chinese-English-90k` on ModelScope):
- `sharegpt_jsonl/common_zh_70k.jsonl`   (70,000 Chinese multi-turn dialogues)
- `sharegpt_jsonl/computer_zh_26k.jsonl` (26,000 Chinese CS/programming multi-turn dialogues)
- `sharegpt_jsonl/common_en_70k.jsonl`   (70,000 English multi-turn dialogues)
- `sharegpt_jsonl/computer_en_26k.jsonl` (26,000 English CS/programming multi-turn dialogues)
Total capacity: > 192,000 multi-turn conversations streamed over domestic CDN.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Force IPv4 resolution first to avoid WSL2 broken IPv6 [Errno 101] Network is unreachable
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig_getaddrinfo(host, port, family, type, proto, flags)
    ipv4 = [r for r in res if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else res


socket.getaddrinfo = _ipv4_first_getaddrinfo


MODELSCOPE_BASE = (
    "https://www.modelscope.cn/api/v1/datasets/"
    "AI-ModelScope/ShareGPT-Chinese-English-90k/repo?Revision=master&FilePath="
)

MODELSCOPE_SPLITS: Dict[str, List[Tuple[str, float]]] = {
    "mixed": [
        ("sharegpt_jsonl/common_zh_70k.jsonl", 0.35),
        ("sharegpt_jsonl/computer_zh_26k.jsonl", 0.15),
        ("sharegpt_jsonl/common_en_70k.jsonl", 0.35),
        ("sharegpt_jsonl/computer_en_26k.jsonl", 0.15),
    ],
    "zh": [
        ("sharegpt_jsonl/common_zh_70k.jsonl", 0.73),
        ("sharegpt_jsonl/computer_zh_26k.jsonl", 0.27),
    ],
    "en": [
        ("sharegpt_jsonl/common_en_70k.jsonl", 0.73),
        ("sharegpt_jsonl/computer_en_26k.jsonl", 0.27),
    ],
    "code": [
        ("sharegpt_jsonl/computer_zh_26k.jsonl", 0.50),
        ("sharegpt_jsonl/computer_en_26k.jsonl", 0.50),
    ],
}


def format_sharegpt_conversation(obj: Dict[str, object]) -> Optional[str]:
    """
    Convert a ShareGPT JSON object (`{"conversation": [{"human": ..., "assistant": ...}]}`)
    into uTTT's `<|user|> ... \n<|assistant|> ...` sequence format.
    """
    conv = obj.get("conversation")
    if not isinstance(conv, list) or not conv:
        return None

    category = str(obj.get("category") or "").strip()
    lines: List[str] = []
    if category:
        lines.append(f"<|user|> [Context Category: {category}]")

    valid_turns = 0
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        human_msg = str(turn.get("human") or "").strip()
        assistant_msg = str(turn.get("assistant") or "").strip()
        if not human_msg or not assistant_msg:
            continue
        lines.append(f"<|user|> {human_msg}")
        lines.append(f"<|assistant|> {assistant_msg}")
        valid_turns += 1

    if valid_turns == 0:
        return None

    full_text = "\n".join(lines)
    return full_text if len(full_text) >= 100 else None


def stream_modelscope_split(
    file_path: str,
    target_count: int,
    timeout: int = 30,
) -> List[Dict[str, str]]:
    """
    Stream-download up to `target_count` multi-turn conversations from a ModelScope
    JSONL file over HTTP without buffering the entire archive in memory.
    """
    url = f"{MODELSCOPE_BASE}{file_path}"
    headers = {"User-Agent": "Mozilla/5.0 (uTTT-Dataset-Streamer/1.0)"}
    samples: List[Dict[str, str]] = []

    print(
        f"[Dataset] Streaming up to {target_count:,} dialogues from ModelScope: {file_path} ...",
        flush=True,
    )
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            if len(samples) >= target_count:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            formatted = format_sharegpt_conversation(obj)
            if formatted is not None:
                samples.append(
                    {
                        "source": f"AI-ModelScope/ShareGPT-Chinese-English-90k/{file_path}",
                        "text": formatted,
                    }
                )
                if len(samples) % 5000 == 0:
                    print(
                        f"  -> Progress ({file_path}): {len(samples):,} / {target_count:,} samples",
                        flush=True,
                    )

    print(
        f"[Dataset] Finished {file_path}: collected {len(samples):,} multi-turn dialogues.",
        flush=True,
    )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download & Format Public Multi-Turn Datasets for uTTT (ModelScope CDN)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["mixed", "zh", "en", "code", "personachat", "belle_zh"],
        default="mixed",
        help="Dataset preset: 'mixed' (ZH+EN General & CS), 'zh', 'en', or 'code'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Total number of multi-turn dialogue samples to prepare (up to 192,000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/uttt_persona_train.jsonl",
        help="Output JSONL path relative to project root.",
    )
    args = parser.parse_args()

    # Map legacy preset aliases cleanly
    preset_key = args.preset
    if preset_key == "personachat":
        preset_key = "en"
    elif preset_key == "belle_zh":
        preset_key = "zh"

    splits = MODELSCOPE_SPLITS[preset_key]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples: List[Dict[str, str]] = []
    remaining = max(1, int(args.max_samples))

    for idx, (file_path, ratio) in enumerate(splits):
        if remaining <= 0:
            break
        if idx == len(splits) - 1:
            quota = remaining
        else:
            quota = max(1, int(round(args.max_samples * ratio)))
            quota = min(quota, remaining)

        try:
            chunk = stream_modelscope_split(file_path=file_path, target_count=quota)
            all_samples.extend(chunk)
            remaining = args.max_samples - len(all_samples)
        except Exception as exc:
            print(f"[WARN] Error while streaming {file_path}: {exc}", flush=True)

    if not all_samples:
        raise RuntimeError("No samples were downloaded. Please check network connectivity.")

    random.seed(42)
    random.shuffle(all_samples)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"\n[Dataset Ready] Saved {len(all_samples):,} multi-turn dialogues -> "
        f"{out_path} ({size_mb:.2f} MB)",
        flush=True,
    )
    print(
        f"[Next Step] Train with this dataset using:\n"
        f"  ./deploy/remote_ctl.sh train --data-path {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

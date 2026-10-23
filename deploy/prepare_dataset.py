#!/usr/bin/env python3
"""
deploy/prepare_dataset.py
=========================
Public Dataset Downloader & Formatter for uTTT (Universal Test-Time Training).

Downloads and formats real open-source Persona & Multi-Turn Dialogue datasets
into `/root/uTTT/data/uttt_persona_train.jsonl` (using standard `<|user|>` and
`<|assistant|>` tags compatible with `train_5090.py --data-path`).

Supported Public Datasets:
1. `google/Synthetic-Persona-Chat` (English Persona-Memory Multi-Turn Dialogues):
   Each conversation begins with explicit User Persona facts followed by a 10-20 turn
   dialogue, directly training uTTT's SharedFastWeightPool to compress user profiles
   in early chunks and recall them in later chunks.
2. `BelleGroup/multiturn_chat_0.8M` (Chinese Multi-Turn Instruction & Dialogue):
   Stream-downloads the first N multi-turn Chinese dialogues for bilingual capability.
3. `mixed` (Default): Combines both datasets into a unified shuffled JSONL corpus.

Automatically uses `https://hf-mirror.com` for high-speed domestic downloads with
automatic fallback to `https://huggingface.co`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional


HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]


def open_url_with_fallback(rel_path: str, timeout: int = 30):
    """Open a dataset URL trying hf-mirror.com first, then huggingface.co."""
    headers = {"User-Agent": "uTTT-Dataset-Downloader/1.0"}
    last_err: Optional[Exception] = None
    for base in HF_MIRRORS:
        url = f"{base.rstrip('/')}/{rel_path.lstrip('/')}"
        try:
            print(f"[Dataset] Connecting to: {url} ...", flush=True)
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as exc:
            print(f"[Dataset] Mirror {base} failed ({exc}), trying next...", flush=True)
            last_err = exc
    raise RuntimeError(f"Failed to download {rel_path} from all mirrors: {last_err}")


def fetch_google_persona_chat(max_samples: int = 5000) -> List[Dict[str, str]]:
    """
    Download and format `google/Synthetic-Persona-Chat` (`Synthetic-Persona-Chat_train.csv`).
    """
    rel_path = "datasets/google/Synthetic-Persona-Chat/resolve/main/Synthetic-Persona-Chat_train.csv"
    samples: List[Dict[str, str]] = []

    with open_url_with_fallback(rel_path, timeout=60) as resp:
        text_stream = io.TextIOWrapper(resp, encoding="utf-8", errors="replace")
        reader = csv.DictReader(text_stream)
        for row in reader:
            persona_1 = (row.get("user 1 personas") or "").strip()
            persona_2 = (row.get("user 2 personas") or "").strip()
            conv = (row.get("Best Generated Conversation") or "").strip()
            if not conv:
                continue

            # Format persona header + multi-turn conversation with <|user|> / <|assistant|>
            formatted_lines: List[str] = []
            if persona_1:
                persona_clean = " ".join(persona_1.splitlines())
                formatted_lines.append(
                    f"<|user|> [User Persona Profile] {persona_clean}"
                )
                formatted_lines.append(
                    "<|assistant|> I have noted your personal profile and preferences in my fast weights."
                )

            for raw_turn in conv.splitlines():
                turn = raw_turn.strip()
                if not turn:
                    continue
                if turn.startswith("User 1:"):
                    msg = turn[len("User 1:") :].strip()
                    formatted_lines.append(f"<|user|> {msg}")
                elif turn.startswith("User 2:"):
                    msg = turn[len("User 2:") :].strip()
                    formatted_lines.append(f"<|assistant|> {msg}")
                else:
                    formatted_lines.append(turn)

            full_text = "\n".join(formatted_lines)
            if len(full_text) >= 128:
                samples.append(
                    {
                        "source": "google/Synthetic-Persona-Chat",
                        "text": full_text,
                    }
                )
            if len(samples) >= max_samples:
                break

    print(f"[Dataset] Loaded {len(samples)} samples from google/Synthetic-Persona-Chat.", flush=True)
    return samples


def fetch_belle_multiturn_zh(max_samples: int = 5000) -> List[Dict[str, str]]:
    """
    Stream-download the first `max_samples` Chinese multi-turn dialogues from
    `BelleGroup/multiturn_chat_0.8M` without needing to download the entire 800k file.
    """
    rel_path = "datasets/BelleGroup/multiturn_chat_0.8M/resolve/main/multiturn_chat_0.8M.json"
    samples: List[Dict[str, str]] = []

    with open_url_with_fallback(rel_path, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = (obj.get("instruction") or "").strip()
            output = (obj.get("output") or "").strip()
            if not instruction or not output:
                continue

            # Convert "Human: ... Assistant: ..." into "<|user|> ... <|assistant|> ..."
            conv_text = re.sub(r"\bHuman:\s*", "\n<|user|> ", instruction)
            conv_text = re.sub(r"\bAssistant:\s*", "\n<|assistant|> ", conv_text).strip()
            full_text = f"{conv_text}\n{output}\n"

            if len(full_text) >= 128:
                samples.append(
                    {
                        "source": "BelleGroup/multiturn_chat_0.8M",
                        "text": full_text,
                    }
                )
            if len(samples) >= max_samples:
                break

    print(f"[Dataset] Streamed {len(samples)} samples from BelleGroup/multiturn_chat_0.8M.", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & Format Public Persona Datasets for uTTT")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["mixed", "personachat", "belle_zh"],
        default="mixed",
        help="Dataset preset: 'mixed' (EN Persona-Chat + ZH Multi-Turn), 'personachat', or 'belle_zh'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8000,
        help="Total number of multi-turn dialogue samples to prepare.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/uttt_persona_train.jsonl",
        help="Output JSONL path relative to project root.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples: List[Dict[str, str]] = []
    if args.preset == "personachat":
        all_samples.extend(fetch_google_persona_chat(max_samples=args.max_samples))
    elif args.preset == "belle_zh":
        all_samples.extend(fetch_belle_multiturn_zh(max_samples=args.max_samples))
    else:
        half = max(1, args.max_samples // 2)
        try:
            all_samples.extend(fetch_google_persona_chat(max_samples=half))
        except Exception as exc:
            print(f"[WARN] Could not fetch google/Synthetic-Persona-Chat: {exc}")
        try:
            all_samples.extend(fetch_belle_multiturn_zh(max_samples=args.max_samples - len(all_samples)))
        except Exception as exc:
            print(f"[WARN] Could not fetch BelleGroup/multiturn_chat_0.8M: {exc}")

    if not all_samples:
        raise RuntimeError("No samples were downloaded. Please check network connectivity.")

    random.seed(42)
    random.shuffle(all_samples)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"\n[Dataset Ready] Saved {len(all_samples)} multi-turn persona dialogues -> "
        f"{out_path} ({size_mb:.2f} MB)"
    )
    print(
        f"[Next Step] Train with this dataset using:\n"
        f"  ./deploy/remote_ctl.sh train --data-path {out_path}"
    )


if __name__ == "__main__":
    main()

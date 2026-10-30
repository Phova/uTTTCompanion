"""Read-only, fixed-seed audit of repeated token targets in the packed corpus.

Uses CPU NumPy/tokenizers only; it neither loads a model nor changes corpus
files, labels, checkpoints, or training configuration.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tokenizers import Tokenizer
from deploy.corpus_repetition import analyze_repeated_spans, detected_language


def audit(manifest_path, *, samples=5000, seq_len=1024, seed=42, split="train",
          ngram_size=16, keep_occurrences=2, top_k=20):
    if samples < 1 or seq_len < 2 or top_k < 0:
        raise ValueError("samples/sequence length must be positive and top_k nonnegative")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "uttt-token-corpus-v1" or split not in ("train", "validation"):
        raise ValueError("Invalid token corpus or split")
    dtype = np.dtype(manifest["dtype"])
    if dtype not in (np.dtype("uint16"), np.dtype("uint32")):
        raise ValueError("Unsupported corpus dtype")
    info = manifest["splits"][split]
    path = (manifest_path.parent / info["file"]).resolve()
    tokenizer_path = (manifest_path.parent / manifest["tokenizer"]).resolve()
    if not path.is_relative_to(manifest_path.parent) or not tokenizer_path.is_relative_to(manifest_path.parent):
        raise ValueError("Corpus paths must remain within the manifest folder")
    token_count = int(info["tokens"])
    if path.stat().st_size != token_count * dtype.itemsize:
        raise ValueError("Corpus file size does not match manifest")
    size = (token_count - 1) // (seq_len - 1)
    if size < 1:
        raise ValueError("Corpus is shorter than a training window")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    corpus = np.memmap(path, mode="r", dtype=dtype)
    generator = np.random.default_rng(seed)
    indices = generator.choice(size, min(samples, size), replace=False)
    fractions, top = [], []
    total = Counter()
    groups = defaultdict(Counter)
    started = time.monotonic()
    for value in indices:
        index = int(value)
        start = index * (seq_len - 1)
        tokens = corpus[start:start + seq_len].tolist()
        analysis = analyze_repeated_spans(tokens, ngram_size, keep_occurrences)
        fraction = analysis["fraction"]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        language = detected_language(text)
        for counters in (total, groups[language]):
            counters["windows"] += 1
            counters["supervised_targets"] += analysis["supervised_targets"]
            counters["repeated_targets"] += analysis["repeated_targets"]
            counters["windows_with_any_repeated_targets"] += analysis["repeated_targets"] > 0
            for threshold in (.01, .05, .1, .25, .5):
                counters[f"windows_fraction_at_least_{threshold:g}"] += fraction >= threshold
        fractions.append(fraction)
        if top_k and analysis["repeated_targets"]:
            first = analysis["mask"].index(True)
            begin, end = max(0, first - 96), min(len(tokens), first + 192)
            top.append({"window_index": index, "token_offset": start, "detected_language": language,
                        "supervised_targets": analysis["supervised_targets"],
                        "repeated_targets": analysis["repeated_targets"], "fraction": fraction,
                        "longest_masked_run": analysis["longest_masked_run"],
                        "most_repeated_span_occurrences": analysis["most_repeated_span_occurrences"],
                        "repeated_span": tokenizer.decode(analysis["most_repeated_span"], skip_special_tokens=True),
                        "snippet_token_start": start + begin,
                        "snippet": tokenizer.decode(tokens[begin:end], skip_special_tokens=True)})
            top.sort(key=lambda item: (-item["fraction"], item["window_index"]))
            del top[top_k:]

    def summarize(counters):
        result = dict(counters)
        result["repeated_target_fraction"] = counters["repeated_targets"] / max(1, counters["supervised_targets"])
        return result

    return {"manifest": str(manifest_path), "corpus_fingerprint": manifest["fingerprint"],
            "split": split, "window_count": size, "requested_samples": samples,
            "samples": len(indices), "seq_len": seq_len, "seed": seed,
            "ngram_size": ngram_size, "keep_occurrences": keep_occurrences,
            "boundaries_reset_counts": [0, 1, 2, 3, 4, 5],
            "criterion": "targets ending third-or-later exact long-span occurrences within a protected-boundary segment; overlapping occurrences count",
            "summary": summarize(total),
            "fraction_percentiles": {str(percentile): float(np.percentile(fractions, percentile))
                                     for percentile in (0, 50, 90, 95, 99, 100)},
            "detected_language_groups": {name: summarize(value) for name, value in groups.items()},
            "top_repeated_windows": top,
            "source_attribution": "Packed binary files do not retain a per-document source index; offsets and script-based language groups are provided, not inferred source URLs.",
            "elapsed_seconds": time.monotonic() - started,
            "training_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--ngram-size", type=int, default=16)
    parser.add_argument("--keep-occurrences", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.manifest, samples=args.samples, seq_len=args.seq_len, seed=args.seed,
                       split=args.split, ngram_size=args.ngram_size, keep_occurrences=args.keep_occurrences, top_k=args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        # A report can never replace any existing file, especially corpus data.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
    print(content)


if __name__ == "__main__":
    main()

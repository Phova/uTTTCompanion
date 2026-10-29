"""Train a local, lossless byte-level BPE before training a NEW model."""
import argparse
import json
import sys
import hashlib
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.tokenizer import SPECIAL_TOKENS


def train_tokenizer(data_path, output_path, vocab_size=32000, max_samples=None, validation_fraction=0.02, split_seed=42):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if vocab_size < 262:
        raise ValueError("Byte-level BPE needs at least 256 byte tokens and 6 special tokens")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    samples_seen = 0

    def texts():
        nonlocal samples_seen
        seen = set()
        with open(data_path, encoding="utf-8") as source:
            for index, line in enumerate(source):
                if max_samples is not None and index >= max_samples:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line) if line.lstrip().startswith("{") else {"text": line}
                text = obj.get("text") or obj.get("content")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing text at line {index + 1}")
                text = re.sub(r"^<\|user\|> \[Context Category:[^\n]*\n", "", text).strip()
                identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if identity in seen:
                    continue
                seen.add(identity)
                if int(hashlib.sha256(f"{split_seed}:{identity}".encode()).hexdigest()[:16], 16) / 2**64 < validation_fraction:
                    continue
                samples_seen += 1
                yield text
    tokenizer.train_from_iterator(texts(), trainer=trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ))
    if not samples_seen:
        raise ValueError("No nonempty training samples remained after validation split")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    print(f"Saved tokenizer with {tokenizer.get_vocab_size():,} tokens to {output}")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    train_tokenizer(args.data_path, args.output, args.vocab_size, args.max_samples, args.validation_fraction, args.split_seed)

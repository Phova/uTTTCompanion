"""Clean, deduplicate, split, train BPE and tokenize downloaded bilingual shards."""
import argparse
import collections
import hashlib
import json
import os
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from model.tokenizer import SPECIAL_TOKENS, BPETokenizer
from engine.corpus import TokenizedCorpusDataset


def normalize(text):
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not 100 <= len(text) <= 100000 or text.count("\ufffd") > len(text) * 0.001:
        return None
    if sum(char.isalnum() for char in text) < len(text) * 0.3:
        return None
    return text


def identity_and_split(text, seed=42, validation_fraction=0.02):
    identity = hashlib.sha256(text.encode("utf-8")).digest()
    is_validation = int.from_bytes(hashlib.sha256(str(seed).encode() + b":" + identity).digest()[:8], "big") / 2**64 < validation_fraction
    return identity, "validation" if is_validation else "train"


def batches(root, sources):
    readers = []
    for source in sources:
        parquet = pq.ParquetFile(root / source["local_file"])
        names = parquet.schema_arrow.names
        columns = [name for name in ("text", "score", "int_score") if name in names]
        if "text" not in columns:
            raise ValueError(f"Missing text column: {source['local_file']}")
        readers.append((source, iter(parquet.iter_batches(batch_size=256, columns=columns))))
    while readers:
        following = []
        for source, reader in readers:
            try:
                batch = next(reader).to_pydict()
            except StopIteration:
                continue
            following.append((source, reader))
            yield source, batch
        readers = following


def clean_batch(source, batch):
    scores, integer_scores = batch.get("score"), batch.get("int_score")
    for index, raw in enumerate(batch["text"]):
        if source["language"] == "zh" and scores is not None and scores[index] < 0.75:
            continue
        if source["language"] == "en" and integer_scores is not None and integer_scores[index] < 3:
            continue
        text = normalize(raw)
        if text:
            yield text


def prepare(root, tokenizer_documents=100000, seed=42, validation_fraction=0.02):
    root = Path(root)
    sources = json.loads((root / "sources.json").read_text())["sources"]
    for entry in sources:
        path = root / entry["local_file"]
        if not path.exists() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Shard missing/incomplete: {path}")
    output = root / "tokenized"
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        existing = json.loads((output / "manifest.json").read_text())
        for split in ("train", "validation"):
            TokenizedCorpusDataset(output / "manifest.json", split=split)
        if hashlib.sha256((output / existing["tokenizer"]).read_bytes()).hexdigest() != existing["tokenizer_sha256"]:
            raise ValueError("Existing corpus tokenizer fingerprint does not match")
        print(f"CORPUS_ALREADY_PREPARED {output / 'manifest.json'}", flush=True)
        return output / "manifest.json"
    tokenizer_path = output / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        count = 0
        chars = 0
        seen = set()

        def sample():
            nonlocal count, chars
            for source, batch in batches(root, sources):
                for text in clean_batch(source, batch):
                    identity, split = identity_and_split(text, seed, validation_fraction)
                    if split != "train" or identity in seen:
                        continue
                    seen.add(identity)
                    count += 1
                    chars += len(text)
                    yield text
                    if count % 10000 == 0:
                        print(f"[BPE sample] {count:,} documents, {chars / 1e6:.1f}M characters", flush=True)
                    if count >= tokenizer_documents or chars >= 150000000:
                        return
        tokenizer.train_from_iterator(sample(), trainer=trainers.BpeTrainer(
            vocab_size=32000, min_frequency=3, special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ))
        if count < 1000:
            raise ValueError("Too few clean tokenizer training documents")
        tokenizer.save(str(tokenizer_path))
        print(f"[BPE complete] {tokenizer.get_vocab_size():,} tokens from {count:,} documents", flush=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    wrapped = BPETokenizer(tokenizer.to_str(), 32000)
    for sample in ("Hello, world!", "你好，世界！", "数学与科学 1+1=2", "é λ 🙂"):
        if wrapped.decode(wrapped.encode(sample)) != sample:
            raise ValueError("Tokenizer round trip failed")
    database = output / "dedup.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("DROP TABLE IF EXISTS documents")
    connection.execute("CREATE TABLE documents (digest BLOB PRIMARY KEY) WITHOUT ROWID")
    counters = collections.Counter()
    hashes = {split: hashlib.sha256() for split in ("train", "validation")}
    files = {split: (output / f"{split}.bin.tmp").open("wb", buffering=4 * 1024 * 1024) for split in hashes}
    started = time.monotonic()
    try:
        for batch_index, (source, batch) in enumerate(batches(root, sources)):
            texts, splits = [], []
            for text in clean_batch(source, batch):
                identity, split = identity_and_split(text, seed, validation_fraction)
                cursor = connection.execute("INSERT OR IGNORE INTO documents VALUES (?)", (identity,))
                if not cursor.rowcount:
                    counters["duplicates"] += 1
                    continue
                texts.append(text)
                splits.append(split)
                counters[f"{split}_documents"] += 1
                counters[f"{source['language']}_documents"] += 1
            if texts:
                encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
                for encoding, split in zip(encoded, splits):
                    values = np.asarray([1, *encoding.ids, 2], dtype="<u2")
                    data = values.tobytes()
                    files[split].write(data)
                    hashes[split].update(data)
                    counters[f"{split}_tokens"] += len(values)
                    counters[f"{source['language']}_tokens"] += len(values)
            if batch_index % 100 == 0:
                connection.commit()
                print(f"[Tokenizing] docs={counters['train_documents'] + counters['validation_documents']:,} tokens={counters['train_tokens'] + counters['validation_tokens']:,} elapsed={time.monotonic() - started:.0f}s", flush=True)
        connection.commit()
    finally:
        for stream in files.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        connection.close()
    if counters["validation_tokens"] < 65536 or counters["train_tokens"] < 1000000:
        raise ValueError("Prepared corpus is unexpectedly small")
    split_info = {}
    for split in hashes:
        (output / f"{split}.bin.tmp").replace(output / f"{split}.bin")
        split_info[split] = {"file": f"{split}.bin", "tokens": counters[f"{split}_tokens"], "sha256": hashes[split].hexdigest()}
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256((tokenizer_hash + hashes["train"].hexdigest() + hashes["validation"].hexdigest()).encode()).hexdigest()
    manifest = {"format": "uttt-token-corpus-v1", "dtype": "uint16", "tokenizer": "tokenizer.json",
                "tokenizer_sha256": tokenizer_hash, "vocab_size": wrapped.vocab_size,
                "fingerprint": fingerprint, "splits": split_info, "statistics": dict(counters),
                "split_seed": seed, "validation_fraction": validation_fraction, "sources": sources}
    temporary_manifest = output / "manifest.json.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_manifest.replace(output / "manifest.json")
    print("CORPUS_PREPARATION_COMPLETE " + json.dumps(dict(counters)), flush=True)
    return output / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer-documents", type=int, default=100000)
    args = parser.parse_args()
    prepare(args.root, args.tokenizer_documents)

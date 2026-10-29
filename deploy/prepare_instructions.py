"""Prepare fresh Chinese human-reviewed and English synthetic instruction data."""
import argparse
import collections
import hashlib
import json
import random
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from deploy.download_pretrain import list_files, download


def format_messages(messages):
    lines, system, answered = [], [], False
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        content = content.strip().replace("\x00", "")
        if role == "system":
            system.append(content)
        elif role == "user":
            if system:
                content = "\n\n".join([*system, content])
                system.clear()
            lines.append(f"<|user|> {content}")
        elif role == "assistant" and lines:
            lines.append(f"<|assistant|> {content}<|eos|>")
            answered = True
        else:
            return None
    return "\n".join(lines) if answered else None


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = [{
        "repository": "AI-ModelScope/COIG-CQIA", "path": "COIG-CQIA-full.jsonl",
        "size": 149472971, "sha256": "f2a6a758362b7fa5d5c29c9ae0946f2fbf8c9318b6fb48c0e62f8b774d8ff518",
        "local_file": "raw/COIG-CQIA-full.jsonl", "language": "zh",
        "upstream": "https://huggingface.co/datasets/m-a-p/COIG-CQIA",
        "license": "See upstream dataset card and per-record copyright fields",
    }]
    for subset, limit in [("smol-magpie-ultra", 40000), ("smol-constraints", 10000), ("everyday-conversations", 10000)]:
        files = [entry for entry in list_files("HuggingFaceTB/smoltalk", f"data/{subset}") if "/train-" in entry["Path"]]
        entry = random.Random(42).choice(files)
        sources.append({"repository": "HuggingFaceTB/smoltalk", "path": entry["Path"],
                        "size": entry["Size"], "sha256": entry["Sha256"], "language": "en",
                        "local_file": f"raw/{subset}/{Path(entry['Path']).name}", "limit": limit,
                        "upstream": "https://huggingface.co/datasets/HuggingFaceTB/smoltalk", "license": "Apache-2.0"})
    rows, seen, counts = [], set(), collections.Counter()
    for source in sources:
        query = urllib.parse.urlencode({"Revision": "master", "FilePath": source["path"]})
        source["download_url"] = f"https://www.modelscope.cn/api/v1/datasets/{source['repository']}/repo?{query}"
        download(source, output)
        path = output / source["local_file"]
        if source["language"] == "zh":
            def records():
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        yield json.loads(line)
            selected = records()
        else:
            # Reservoir sampling avoids taking only the beginning of a topic-ordered shard.
            reservoir, rng, visited = [], random.Random(42), 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    visited += 1
                    if len(reservoir) < source["limit"]:
                        reservoir.append(row)
                    else:
                        index = rng.randrange(visited)
                        if index < len(reservoir):
                            reservoir[index] = row
            selected = reservoir
        for row in selected:
            if source["language"] == "zh":
                instruction, answer = row.get("instruction"), row.get("output")
                if not isinstance(instruction, str) or not isinstance(answer, str):
                    continue
                context = row.get("input") or ""
                if not isinstance(context, str):
                    continue
                user = instruction + ("\n\n" + context if context else "")
                text = format_messages([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
            else:
                text = format_messages(row.get("messages", []))
            if not text or len(text) > 100000 or "\ufffd" in text:
                continue
            identity = hashlib.sha256(text.encode()).digest()
            if identity in seen:
                continue
            seen.add(identity)
            record = {"text": text, "source": source["upstream"], "source_file": source["path"]}
            if row.get("copyright"):
                record["copyright"] = row["copyright"]
            rows.append(record)
            counts[source["language"]] += 1
        print(f"[Instruction source] {source['path']} totals={dict(counts)}", flush=True)
    random.Random(42).shuffle(rows)
    if counts.get("zh", 0) < 1000 or counts.get("en", 0) < 1000:
        raise ValueError("Instruction mixture is missing enough Chinese or English documents")
    with (output / "train.jsonl.tmp").open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "train.jsonl.tmp").replace(output / "train.jsonl")
    (output / "sources.json").write_text(json.dumps({"sources": sources, "documents": dict(counts)}, indent=2))
    print("INSTRUCTION_PREPARATION_COMPLETE " + json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/instruction_20261029")
    args = parser.parse_args()
    prepare(args.output)

"""Download a reproducible bilingual FineWeb shard selection with hash verification."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path


def request(url, **kwargs):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "uTTT-pretraining/1.0", **kwargs}), timeout=60)


def list_files(repo, root):
    query = urllib.parse.urlencode({"Revision": "master", "Root": root, "PageNumber": 1, "PageSize": 1000})
    with request(f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo/tree?{query}") as response:
        result = json.load(response)
    if result.get("Code") != 200:
        raise RuntimeError(result)
    return sorted((entry for entry in result["Data"]["Files"] if entry["Path"].endswith(".parquet")), key=lambda entry: entry["Path"])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, output):
    target = output / entry["local_file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry["size"] and sha256(target) == entry["sha256"]:
        print(f"[Verified existing] {entry['local_file']}", flush=True)
        return
    temporary = target.with_suffix(".part")
    for attempt in range(4):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset >= entry["size"]:
                if offset == entry["size"] and sha256(temporary) == entry["sha256"]:
                    temporary.replace(target)
                    print(f"[Verified completed partial] {entry['local_file']}", flush=True)
                    return
                temporary.unlink(missing_ok=True)
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with request(entry["download_url"], **headers) as response:
                if offset and response.status != 206:
                    offset = 0
                with temporary.open("ab" if offset else "wb") as destination:
                    while block := response.read(4 * 1024 * 1024):
                        destination.write(block)
            if temporary.stat().st_size != entry["size"] or sha256(temporary) != entry["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError("Downloaded shard failed size/SHA-256 verification")
            temporary.replace(target)
            print(f"[Downloaded + verified] {entry['local_file']} ({entry['size'] / 1e9:.2f} GB)", flush=True)
            return
        except Exception as error:
            if attempt == 3:
                raise
            print(f"[Retry {attempt + 1}] {entry['local_file']}: {error}", flush=True)
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zh-shards", type=int, default=12)
    parser.add_argument("--en-shards", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "sources.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = {"seed": args.seed, "sources": []}
        for language, repo, root, count, license_name in [
            ("zh", "opencsg/chinese-fineweb-edu-v2", "data", args.zh_shards, "Apache-2.0 dataset card; see upstream OpenCSG community license file"),
            ("en", "HuggingFaceFW/fineweb-edu", "sample/10BT", args.en_shards, "ODC-BY"),
        ]:
            entries = list_files(repo, root)
            if len(entries) < count:
                raise ValueError(f"Not enough shards in {repo}: {len(entries)} < {count}")
            selected = random.Random(args.seed).sample(entries, count)
            for entry in selected:
                query = urllib.parse.urlencode({"Revision": "master", "FilePath": entry["Path"]})
                plan["sources"].append({
                    "language": language, "repository": repo, "path": entry["Path"],
                    "size": entry["Size"], "sha256": entry["Sha256"], "license": license_name,
                    "upstream": f"https://huggingface.co/datasets/{repo}",
                    "download_url": f"https://www.modelscope.cn/api/v1/datasets/{repo}/repo?{query}",
                    "local_file": f"raw/{language}/{Path(entry['Path']).name}",
                })
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[Plan] {len(plan['sources'])} shards, {sum(item['size'] for item in plan['sources']) / 1e9:.2f} GB", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda entry: download(entry, output), plan["sources"]))
    print("CORPUS_DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

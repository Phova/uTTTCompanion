"""Constant-memory, packed language-model datasets prepared on disk."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class ResumableBatchSampler(Sampler):
    """Keep the consumed cursor separate from DataLoader's speculative prefetch.

    Iteration never commits progress. The trainer advances only after a batch
    succeeds, and can restore the step's cursor before retrying an OOM.
    """
    def __init__(self, size, batch_size, seed=42):
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset size and batch size must be positive")
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        self.epoch, self.offset = 0, 0
        self._permutation_key, self._permutation = None, None

    def _order_for_epoch(self):
        key = (self.size, self.seed, self.epoch)
        if self._permutation_key != key:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            # Keep the exact legacy randperm, but avoid millions of boxed Python
            # integers. NumPy shares this CPU tensor's storage without copying.
            self._permutation = torch.randperm(self.size, generator=generator).numpy()
            self._permutation_key = key
        return self._permutation

    def __iter__(self):
        order = memoryview(self._order_for_epoch())
        size, batch_size, offset = self.size, self.batch_size, self.offset
        # An OOM can change batch size at an offset that is not divisible by
        # the new size. Drop only the actual remaining tail, not a global tail.
        limit = offset + ((size - offset) // batch_size) * batch_size if size >= batch_size else size
        if batch_size == 1:
            for start in range(offset, limit):
                yield [order[start]]
            return
        for start in range(offset, limit, batch_size):
            end = min(start + batch_size, limit)
            if end - start < batch_size and size >= batch_size:
                break
            # Only box the small batch crossing the DataLoader queue boundary.
            yield order[start:end].tolist()

    def __len__(self):
        remaining = self.size - self.offset
        return remaining // self.batch_size if self.size >= self.batch_size else int(remaining > 0)

    def set_batch_size(self, batch_size):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.batch_size = int(batch_size)
        if self.size >= self.batch_size and self.size - self.offset < self.batch_size:
            self.epoch += 1
            self.offset = 0

    def advance(self, count):
        expected = min(self.batch_size, self.size - self.offset)
        if count != expected:
            raise ValueError("Consumed batch does not match the data cursor")
        self.offset += count
        # Drop the incomplete tail exactly as the original training loader did.
        if self.offset == self.size or (self.size >= self.batch_size and self.size - self.offset < self.batch_size):
            self.epoch += 1
            self.offset = 0

    def state_dict(self):
        return {"version": 1, "size": self.size, "seed": self.seed,
                "epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("size") != self.size:
            raise ValueError("Checkpoint data cursor is incompatible with the dataset")
        epoch, offset = int(state["epoch"]), int(state["offset"])
        if epoch < 0 or not 0 <= offset < self.size:
            raise ValueError("Invalid checkpoint data cursor")
        self.seed, self.epoch, self.offset = int(state["seed"]), epoch, offset


def read_manifest(path):
    if not path or Path(path).suffix != ".json":
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if value.get("format") == "uttt-token-corpus-v1" else None


class TokenizedCorpusDataset(Dataset):
    def __init__(self, manifest_path, seq_len=1024, split="train"):
        if seq_len < 2 or split not in {"train", "validation"}:
            raise ValueError("Invalid sequence length or corpus split")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = read_manifest(self.manifest_path)
        if self.manifest is None:
            raise ValueError("Not a uTTT token corpus manifest")
        self.seq_len, self.split = seq_len, split
        self.dtype = np.dtype(self.manifest["dtype"])
        if self.dtype not in (np.dtype("uint16"), np.dtype("uint32")):
            raise ValueError("Unsupported corpus token dtype")
        info = self.manifest["splits"][split]
        self.path = (self.manifest_path.parent / info["file"]).resolve()
        if not self.path.is_relative_to(self.manifest_path.parent):
            raise ValueError("Corpus files must stay inside the manifest folder")
        self.token_count = int(info["tokens"])
        if self.path.stat().st_size != self.token_count * self.dtype.itemsize:
            raise ValueError("Corpus token count does not match file size")
        if self.token_count < seq_len:
            raise ValueError("Corpus split is too short for the training sequence")
        self.fingerprint = self.manifest["fingerprint"] + ":" + split
        self._tokens = None
        self._pid = None

    def __len__(self):
        return (self.token_count - 1) // (self.seq_len - 1)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._tokens is None or self._pid != os.getpid():
            self._tokens = np.memmap(self.path, mode="r", dtype=self.dtype)
            self._pid = os.getpid()
        start = index * (self.seq_len - 1)
        ids = torch.from_numpy(np.array(self._tokens[start:start + self.seq_len], dtype=np.int64, copy=True))
        labels = ids.clone()
        labels[(labels == 0) | (labels == 1)] = -100
        return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones(self.seq_len)}

    def __getstate__(self):
        return {**self.__dict__, "_tokens": None, "_pid": None}

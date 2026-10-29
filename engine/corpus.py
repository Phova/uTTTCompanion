"""Constant-memory, packed language-model datasets prepared on disk."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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

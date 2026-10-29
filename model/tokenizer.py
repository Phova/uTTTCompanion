"""Checkpoint-contained BPE, with backward compatibility for hybrid token IDs."""
from model.uttt_model import SimpleUTF8Tokenizer


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|user|>", "<|assistant|>"]


class BPETokenizer(SimpleUTF8Tokenizer):
    def __init__(self, tokenizer_json, vocab_size):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_str(tokenizer_json)
        self.backend.no_truncation()
        self.backend.no_padding()
        self.vocab_size = self.backend.get_vocab_size()
        if self.vocab_size > vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
        for index, token in enumerate(SPECIAL_TOKENS):
            if self.backend.token_to_id(token) != index:
                raise ValueError(f"Tokenizer must assign {token} ID {index}")

    def encode(self, text, add_bos=True, add_eos=False, max_length=None):
        ids = ([self.BOS_ID] if add_bos else []) + self.backend.encode(text, add_special_tokens=False).ids
        if add_eos:
            ids.append(self.EOS_ID)
        return ids if max_length is None else ids[:max_length]

    def decode(self, token_ids, skip_special_tokens=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens)


def build_tokenizer(config):
    if config.tokenizer_json:
        return BPETokenizer(config.tokenizer_json, config.vocab_size)
    return SimpleUTF8Tokenizer(config.vocab_size)

"""Build chat inputs ending at a stable assistant token boundary."""


MAX_HISTORY_MESSAGES = 16
MAX_HISTORY_MESSAGE_CHARS = 4096
MAX_HISTORY_CHARS = 16384
_TEMPLATE_MARKERS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|user|>", "<|assistant|>")


def _reject_template_markers(text):
    if any(marker in text for marker in _TEMPLATE_MARKERS):
        raise ValueError("Chat text must not contain reserved template markers")


def validate_chat_history(history):
    """Only bounded, complete user/assistant pairs may precede this turn."""
    if not isinstance(history, list) or len(history) > MAX_HISTORY_MESSAGES or len(history) % 2:
        raise ValueError(f"history must contain complete user/assistant pairs, at most {MAX_HISTORY_MESSAGES} messages")
    normalized = []
    total = 0
    for index, message in enumerate(history):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each history message must contain only role and content")
        role = "user" if index % 2 == 0 else "assistant"
        content = message["content"]
        if message["role"] != role or not isinstance(content, str) or not content.strip():
            raise ValueError("history must alternate nonblank user and assistant messages")
        if len(content) > MAX_HISTORY_MESSAGE_CHARS:
            raise ValueError(f"Each history message must be at most {MAX_HISTORY_MESSAGE_CHARS} characters")
        _reject_template_markers(content)
        total += len(content)
        normalized.append({"role": role, "content": content.strip()})
    if total > MAX_HISTORY_CHARS:
        raise ValueError(f"history must be at most {MAX_HISTORY_CHARS} characters in total")
    return normalized


def _history_prefix(history):
    return "".join(
        f"<|user|> {history[index]['content']}\n<|assistant|> {history[index + 1]['content']}<|eos|>\n"
        for index in range(0, len(history), 2)
    )


def format_chat_prompt(prompt, history=None):
    """Leave answer whitespace to the model, as it appears in SFT targets.

    Byte-level BPE can merge a space with the answer's first word. Encoding a
    trailing space without that word creates a different token sequence from
    the training prefix. The special assistant token is a stable boundary for
    English, Chinese and numeric answers alike.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be nonblank text")
    _reject_template_markers(prompt)
    messages = validate_chat_history(history) if history is not None else []
    return _history_prefix(messages) + f"<|user|> {prompt.strip()}\n<|assistant|>"


def encode_chat_prompt(tokenizer, prompt, max_length, history=None):
    """Keep the chat prefix and most recent input when the prompt is too long.

    Tokenize the complete template first so ordinary inputs retain exactly the
    same BPE tokens. Truncation retains its original tail, including the newline
    and assistant prefix, instead of cutting off the role that should answer.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Chat input token budget must be a positive integer")
    messages = validate_chat_history(history) if history is not None else []
    text = format_chat_prompt(prompt)
    prefix = tokenizer.encode("<|user|>", add_bos=True, add_eos=False)
    suffix = tokenizer.encode("\n<|assistant|>", add_bos=False, add_eos=False)
    if max_length < len(prefix) + len(suffix) + 1:
        raise ValueError("Token budget must leave room for the chat template and at least one prompt token; reduce max_new_tokens")
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    if len(ids) <= max_length:
        # Keep complete recent pairs only. Never cut a role, a previous answer,
        # or the current question merely to squeeze in older conversation.
        for first in range(0, len(messages), 2):
            candidate = tokenizer.encode(_history_prefix(messages[first:]) + text, add_bos=True, add_eos=False)
            if len(candidate) <= max_length:
                return candidate
        return ids
    return ids[:len(prefix)] + ids[-(max_length - len(prefix)):]

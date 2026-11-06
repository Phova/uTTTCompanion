"""Build chat inputs ending at a stable assistant token boundary."""


def format_chat_prompt(prompt):
    """Leave answer whitespace to the model, as it appears in SFT targets.

    Byte-level BPE can merge a space with the answer's first word. Encoding a
    trailing space without that word creates a different token sequence from
    the training prefix. The special assistant token is a stable boundary for
    English, Chinese and numeric answers alike.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be nonblank text")
    return f"<|user|> {prompt.strip()}\n<|assistant|>"


def encode_chat_prompt(tokenizer, prompt, max_length):
    """Keep the chat prefix and most recent input when the prompt is too long.

    Tokenize the complete template first so ordinary inputs retain exactly the
    same BPE tokens. Truncation retains its original tail, including the newline
    and assistant prefix, instead of cutting off the role that should answer.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Chat input token budget must be a positive integer")
    text = format_chat_prompt(prompt)
    prefix = tokenizer.encode("<|user|>", add_bos=True, add_eos=False)
    suffix = tokenizer.encode("\n<|assistant|>", add_bos=False, add_eos=False)
    if max_length < len(prefix) + len(suffix) + 1:
        raise ValueError("Token budget must leave room for the chat template and at least one prompt token; reduce max_new_tokens")
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    if len(ids) <= max_length:
        return ids
    return ids[:len(prefix)] + ids[-(max_length - len(prefix)):]

"""Build bounded chat inputs without removing the assistant role boundary."""


def encode_chat_prompt(tokenizer, prompt, max_length):
    """Keep the chat prefix and most recent input when the prompt is too long.

    Tokenize the complete template first so ordinary inputs retain exactly the
    same BPE tokens. Truncation retains its original tail, including the newline
    and assistant prefix, instead of cutting off the role that should answer.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Chat input token budget must be a positive integer")
    text = prompt.strip()
    if not text:
        raise ValueError("Prompt must not be blank")
    prefix = tokenizer.encode("<|user|>", add_bos=True, add_eos=False)
    suffix = tokenizer.encode("\n<|assistant|> ", add_bos=False, add_eos=False)
    if max_length < len(prefix) + len(suffix) + 1:
        raise ValueError("Token budget must leave room for the chat template and at least one prompt token; reduce max_new_tokens")
    ids = tokenizer.encode(f"<|user|> {text}\n<|assistant|> ", add_bos=True, add_eos=False)
    if len(ids) <= max_length:
        return ids
    return ids[:len(prefix)] + ids[-(max_length - len(prefix)):]

"""Small, dependency-free generation diagnostics; these are not quality scores."""


def repeated_ngram_fraction(values, n=4):
    """Fraction of n-gram occurrences that repeat an earlier occurrence."""
    if n < 1:
        raise ValueError("n must be positive")
    if len(values) < n:
        return 0.0
    grams = [tuple(values[index:index + n]) for index in range(len(values) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def repetition_metrics(token_ids, text):
    longest = run = 0
    previous = None
    for token in token_ids:
        run = run + 1 if token == previous else 1
        longest = max(longest, run)
        previous = token
    normalized = " ".join(text.split())
    return {
        "token_4gram_repetition": repeated_ngram_fraction(token_ids),
        "text_4gram_repetition": repeated_ngram_fraction(normalized),
        "max_token_run": longest,
    }


def repeated_ngram_continuations(token_ids, size):
    """Next tokens that would repeat an existing generated n-gram.

    This is an optional decoding control, never a training-quality measurement.
    It considers generated tokens only and does not penalize prompt contents.
    """
    if size < 1:
        raise ValueError("n-gram size must be positive")
    if len(token_ids) + 1 < size:
        return set()
    prefix = tuple(token_ids[-(size - 1):]) if size > 1 else ()
    return {token_ids[index + size - 1]
            for index in range(len(token_ids) - size + 1)
            if tuple(token_ids[index:index + size - 1]) == prefix}


def generation_summary(results):
    """Aggregate observable decoding behavior without calling it model quality."""
    count = len(results)
    denominator = max(count, 1)
    return {
        "samples": count,
        "eos_rate": sum(result["eos_generated"] for result in results) / denominator,
        "token_limit_rate": sum(result["stop_reason"] == "token_limit" for result in results) / denominator,
        "mean_generated_tokens": sum(result["generated_tokens"] for result in results) / denominator,
        "mean_token_4gram_repetition": sum(result["token_4gram_repetition"] for result in results) / denominator,
        "mean_text_4gram_repetition": sum(result["text_4gram_repetition"] for result in results) / denominator,
        "accepted_exact_matches": sum(result.get("exact_match", False) for result in results),
        "accepted_tasks": sum("exact_match" in result for result in results),
    }

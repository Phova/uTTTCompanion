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

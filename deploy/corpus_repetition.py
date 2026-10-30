"""Dependency-free diagnostics for repeated long token spans within documents.

These diagnostics do not alter training labels or impose a generation penalty.
"""
from collections import Counter, deque


def analyze_repeated_spans(token_ids, ngram_size=16, keep_occurrences=2, boundary_ids=(0, 1, 2, 3, 4, 5)):
    """Find targets ending the third (by default) occurrence of an exact span.

    Counts restart at each protected boundary. Overlapping occurrences count,
    which also detects long single-token and short-period degenerate runs.
    The first window token has no next-token supervision and is not counted.
    """
    if ngram_size < 2 or keep_occurrences < 1:
        raise ValueError("ngram_size must be at least two and keep_occurrences positive")
    tokens = [int(value) for value in token_ids]
    boundaries = frozenset(int(value) for value in boundary_ids)
    recent = deque(maxlen=ngram_size)
    counts = Counter()
    mask = [False] * len(tokens)
    top_span, top_count = (), 0
    longest_run = run = 0
    for index, token in enumerate(tokens):
        if token in boundaries:
            recent.clear()
            counts.clear()
            run = 0
            continue
        recent.append(token)
        if len(recent) != ngram_size:
            run = 0
            continue
        span = tuple(recent)
        counts[span] += 1
        if counts[span] > top_count:
            top_span, top_count = span, counts[span]
        if counts[span] > keep_occurrences and index > 0:
            mask[index] = True
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0
    supervised = sum(token not in (0, 1) for token in tokens[1:])
    repeated = sum(mask)
    return {"mask": mask, "supervised_targets": supervised, "repeated_targets": repeated,
            "fraction": repeated / supervised if supervised else 0.0,
            "longest_masked_run": longest_run,
            "most_repeated_span": list(top_span) if top_count > keep_occurrences else [],
            "most_repeated_span_occurrences": top_count if top_count > keep_occurrences else 0}


def detected_language(text):
    """Coarse script-based grouping, not an upstream source or language label."""
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    letters = sum(char.isalpha() for char in text)
    if letters and cjk / letters >= .2:
        return "zh_or_mixed_cjk"
    if letters and latin / letters >= .8:
        return "latin_dominant"
    return "other_or_mixed"

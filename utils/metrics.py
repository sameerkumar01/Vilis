"""Evaluation metrics for Visual Speech Recognition: Character Error Rate (CER) and Word Error Rate (WER)."""

from __future__ import annotations

from typing import List


def levenshtein_distance(seq1: List | str, seq2: List | str) -> int:
    """Compute Levenshtein distance between two sequences or strings."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]

    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = matrix[x - 1][y - 1]
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,       # Deletion
                    matrix[x][y - 1] + 1,       # Insertion
                    matrix[x - 1][y - 1] + 1    # Substitution
                )
    return matrix[size_x - 1][size_y - 1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) = Levenshtein(ref, hyp) / len(ref)."""
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist) / len(ref_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) = Levenshtein(ref_words, hyp_words) / len(ref_words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist) / len(ref_words)

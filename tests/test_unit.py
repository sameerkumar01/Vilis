"""Unit test verifying ViLiS components and tokenizer without requiring GPU."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import compute_cer, compute_wer
from utils.tokenizer import ViLiSTokenizer


def test_metrics():
    ref = "नमस्ते आप कैसे हैं"
    hyp = "नमस्ते आप कैसे हैं"
    assert compute_cer(ref, hyp) == 0.0
    assert compute_wer(ref, hyp) == 0.0

    hyp_sub = "नमस्ते आप कैसे हो"
    assert compute_wer(ref, hyp_sub) > 0.0
    print("Metrics test passed!")


def test_tokenizer():
    tok = ViLiSTokenizer()
    text = "नमस्ते दुनिया"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
    print(f"Tokenizer test passed! Encoded: {encoded} -> Decoded: '{decoded}'")


if __name__ == "__main__":
    test_metrics()
    test_tokenizer()

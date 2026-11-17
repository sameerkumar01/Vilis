"""Hindi Devanagari Tokenizer for ViLiS.

Provides a unified interface using either SentencePiece or a robust
character-level Devanagari fallback tokenizer for CTC decoding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Standard Hindi Devanagari characters + numerals + punctuation
DEVANAGARI_CHARS = [
    # Special tokens
    "<blank>",  # CTC blank token = index 0
    "<unk>",
    "<pad>",
    " ",        # Space character
    # Vowels (Swar)
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", "अं", "अः",
    # Matras (Vowel signs)
    "ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ", "ं", "ः", "्", "ॅ", "ॉ", "ँ", "़",
    # Consonants (Vyanjan)
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
    "क्ष", "त्र", "ज्ञ", "श्र",
    # Nuqta consonants
    "क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़",
    # Digits
    "०", "१", "२", "३", "४", "५", "६", "७", "८", "९",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Punctuation
    "।", "?", "!", ",", "."
]


class ViLiSTokenizer:
    """Tokenizer supporting SentencePiece model with built-in Devanagari fallback."""

    def __init__(self, sp_model_path: Optional[str | Path] = None) -> None:
        self.sp = None
        self.sp_path = None
        self.char2idx = {}
        self.idx2char = {}

        if sp_model_path and Path(sp_model_path).exists():
            try:
                import sentencepiece as spm
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(str(sp_model_path))
                self.sp_path = str(sp_model_path)
                logger.info("Loaded SentencePiece model from %s (vocab: %d)", sp_model_path, len(self.sp))
            except Exception as e:
                logger.warning("Could not load SentencePiece model: %s. Falling back to character tokenizer.", e)
                self._build_char_vocab()
        else:
            self._build_char_vocab()

    def _build_char_vocab(self) -> None:
        """Build character-level vocabulary for Devanagari script."""
        unique_chars = []
        for ch in DEVANAGARI_CHARS:
            if ch not in unique_chars:
                unique_chars.append(ch)
        self.char2idx = {ch: idx for idx, ch in enumerate(unique_chars)}
        self.idx2char = {idx: ch for idx, ch in enumerate(unique_chars)}

    @property
    def vocab_size(self) -> int:
        if self.sp is not None:
            return len(self.sp)
        return len(self.char2idx)

    def encode(self, text: str) -> List[int]:
        """Encode text string into token IDs."""
        if self.sp is not None:
            return self.sp.encode_as_ids(text)
        
        # Character-level encoding
        unk_idx = self.char2idx.get("<unk>", 1)
        tokens = []
        for ch in text.strip():
            tokens.append(self.char2idx.get(ch, unk_idx))
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into text string."""
        if self.sp is not None:
            return self.sp.decode_ids(ids)

        chars = []
        for i in ids:
            if i == 0:  # <blank>
                continue
            char = self.idx2char.get(i, "")
            if char not in ("<pad>", "<unk>", "<blank>"):
                chars.append(char)
        return "".join(chars).strip()

    def ctc_greedy_decode(self, logits: List[int] | np.ndarray) -> str:
        """Collapse repeated tokens and remove blanks (index 0)."""
        collapsed = []
        prev = None
        for token_id in logits:
            if token_id != prev and token_id != 0:
                collapsed.append(int(token_id))
            prev = token_id
        return self.decode(collapsed)

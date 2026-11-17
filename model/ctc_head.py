"""
CTC Prediction Head for ViLiS.

Produces per-frame log-probabilities over the vocabulary (including CTC blank).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    """LayerNorm → Linear projection → log-softmax over vocab dimension."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — encoder output features

        Returns:
            (B, T, vocab_size) — log-probabilities
        """
        x = self.layer_norm(x)
        logits = self.projection(x)
        return F.log_softmax(logits, dim=-1)

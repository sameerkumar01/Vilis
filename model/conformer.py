"""
Conformer Encoder for ViLiS.

Architecture per block (Macaron-style):
    x → ½ FFN → MHSA → DepthwiseConv → ½ FFN → LayerNorm

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", Interspeech 2020.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# Positional Encoding (sinusoidal)
# ──────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)              # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                             # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────
# Feed-Forward Module
# ──────────────────────────────────────────────────────────

class FeedForwardModule(nn.Module):
    """Position-wise FFN with SiLU (Swish) activation and half-step residual."""

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        d_inner = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_inner)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the FFN output *without* the residual add (caller scales by 0.5)."""
        out = self.layer_norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return out


# ──────────────────────────────────────────────────────────
# Multi-Head Self-Attention Module
# ──────────────────────────────────────────────────────────

class MultiHeadSelfAttentionModule(nn.Module):
    """MHSA wrapper with pre-norm and residual."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns MHSA output *without* the residual add (caller handles it)."""
        out = self.layer_norm(x)
        out, _ = self.mhsa(out, out, out)
        out = self.dropout(out)
        return out


# ──────────────────────────────────────────────────────────
# Depthwise Convolution Module
# ──────────────────────────────────────────────────────────

class DepthwiseConvModule(nn.Module):
    """Conformer convolution module.

    Pointwise → GLU → DepthwiseConv1d → BatchNorm → SiLU → Pointwise
    """

    def __init__(self, d_model: int, conv_kernel: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expansion (×2 for GLU gating)
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        self.glu = nn.GLU(dim=-1)

        # Depthwise separable conv (groups = channels)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=d_model,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()

        # Pointwise projection back
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns conv output *without* the residual add."""
        out = self.layer_norm(x)                         # (B, T, D)
        out = self.pointwise1(out)                       # (B, T, 2D)
        out = self.glu(out)                              # (B, T, D)

        out = out.transpose(1, 2)                        # (B, D, T) for Conv1d
        out = self.depthwise_conv(out)                   # (B, D, T)
        out = self.batch_norm(out)                       # (B, D, T)
        out = self.activation(out)
        out = out.transpose(1, 2)                        # (B, T, D)

        out = self.pointwise2(out)                       # (B, T, D)
        out = self.dropout(out)
        return out


# ──────────────────────────────────────────────────────────
# Conformer Block
# ──────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """Single Conformer block (Macaron-style).

    x → +0.5·FFN₁ → +MHSA → +Conv → +0.5·FFN₂ → LayerNorm
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ff_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, nhead, dropout)
        self.conv = DepthwiseConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Macaron FFN (half-step)
        x = x + 0.5 * self.ffn1(x)
        # Multi-Head Self-Attention
        x = x + self.mhsa(x)
        # Convolution module
        x = x + self.conv(x)
        # Second FFN (half-step)
        x = x + 0.5 * self.ffn2(x)
        # Final layer norm
        x = self.final_layer_norm(x)
        return x


# ──────────────────────────────────────────────────────────
# Conformer Encoder (stack of blocks)
# ──────────────────────────────────────────────────────────

class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks with sinusoidal positional encoding.

    Input:  (B, T, d_model)
    Output: (B, T, d_model)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 12,
        nhead: int = 8,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ff_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(d_model, nhead, conv_kernel, dropout, ff_expansion)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

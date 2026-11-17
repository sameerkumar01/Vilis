"""
ViLiS — Visual Lip-reading System for Hindi.

End-to-end model: SpatioTemporalFrontend → ConformerEncoder → CTCHead

Input:  (B, T, 1, 96, 96) grayscale lip-crop video
Output: (B, T, vocab_size) CTC log-probabilities
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.frontend import SpatioTemporalFrontend
from model.conformer import ConformerEncoder
from model.ctc_head import CTCHead


class ViLiSModel(nn.Module):
    """Full ViLiS lip-reading model."""

    def __init__(self, config) -> None:
        """
        Args:
            config: Object with attributes —
                d_model (int), num_layers (int), nhead (int),
                conv_kernel (int), dropout (float), vocab_size (int)

                These map to the ViLiSConfig from config.config.
        """
        super().__init__()

        self.config = config

        d_model = getattr(config, "D_MODEL", getattr(config, "d_model", 512))
        num_layers = getattr(config, "NUM_LAYERS", getattr(config, "num_layers", 12))
        nhead = getattr(config, "NHEAD", getattr(config, "nhead", 8))
        conv_kernel = getattr(config, "CONV_KERNEL", getattr(config, "conv_kernel", 31))
        dropout = getattr(config, "DROPOUT", getattr(config, "dropout", 0.1))
        vocab_size = getattr(config, "VOCAB_SIZE", getattr(config, "vocab_size", 4000))

        self.frontend = SpatioTemporalFrontend()

        self.conformer_encoder = ConformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )

        self.ctc_head = CTCHead(d_model=d_model, vocab_size=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1, 96, 96) — grayscale lip-crop frames

        Returns:
            (B, T, vocab_size) — CTC log-probabilities per frame
        """
        # Visual feature extraction
        features = self.frontend(x)              # (B, T, 512)

        # Conformer encoding
        encoded = self.conformer_encoder(features)  # (B, T, 512)

        # CTC prediction
        logits = self.ctc_head(encoded)          # (B, T, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def export_onnx(self, path: str | Path, opset_version: int = 17) -> None:
        """Export the model to ONNX format.

        Args:
            path: Destination file path for the .onnx model.
            opset_version: ONNX opset version (default 17).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        device = next(self.parameters()).device

        # Dummy input: single 75-frame clip
        dummy = torch.randn(1, 75, 1, 96, 96, device=device)

        torch.onnx.export(
            self,
            dummy,
            str(path),
            opset_version=opset_version,
            input_names=["video"],
            output_names=["logits"],
            dynamic_axes={
                "video": {0: "batch", 1: "time"},
                "logits": {0: "batch", 1: "time"},
            },
        )
        print(f"ONNX model exported to {path}")
